"""Background scheduler that watches trigger definitions and executes them."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

from ..agents.execution_agent.batch_manager import ExecutionBatchManager
from ..agents.execution_agent.runtime import ExecutionResult
from ..logging_config import logger
from .execution import get_agent_registry
from .triggers import TriggerRecord, get_trigger_service


UTC = timezone.utc


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _isoformat(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


class TriggerScheduler:
    """Polls stored triggers and launches execution agents when due."""

    def __init__(self, poll_interval_seconds: float = 10.0) -> None:
        self._poll_interval = poll_interval_seconds
        self._service = get_trigger_service()
        self._task: Optional[asyncio.Task[None]] = None
        self._running = False
        self._in_flight: Set[int] = set()
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        async with self._lock:
            if self._task and not self._task.done():
                return
            loop = asyncio.get_running_loop()
            self._running = True
            self._task = loop.create_task(self._run(), name="trigger-scheduler")
            logger.info("Trigger scheduler started", extra={"interval": self._poll_interval})

    async def stop(self) -> None:
        async with self._lock:
            self._running = False
            if self._task:
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
                self._task = None
                logger.info("Trigger scheduler stopped")

    async def _run(self) -> None:
        try:
            while self._running:
                await self._poll_once()
                await asyncio.sleep(self._poll_interval)
        except asyncio.CancelledError:  # pragma: no cover - shutdown path
            raise
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Trigger scheduler loop crashed", extra={"error": str(exc)})

    async def _poll_once(self) -> None:
        now = _utc_now()
        due_triggers = self._service.get_due_triggers(before=now)
        if not due_triggers:
            return

        pending = [t for t in due_triggers if t.id not in self._in_flight]
        # Several reminders due for the same agent in one poll are one wake-up, not N: each
        # execution is a full strong-model run over that agent's history.
        groups: List[List[TriggerRecord]] = []
        if self._coalesce_enabled():
            cap = self._coalesce_max()
            by_agent: Dict[str, List[TriggerRecord]] = {}
            for trigger in pending:
                by_agent.setdefault(trigger.agent_name, []).append(trigger)
            for triggers in by_agent.values():
                groups.extend(triggers[i : i + cap] for i in range(0, len(triggers), cap))
        else:
            groups = [[t] for t in pending]
        for group in groups:
            for trigger in group:
                self._in_flight.add(trigger.id)
            name = "trigger-" + "-".join(str(t.id) for t in group)
            asyncio.create_task(self._execute_triggers(group), name=name)

    @staticmethod
    def _coalesce_enabled() -> bool:
        return os.getenv("OPENPOKE_TRIGGER_COALESCE", "1") != "0"

    @staticmethod
    def _coalesce_max() -> int:
        """A combined run shares one timeout and one tool-iteration budget; keep groups small."""
        try:
            return max(1, int(os.getenv("OPENPOKE_TRIGGER_COALESCE_MAX", "3")))
        except ValueError:
            return 3

    async def _execute_trigger(self, trigger: TriggerRecord) -> None:
        await self._execute_triggers([trigger])

    async def _execute_triggers(self, triggers: List[TriggerRecord]) -> None:
        agent_name = triggers[0].agent_name
        try:
            fired_at = _utc_now()
            instructions = self._format_instructions(triggers[0], fired_at) if len(triggers) == 1 \
                else self._format_batch_instructions(triggers, fired_at)
            logger.info(
                "Dispatching trigger" if len(triggers) == 1 else "Dispatching coalesced triggers",
                extra={
                    "trigger_ids": [t.id for t in triggers],
                    "agent": agent_name,
                    "scheduled_for": triggers[0].next_trigger,
                },
            )
            # A firing trigger is activity: re-activate the owning agent so it is not
            # treated as dormant/archived while it still has scheduled work.
            get_agent_registry().touch(agent_name, reason="trigger")
            execution_manager = ExecutionBatchManager()
            result = await execution_manager.execute_agent(
                agent_name,
                instructions,
                source="trigger",
            )
            if not result.success and len(triggers) > 1:
                # The combined run failed (timeout, iteration budget, model error). Do not
                # charge every reminder for it: give each its own run, with its own bookkeeping.
                logger.warning("Coalesced trigger run failed; retrying each trigger alone",
                               extra={"trigger_ids": [t.id for t in triggers], "agent": agent_name})
                for trigger in triggers:
                    await self._execute_triggers([trigger])
                return
            for trigger in triggers:
                if result.success:
                    self._handle_success(trigger, fired_at)
                else:
                    self._handle_failure(trigger, fired_at, result.error or result.response)
        except Exception as exc:  # pragma: no cover - defensive
            for trigger in triggers:
                self._handle_failure(trigger, _utc_now(), str(exc))
            logger.exception(
                "Trigger execution failed unexpectedly",
                extra={"trigger_ids": [t.id for t in triggers], "agent": agent_name},
            )
        finally:
            for trigger in triggers:
                self._in_flight.discard(trigger.id)

    def _format_batch_instructions(self, triggers: List[TriggerRecord], fired_at: datetime) -> str:
        parts = [f"{len(triggers)} reminders for this agent fired together at {_isoformat(fired_at)} (UTC). "
                 "Handle each one; report on all of them in your final message.\n"]
        for index, trigger in enumerate(triggers, 1):
            parts.append(f"=== Reminder {index} of {len(triggers)} ===\n" + self._format_instructions(trigger, fired_at))
        return "\n\n".join(parts)

    def _handle_success(self, trigger: TriggerRecord, fired_at: datetime) -> None:
        logger.info(
            "Trigger completed",
            extra={"trigger_id": trigger.id, "agent": trigger.agent_name},
        )
        self._service.schedule_next_occurrence(trigger, fired_at=fired_at)

    def _handle_failure(self, trigger: TriggerRecord, fired_at: datetime, error: str) -> None:
        logger.warning(
            "Trigger execution failed",
            extra={
                "trigger_id": trigger.id,
                "agent": trigger.agent_name,
                "error": error,
            },
        )
        self._service.record_failure(trigger, error)
        if trigger.recurrence_rule:
            self._service.schedule_next_occurrence(trigger, fired_at=fired_at)
        else:
            self._service.clear_next_fire(trigger.id, agent_name=trigger.agent_name)

    def _format_instructions(self, trigger: TriggerRecord, fired_at: datetime) -> str:
        scheduled_for = trigger.next_trigger or _isoformat(fired_at)
        metadata_lines = [f"Trigger ID: {trigger.id}"]
        if trigger.recurrence_rule:
            metadata_lines.append(f"Recurrence: {trigger.recurrence_rule}")
        if trigger.timezone:
            metadata_lines.append(f"Timezone: {trigger.timezone}")
        if trigger.start_time:
            metadata_lines.append(f"Start Time (UTC): {trigger.start_time}")

        metadata = "\n".join(f"- {line}" for line in metadata_lines)
        return (
            f"Trigger fired at {_isoformat(fired_at)} (UTC).\n"
            f"Scheduled occurrence time: {scheduled_for}.\n\n"
            f"Metadata:\n{metadata}\n\n"
            f"Payload:\n{trigger.payload}"
        )


_scheduler_instance: Optional[TriggerScheduler] = None


def get_trigger_scheduler() -> TriggerScheduler:
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = TriggerScheduler()
    return _scheduler_instance


__all__ = ["TriggerScheduler", "get_trigger_scheduler"]
