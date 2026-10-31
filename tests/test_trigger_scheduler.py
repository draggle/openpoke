"""Trigger scheduler: several reminders due for one agent become one execution."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import List

import pytest

from server.services import trigger_scheduler as ts
from server.services.triggers.models import TriggerRecord


def _record(id_: int, agent: str, payload: str, rrule: str | None = None) -> TriggerRecord:
    stamp = "2026-10-31T09:00:00Z"
    return TriggerRecord(id=id_, agent_name=agent, payload=payload, start_time=stamp, next_trigger=stamp,
                         recurrence_rule=rrule, timezone="UTC", status="active", created_at=stamp, updated_at=stamp)


class FakeService:
    def __init__(self, due: List[TriggerRecord]) -> None:
        self.due = due
        self.completed: List[int] = []
        self.failed: List[int] = []
        self.cleared: List[int] = []

    def get_due_triggers(self, *, before, agent_name=None):
        return list(self.due)

    def schedule_next_occurrence(self, trigger, *, fired_at):
        self.completed.append(trigger.id)

    def record_failure(self, trigger, error):
        self.failed.append(trigger.id)

    def clear_next_fire(self, trigger_id, *, agent_name):
        self.cleared.append(trigger_id)


class FakeManager:
    calls: List[tuple] = []
    succeed = True

    async def execute_agent(self, agent_name, instructions, *, source="user"):
        FakeManager.calls.append((agent_name, instructions, source))
        return ts.ExecutionResult(agent_name=agent_name, success=FakeManager.succeed,
                                  response="ok" if FakeManager.succeed else "boom", error=None if FakeManager.succeed else "boom")


class StubRegistry:
    def touch(self, *a, **k): pass


@pytest.fixture
def scheduler(monkeypatch):
    FakeManager.calls = []
    FakeManager.succeed = True
    monkeypatch.setattr(ts, "ExecutionBatchManager", FakeManager)
    monkeypatch.setattr(ts, "get_agent_registry", lambda: StubRegistry())

    def make(due):
        service = FakeService(due)
        monkeypatch.setattr(ts, "get_trigger_service", lambda: service)
        return ts.TriggerScheduler(poll_interval_seconds=0.01), service
    return make


async def _poll_and_drain(sched):
    await sched._poll_once()
    for _ in range(5):
        await asyncio.sleep(0)


async def test_due_triggers_for_same_agent_coalesce(scheduler, monkeypatch):
    monkeypatch.delenv("OPENPOKE_TRIGGER_COALESCE", raising=False)
    sched, service = scheduler([_record(1, "Alice Lunch", "send the invoice"),
                                _record(2, "Alice Lunch", "ask about Thursday", "FREQ=WEEKLY"),
                                _record(3, "Bob Flight", "check the booking")])
    await _poll_and_drain(sched)
    assert len(FakeManager.calls) == 2
    alice = next(c for c in FakeManager.calls if c[0] == "Alice Lunch")
    assert "Trigger ID: 1" in alice[1] and "Trigger ID: 2" in alice[1]
    assert "send the invoice" in alice[1] and "ask about Thursday" in alice[1]
    assert alice[2] == "trigger"
    assert sorted(service.completed) == [1, 2, 3]


async def test_coalesced_failure_marks_every_trigger(scheduler, monkeypatch):
    monkeypatch.delenv("OPENPOKE_TRIGGER_COALESCE", raising=False)
    FakeManager.succeed = False
    sched, service = scheduler([_record(1, "Alice Lunch", "a"), _record(2, "Alice Lunch", "b", "FREQ=DAILY")])
    await _poll_and_drain(sched)
    assert len(FakeManager.calls) == 1
    assert sorted(service.failed) == [1, 2]
    assert service.cleared == [1]  # one-shot: next fire cleared
    assert service.completed == [2]  # recurring: rescheduled despite the failure (as upstream)


async def test_coalesce_flag_off_dispatches_separately(scheduler, monkeypatch):
    monkeypatch.setenv("OPENPOKE_TRIGGER_COALESCE", "0")
    sched, service = scheduler([_record(1, "Alice Lunch", "a"), _record(2, "Alice Lunch", "b")])
    await _poll_and_drain(sched)
    assert len(FakeManager.calls) == 2
    assert sorted(service.completed) == [1, 2]
