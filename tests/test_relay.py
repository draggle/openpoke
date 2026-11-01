"""Spike: relay an execution batch's user summaries straight to the user, no second interaction call."""

from __future__ import annotations

import asyncio

from server.agents.execution_agent import batch_manager as bm
from server.agents.execution_agent.relay import relay_message, strip_summary, user_summary
from server.agents.execution_agent.runtime import ExecutionResult


def test_user_summary_is_extracted_and_stripped():
    text = "Created trigger T1 for 10:00.\n<user_summary>done, reminder moved to 10am tomorrow.</user_summary>"
    assert user_summary(text) == "done, reminder moved to 10am tomorrow."
    assert strip_summary(text) == "Created trigger T1 for 10:00.\ndone, reminder moved to 10am tomorrow."
    assert user_summary("no tag here") is None and strip_summary("no tag here") == "no tag here"
    assert user_summary("<user_summary>   </user_summary>") is None


def test_relay_message_needs_every_result_successful_with_a_summary():
    ok = ExecutionResult(agent_name="A", success=True, response="x <user_summary>a done</user_summary>")
    ok2 = ExecutionResult(agent_name="B", success=True, response="<user_summary>b done</user_summary>")
    bare = ExecutionResult(agent_name="C", success=True, response="did it")
    failed = ExecutionResult(agent_name="D", success=False, response="<user_summary>nope</user_summary>")
    assert relay_message([ok, ok2]) == "a done\nb done"
    assert relay_message([ok, bare]) is None
    assert relay_message([ok, failed]) is None
    assert relay_message([]) is None


class _Log:
    def __init__(self):
        self.agent_messages, self.replies = [], []

    def record_agent_message(self, m): self.agent_messages.append(m)
    def record_reply(self, m): self.replies.append(m)


def _manager(monkeypatch, response, success=True):
    class FakeRuntime:
        def __init__(self, agent_name, source="user", send_authorisation=None): self.agent_name = agent_name
        async def execute(self, instructions):
            return ExecutionResult(agent_name=self.agent_name, success=success, response=response)

    monkeypatch.setattr(bm, "ExecutionAgentRuntime", FakeRuntime)
    manager = bm.ExecutionBatchManager(timeout_seconds=5)
    log = _Log()
    monkeypatch.setattr(bm, "_conversation_log", lambda: log)
    dispatched = []

    async def fake_dispatch(payload): dispatched.append(payload)

    monkeypatch.setattr(manager, "_dispatch_to_interaction_agent", fake_dispatch)
    return manager, log, dispatched


def test_batch_with_summaries_is_relayed_without_the_interaction_agent(monkeypatch):
    monkeypatch.delenv("OPENPOKE_DIRECT_RELAY", raising=False)
    manager, log, dispatched = _manager(monkeypatch, "Trigger set.\n<user_summary>done, 10am tomorrow.</user_summary>")
    asyncio.run(manager.execute_agent("Alice reminder", "move it"))
    assert dispatched == []
    assert log.replies == ["done, 10am tomorrow."]
    assert log.agent_messages == ["[SUCCESS] Alice reminder: Trigger set.\ndone, 10am tomorrow."]


def test_batch_without_summary_goes_through_the_interaction_agent(monkeypatch):
    monkeypatch.delenv("OPENPOKE_DIRECT_RELAY", raising=False)
    manager, log, dispatched = _manager(monkeypatch, "Trigger set.")
    asyncio.run(manager.execute_agent("Alice reminder", "move it"))
    assert dispatched == ["[SUCCESS] Alice reminder: Trigger set."] and log.replies == []


def test_failed_result_goes_through_the_interaction_agent(monkeypatch):
    monkeypatch.delenv("OPENPOKE_DIRECT_RELAY", raising=False)
    manager, log, dispatched = _manager(monkeypatch, "<user_summary>nope</user_summary>", success=False)
    asyncio.run(manager.execute_agent("Alice reminder", "move it"))
    assert len(dispatched) == 1 and dispatched[0].startswith("[FAILED]") and log.replies == []


def test_relay_flag_off_restores_the_old_path(monkeypatch):
    monkeypatch.setenv("OPENPOKE_DIRECT_RELAY", "0")
    manager, log, dispatched = _manager(monkeypatch, "<user_summary>done</user_summary>")
    asyncio.run(manager.execute_agent("Alice reminder", "move it"))
    assert dispatched == ["[SUCCESS] Alice reminder: done"] and log.replies == []


def test_trigger_sourced_runs_are_never_relayed(monkeypatch):
    """A reminder firing in the background keeps the interaction agent in the loop: it decides
    how to tell the user, the execution agent's line is written for a request it was given."""
    monkeypatch.delenv("OPENPOKE_DIRECT_RELAY", raising=False)
    manager, log, dispatched = _manager(monkeypatch, "<user_summary>reminder: call the dentist</user_summary>")
    asyncio.run(manager.execute_agent("Dentist reminder", "fire", source="trigger"))
    assert len(dispatched) == 1 and log.replies == []
