"""Hardening: two runs for the same agent never interleave (their log and raw-journal writes
would otherwise swap request contexts); runs for different agents still overlap."""

from __future__ import annotations

import asyncio

from server.agents.execution_agent import batch_manager as bm
from server.agents.execution_agent.runtime import ExecutionResult


def _manager_with_fake_runtime(monkeypatch, events):
    class FakeRuntime:
        def __init__(self, agent_name, source="user", send_authorisation=None):
            self.agent_name = agent_name

        async def execute(self, instructions):
            events.append(("start", self.agent_name, instructions))
            await asyncio.sleep(0.02)
            events.append(("end", self.agent_name, instructions))
            return ExecutionResult(agent_name=self.agent_name, success=True, response="ok")

    monkeypatch.setattr(bm, "ExecutionAgentRuntime", FakeRuntime)
    manager = bm.ExecutionBatchManager(timeout_seconds=5)

    async def no_dispatch(*a, **k):
        return None

    monkeypatch.setattr(manager, "_complete_execution", no_dispatch)
    return manager


async def test_runs_for_the_same_agent_are_serialised(monkeypatch):
    events = []
    manager = _manager_with_fake_runtime(monkeypatch, events)
    await asyncio.gather(manager.execute_agent("Alice Lunch", "first"), manager.execute_agent("Alice Lunch", "second"))
    assert events == [("start", "Alice Lunch", "first"), ("end", "Alice Lunch", "first"),
                      ("start", "Alice Lunch", "second"), ("end", "Alice Lunch", "second")]


async def test_runs_for_different_agents_overlap(monkeypatch):
    events = []
    manager = _manager_with_fake_runtime(monkeypatch, events)
    await asyncio.gather(manager.execute_agent("Alice Lunch", "a"), manager.execute_agent("Bob Flight", "b"))
    assert [e[0] for e in events] == ["start", "start", "end", "end"]
