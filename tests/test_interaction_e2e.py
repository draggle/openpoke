"""End-to-end: the real InteractionAgentRuntime loop with a scripted LLM.

Checks that (1) the prompt the model sees carries a *bounded* roster with a roster note even
when 60 agents exist, (2) the model can call ``search_agents`` and get the hidden agent back,
and (3) delegating to that agent registers/touches it in the registry.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest

from server.agents.interaction_agent import agent as agent_module
from server.agents.interaction_agent import tools as tools_module
from server.agents.interaction_agent.runtime import InteractionAgentRuntime
from server.openrouter_client import client as oc
from server.services.execution.registry import AgentRegistry, RegistryConfig

DAY = 86400.0


class StubConversationLog:
    def __init__(self) -> None:
        self.entries = []

    def record_user_message(self, m): self.entries.append(("user", m))
    def record_agent_message(self, m): self.entries.append(("agent", m))
    def record_reply(self, m): self.entries.append(("reply", m))
    def record_wait(self, m): self.entries.append(("wait", m))
    def load_transcript(self): return ""


class StubRoster:
    def __init__(self): self.names = []
    def load(self): pass
    def get_agents(self): return list(self.names)
    def add_agent(self, n): self.names.append(n)


class StubLogs:
    def record_request(self, *a, **k): pass


def tool_call(name: str, args: dict, call_id: str = "call_1") -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


@pytest.fixture
def registry(tmp_path: Path) -> AgentRegistry:
    now = 1_800_000_000.0
    clock = {"t": now - 10 * DAY}
    reg = AgentRegistry(tmp_path / "registry.json", config=RegistryConfig(max_roster=12),
                        clock=lambda: clock["t"])
    reg.register("Invoice from Priya", "Chase Priya for the October design invoice")
    reg.update_summary("Invoice from Priya", "Sent reminder; awaiting Priya's reply")
    for i in range(59):
        clock["t"] += 3 * 3600
        reg.register(f"Agent {i}", f"Generic task {i} about topic{i}")
    clock["t"] = now
    return reg


async def test_bounded_roster_search_fallback_and_delegation(monkeypatch, registry):
    monkeypatch.setattr(agent_module, "get_agent_registry", lambda: registry)
    monkeypatch.setattr(tools_module, "get_agent_registry", lambda: registry)
    monkeypatch.setattr(tools_module, "get_agent_roster", lambda: StubRoster())
    monkeypatch.setattr(tools_module, "get_execution_agent_logs", lambda: StubLogs())
    conversation = StubConversationLog()
    monkeypatch.setattr(tools_module, "get_conversation_log", lambda: conversation)

    dispatched = {}

    async def fake_execute_agent(agent_name, instructions):
        dispatched["agent"] = agent_name
        from server.agents.execution_agent.runtime import ExecutionResult
        return ExecutionResult(agent_name=agent_name, success=True, response="ok")

    monkeypatch.setattr(tools_module._EXECUTION_BATCH_MANAGER, "execute_agent", fake_execute_agent)

    seen_prompts = []

    async def scripted_llm(*, model, messages, system=None, api_key=None, tools=None, base_url=None):
        seen_prompts.append(messages)
        n = len(seen_prompts)
        if n == 1:
            return {"choices": [{"message": {"content": "", "tool_calls": [
                tool_call("search_agents", {"query": "priya invoice"})]}}]}
        if n == 2:
            result = json.loads(messages[-1]["content"])
            names = [m["agent_name"] for m in result["result"]["matches"]]
            assert names[0] == "Invoice from Priya", names
            return {"choices": [{"message": {"content": "", "tool_calls": [
                tool_call("send_message_to_agent", {"agent_name": "Invoice from Priya",
                                                    "instructions": "Check if Priya replied"}, "call_2")]}}]}
        if n == 3:
            return {"choices": [{"message": {"content": "", "tool_calls": [
                tool_call("send_message_to_user", {"message": "checking with priya's thread now"}, "call_3")]}}]}
        return {"choices": [{"message": {"content": ""}}]}  # no tool calls -> loop ends

    monkeypatch.setattr(oc, "request_chat_completion", scripted_llm)

    runtime = InteractionAgentRuntime()
    runtime.conversation_log = conversation
    result = await runtime.execute("did priya ever get back to me about that invoice?")
    await asyncio.sleep(0)  # let the fire-and-forget execution task run

    assert result.success, result.error
    assert result.response == "checking with priya's thread now"

    # (1) bounded roster in the first prompt
    first = seen_prompts[0][0]["content"]
    roster = first.split("<active_agents>")[1].split("</active_agents>")[0]
    assert len(re.findall(r"<agent ", roster)) <= 12
    assert "<roster_note>" in roster and "60 agents" in roster
    # (2)+(3) search found the buried agent, delegation reached it, registry recorded the touch
    assert dispatched["agent"] == "Invoice from Priya"
    assert registry.get("Invoice from Priya").use_count == 2
    assert result.execution_agents_used == 1


async def test_all_strategy_prompt_matches_upstream_shape(monkeypatch, tmp_path):
    reg = AgentRegistry(tmp_path / "r.json", config=RegistryConfig(strategy="all"))
    reg.register("Alpha", "a")
    reg.register("Beta", "b")
    monkeypatch.setattr(agent_module, "get_agent_registry", lambda: reg)
    content = agent_module.prepare_message_with_history("hi", "", "user")[0]["content"]
    assert '<active_agents>\n<agent name="Alpha" />\n<agent name="Beta" />\n</active_agents>' in content


async def test_single_roster_selection_per_turn(monkeypatch, registry):
    """The roster is selected once per user turn; the routing-hint path reuses it."""
    monkeypatch.setattr(agent_module, "get_agent_registry", lambda: registry)
    conversation = StubConversationLog()
    calls = {"select": 0}
    real_select = registry.select

    def counting_select(*a, **k):
        calls["select"] += 1
        return real_select(*a, **k)

    monkeypatch.setattr(registry, "select", counting_select)

    async def scripted_llm(*, model, messages, system=None, api_key=None, tools=None, base_url=None):
        return {"choices": [{"message": {"content": "hello"}}]}

    monkeypatch.setattr(oc, "request_chat_completion", scripted_llm)
    runtime = InteractionAgentRuntime()
    runtime.conversation_log = conversation
    result = await runtime.execute("hi there")
    assert result.success
    assert calls["select"] == 1
