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


class StubMemory:
    def note_request(self, *a, **k): pass


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
    monkeypatch.setattr(tools_module, "get_agent_memory", lambda: StubMemory())
    conversation = StubConversationLog()
    monkeypatch.setattr(tools_module, "get_conversation_log", lambda: conversation)

    dispatched = {}

    async def fake_execute_agent(agent_name, instructions, send_authorisation=None):
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
    # Keyword-free follow-up: nothing in it retrieves the Priya agent, so it is genuinely
    # outside the first roster and only search_agents can bring it back.
    result = await runtime.execute("did she ever get back to me about that thing?")
    await asyncio.sleep(0)  # let the fire-and-forget execution task run

    assert result.success, result.error
    assert result.response == "checking with priya's thread now"

    # (1) bounded roster in the first prompt, and the target is NOT in it
    first = seen_prompts[0][0]["content"]
    roster = first.split("<active_agents>")[1].split("</active_agents>")[0]
    assert len(re.findall(r"<agent ", roster)) <= 12
    assert "<roster_note>" in roster and "60 agents" in roster
    assert "Invoice from Priya" not in roster
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


def test_agent_names_in_transcript():
    transcript = "\n".join([
        '<user_message timestamp="t1">chase tara for the invoice</user_message>',
        '<agent_message timestamp="t2">[SUCCESS] Tara October Invoice: sent a nudge</agent_message>',
        '<poke_reply timestamp="t3">done</poke_reply>',
        '<user_message timestamp="t4">book the dentist</user_message>',
        '<agent_message timestamp="t5">[SUCCESS] Dentist Appointment with Alice: booked Thursday</agent_message>',
    ])
    known = ["Tara October Invoice", "Dentist Appointment with Alice", "Alice Lunch", "Tara Conference Talk"]
    assert agent_module.agent_names_in_transcript(transcript, known) == [
        "Dentist Appointment with Alice", "Tara October Invoice",
    ]
    assert agent_module.agent_names_in_transcript("", known) == []


def test_system_prompt_names_last_touched_rule():
    from server.agents.interaction_agent.agent import build_system_prompt

    prompt = build_system_prompt()
    assert 'last_touched="true"' in prompt
    paragraph = next(p for p in prompt.split("\n") if "names a person" in p)
    assert "search_agents" in paragraph


def _guard_registry(tmp_path):
    reg = AgentRegistry(tmp_path / "g.json", config=RegistryConfig(max_roster=12))
    reg.register("Follow up with Tomasz - flight", "Check whether Tomasz has replied about the flight confirmation")
    reg.register("Flight Confirmation with Bob", "Find the latest email from Bob about the flight confirmation")
    return reg


def test_send_message_guard_fires_once(monkeypatch, tmp_path):
    reg = _guard_registry(tmp_path)
    monkeypatch.setattr(tools_module, "get_agent_registry", lambda: reg)
    monkeypatch.setattr(tools_module, "get_agent_roster", lambda: StubRoster())
    monkeypatch.setattr(tools_module, "get_execution_agent_logs", lambda: StubLogs())
    monkeypatch.setattr(tools_module, "get_agent_memory", lambda: StubMemory())
    dispatched = []

    async def fake_execute_agent(agent_name, instructions, send_authorisation=None):
        dispatched.append(agent_name)
        from server.agents.execution_agent.runtime import ExecutionResult
        return ExecutionResult(agent_name=agent_name, success=True, response="ok")

    monkeypatch.setattr(tools_module._EXECUTION_BATCH_MANAGER, "execute_agent", fake_execute_agent)
    monkeypatch.delenv("OPENPOKE_ROUTING_GUARD", raising=False)
    tools_module._GUARD_SEEN.clear()

    async def run():
        first = tools_module.send_message_to_agent("Follow up with Tomasz - flight",
                                                   "What did Bob say about the flight confirmation?")
        assert not first.success
        assert first.payload["did_you_mean"][0] == "Flight Confirmation with Bob"
        assert dispatched == []
        second = tools_module.send_message_to_agent("Follow up with Tomasz - flight",
                                                    "What did Bob say about the flight confirmation?")
        assert second.success
        await asyncio.sleep(0)
        assert dispatched == ["Follow up with Tomasz - flight"]

    asyncio.run(run())


def test_send_message_guard_flag_off(monkeypatch, tmp_path):
    reg = _guard_registry(tmp_path)
    monkeypatch.setattr(tools_module, "get_agent_registry", lambda: reg)
    monkeypatch.setattr(tools_module, "get_agent_roster", lambda: StubRoster())
    monkeypatch.setattr(tools_module, "get_execution_agent_logs", lambda: StubLogs())
    monkeypatch.setattr(tools_module, "get_agent_memory", lambda: StubMemory())

    async def fake_execute_agent(agent_name, instructions, send_authorisation=None):
        from server.agents.execution_agent.runtime import ExecutionResult
        return ExecutionResult(agent_name=agent_name, success=True, response="ok")

    monkeypatch.setattr(tools_module._EXECUTION_BATCH_MANAGER, "execute_agent", fake_execute_agent)
    monkeypatch.setenv("OPENPOKE_ROUTING_GUARD", "0")
    tools_module._GUARD_SEEN.clear()

    async def run():
        first = tools_module.send_message_to_agent("Follow up with Tomasz - flight",
                                                   "What did Bob say about the flight confirmation?")
        assert first.success

    asyncio.run(run())


# ---- round three: the turn ends once the user is told and the agent dispatched ------------


def _loop_fixture(monkeypatch, tmp_path, script, extra_agents=()):
    reg = AgentRegistry(tmp_path / "l.json", config=RegistryConfig(max_roster=12))
    reg.register("Alice Lunch", "schedule lunch with Alice")
    for name, description in extra_agents:
        reg.register(name, description)
    tools_module._GUARD_SEEN.clear()
    monkeypatch.setattr(agent_module, "get_agent_registry", lambda: reg)
    monkeypatch.setattr(tools_module, "get_agent_registry", lambda: reg)
    monkeypatch.setattr(tools_module, "get_agent_roster", lambda: StubRoster())
    monkeypatch.setattr(tools_module, "get_execution_agent_logs", lambda: StubLogs())
    monkeypatch.setattr(tools_module, "get_agent_memory", lambda: StubMemory())
    conversation = StubConversationLog()
    monkeypatch.setattr(tools_module, "get_conversation_log", lambda: conversation)

    async def fake_execute_agent(agent_name, instructions, send_authorisation=None):
        from server.agents.execution_agent.runtime import ExecutionResult
        return ExecutionResult(agent_name=agent_name, success=True, response="ok")

    monkeypatch.setattr(tools_module._EXECUTION_BATCH_MANAGER, "execute_agent", fake_execute_agent)
    calls = {"n": 0}

    async def scripted_llm(*, model, messages, system=None, api_key=None, tools=None, base_url=None):
        calls["n"] += 1
        step = script[min(calls["n"], len(script)) - 1]
        return {"choices": [{"message": {"content": "", "tool_calls": step}}]} if step else \
            {"choices": [{"message": {"content": ""}}]}

    monkeypatch.setattr(oc, "request_chat_completion", scripted_llm)
    runtime = InteractionAgentRuntime()
    runtime.conversation_log = conversation
    return runtime, calls


ANNOUNCE = tool_call("send_message_to_user", {"message": "on it"}, "u1")
DELEGATE = tool_call("send_message_to_agent", {"agent_name": "Alice Lunch", "instructions": "move lunch"}, "a1")


async def test_early_exit_after_announce_and_delegate(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENPOKE_INTERACTION_EARLY_EXIT", raising=False)
    runtime, calls = _loop_fixture(monkeypatch, tmp_path, [[ANNOUNCE, DELEGATE], None])
    result = await runtime.execute("move lunch with alice to thursday")
    assert result.success and result.response == "on it"
    assert calls["n"] == 1


async def test_no_early_exit_after_announce_only(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENPOKE_INTERACTION_EARLY_EXIT", raising=False)
    runtime, calls = _loop_fixture(monkeypatch, tmp_path, [[ANNOUNCE], [DELEGATE], None])
    result = await runtime.execute("move lunch with alice to thursday")
    assert result.success and result.execution_agents_used == 1
    assert calls["n"] == 2  # announce (keep going) -> delegate (exit); the empty third call is skipped


async def test_agent_turn_exits_after_reply(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENPOKE_INTERACTION_EARLY_EXIT", raising=False)
    runtime, calls = _loop_fixture(monkeypatch, tmp_path, [[ANNOUNCE], None])
    result = await runtime.handle_agent_message("[SUCCESS] Alice Lunch: moved to Thursday")
    assert result.success and result.response == "on it"
    assert calls["n"] == 1


async def test_early_exit_flag_off_keeps_upstream_loop(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENPOKE_INTERACTION_EARLY_EXIT", "0")
    runtime, calls = _loop_fixture(monkeypatch, tmp_path, [[ANNOUNCE, DELEGATE], None])
    result = await runtime.execute("move lunch with alice to thursday")
    assert result.success
    assert calls["n"] == 2


def test_send_message_guard_fires_once_per_turn_even_when_rephrased(monkeypatch, tmp_path):
    reg = _guard_registry(tmp_path)
    monkeypatch.setattr(tools_module, "get_agent_registry", lambda: reg)
    monkeypatch.setattr(tools_module, "get_agent_roster", lambda: StubRoster())
    monkeypatch.setattr(tools_module, "get_execution_agent_logs", lambda: StubLogs())
    monkeypatch.setattr(tools_module, "get_agent_memory", lambda: StubMemory())

    async def fake_execute_agent(agent_name, instructions, send_authorisation=None):
        from server.agents.execution_agent.runtime import ExecutionResult
        return ExecutionResult(agent_name=agent_name, success=True, response="ok")

    monkeypatch.setattr(tools_module._EXECUTION_BATCH_MANAGER, "execute_agent", fake_execute_agent)
    monkeypatch.delenv("OPENPOKE_ROUTING_GUARD", raising=False)

    async def run():
        tools_module.begin_turn()
        first = tools_module.send_message_to_agent("Follow up with Tomasz - flight", "What did Bob say about the flight?")
        assert not first.success
        second = tools_module.send_message_to_agent("Follow up with Tomasz - flight", "Check Bob's reply on the flight confirmation")
        assert second.success  # rephrased, same turn: honoured, no loop
        tools_module.begin_turn()
        third = tools_module.send_message_to_agent("Follow up with Tomasz - flight", "What did Bob say about the flight?")
        assert not third.success  # a new turn gets one fresh challenge

    asyncio.run(run())


async def test_runtime_begins_a_turn_per_message(monkeypatch, tmp_path):
    runtime, calls = _loop_fixture(monkeypatch, tmp_path, [None])
    before = tools_module.current_turn()
    await runtime.execute("hi")
    await runtime.handle_agent_message("[SUCCESS] Alice Lunch: ok")
    assert tools_module.current_turn() == before + 2


# ---- hardening: `wait` is not a reply; delegate-then-reply exits; a guarded dispatch loops ---

WAIT = tool_call("wait", {"reason": "waiting for the agent"}, "w1")


async def test_wait_after_delegate_is_not_a_reply(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENPOKE_INTERACTION_EARLY_EXIT", raising=False)
    runtime, calls = _loop_fixture(monkeypatch, tmp_path, [[DELEGATE, WAIT], [ANNOUNCE], None])
    result = await runtime.execute("move lunch with alice to thursday")
    assert result.success and result.response == "on it"
    assert calls["n"] == 2  # the wait did not end the turn; the reply did


async def test_delegate_then_reply_exits_without_a_third_call(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENPOKE_INTERACTION_EARLY_EXIT", raising=False)
    runtime, calls = _loop_fixture(monkeypatch, tmp_path, [[DELEGATE], [ANNOUNCE], None])
    result = await runtime.execute("move lunch with alice to thursday")
    assert result.success and result.response == "on it" and result.execution_agents_used == 1
    assert calls["n"] == 2


async def test_guard_failed_dispatch_keeps_the_loop_going(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENPOKE_INTERACTION_EARLY_EXIT", raising=False)
    monkeypatch.delenv("OPENPOKE_ROUTING_GUARD", raising=False)
    wrong = tool_call("send_message_to_agent",
                      {"agent_name": "Alice Lunch", "instructions": "What did Bob say about the flight confirmation?"}, "g1")
    right = tool_call("send_message_to_agent",
                      {"agent_name": "Bob Flight", "instructions": "What did Bob say about the flight confirmation?"}, "g2")
    runtime, calls = _loop_fixture(monkeypatch, tmp_path, [[ANNOUNCE, wrong], [right], None],
                                   extra_agents=[("Bob Flight", "Find the latest email from Bob about the flight confirmation")])
    result = await runtime.execute("what did bob say about the flight?")
    assert result.success and result.execution_agents_used == 1
    assert calls["n"] == 2  # guard refused the first pick; the corrected pick ended the turn


def test_system_prompt_says_new_unrelated_reminders_get_their_own_agent():
    """A message like "also remind me Friday to call the dentist" must not be filed under the
    invoice agent just because that agent owns a reminder and sits at the top of the roster."""
    prompt = agent_module.build_system_prompt()
    paragraph = next(p for p in prompt.split("\n") if "has_reminders" in p)
    assert "own agent" in paragraph and "different matter" in paragraph


def test_system_prompt_says_cards_are_for_choosing_not_answering():
    prompt = agent_module.build_system_prompt()
    paragraph = next(p for p in prompt.split("\n") if "may be stale" in p)
    assert "delegate" in paragraph.lower() and "replied" in paragraph
