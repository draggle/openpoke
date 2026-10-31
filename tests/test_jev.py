"""Jev integration: typed decisions gate the chat model, and degrade gracefully."""

from __future__ import annotations

import json

import pytest

from evals.email_eval import load_fixtures, to_processed
from server.agents.interaction_agent import agent as agent_module
from server.openrouter_client import client as oc
from server.services.execution.registry import AgentRegistry, RegistryConfig
from server.services.execution.routing_hint import NONE_OPTION, compute_routing_hint
from server.services.gmail.importance_classifier import classify_email_importance
from server.services.llm import get_usage_ledger
from server.services.llm.jev import Answer, FakeJev, JevError, _parse_answer, choice, set_jev_client

ROWS = {r["id"]: r for r in load_fixtures()}


@pytest.fixture
def llm_calls(monkeypatch):
    calls = []

    async def fake(*, model, messages, system=None, api_key=None, tools=None, base_url=None):
        calls.append(messages)
        return {"choices": [{"message": {"content": "", "tool_calls": [{
            "id": "c1", "type": "function",
            "function": {"name": "mark_email_importance",
                         "arguments": json.dumps({"important": True, "summary": "Alice confirmed lunch."})}}]}}]}

    monkeypatch.setattr(oc, "request_chat_completion", fake)
    return calls


@pytest.fixture
def jev_scripted():
    def install(prob_by_state=None, choice_answer=None):
        def answers_for(state, questions):
            out = {}
            for qid, q in questions.items():
                if q["type"] == "noul":
                    p = 0.9 if (prob_by_state is None) else prob_by_state(state)
                    out[qid] = Answer(kind="noul", probability=p)
                else:
                    label, conf = choice_answer(state, q["criteria"])
                    out[qid] = Answer(kind="choice", choice=label, confidence=conf,
                                      probabilities={label: conf})
            return out
        fake = FakeJev(answers_for)
        set_jev_client(fake)
        return fake
    yield install
    set_jev_client(None)


# ---- parsing ---------------------------------------------------------------------------


def test_parse_answer_shapes():
    assert _parse_answer({"noul": 0.73}).probability == 0.73
    a = _parse_answer({"choice": "x", "probabilities": {"x": 0.8, "y": 0.2}, "confidence": 0.8})
    assert a.choice == "x" and a.confidence == 0.8
    a2 = _parse_answer({"selected": "y", "distribution": [{"label": "y", "probability": 0.6}]})
    assert a2.probabilities == {"y": 0.6}
    with pytest.raises(JevError):
        _parse_answer({"weird": 1})


def test_choice_option_limit():
    with pytest.raises(JevError):
        choice("q", {str(i): "" for i in range(256)})


# ---- email gating ----------------------------------------------------------------------


async def test_jev_low_probability_skips_llm(jev_scripted, llm_calls):
    fake = jev_scripted(prob_by_state=lambda s: 0.05)
    out = await classify_email_importance(to_processed(ROWS["n15"]))
    assert out is None
    assert len(fake.calls) == 1 and llm_calls == []
    [rec] = get_usage_ledger().records()
    assert rec.role == "jev_email_importance" and rec.completion_tokens == 0


async def test_jev_high_probability_lets_llm_summarise(jev_scripted, llm_calls):
    jev_scripted(prob_by_state=lambda s: 0.95)
    out = await classify_email_importance(to_processed(ROWS["i01"]))
    assert out == "Alice confirmed lunch."
    assert len(llm_calls) == 1


async def test_jev_failure_falls_back_to_llm(llm_calls):
    class Broken:
        async def ask(self, *a, **k):
            raise JevError("down")
    set_jev_client(Broken())
    try:
        out = await classify_email_importance(to_processed(ROWS["i01"]))
    finally:
        set_jev_client(None)
    assert out == "Alice confirmed lunch." and len(llm_calls) == 1


async def test_jev_disabled_is_upstream_behaviour(llm_calls, monkeypatch):
    monkeypatch.delenv("OPENPOKE_JEV", raising=False)
    set_jev_client(None)
    out = await classify_email_importance(to_processed(ROWS["i01"]))
    assert out == "Alice confirmed lunch." and len(llm_calls) == 1


async def test_threshold_env(jev_scripted, llm_calls, monkeypatch):
    monkeypatch.setenv("OPENPOKE_JEV_EMAIL_THRESHOLD", "0.95")
    jev_scripted(prob_by_state=lambda s: 0.9)
    assert await classify_email_importance(to_processed(ROWS["i01"])) is None
    assert llm_calls == []


# ---- routing hint ----------------------------------------------------------------------


@pytest.fixture
def registry(tmp_path):
    reg = AgentRegistry(tmp_path / "r.json", config=RegistryConfig(max_roster=5))
    reg.register("Invoice from Priya", "Chase Priya for the October invoice")
    reg.register("Lunch with Alice", "Invite Alice to lunch")
    return reg


async def test_routing_hint_confident(jev_scripted, registry):
    fake = jev_scripted(choice_answer=lambda s, opts: ("Invoice from Priya", 0.92))
    sel = registry.select("did priya reply?")
    hint = await compute_routing_hint(registry, sel, "did priya reply?")
    assert hint and hint.agent_name == "Invoice from Priya" and hint.confidence == 0.92
    assert 'agent="Invoice from Priya"' in hint.render()
    # candidates offered = shown roster + none
    offered = fake.calls[0]["questions"]["agent"]["criteria"]
    assert set(offered) == {"Invoice from Priya", "Lunch with Alice", NONE_OPTION}


async def test_routing_hint_none_and_low_confidence(jev_scripted, registry):
    jev_scripted(choice_answer=lambda s, opts: (NONE_OPTION, 0.8))
    hint = await compute_routing_hint(registry, registry.select("book me a dentist"), "book me a dentist")
    assert hint and hint.agent_name is None and 'agent=""' in hint.render()

    jev_scripted(choice_answer=lambda s, opts: ("Lunch with Alice", 0.3))
    assert await compute_routing_hint(registry, registry.select("hmm"), "hmm") is None


async def test_routing_hint_injected_into_prompt(jev_scripted, registry, monkeypatch):
    jev_scripted(choice_answer=lambda s, opts: ("Lunch with Alice", 0.9))
    monkeypatch.setattr(agent_module, "get_agent_registry", lambda: registry)
    hint = await agent_module.build_routing_hint("is alice coming to lunch")
    content = agent_module.prepare_message_with_history("is alice coming to lunch", "", "user", routing_hint=hint)[0]["content"]
    assert "<routing_hint" in content and content.index("<routing_hint") < content.index("<new_user_message>")


async def test_no_hint_when_jev_disabled(registry, monkeypatch):
    set_jev_client(None)
    monkeypatch.delenv("OPENPOKE_JEV", raising=False)
    monkeypatch.setattr(agent_module, "get_agent_registry", lambda: registry)
    assert await agent_module.build_routing_hint("is alice coming") is None
