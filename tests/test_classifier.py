"""Importance classifier: the prompt names the categories it missed; Jev-decided mail survives."""

from __future__ import annotations

import json

import pytest

from evals.email_eval import load_fixtures, to_processed
from server.openrouter_client import client as oc
from server.services.gmail import importance_classifier as ic
from server.services.llm.jev import Answer, FakeJev, set_jev_client

ROWS = {r["id"]: r for r in load_fixtures()}


def test_prompt_lists_categories():
    prompt = ic._SYSTEM_PROMPT.lower()
    for needle in ("balance", "recruiter", "immigration", "replying", "unsubscribe", "declined", "payroll"):
        assert needle in prompt, needle


@pytest.fixture
def jev_says_important():
    fake = FakeJev(lambda state, questions: {q: Answer(kind="noul", probability=0.9) for q in questions})
    set_jev_client(fake)
    yield fake
    set_jev_client(None)


async def test_summary_only_fallback_when_llm_gives_no_summary(jev_says_important, monkeypatch):
    async def fake(*, model, messages, system=None, api_key=None, tools=None, base_url=None):
        return {"choices": [{"message": {"content": "", "tool_calls": [{
            "id": "c1", "type": "function",
            "function": {"name": "mark_email_importance", "arguments": json.dumps({"important": False})}}]}}]}

    monkeypatch.setattr(oc, "request_chat_completion", fake)
    out = await ic.classify_email_importance(to_processed(ROWS["a01"]))
    assert isinstance(out, str) and "Low balance alert" in out and "Chase" in out


async def test_summary_only_fallback_when_llm_makes_no_tool_call(jev_says_important, monkeypatch):
    async def fake(*, model, messages, system=None, api_key=None, tools=None, base_url=None):
        return {"choices": [{"message": {"content": "Sure."}}]}

    monkeypatch.setattr(oc, "request_chat_completion", fake)
    out = await ic.classify_email_importance(to_processed(ROWS["a05"]))
    assert isinstance(out, str) and "visa application" in out


# ---- hardening: Jev-decided mail survives a transport error; bad legacy threshold ignored ---


async def test_jev_decided_mail_survives_llm_transport_error(jev_says_important, monkeypatch):
    from server.openrouter_client import OpenRouterError

    async def fake(*, model, messages, system=None, api_key=None, tools=None, base_url=None):
        raise OpenRouterError("502 bad gateway")

    monkeypatch.setattr(oc, "request_chat_completion", fake)
    out = await ic.classify_email_importance(to_processed(ROWS["a01"]))
    assert isinstance(out, str) and "Low balance alert" in out


async def test_jev_decided_mail_survives_invalid_tool_arguments(jev_says_important, monkeypatch):
    async def fake(*, model, messages, system=None, api_key=None, tools=None, base_url=None):
        return {"choices": [{"message": {"content": "", "tool_calls": [{
            "id": "c1", "type": "function",
            "function": {"name": "mark_email_importance", "arguments": "{not json"}}]}}]}

    monkeypatch.setattr(oc, "request_chat_completion", fake)
    out = await ic.classify_email_importance(to_processed(ROWS["a05"]))
    assert isinstance(out, str) and "visa application" in out


async def test_undecided_mail_is_still_dropped_on_transport_error(monkeypatch):
    from server.openrouter_client import OpenRouterError

    set_jev_client(None)

    async def fake(*, model, messages, system=None, api_key=None, tools=None, base_url=None):
        raise OpenRouterError("502 bad gateway")

    monkeypatch.setattr(oc, "request_chat_completion", fake)
    assert await ic.classify_email_importance(to_processed(ROWS["a01"])) is None


def test_non_numeric_legacy_threshold_falls_back_to_defaults(monkeypatch):
    monkeypatch.delenv("OPENPOKE_JEV_EMAIL_LOW", raising=False)
    monkeypatch.delenv("OPENPOKE_JEV_EMAIL_HIGH", raising=False)
    monkeypatch.setenv("OPENPOKE_JEV_EMAIL_THRESHOLD", "half")
    assert ic.gate_thresholds() == ic.DEFAULT_GATE


def test_default_low_gate_sits_below_the_lowest_recorded_positive():
    """a04 (the recruiter) was recorded at p = 0.10; a strict ``<`` at exactly 0.1 is one
    rounding error from dismissing it."""
    low, high = ic.DEFAULT_GATE
    assert low < 0.10 and high == 0.5
