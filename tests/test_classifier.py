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
