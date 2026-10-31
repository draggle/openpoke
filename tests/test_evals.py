"""The evaluators' own plumbing: stratified workloads, fair history for both arms, intervals."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from evals.synthetic import generate


def test_stratified_workload_has_equal_kinds():
    w = generate(150, seed=7, n_queries=40, stratified=True)
    assert Counter(q.kind for q in w.queries) == {"keyword": 10, "name_only": 10, "topic_only": 10, "recency": 10}
    # still deterministic
    again = generate(150, seed=7, n_queries=40, stratified=True)
    assert [q.text for q in again.queries] == [q.text for q in w.queries]


def test_transcript_for_names_recent_agents():
    from evals.synthetic import transcript_for

    w = generate(150, seed=7, n_queries=40)
    transcript = transcript_for(w, k=6)
    lines = [l for l in transcript.splitlines() if l.strip()]
    assert len(lines) == 12  # a user message and an agent result per interaction
    assert all(l.startswith("<user_message") or l.startswith("<agent_message") for l in lines)
    most_recent = max(w.agents, key=lambda a: a.last_used_at)
    assert most_recent.name in lines[-1]  # the last thing that happened is the most recent agent
    named = [a.name for a in sorted(w.agents, key=lambda a: -a.last_used_at)[:6]]
    assert all(any(n in l for l in lines) for n in named)


def test_wilson():
    from evals.stats import wilson

    lo, hi = wilson(36, 40)
    assert (round(lo, 2), round(hi, 2)) == (0.77, 0.96)
    assert wilson(0, 0) == (0.0, 1.0)


async def test_decide_counts_search_calls(tmp_path, monkeypatch):
    from evals.live_routing_eval import decide
    from evals.roster_eval import build_registry
    from server.services.execution.registry import RegistryConfig
    from server.services.llm import gateway

    w = generate(60, seed=7, n_queries=5)
    registry = build_registry(w, tmp_path, RegistryConfig(strategy="budgeted"))
    target = w.queries[0].target
    calls = {"n": 0}

    async def scripted(*, model, messages, system=None, api_key=None, tools=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"choices": [{"message": {"content": "", "tool_calls": [{"id": "c1", "type": "function",
                    "function": {"name": "search_agents", "arguments": json.dumps({"query": target})}}]}}]}
        return {"choices": [{"message": {"content": "", "tool_calls": [{"id": "c2", "type": "function",
                "function": {"name": "send_message_to_agent",
                             "arguments": json.dumps({"agent_name": target, "instructions": "go"})}}]}}]}

    monkeypatch.setattr(gateway._openrouter, "request_chat_completion", scripted)
    decision = await decide(registry, w, w.queries[0].text, history=True)
    assert decision.agent == target
    assert decision.search_calls == 1
    assert decision.none_text == ""
