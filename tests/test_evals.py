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


def test_memory_eval_detects_stale_value():
    from evals.memory_eval import Needle, score_retention

    needles = [Needle("changed", "$700", 4, stale="$500"), Needle("buried", "REF-1", 1),
               Needle("constraint", "Do not mention the deposit.", 2)]
    kept_old = "## Facts\n- invoice: $500\n\n# Recent History (2 most recent entries, verbatim)\n<x>z</x>"
    r = score_retention(needles, kept_old, search=lambda q: [])
    assert r["changed_current"] == 0.0 and r["stale"] == 1.0
    assert r["buried"] == 0.0 and r["recoverable"] == 0.0 and r["constraint"] == 0.0

    kept_new = ("## Commitments (verbatim)\n- Do not mention the deposit.\n## Facts\n- invoice: $700\n"
                "## Changed facts (old → new)\n- invoice: $500 → $700\n\n# Recent History (0 most recent entries, verbatim)\n")
    r = score_retention(needles, kept_new, search=lambda q: [{"text": "reference REF-1"}] if "REF-1" in q else [])
    assert r["changed_current"] == 1.0 and r["stale"] == 0.0
    assert r["buried"] == 0.0 and r["recoverable"] == 1.0 and r["constraint"] == 1.0

    # The timeline may legitimately narrate the change ("now $700, was $500"); only the Facts
    # block asserts what is current, so an old value there is not stale.
    narrated = ("## Facts\n- invoice: $700\n## Timeline\n- [t] Updated: the invoice is $700 (was $500).\n\n"
                "# Recent History (0 most recent entries, verbatim)\n")
    r = score_retention(needles, narrated, search=lambda q: [])
    assert r["changed_current"] == 1.0 and r["stale"] == 0.0

    # A fact that carries its own correction history is current, not stale.
    corrected = "## Facts\n- deposit: $700 (corrected from $500)\n- rent: $900, was $850\n\n# Recent History (0 most recent entries, verbatim)\n"
    r = score_retention(needles, corrected, search=lambda q: [])
    assert r["changed_current"] == 1.0 and r["stale"] == 0.0
    # ...but a standalone old value in Facts is.
    both = "## Facts\n- deposit: $700\n- deposit (old): $500\n\n# Recent History (0 most recent entries, verbatim)\n"
    assert score_retention(needles, both, search=lambda q: [])["stale"] == 1.0

    # An uncompacted agent (raw transcript, no memory section) asserts nothing as current, so
    # the old value in the raw log is not "stale memory".
    raw = "<agent_request>invoice $500</agent_request>\n<agent_request>now $700 not $500</agent_request>"
    r = score_retention(needles, raw, search=lambda q: [])
    assert r["changed_current"] == 1.0 and r["stale"] == 0.0


def test_paraphrase_set_resolves_targets():
    from evals.paraphrases import PARAPHRASES, resolve

    w = generate(150, seed=7)
    assert len(PARAPHRASES) >= 30
    unresolved = [p["text"] for p in PARAPHRASES if resolve(w, p) is None]
    assert unresolved == [], unresolved
    # every paraphrase avoids the templated wording the lexical index was built from
    banned = ("what did", "any update on", "get back to me", "cancel the", "remind me what", "did .* reply",
              "anything from", "nudge .* again", "what's the latest", "where are we on", "sorted yet")
    import re
    for p in PARAPHRASES:
        assert not any(re.search(b, p["text"].lower()) for b in banned), p["text"]


def test_cost_sensitivity_rows(tmp_path):
    from evals.cost_eval import sensitivity
    from server.services.llm.router import ModelRouter

    rows = sensitivity(ModelRouter(), agents=200, history_entries=(0, 40, 80, 160), runs_per_turn=(0.5, 1.2),
                       out_dir=tmp_path)
    assert len(rows) == 8
    for runs in (0.5, 1.2):
        sub = [r for r in rows if r["execution_runs_per_user_turn"] == runs]
        savings = {r["history_entries"]: r["saving_pct"] for r in sub}
        # Short histories never reach the compaction budget, so their saving is roster+routing
        # only; the memory saving appears once the log is long enough to compact.
        assert savings[160] > savings[0], savings
        assert all(0 <= s < 100 for s in savings.values())
