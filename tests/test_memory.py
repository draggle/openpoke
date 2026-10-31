"""Unit tests for per-agent tiered memory and compaction."""

from __future__ import annotations

from pathlib import Path

import pytest

from server.services.execution.log_store import ExecutionAgentLogStore
from server.services.execution.memory import (
    AgentMemory,
    Extraction,
    HeuristicExtractor,
    LLMExtractor,
    MemoryConfig,
    MemoryState,
    Thread,
)


@pytest.fixture
def store(tmp_path: Path) -> ExecutionAgentLogStore:
    return ExecutionAgentLogStore(tmp_path / "logs")


def fill_log(store: ExecutionAgentLogStore, agent: str, rounds: int, payload_chars: int = 300, start: int = 0) -> None:
    for i in range(start, start + rounds):
        store.record_request(agent, f"Round {i}: ask Alice (alice@example.com) about item {i}")
        store.record_action(agent, f"Calling gmail_create_draft with: {{'to': 'alice@example.com'}}")
        store.record_tool_response(agent, "gmail_create_draft", "x" * payload_chars)
        store.record_agent_response(agent, f"Draft {i} created for Alice; awaiting her reply")


def make_memory(tmp_path, store, extractor=None, **cfg) -> AgentMemory:
    config = MemoryConfig(**{"budget_chars": 2000, "tail_entries": 4, **cfg})
    return AgentMemory(tmp_path / "mem", log_store=store, config=config,
                       extractor=extractor or HeuristicExtractor())


# ---- behaviour before compaction: identical to upstream ----------------------------


def test_young_agent_gets_full_transcript(tmp_path, store):
    fill_log(store, "a", rounds=1)
    mem = make_memory(tmp_path, store)
    assert mem.needs_compaction("a") is False
    assert mem.render_history("a") == store.load_transcript("a")


def test_disabled_memory_never_compacts(tmp_path, store):
    fill_log(store, "a", rounds=20)
    mem = make_memory(tmp_path, store, enabled=False)
    assert mem.needs_compaction("a") is False
    assert mem.render_history("a") == store.load_transcript("a")


# ---- compaction ---------------------------------------------------------------------


async def test_compaction_bounds_prompt_and_keeps_tail(tmp_path, store):
    fill_log(store, "a", rounds=20)  # 80 entries, ~8k+ chars
    mem = make_memory(tmp_path, store)
    before = len(mem.render_history("a"))
    assert mem.needs_compaction("a")

    state = await mem.compact("a")
    assert state is not None
    assert state.compacted_through == 80 - 4
    after = mem.render_history("a")

    assert len(after) < before / 2
    assert "# Memory" in after and "# Recent History (4 most recent entries" in after
    # the raw tail is verbatim
    assert "Round 19" in after
    # and older events survive as timeline episodes rather than raw payloads
    assert "Asked: Round 0" in after
    assert "xxxxxxxxxx" not in after.split("# Recent History")[0]


async def test_compaction_is_incremental(tmp_path, store):
    fill_log(store, "a", rounds=10)
    mem = make_memory(tmp_path, store)
    s1 = await mem.compact("a", force=True)
    fill_log(store, "a", rounds=10, start=10)
    s2 = await mem.compact("a", force=True)
    assert s2.compactions == 2
    assert s2.compacted_through > s1.compacted_through
    # episodes from both batches, no duplicates of batch one
    texts = [e.text for e in s2.episodes]
    assert texts.count("Asked: Round 0: ask Alice (alice@example.com) about item 0") == 1


async def test_heuristic_extracts_email_fact_and_thread(tmp_path, store):
    fill_log(store, "a", rounds=6)
    mem = make_memory(tmp_path, store)
    state = await mem.compact("a", force=True)
    facts = {f.key: f.value for f in state.active_facts()}
    assert facts.get("Alice email") == "alice@example.com"
    assert state.threads and state.threads[-1].waiting_on == "reply"
    assert "awaiting" in mem.one_line(state).lower() or "reply" in mem.one_line(state).lower()


# ---- fact invalidation (the knowledge-graph idea) -----------------------------------


class ScriptedExtractor:
    def __init__(self, *extractions: Extraction) -> None:
        self._queue = list(extractions)

    async def __call__(self, entries, state) -> Extraction:
        return self._queue.pop(0) if self._queue else Extraction()


async def test_new_value_invalidates_old_fact_but_keeps_history(tmp_path, store):
    fill_log(store, "a", rounds=4)
    ext = ScriptedExtractor(
        Extraction(facts=[("Lunch with Alice", "Friday 12:30")]),
        Extraction(facts=[("lunch with alice", "Thursday 1pm")]),  # key match is case-insensitive
    )
    mem = make_memory(tmp_path, store, extractor=ext)
    await mem.compact("a", force=True)
    fill_log(store, "a", rounds=2)
    state = await mem.compact("a", force=True)

    active = state.active_facts()
    assert len(active) == 1 and active[0].value == "Thursday 1pm"
    old = [f for f in state.facts if not f.active]
    assert len(old) == 1 and old[0].superseded_by == "Thursday 1pm"

    rendered = mem.render_history("a")
    assert "Thursday 1pm" in rendered
    assert "Friday 12:30 → Thursday 1pm" in rendered


async def test_same_value_does_not_duplicate_fact(tmp_path, store):
    fill_log(store, "a", rounds=4)
    ext = ScriptedExtractor(Extraction(facts=[("Bob email", "bob@x.com")]),
                            Extraction(facts=[("Bob email", "BOB@x.com")]))
    mem = make_memory(tmp_path, store, extractor=ext)
    await mem.compact("a", force=True)
    fill_log(store, "a", rounds=2)
    state = await mem.compact("a", force=True)
    assert len(state.facts) == 1


async def test_threads_open_update_and_resolve(tmp_path, store):
    fill_log(store, "a", rounds=4)
    ext = ScriptedExtractor(
        Extraction(threads=[Thread("Alice lunch", "sent invite", "reply")]),
        Extraction(threads=[Thread("alice lunch", "nudged her", "reply")]),
        Extraction(resolved_threads=["Alice lunch"]),
    )
    mem = make_memory(tmp_path, store, extractor=ext)
    await mem.compact("a", force=True)
    fill_log(store, "a", rounds=1); s2 = await mem.compact("a", force=True)
    assert len(s2.threads) == 1 and s2.threads[0].last_action == "nudged her"
    fill_log(store, "a", rounds=1); s3 = await mem.compact("a", force=True)
    assert s3.threads == []


# ---- commitments survive compaction (the thing summarisation tends to lose) ----------


async def test_reminder_commitment_survives_compaction(tmp_path, store):
    agent = "Dentist"
    store.record_request(agent, "Remind me Friday at 9am to call the dentist about the crown")
    store.record_action(agent, "Calling createTrigger with: {'start_time': '2026-11-06T09:00'}")
    store.record_tool_response(agent, "createTrigger", "{'id': 7, 'status': 'active'}")
    store.record_agent_response(agent, "Reminder set for Friday 9am (trigger 7) to call the dentist about the crown")
    fill_log(store, agent, rounds=15)  # lots of unrelated noise after it

    mem = make_memory(tmp_path, store)
    await mem.compact(agent, force=True)
    rendered = mem.render_history(agent)
    head = rendered.split("# Recent History")[0]
    assert "Friday" in head and "dentist" in head and "crown" in head


# ---- LLM extractor: parsing + fallback ----------------------------------------------


async def test_llm_extractor_parses_json_and_falls_back(tmp_path, store, monkeypatch):
    from server.openrouter_client import client as oc

    fill_log(store, "a", rounds=3)
    calls = {"n": 0}

    async def fake(*, model, messages, system=None, api_key=None, tools=None, base_url=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"choices": [{"message": {"content": '```json\n{"facts":[{"key":"Alice email","value":"a@b.c"}],"threads":[],"resolved_threads":[],"episodes":["Drafted lunch invite"]}\n```'}}]}
        raise RuntimeError("boom")

    monkeypatch.setattr(oc, "request_chat_completion", fake)
    ext = LLMExtractor()
    e1 = await ext(list(store.iter_entries("a")), MemoryState())
    assert e1.facts == [("Alice email", "a@b.c")] and e1.episodes == ["Drafted lunch invite"]
    e2 = await ext(list(store.iter_entries("a")), MemoryState())  # falls back to heuristic
    assert any(ep.startswith("Asked:") for ep in e2.episodes)


async def test_summary_callback_feeds_registry(tmp_path, store):
    seen = {}
    fill_log(store, "a", rounds=6)
    mem = AgentMemory(tmp_path / "mem", log_store=store,
                      config=MemoryConfig(budget_chars=100, tail_entries=2),
                      extractor=HeuristicExtractor(),
                      on_summary=lambda name, summary: seen.update({name: summary}))
    await mem.compact("a")
    assert "a" in seen and seen["a"]
