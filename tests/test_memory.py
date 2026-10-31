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


# ---- regressions from review ---------------------------------------------------------


async def test_cleared_log_resets_memory(tmp_path, store):
    fill_log(store, "a", rounds=10)
    mem = make_memory(tmp_path, store)
    await mem.compact("a", force=True)
    store.clear_all()
    assert mem.load("a").compacted_through == 0
    assert mem.render_history("a") == ""


async def test_response_only_resolves_matching_thread(tmp_path, store):
    agent = "a"
    store.record_request(agent, "Ask Alice about lunch")
    store.record_agent_response(agent, "Draft sent to Alice about lunch; awaiting reply")
    store.record_request(agent, "Ask Bob about the contract")
    store.record_agent_response(agent, "Draft sent to Bob about the contract; awaiting reply")
    mem = make_memory(tmp_path, store, tail_entries=0)
    s1 = await mem.compact(agent, force=True)
    assert len(s1.threads) == 2
    store.record_request(agent, "check")
    store.record_agent_response(agent, "Bob replied and confirmed the contract is signed")
    s2 = await mem.compact(agent, force=True)
    subjects = [t.subject for t in s2.threads]
    assert any("Alice" in s for s in subjects) and not any("Bob" in s for s in subjects)


async def test_active_facts_are_capped(tmp_path, store):
    fill_log(store, "a", rounds=2)
    ext = ScriptedExtractor(Extraction(facts=[(f"k{i}", f"v{i}") for i in range(100)]))
    mem = make_memory(tmp_path, store, extractor=ext, max_facts=20)
    state = await mem.compact("a", force=True)
    assert len(state.facts) == 20 and state.facts[-1].key == "k99"


async def test_concurrent_compactions_do_not_duplicate(tmp_path, store):
    import asyncio

    fill_log(store, "a", rounds=10)
    mem = make_memory(tmp_path, store)
    await asyncio.gather(mem.compact("a", force=True), mem.compact("a", force=True))
    state = mem.load("a")
    assert state.compactions == 1
    texts = [e.text for e in state.episodes]
    assert len(texts) == len(set(texts))


# ---- round two: no silent loss, atomic save -----------------------------------------


async def test_empty_extraction_falls_back_to_heuristic(tmp_path, store):
    """An LLM answer with no facts/episodes must not swallow the batch: the heuristic absorbs it."""
    class Nothing:
        async def __call__(self, entries, state):
            return Extraction()

    fill_log(store, "a", rounds=10)
    mem = make_memory(tmp_path, store, extractor=Nothing())
    state = await mem.compact("a", force=True)
    assert state is not None and state.compacted_through > 0
    assert state.episodes, "fallback extractor should have produced the timeline"


async def test_unextractable_batch_does_not_advance_cursor(tmp_path, store):
    """When neither extractor finds anything, the cursor stays so the entries remain in the prompt."""
    class Nothing:
        async def __call__(self, entries, state):
            return Extraction()

    for i in range(10):
        store.record_tool_response("a", "noop", "{}")  # nothing a heuristic can name
    mem = make_memory(tmp_path, store, extractor=Nothing())
    assert await mem.compact("a", force=True) is None
    assert mem.load("a").compacted_through == 0


def test_save_is_atomic(tmp_path, store, monkeypatch):
    """A crash mid-write must never leave a truncated memory file: write to a temp file, then replace."""
    import os
    from server.services.execution import memory as memory_module

    replaced = []
    real_replace = os.replace
    monkeypatch.setattr(memory_module.os, "replace", lambda src, dst: (replaced.append((src, dst)), real_replace(src, dst)))
    fill_log(store, "a", rounds=2)
    mem = make_memory(tmp_path, store)
    mem.save("a", MemoryState(compactions=1))
    assert replaced and str(replaced[0][1]).endswith("a.memory.json")
    assert not list((tmp_path / "mem").glob("*.tmp"))
    assert mem.load("a").compactions == 1


# ---- round two: fact keys are normalised before supersession ------------------------


def test_normalise_key():
    from server.services.execution.memory import normalise_key
    assert normalise_key("Alice's Email Address") == "alice email address"
    assert normalise_key("the lunch date") == "lunch date"
    assert normalise_key("Bob’s phone #") == "bob phone"


async def test_fact_keys_normalised_for_supersession(tmp_path, store):
    fill_log(store, "a", rounds=4)
    ext = ScriptedExtractor(
        Extraction(facts=[("Alice's email address", "a@x")]),
        Extraction(facts=[("alice email address", "b@x")]),
    )
    mem = make_memory(tmp_path, store, extractor=ext)
    await mem.compact("a", force=True)
    fill_log(store, "a", rounds=2)
    state = await mem.compact("a", force=True)
    active = [f for f in state.active_facts() if "email" in f.key.lower()]
    assert len(active) == 1 and active[0].value == "b@x"
    assert any(f.superseded_by == "b@x" for f in state.facts if not f.active)


# ---- round two: commitments tier, verbatim and never re-summarised -------------------


def test_extract_commitments():
    from server.services.execution.memory import extract_commitments
    text = "Remind me Friday at 9am about the crown. Also never email Bob after 6pm. Find the latest invoice from Alice."
    assert extract_commitments(text) == [
        "Remind me Friday at 9am about the crown.",
        "Also never email Bob after 6pm.",
    ]
    assert extract_commitments("Find the latest invoice from Alice") == []
    assert extract_commitments("Always cc my assistant on anything to the landlord") == [
        "Always cc my assistant on anything to the landlord"
    ]


async def test_commitments_survive_many_compactions(tmp_path, store):
    agent = "Landlord"
    mem = make_memory(tmp_path, store, extractor=ScriptedExtractor())  # LLM returns nothing useful
    mem.note_request(agent, "Remind me on the 1st to pay the rent. Do not mention the deposit dispute.")
    store.record_request(agent, "Remind me on the 1st to pay the rent. Do not mention the deposit dispute.")
    for _ in range(6):
        fill_log(store, agent, rounds=6)
        await mem.compact(agent, force=True)
    rendered = mem.render_history(agent)
    head = rendered.split("# Recent History")[0]
    assert "## Commitments (verbatim" in head
    assert "Remind me on the 1st to pay the rent." in head
    assert "Do not mention the deposit dispute." in head


def test_commitment_truncated_and_escaped(tmp_path, store):
    agent = "Long"
    mem = make_memory(tmp_path, store)
    sentence = "Never send <b>anything</b> to " + "x" * 600
    mem.note_request(agent, sentence)
    state = mem.load(agent)
    assert len(state.commitments) == 1 and len(state.commitments[0].text) <= 300
    store.record_request(agent, "hello")
    rendered = mem.render_history(agent)
    assert "&lt;b&gt;" in rendered and "<b>" not in rendered


def test_young_agent_sees_commitments_before_transcript(tmp_path, store):
    agent = "Young"
    mem = make_memory(tmp_path, store)
    mem.note_request(agent, "Always reply in French to Marie.")
    store.record_request(agent, "Always reply in French to Marie.")
    rendered = mem.render_history(agent)
    assert rendered.startswith("## Commitments (verbatim")
    assert store.load_transcript(agent) in rendered


# ---- review fixes: commitments race and regex precision -------------------------------


async def test_commitment_noted_during_compaction_survives(tmp_path, store):
    """compact() awaits the extractor; a request that arrives meanwhile must not be erased."""
    agent = "Race"
    fill_log(store, agent, rounds=10)
    holder = {}

    class SlowExtractor:
        async def __call__(self, entries, state):
            holder["mem"].note_request(agent, "Never email Bob about the merger.")
            return Extraction(episodes=["did things"])

    mem = make_memory(tmp_path, store, extractor=SlowExtractor())
    holder["mem"] = mem
    await mem.compact(agent, force=True)
    assert [c.text for c in mem.load(agent).commitments] == ["Never email Bob about the merger."]


def test_commitment_regex_ignores_one_off_instructions():
    from server.services.execution.memory import extract_commitments
    one_offs = [
        "Do not send it yet, only create the draft.",
        "Make sure to include the thread id in your answer.",
        "Only look at the last month.",
        "I don't think he has.",
        "Just check whether she replied this time.",
        "Find emails from Dr. Chen about the lease.",
    ]
    for text in one_offs:
        assert extract_commitments(text) == [], text
    durable = [
        ("Never cc her manager on these emails.", "Never cc her manager on these emails."),
        ("From now on, reply in French to Marie.", "From now on, reply in French to Marie."),
        ("Do not mention the deposit dispute to Hannah.", "Do not mention the deposit dispute to Hannah."),
        ("Please don't ever forward her messages.", "Please don't ever forward her messages."),
        ("Remind me every Monday at 9am to check the invoices.", "Remind me every Monday at 9am to check the invoices."),
    ]
    for text, expected in durable:
        assert extract_commitments(text) == [expected], text


# ---- review fix: real schedule_compaction is tracked and cancellable ----------------


async def test_schedule_compaction_runs_the_real_compact(tmp_path, store):
    fill_log(store, "a", rounds=12)
    mem = make_memory(tmp_path, store)
    assert mem.needs_compaction("a")
    task = mem.schedule_compaction("a")
    assert task is not None and task in mem.pending_tasks()
    await task
    assert mem.load("a").compacted_through > 0
    assert task not in mem.pending_tasks()


async def test_shutdown_cancels_pending_compaction(tmp_path, store):
    import asyncio

    class Slow:
        async def __call__(self, entries, state):
            await asyncio.sleep(5)
            return Extraction(episodes=["late"])

    fill_log(store, "a", rounds=12)
    mem = make_memory(tmp_path, store, extractor=Slow())
    task = mem.schedule_compaction("a")
    await asyncio.sleep(0)
    await mem.shutdown()
    assert task.cancelled() or task.done()
    assert mem.load("a").compacted_through == 0
    assert not mem.pending_tasks()
