"""Unit tests for per-agent tiered memory and compaction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from server.services.execution.log_store import ExecutionAgentLogStore
from server.services.execution.memory import (
    AgentMemory,
    Extraction,
    Fact,
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


# ---- round three: corrections and key replacement in the extractor contract --------------


async def test_heuristic_extracts_corrections(tmp_path, store):
    store.record_request("a", "Priya says the invoice was wrong: it is $700, not $500. Update our records.")
    store.record_agent_response("a", "Updated: the deposit is $900 (was $850).")
    ext = await HeuristicExtractor()(list(store.iter_entries("a")), MemoryState())
    assert ("$500", "$700") in ext.corrections
    assert ("$850", "$900") in ext.corrections


def test_merge_correction_invalidates_matching_fact(tmp_path, store):
    mem = make_memory(tmp_path, store)
    state = MemoryState(facts=[Fact(key="Priya invoice amount", value="$500", recorded_at="t0"),
                               Fact(key="Bob deposit", value="$500", recorded_at="t0")])
    extraction = Extraction(facts=[("Priya invoice", "$700")], corrections=[("$500", "$700")])
    mem._merge(state, extraction, "t1")
    by_key = {f.key: f for f in state.facts}
    assert by_key["Priya invoice amount"].active is False
    assert by_key["Priya invoice amount"].superseded_by == "$700"
    assert by_key["Bob deposit"].active is True  # shares no key token with the correcting fact
    assert by_key["Priya invoice"].active and by_key["Priya invoice"].value == "$700"


def test_merge_replaces_invalidates_named_key(tmp_path, store):
    mem = make_memory(tmp_path, store)
    state = MemoryState(facts=[Fact(key="October invoice", value="$500", recorded_at="t0")])
    extraction = Extraction(facts=[("Priya October invoice amount", "$700")],
                            replaces={"Priya October invoice amount": "October invoice"})
    mem._merge(state, extraction, "t1")
    old = next(f for f in state.facts if f.key == "October invoice")
    assert old.active is False and old.superseded_by == "$700"
    assert len(state.active_facts()) == 1


async def test_llm_extractor_parses_replaces_and_corrections(tmp_path, store, monkeypatch):
    from server.openrouter_client import client as oc

    fill_log(store, "a", rounds=1)

    async def fake(*, model, messages, system=None, api_key=None, tools=None, base_url=None):
        assert "replaces" in (system if isinstance(system, str) else system[0]["text"])
        return {"choices": [{"message": {"content": json.dumps({
            "facts": [{"key": "Priya invoice amount", "value": "$700", "replaces": "invoice"}],
            "corrections": [{"old": "$500", "new": "$700"}],
            "threads": [], "resolved_threads": [], "episodes": []})}}]}

    monkeypatch.setattr(oc, "request_chat_completion", fake)
    ext = await LLMExtractor()(list(store.iter_entries("a")), MemoryState())
    assert ext.facts == [("Priya invoice amount", "$700")]
    assert ext.replaces == {"Priya invoice amount": "invoice"}
    assert ext.corrections == [("$500", "$700")]


# ---- round three, after the live run: corrections must not depend on the LLM noticing -----


def test_merge_correction_matches_old_value_in_key(tmp_path, store):
    """The live extractor wrote 'Hannah refund $3579: confirmed by Nov 4'; the amount sits in
    the key, and no fact carried the new value. The correction alone must retire it."""
    mem = make_memory(tmp_path, store)
    state = MemoryState(facts=[Fact(key="Hannah refund $3579", value="confirmed payment by Nov 4", recorded_at="t0"),
                               Fact(key="Bob deposit", value="$3579", recorded_at="t0")])
    ext = Extraction(corrections=[("$3579", "$3911")],
                     correction_context={("$3579", "$3911"): "Hannah says the refund was wrong: it is $3911, not $3579."})
    mem._merge(state, ext, "t1")
    by_key = {f.key: f for f in state.facts}
    assert by_key["Hannah refund $3579"].active is False and by_key["Hannah refund $3579"].superseded_by == "$3911"
    # anchored on the sentence's own words (Hannah, refund): Bob's deposit is untouched
    assert by_key["Bob deposit"].active is True


async def test_llm_extractor_unions_regex_corrections(tmp_path, store, monkeypatch):
    from server.openrouter_client import client as oc

    store.record_request("a", "Hannah says the refund was wrong: it is $3911, not $3579. Update our records.")

    async def fake(*, model, messages, system=None, api_key=None, tools=None, base_url=None):
        return {"choices": [{"message": {"content": json.dumps({"facts": [], "corrections": [], "threads": [],
                                                                  "resolved_threads": [], "episodes": ["Updated the refund"]})}}]}

    monkeypatch.setattr(oc, "request_chat_completion", fake)
    ext = await LLMExtractor()(list(store.iter_entries("a")), MemoryState())
    assert ext.corrections == [("$3579", "$3911")]


async def test_correction_in_the_verbatim_tail_retires_the_fact(tmp_path, store):
    """The fact block must not assert $500 while the raw tail, rendered right below it, says
    'it is $700, not $500'. Applied at compaction and again at render time."""
    fill_log(store, "a", rounds=6)
    store.record_request("a", "Ask Alice whether the invoice for $500 can be settled by Nov 8")
    store.record_agent_response("a", "Alice confirmed the invoice for $500 by Nov 8")
    mem = make_memory(tmp_path, store, tail_entries=2)
    state = await mem.compact("a", force=True)
    state.facts.append(Fact(key="Alice invoice $500", value="confirmed by Nov 8", recorded_at="t0"))
    mem.save("a", state)
    # the correction lands after compaction, inside the verbatim tail
    store.record_request("a", "Alice says the invoice was wrong: it is $700, not $500. Update our records.")
    store.record_agent_response("a", "Updated: the invoice is $700 (was $500).")
    rendered = mem.render_history("a")
    facts_block = rendered.split("## Facts", 1)[1].split("##", 1)[0]
    assert "$500" not in facts_block
    assert "Alice invoice $500" not in facts_block
    # and the next compaction persists it
    state = await mem.compact("a", force=True)
    old = next(f for f in state.facts if f.key == "Alice invoice $500")
    assert old.active is False and old.superseded_by == "$700"


# ---- review fix pass: corrections come from the user's words, never from email bodies -------


def test_tail_corrections_ignore_tool_responses_and_unrelated_facts(tmp_path, store):
    fill_log(store, "a", rounds=6)
    mem = make_memory(tmp_path, store, tail_entries=2)
    state = MemoryState(facts=[Fact(key="Priya invoice amount", value="$500", recorded_at="t0"),
                               Fact(key="Bob deposit", value="$500", recorded_at="t0"),
                               Fact(key="Gym membership", value="$99/month", recorded_at="t0")],
                        compacted_through=24)
    mem.save("a", state)
    store.record_request("a", "Priya says the invoice was wrong: it is $700, not $500. Update our records.")
    store.record_tool_response("a", "gmail_fetch_emails", "Newsletter: Premium plan now $49 (was $99). Unsubscribe.")
    rendered = mem.render_history("a")
    facts_block = rendered.split("## Facts", 1)[1].split("## ", 1)[0]
    assert "Priya invoice amount" not in facts_block
    assert "Bob deposit: $500" in facts_block  # shares nothing with "Priya ... invoice"
    assert "Gym membership: $99/month" in facts_block  # the "correction" was in an email body


async def test_heuristic_corrections_only_from_requests_and_responses(tmp_path, store):
    store.record_tool_response("a", "gmail_fetch_emails", "Premium plan now $49 (was $99).")
    store.record_agent_response("a", "Updated: the deposit is $900 (was $850).")
    ext = await HeuristicExtractor()(list(store.iter_entries("a")), MemoryState())
    assert ext.corrections == [("$850", "$900")]


def test_correction_without_any_anchor_retires_nothing(tmp_path, store):
    mem = make_memory(tmp_path, store)
    state = MemoryState(facts=[Fact(key="Bob deposit", value="$500", recorded_at="t0")])
    mem._merge(state, Extraction(corrections=[("$500", "$700")]), "t1")  # no sentence, no carrier
    assert state.facts[0].active


def test_correction_new_value_is_matched_as_a_whole_amount(tmp_path, store):
    mem = make_memory(tmp_path, store)
    state = MemoryState(facts=[Fact(key="Priya invoice amount", value="$500", recorded_at="t0")])
    ext = Extraction(corrections=[("$500", "$50")], correction_context={("$500", "$50"): "Priya's invoice is $50, not $500."})
    mem._merge(state, ext, "t1")
    assert state.facts[0].active is False and state.facts[0].superseded_by == "$50"


# ---- hardening: amounts compare by value, not by spelling ---------------------------------


def test_contains_amount_matches_equal_values_spelled_differently():
    from server.services.execution.memory import _contains_amount

    assert _contains_amount("deposit is $500.00 as agreed", "$500")
    assert _contains_amount("invoice total $1,200", "$1200")
    assert _contains_amount("fee $500", "$500.00")
    assert not _contains_amount("deposit is $5000", "$500")
    assert not _contains_amount("deposit is $500.50", "$500")


def test_correction_retires_fact_whose_amount_is_spelled_with_cents(tmp_path, store):
    mem = make_memory(tmp_path, store)
    state = MemoryState(facts=[Fact(key="Priya invoice amount", value="$500.00", recorded_at="t0")])
    ext = Extraction(corrections=[("$500", "$700")], correction_context={("$500", "$700"): "Priya's invoice is $700, not $500."})
    mem._merge(state, ext, "t1")
    assert state.facts[0].active is False and state.facts[0].superseded_by == "$700"
