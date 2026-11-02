"""Unit tests for the budgeted AgentRegistry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from server.services.execution.registry import (
    AgentRegistry,
    AgentStatus,
    RegistryConfig,
    tokenize,
)

HOUR = 3600.0
DAY = 24 * HOUR


class Clock:
    def __init__(self, start: float = 1_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> Clock:
    return Clock()


def make_registry(
    tmp_path: Path,
    clock: Clock,
    *,
    pinned: set[str] | None = None,
    **overrides,
) -> AgentRegistry:
    config = RegistryConfig(**{"max_roster": 5, "hot_cache_size": 2, "retrieval_k": 4, **overrides})
    pinned = pinned or set()
    return AgentRegistry(
        tmp_path / "registry.json",
        config=config,
        clock=clock,
        has_pending_triggers=lambda name: name in pinned,
    )


# ---- basic lifecycle ------------------------------------------------------------------


def test_register_creates_then_touches(tmp_path, clock):
    reg = make_registry(tmp_path, clock)
    assert reg.register("Email to Alice", "Draft a lunch invite to Alice for Friday") is True
    assert reg.register("Email to Alice", "Ask Alice if Thursday works instead") is False
    rec = reg.get("Email to Alice")
    assert rec.use_count == 2
    assert rec.description.startswith("Draft a lunch invite")  # first instruction sticks
    assert len(rec.recent_requests) == 2


def test_persistence_round_trip(tmp_path, clock):
    reg = make_registry(tmp_path, clock)
    reg.register("Vercel Job Offer", "Find the Vercel offer email and summarise the comp")
    reg.update_summary("Vercel Job Offer", "Offer is $180k base; awaiting equity details")

    reloaded = make_registry(tmp_path, clock)
    rec = reloaded.get("Vercel Job Offer")
    assert rec is not None
    assert rec.summary.startswith("Offer is $180k")
    assert rec.use_count == 1


def test_migrates_legacy_roster(tmp_path, clock):
    legacy = tmp_path / "roster.json"
    legacy.write_text(json.dumps(["Old Agent A", "Old Agent B"]))
    reg = AgentRegistry(
        tmp_path / "registry.json",
        config=RegistryConfig(),
        clock=clock,
        legacy_roster_path=legacy,
    )
    assert set(reg.names()) == {"Old Agent A", "Old Agent B"}
    assert reg.get("Old Agent A").last_reason == "migrated"


def test_status_transitions_with_idle_time(tmp_path, clock):
    reg = make_registry(tmp_path, clock, active_ttl_s=2 * DAY, archive_ttl_s=14 * DAY)
    reg.register("Dentist Reminder", "Remind me about the dentist on the 14th")
    rec = reg.get("Dentist Reminder")

    assert reg.status_of(rec) == AgentStatus.ACTIVE
    clock.advance(3 * DAY)
    assert reg.status_of(rec) == AgentStatus.DORMANT
    clock.advance(12 * DAY)
    assert reg.status_of(rec) == AgentStatus.ARCHIVED


def test_pending_trigger_prevents_archive(tmp_path, clock):
    reg = make_registry(
        tmp_path, clock, pinned={"Dentist Reminder"}, active_ttl_s=DAY, archive_ttl_s=2 * DAY
    )
    reg.register("Dentist Reminder", "Remind me about the dentist")
    reg.register("Throwaway", "One-off lookup")
    clock.advance(30 * DAY)

    assert reg.status_of(reg.get("Dentist Reminder")) == AgentStatus.DORMANT
    assert reg.status_of(reg.get("Throwaway")) == AgentStatus.ARCHIVED

    # ...and it is still findable, and flagged, when something relevant is asked.
    selection = reg.select("can you move my dentist reminder")
    names = selection.names()
    assert "Dentist Reminder" in names
    entry = next(e for e in selection.shown if e.record.name == "Dentist Reminder")
    assert entry.pinned is True


def test_trigger_touch_reactivates_agent(tmp_path, clock):
    reg = make_registry(tmp_path, clock, active_ttl_s=DAY, archive_ttl_s=2 * DAY)
    reg.register("Weekly Report", "Send me a weekly summary every Monday")
    clock.advance(10 * DAY)
    assert reg.status_of(reg.get("Weekly Report")) == AgentStatus.ARCHIVED
    reg.touch("Weekly Report", reason="trigger")
    assert reg.status_of(reg.get("Weekly Report")) == AgentStatus.ACTIVE
    assert reg.get("Weekly Report").last_reason == "trigger"


# ---- selection --------------------------------------------------------------------------


def seed_many(reg: AgentRegistry, clock: Clock, n: int) -> None:
    for i in range(n):
        reg.register(f"Agent {i}", f"Generic task number {i} about topic{i}")
        clock.advance(HOUR)


def test_budget_is_respected(tmp_path, clock):
    reg = make_registry(tmp_path, clock, max_roster=5)
    seed_many(reg, clock, 50)
    selection = reg.select("anything at all")
    assert len(selection.shown) <= 5
    assert selection.total == 50
    assert selection.hidden == 50 - len(selection.shown)


def test_hot_cache_included_without_keywords(tmp_path, clock):
    reg = make_registry(tmp_path, clock, hot_cache_size=2, max_roster=4)
    seed_many(reg, clock, 30)
    reg.register("Email to Alice", "Ask Alice about lunch")
    clock.advance(60)
    reg.register("Email to Bob", "Ask Bob about lunch")

    # A follow-up with no retrievable keywords still sees the two most recent agents.
    selection = reg.select("did she reply?")
    names = selection.names()
    assert "Email to Bob" in names and "Email to Alice" in names


def test_retrieval_surfaces_old_relevant_agent(tmp_path, clock):
    reg = make_registry(tmp_path, clock, hot_cache_size=2, max_roster=5)
    reg.register("Invoice from Priya", "Find Priya's invoice for the October design work")
    clock.advance(5 * DAY)
    seed_many(reg, clock, 40)  # bury it under 40 newer agents

    selection = reg.select("what did priya say about the invoice")
    assert "Invoice from Priya" in selection.names()
    entry = next(e for e in selection.shown if e.record.name == "Invoice from Priya")
    assert entry.reason == "retrieved"


def test_same_name_different_topic_disambiguated(tmp_path, clock):
    reg = make_registry(tmp_path, clock, hot_cache_size=1, max_roster=3)
    reg.register("Alice Lunch", "Invite Alice to lunch on Friday")
    clock.advance(HOUR)
    reg.register("Alice Contract", "Ask Alice to countersign the consulting contract")
    clock.advance(HOUR)
    seed_many(reg, clock, 20)

    selection = reg.select("has alice signed the contract yet")
    shown = selection.shown
    top_retrieved = [e for e in shown if e.reason == "retrieved"]
    assert top_retrieved and top_retrieved[0].record.name == "Alice Contract"


def test_archived_excluded_from_roster_but_searchable(tmp_path, clock):
    reg = make_registry(tmp_path, clock, active_ttl_s=DAY, archive_ttl_s=2 * DAY, max_roster=5)
    reg.register("Flight to Lisbon", "Find my Lisbon flight confirmation")
    clock.advance(30 * DAY)
    reg.register("Fresh", "something new")

    selection = reg.select("lisbon flight")
    assert "Flight to Lisbon" not in selection.names()
    assert selection.archived == 1

    hits = reg.search("lisbon flight", include_archived=True)
    assert hits and hits[0][0].name == "Flight to Lisbon"


def test_all_strategy_matches_upstream_rendering(tmp_path, clock):
    reg = make_registry(tmp_path, clock, strategy="all")
    reg.register("B agent", "x")
    reg.register("A agent", "y")
    rendered = reg.render(reg.select("whatever"))
    # insertion order, like upstream roster.json
    assert rendered == '<agent name="B agent" />\n<agent name="A agent" />'


def test_budgeted_rendering_has_note_and_summaries(tmp_path, clock):
    reg = make_registry(tmp_path, clock, max_roster=2)
    seed_many(reg, clock, 5)
    reg.update_summary("Agent 4", "Waiting on reply")
    rendered = reg.render(reg.select("topic4"))
    assert "<roster_note>" in rendered
    assert 'status="active"' in rendered
    assert "Waiting on reply" in rendered
    assert "search_agents" in rendered


def test_empty_registry_renders_none(tmp_path, clock):
    reg = make_registry(tmp_path, clock)
    assert reg.render(reg.select("hi")) == "None"


def test_tokenize_drops_stopwords_and_punctuation():
    assert tokenize("Did Alice reply about the invoice?") == ["alice", "invoice"]


# ---- consolidation ---------------------------------------------------------------------


def test_find_duplicate_merges_renamed_same_thread(tmp_path, clock):
    reg = make_registry(tmp_path, clock)
    reg.register("Email to Alice about lunch", "Invite Alice to lunch Friday")
    assert reg.find_duplicate("Alice Lunch").name == "Email to Alice about lunch"
    assert reg.find_duplicate("Lunch with Alice").name == "Email to Alice about lunch"


def test_find_duplicate_respects_topic_difference(tmp_path, clock):
    reg = make_registry(tmp_path, clock)
    reg.register("Email to Alice about lunch", "Invite Alice to lunch")
    assert reg.find_duplicate("Alice Contract") is None
    assert reg.find_duplicate("Alice") is None  # single token: too little signal


def test_find_duplicate_ignores_archived_and_can_be_disabled(tmp_path, clock):
    reg = make_registry(tmp_path, clock, active_ttl_s=DAY, archive_ttl_s=2 * DAY)
    reg.register("Email to Alice about lunch", "x")
    clock.advance(30 * DAY)
    assert reg.find_duplicate("Alice Lunch") is None
    reg2 = make_registry(tmp_path, clock, consolidate=False)
    reg2.register("Email to Bob about rent", "x")
    assert reg2.find_duplicate("Bob Rent") is None


def test_roster_is_ordered_most_recent_first(tmp_path, clock):
    reg = make_registry(tmp_path, clock, max_roster=5, hot_cache_size=1)
    reg.register("Tara October Invoice", "chase Tara's invoice")
    clock.advance(HOUR)
    reg.register("Tara Conference Talk", "ask Tara about the talk")
    clock.advance(HOUR)
    reg.register("Tara Flight", "Tara's flight details")
    names = reg.select("nudge tara again").names()
    assert names[:3] == ["Tara Flight", "Tara Conference Talk", "Tara October Invoice"]


# ---- round two: consolidation needs evidence from the instructions ----------------------


@pytest.mark.parametrize("existing,existing_instr,new,instr,expect", [
    ("Alice lunch invoice reconciliation", "reconcile the lunch invoice with Alice", "Alice lunch", "schedule lunch with Alice", False),
    ("Alice Smith Lunch", "lunch with Alice Smith on Friday", "Alice Lunch", "move the lunch with Alice Smith to Thursday", True),
    ("Alice Lunch Invite", "schedule lunch with Alice next week", "Alice Lunch Reschedule", "move the lunch with Alice to Thursday", True),
    ("Email to Bob about dinner", "book dinner with Bob on Saturday", "Bob Dinner Cancellation", "cancel the dinner with Bob on Saturday", True),
    ("Tara October Invoice", "chase Tara for the October invoice", "Tara November Invoice", "chase Tara for the November invoice", False),
    ("Dinner Plans with Bob", "book dinner with Bob", "Dinner Plans with Alice", "book dinner with Alice", False),
    ("Flight Confirmation with Liam", "find Liam's flight confirmation", "Liam Flight Cancellation", "cancel Liam's flight booking", True),
])
def test_consolidation_table(tmp_path, clock, existing, existing_instr, new, instr, expect):
    reg = make_registry(tmp_path, clock)
    reg.register(existing, existing_instr)
    assert (reg.find_duplicate(new, instr) is not None) is expect


# ---- round two: pinned agents are always shown; migration keeps real recency -----------


def test_pinned_agents_are_always_in_roster(tmp_path, clock):
    reg = make_registry(tmp_path, clock, pinned={"Reminder: dentist (Alice)"}, max_roster=6, hot_cache_size=2)
    reg.register("Reminder: dentist (Alice)", "remind me about the dentist")
    clock.advance(10 * DAY)
    for i in range(30):
        clock.advance(HOUR)
        reg.register(f"Agent {i}", f"topic {i}")
    sel = reg.select("")
    entry = next((e for e in sel.shown if e.record.name == "Reminder: dentist (Alice)"), None)
    assert entry is not None and entry.reason == "pinned" and entry.pinned is True


def test_pinned_slots_reserved_most_recent_first(tmp_path, clock):
    """Five old reminder agents, three brand-new agents, budget 4: min(3, 4//2)=2 most recent
    pinned + the hot/recent new ones. Pinned agents never crowd out all current work."""
    names = [f"Reminder {i}" for i in range(5)]
    reg = make_registry(tmp_path, clock, pinned=set(names), max_roster=4, hot_cache_size=1)
    for n in names:
        clock.advance(HOUR)
        reg.register(n, "remind me")
    clock.advance(5 * DAY)
    for i in range(3):
        clock.advance(HOUR)
        reg.register(f"New {i}", f"fresh task {i}")
    sel = reg.select("")
    assert len(sel.shown) == 4
    assert set(sel.names()) == {"Reminder 4", "Reminder 3", "New 2", "New 1"}


def test_migration_uses_log_mtime(tmp_path, clock):
    import os
    from server.services.execution.log_store import _slugify

    legacy = tmp_path / "roster.json"
    legacy.write_text(json.dumps(["Old Agent", "Fresh Agent"]))
    logs = tmp_path / "logs"
    logs.mkdir()
    log = logs / f"{_slugify('Old Agent')}.log"
    log.write_text("<agent_request>x</agent_request>\n")
    five_days_ago = clock() - 5 * DAY
    os.utime(log, (five_days_ago, five_days_ago))
    reg = AgentRegistry(tmp_path / "registry.json", config=RegistryConfig(), clock=clock,
                        legacy_roster_path=legacy, log_dir=logs)
    assert reg.get("Old Agent").last_used_at == pytest.approx(five_days_ago, abs=1)
    assert reg.get("Fresh Agent").last_used_at == pytest.approx(clock(), abs=1)


# ---- round two: agents named in the recent transcript are guaranteed and boosted --------


def test_recent_agent_names_guarantee_inclusion(tmp_path, clock):
    reg = make_registry(tmp_path, clock, max_roster=6, hot_cache_size=2)
    reg.register("Dentist Appointment with Alice", "reschedule the dentist with Alice")
    clock.advance(10 * DAY)
    for i in range(60):
        clock.advance(HOUR)
        reg.register(f"Agent {i}", f"topic {i}")
    sel = reg.select("did she reply?", recent_agent_names=["Dentist Appointment with Alice"])
    entry = next((e for e in sel.shown if e.record.name == "Dentist Appointment with Alice"), None)
    assert entry is not None and entry.reason == "transcript"
    assert len(sel.shown) <= 6


def test_transcript_names_outrank_lexical_ties(tmp_path, clock):
    reg = make_registry(tmp_path, clock, max_roster=6, hot_cache_size=1, retrieval_k=4)
    reg.register("Tara October Invoice", "chase Tara for the October invoice")
    clock.advance(DAY)
    reg.register("Tara Conference Talk", "ask Tara about the conference talk")
    clock.advance(DAY)
    hits = reg.search("nudge tara again", recent_agent_names=["Tara October Invoice"])
    assert hits[0][0].name == "Tara October Invoice"


# ---- round two: optional trigram fallback for misspelt follow-ups ---------------------


def test_fuzzy_fallback_matches_misspelling(tmp_path, clock):
    reg = make_registry(tmp_path, clock, fuzzy=True)
    reg.register("Email to Priya about invoice", "chase Priya for the invoice")
    reg.register("Dinner Plans with Bob", "book dinner with Bob")
    hits = reg.search("any word from Priyaa?")
    assert hits and hits[0][0].name == "Email to Priya about invoice"
    strict = make_registry(tmp_path / "strict", clock, fuzzy=False)
    strict.register("Email to Priya about invoice", "chase Priya for the invoice")
    assert strict.search("any word from Priyaa?") == []


# ---- review fixes: full-name entities; transcript tier cannot crowd out retrieval ------


@pytest.mark.parametrize("existing,existing_instr,new,instr,expect", [
    ("Alice Wong lunch", "Email Alice Wong about lunch on Thursday.", "Alice Chen lunch", "Email Alice Chen about lunch on Friday.", False),
    ("Alice Wong lunch", "Email Alice Wong about lunch on Thursday.", "Alice lunch", "Move the lunch with Alice Wong to Friday.", True),
    ("Email to Bob about dinner", "Ask Bob if Saturday works. Check his reply.", "Bob Dinner Cancellation", "Draft a note to Bob cancelling dinner. Ask Bob to confirm.", True),
    # mirror of the comment's forbidden case: a more specific new topic must not land on the generic agent
    ("Alice lunch", "Invite Alice to lunch Friday", "Alice lunch invoice reconciliation", "Reconcile the lunch invoice with Alice", False),
])
def test_consolidation_full_names_and_capitalised_instructions(tmp_path, clock, existing, existing_instr, new, instr, expect):
    reg = make_registry(tmp_path, clock)
    reg.register(existing, existing_instr)
    assert (reg.find_duplicate(new, instr) is not None) is expect


def test_transcript_tier_does_not_hide_explicit_query_target(tmp_path, clock):
    """3 pinned + 6 transcript-named + hot agents must not push out the agent the user names."""
    pinned = {f"Reminder {i}" for i in range(3)}
    reg = make_registry(tmp_path, clock, pinned=pinned, max_roster=12, hot_cache_size=4, retrieval_k=8)
    reg.register("Acme invoice dispute", "chase Acme about the disputed invoice")
    clock.advance(5 * DAY)
    for n in sorted(pinned):
        clock.advance(HOUR)
        reg.register(n, "remind me")
    recent = [f"Recent {i}" for i in range(10)]
    for n in recent:
        clock.advance(HOUR)
        reg.register(n, f"generic task {n}")
    # transcript names older than the hot four, so pinned(3) + transcript(6) + hot(4) > 12
    sel = reg.select("what happened with the acme invoice dispute?", recent_agent_names=recent[:6])
    assert "Acme invoice dispute" in sel.names(), sel.names()
    assert len(sel.shown) <= 12


# ---- round three: the thread the user touched last is marked -------------------------------


def test_render_marks_most_recent_agent_last_touched(tmp_path, clock):
    reg = make_registry(tmp_path, clock, max_roster=5)
    reg.register("Tara October Invoice", "chase Tara's invoice")
    clock.advance(HOUR)
    reg.register("Bob Flight", "find Bob's flight")
    rendered = reg.render(reg.select("any news?"))
    lines = [line for line in rendered.splitlines() if line.startswith("<agent ")]
    assert rendered.count('last_touched="true"') == 1
    assert 'name="Bob Flight"' in lines[0] and 'last_touched="true"' in lines[0]

    upstream = AgentRegistry(tmp_path / "all.json", config=RegistryConfig(strategy="all"), clock=clock)
    upstream.register("Bob Flight", "find Bob's flight")
    assert "last_touched" not in upstream.render(upstream.select("any news?"))


# ---- round three: did-you-mean guard when the chosen agent is about someone else ---------


@pytest.mark.parametrize("message,chosen,expect", [
    ("what did bob say about the flight confirmation?", "Follow up with Tomasz - flight", ["Flight Confirmation with Bob"]),
    ("ask Tomasz whether Bob is coming", "Follow up with Tomasz - flight", None),
    ("any update on the flight?", "Follow up with Tomasz - flight", None),
    ("did bob reply?", "Flight Confirmation with Bob", None),
    ("did bob reply?", "No Such Agent", None),
])
def test_routing_mismatch(tmp_path, clock, message, chosen, expect):
    reg = make_registry(tmp_path, clock)
    reg.register("Follow up with Tomasz - flight", "Check whether Tomasz has replied about the flight confirmation")
    clock.advance(HOUR)
    reg.register("Flight Confirmation with Bob", "Find the latest email from Bob about the flight confirmation")
    clock.advance(HOUR)
    reg.register("Dinner Plans with Alice", "book dinner with Alice")
    mismatch = reg.routing_mismatch(chosen, message)
    if expect is None:
        assert mismatch is None
    else:
        assert mismatch is not None
        assert mismatch.candidates[: len(expect)] == expect
        assert "bob" in mismatch.tokens


# ---- review fix pass: the guard sees people, not weekdays, products or topic anchors --------


@pytest.mark.parametrize("message,chosen,fires", [
    ("Tell them I'm free Monday", "Acme recruiter", False),
    ("Send the Zoom link for the invoice review", "Priya October Invoice", False),
    ("Draft a reply to the recruiter and cc Priya", "Acme recruiter", False),
    ("what did bob say about the flight confirmation?", "Follow up with Tomasz - flight", True),
])
def test_routing_guard_precision(tmp_path, clock, message, chosen, fires):
    reg = make_registry(tmp_path, clock)
    reg.register("Acme recruiter", "Reply to the recruiter at Acme about the senior role")
    reg.register("Zoom call with Bob", "Set up a Zoom call with Bob on Monday")
    reg.register("Priya October Invoice", "Chase Priya for the October invoice")
    reg.register("Follow up with Tomasz - flight", "Check whether Tomasz has replied about the flight confirmation")
    reg.register("Flight Confirmation with Bob", "Find the latest email from Bob about the flight confirmation")
    assert (reg.routing_mismatch(chosen, message) is not None) is fires


# ---- hardening: a background trigger fire is not "the thread the user touched last" -------


def test_last_touched_skips_agents_whose_last_use_was_a_trigger(tmp_path, clock):
    reg = make_registry(tmp_path, clock, max_roster=5)
    reg.register("Tara October Invoice", "chase Tara's invoice")
    clock.advance(HOUR)
    reg.register("Bob Flight", "find Bob's flight")
    clock.advance(HOUR)
    reg.touch("Tara October Invoice", reason="trigger")  # a reminder fired in the background
    rendered = reg.render(reg.select("ok send it"))
    lines = [line for line in rendered.splitlines() if line.startswith("<agent ")]
    assert rendered.count('last_touched="true"') == 1
    assert 'name="Tara October Invoice"' in lines[0] and "last_touched" not in lines[0]
    assert 'name="Bob Flight"' in lines[1] and 'last_touched="true"' in lines[1]


def test_last_touched_falls_back_to_first_entry_when_every_use_was_a_trigger(tmp_path, clock):
    reg = make_registry(tmp_path, clock, max_roster=5)
    reg.register("Tara October Invoice", "chase Tara's invoice")
    clock.advance(HOUR)
    reg.touch("Tara October Invoice", reason="trigger")
    rendered = reg.render(reg.select("ok send it"))
    assert rendered.count('last_touched="true"') == 1


# ---- held-out findings: the guard anchors on the user's words, and covers new agents -------


def _heldout_registry(tmp_path, clock):
    reg = make_registry(tmp_path, clock, max_roster=12)
    reg.register("Gym membership freeze", "Email Ironwood Fitness to freeze my membership for June and July")
    clock.advance(HOUR)
    reg.register("Bob flight confirmation", "Find the latest email from Bob about the flight confirmation")
    clock.advance(HOUR)
    reg.register("Grandma's birthday flowers", "Order flowers from Petal & Stem for Grandma's birthday on the 9th")
    clock.advance(HOUR)
    reg.register("Inbox check", "Check my inbox for anything urgent from the last 24 hours")
    return reg


def test_guard_anchors_on_the_users_message_not_the_instructions(tmp_path, clock):
    """The model's instructions always echo the chosen agent's name ("check the inbox for ...
    from Bob"), which used to count as an anchor and silence the guard. The anchor must come
    from what the user said."""
    reg = _heldout_registry(tmp_path, clock)
    instructions = "Check my inbox for any recent email from Bob and summarise it"
    assert reg.routing_mismatch("Inbox check", instructions) is None  # instructions alone: anchored, as before
    mismatch = reg.routing_mismatch("Inbox check", instructions, anchor_text="anything from Bob?")
    assert mismatch is not None and mismatch.candidates[0] == "Bob flight confirmation"


def test_guard_still_respects_an_anchor_in_the_users_message(tmp_path, clock):
    reg = _heldout_registry(tmp_path, clock)
    reg.register("Acme recruiter", "Reply to the recruiter at Acme about the senior role")
    assert reg.routing_mismatch("Acme recruiter", "Draft a reply to the recruiter and cc Bob",
                                anchor_text="reply to the recruiter and cc Bob") is None


@pytest.mark.parametrize("message, new_name, expect", [
    ("did Ironwood get back to me?", "Ironwood email check", "Gym membership freeze"),
    ("has Petal & Stem shipped?", "Petal & Stem shipment check", "Grandma's birthday flowers"),
    ("any word from Bob?", "Bob follow-up", "Bob flight confirmation"),
    ("email Priya about the gym schedule", "Priya gym schedule", None),  # new task, nobody has it
    ("book a table at Osteria Nova for Friday", "Osteria Nova booking", None),
    ("email Ironwood Fitness to cancel my membership outright", "Ironwood cancellation", None),  # new task, not a follow-up
])
def test_new_agent_guard_catches_follow_ups_about_an_entity_someone_already_owns(tmp_path, clock, message, new_name, expect):
    reg = _heldout_registry(tmp_path, clock)
    mismatch = reg.new_agent_mismatch(new_name, message, anchor_text=message)
    if expect is None:
        assert mismatch is None
    else:
        assert mismatch is not None and mismatch.candidates[0] == expect


def test_fuzzy_matches_a_misspelt_name_even_when_other_words_match(tmp_path, clock):
    """"anything from Sharanjit?" has an exact token ("anything") somewhere in the index, so the
    old all-or-nothing fallback never ran; the misspelt name must be matched on its own."""
    reg = make_registry(tmp_path, clock, max_roster=12, fuzzy=True)
    reg.register("Sharanjeet book club", "Email Sharanjeet about moving book club to Thursdays")
    clock.advance(HOUR)
    reg.register("Inbox check", "Check my inbox for anything urgent from the last 24 hours")
    hits = [r.name for r, _ in reg.search("anything from Sharanjit?", limit=3)]
    assert hits and hits[0] == "Sharanjeet book club"
    hits = [r.name for r, _ in reg.search("what did Katherin say?", limit=3)]
    assert "Sharanjeet book club" not in hits[:1]


def test_fuzzy_handles_transposed_letters_and_matches_people_only(tmp_path, clock):
    reg = make_registry(tmp_path, clock, max_roster=12, fuzzy=True)
    reg.register("Priya Nair birthday dinner", "Email Priya Nair to pick a restaurant for her birthday dinner")
    clock.advance(HOUR)
    reg.register("Reminder: dentist (Mei)", "Remind me Friday at 8am to call Mei's dentist about the retainer")
    clock.advance(HOUR)
    reg.register("Dinner Plans with Ingrid", "Email Ingrid about dinner on Saturday")
    assert [r.name for r, _ in reg.search("did Pryia reply about dinner?", limit=2)][0] == "Priya Nair birthday dinner"
    assert [r.name for r, _ in reg.search("did Ingird confirm Saturday?", limit=2)][0] == "Dinner Plans with Ingrid"
    # A misspelt ordinary word ("retianer") is not a misspelt person: the fuzzy pass ignores it.
    assert all(r.name != "Reminder: dentist (Mei)" for r, _ in reg.search("any update on the retianer", limit=3))


def test_fan_out_does_not_move_last_touched(tmp_path, clock):
    """"What reminders do I have?" asks three agents at once; none of them is the thread the
    user touched. The marker stays on the last single-agent dispatch."""
    reg = make_registry(tmp_path, clock, max_roster=5)
    reg.register("Call Mom Reminder", "remind me to call mom")
    clock.advance(HOUR)
    reg.register("Dentist Reminder", "move the dentist one to 9am")
    clock.advance(HOUR)
    reg.register("Call Mom Reminder", "list your reminders")
    reg.register("Dentist Reminder", "list your reminders")
    reg.register("Alice Invoice Reminder", "list your reminders")
    reg.mark_fan_out(["Call Mom Reminder", "Dentist Reminder", "Alice Invoice Reminder"])
    rendered = reg.render(reg.select("cancel it"))
    touched = [line for line in rendered.splitlines() if 'last_touched="true"' in line]
    assert len(touched) == 1 and 'name="Dentist Reminder"' in touched[0], rendered
    # The next single dispatch takes the marker back.
    clock.advance(HOUR)
    reg.register("Alice Invoice Reminder", "nudge alice")
    rendered = reg.render(reg.select("cancel it"))
    touched = [line for line in rendered.splitlines() if 'last_touched="true"' in line]
    assert 'name="Alice Invoice Reminder"' in touched[0]


def test_touch_by_the_user_moves_last_touched(tmp_path, clock):
    """The evals replay a workload with register() then touch(reason="user") at each agent's
    real last-use time; the marker must follow the touch, not the registration order."""
    reg = make_registry(tmp_path, clock, max_roster=5)
    reg.register("Tara October Invoice", "chase Tara's invoice")
    clock.advance(HOUR)
    reg.register("Bob Flight", "find Bob's flight")
    clock.advance(HOUR)
    reg.touch("Tara October Invoice", reason="user")
    rendered = reg.render(reg.select("any news?"))
    touched = [line for line in rendered.splitlines() if 'last_touched="true"' in line]
    assert len(touched) == 1 and 'name="Tara October Invoice"' in touched[0], rendered
    clock.advance(HOUR)
    reg.touch("Bob Flight", reason="trigger")  # a reminder firing does not move it
    rendered = reg.render(reg.select("any news?"))
    touched = [line for line in rendered.splitlines() if 'last_touched="true"' in line]
    assert 'name="Tara October Invoice"' in touched[0], rendered
