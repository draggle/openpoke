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
    assert rendered == '<agent name="A agent" />\n<agent name="B agent" />'


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
