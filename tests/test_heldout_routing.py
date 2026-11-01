"""The hand-written held-out routing set loads into the same Workload the live eval consumes."""

from __future__ import annotations

from pathlib import Path

from evals.heldout_routing import FIXTURE, load_fixture


def test_fixture_loads_and_is_internally_consistent():
    w = load_fixture(FIXTURE)
    names = [a.name for a in w.agents]
    assert len(names) == len(set(names)), "duplicate agent names"
    assert len(w.agents) >= 40 and len(w.queries) >= 60
    missing = [q.target for q in w.queries if q.target not in set(names)]
    assert not missing, missing
    most_recent = max(w.agents, key=lambda a: a.last_used_at)
    for q in w.queries:
        if q.kind == "recency":
            assert q.target == most_recent.name
        assert q.target_age_days >= 0
    kinds = {q.kind for q in w.queries}
    assert {"keyword", "name_only", "topic_only", "recency", "paraphrase", "nickname", "typo", "name_opaque", "lookalike"} <= kinds


def test_fixture_states_its_expectations_before_the_run():
    import json

    data = json.loads(Path(FIXTURE).read_text())
    exp = data["expectations"]
    assert exp["budgeted_accuracy_at_least"] >= 0.8 and exp["upstream_accuracy_at_most"] <= 0.75
    assert "written_before_any_run" in exp and exp["written_before_any_run"] is True
