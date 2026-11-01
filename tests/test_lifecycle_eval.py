"""Sixty simulated days of use: the budgeted roster's prompt cost stays flat, archiving keeps
the live set bounded, and an agent that owns a reminder is never archived."""

from __future__ import annotations

from evals.lifecycle_eval import simulate


def test_budgeted_roster_cost_is_flat_while_upstream_grows(tmp_path):
    out = simulate(days=40, tmp_dir=tmp_path, seed=3)
    upstream, fork = out["all"], out["budgeted"]
    assert upstream["tokens_day_last"] > 2 * upstream["tokens_day_5"]
    assert fork["tokens_day_last"] <= 1.3 * fork["tokens_day_5"]
    assert fork["tokens_per_day_slope"] < 0.1 * upstream["tokens_per_day_slope"]


def test_archiving_bounds_the_live_set_and_spares_reminder_owners(tmp_path):
    out = simulate(days=40, tmp_dir=tmp_path, seed=3)
    fork = out["budgeted"]
    assert fork["total_agents_last"] > fork["live_agents_last"] > 0
    assert fork["archived_last"] > 0
    assert fork["reminder_owners_archived"] == 0
    assert fork["live_agents_last"] <= fork["live_agents_peak"]
