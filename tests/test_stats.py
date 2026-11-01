"""Paired statistics for the two-arm routing eval: both arms answer the same queries, so the
right test is on the pairs, not on two independent proportions."""

from __future__ import annotations

import pytest

from evals.stats import mcnemar_exact, paired_bootstrap_diff, pair_rows


def _rows(oks, seed=7):
    return [{"seed": seed, "text": f"q{i}", "target": f"a{i}", "ok": ok} for i, ok in enumerate(oks)]


def test_pair_rows_matches_on_seed_text_and_target():
    a = _rows([True, False, True])
    b = _rows([True, True, False])
    pairs = pair_rows(a, b)
    assert pairs == [(True, True), (False, True), (True, False)]


def test_pair_rows_ignores_order_and_unmatched():
    a = _rows([True, False])
    b = list(reversed(_rows([True, True]))) + [{"seed": 9, "text": "extra", "target": "x", "ok": True}]
    assert pair_rows(a, b) == [(True, True), (False, True)]


def test_mcnemar_exact_counts_discordant_pairs_only():
    # 10 pairs where B is right and A wrong, 0 the other way, 20 agreements.
    pairs = [(False, True)] * 10 + [(True, True)] * 15 + [(False, False)] * 5
    out = mcnemar_exact(pairs)
    assert out["b_only"] == 10 and out["a_only"] == 0 and out["discordant"] == 10
    assert out["p_value"] == pytest.approx(2 * 0.5 ** 10, rel=1e-9)


def test_mcnemar_exact_no_discordant_pairs_is_p_one():
    assert mcnemar_exact([(True, True)] * 5)["p_value"] == 1.0


def test_paired_bootstrap_diff_interval_brackets_the_point_estimate():
    pairs = [(False, True)] * 30 + [(True, True)] * 60 + [(False, False)] * 20 + [(True, False)] * 10
    out = paired_bootstrap_diff(pairs, n_boot=2000, seed=1)
    assert out["diff"] == pytest.approx(20 / 120)
    lo, hi = out["ci95"]
    assert lo < out["diff"] < hi and lo > 0.0


def test_paired_bootstrap_diff_is_deterministic_for_a_seed():
    pairs = [(False, True)] * 8 + [(True, True)] * 8 + [(True, False)] * 2
    assert paired_bootstrap_diff(pairs, n_boot=500, seed=3) == paired_bootstrap_diff(pairs, n_boot=500, seed=3)
