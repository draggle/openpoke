"""Statistics helpers for the evaluators (no scipy): Wilson intervals, and paired tests for
two-arm evals that answer the same queries.

The routing eval runs every query through the upstream arm and the fork's arm. Two Wilson
intervals that do not overlap are already strong, but the honest test is on the *pairs*: how
often did one arm get a query right that the other got wrong? That is McNemar's exact test on
the discordant pairs, and a paired bootstrap for the interval on the accuracy difference.

    python -m evals.stats docs/live/final/live_routing_results_history.json
"""

from __future__ import annotations

import json
import random
import sys
from math import comb
from pathlib import Path
from math import sqrt
from typing import Dict, Iterable, List, Sequence, Tuple

Pair = Tuple[bool, bool]  # (arm A correct, arm B correct) for one query


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score 95% interval for k successes in n trials."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def pair_rows(rows_a: Iterable[Dict], rows_b: Iterable[Dict]) -> List[Pair]:
    """Match rows by (seed, text, target) and return (ok_a, ok_b) per matched query, in A's order."""
    key = lambda r: (r.get("seed"), r["text"], r["target"])  # noqa: E731
    by_b = {key(r): bool(r["ok"]) for r in rows_b}
    return [(bool(r["ok"]), by_b[key(r)]) for r in rows_a if key(r) in by_b]


def mcnemar_exact(pairs: Sequence[Pair]) -> Dict[str, float]:
    """Two-sided exact McNemar test on the discordant pairs (binomial, p = 0.5)."""
    a_only = sum(1 for a, b in pairs if a and not b)
    b_only = sum(1 for a, b in pairs if b and not a)
    n = a_only + b_only
    if n == 0:
        p = 1.0
    else:
        k = min(a_only, b_only)
        tail = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
        p = min(1.0, 2 * tail)
    return {"a_only": a_only, "b_only": b_only, "discordant": n, "p_value": p}


def paired_bootstrap_diff(pairs: Sequence[Pair], *, n_boot: int = 5000, seed: int = 0) -> Dict:
    """Accuracy(B) − accuracy(A) with a 95% percentile interval from resampling the pairs."""
    n = len(pairs)
    if n == 0:
        return {"diff": 0.0, "ci95": (0.0, 0.0), "n": 0}
    diff = (sum(b for _, b in pairs) - sum(a for a, _ in pairs)) / n
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        diffs.append((sum(b for _, b in sample) - sum(a for a, _ in sample)) / n)
    diffs.sort()
    lo, hi = diffs[int(0.025 * n_boot)], diffs[min(n_boot - 1, int(0.975 * n_boot))]
    return {"diff": diff, "ci95": (lo, hi), "n": n, "n_boot": n_boot}


def paired_summary(rows_a: Sequence[Dict], rows_b: Sequence[Dict], *, label_a: str = "all", label_b: str = "budgeted") -> Dict:
    pairs = pair_rows(rows_a, rows_b)
    return {"arms": [label_a, label_b], "pairs": len(pairs), "mcnemar": mcnemar_exact(pairs),
            "bootstrap": paired_bootstrap_diff(pairs)}


def main(argv: Sequence[str] = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(__doc__)
        return
    results = json.loads(Path(args[0]).read_text())
    arms = [k for k in results if isinstance(results[k], dict) and "rows" in results[k]]
    if len(arms) < 2:
        print(f"need two arms with rows; found {arms}")
        return
    a, b = arms[0], arms[1]
    out = paired_summary(results[a]["rows"], results[b]["rows"], label_a=a, label_b=b)
    m, bs = out["mcnemar"], out["bootstrap"]
    print(f"{a} vs {b}: {out['pairs']} paired queries")
    print(f"  {b} right / {a} wrong: {m['b_only']}   {a} right / {b} wrong: {m['a_only']}   "
          f"exact McNemar p = {m['p_value']:.2e}")
    print(f"  accuracy difference {bs['diff']:+.1%}, paired bootstrap 95% CI [{bs['ci95'][0]:+.1%}, {bs['ci95'][1]:+.1%}]")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()


__all__ = ["mcnemar_exact", "pair_rows", "paired_bootstrap_diff", "paired_summary", "wilson"]
