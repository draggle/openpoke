"""A hand-written held-out routing set.

The synthetic generator draws names, topics and query phrasings from the same lists the fork's
roster selection was tuned against. This set was written by hand, in one sitting, before it was
ever run, and it is never used for tuning: 50 agents and 72 follow-ups with the failure modes a
real roster has and the generator does not draw deliberately. Every query names its kind so the
table can say which ones a roster design survives:

    keyword      person + topic ("what did Priya say about the invoice")
    name_only    person only, who has several threads ("anything from Priya?")
    topic_only   topic only, several people ("is the flight sorted?")
    recency      bare follow-up to the last thread ("ok send it")
    lookalike    two agents for the same person, told apart by one topic word
    nickname     the user says Alex, Kate, Sam; the agent says Alexander, Katherine, Samantha
    typo         the user misspells the person
    paraphrase   none of the agent's own words ("the apartment thing" for a lease renewal)
    name_opaque  the entity the user names lives only in the agent's instructions, not its name

The expectations block was written before the first run and is checked by the test, so the
set cannot quietly become a tuning set.

    python -m evals.live_routing_eval --fixture evals/heldout_routing.json --history
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from evals.synthetic import DAY, Query, SyntheticAgent, Workload

FIXTURE = Path(__file__).resolve().parent / "heldout_routing.json"
NOW = 1_800_000_000.0


def load_fixture(path: Path = FIXTURE) -> Workload:
    data: Dict = json.loads(Path(path).read_text())
    agents: List[SyntheticAgent] = []
    for a in data["agents"]:
        last_used = NOW - a["last_used_days_ago"] * DAY
        created = NOW - a.get("created_days_ago", a["last_used_days_ago"] + 1) * DAY
        agents.append(SyntheticAgent(
            name=a["name"], person=a.get("person", ""), topic=a.get("topic", ""), desc=a.get("desc", a.get("topic", "")),
            instructions=list(a["instructions"]), summary=a.get("summary", ""),
            created_at=min(created, last_used), last_used_at=last_used,
            use_count=int(a.get("use_count", len(a["instructions"]))), has_trigger=bool(a.get("has_trigger", False)),
        ))
    by_name = {a.name: a for a in agents}
    queries = [Query(text=q["text"], target=q["target"], kind=q["kind"],
                     target_age_days=(NOW - by_name[q["target"]].last_used_at) / DAY if q["target"] in by_name else -1.0)
               for q in data["queries"]]
    return Workload(now=NOW, agents=agents, queries=queries, seed=0, span_days=int(data.get("span_days", 45)))


__all__ = ["FIXTURE", "NOW", "load_fixture"]


def grade(results: Dict, expectations: Dict = None) -> List[Dict]:
    """Score a live run of the fixture against the bars written before it ran."""
    exp = expectations if expectations is not None else json.loads(FIXTURE.read_text())["expectations"]
    rows: List[Dict] = []
    budgeted, upstream = results.get("budgeted", {}), results.get("all", {})
    if "budgeted_accuracy_at_least" in exp:
        v = budgeted.get("accuracy", 0.0)
        rows.append({"bar": "budgeted_accuracy_at_least", "value": v, "bar_value": exp["budgeted_accuracy_at_least"],
                     "passed": v >= exp["budgeted_accuracy_at_least"]})
    if "upstream_accuracy_at_most" in exp:
        v = upstream.get("accuracy", 1.0)
        rows.append({"bar": "upstream_accuracy_at_most", "value": v, "bar_value": exp["upstream_accuracy_at_most"],
                     "passed": v <= exp["upstream_accuracy_at_most"]})
    if exp.get("paired_bootstrap_ci_excludes_zero"):
        lo, hi = results.get("paired", {}).get("bootstrap", {}).get("ci95", (0.0, 0.0))
        rows.append({"bar": "paired_bootstrap_ci_excludes_zero", "value": [lo, hi], "bar_value": "0 outside",
                     "passed": lo > 0.0 or hi < 0.0})
    for kind, bar in exp.get("budgeted_by_kind_at_least", {}).items():
        v = budgeted.get("by_kind", {}).get(kind, {}).get("acc", 0.0)
        rows.append({"bar": f"kind:{kind}", "value": v, "bar_value": bar, "passed": v >= bar})
    return rows


def main() -> None:
    import sys

    if len(sys.argv) < 2:
        print(__doc__)
        return
    results = json.loads(Path(sys.argv[1]).read_text())
    report = grade(results)
    for r in report:
        mark = "PASS" if r["passed"] else "MISS"
        value = r["value"] if not isinstance(r["value"], float) else f"{r['value']:.2f}"
        print(f"{mark}  {r['bar']:<36} {value}  (bar {r['bar_value']})")
    print(f"{sum(r['passed'] for r in report)} of {len(report)} bars met")


if __name__ == "__main__":
    main()
