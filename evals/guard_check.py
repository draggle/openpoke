"""Offline check of the did-you-mean routing guard on the synthetic workload (seeds 7-9).

Reports (a) false fires: the guard challenging the *correct* target, and (b) catches: the guard
challenging a same-topic agent about a different person. Writes evals/out/guard_check.json.

    python -m evals.guard_check
"""

from __future__ import annotations

import json
from pathlib import Path

from server.services.execution.registry import RegistryConfig

from .roster_eval import build_registry
from .synthetic import generate


def run(out_dir: Path = Path("evals/out"), seeds=(7, 8, 9), n_agents: int = 150, n_queries: int = 40) -> dict:
    false_fires = caught = total = candidates = 0
    examples = []
    for seed in seeds:
        workload = generate(n_agents, seed=seed, n_queries=n_queries, stratified=True)
        registry = build_registry(workload, out_dir / "tmp", RegistryConfig())
        by_name = workload.by_name()
        for query in workload.queries:
            total += 1
            if registry.routing_mismatch(query.target, query.text) is not None:
                false_fires += 1
                examples.append({"seed": seed, "text": query.text, "target": query.target})
            target = by_name[query.target]
            wrong = [a for a in workload.agents if a.topic == target.topic and a.person != target.person]
            if wrong and query.kind in ("keyword", "name_only"):
                candidates += 1
                caught += registry.routing_mismatch(wrong[0].name, query.text) is not None
    result = {"queries": total, "false_fires_on_correct_target": false_fires, "false_fire_examples": examples,
              "wrong_person_picks": candidates, "wrong_person_picks_caught": caught}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "guard_check.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
