"""Cost eval: estimated $/user/day under four configurations.

This is a *model*, not a measurement: it combines the measured token savings from
``roster_eval`` and ``memory_eval`` with an explicit, editable assumption about a day's call
mix, and prices each call with the router's price table. All assumptions are printed with the
results so they can be argued with.

Configurations:
    upstream        strong model everywhere, full roster, full per-agent history
    routing_only    cheap tier for mechanical roles + email pre-filter; context unchanged
    context_only    budgeted roster + tiered memory; strong model everywhere
    both            everything on (this repo's defaults)

Usage:
    python -m evals.cost_eval [--agents 200] [--history-entries 160]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

from server.services.llm.router import (
    ROLE_EMAIL_CLASSIFIER,
    ROLE_EXECUTION,
    ROLE_EXECUTION_SEARCH,
    ROLE_INTERACTION,
    ROLE_MEMORY_COMPACTOR,
    ROLE_SUMMARIZER,
    ModelRouter,
)

# ---- a day in the life (editable) -----------------------------------------------------
ASSUMPTIONS: Dict[str, float] = {
    "user_turns_per_day": 40,
    "interaction_calls_per_user_turn": 1.6,  # tool loop averages >1 call
    "agent_result_turns_per_user_turn": 0.8,  # execution results re-enter the interaction agent
    "interaction_calls_per_agent_turn": 1.2,
    "interaction_base_prompt_tokens": 2500,  # system prompt + conversation window, excl. roster
    "interaction_completion_tokens": 120,
    "execution_runs_per_user_turn": 1.2,
    "execution_calls_per_run": 2.5,
    "execution_base_prompt_tokens": 1500,  # system prompt + tools, excl. history
    "execution_completion_tokens": 200,
    "email_searches_per_user_turn": 0.3,
    "email_search_calls": 3,
    "email_search_prompt_tokens": 3000,
    "email_search_completion_tokens": 150,
    "emails_classified_per_day": 120,
    "classifier_prompt_tokens": 1200,
    "classifier_completion_tokens": 40,
    "summaries_per_day": 1,
    "summary_prompt_tokens": 8000,
    "summary_completion_tokens": 600,
    "prefilter_skip_rate": 0.50,  # measured on tests/fixtures/emails.json (evals/email_eval.py)
    "compactions_per_day": 6,  # only when tiered memory is on
    "compaction_prompt_tokens": 4000,
    "compaction_completion_tokens": 400,
}


@dataclass
class ConfigCost:
    config: str
    usd_per_day: float
    usd_per_month: float
    by_role: Dict[str, float]
    tokens_per_day: int


def _load_measurement(path: Path, key: str, match: Dict[str, object], default: float) -> float:
    if not path.exists():
        return default
    try:
        for row in json.loads(path.read_text()):
            if all(row.get(k) == v for k, v in match.items()):
                return float(row[key])
    except Exception:
        pass
    return default


def estimate(router: ModelRouter, *, roster_tokens: float, history_tokens: float,
             tiered_memory: bool, prefilter: bool = False,
             a: Dict[str, float] = ASSUMPTIONS) -> ConfigCost:
    by_role: Dict[str, float] = {}
    tokens_total = 0

    def add(role: str, calls: float, prompt: float, completion: float) -> None:
        nonlocal tokens_total
        price = router.price_for(router.model_for(role))
        cost = calls * price.cost(int(prompt), int(completion))
        by_role[role] = by_role.get(role, 0.0) + cost
        tokens_total += int(calls * (prompt + completion))

    turns = a["user_turns_per_day"]
    interaction_calls = turns * a["interaction_calls_per_user_turn"] + \
        turns * a["agent_result_turns_per_user_turn"] * a["interaction_calls_per_agent_turn"]
    add(ROLE_INTERACTION, interaction_calls,
        a["interaction_base_prompt_tokens"] + roster_tokens, a["interaction_completion_tokens"])
    add(ROLE_EXECUTION, turns * a["execution_runs_per_user_turn"] * a["execution_calls_per_run"],
        a["execution_base_prompt_tokens"] + history_tokens, a["execution_completion_tokens"])
    add(ROLE_EXECUTION_SEARCH, turns * a["email_searches_per_user_turn"] * a["email_search_calls"],
        a["email_search_prompt_tokens"], a["email_search_completion_tokens"])
    classified = a["emails_classified_per_day"] * ((1 - a["prefilter_skip_rate"]) if prefilter else 1.0)
    add(ROLE_EMAIL_CLASSIFIER, classified,
        a["classifier_prompt_tokens"], a["classifier_completion_tokens"])
    add(ROLE_SUMMARIZER, a["summaries_per_day"], a["summary_prompt_tokens"], a["summary_completion_tokens"])
    if tiered_memory:
        add(ROLE_MEMORY_COMPACTOR, a["compactions_per_day"],
            a["compaction_prompt_tokens"], a["compaction_completion_tokens"])

    day = sum(by_role.values())
    return ConfigCost(config="", usd_per_day=day, usd_per_month=day * 30,
                      by_role={k: round(v, 4) for k, v in by_role.items()}, tokens_per_day=tokens_total)


def run(agents: int, history_entries: int, out_dir: Path) -> List[ConfigCost]:
    roster_all = _load_measurement(out_dir / "roster_results.json", "roster_tokens_mean",
                                   {"strategy": "all", "n_agents": agents}, default=agents * 11.2)
    roster_budgeted = _load_measurement(out_dir / "roster_results.json", "roster_tokens_mean",
                                        {"strategy": "budgeted", "n_agents": agents}, default=500.0)
    hist_full = _load_measurement(out_dir / "memory_results.json", "upstream_chars",
                                  {"entries": history_entries}, default=history_entries * 224) / 4
    hist_tiered = _load_measurement(out_dir / "memory_results.json", "compacted_chars",
                                    {"entries": history_entries}, default=7000) / 4

    strong_only = ModelRouter(mode="strong_only")
    tiered = ModelRouter(mode="tiered")
    rows = [
        ("upstream", estimate(strong_only, roster_tokens=roster_all, history_tokens=hist_full, tiered_memory=False)),
        ("routing_only", estimate(tiered, roster_tokens=roster_all, history_tokens=hist_full, tiered_memory=False, prefilter=True)),
        ("context_only", estimate(strong_only, roster_tokens=roster_budgeted, history_tokens=hist_tiered, tiered_memory=True)),
        ("both", estimate(tiered, roster_tokens=roster_budgeted, history_tokens=hist_tiered, tiered_memory=True, prefilter=True)),
    ]
    results = []
    for name, cost in rows:
        cost.config = name
        results.append(cost)
    payload = {
        "inputs": {
            "agents": agents, "history_entries": history_entries,
            "roster_tokens_all": roster_all, "roster_tokens_budgeted": roster_budgeted,
            "history_tokens_full": hist_full, "history_tokens_tiered": hist_tiered,
            "strong_model": tiered.strong_model, "cheap_model": tiered.cheap_model,
            "prices_usd_per_1M": {m: list(p) for m, p in tiered.prices.items()},
        },
        "assumptions": ASSUMPTIONS,
        "results": [asdict(r) for r in results],
    }
    (out_dir / "cost_results.json").write_text(json.dumps(payload, indent=2))
    return results


def print_table(results: List[ConfigCost]) -> None:
    base = results[0].usd_per_day
    print(f"{'config':<14} {'$/day':>8} {'$/month':>9} {'vs upstream':>12} {'tokens/day':>11}")
    for r in results:
        print(f"{r.config:<14} {r.usd_per_day:>8.2f} {r.usd_per_month:>9.2f} {(1 - r.usd_per_day / base) * 100:>11.0f}% {r.tokens_per_day:>11,}")
    print("\nby role ($/day):")
    roles = sorted({k for r in results for k in r.by_role})
    print(f"{'role':<18}" + "".join(f"{r.config:>14}" for r in results))
    for role in roles:
        print(f"{role:<18}" + "".join(f"{r.by_role.get(role, 0.0):>14.3f}" for r in results))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agents", type=int, default=200, help="population size to read from roster_results.json")
    parser.add_argument("--history-entries", type=int, default=160, help="agent log length to read from memory_results.json")
    parser.add_argument("--out", default="evals/out")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = run(args.agents, args.history_entries, out_dir)
    print(f"inputs: {args.agents} agents, {args.history_entries}-entry agent history "
          f"(see {out_dir / 'cost_results.json'} for every assumption)\n")
    print_table(results)


if __name__ == "__main__":
    main()
