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
    "prefilter_skip_rate": 0.26,  # measured: 12 of 46 dev emails (evals/email_eval.py, round three)
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
    """Value of ``key`` in the row matching ``match``; for ``entries`` the nearest row wins."""
    if not path.exists():
        return default
    try:
        rows = json.loads(path.read_text())
        exact = [r for r in rows if all(r.get(k) == v for k, v in match.items())]
        if exact:
            return float(exact[0][key])
        if "entries" in match:
            others = {k: v for k, v in match.items() if k != "entries"}
            near = [r for r in rows if all(r.get(k) == v for k, v in others.items()) and "entries" in r]
            if near:
                best = min(near, key=lambda r: abs(int(r["entries"]) - int(match["entries"])))
                return float(best[key])
    except Exception:
        pass
    return default


_DOCS_LIVE = Path(__file__).resolve().parent.parent / "docs" / "live"
_LIVE_MEMORY_CANDIDATES = (_DOCS_LIVE / "round3" / "memory_results_live.json", _DOCS_LIVE / "round2" / "memory_results_live.json",
                           _DOCS_LIVE / "memory_results_live.json")
_LIVE_MEMORY = next((p for p in _LIVE_MEMORY_CANDIDATES if p.exists()), _LIVE_MEMORY_CANDIDATES[0])


def _history_measurements(out_dir: Path, history_entries: int) -> tuple:
    """(full, tiered) history tokens. Prefers the LLM extractor's live measurement, which is the
    runtime default; the heuristic's smaller output would overstate the saving."""
    source = _LIVE_MEMORY if _LIVE_MEMORY.exists() else out_dir / "memory_results.json"
    full = _load_measurement(source, "upstream_chars", {"entries": history_entries}, default=history_entries * 224) / 4
    tiered = _load_measurement(source, "compacted_chars", {"entries": history_entries}, default=min(full, 2600)) / 4
    if history_entries == 0:
        full = tiered = 0.0
    return full, tiered, str(source)


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
    hist_full, hist_tiered, _source = _history_measurements(out_dir, history_entries)

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


def sensitivity(router: ModelRouter, *, agents: int, history_entries=(0, 40, 80, 160),
                runs_per_turn=(0.5, 1.2), out_dir: Path) -> List[Dict[str, float]]:
    """How the headline saving moves with the two assumptions that drive it: how long the
    typical agent's history is, and how many execution runs a user turn causes."""
    roster_all = _load_measurement(out_dir / "roster_results.json", "roster_tokens_mean",
                                   {"strategy": "all", "n_agents": agents}, default=agents * 11.2)
    roster_budgeted = _load_measurement(out_dir / "roster_results.json", "roster_tokens_mean",
                                        {"strategy": "budgeted", "n_agents": agents}, default=500.0)
    rows: List[Dict[str, float]] = []
    for runs in runs_per_turn:
        a = {**ASSUMPTIONS, "execution_runs_per_user_turn": runs}
        for entries in history_entries:
            full, tiered, _ = _history_measurements(out_dir, entries)
            base = estimate(router, roster_tokens=roster_all, history_tokens=full, tiered_memory=False, a=a)
            both = estimate(router, roster_tokens=roster_budgeted, history_tokens=tiered, tiered_memory=True,
                            prefilter=True, a=a)
            exec_share = both.by_role.get("execution", 0.0) / max(both.usd_per_day, 1e-9)
            rows.append({
                "execution_runs_per_user_turn": runs, "history_entries": entries,
                "upstream_usd_per_day": round(base.usd_per_day, 2), "both_usd_per_day": round(both.usd_per_day, 2),
                "saving_pct": round((1 - both.usd_per_day / base.usd_per_day) * 100, 1),
                "execution_share_after": round(exec_share * 100, 1),
            })
    return rows


def print_sensitivity(rows: List[Dict[str, float]]) -> None:
    print("\nsensitivity (saving vs upstream, both fixes):")
    print(f"{'runs/turn':>9} {'history':>8} {'upstream $/d':>13} {'fixed $/d':>10} {'saving':>7} {'exec share':>11}")
    for r in rows:
        print(f"{r['execution_runs_per_user_turn']:>9} {r['history_entries']:>8} {r['upstream_usd_per_day']:>13.2f} "
              f"{r['both_usd_per_day']:>10.2f} {r['saving_pct']:>6.0f}% {r['execution_share_after']:>10.0f}%")


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
    from server.services.llm.router import get_model_router

    rows = sensitivity(get_model_router(), agents=args.agents, out_dir=out_dir)
    print_sensitivity(rows)
    (out_dir / "cost_sensitivity.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
