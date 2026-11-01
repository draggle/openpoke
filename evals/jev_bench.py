"""Jev benchmark: is a typed classifier worth it? Before vs after, side by side.

Two places where OpenPoke makes a *decision* with a chat model:

  email   "should I interrupt the user about this email?"   -> Jev noul gate, LLM only summarises
  routing "which existing agent does this message belong to?" -> Jev choice over the roster

OFFLINE (default, no keys): counts model calls, tokens and modelled $ for both configs over
the 40-email labelled set (scaled to a day) and the synthetic routing workload, with a stated
per-call latency model. The Jev stand-in is an oracle, so offline *accuracy* is not measured.

LIVE (``--live``; needs TYPESAFE_API_KEY and OPENROUTER_API_KEY): measures real wall-clock
latency and accuracy for both configs:
  email   LLM classifier vs Jev gate: precision / recall / F1 / p50 latency / $ per 120 emails
  routing LLM-with-roster vs Jev choice: accuracy / p50 latency / $ per decision

Usage:
    python -m evals.jev_bench                 # offline
    python -m evals.jev_bench --live --queries 40
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional

from server.services.execution.registry import RegistryConfig
from server.services.execution.routing_hint import NONE_OPTION, compute_routing_hint
from server.services.gmail.prefilter import prefilter
from server.services.llm import UsageLedger, get_usage_ledger, set_jev_client, set_usage_ledger
from server.services.llm.jev import Answer, FakeJev, JevClient, get_jev_client
from server.services.llm.router import ROLE_EMAIL_CLASSIFIER, get_model_router

from .email_eval import load_fixtures, to_processed
from .roster_eval import build_registry
from .synthetic import generate

# Per-call latency model for the offline estimate. Edit freely; printed with results.
LATENCY_S = {
    "llm_classify": 1.8,  # chat model, tool-call JSON, ~1.2k-token prompt
    "llm_summarise": 1.8,
    "jev": 0.15,  # typed decision, parallel decoding (vendor/user reports: 5-18x faster than LLM classify)
}
EMAILS_PER_DAY = 120
REALISTIC_IMPORTANT_RATE = 0.10  # share of non-prefiltered mail that is truly important


# =============================================================================
# EMAIL
# =============================================================================


async def email_offline() -> Dict[str, Dict]:
    rows = load_fixtures()
    price = get_model_router().price_for(get_model_router().model_for(ROLE_EMAIL_CLASSIFIER))
    llm_classify_cost = price.cost(1200, 40)
    jev_cost = 1200 * 0.0578 / 1_000_000

    kept = [r for r in rows if not prefilter(to_processed(r)).skip]
    fixture_rate = sum(r["important"] for r in kept) / len(kept)
    kept_per_day = EMAILS_PER_DAY * len(kept) / len(rows)

    out = {}
    # The fixture is deliberately important-heavy (to test the prefilter); a real inbox is
    # mostly noise. Report both base rates so the saving is not overstated or understated.
    for label, rate in (("fixture base rate", fixture_rate), ("realistic inbox", REALISTIC_IMPORTANT_RATE)):
        before = {"llm": kept_per_day, "jev": 0.0}
        after = {"llm": kept_per_day * rate, "jev": kept_per_day}  # oracle Jev: LLM only for important
        for name, c in ((f"before (prefilter + LLM) @ {label} {rate:.0%}", before),
                        (f"after  (prefilter + Jev gate + LLM summary) @ {label} {rate:.0%}", after)):
            usd = c["llm"] * llm_classify_cost + c["jev"] * jev_cost
            latency = c["llm"] * LATENCY_S["llm_classify"] + c["jev"] * LATENCY_S["jev"]
            out[name] = {
                "llm_calls_per_day": round(c["llm"], 1),
                "jev_calls_per_day": round(c["jev"], 1),
                "usd_per_day": round(usd, 4),
                "model_seconds_per_day": round(latency, 1),
                "p50_decision_latency_s": LATENCY_S["jev"] if c["jev"] else LATENCY_S["llm_classify"],
            }
    return out


def gate_sweep(rows: List[Dict], *, lows=(0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5), highs=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95)) -> List[Dict]:
    """Recall / precision / LLM calls of a two-sided gate over recorded per-email rows.

    Each row: ``p`` (Jev P(important)), ``llm_important`` (what the full classifier said when
    it ran), ``truth``. Below ``low`` the mail is dropped with no LLM call; at or above ``high``
    it is surfaced (summary-only LLM call, cannot veto); in between the LLM verdict decides.
    """
    table: List[Dict] = []
    for low in lows:
        for high in highs:
            if high < low:
                continue
            tp = fp = fn = llm = 0
            for r in rows:
                p = r["p"]
                if p < low:
                    predicted = False
                elif p >= high:
                    predicted = True
                    llm += 1
                else:
                    predicted = bool(r["llm_important"])
                    llm += 1
                tp += predicted and r["truth"]
                fp += predicted and not r["truth"]
                fn += (not predicted) and r["truth"]
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            table.append({"low": low, "high": high, "recall": round(recall, 3), "precision": round(precision, 3),
                          "llm_calls": llm, "f1": round(2 * precision * recall / (precision + recall), 3) if precision + recall else 0.0})
    return table


def best_gate(table: List[Dict], floor_recall: float) -> Optional[Dict]:
    """The cheapest (fewest LLM calls) gate whose recall meets ``floor_recall`` at precision 1.0;
    ties broken by recall, then by the narrower band."""
    ok = [r for r in table if r["recall"] >= floor_recall and r["precision"] >= 0.999]
    if not ok:
        return None
    return min(ok, key=lambda r: (r["llm_calls"], -r["recall"], r["high"] - r["low"]))


async def email_live() -> Dict[str, Dict]:
    from server.services.gmail.importance_classifier import classify_email_importance, jev_importance_probability

    rows = [r for r in load_fixtures() if not prefilter(to_processed(r)).skip]
    results: Dict[str, Dict] = {}

    # BEFORE: chat-model classifier alone (Jev disabled). Keep each verdict for the sweep.
    set_jev_client(None)
    os.environ["OPENPOKE_JEV"] = "0"
    llm_verdicts: Dict[str, bool] = {}

    async def llm_only(e):
        verdict = (await classify_email_importance(e)) is not None
        llm_verdicts[e.id] = verdict
        return verdict

    results["before: LLM classifier"] = await _email_live_config(rows, llm_only)

    # AFTER: Jev gate (+ LLM summary for positives). Keep each probability for the sweep.
    os.environ["OPENPOKE_JEV"] = "1"
    set_jev_client(JevClient())
    threshold = float(os.getenv("OPENPOKE_JEV_EMAIL_THRESHOLD", "0.5"))
    probabilities: Dict[str, float] = {}

    async def jev_only(e):
        p = await jev_importance_probability(e)
        probabilities[e.id] = -1.0 if p is None else p
        return p is not None and p >= threshold

    results["after: Jev decision only"] = await _email_live_config(rows, jev_only)
    results["after: Jev gate + LLM summary"] = await _email_live_config(rows, lambda e: classify_email_importance(e))
    set_jev_client(None)

    per_email = [{"id": r["id"], "truth": bool(r["important"]), "p": probabilities.get(r["id"], -1.0),
                  "llm_important": llm_verdicts.get(r["id"], False)} for r in rows]
    sweep = gate_sweep([r for r in per_email if r["p"] >= 0])
    results["per_email"] = {"rows": per_email}
    results["gate_sweep"] = {"table": sweep,
                             "best_at_llm_recall": best_gate(sweep, results["before: LLM classifier"]["recall"]),
                             "note": "chosen on the dev set; check it on the held-out set before trusting it"}
    return results


async def _email_live_config(rows, decide) -> Dict:
    ledger = UsageLedger()
    set_usage_ledger(ledger)
    latencies: List[float] = []
    tp = fp = fn = 0
    for row in rows:
        t0 = time.perf_counter()
        predicted = bool(await decide(to_processed(row)))
        latencies.append(time.perf_counter() - t0)
        truth = row["important"]
        tp += predicted and truth
        fp += predicted and not truth
        fn += (not predicted) and truth
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    usd = ledger.totals()["total"]["cost_usd"]
    set_usage_ledger(None)
    return {
        "emails": len(rows), "precision": round(precision, 3), "recall": round(recall, 3),
        "f1": round(2 * precision * recall / (precision + recall), 3) if precision + recall else 0.0,
        "p50_latency_s": round(statistics.median(latencies), 3),
        "p95_latency_s": round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 3),
        "usd_per_120_emails": round(usd * EMAILS_PER_DAY / len(rows), 4),
    }


# =============================================================================
# ROUTING
# =============================================================================


def _oracle_jev(workload):
    target_of = {q.text: q.target for q in workload.queries}

    def answers_for(state, questions):
        q = questions["agent"]
        target = target_of.get(state.get("message", ""))
        label = target if target in q["criteria"] else NONE_OPTION
        return {"agent": Answer(kind="choice", choice=label, confidence=0.9, probabilities={label: 0.9})}

    return FakeJev(answers_for)


async def routing_offline(n_agents: int, n_queries: int, out_dir: Path) -> Dict[str, Dict]:
    workload = generate(n_agents, seed=7, n_queries=n_queries)
    registry = build_registry(workload, out_dir / "tmp", RegistryConfig())
    set_jev_client(_oracle_jev(workload))
    ledger = UsageLedger()
    set_usage_ledger(ledger)
    covered = hinted = 0
    # The hint is off by default in the runtime (it lost the live benchmark); the bench
    # exists to measure it, so enable it for the duration of this function only.
    previous = os.environ.get("OPENPOKE_JEV_ROUTING")
    os.environ["OPENPOKE_JEV_ROUTING"] = "1"
    try:
        for q in workload.queries:
            sel = registry.select(q.text, now=workload.now)
            covered += q.target in sel.names()
            hint = await compute_routing_hint(registry, sel, q.text)
            hinted += hint is not None and hint.agent_name == q.target
    finally:
        if previous is None:
            os.environ.pop("OPENPOKE_JEV_ROUTING", None)
        else:
            os.environ["OPENPOKE_JEV_ROUTING"] = previous
    jev_tokens = ledger.totals()["total"]["prompt_tokens"] / len(workload.queries)
    set_usage_ledger(None)
    set_jev_client(None)
    # LLM-side comparison: the Interaction Agent's own reasoning step is what the hint replaces.
    llm_price = get_model_router().price_for(get_model_router().strong_model)
    return {
        "candidates_contain_target": round(covered / len(workload.queries), 3),
        "oracle_hint_correct": round(hinted / len(workload.queries), 3),
        "jev_tokens_per_decision": round(jev_tokens),
        "jev_usd_per_decision": round(jev_tokens * 0.0578 / 1_000_000, 6),
        "llm_usd_per_extra_reasoning_call(3k prompt)": round(llm_price.cost(3000, 150), 5),
        "latency_model_s": {"jev": LATENCY_S["jev"], "llm_call": LATENCY_S["llm_classify"]},
        "note": "offline Jev is an oracle; accuracy needs --live",
    }


async def routing_live(n_agents: int, n_queries: int, out_dir: Path) -> Dict[str, Dict]:
    from .live_routing_eval import decide as llm_decide

    workload = generate(n_agents, seed=7, n_queries=n_queries)
    registry = build_registry(workload, out_dir / "tmp", RegistryConfig())
    results: Dict[str, Dict] = {}

    # BEFORE: LLM picks from the budgeted roster (no hint)
    set_jev_client(None)
    os.environ["OPENPOKE_JEV"] = "0"
    ledger = UsageLedger(); set_usage_ledger(ledger)
    lat, ok = [], 0
    for q in workload.queries:
        t0 = time.perf_counter()
        chosen = (await llm_decide(registry, workload, q.text)).agent
        lat.append(time.perf_counter() - t0)
        ok += chosen == q.target
    results["before: LLM over budgeted roster"] = _route_summary(ok, lat, ledger, len(workload.queries))

    # AFTER: Jev choice over the same candidates
    os.environ["OPENPOKE_JEV"] = "1"
    os.environ["OPENPOKE_JEV_ROUTING"] = "1"  # the bench measures the hint; the runtime default is off
    set_jev_client(JevClient())
    ledger = UsageLedger(); set_usage_ledger(ledger)
    lat, ok, abstain = [], 0, 0
    for q in workload.queries:
        sel = registry.select(q.text, now=workload.now)
        t0 = time.perf_counter()
        hint = await compute_routing_hint(registry, sel, q.text)
        lat.append(time.perf_counter() - t0)
        if hint is None:
            abstain += 1
        ok += hint is not None and hint.agent_name == q.target
    results["after: Jev choice"] = {**_route_summary(ok, lat, ledger, len(workload.queries)), "abstained": abstain}
    set_jev_client(None)
    set_usage_ledger(None)
    return results


def _route_summary(ok, lat, ledger, n) -> Dict:
    return {
        "accuracy": round(ok / n, 3),
        "p50_latency_s": round(statistics.median(lat), 3),
        "p95_latency_s": round(sorted(lat)[int(0.95 * (len(lat) - 1))], 3),
        "usd_per_decision": round(ledger.totals()["total"]["cost_usd"] / n, 6),
    }


# =============================================================================


def _print(title: str, table: Dict[str, Dict]) -> None:
    print(f"\n{title}")
    for config, metrics in table.items():
        if config == "per_email":
            continue
        if config == "gate_sweep":
            best = metrics.get("best_at_llm_recall")
            print(f"  gate sweep: {len(metrics['table'])} (low, high) pairs; best at LLM recall: {best}")
            continue
        print(f"  {config}")
        for k, v in metrics.items():
            print(f"      {k:<42} {v}")


async def main_async(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {"latency_model_s": LATENCY_S}
    if args.live:
        missing = [k for k in ("TYPESAFE_API_KEY", "OPENROUTER_API_KEY") if not os.getenv(k)]
        if missing:
            raise SystemExit(f"--live needs {', '.join(missing)}")
        results["email"] = await email_live()
        results["routing"] = await routing_live(args.agents, args.queries, out)
    else:
        print(f"offline mode: Jev is an oracle; latency model {LATENCY_S} (edit in evals/jev_bench.py)")
        results["email"] = await email_offline()
        results["routing"] = {"after: Jev choice over budgeted roster (oracle)": await routing_offline(args.agents, args.queries, out)}
    _print("EMAIL importance", results["email"])
    _print("AGENT routing", results["routing"])
    path = out / ("jev_bench_live.json" if args.live else "jev_bench.json")
    path.write_text(json.dumps(results, indent=2))
    print(f"\njson: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--agents", type=int, default=200)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--out", default="evals/out")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
