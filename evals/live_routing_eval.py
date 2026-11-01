"""LIVE routing eval (needs an API key): does the real Interaction model pick the right agent,
upstream roster vs budgeted roster + search_agents?

For each synthetic follow-up query we build the exact prompt the Interaction Agent would see
(system prompt + <conversation_history> + <active_agents> + <new_user_message>) and let the
model act with the real tool schemas. We follow at most 3 tool steps:

- ``send_message_to_agent(agent_name=X)``  -> decision = X
- ``search_agents(query)``                 -> we run the search and feed the result back
- anything else / no decision              -> counted as "no delegation"

A decision is correct if X equals the ground-truth target. Creating a new agent for a topic
that already has one counts as wrong (that *is* the overload failure mode).

Fairness controls (round two):
- ``--history``: both arms see the same conversation tail (the last 6 interactions), which is
  the signal upstream relies on for "did she reply?". Without it the baseline is handicapped.
- ``--stratified``: equal numbers of each query kind, so no cell rests on one query.
- ``--seeds 7,8,9``: several workloads; results are pooled and reported with Wilson 95% CIs.
- ``search_agents`` calls are counted, and the assistant's text is kept for every non-delegation.

Usage:
    python -m evals.live_routing_eval [--agents 150] [--queries 40] [--history] [--stratified] [--seeds 7,8,9]

Cost: ~2 x queries x seeds x (1-3) model calls on the interaction tier.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from server.agents.interaction_agent.agent import agent_names_in_transcript, build_system_prompt
from server.agents.interaction_agent.tools import get_tool_schemas
from server.services.execution.registry import AgentRegistry, RegistryConfig
from server.services.llm import ROLE_INTERACTION, complete, set_usage_ledger
from server.services.llm.usage import UsageLedger

from .roster_eval import build_registry
from .stats import wilson
from .synthetic import Workload, generate, transcript_for


@dataclass
class Decision:
    agent: Optional[str]
    search_calls: int = 0
    none_text: str = ""  # what the model said when it did not delegate
    target_shown: bool = True  # was the target in the first roster the model saw?
    first_pick: Optional[str] = None  # the first agent the model tried to dispatch to
    guard_fired: bool = False  # the did-you-mean guard challenged that first pick


def _prompt(registry: AgentRegistry, workload: Workload, text: str, history: bool) -> tuple[str, bool]:
    transcript = transcript_for(workload) if history else ""
    recent = agent_names_in_transcript(transcript, registry.names()) if transcript else []
    selection = registry.select(text, now=workload.now, recent_agent_names=recent)
    roster = registry.render(selection, now=workload.now)
    shown_names = set(selection.names())
    body = (
        f"<conversation_history>\n{transcript or 'None'}\n</conversation_history>\n\n"
        f"<active_agents>\n{roster}\n</active_agents>\n\n"
        f"<new_user_message>\n{text}\n</new_user_message>"
    )
    return body, shown_names


async def decide(registry: AgentRegistry, workload: Workload, text: str, *, history: bool = False,
                 target: Optional[str] = None) -> Decision:
    body, shown = _prompt(registry, workload, text, history)
    messages = [{"role": "user", "content": body}]
    decision = Decision(agent=None, target_shown=(target in shown) if target else True)
    last_text = ""
    guarded = registry.config.strategy == "budgeted"  # upstream arm has no guard
    for _ in range(4):
        response = await complete(role=ROLE_INTERACTION, messages=messages, system=build_system_prompt(),
                                  tools=get_tool_schemas(), tag="live_routing_eval")
        message = (response.get("choices") or [{}])[0].get("message") or {}
        last_text = message.get("content") or last_text
        messages.append({"role": "assistant", "content": message.get("content") or "",
                         **({"tool_calls": message["tool_calls"]} if message.get("tool_calls") else {})})
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            name = fn.get("name")
            if name == "send_message_to_agent":
                chosen = args.get("agent_name")
                if decision.first_pick is None:
                    decision.first_pick = chosen
                mismatch = None
                if guarded and not decision.guard_fired and isinstance(chosen, str):
                    instructions = args.get("instructions") or text
                    mismatch = (registry.new_agent_mismatch(chosen, instructions, anchor_text=text)
                                if registry.get(chosen) is None
                                else registry.routing_mismatch(chosen, instructions, anchor_text=text))
                if mismatch is None:
                    decision.agent = chosen
                    return decision
                decision.guard_fired = True
                messages.append({"role": "tool", "tool_call_id": call.get("id") or "send",
                                 "content": json.dumps({"tool": name, "status": "error",
                                                        "arguments": args,
                                                        "error": {"error": mismatch.render(),
                                                                  "did_you_mean": mismatch.candidates}})})
                continue
            if name == "search_agents":
                decision.search_calls += 1
                hits = registry.search(args.get("query", ""), limit=int(args.get("limit", 5) or 5))
                payload = {"query": args.get("query"), "matches": registry.render_search_results(hits),
                           "total_agents": len(registry)}
                messages.append({"role": "tool", "tool_call_id": call.get("id") or "search",
                                 "content": json.dumps({"tool": name, "status": "success", "result": payload})})
            else:
                if name == "send_message_to_user":
                    last_text = args.get("message") or last_text
                messages.append({"role": "tool", "tool_call_id": call.get("id") or name,
                                 "content": json.dumps({"tool": name, "status": "success", "result": {"status": "ok"}})})
        if not message.get("tool_calls"):
            break
    decision.none_text = (last_text or "")[:200]
    return decision


def _answered_from_card(none_text: str, target) -> bool:
    """The model told the user something instead of delegating, and what it said shares at
    least two content words with the target agent's summary: it answered from the card."""
    if not none_text or target is None or none_text.rstrip().endswith("?"):
        return False
    from server.services.execution.registry import tokenize

    said = set(tokenize(none_text))
    card = set(tokenize(f"{target.summary} {target.description}"))
    return len(said & card) >= 2


def _summary(rows: List[Dict], ledger_totals: Dict, n_queries: int) -> Dict:
    ok = sum(r["ok"] for r in rows)
    n = len(rows)
    by_kind: Dict[str, Dict] = {}
    for kind in sorted({r["kind"] for r in rows}):
        sub = [r for r in rows if r["kind"] == kind]
        k = sum(r["ok"] for r in sub)
        lo, hi = wilson(k, len(sub))
        by_kind[kind] = {"acc": round(k / len(sub), 3), "n": len(sub), "ci95": [round(lo, 3), round(hi, 3)]}
    lo, hi = wilson(ok, n)
    needed = [r for r in rows if not r["target_shown"]]
    return {
        "accuracy": round(ok / n, 4),
        "n": n,
        "ci95": [round(lo, 3), round(hi, 3)],
        "by_kind": by_kind,
        "spawned_new_agent": sum(1 for r in rows if r["chosen"] is not None and not r["known"]),
        "no_delegation": sum(1 for r in rows if r["chosen"] is None),
        "search_calls": sum(r["search_calls"] for r in rows),
        "search_needed": len(needed),
        "search_hits": sum(r["ok"] for r in needed),
        "wrong_agent": sum(1 for r in rows if r["chosen"] is not None and not r["ok"]),
        # Did not delegate but answered the question from the roster card (or a search hit): the
        # content may be right, the policy is wrong (the card may be stale), so it is its own row.
        "answered_from_card": sum(1 for r in rows if r.get("answered_from_card")),
        "guard_fired": sum(1 for r in rows if r.get("guard_fired")),
        "guard_helped": sum(1 for r in rows if r.get("guard_fired") and r["ok"] and r.get("first_pick") != r["target"]),
        "guard_hurt": sum(1 for r in rows if r.get("guard_fired") and not r["ok"] and r.get("first_pick") == r["target"]),
        "prompt_tokens_per_query": round(ledger_totals["prompt_tokens"] / n, 1),
        "cached_tokens_per_query": round(ledger_totals.get("cached_tokens", 0) / n, 1),
        "usd_total": round(ledger_totals["cost_usd"], 4),
        "none_texts": [r["none_text"] for r in rows if r["chosen"] is None],
    }


async def run(n_agents: int, n_queries: int, out_dir: Path, strategies=("all", "budgeted"), *,
              history: bool = False, stratified: bool = False, seeds: Sequence[int] = (7,),
              fixture: Optional[Path] = None) -> Dict[str, Dict]:
    """``fixture`` replaces the generated workload with a hand-written one (evals/heldout_routing.json)."""
    results: Dict[str, Dict] = {}
    if fixture is not None:
        from evals.heldout_routing import load_fixture

        seeds = [0]
    for strategy in strategies:
        ledger = UsageLedger()
        set_usage_ledger(ledger)
        rows: List[Dict] = []
        for seed in seeds:
            workload = load_fixture(fixture) if fixture is not None else \
                generate(n_agents, seed=seed, n_queries=n_queries, stratified=stratified)
            registry = build_registry(workload, out_dir / "tmp", RegistryConfig(strategy=strategy))
            for q in workload.queries:
                d = await decide(registry, workload, q.text, history=history, target=q.target)
                ok = d.agent == q.target
                answered = d.agent is None and _answered_from_card(d.none_text, registry.get(q.target))
                rows.append({"seed": seed, "kind": q.kind, "text": q.text, "target": q.target, "chosen": d.agent,
                             "ok": ok, "known": d.agent in registry if d.agent else False,
                             "search_calls": d.search_calls, "target_shown": d.target_shown, "none_text": d.none_text,
                             "state": q.state, "first_pick": d.first_pick, "guard_fired": d.guard_fired,
                             "answered_from_card": answered})
                flag = "OK " if ok else "MISS"
                extra = f" [search x{d.search_calls}]" if d.search_calls else ""
                if d.guard_fired:
                    extra += f" [guard: first {d.first_pick!r}]"
                print(f"[{strategy} s{seed}] {flag} {q.kind:<10} {q.text[:45]:<45} -> {d.agent!r} (want {q.target!r}){extra}", flush=True)
        results[strategy] = _summary(rows, ledger.totals()["total"], n_queries)
        results[strategy]["rows"] = rows
    set_usage_ledger(None)
    if "all" in results and "budgeted" in results:
        # Both arms answered the same queries: the paired tests are the ones that count.
        from evals.stats import paired_summary

        results["paired"] = paired_summary(results["all"]["rows"], results["budgeted"]["rows"])
    suffix = "" if set(strategies) == {"all", "budgeted"} else "_" + "_".join(strategies)
    if history:
        suffix += "_history"
    if fixture is not None:
        suffix += "_heldout"
    (out_dir / f"live_routing_results{suffix}.json").write_text(json.dumps(results, indent=2))
    return results


def print_table(results: Dict[str, Dict]) -> None:
    for strategy, r in results.items():
        if strategy == "paired":
            m, bs = r["mcnemar"], r["bootstrap"]
            print(f"\npaired ({r['pairs']} queries through both arms): budgeted right / upstream wrong {m['b_only']}, "
                  f"the reverse {m['a_only']}, exact McNemar p = {m['p_value']:.1e}; "
                  f"difference {bs['diff']:+.1%}, paired bootstrap 95% CI [{bs['ci95'][0]:+.1%}, {bs['ci95'][1]:+.1%}]")
            continue
        lo, hi = r["ci95"]
        print(f"\n{strategy}: {r['accuracy']:.1%} (n={r['n']}, 95% CI {lo:.0%}-{hi:.0%}); "
              f"no delegation {r['no_delegation']}, new agents {r['spawned_new_agent']}, "
              f"search calls {r['search_calls']} (needed {r['search_needed']}, hit {r['search_hits']}); "
              f"wrong agent {r['wrong_agent']}, answered from card {r['answered_from_card']}, guard fired {r['guard_fired']} (helped {r['guard_helped']}, hurt {r['guard_hurt']}); "
              f"{r['prompt_tokens_per_query']} prompt tok/query, {r['cached_tokens_per_query']} cached; ${r['usd_total']}")
        for kind, k in r["by_kind"].items():
            print(f"  {kind:<10} {k['acc']:.0%}  n={k['n']:<3} CI {k['ci95'][0]:.0%}-{k['ci95'][1]:.0%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agents", type=int, default=150)
    parser.add_argument("--queries", type=int, default=40, help="queries per seed")
    parser.add_argument("--out", default="evals/out")
    parser.add_argument("--strategies", default="all,budgeted", help="comma-separated: all, budgeted")
    parser.add_argument("--history", action="store_true", help="give both arms the last 6 interactions")
    parser.add_argument("--stratified", action="store_true", help="equal numbers of each query kind")
    parser.add_argument("--seeds", default="7", help="comma-separated workload seeds")
    parser.add_argument("--fixture", default=None, help="hand-written workload JSON (evals/heldout_routing.json) instead of the generator")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    strategies = tuple(s.strip() for s in args.strategies.split(",") if s.strip())
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    results = asyncio.run(run(args.agents, args.queries, out, strategies, history=args.history,
                              stratified=args.stratified, seeds=seeds,
                              fixture=Path(args.fixture) if args.fixture else None))
    print_table(results)


if __name__ == "__main__":
    main()
