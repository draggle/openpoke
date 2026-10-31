"""LIVE routing eval (needs OPENROUTER_API_KEY): does the real Interaction model pick the
right agent, upstream roster vs budgeted roster + search_agents?

For each synthetic follow-up query we build the exact prompt the Interaction Agent would see
(system prompt + <conversation_history> + <active_agents> + <new_user_message>) and let the
model act with the real tool schemas. We follow at most 3 tool steps:

- ``send_message_to_agent(agent_name=X)``  -> decision = X
- ``search_agents(query)``                 -> we run the search and feed the result back
- anything else / no decision              -> counted as "no delegation"

A decision is correct if X equals the ground-truth target. Creating a new agent for a topic
that already has one counts as wrong (that *is* the overload failure mode).

Usage:
    OPENROUTER_API_KEY=... python -m evals.live_routing_eval [--agents 150] [--queries 40]

Cost: ~2 x queries x (1-3) model calls on the interaction tier. Keep --queries small.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

from server.agents.interaction_agent.agent import build_system_prompt
from server.agents.interaction_agent.tools import get_tool_schemas
from server.services.execution.registry import AgentRegistry, RegistryConfig
from server.services.llm import ROLE_INTERACTION, complete, get_usage_ledger, set_usage_ledger
from server.services.llm.usage import UsageLedger

from .roster_eval import build_registry
from .synthetic import Workload, generate


def _prompt(registry: AgentRegistry, workload: Workload, text: str) -> str:
    selection = registry.select(text, now=workload.now)
    roster = registry.render(selection, now=workload.now)
    return (
        "<conversation_history>\nNone\n</conversation_history>\n\n"
        f"<active_agents>\n{roster}\n</active_agents>\n\n"
        f"<new_user_message>\n{text}\n</new_user_message>"
    )


async def decide(registry: AgentRegistry, workload: Workload, text: str) -> Optional[str]:
    messages = [{"role": "user", "content": _prompt(registry, workload, text)}]
    for _ in range(3):
        response = await complete(role=ROLE_INTERACTION, messages=messages, system=build_system_prompt(),
                                  tools=get_tool_schemas(), tag="live_routing_eval")
        message = (response.get("choices") or [{}])[0].get("message") or {}
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
                return args.get("agent_name")
            if name == "search_agents":
                hits = registry.search(args.get("query", ""), limit=int(args.get("limit", 5) or 5))
                payload = {"query": args.get("query"), "matches": registry.render_search_results(hits),
                           "total_agents": len(registry)}
                messages.append({"role": "tool", "tool_call_id": call.get("id") or "search",
                                 "content": json.dumps({"tool": name, "status": "success", "result": payload})})
            else:
                messages.append({"role": "tool", "tool_call_id": call.get("id") or name,
                                 "content": json.dumps({"tool": name, "status": "success", "result": {"status": "ok"}})})
        if not message.get("tool_calls"):
            break
    return None


async def run(n_agents: int, n_queries: int, out_dir: Path) -> Dict[str, Dict]:
    workload = generate(n_agents, seed=7, n_queries=n_queries)
    results: Dict[str, Dict] = {}
    for strategy in ("all", "budgeted"):
        registry = build_registry(workload, out_dir / "tmp", RegistryConfig(strategy=strategy))
        ledger = UsageLedger()
        set_usage_ledger(ledger)
        correct = 0
        by_kind: Dict[str, list] = defaultdict(list)
        new_agent = none = 0
        for q in workload.queries:
            chosen = await decide(registry, workload, q.text)
            ok = chosen == q.target
            correct += ok
            by_kind[q.kind].append(ok)
            if chosen is None:
                none += 1
            elif chosen not in registry:
                new_agent += 1
            print(f"[{strategy}] {'OK ' if ok else 'MISS'} {q.kind:<10} {q.text[:45]:<45} -> {chosen!r} (want {q.target!r})")
        totals = ledger.totals()["total"]
        results[strategy] = {
            "accuracy": correct / len(workload.queries),
            "by_kind": {k: sum(v) / len(v) for k, v in by_kind.items()},
            "spawned_new_agent": new_agent,
            "no_delegation": none,
            "prompt_tokens_per_query": totals["prompt_tokens"] / len(workload.queries),
            "usd_total": totals["cost_usd"],
        }
    set_usage_ledger(None)
    (out_dir / "live_routing_results.json").write_text(json.dumps(results, indent=2))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agents", type=int, default=150)
    parser.add_argument("--queries", type=int, default=40)
    parser.add_argument("--out", default="evals/out")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = asyncio.run(run(args.agents, args.queries, out))
    print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
