"""Roster eval: prompt cost vs. ability to find the right agent, baseline vs budgeted.

For each population size N we build a registry from the synthetic workload and, for every
query, ask: *is the correct agent in the roster the Interaction Agent will see?*

Baseline (upstream, ``strategy=all``) trivially shows everything, so its recall is 1.0 and
its cost grows linearly with N. The budgeted roster caps cost; we report how much recall it
gives up, and how much of that the ``search_agents`` fallback recovers.

Metrics per (N, strategy):
    roster_tokens        mean estimated tokens of the <active_agents> block per turn
    recall_shown         P(target in roster)
    recall_with_search   P(target in roster OR in top-5 search_agents(query))
    by_kind              recall_shown broken down by query flavour

Usage:
    python -m evals.roster_eval [--sizes 10,25,50,100,200,400] [--queries 300] [--out evals/out]
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

from server.services.execution.registry import AgentRegistry, RegistryConfig
from server.services.llm.usage import estimate_tokens

from .synthetic import Workload, generate


@dataclass
class StrategyResult:
    strategy: str
    n_agents: int
    roster_tokens_mean: float
    roster_entries_mean: float
    recall_shown: float
    recall_with_search: float
    recall_live: float  # recall_shown restricted to targets that are not archived
    by_kind: Dict[str, float]
    archived_targets: int  # queries whose target was archived (hard cases)


class _FixedClock:
    def __init__(self, t: float) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def build_registry(workload: Workload, tmp_dir: Path, config: RegistryConfig) -> AgentRegistry:
    """Replay the workload into a registry with a controllable clock."""
    clock = _FixedClock(0.0)
    pinned = {a.name for a in workload.agents if a.has_trigger}
    registry = AgentRegistry(
        tmp_dir / f"registry_{config.strategy}_{len(workload.agents)}.json",
        config=config,
        clock=clock,
        has_pending_triggers=lambda name: name in pinned,
    )
    registry.clear()
    # Replay in creation order, then set each agent's final last_used_at.
    for agent in sorted(workload.agents, key=lambda a: a.created_at):
        clock.t = agent.created_at
        for instr in agent.instructions:
            registry.register(agent.name, instr)
        registry.update_summary(agent.name, agent.summary)
    for agent in workload.agents:
        clock.t = agent.last_used_at
        registry.touch(agent.name, reason="user")
        rec = registry.get(agent.name)
        rec.use_count = agent.use_count
    clock.t = workload.now
    registry.save()
    return registry


def evaluate(workload: Workload, registry: AgentRegistry, strategy: str) -> StrategyResult:
    tokens: List[int] = []
    entries: List[int] = []
    hits_shown: List[bool] = []
    hits_search: List[bool] = []
    by_kind: Dict[str, List[bool]] = defaultdict(list)
    live_hits: List[bool] = []
    archived_targets = 0
    now = workload.now

    for query in workload.queries:
        selection = registry.select(query.text, now=now)
        rendered = registry.render(selection, now=now)
        tokens.append(estimate_tokens(rendered))
        entries.append(len(selection.shown))
        names = set(selection.names())
        shown = query.target in names
        if not shown and strategy != "all":
            search_hits = {r.name for r, _ in registry.search(query.text, limit=5)}
            with_search = query.target in search_hits
        else:
            with_search = shown
        hits_shown.append(shown)
        hits_search.append(with_search)
        by_kind[query.kind].append(shown)
        target = registry.get(query.target)
        if target is not None and registry.status_of(target, now).value == "archived":
            archived_targets += 1
        else:
            live_hits.append(shown)

    return StrategyResult(
        strategy=strategy,
        n_agents=len(workload.agents),
        roster_tokens_mean=statistics.mean(tokens),
        roster_entries_mean=statistics.mean(entries),
        recall_shown=sum(hits_shown) / len(hits_shown),
        recall_with_search=sum(hits_search) / len(hits_search),
        recall_live=(sum(live_hits) / len(live_hits)) if live_hits else 1.0,
        by_kind={k: sum(v) / len(v) for k, v in sorted(by_kind.items())},
        archived_targets=archived_targets,
    )


def run(sizes: List[int], n_queries: int, out_dir: Path, seed: int = 7) -> List[StrategyResult]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "tmp"
    tmp.mkdir(exist_ok=True)
    results: List[StrategyResult] = []
    configs = {
        "all": RegistryConfig(strategy="all"),
        "budgeted": RegistryConfig(strategy="budgeted"),
    }
    for n in sizes:
        workload = generate(n, seed=seed, n_queries=n_queries)
        for strategy, config in configs.items():
            registry = build_registry(workload, tmp, config)
            results.append(evaluate(workload, registry, strategy))
    (out_dir / "roster_results.json").write_text(json.dumps([asdict(r) for r in results], indent=2))
    return results


def print_table(results: List[StrategyResult]) -> None:
    print(f"{'N':>5} {'strategy':<9} {'entries':>7} {'tokens':>7} {'recall':>7} {'live':>6} {'+search':>8} {'arch.tgt':>8}  by kind")
    for r in results:
        kinds = " ".join(f"{k}={v:.2f}" for k, v in r.by_kind.items())
        print(f"{r.n_agents:>5} {r.strategy:<9} {r.roster_entries_mean:>7.1f} {r.roster_tokens_mean:>7.0f} "
              f"{r.recall_shown:>7.2f} {r.recall_live:>6.2f} {r.recall_with_search:>8.2f} {r.archived_targets:>8}  {kinds}")


def plot(results: List[StrategyResult], out_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"all": "#2a78d6", "budgeted": "#eb6834"}
    labels = {"all": "upstream (all agents)", "budgeted": "budgeted roster"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)
    for strategy in ("all", "budgeted"):
        rows = [r for r in results if r.strategy == strategy]
        xs = [r.n_agents for r in rows]
        axes[0].plot(xs, [r.roster_tokens_mean for r in rows], marker="o", ms=5, lw=2,
                     color=colours[strategy], label=labels[strategy])
        axes[1].plot(xs, [r.recall_shown for r in rows], marker="o", ms=5, lw=2,
                     color=colours[strategy], label=labels[strategy])
        if strategy == "budgeted":
            axes[1].plot(xs, [r.recall_with_search for r in rows], marker="o", ms=5, lw=2,
                         ls="--", color=colours[strategy], label="budgeted + search_agents")
    axes[0].set_title("Roster tokens per turn", fontsize=11)
    axes[0].set_xlabel("agents ever created")
    axes[0].set_ylabel("est. tokens in <active_agents>")
    axes[1].set_title("Correct agent reachable", fontsize=11)
    axes[1].set_xlabel("agents ever created")
    axes[1].set_ylabel("recall")
    axes[1].set_ylim(0, 1.05)
    for ax in axes:
        ax.set_xscale("log")
        ax.grid(True, alpha=0.25)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    path = out_dir / "roster_tokens_vs_agents.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", default="10,25,50,100,200,400")
    parser.add_argument("--queries", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="evals/out")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    out_dir = Path(args.out)
    results = run(sizes, args.queries, out_dir, seed=args.seed)
    print_table(results)
    if not args.no_plot:
        print(f"\nchart: {plot(results, out_dir)}")
    print(f"json:  {out_dir / 'roster_results.json'}")


if __name__ == "__main__":
    main()
