"""Lifecycle over time: does the roster's cost stay flat as the weeks go by?

A roster that is bounded per turn can still rot over months: agents that are never archived
keep competing for the budgeted slots, and the upstream roster grows without limit. This
simulation plays ``days`` of use against a registry with a controllable clock: every day the
user opens a few new threads, returns to a few recent ones, and sets the occasional reminder.
After each day's traffic one follow-up is rendered through both strategies and its
``<active_agents>`` block is measured. Deterministic, no model.

    python -m evals.lifecycle_eval [--days 60] [--seed 3]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Dict, List

from evals.synthetic import DAY, PEOPLE, TOPICS
from server.services.execution.registry import AgentRegistry, AgentStatus, RegistryConfig
from server.services.llm.usage import estimate_tokens

_FOLLOW_UPS = ["any update?", "did they reply?", "where are we on that", "ok send it", "and then?"]


class _Clock:
    def __init__(self, t: float) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _slope(ys: List[float]) -> float:
    """Least-squares slope of ``ys`` against day index (tokens per day)."""
    n = len(ys)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0


def simulate(*, days: int = 60, tmp_dir: Path = Path("evals/out/tmp"), seed: int = 3,
             new_per_day: int = 3, reuse_per_day: int = 2, reminder_every: int = 4) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for strategy in ("all", "budgeted"):
        rng = random.Random(seed)
        start = 1_800_000_000.0
        clock = _Clock(start)
        triggers: set = set()
        registry = AgentRegistry(Path(tmp_dir) / f"lifecycle_{strategy}.json", config=RegistryConfig(strategy=strategy),
                                 clock=clock, has_pending_triggers=lambda name: name in triggers)
        registry.clear()
        per_day: List[Dict] = []
        reminder_owners_archived = 0
        for day in range(days):
            clock.t = start + day * DAY + 9 * 3600
            names = registry.names()
            for i in range(new_per_day):
                person = rng.choice(PEOPLE)
                topic, desc, verb = rng.choice(TOPICS)
                name = f"{person} {desc.title()} d{day}"
                registry.register(name, f"{verb} {person} about the {desc}")
                if day % reminder_every == 0 and i == 0:
                    triggers.add(name)
                clock.t += 600
            for _ in range(min(reuse_per_day, len(names))):
                recent = sorted(names, key=lambda n: -registry.get(n).last_used_at)[: 15]
                target = rng.choice(recent)
                registry.register(target, "follow up on this")
                clock.t += 600
            text = rng.choice(_FOLLOW_UPS)
            selection = registry.select(text, now=clock.t)
            rendered = registry.render(selection, now=clock.t)
            archived = selection.archived
            total = selection.total
            live = total - archived
            for name in triggers:
                if registry.status_of(registry.get(name), now=clock.t) == AgentStatus.ARCHIVED:
                    reminder_owners_archived += 1
            per_day.append({"day": day, "total": total, "live": live, "archived": archived,
                            "shown": len(selection.shown), "tokens": estimate_tokens(rendered)})
        tokens = [d["tokens"] for d in per_day]
        out[strategy] = {
            "days": days,
            "tokens_day_5": tokens[min(4, len(tokens) - 1)],
            "tokens_day_last": tokens[-1],
            "tokens_mean": round(statistics.mean(tokens), 1),
            "tokens_per_day_slope": round(_slope(tokens), 2),
            "total_agents_last": per_day[-1]["total"],
            "live_agents_last": per_day[-1]["live"],
            "live_agents_peak": max(d["live"] for d in per_day),
            "archived_last": per_day[-1]["archived"],
            "shown_last": per_day[-1]["shown"],
            "reminder_owners": len(triggers),
            "reminder_owners_archived": reminder_owners_archived,
            "per_day": per_day,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--out", default="evals/out")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = simulate(days=args.days, tmp_dir=out_dir / "tmp", seed=args.seed)
    print(f"{'strategy':<9} {'day5 tok':>8} {'last tok':>8} {'slope/day':>9} {'total':>6} {'live':>5} {'archived':>8} {'shown':>5} {'reminders archived':>18}")
    for s, r in results.items():
        print(f"{s:<9} {r['tokens_day_5']:>8} {r['tokens_day_last']:>8} {r['tokens_per_day_slope']:>9} {r['total_agents_last']:>6} "
              f"{r['live_agents_last']:>5} {r['archived_last']:>8} {r['shown_last']:>5} {r['reminder_owners_archived']:>6}/{r['reminder_owners']}")
    (out_dir / "lifecycle_results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
