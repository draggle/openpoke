"""Memory eval: how much prompt does compaction save, and what does it lose?

We synthesise a long-lived execution agent (one person, many rounds of email back-and-forth)
and plant *needles* - facts the agent should still know later:

- ``surface`` needles appear in agent requests/responses (where the LLM writes its summary)
- ``buried``  needles appear only inside raw tool payloads (an email body returned by Gmail)
- ``commitment`` needles are reminder requests ("remind me Friday ... crown")

Then, for increasing history lengths, we compare the upstream prompt (full transcript) against
the compacted prompt, measuring:

    history_chars           upstream vs compacted (and the ratio)
    retention[kind]         fraction of planted needles still present in the compacted prompt

The offline run uses the deterministic ``HeuristicExtractor``. It is expected to keep surface
and commitment needles and to *lose buried ones* - that is the honest limitation of a
non-LLM extractor and the reason the runtime default is ``LLMExtractor`` (which reads the
payloads). Pass ``--live`` with ``OPENROUTER_API_KEY`` set to run the LLM extractor instead.

Usage:
    python -m evals.memory_eval [--rounds 5,10,20,40,80] [--live]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from server.services.execution.log_store import ExecutionAgentLogStore
from server.services.execution.memory import AgentMemory, HeuristicExtractor, LLMExtractor, MemoryConfig

from .synthetic import PEOPLE

TOOL_NOISE = (
    '{"id": "18f3a2c9e7b1", "threadId": "18f3a2c9e7b1", "labelIds": ["INBOX", "IMPORTANT"], '
    '"snippet": "Thanks for reaching out, I will take a look and get back to you shortly...", '
    '"payload": {"headers": [{"name": "From", "value": "{person} <{email}>"}, {"name": "Date", '
    '"value": "Tue, 14 Oct 2026 09:12:33 -0400"}], "mimeType": "multipart/alternative"}, '
    '"sizeEstimate": 48213, "historyId": "9988123"}'
)


@dataclass
class Needle:
    kind: str  # surface | buried | commitment
    text: str  # substring that must survive
    round: int


@dataclass
class RoundResult:
    rounds: int
    entries: int
    upstream_chars: int
    compacted_chars: int
    ratio: float
    retention: Dict[str, float]
    compactions: int


def build_log(store: ExecutionAgentLogStore, agent: str, rounds: int, rng: random.Random) -> List[Needle]:
    person = agent.split()[-1]
    email = f"{person.lower()}@example.com"
    needles: List[Needle] = []
    for i in range(rounds):
        kind = rng.random()
        if i % 7 == 3:
            # commitment: a reminder the user asked for
            day = rng.choice(["Friday", "Monday", "Thursday"])
            thing = rng.choice(["crown", "lease renewal", "passport pickup", "car inspection"])
            store.record_request(agent, f"Remind me {day} at 9am about the {thing}")
            store.record_action(agent, "Calling createTrigger with: {'payload': 'reminder'}")
            store.record_tool_response(agent, "createTrigger", "{'id': %d, 'status': 'active'}" % (100 + i))
            store.record_agent_response(agent, f"Reminder set for {day} 9am about the {thing} (trigger {100 + i})")
            needles.append(Needle("commitment", f"{day}", i))
            needles.append(Needle("commitment", thing, i))
            continue

        amount = rng.randint(120, 4800)
        date = f"{rng.choice(['Nov', 'Dec', 'Jan'])} {rng.randint(1, 28)}"
        store.record_request(agent, f"Ask {person} whether the {['invoice', 'deposit', 'refund'][i % 3]} for ${amount} can be settled by {date}")
        store.record_action(agent, f"Calling gmail_create_draft with: {{'to': '{email}', 'subject': 'Re: payment'}}")
        # buried needle: lives only in the tool payload
        code = f"REF-{rng.randint(10000, 99999)}"
        payload = TOOL_NOISE.replace("{person}", person).replace("{email}", email)
        payload += f' "body": "Hi, confirming receipt. Our internal reference is {code}. Regards, {person}"'
        store.record_tool_response(agent, "gmail_fetch_emails", payload)
        if kind < 0.5:
            store.record_agent_response(agent, f"Draft sent to {person} about ${amount} due {date}; awaiting reply")
        else:
            store.record_agent_response(agent, f"{person} confirmed ${amount} will be paid by {date}")
        needles.append(Needle("surface", f"${amount}", i))
        needles.append(Needle("buried", code, i))
    return needles


async def evaluate(rounds_list: List[int], live: bool, out_dir: Path, seed: int = 11) -> List[RoundResult]:
    results: List[RoundResult] = []
    for rounds in rounds_list:
        rng = random.Random(seed + rounds)
        work = out_dir / "tmp" / f"memory_{rounds}"
        if work.exists():
            for f in work.rglob("*"):
                if f.is_file():
                    f.unlink()
        store = ExecutionAgentLogStore(work / "logs")
        agent = f"Email to {rng.choice(PEOPLE)}"
        needles = build_log(store, agent, rounds, rng)

        upstream = store.load_transcript(agent)
        extractor = LLMExtractor() if live else HeuristicExtractor()
        memory = AgentMemory(work / "mem", log_store=store, extractor=extractor,
                             config=MemoryConfig(budget_chars=6000, tail_entries=12))
        # Simulate compaction happening as the log grew: compact every ~10 rounds' worth.
        compactions = 0
        entries = list(store.iter_entries(agent))
        step = 40  # entries per simulated "run" between compaction checks
        for cutoff in range(step, len(entries) + 1, step):
            # Temporarily present a truncated view by re-creating a store up to `cutoff`
            partial = ExecutionAgentLogStore(work / f"partial_{cutoff}")
            for tag, _ts, payload in entries[:cutoff]:
                partial._append(agent, tag, payload)  # noqa: SLF001 - eval replays the journal
            memory._log_store = partial  # noqa: SLF001
            if memory.needs_compaction(agent):
                await memory.compact(agent)
                compactions += 1
        memory._log_store = store  # noqa: SLF001
        if memory.needs_compaction(agent):
            await memory.compact(agent)
            compactions += 1
        compacted = memory.render_history(agent)

        retention: Dict[str, List[bool]] = {}
        for needle in needles:
            retention.setdefault(needle.kind, []).append(needle.text in compacted)
        results.append(
            RoundResult(
                rounds=rounds,
                entries=len(entries),
                upstream_chars=len(upstream),
                compacted_chars=len(compacted),
                ratio=len(compacted) / max(1, len(upstream)),
                retention={k: sum(v) / len(v) for k, v in sorted(retention.items())},
                compactions=compactions,
            )
        )
    (out_dir / ("memory_results_live.json" if live else "memory_results.json")).write_text(
        json.dumps([asdict(r) for r in results], indent=2)
    )
    return results


def print_table(results: List[RoundResult]) -> None:
    print(f"{'rounds':>6} {'entries':>7} {'upstream_tok':>12} {'compacted_tok':>13} {'ratio':>6} {'compactions':>11}  retention")
    for r in results:
        ret = " ".join(f"{k}={v:.2f}" for k, v in r.retention.items())
        print(f"{r.rounds:>6} {r.entries:>7} {r.upstream_chars // 4:>12} {r.compacted_chars // 4:>13} {r.ratio:>6.2f} {r.compactions:>11}  {ret}")


def plot(results: List[RoundResult], out_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 3.8), dpi=150)
    xs = [r.entries for r in results]
    ax.plot(xs, [r.upstream_chars / 4 for r in results], marker="o", ms=5, lw=2, color="#2a78d6",
            label="upstream (full transcript)")
    ax.plot(xs, [r.compacted_chars / 4 for r in results], marker="o", ms=5, lw=2, color="#eb6834",
            label="tiered memory")
    ax.set_title("Execution-agent history in system prompt", fontsize=11)
    ax.set_xlabel("log entries for one agent")
    ax.set_ylabel("est. tokens")
    ax.grid(True, alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    path = out_dir / "memory_tokens_vs_history.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds", default="5,10,20,40,80")
    parser.add_argument("--live", action="store_true", help="use the LLM extractor (needs OPENROUTER_API_KEY)")
    parser.add_argument("--out", default="evals/out")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rounds = [int(x) for x in args.rounds.split(",") if x.strip()]
    results = asyncio.run(evaluate(rounds, args.live, out_dir))
    print_table(results)
    if not args.no_plot:
        print(f"\nchart: {plot(results, out_dir)}")


if __name__ == "__main__":
    main()
