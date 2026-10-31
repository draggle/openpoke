"""Memory eval: how much prompt does compaction save, and what does it lose?

We synthesise a long-lived execution agent (one person, many rounds of email back-and-forth)
and plant *needles* - facts the agent should still know later:

- ``surface`` needles appear in agent requests/responses (where the LLM writes its summary)
- ``buried``  needles appear only inside raw tool payloads (an email body returned by Gmail)
- ``commitment`` needles are reminder requests ("remind me Friday ... crown")
- ``changed`` needles are amounts that were later corrected: the NEW value must be present and
  the OLD value must not be presented as current (a compactor that keeps the stale number, or
  both, scores badly; presence-only metrics cannot see this)
- ``constraint`` needles are "do not ..." rules the user stated, which compaction studies show
  summaries drop first (arXiv 2610.31778)

Then, for increasing history lengths, we compare the upstream prompt (full transcript) against
the compacted prompt, measuring:

    history_chars           upstream vs compacted (and the ratio)
    retention[kind]         fraction of planted needles still present in the compacted prompt
    retention[changed_current] / [stale]   new value present / old value shown as current
    retention[recoverable]  buried needles absent from the prompt but found by search_history

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
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from server.services.execution.log_store import ExecutionAgentLogStore
from server.services.execution.memory import AgentMemory, HeuristicExtractor, LLMExtractor, MemoryConfig

from .synthetic import PEOPLE

def gmail_fetch_result(person: str, email: str, body: str) -> str:
    """A realistic GMAIL_FETCH_EMAILS result (valid JSON) with the needle deep in the body."""
    import json as _json

    boilerplate = ("Thanks for reaching out. Per our earlier thread, here is the status update you asked "
                   "for. Please find the details below and let me know if anything is unclear. ") * 3
    return _json.dumps({"status": "success", "emails": [{
        "id": "18f3a2c9e7b1", "thread_id": "18f3a2c9e7b1", "subject": "Re: payment",
        "sender": f"{person} <{email}>", "recipient": "me@example.com",
        "timestamp": "2026-10-14T09:12:33-04:00", "label_ids": ["INBOX", "IMPORTANT"],
        "clean_text": boilerplate + body, "has_attachments": False, "attachment_count": 0,
        "attachment_filenames": [],
    }], "message_ids": ["18f3a2c9e7b1"]})


def record_tool_result(store: ExecutionAgentLogStore, agent: str, tool: str, raw: str) -> None:
    """Write a tool result the way the runtime does: compacted in the log, raw in the sidecar."""
    from server.services.execution.payloads import compact_tool_payload

    store.record_tool_response(agent, tool, compact_tool_payload(raw, limit=500), raw=raw)


@dataclass
class Needle:
    kind: str  # surface | buried | commitment | changed | constraint
    text: str  # substring that must survive
    round: int
    stale: str = ""  # for ``changed``: the superseded value that must not be presented as current


def score_retention(needles: List[Needle], compacted: str, search) -> Dict[str, float]:
    """Score what survived. ``search(query)`` returns search_history-style matches."""
    # Only a compacted prompt has a memory section that *asserts* values; a raw transcript
    # merely records history, so old values in it are not stale memory.
    head = compacted.split("# Recent History")[0] if "# Recent History" in compacted else ""
    # Only the Facts block asserts what is *current*. The changed-facts block and the timeline
    # legitimately narrate old values ("$700, was $500"), so they are excluded.
    facts_block = ""
    if "## Facts" in head:
        facts_block = head.split("## Facts", 1)[1]
        for stop in ("## Changed facts", "## Open threads", "## Timeline"):
            facts_block = facts_block.split(stop, 1)[0]
    # Narrated corrections inside a fact ("$700 (corrected from $500)", "$900, was $850") are
    # history the model can use, not a stale assertion.
    head_current = re.sub(
        r"\((?:corrected|changed|updated|revised)[^)]*\)|,?\s*(?:was|previously|formerly|not|up from|down from)\s+\$?[\d,.]+",
        "", facts_block, flags=re.I,
    )
    present: Dict[str, List[bool]] = {}
    stale: List[bool] = []
    recoverable: List[bool] = []
    for needle in needles:
        if needle.kind == "changed":
            present.setdefault("changed_current", []).append(needle.text in compacted)
            stale.append(needle.stale in head_current)
            continue
        hit = needle.text in compacted
        present.setdefault(needle.kind, []).append(hit)
        if needle.kind == "buried" and not hit:
            matches = search(needle.text)
            recoverable.append(any(needle.text in (m.get("text") or "") for m in matches))
        elif needle.kind == "buried":
            recoverable.append(True)
    out = {k: sum(v) / len(v) for k, v in sorted(present.items())}
    if stale:
        out["stale"] = sum(stale) / len(stale)
    if recoverable:
        out["recoverable"] = sum(recoverable) / len(recoverable)
    return out


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
    amounts: List[Tuple[int, str]] = []  # (amount, item) for later corrections
    for i in range(rounds):
        kind = rng.random()
        if i % 9 == 2:
            # constraint: a rule the user stated; must survive verbatim
            rule = rng.choice([
                f"Do not mention the deposit dispute to {person}.",
                f"Never cc {person}'s manager on these emails.",
                f"Always send {person} the PDF version, not a link.",
            ])
            store.record_request(agent, f"Follow up with {person} on the open items. {rule}")
            store.record_agent_response(agent, f"Followed up with {person}; noted the rule.")
            needles.append(Needle("constraint", rule, i))
            continue
        if i % 5 == 4 and amounts:
            # changed: a correction to an earlier amount
            old_amount, item = rng.choice(amounts)
            new_amount = old_amount + rng.randint(50, 400)
            store.record_request(agent, f"{person} says the {item} was wrong: it is ${new_amount}, not ${old_amount}. Update our records.")
            store.record_agent_response(agent, f"Updated: the {item} is ${new_amount} (was ${old_amount}).")
            needles.append(Needle("changed", f"${new_amount}", i, stale=f"${old_amount}"))
            needles = [n for n in needles if not (n.kind == "surface" and n.text == f"${old_amount}")]
            amounts = [(a, it) for a, it in amounts if a != old_amount]
            continue
        if i % 7 == 3:
            # commitment: a reminder the user asked for
            day = rng.choice(["Friday", "Monday", "Thursday"])
            thing = rng.choice(["crown", "lease renewal", "passport pickup", "car inspection"])
            store.record_request(agent, f"Remind me {day} at 9am about the {thing}")
            store.record_action(agent, "Calling createTrigger with: {'payload': 'reminder'}")
            record_tool_result(store, agent, "createTrigger", '{"id": %d, "status": "active"}' % (100 + i))
            store.record_agent_response(agent, f"Reminder set for {day} 9am about the {thing} (trigger {100 + i})")
            needles.append(Needle("commitment", f"{day}", i))
            needles.append(Needle("commitment", thing, i))
            continue

        amount = rng.randint(120, 4800)
        item = ["invoice", "deposit", "refund"][i % 3]
        amounts.append((amount, item))
        date = f"{rng.choice(['Nov', 'Dec', 'Jan'])} {rng.randint(1, 28)}"
        store.record_request(agent, f"Ask {person} whether the {item} for ${amount} can be settled by {date}")
        store.record_action(agent, f"Calling gmail_create_draft with: {{'to': '{email}', 'subject': 'Re: payment'}}")
        # buried needle: lives only inside the tool payload's email body
        code = f"REF-{rng.randint(10000, 99999)}"
        raw = gmail_fetch_result(person, email, f"Hi, confirming receipt. Our internal reference is {code}. Regards, {person}")
        record_tool_result(store, agent, "gmail_fetch_emails", raw)
        if kind < 0.5:
            store.record_agent_response(agent, f"Draft sent to {person} about ${amount} due {date}; awaiting reply")
        else:
            store.record_agent_response(agent, f"{person} confirmed ${amount} will be paid by {date}")
        needles.append(Needle("surface", f"${amount}", i))
        needles.append(Needle("buried", code, i))
    return needles


async def evaluate(rounds_list: List[int], live: bool, out_dir: Path, seed: int = 11,
                   budget_chars: int = 12_000) -> List[RoundResult]:
    from server.agents.execution_agent.tools.history import search_history

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
                             config=MemoryConfig(budget_chars=budget_chars, tail_entries=12))
        # The runtime records commitments at dispatch time; replay that for every request.
        for tag, _ts, payload in store.iter_entries(agent):
            if tag == "agent_request":
                memory.note_request(agent, payload)
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
        retention = score_retention(
            needles, compacted, search=lambda q: search_history(agent, q, limit=5, log_store=store)["matches"]
        )
        results.append(
            RoundResult(
                rounds=rounds,
                entries=len(entries),
                upstream_chars=len(upstream),
                compacted_chars=len(compacted),
                ratio=len(compacted) / max(1, len(upstream)),
                retention=retention,
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
    parser.add_argument("--budget", type=int, default=12_000, help="compaction budget in chars (runtime default 12000)")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rounds = [int(x) for x in args.rounds.split(",") if x.strip()]
    results = asyncio.run(evaluate(rounds, args.live, out_dir, budget_chars=args.budget))
    print_table(results)
    if not args.no_plot:
        print(f"\nchart: {plot(results, out_dir)}")


if __name__ == "__main__":
    main()
