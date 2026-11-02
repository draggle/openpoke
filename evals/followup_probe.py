"""Follow-up probe: bare follow-ups that are NOT ambiguous must be acted on, not asked about.

The session probe's sixth turn checks that "cancel it" after a list of three gets a question.
This probe checks the other side of that rule, so it cannot be satisfied by always asking:

- "cancel it" right after creating one reminder cancels it without a question
- "move it to 1pm" after two single-agent turns goes to the thread touched last, no question
- "cancel the insurance one" after a list names its target and is acted on, no question
- a whole-reminder question after a cancellation lists what is left and not what was cancelled

Runs the shipping configuration (early exit + direct relay) through the in-process app in an
isolated temporary data dir, as evals.session_probe does.

Usage:
    python -m evals.followup_probe --live [--out evals/out]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from evals.session_probe import _agent_names, _settle, report

TURNS: List[str] = [
    "Remind me tomorrow at 9am to call the dentist.",
    "cancel it",
    "Remind me Friday at 8am to renew the car insurance.",
    "Set a reminder for Monday at noon to send Alice the invoice.",
    "move it to 1pm",
    "what reminders do I have?",
    "cancel the insurance one",
]


def _asks(text: str) -> bool:
    lowered = text.lower()
    return "which one" in lowered or "which reminder" in lowered or lowered.rstrip().endswith("?")


def checks(replies_by_turn: List[List[str]]) -> Dict[str, Any]:
    """Each entry is the assistant replies produced for that turn, in order."""
    t = [" ".join(r).lower() for r in replies_by_turn]
    while len(t) < len(TURNS):
        t.append("")
    listing = t[5]
    return {
        "bare_cancel_after_one_reminder_acts": ("cancel" in t[1]) and not _asks(t[1]),
        "move_it_goes_to_last_touched_no_question": ("1pm" in t[4] or "1 pm" in t[4] or "13:00" in t[4]) and "insurance" not in t[4] and not _asks(t[4]),
        "list_shows_only_live_reminders": "insurance" in listing and ("invoice" in listing or "alice" in listing) and "dentist" not in listing,
        "named_cancel_after_list_acts": "insurance" in t[6] and "cancel" in t[6] and not _asks(t[6]),
        "no_internal_ids_shown": all("trigger" not in x and "rrule" not in x for x in t),
    }


async def _run() -> Dict[str, Any]:
    import httpx

    assert os.environ.get("OPENPOKE_DATA_DIR")
    from server.app import app
    from server.services.llm import set_usage_ledger
    from server.services.llm.usage import UsageLedger

    os.environ["OPENPOKE_INTERACTION_EARLY_EXIT"] = "1"
    os.environ["OPENPOKE_DIRECT_RELAY"] = "1"
    ledger = UsageLedger()
    set_usage_ledger(ledger)
    transport = httpx.ASGITransport(app=app)
    replies_by_turn: List[List[str]] = []
    async with httpx.AsyncClient(transport=transport, base_url="http://probe") as client:
        await client.delete("/api/v1/chat/history")
        count = 0
        for turn in TURNS:
            before = len([m for m in (await client.get("/api/v1/chat/history")).json().get("messages", []) if m.get("role") == "assistant"])
            response = await client.post("/api/v1/chat/send", json={"messages": [{"role": "user", "content": turn}]})
            assert response.status_code in (200, 202), response.text
            count = await _settle(client, count)
            history = (await client.get("/api/v1/chat/history")).json().get("messages", [])
            assistant = [m["content"] for m in history if m.get("role") == "assistant"]
            replies_by_turn.append(assistant[before:])
        agents = (await client.get("/api/v1/meta/agents")).json()
        # The temporary data dir is left in place so the agents' tool results can be inspected.
    await asyncio.sleep(2.0)
    set_usage_ledger(None)
    out = report(ledger.totals()["by_role"], early_exit=True, turns=len(TURNS), relay=True)
    out["turns"] = TURNS
    out["replies_by_turn"] = replies_by_turn
    out["checks"] = checks(replies_by_turn)
    out["agents_after"] = _agent_names(agents)
    return out


async def run(out_dir: Path) -> Dict[str, Any]:
    import tempfile

    os.environ["OPENPOKE_DATA_DIR"] = tempfile.mkdtemp(prefix="openpoke-followup-")
    results = await _run()
    results["data_dir"] = os.environ["OPENPOKE_DATA_DIR"]
    (out_dir / "followup_probe.json").write_text(json.dumps(results, indent=2))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--out", default="evals/out")
    args = parser.parse_args()
    if not args.live:
        raise SystemExit("followup_probe only has a live mode: pass --live")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = asyncio.run(run(out_dir))
    print(f"\n{results['total']['calls']} calls, ${results['total']['cost_usd']}")
    for turn, replies in zip(TURNS, results["replies_by_turn"]):
        print(f"  user> {turn}")
        for r in replies:
            print(f"    > {r[:160]!r}")
    print(f"  agents after: {results['agents_after']}")
    print(f"  checks: {results['checks']}")


if __name__ == "__main__":
    main()
