"""Session probe: LLM calls, cache share and cost by role for a real three-turn chat session.

Boots the FastAPI app in-process (no network server, no background scheduler) and drives the
real chat endpoint through the full path: Interaction Agent -> execution agent -> result back
to the Interaction Agent -> reply. The usage ledger is swapped for a fresh one per run, so the
numbers are exactly this session's. Runs once with the early-exit loop (round three) and once
with it off (round two / upstream loop) so the saving is measured, not modelled.

The session is the one used for the round-two cache probe: create a reminder with a standing
rule, move it, ask what is set. Composio need not be connected; only reminder tools run.

State isolation: the probe sets ``OPENPOKE_DATA_DIR`` to a fresh temporary directory before the
app is imported, so a working installation's conversation, agents, memory and reminders under
``server/data`` are never read or cleared. (Round two's probe cleared the real state.)

Usage:
    python -m evals.session_probe --live [--out evals/out]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

TURNS: List[str] = [
    "Set a reminder for tomorrow at 9am to send Alice the lunch invoice. Important: never cc her "
    "manager on any emails related to this.",
    "Change the reminder time from 9am to 10am tomorrow.",
    "What reminders do I have set?",
    # Two requests in one message: the early-exit loop must not drop the second dispatch.
    "Also remind me Friday at 8am to call the dentist, and set a separate reminder for Monday at "
    "noon to renew the car insurance.",
]


def report(by_role: Dict[str, Dict[str, Any]], *, early_exit: bool, turns: int, relay: bool = False) -> Dict[str, Any]:
    """Summarise ledger totals-by-role for one session."""
    roles: Dict[str, Dict[str, Any]] = {}
    total = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0,
             "cache_write_tokens": 0, "cost_usd": 0.0}
    for role, row in sorted(by_role.items()):
        prompt = int(row.get("prompt_tokens", 0))
        cached = int(row.get("cached_tokens", 0))
        roles[role] = {
            "calls": int(row.get("calls", 0)),
            "calls_per_turn": round(int(row.get("calls", 0)) / max(1, turns), 2),
            "prompt_tokens": prompt,
            "completion_tokens": int(row.get("completion_tokens", 0)),
            "cached_tokens": cached,
            "cached_share": round(cached / prompt, 2) if prompt else 0.0,
            "cost_usd": round(float(row.get("cost_usd", 0.0)), 4),
        }
        for key in total:
            total[key] += row.get(key, 0) or 0
    total["cost_usd"] = round(total["cost_usd"], 4)
    return {
        "early_exit": early_exit,
        "turns": turns,
        "direct_relay": relay,
        "calls_per_turn": round(total["calls"] / max(1, turns), 2),
        "cached_share": round(total["cached_tokens"] / total["prompt_tokens"], 2) if total["prompt_tokens"] else 0.0,
        "total": total,
        "by_role": roles,
    }


def _work_in_flight() -> bool:
    """An execution agent still running, or an interaction turn still delivering its result."""
    from server.agents.interaction_agent.tools import _EXECUTION_BATCH_MANAGER

    if _EXECUTION_BATCH_MANAGER.get_pending_executions():
        return True
    return any(t.get_name() == "interaction-agent-message" and not t.done() for t in asyncio.all_tasks())


async def _settle(client, previous: int, timeout_s: float = 180.0, quiet_s: float = 4.0) -> int:
    """Wait until the chat history has grown past ``previous``, no execution or result delivery
    is in flight, and the history has stayed still for ``quiet_s``."""
    deadline = time.monotonic() + timeout_s
    last_count, last_change = previous, time.monotonic()
    while time.monotonic() < deadline:
        await asyncio.sleep(1.0)
        response = await client.get("/api/v1/chat/history")
        count = len(response.json().get("messages", []))
        if count != last_count:
            last_count, last_change = count, time.monotonic()
        elif count > previous and not _work_in_flight() and time.monotonic() - last_change >= quiet_s:
            break
    return last_count


async def _run_session(early_exit: bool, relay: bool = False) -> Dict[str, Any]:
    import httpx

    assert os.environ.get("OPENPOKE_DATA_DIR"), "run() sets a temporary data dir before importing the app"
    from server.app import app
    from server.services.llm import set_usage_ledger
    from server.services.llm.usage import UsageLedger

    os.environ["OPENPOKE_INTERACTION_EARLY_EXIT"] = "1" if early_exit else "0"
    os.environ["OPENPOKE_DIRECT_RELAY"] = "1" if relay else "0"
    ledger = UsageLedger()
    set_usage_ledger(ledger)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://probe") as client:
        await client.delete("/api/v1/chat/history")
        count = 0
        for turn in TURNS:
            response = await client.post("/api/v1/chat/send", json={"messages": [{"role": "user", "content": turn}]})
            assert response.status_code in (200, 202), response.text
            count = await _settle(client, count)
        history = (await client.get("/api/v1/chat/history")).json().get("messages", [])
        replies = [m["content"] for m in history if m.get("role") == "assistant"]
        agents = (await client.get("/api/v1/meta/agents")).json()
        await client.delete("/api/v1/chat/history")
    # Let fire-and-forget tasks (background compaction, agent result relays) drain.
    await asyncio.sleep(2.0)
    set_usage_ledger(None)
    out = report(ledger.totals()["by_role"], early_exit=early_exit, turns=len(TURNS), relay=relay)
    out["replies"] = replies
    out["checks"] = _checks(replies)
    out["agents_after"] = _agent_names(agents)
    return out


def _checks(replies: List[str]) -> Dict[str, bool]:
    """Did the user get told what they needed? Cheap substring checks on the assistant replies."""
    text = "\n".join(replies).lower()
    return {
        "moved_to_10am_confirmed": "10" in text,
        "reminder_list_names_the_invoice": "invoice" in text,
        "dentist_and_insurance_confirmed": "dentist" in text and "insurance" in text,
        "no_internal_ids_shown": "trigger" not in text and "rrule" not in text,
        "replies": len(replies),
    }


def _agent_names(payload: Any) -> List[str]:
    items = payload.get("agents", payload) if isinstance(payload, dict) else payload
    names = []
    for item in items if isinstance(items, list) else []:
        name = item.get("name") if isinstance(item, dict) else None
        if name:
            names.append(name)
    return names


async def run(out_dir: Path) -> Dict[str, Any]:
    import tempfile

    os.environ["OPENPOKE_DATA_DIR"] = tempfile.mkdtemp(prefix="openpoke-probe-")
    results = {"what": "four-turn reminder session through the in-process app (isolated data dir); ledger reset per run",
               "turns": TURNS, "with_early_exit": await _run_session(True),
               "without_early_exit": await _run_session(False),
               "with_relay": await _run_session(True, relay=True)}
    a, b, c = results["with_early_exit"], results["without_early_exit"], results["with_relay"]
    results["saving"] = {
        "calls": b["total"]["calls"] - a["total"]["calls"],
        "cost_share": round(1 - a["total"]["cost_usd"] / b["total"]["cost_usd"], 3) if b["total"]["cost_usd"] else 0.0,
        "relay_calls_vs_early_exit": a["total"]["calls"] - c["total"]["calls"],
        "relay_cost_share_vs_early_exit": round(1 - c["total"]["cost_usd"] / a["total"]["cost_usd"], 3) if a["total"]["cost_usd"] else 0.0,
    }
    (out_dir / "session_probe.json").write_text(json.dumps(results, indent=2))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--out", default="evals/out")
    args = parser.parse_args()
    if not args.live:
        raise SystemExit("session_probe only has a live mode: pass --live")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = asyncio.run(run(out_dir))
    for label in ("with_early_exit", "without_early_exit", "with_relay"):
        r = results[label]
        print(f"\n{label}: {r['total']['calls']} calls ({r['calls_per_turn']}/turn), "
              f"{r['total']['prompt_tokens']} prompt tok, {r['cached_share']:.0%} cached, ${r['total']['cost_usd']}")
        for role, row in r["by_role"].items():
            print(f"  {role:<12} calls {row['calls']:<3} cached {row['cached_share']:.0%}  ${row['cost_usd']}")
        print(f"  agents after: {r.get('agents_after')}")
        print(f"  checks: {r.get('checks')}")
        for reply in r.get("replies", []):
            print(f"    > {reply[:160]!r}")
    print(f"\nsaving: {results['saving']}")


if __name__ == "__main__":
    main()
