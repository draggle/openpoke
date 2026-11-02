"""Hand-session runner: type a script of user turns into the real app and keep the transcript.

The scripted equivalent of sitting at the chat UI. Boots the app in-process in a temporary data
dir (nothing of a working installation is read or written), sends each turn, waits the way a
person would (until the assistant has answered and the agents have settled), and writes the
full transcript plus the backend log for review. Options:

  --scheduler   run the real trigger scheduler (3 s poll) so "remind me in 1 minute" fires
  --fake-gmail  install an in-memory inbox and outbound stubs in every execution agent, so the
                email paths (search, draft, reply, forward, the send gate) run without Composio;
                every outbound call is recorded in the output

A script is JSON: {"name": ..., "turns": [{"say": "...", "wait_s": 0}, ...]}. ``wait_s`` adds a
pause after that turn, e.g. 75 for a one-minute reminder to fire.

Usage:
    python -m evals.hand_session --live --script evals/hand_scripts/reminders.json [--scheduler] [--fake-gmail]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from evals.session_probe import _agent_names, _settle

FAKE_INBOX: List[Dict[str, Any]] = [
    {"id": "m-101", "thread_id": "t-alice", "sender": "alice@example.com", "subject": "Lunch invoice",
     "date": "2026-10-30", "body": "Hi! Attaching the invoice for last week's lunch, $42.50. Can you settle it by Friday? Thanks, Alice"},
    {"id": "m-102", "thread_id": "t-bob", "sender": "bob@example.com", "subject": "Flight LO 282",
     "date": "2026-10-31", "body": "Your flight to Krakow is confirmed: LO 282, Friday 7am. Bob"},
    {"id": "m-103", "thread_id": "t-news", "sender": "deals@shopmail.example", "subject": "50% off everything",
     "date": "2026-11-01", "body": "Big sale this weekend only. Unsubscribe at the bottom."},
    {"id": "m-104", "thread_id": "t-priya", "sender": "priya@example.com", "subject": "Reference letter",
     "date": "2026-11-01", "body": "Could you write me a reference letter for the Northwind role by the 10th? Priya"},
]

OUTBOUND: List[Dict[str, Any]] = []
DRAFTS: Dict[str, Dict[str, Any]] = {}


def _install_fake_gmail() -> None:
    from server.agents.execution_agent import runtime as rt

    original = rt.get_tool_registry

    def search(**kw):
        q = " ".join(str(v) for v in kw.values() if isinstance(v, str)).lower()
        hits = [m for m in FAKE_INBOX if not q or any(w in (m["sender"] + " " + m["subject"] + " " + m["body"]).lower() for w in q.split() if len(w) > 2)]
        return {"emails": hits or FAKE_INBOX, "count": len(hits or FAKE_INBOX)}

    def create_draft(**kw):
        draft_id = f"d-{len(DRAFTS) + 1}"
        DRAFTS[draft_id] = kw
        return {"status": "draft_created", "draft_id": draft_id, **kw}

    def outbound(tool):
        def call(**kw):
            OUTBOUND.append({"tool": tool, "args": kw, "at": time.strftime("%H:%M:%S")})
            return {"status": "sent", "id": f"sent-{len(OUTBOUND)}"}
        return call

    def patched(agent_name: str):
        registry = original(agent_name=agent_name)
        registry["task_email_search"] = search
        registry["gmail_create_draft"] = create_draft
        registry["gmail_list_drafts"] = lambda **kw: {"drafts": [{"id": k, **v} for k, v in DRAFTS.items()]}
        registry["gmail_delete_draft"] = lambda **kw: {"status": "deleted"}
        for tool in ("gmail_execute_draft", "gmail_forward_email", "gmail_reply_to_thread"):
            registry[tool] = outbound(tool)
        registry["gmail_get_contacts"] = lambda **kw: {"contacts": [{"name": "Alice", "email": "alice@example.com"}, {"name": "Bob", "email": "bob@example.com"}, {"name": "Priya", "email": "priya@example.com"}]}
        registry["gmail_search_people"] = registry["gmail_get_contacts"]
        registry["gmail_get_people"] = registry["gmail_get_contacts"]
        return registry

    rt.get_tool_registry = patched


async def _run(script: Dict[str, Any], *, scheduler: bool, fake_gmail: bool) -> Dict[str, Any]:
    import httpx

    assert os.environ.get("OPENPOKE_DATA_DIR")
    from server.app import app
    from server.services.llm import set_usage_ledger
    from server.services.llm.usage import UsageLedger

    if fake_gmail:
        _install_fake_gmail()
    ledger = UsageLedger()
    set_usage_ledger(ledger)
    sched = None
    if scheduler:
        from server.services import get_trigger_scheduler

        sched = get_trigger_scheduler()
        sched._poll_interval = 3.0
        await sched.start()
    transport = httpx.ASGITransport(app=app)
    turns_out: List[Dict[str, Any]] = []
    async with httpx.AsyncClient(transport=transport, base_url="http://hand") as client:
        await client.delete("/api/v1/chat/history")
        count = 0
        for turn in script["turns"]:
            before = (await client.get("/api/v1/chat/history")).json().get("messages", [])
            n_before = len([m for m in before if m.get("role") == "assistant"])
            started = time.monotonic()
            response = await client.post("/api/v1/chat/send", json={"messages": [{"role": "user", "content": turn["say"]}]})
            assert response.status_code in (200, 202), response.text
            count = await _settle(client, count)
            extra = float(turn.get("wait_s", 0) or 0)
            if extra:
                await asyncio.sleep(extra)
                count = await _settle(client, count, quiet_s=2.0)
            history = (await client.get("/api/v1/chat/history")).json().get("messages", [])
            assistant = [m["content"] for m in history if m.get("role") == "assistant"]
            turns_out.append({"say": turn["say"], "replies": assistant[n_before:], "seconds": round(time.monotonic() - started, 1)})
        agents = (await client.get("/api/v1/meta/agents")).json()
    if sched is not None:
        await sched.stop()
    await asyncio.sleep(2.0)
    set_usage_ledger(None)
    total = ledger.totals().get("total", {})
    return {
        "name": script["name"],
        "turns": turns_out,
        "agents_after": _agent_names(agents),
        "outbound_calls": list(OUTBOUND),
        "drafts": dict(DRAFTS),
        "calls": int(total.get("calls", 0)),
        "cost_usd": round(float(total.get("cost_usd", 0.0)), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--script", required=True)
    parser.add_argument("--out", default="evals/out")
    parser.add_argument("--scheduler", action="store_true")
    parser.add_argument("--fake-gmail", action="store_true")
    args = parser.parse_args()
    if not args.live:
        raise SystemExit("hand_session only has a live mode: pass --live")
    import tempfile

    os.environ["OPENPOKE_DATA_DIR"] = tempfile.mkdtemp(prefix="openpoke-hand-")
    script = json.loads(Path(args.script).read_text())
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = asyncio.run(_run(script, scheduler=args.scheduler, fake_gmail=args.fake_gmail))
    (out_dir / f"hand_{script['name']}.json").write_text(json.dumps(result, indent=2))
    print(f"\n# {script['name']}: {result['calls']} calls, ${result['cost_usd']}")
    for t in result["turns"]:
        print(f"\nuser> {t['say']}   ({t['seconds']}s)")
        for r in t["replies"]:
            print(f"  poke> {r}")
    print(f"\nagents after: {result['agents_after']}")
    if result["outbound_calls"]:
        print("outbound calls:")
        for c in result["outbound_calls"]:
            print(f"  {c['at']} {c['tool']} {json.dumps(c['args'])[:160]}")


if __name__ == "__main__":
    main()
