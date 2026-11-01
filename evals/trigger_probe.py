"""Live: a reminder fires in the background and the user is told, through the interaction agent.

The direct relay (``server/agents/execution_agent/relay.py``) is for batches the user asked for;
a batch a trigger fire joined must still take the full path. This probe creates a reminder
through the real app, moves its next fire to now, runs the real scheduler, and checks that
(1) a new assistant message about the reminder reached the conversation, (2) the execution run
was billed to the trigger role, (3) the interaction agent made the call that delivered it, and
(4) nothing internal ("[SUCCESS]", trigger ids) leaked to the user.

Usage:  python -m evals.trigger_probe --live   (needs a key; isolated temporary data dir)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

REQUEST = "Set a reminder for tomorrow at 9am to stretch my legs after the standup."


async def _run() -> Dict[str, Any]:
    import httpx

    from evals.session_probe import _agent_names, _settle
    from server.app import app
    from server.services import get_trigger_scheduler, get_trigger_service
    from server.services.llm import set_usage_ledger
    from server.services.llm.usage import UsageLedger

    ledger = UsageLedger()
    set_usage_ledger(ledger)
    out: Dict[str, Any] = {"request": REQUEST, "direct_relay": os.environ.get("OPENPOKE_DIRECT_RELAY", "1")}
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://probe") as client:
        await client.delete("/api/v1/chat/history")
        response = await client.post("/api/v1/chat/send", json={"messages": [{"role": "user", "content": REQUEST}]})
        assert response.status_code in (200, 202), response.text
        count = await _settle(client, 0)
        before = ledger.totals()["by_role"]
        out["calls_before_fire"] = {role: row["calls"] for role, row in before.items()}
        agents = _agent_names((await client.get("/api/v1/meta/agents")).json())
        out["agents"] = agents

        service = get_trigger_service()
        triggers: List = []
        for name in agents:
            triggers += service.list_triggers(agent_name=name)
        out["triggers_created"] = [{"agent": t.agent_name, "payload": t.payload, "next": t.next_trigger} for t in triggers]
        if not triggers:
            out["error"] = "no trigger was created"
            await client.delete("/api/v1/chat/history")
            set_usage_ledger(None)
            return out
        target = triggers[0]
        soon = (datetime.now(timezone.utc) + timedelta(seconds=2)).isoformat()
        service.update_trigger(target.id, agent_name=target.agent_name, start_time=soon, timezone_name="UTC")

        scheduler = get_trigger_scheduler()
        scheduler._poll_interval = 3.0
        await scheduler.start()
        fired_at = time.monotonic()
        count_after = await _settle(client, count, timeout_s=180.0, quiet_s=10.0)
        await scheduler.stop()
        out["seconds_to_notify"] = round(time.monotonic() - fired_at, 1)

        history = (await client.get("/api/v1/chat/history")).json().get("messages", [])
        new_messages = history[count:]
        new_replies = [m["content"] for m in new_messages if m.get("role") == "assistant"]
        out["replies_after_fire"] = new_replies
        after = ledger.totals()["by_role"]
        out["calls_after_fire"] = {role: row["calls"] for role, row in after.items()}
        text = "\n".join(new_replies).lower()
        out["checks"] = {
            "user_was_notified": bool(new_replies),
            "notification_names_the_reminder": "stretch" in text,
            "billed_to_trigger_role": after.get("execution_trigger", {}).get("calls", 0) > before.get("execution_trigger", {}).get("calls", 0),
            "interaction_agent_delivered_it": after.get("interaction", {}).get("calls", 0) > before.get("interaction", {}).get("calls", 0),
            "nothing_internal_leaked": "[success]" not in text and "trigger" not in text,
        }
        out["cost_usd"] = round(ledger.totals()["total"]["cost_usd"], 4) if "total" in ledger.totals() else None
        await client.delete("/api/v1/chat/history")
    await asyncio.sleep(2.0)
    set_usage_ledger(None)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--out", default="evals/out")
    args = parser.parse_args()
    if not args.live:
        print("trigger_probe is live-only: pass --live with a key in the environment")
        return
    os.environ["OPENPOKE_DATA_DIR"] = tempfile.mkdtemp(prefix="openpoke-trigger-probe-")
    result = asyncio.run(_run())
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "trigger_probe.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
