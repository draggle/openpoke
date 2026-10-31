"""Tool-payload compaction, classifier concurrency and trigger-tier routing."""

from __future__ import annotations

import asyncio
import json

from server.services.execution.payloads import compact_tool_payload
from server.services.llm.router import ROLE_EXECUTION, ROLE_EXECUTION_TRIGGER, ModelRouter


def test_compact_keeps_body_drops_noise():
    raw = json.dumps({
        "id": "18f3a2c9", "threadId": "18f3a2c9", "historyId": "9988", "sizeEstimate": 48213,
        "labelIds": ["INBOX", "IMPORTANT"],
        "payload": {"headers": [{"name": "From", "value": "x"}] * 20, "mimeType": "multipart"},
        "subject": "Re: invoice", "from": "priya@studio.design",
        "snippet": "Confirming receipt. Reference REF-55512. " + "filler " * 80,
    })
    assert len(raw) > 500
    out = compact_tool_payload(raw, limit=500)
    assert len(out) <= 500
    assert "REF-55512" in out and "priya@studio.design" in out
    assert "historyId" not in out and "headers" not in out


def test_compact_short_and_non_json_passthrough():
    assert compact_tool_payload("ok", limit=500) == "ok"
    long_text = "x" * 1000
    assert compact_tool_payload(long_text, limit=100) == "x" * 100


def test_trigger_role_has_its_own_tier():
    r = ModelRouter(strong_model="s", cheap_model="c", tiers={ROLE_EXECUTION_TRIGGER: "cheap"})
    assert r.model_for(ROLE_EXECUTION) == "s"
    assert r.model_for(ROLE_EXECUTION_TRIGGER) == "c"
    assert ModelRouter(strong_model="s", cheap_model="c").model_for(ROLE_EXECUTION_TRIGGER) == "s"


def test_execution_runtime_uses_trigger_role(monkeypatch):
    from server.agents.execution_agent.runtime import ExecutionAgentRuntime
    assert ExecutionAgentRuntime("a", source="trigger").role == ROLE_EXECUTION_TRIGGER
    assert ExecutionAgentRuntime("a").role == ROLE_EXECUTION


async def test_watcher_classifies_concurrently(monkeypatch, tmp_path):
    """Drive the real poll loop: 8 unseen emails, classifier 50ms each, concurrency 4 ->
    at most 4 in flight at once, finished in ~2 rounds, every email marked seen, summaries sent."""
    from datetime import datetime, timezone
    from server.services.gmail import importance_watcher as w
    from server.services.gmail.seen_store import GmailSeenStore
    from evals.email_eval import load_fixtures, to_processed

    now = datetime.now(timezone.utc)
    emails = [to_processed(r) for r in load_fixtures() if r["id"].startswith("i")][:8]
    emails = [type(e)(**{**e.__dict__, "timestamp": now}) for e in emails]

    in_flight = {"now": 0, "max": 0}

    async def fake_classify(email):
        in_flight["now"] += 1
        in_flight["max"] = max(in_flight["max"], in_flight["now"])
        await asyncio.sleep(0.05)
        in_flight["now"] -= 1
        return f"summary for {email.id}"

    sent = []

    async def fake_dispatch(self, summary):
        sent.append(summary)

    monkeypatch.setattr(w, "classify_email_importance", fake_classify)
    monkeypatch.setattr(w, "prefilter", lambda e: type("D", (), {"skip": False, "reason": ""})())
    monkeypatch.setattr(w, "get_active_gmail_user_id", lambda: "user-1")
    monkeypatch.setattr(w, "execute_gmail_tool", lambda *a, **k: {"raw": True})
    monkeypatch.setattr(w, "parse_gmail_fetch_response", lambda raw, *, query, cleaner: (emails, None))
    monkeypatch.setattr(w.ImportantEmailWatcher, "_dispatch_summary", fake_dispatch)
    monkeypatch.setenv("OPENPOKE_CLASSIFIER_CONCURRENCY", "4")

    watcher = w.ImportantEmailWatcher(poll_interval_seconds=60, seen_store=GmailSeenStore(tmp_path / "seen.json"))
    watcher._has_seeded_initial_snapshot = True  # skip the warm-up poll that only marks mail seen

    t0 = asyncio.get_event_loop().time()
    await watcher._poll_once()
    elapsed = asyncio.get_event_loop().time() - t0

    assert in_flight["max"] == 4
    assert elapsed < 0.05 * 8 * 0.6, elapsed  # clearly faster than sequential
    assert len(sent) == 8
    assert all(watcher._seen_store.is_seen(e.id) for e in emails)


async def test_execution_runtime_returns_before_compaction(monkeypatch, tmp_path):
    """Compaction is scheduled after the result is produced, never awaited on the request path."""
    from server.agents.execution_agent import agent as agent_module
    from server.agents.execution_agent import runtime as runtime_module
    from server.openrouter_client import client as oc
    from server.services.execution.log_store import ExecutionAgentLogStore

    store = ExecutionAgentLogStore(tmp_path / "logs")
    monkeypatch.setattr(agent_module, "get_execution_agent_logs", lambda: store)

    flag = {"compacted": False}

    class SlowMemory:
        config = type("C", (), {"enabled": True})()

        def render_history(self, name):
            return store.load_transcript(name)

        def needs_compaction(self, name, state=None):
            return True

        async def compact(self, name, *, force=False):
            await asyncio.sleep(0.2)
            flag["compacted"] = True

        def schedule_compaction(self, name):
            if self.needs_compaction(name):
                return asyncio.get_running_loop().create_task(self.compact(name))
            return None

    memory = SlowMemory()
    monkeypatch.setattr(agent_module, "get_agent_memory", lambda: memory)
    monkeypatch.setattr(runtime_module, "get_agent_memory", lambda: memory)

    async def fake_llm(**kw):
        return {"choices": [{"message": {"role": "assistant", "content": "done"}}]}

    monkeypatch.setattr(oc, "request_chat_completion", fake_llm)

    runtime = runtime_module.ExecutionAgentRuntime("Agent X")
    t0 = asyncio.get_event_loop().time()
    result = await runtime.execute("do the thing")
    assert result.success and result.response == "done"
    assert asyncio.get_event_loop().time() - t0 < 0.15
    assert flag["compacted"] is False
    await asyncio.sleep(0.3)
    assert flag["compacted"] is True


def _search_result(n: int) -> str:
    emails = [{
        "id": f"18f3a{i}b2c4d5e6f7", "thread_id": f"18f3a{i}00000000a1", "query": "from:alice",
        "subject": f"Re: Lunch Thursday? ({i})", "sender": "Alice Smith <alice@example.com>",
        "recipient": "me@example.com", "timestamp": "2026-10-28T10:12:00Z",
        "label_ids": ["INBOX", "IMPORTANT"],
        "clean_text": "Hi! Thursday works for me. Shall we do 12:30 at the usual place near the office? "
                      "I have a meeting until 12 so slightly later is better. Let me know and I will book "
                      "a table. Also, did you get the invoice I forwarded last week?",
        "has_attachments": False, "attachment_count": 0, "attachment_filenames": [],
    } for i in range(n)]
    return json.dumps({"status": "success", "emails": emails, "message_ids": [e["id"] for e in emails]})


def test_second_email_thread_id_survives():
    out = compact_tool_payload(_search_result(2), limit=500)
    assert len(out) <= 500
    assert "18f3a000000000a1" in out and "18f3a100000000a1" in out, out
    assert "Lunch Thursday? (0)" in out and "Lunch Thursday? (1)" in out


def test_camel_thread_id_kept():
    raw = json.dumps({"successful": True, "data": {"response_data": {
        "id": "r-123", "message": {"id": "18f3abc", "threadId": "18f3thread", "labelIds": ["DRAFT"],
                                   "snippet": "x" * 600}}}, "error": None})
    out = compact_tool_payload(raw, limit=500)
    assert "18f3thread" in out


def test_execution_static_block_is_cacheable_and_history_is_not(monkeypatch, tmp_path):
    from server.agents.execution_agent import agent as agent_module
    from server.services.execution.log_store import ExecutionAgentLogStore

    store = ExecutionAgentLogStore(tmp_path / "logs")
    monkeypatch.setattr(agent_module, "get_execution_agent_logs", lambda: store)

    class Mem:
        def render_history(self, name):
            return store.load_transcript(name)

    monkeypatch.setattr(agent_module, "get_agent_memory", lambda: Mem())
    agent = agent_module.ExecutionAgent("Email to Priya")
    blocks = agent.build_system_blocks()
    assert len(blocks) == 1 and blocks[0]["cacheable"] is True and "Email to Priya" in blocks[0]["text"]

    store.record_request("Email to Priya", "chase the invoice")
    blocks = agent.build_system_blocks()
    assert len(blocks) == 2
    assert blocks[0]["cacheable"] is True and "chase the invoice" not in blocks[0]["text"]
    assert blocks[1].get("cacheable", False) is False and "chase the invoice" in blocks[1]["text"]
    # the legacy string form is the blocks joined
    assert agent.build_system_prompt_with_history() == blocks[0]["text"] + "\n\n" + blocks[1]["text"]


def test_trigger_list_keeps_its_fields():
    raw = json.dumps({"triggers": [
        {"id": i, "status": "active", "payload": f"remind me about thing {i} " * 8,
         "next_trigger": f"2026-11-0{i+1}T09:00:00", "recurrence_rule": "FREQ=WEEKLY", "created_at": "2026-10-31"}
        for i in range(3)]})
    assert len(raw) > 500
    out = compact_tool_payload(raw, limit=800)
    for i in range(3):
        assert f"2026-11-0{i+1}T09:00:00" in out
    assert "FREQ=WEEKLY" in out and "remind me about thing 0" in out


def test_record_tool_response_writes_raw_sidecar(tmp_path):
    from server.services.execution.log_store import ExecutionAgentLogStore

    store = ExecutionAgentLogStore(tmp_path / "logs")
    store.record_tool_response("A", "tool", "short", raw="short")  # identical: no sidecar entry
    store.record_tool_response("A", "tool", "cut", raw="the full " + "payload " * 50)
    entries = list(store.iter_raw_tool_responses("A"))
    assert len(entries) == 1
    ts, tool, raw = entries[0]
    assert tool == "tool" and raw.startswith("the full payload")
    assert [p for _, _, p in store.iter_entries("A")] == ["tool: short", "tool: cut"]
