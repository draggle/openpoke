"""search_history: the execution agent's own raw log is searchable, so compaction is lossy but
never unrecoverable."""

from __future__ import annotations

from server.agents.execution_agent.tools.history import search_history
from server.services.execution.log_store import ExecutionAgentLogStore


def _store_with_buried_ref(tmp_path):
    store = ExecutionAgentLogStore(tmp_path / "logs")
    agent = "Email to Priya"
    for i in range(30):
        store.record_request(agent, f"Round {i}: ask Priya about item {i}")
        payload = '{"id": "m%d", "snippet": "boilerplate"}' % i
        if i == 11:
            payload = '{"id": "m11", "body": "Confirming receipt. Our internal reference is REF-55555."}'
        store.record_tool_response(agent, "gmail_fetch_emails", payload)
        store.record_agent_response(agent, f"Done with item {i}")
    return store, agent


def test_search_history_finds_buried_payload(tmp_path):
    from server.agents.execution_agent.tools.history import search_history

    store, agent = _store_with_buried_ref(tmp_path)
    result = search_history(agent, "REF-55555", limit=3, log_store=store)
    assert result["total_entries"] == 90
    assert result["matches"], result
    assert "REF-55555" in result["matches"][0]["text"]
    assert result["matches"][0]["tag"] == "tool_response"


def test_search_history_ranks_lexical_matches(tmp_path):
    from server.agents.execution_agent.tools.history import search_history

    store, agent = _store_with_buried_ref(tmp_path)
    result = search_history(agent, "ask Priya about item 7", limit=2, log_store=store)
    assert "item 7" in result["matches"][0]["text"]


def test_search_history_empty_query_or_no_log(tmp_path):
    from server.agents.execution_agent.tools.history import search_history

    store, agent = _store_with_buried_ref(tmp_path)
    assert search_history(agent, "", log_store=store)["matches"] == []
    assert search_history("Nobody", "anything", log_store=store) == {"query": "anything", "matches": [], "total_entries": 0}


def test_schema_and_registry_registered():
    from server.agents.execution_agent.tools.registry import get_tool_registry, get_tool_schemas

    assert "search_history" in {s["function"]["name"] for s in get_tool_schemas()}
    assert "search_history" in get_tool_registry("Email to Priya")


def test_search_history_finds_needle_only_in_raw_sidecar(tmp_path):
    """The log keeps the compacted payload; the raw sidecar keeps everything, so a reference the
    compaction dropped is still recoverable."""
    from server.agents.execution_agent.tools.history import search_history

    store = ExecutionAgentLogStore(tmp_path / "logs")
    agent = "Email to Priya"
    compacted = '{"status":"success","emails":[{"id":"m1","subject":"Re: invoice"}]}'
    raw = '{"status":"success","emails":[{"id":"m1","subject":"Re: invoice","clean_text":"' + "x" * 700 + ' reference REF-77777 end"}]}'
    store.record_tool_response(agent, "gmail_fetch_emails", compacted, raw=raw)
    result = search_history(agent, "REF-77777", log_store=store)
    assert result["matches"], result
    assert "REF-77777" in result["matches"][0]["text"]
    assert result["matches"][0]["tag"] == "tool_response_raw"


def test_snippet_window_contains_the_match(tmp_path):
    from server.agents.execution_agent.tools.history import search_history

    store = ExecutionAgentLogStore(tmp_path / "logs")
    agent = "Long"
    store.record_agent_response(agent, "a" * 900 + " the code is REF-31337 " + "b" * 300)
    result = search_history(agent, "REF-31337", log_store=store)
    assert "REF-31337" in result["matches"][0]["text"]
    assert len(result["matches"][0]["text"]) <= 600


# ---- round three, after the live run: a raw payload is findable by the request it answered --


def test_raw_payload_found_through_its_request_context(tmp_path):
    """Gmail results carry no amount; the request that triggered the fetch does. A realistic
    query ('$182 reference') must still rank the right raw payload first."""
    from server.services.execution.log_store import ExecutionAgentLogStore

    store = ExecutionAgentLogStore(tmp_path / "logs")
    body = "Thanks for reaching out. " * 20
    for amount, code in (("$454", "REF-11111"), ("$182", "REF-22222"), ("$2532", "REF-33333")):
        store.record_request("a", f"Ask Hannah whether the refund for {amount} can be settled by Nov 1")
        store.record_action("a", "Calling gmail_fetch_emails with: {}")
        raw = '{"emails": [{"subject": "Re: payment", "clean_text": "%sOur internal reference is %s. Regards, Hannah"}]}' % (body, code)
        store.record_tool_response("a", "gmail_fetch_emails", raw[:500], raw=raw)
        store.record_agent_response("a", f"Hannah confirmed {amount}")
    matches = search_history("a", "$182 reference", limit=3, log_store=store)["matches"]
    raw_hits = [m for m in matches if m["tag"] == "tool_response_raw"]
    assert raw_hits and "REF-22222" in raw_hits[0]["text"]
    assert "REF-11111" not in raw_hits[0]["text"]


def test_raw_context_survives_a_short_result_without_a_raw_row(tmp_path):
    """A result under the cap is written to the log only (no raw row). The raw rows after it
    must still map to their own requests, not slide one request back."""
    from server.services.execution.log_store import ExecutionAgentLogStore

    store = ExecutionAgentLogStore(tmp_path / "logs")
    body = "Thanks for reaching out. " * 20
    store.record_request("a", "Remind me Monday about the passport")
    short = '{"id": 103, "status": "active"}'
    store.record_tool_response("a", "createTrigger", short, raw=short)  # identical: no raw row
    store.record_request("a", "Ask Hannah whether the refund for $182 can be settled by Nov 1")
    raw = '{"emails": [{"clean_text": "%sOur internal reference is REF-22222."}]}' % body
    store.record_tool_response("a", "gmail_fetch_emails", raw[:500], raw=raw)
    store.record_request("a", "Ask Hannah whether the deposit for $454 can be settled by Dec 1")
    raw2 = '{"emails": [{"clean_text": "%sOur internal reference is REF-11111."}]}' % body
    store.record_tool_response("a", "gmail_fetch_emails", raw2[:500], raw=raw2)
    hits = [m for m in search_history("a", "$182 refund reference", limit=5, log_store=store)["matches"]
            if m["tag"] == "tool_response_raw"]
    assert hits and "REF-22222" in hits[0]["text"]
