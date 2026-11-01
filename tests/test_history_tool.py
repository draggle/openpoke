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


def test_raw_match_carries_the_request_it_answered(tmp_path):
    """Three near-identical email bodies each hold a reference code; the model can only tell
    which is which if the match says what request produced it."""
    from server.services.execution.log_store import ExecutionAgentLogStore

    store = ExecutionAgentLogStore(tmp_path / "logs")
    body = "Thanks for reaching out. " * 20
    for amount, code in (("$454", "REF-11111"), ("$182", "REF-22222")):
        store.record_request("a", f"Ask Hannah whether the refund for {amount} can be settled by Nov 1")
        raw = '{"emails": [{"clean_text": "%sOur internal reference is %s."}]}' % (body, code)
        store.record_tool_response("a", "gmail_fetch_emails", raw[:500], raw=raw)
    hit = next(m for m in search_history("a", "$182 refund reference", limit=5, log_store=store)["matches"]
               if m["tag"] == "tool_response_raw")
    assert "REF-22222" in hit["text"]
    assert "$182" in hit["context"] and hit["context"].startswith("Ask Hannah")


def test_snippet_window_covers_the_most_query_tokens(tmp_path):
    """'Hannah' appears in the sender field at the top of a long payload and 'reference' near
    the end; the window must sit where most query tokens are, not at the first hit."""
    from server.agents.execution_agent.tools.history import _window

    payload = ('{"sender": "Hannah <h@x.com>", "clean_text": "' + "Per our earlier thread, here is the status update. " * 30
               + 'Our internal reference is REF-29252. Regards, Hannah"}')
    assert len(payload) > 900
    snippet = _window(payload, "deposit $2532 Hannah reference number", ["deposit", "2532", "hannah", "reference", "number"])
    assert "REF-29252" in snippet


# ---- review fix pass: keys survive the log's newline encoding; unkeyed rows get no context ---


def test_raw_key_survives_payloads_with_newlines(tmp_path):
    from server.services.execution.log_store import ExecutionAgentLogStore

    store = ExecutionAgentLogStore(tmp_path / "logs")
    body = "Thanks for reaching out. " * 20
    store.record_request("a", "Find the $50 parking email")
    raw0 = '{"emails": [{"snippet": "line one\\nline two", "clean_text": "%sOur internal reference is REF-PARK."}]}' % body
    store.record_tool_response("a", "gmail_fetch_emails", raw0[:500] + "\n", raw=raw0)
    store.record_request("a", "Ask Hannah whether the refund for $182 can be settled by Nov 1")
    raw1 = '{"emails": [{"snippet": "first\\nsecond", "clean_text": "%sOur internal reference is REF-182A."}]}' % body
    store.record_tool_response("a", "gmail_fetch_emails", raw1[:500] + "\n", raw=raw1)
    hit = next(m for m in search_history("a", "$182 refund reference", limit=5, log_store=store)["matches"]
               if m["tag"] == "tool_response_raw")
    assert "REF-182A" in hit["text"] and "$182" in hit["context"]


def test_unkeyed_legacy_raw_rows_get_no_context(tmp_path):
    from server.services.execution.log_store import ExecutionAgentLogStore

    store = ExecutionAgentLogStore(tmp_path / "logs")
    store.record_request("a", "Find the $50 parking email")
    store.record_tool_response("a", "createTrigger", '{"id": 1}', raw='{"id": 1}')  # identical: no raw row
    store.record_request("a", "Ask Hannah about the $182 refund")
    store._append_raw("a", "gmail_fetch_emails", "Our internal reference is REF-182A.")  # round-two row, no key
    store.record_tool_response("a", "gmail_fetch_emails", "compacted", raw=None)
    hit = next(m for m in search_history("a", "reference", limit=5, log_store=store)["matches"]
               if m["tag"] == "tool_response_raw")
    assert "REF-182A" in hit["text"] and hit["context"] == ""


# ---- hardening: the window counts whole tokens and runs in linear time --------------------


def test_snippet_window_counts_whole_tokens_not_substrings():
    """'ref' inside 'preference' is not a hit for the query token 'ref'; the span with the
    whole-word hits wins."""
    from server.agents.execution_agent.tools.history import _window

    top = "preference number " * 120          # substring 'ref' + whole 'number' -> 1 real token
    bottom = "ticket ref 1234 closed"          # whole 'ref' + '1234' -> 2 real tokens
    payload = top + bottom
    assert len(payload) > 900
    snippet = _window(payload, "ref number 1234", ["ref", "number", "1234"])
    assert "1234" in snippet


def test_snippet_window_is_fast_on_dense_payloads():
    import time
    from server.agents.execution_agent.tools.history import _window

    payload = ("alpha beta gamma delta " * 1000) + "omega REF-77"  # ~23 KB, ~4000 token hits
    start = time.perf_counter()
    for _ in range(5):
        snippet = _window(payload, "alpha beta gamma omega", ["alpha", "beta", "gamma", "omega"])
    elapsed = time.perf_counter() - start
    assert "omega" in snippet
    assert elapsed < 1.0, f"{elapsed:.2f}s for five windows"
