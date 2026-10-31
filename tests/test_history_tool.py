"""search_history: the execution agent's own raw log is searchable, so compaction is lossy but
never unrecoverable."""

from __future__ import annotations

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
