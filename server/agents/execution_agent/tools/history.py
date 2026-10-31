"""``search_history``: let an execution agent search its own raw log.

Tiered memory keeps the system prompt bounded, which makes it lossy. Rather than making the
compactor cleverer, give the agent a retrieval tool over the full journal so anything it
dropped is one call away (AMA-Bench's finding: retrieval exposed as a tool the agent queries
itself beats fixed top-k lookups on trajectory-style memory).
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, List, Optional

from rank_bm25 import BM25Okapi

from server.services.execution import get_execution_agent_logs
from server.services.execution.log_store import ExecutionAgentLogStore
from server.services.execution.registry import tokenize

_SNIPPET_CHARS = 600
_EXACT_BONUS = 1.0  # a verbatim substring hit (reference codes, ids) outranks lexical overlap

_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_history",
            "description": (
                "Search this agent's full activity log (every request, tool result and reply, "
                "including ones compressed out of the history above). Use it when you need an exact "
                "detail from earlier work: a reference number, an email id, what someone said, a date."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keywords or an exact code/id to look for."},
                    "limit": {"type": "integer", "description": "Maximum entries to return (default 5)."},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }
]


def get_schemas() -> List[Dict[str, Any]]:
    return list(_SCHEMAS)


def search_history(
    agent_name: str,
    query: str,
    limit: int = 5,
    *,
    log_store: Optional[ExecutionAgentLogStore] = None,
) -> Dict[str, Any]:
    """Return the log entries best matching ``query``, newest first among ties."""
    store = log_store or get_execution_agent_logs()
    entries = list(store.iter_entries(agent_name))
    # The raw sidecar holds full tool results; the log holds the compacted form that prompts
    # render. Search both, so anything compaction dropped is still findable.
    entries += [("tool_response_raw", ts, f"{tool}: {raw}") for ts, tool, raw in store.iter_raw_tool_responses(agent_name)]
    query = (query or "").strip()
    try:
        limit = max(1, min(int(limit), 20))
    except (TypeError, ValueError):
        limit = 5
    if not entries or not query:
        return {"query": query, "matches": [], "total_entries": len(entries)}

    query_tokens = tokenize(query)
    corpus = [tokenize(payload) or ["_"] for _, _, payload in entries]
    bm25_scores = BM25Okapi(corpus).get_scores(query_tokens) if query_tokens else [0.0] * len(entries)
    max_bm = max((float(b) for b in bm25_scores), default=0.0)
    lowered = query.lower()

    scored = []
    for index, ((tag, ts, payload), bm) in enumerate(zip(entries, bm25_scores)):
        score = (float(bm) / max_bm) if max_bm > 0 else 0.0
        if lowered in payload.lower():
            score += _EXACT_BONUS
        if score > 0:
            scored.append((score, index, tag, ts, payload))
    scored.sort(key=lambda item: (-item[0], -item[1]))

    matches = [
        {"tag": tag, "timestamp": ts, "text": _window(payload, query, query_tokens)}
        for _, _, tag, ts, payload in scored[:limit]
    ]
    return {"query": query, "matches": matches, "total_entries": len(entries)}


def _window(payload: str, query: str, query_tokens: List[str]) -> str:
    """A snippet that contains the match: centred on the exact query if present, else on the
    first query token, else the start of the entry."""
    if len(payload) <= _SNIPPET_CHARS:
        return payload
    lowered = payload.lower()
    position = lowered.find(query.lower())
    if position < 0:
        position = min((p for p in (lowered.find(t) for t in query_tokens) if p >= 0), default=0)
    start = max(0, position - _SNIPPET_CHARS // 3)
    end = min(len(payload), start + _SNIPPET_CHARS)
    start = max(0, end - _SNIPPET_CHARS)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(payload) else ""
    return (prefix + payload[start:end] + suffix)[: _SNIPPET_CHARS]


def build_registry(agent_name: str) -> Dict[str, Callable[..., Any]]:
    return {"search_history": partial(search_history, agent_name)}


__all__ = ["build_registry", "get_schemas", "search_history"]
