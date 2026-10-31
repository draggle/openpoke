"""``search_history``: let an execution agent search its own raw log.

Tiered memory keeps the system prompt bounded, which makes it lossy. Rather than making the
compactor cleverer, give the agent a retrieval tool over the full journal so anything it
dropped is one call away (AMA-Bench's finding: retrieval exposed as a tool the agent queries
itself beats fixed top-k lookups on trajectory-style memory).
"""

from __future__ import annotations

import re
from functools import partial
from typing import Any, Callable, Dict, List, Optional

from rank_bm25 import BM25Okapi

from server.services.execution import get_execution_agent_logs
from server.services.execution.log_store import ExecutionAgentLogStore, raw_key
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
    # render. Search both, so anything compaction dropped is still findable. A raw result is
    # indexed together with the request that led to it: an email body says "our reference is
    # REF-1", the request said "the $182 refund", and the user asks about the $182 refund.
    by_key = _request_contexts(entries)
    raw_entries = []
    index_text: Dict[int, str] = {}
    context_of: Dict[int, str] = {}
    for ts, tool, raw, key in store.iter_raw_tool_responses_keyed(agent_name):
        if key and by_key.get(key):
            # Identical compacted payloads (the same boilerplate under the cap) share a key;
            # raw rows and log entries are both in write order, so consume contexts in order.
            context = by_key[key].pop(0)
        else:
            # Rows written before the key existed, or whose entry we cannot find: no context
            # rather than a guessed one (a wrong request is worse than none).
            context = ""
        index_text[len(entries) + len(raw_entries)] = f"{tool}: {raw} {context}"
        context_of[len(entries) + len(raw_entries)] = context
        raw_entries.append(("tool_response_raw", ts, f"{tool}: {raw}"))
    entries += raw_entries
    query = (query or "").strip()
    try:
        limit = max(1, min(int(limit), 20))
    except (TypeError, ValueError):
        limit = 5
    if not entries or not query:
        return {"query": query, "matches": [], "total_entries": len(entries)}

    query_tokens = tokenize(query)
    corpus = [tokenize(index_text.get(i, payload)) or ["_"] for i, (_, _, payload) in enumerate(entries)]
    bm25_scores = BM25Okapi(corpus).get_scores(query_tokens) if query_tokens else [0.0] * len(entries)
    max_bm = max((float(b) for b in bm25_scores), default=0.0)
    lowered = query.lower()

    unique_query = set(query_tokens)
    scored = []
    for index, ((tag, ts, payload), bm) in enumerate(zip(entries, bm25_scores)):
        # BM25's length normalisation buries a long raw payload under the short request that
        # mentions the same words; plain token overlap restores it (as registry.search does).
        overlap = len(unique_query & set(corpus[index])) / len(unique_query) if unique_query else 0.0
        score = ((float(bm) / max_bm) if max_bm > 0 else 0.0) + overlap
        if lowered in index_text.get(index, payload).lower():
            score += _EXACT_BONUS
        if score > 0:
            scored.append((score, index, tag, ts, payload))
    scored.sort(key=lambda item: (-item[0], -item[1]))

    # A raw result says what the tool returned; ``context`` says what it was asked for, which is
    # how the caller tells three near-identical email bodies apart.
    matches = [
        {"tag": tag, "timestamp": ts, "text": _window(payload, query, query_tokens),
         "context": context_of.get(index, "")[:200]}
        for _, index, tag, ts, payload in scored[:limit]
    ]
    return {"query": query, "matches": matches, "total_entries": len(entries)}


def _request_contexts(entries: List[tuple]) -> Dict[str, List[str]]:
    """The agent request that preceded each ``tool_response`` log entry, keyed by the entry's
    raw_key (how raw sidecar rows are tied to their entry), in write order per key."""
    by_key: Dict[str, List[str]] = {}
    last_request = ""
    for tag, _ts, payload in entries:
        if tag == "agent_request":
            last_request = payload
        elif tag == "tool_response":
            by_key.setdefault(raw_key(payload), []).append(last_request)
    return by_key


def _window(payload: str, query: str, query_tokens: List[str]) -> str:
    """A snippet that contains the match: centred on the exact query if present, else on the
    first query token, else the start of the entry."""
    if len(payload) <= _SNIPPET_CHARS:
        return payload
    lowered = payload.lower()
    position = lowered.find(query.lower())
    if position < 0:
        # No exact hit: put the window where the most distinct query tokens fall, not at the
        # first one ("Hannah" in a sender field at the top; the reference code at the bottom).
        hits = [(m.start(), t) for t in set(query_tokens) for m in re.finditer(re.escape(t), lowered)]
        best, best_count = 0, -1
        for start_pos, _t in hits:
            covered = {t for p, t in hits if start_pos <= p < start_pos + _SNIPPET_CHARS}
            if len(covered) > best_count:
                best, best_count = start_pos, len(covered)
        position = best
    start = max(0, position - _SNIPPET_CHARS // 3)
    end = min(len(payload), start + _SNIPPET_CHARS)
    start = max(0, end - _SNIPPET_CHARS)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(payload) else ""
    return (prefix + payload[start:end] + suffix)[: _SNIPPET_CHARS]


def build_registry(agent_name: str) -> Dict[str, Callable[..., Any]]:
    return {"search_history": partial(search_history, agent_name)}


__all__ = ["build_registry", "get_schemas", "search_history"]
