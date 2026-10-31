"""Compact tool results before they enter an agent's log.

Upstream truncates each tool response to 500 characters, which for a Gmail payload keeps the
*least* useful part (ids, label arrays, header boilerplate) and cuts the body. This keeps the
fields a future turn could plausibly need and drops the rest, so young agents (before memory
compaction kicks in) carry far less noise.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

# Keys that never help a later turn.
_NOISE_KEYS = frozenset(
    "historyId sizeEstimate internalDate labelIds mimeType partId headers raw payload parts "
    "threadId etag kind selfLink resultSizeEstimate nextPageToken".split()
)
# Keys worth keeping first, in this order, when present.
_PRIORITY_KEYS = (
    "id", "messageId", "subject", "from", "sender", "to", "date", "snippet", "preview",
    "body", "clean_text", "text", "status", "success", "error", "draft_id", "trigger_id",
    "next_trigger", "start_time", "recurrence_rule",
)


def compact_tool_payload(result: str, *, limit: int = 500, string_limit: int = 240) -> str:
    """Return a bounded, information-dense rendering of a tool result string."""
    if len(result) <= limit:
        return result
    try:
        data = json.loads(result)
    except (TypeError, ValueError):
        return result[:limit]
    rendered = _compact_value(data, string_limit)
    text = json.dumps(rendered, ensure_ascii=False, separators=(",", ":"), default=str)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _compact_value(value: Any, string_limit: int, depth: int = 0) -> Any:
    if depth > 4:
        return "…"
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        ordered = [k for k in _PRIORITY_KEYS if k in value] + [
            k for k in value if k not in _PRIORITY_KEYS
        ]
        for key in ordered:
            if key in _NOISE_KEYS:
                continue
            out[key] = _compact_value(value[key], string_limit, depth + 1)
        return out
    if isinstance(value, list):
        items: List[Any] = [_compact_value(v, string_limit, depth + 1) for v in value[:5]]
        if len(value) > 5:
            items.append(f"…(+{len(value) - 5} more)")
        return items
    if isinstance(value, str):
        return value if len(value) <= string_limit else value[: string_limit - 1] + "…"
    return value


__all__ = ["compact_tool_payload"]
