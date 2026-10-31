"""Compact tool results before they enter an agent's log.

Upstream truncates each tool response to 500 characters, which for a Gmail payload keeps the
*least* useful part (ids, label arrays, header boilerplate) and cuts the body. This keeps the
fields a future turn could plausibly need and drops the rest, so young agents (before memory
compaction kicks in) carry far less noise.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

# Keys that never help a later turn. ``threadId`` is NOT noise: ``gmail_reply_to_thread``
# needs it on a later turn.
_NOISE_KEYS = frozenset(
    "historyId sizeEstimate internalDate labelIds label_ids mimeType partId headers raw payload parts "
    "etag kind selfLink resultSizeEstimate nextPageToken has_attachments attachment_count "
    "attachment_filenames query recipient".split()
)
# When a list holds several records (search results), each is projected to the fields a
# later turn can act on, so the second and third emails survive the overall cap.
_ITEM_KEYS = (
    "id", "messageId", "thread_id", "threadId", "subject", "from", "sender", "to", "date", "timestamp",
    "snippet", "clean_text", "body", "text", "status", "error",
)
_ITEM_STRING_LIMIT = 120
# A list item is projected only when it looks like an email; other records (triggers, drafts,
# contacts) keep every non-noise field.
_EMAIL_SHAPE_KEYS = frozenset({"subject", "sender", "from", "clean_text", "snippet", "messageId"})
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
            # "payload" is Gmail MIME noise when it is a dict, but a trigger's instruction text
            # when it is a string.
            if key in _NOISE_KEYS and not (key == "payload" and isinstance(value[key], str)):
                continue
            out[key] = _compact_value(value[key], string_limit, depth + 1)
        return out
    if isinstance(value, list):
        items: List[Any] = [
            _project_item(v) if isinstance(v, dict) and (set(v) & _EMAIL_SHAPE_KEYS)
            else _compact_value(v, string_limit, depth + 1)
            for v in value[:5]
        ]
        if len(value) > 5:
            items.append(f"…(+{len(value) - 5} more)")
        return items
    if isinstance(value, str):
        return value if len(value) <= string_limit else value[: string_limit - 1] + "…"
    return value


def _project_item(item: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in _ITEM_KEYS:
        if key in item and item[key] not in (None, "", [], {}):
            value = item[key]
            if isinstance(value, str) and len(value) > _ITEM_STRING_LIMIT:
                value = value[: _ITEM_STRING_LIMIT - 1] + "…"
            out[key] = value
    return out or _compact_value(item, _ITEM_STRING_LIMIT, depth=1)


__all__ = ["compact_tool_payload"]
