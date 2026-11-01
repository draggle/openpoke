"""Direct relay of execution results to the user.

An execution agent ends its final response with a one-line, user-facing summary in a
``<user_summary>`` block. When every result in a batch succeeded and carries one, the batch
manager records those lines as the reply and skips the interaction agent's follow-up call,
which would only have rephrased them. Any failure, or any result without a summary, takes the
usual path through the interaction agent. ``OPENPOKE_DIRECT_RELAY=0`` turns the relay off.
"""

from __future__ import annotations

import os
import re
from typing import List, Optional, Sequence

_SUMMARY_RE = re.compile(r"<user_summary>\s*(.*?)\s*</user_summary>", re.S | re.I)


def relay_enabled() -> bool:
    return os.getenv("OPENPOKE_DIRECT_RELAY", "1") != "0"


def user_summary(text: Optional[str]) -> Optional[str]:
    """The user-facing line an execution agent wrote, or None."""
    match = _SUMMARY_RE.search(text or "")
    if not match:
        return None
    summary = " ".join(match.group(1).split())
    return summary or None


def strip_summary(text: Optional[str]) -> str:
    """The response with the tags removed (the line itself is kept for the interaction agent)."""
    return _SUMMARY_RE.sub(lambda m: " ".join(m.group(1).split()), text or "").strip()


def relay_message(results: Sequence) -> Optional[str]:
    """The reply to record when the whole batch can be relayed as-is, else None."""
    lines: List[str] = []
    for result in results:
        if not getattr(result, "success", False):
            return None
        summary = user_summary(getattr(result, "response", None))
        if summary is None:
            return None
        lines.append(summary)
    return "\n".join(lines) if lines else None


__all__ = ["relay_enabled", "relay_message", "strip_summary", "user_summary"]
