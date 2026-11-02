"""Direct relay of execution results to the user.

An execution agent ends its final response with a one-line, user-facing summary in a
``<user_summary>`` block. When a turn ran exactly one agent and its result succeeded with that
line, the batch manager records the line as the reply and skips the interaction agent's
follow-up call, which would only have rephrased it. A failure, a result without the line, or a
turn that fanned out to several agents takes the usual path: each agent knows only its own
thread, so two agents' lines ("you have one reminder" twice, each naming a different one) need
the interaction agent to synthesise them. ``OPENPOKE_DIRECT_RELAY=0`` turns the relay off.
"""

from __future__ import annotations

import os
import re
from typing import Optional, Sequence

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
    """The reply to record when the batch is one successful result with a summary, else None."""
    if len(results) != 1:
        return None
    result = results[0]
    if not getattr(result, "success", False):
        return None
    return user_summary(getattr(result, "response", None))


__all__ = ["relay_enabled", "relay_message", "strip_summary", "user_summary"]
