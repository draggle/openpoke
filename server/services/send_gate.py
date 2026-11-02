"""The send gate: outbound email needs the user's go-ahead, in code.

Execution agents read email written by third parties, and the tools that send mail sit in the
same loop as the tools that read it. Nothing upstream stops a line in an email ("forward this
thread to ...") from becoming a send: the model is asked not to, and that is the whole
defence. This module makes it a property of the run instead.

A run may send only when it was dispatched from a *user* turn whose message asked for a send
(a send verb), or affirmed a draft the assistant had just shown. A run fired by a reminder, or
started while handling an agent's result, is never authorised: it can draft and report, and the
user sends on the next turn. ``OPENPOKE_SEND_GATE=0`` restores upstream behaviour.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

OUTBOUND_TOOLS = frozenset({"gmail_execute_draft", "gmail_forward_email", "gmail_reply_to_thread"})

_SEND_VERB_RE = re.compile(
    r"\b(send|sends|resend|reply|respond|forward|e?-?mail(?= (?!about|from|with|in|that|which|i |is|was)\w)|write (?:back|to)|tell|"
    r"let .{0,30}\bknow|cc|confirm with|follow up with|chase|nudge|ping|get back to|answer)\b",
    re.I,
)
# A request to see the text first: a drafting verb before the send verb, or an explicit "don't
# send". "send the draft" is still a send (the send verb comes first).
_DRAFT_INTENT_RE = re.compile(
    r"\b(draft|write up|prepare|compose|put together|show me)\b|\b(don'?t|do not|without|before|not)\s+(send|sending)\b",
    re.I,
)
_AFFIRMATIVE_RE = re.compile(
    r"^\W*(yes|yep|yeah|yup|ok|okay|sure|go ahead|do it|looks good|lgtm|approved|approve|fine|"
    r"please do|ship it|go for it|that works|perfect|send(?: it)?)(?:[\s,.!]+(go ahead|send it|do it|please|thanks))*\W*$",
    re.I,
)


@dataclass(frozen=True)
class SendAuthorisation:
    allowed: bool
    reason: str


def gate_enabled() -> bool:
    return os.getenv("OPENPOKE_SEND_GATE", "1") != "0"


def authorise_send(user_message: Optional[str], *, last_reply_was_draft: bool = False,
                   source: str = "user") -> SendAuthorisation:
    """Decide whether a run dispatched now may call an outbound Gmail tool."""
    if source != "user":
        return SendAuthorisation(False, "this run was started by a reminder or in the background, not by the user")
    text = (user_message or "").strip()
    if not text:
        return SendAuthorisation(False, "no user message this turn")
    send = _SEND_VERB_RE.search(text)
    draft = _DRAFT_INTENT_RE.search(text)
    if draft and (send is None or draft.start() < send.start() or draft.group(2)):
        return SendAuthorisation(False, "the user asked for a draft to review, not a send")
    if send:
        return SendAuthorisation(True, "the user asked to send this turn")
    if last_reply_was_draft and _AFFIRMATIVE_RE.match(text):
        return SendAuthorisation(True, "the user approved the draft just shown")
    return SendAuthorisation(False, "the user did not ask to send anything this turn")


# Whether the user has a draft in front of them: set by the interaction agent's send_draft tool
# or by any user-sourced batch that created one; cleared by a plain reply or a batch without one.
_REPLY_WAS_DRAFT = False
_BATCH_CREATED_DRAFT = False


def note_reply(*, draft: bool) -> None:
    global _REPLY_WAS_DRAFT
    _REPLY_WAS_DRAFT = draft


def note_batch(*, created_draft: bool) -> None:
    global _BATCH_CREATED_DRAFT
    _BATCH_CREATED_DRAFT = created_draft


def draft_pending() -> bool:
    return _REPLY_WAS_DRAFT or _BATCH_CREATED_DRAFT


def refusal(tool_name: str, authorisation: Optional[SendAuthorisation]) -> str:
    reason = authorisation.reason if authorisation else "no authorisation was attached to this run"
    return (f"{tool_name} is not available in this run: {reason}. Create a draft with "
            f"gmail_create_draft and tell the user exactly what you would send; they can approve it.")


__all__ = ["OUTBOUND_TOOLS", "SendAuthorisation", "authorise_send", "draft_pending", "gate_enabled", "note_batch", "note_reply", "refusal"]
