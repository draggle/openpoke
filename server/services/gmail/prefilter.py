"""Rule-based pre-filter for the importance classifier.

The watcher runs an LLM classification on *every* new email. A large share of inbound mail
is bulk (newsletters, marketing, automated notifications) that no classifier should need a
model call to dismiss. This module makes the cheap, high-precision call first:

    prefilter(email) -> PrefilterDecision(skip=True, reason=...)   # never important
                     -> PrefilterDecision(skip=False, reason=...)  # ask the model

It is deliberately conservative: anything that *might* be important (OTPs, security notices,
calendar invites, replies, mail from a human) goes to the model. The rules are a known
precision/recall trade-off; ``tests/fixtures/emails.json`` is the labelled set they are
checked against, and ``evals/email_eval.py`` reports precision/recall for the rules alone
and (``--live``) for rules + model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

from .processing import ProcessedEmail

_BULK_SENDER_RE = re.compile(
    r"(^|[<\s.])(no-?reply|do-?not-?reply|noreply|newsletter|newsletters|news|marketing|promo|promotions|"
    r"offers|deals|digest|updates|bounce|mailer-daemon)[@.\-_]",
    re.I,
)
_BULK_SUBJECT_RE = re.compile(
    r"(unsubscribe|newsletter|% off|\bsale\b|limited time|last chance|your weekly|your daily|"
    r"this week in|digest|webinar|product update|we miss you|exclusive offer|flash sale|"
    r"don't miss|new arrivals|earn points|special offer)",
    re.I,
)
_UNSUBSCRIBE_BODY_RE = re.compile(r"unsubscribe|manage (your )?(email )?preferences|opt[- ]out", re.I)
_SIGNALS_KEEP_RE = re.compile(
    r"(\bOTP\b|one[- ]time (pass)?code|verification code|security (alert|code)|sign-?in|login attempt|"
    r"password|2fa|two-factor|invitation:|invite|calendar|meeting|interview|reschedul|urgent|"
    r"action required|re:|fwd?:|payment (failed|due|received)|invoice|receipt for your|"
    r"your (order|flight|booking|appointment)|confirm|fraud|unusual|suspicious|transaction|charge|declined|"
    r"review requested|mentioned you|assigned to you|overdue|deadline|cancel+ed)",
    re.I,
)
_BULK_LABELS = {"CATEGORY_PROMOTIONS", "CATEGORY_SOCIAL", "CATEGORY_FORUMS", "SPAM"}


@dataclass(frozen=True)
class PrefilterDecision:
    skip: bool
    reason: str


def prefilter(email: ProcessedEmail) -> PrefilterDecision:
    """Decide whether an email can be dismissed without a model call."""
    subject = email.subject or ""
    sender = email.sender or ""
    body = email.clean_text or ""
    labels = set(email.label_ids or [])

    # Anything carrying a "could matter" signal always goes to the model.
    if _SIGNALS_KEEP_RE.search(subject) or _SIGNALS_KEEP_RE.search(body[:600]):
        return PrefilterDecision(False, "keep-signal")

    if "SPAM" in labels:
        return PrefilterDecision(True, "label:spam")
    if labels & _BULK_LABELS and _UNSUBSCRIBE_BODY_RE.search(body):
        return PrefilterDecision(True, "label:bulk+unsubscribe")
    if _BULK_SENDER_RE.search(sender) and (_UNSUBSCRIBE_BODY_RE.search(body) or _BULK_SUBJECT_RE.search(subject)):
        return PrefilterDecision(True, "sender:bulk")
    if _BULK_SUBJECT_RE.search(subject) and _UNSUBSCRIBE_BODY_RE.search(body):
        return PrefilterDecision(True, "subject:bulk+unsubscribe")

    return PrefilterDecision(False, "no-rule")


def prefilter_many(emails: Iterable[ProcessedEmail]) -> list[tuple[ProcessedEmail, PrefilterDecision]]:
    return [(e, prefilter(e)) for e in emails]


__all__ = ["PrefilterDecision", "prefilter", "prefilter_many"]
