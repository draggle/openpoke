"""Rule-based pre-filter for the importance classifier.

The watcher runs an LLM classification on *every* new email. A large share of inbound mail
is bulk (newsletters, marketing, automated notifications) that no classifier should need a
model call to dismiss. This module makes the cheap, high-precision call first:

    prefilter(email) -> PrefilterDecision(skip=True, reason=...)   # never important
                     -> PrefilterDecision(skip=False, reason=...)  # ask the model

It is deliberately conservative: only Gmail-labelled promotions with an unsubscribe footer and
spam are dropped; everything else (OTPs, alerts, forums, social, no-reply senders) goes to the model. The rules are a known
precision/recall trade-off; ``tests/fixtures/emails.json`` is the labelled set they are
checked against, and ``evals/email_eval.py`` reports precision/recall for the rules alone
and (``--live``) for rules + model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

from .processing import ProcessedEmail

_UNSUBSCRIBE_BODY_RE = re.compile(r"unsubscribe|manage (your )?(email )?preferences|opt[- ]out", re.I)
_SIGNALS_KEEP_RE = re.compile(
    r"(\bOTP\b|one[- ]time (pass)?code|verification code|security (alert|code)|sign[- ]?in|login attempt|"
    r"password|2fa|two-factor|invitation:|invite|calendar|meeting|interview|reschedul|urgent|"
    r"action required|(^|\s)re:|(^|\s)fwd?:|payment (failed|due|received)|invoice|receipt for your|"
    r"your (order|flight|booking|appointment|application)|confirm|fraud|unusual|suspicious|transaction|charge|declined|"
    r"review requested|mentioned you|assigned to you|overdue|deadline|cancel+ed|balance|alert|"
    r"\b(down|outage|incident|failover)\b|status has changed|code:)",
    re.I,
)
# Only mail Gmail itself has already classified as promotional may be dropped on a footer match.
# Forums (Google Groups), Social (LinkedIn, recruiter mail), Updates (bank alerts, sign-in codes)
# and no-reply senders all carry important mail with unsubscribe footers; those go to the model.
_HARD_SKIP_LABELS = {"CATEGORY_PROMOTIONS"}


@dataclass(frozen=True)
class PrefilterDecision:
    skip: bool
    reason: str


def prefilter(email: ProcessedEmail) -> PrefilterDecision:
    """Decide whether an email can be dismissed without a model call.

    The filter is a *routing* layer, not a classifier: it only short-circuits mail that
    Gmail has labelled promotional and that carries an unsubscribe footer, or spam. Everything
    else, including bulk-looking senders, reaches the model, because the cost of a missed
    alert is far higher than one cheap-model call.
    """
    subject = email.subject or ""
    body = email.clean_text or ""
    labels = set(email.label_ids or [])

    # Anything carrying a "could matter" signal always goes to the model.
    if _SIGNALS_KEEP_RE.search(subject) or _SIGNALS_KEEP_RE.search(body):
        return PrefilterDecision(False, "keep-signal")

    if "SPAM" in labels:
        return PrefilterDecision(True, "label:spam")
    if labels & _HARD_SKIP_LABELS and _UNSUBSCRIBE_BODY_RE.search(body):
        return PrefilterDecision(True, "label:promotions+unsubscribe")

    return PrefilterDecision(False, "no-rule")


def prefilter_many(emails: Iterable[ProcessedEmail]) -> list[tuple[ProcessedEmail, PrefilterDecision]]:
    return [(e, prefilter(e)) for e in emails]


__all__ = ["PrefilterDecision", "prefilter", "prefilter_many"]
