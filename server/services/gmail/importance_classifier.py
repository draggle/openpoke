"""LLM-powered classifier for determining important Gmail emails."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from .processing import ProcessedEmail
from ...config import get_settings
from ...logging_config import logger
from ...openrouter_client import OpenRouterError
from ..llm import ROLE_EMAIL_CLASSIFIER, complete as llm_complete
from ..llm.jev import ROLE_JEV_EMAIL, JevError, get_jev_client, noul


_TOOL_NAME = "mark_email_importance"
_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": _TOOL_NAME,
        "description": (
            "Decide whether an email should be proactively surfaced to the user and, "
            "if so, provide a natural-language summary explaining why."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "important": {
                    "type": "boolean",
                    "description": (
                        "Set to true only when the email requires timely attention, a decision, "
                        "coordination, or contains critical security information (e.g. OTPs)."
                    ),
                },
                "summary": {
                    "type": "string",
                    "description": (
                        "Concise 2-3 sentence summary highlighting sender, topic, and the "
                        "specific action or urgency for the user. Only include when important=true."
                    ),
                },
            },
            "required": ["important"],
            "additionalProperties": False,
        },
    },
}

_SUMMARY_ONLY_SYSTEM_PROMPT = (
    "A fast classifier has already determined that this Gmail message warrants an immediate "
    "proactive notification to the user. Do not second-guess that decision: call the tool with "
    "important=true and write a brief 2-3 sentence summary describing who it is from, what "
    "happened, and why it matters or what action is needed."
)

_SYSTEM_PROMPT = (
    "You review incoming Gmail messages and decide whether they warrant an immediate proactive "
    "notification to the user. Judge the message by what it is about and who it is from, not by "
    "its formatting.\n\n"
    "Mark important=true for:\n"
    "- a person replying to or writing to the user directly, including short or casual replies "
    "that confirm, decline or move a plan (lunch, a call, a meeting), ask the user for something, "
    "or answer something the user asked;\n"
    "- money and account alerts: low balance, declined or failed payment, unusual or suspicious "
    "activity, fraud, payroll errors, a bill or invoice due, an overdue amount;\n"
    "- job and recruiter outreach addressed to the user (a hiring manager, recruiter or "
    "applicant-tracking system proposing a role, an interview or a next step);\n"
    "- government, immigration, legal, tax, medical or insurance status changes, appointments "
    "and decisions about the user's own case or application;\n"
    "- security codes, one-time passcodes and sign-in notices;\n"
    "- deadlines, decisions the user must make, and changes to meetings, interviews, "
    "appointments, flights or bookings the user already has;\n"
    "- outages or incidents on systems the user is responsible for.\n\n"
    "Mark important=false for: order and shipping confirmations, receipts, routine statements "
    "that need no action, marketing and sales, newsletters and digests, social-network "
    "notifications, product update announcements, surveys, mass recruiting campaigns that are not "
    "addressed to the user personally, and low-impact status notifications.\n\n"
    "An unsubscribe, opt-out or manage-preferences footer does NOT make a message promotional "
    "when the body is about the user's own account, application, money, appointment or case, or "
    "comes from a person writing to the user. Decide on the body. Examples: 'Re: coffee?' with "
    "'Tuesday 10 works, see you then' is important; 'Your card ending 2210 was declined at Shell' "
    "with a preferences footer is important; 'You appeared in 5 searches this week' is not.\n\n"
    "When important, write a brief 2-3 sentence summary for the user: who it is from, what "
    "happened, and what action or decision it needs."
)


def _format_email_payload(email: ProcessedEmail) -> str:
    attachments = ", ".join(email.attachment_filenames) if email.attachment_filenames else "None"
    labels = ", ".join(email.label_ids) if email.label_ids else "None"
    header_lines = [
        f"Sender: {email.sender}",
        f"Recipient: {email.recipient}",
        f"Subject: {email.subject}",
        f"Received (user timezone): {email.timestamp.isoformat()}",
        f"Thread ID: {email.thread_id or 'None'}",
        f"Labels: {labels}",
        f"Has attachments: {'Yes' if email.has_attachments else 'No'}",
        f"Attachment filenames: {attachments}",
    ]

    return (
        "Email Metadata:\n"
        + "\n".join(header_lines)
        + "\n\nCleaned Body:\n"
        + (email.clean_text or "(empty body)")
    )


_JEV_QUESTION = noul(
    "Should the assistant proactively interrupt the user about this email right now? "
    "Yes only if it requires timely attention, a decision, coordination, or is a security-"
    "sensitive code/notice (OTP, sign-in). No for receipts, marketing, newsletters, routine "
    "notifications and low-impact updates.",
    true_means="Interrupt the user: timely action, decision, meeting/interview change, or security code.",
    false_means="Do not interrupt: routine, promotional, informational, or can wait.",
)


def _jev_threshold() -> float:
    try:
        return float(os.getenv("OPENPOKE_JEV_EMAIL_THRESHOLD", "0.5"))
    except ValueError:
        return 0.5


async def jev_importance_probability(email: ProcessedEmail) -> Optional[float]:
    """P(important) from Jev, or None when Jev is disabled or fails (caller falls back)."""
    client = get_jev_client()
    if client is None:
        return None
    try:
        answers = await client.ask(_format_email_payload(email), {"important": _JEV_QUESTION},
                                   role=ROLE_JEV_EMAIL, tag=email.id or "")
        return answers["important"].probability
    except (JevError, KeyError, Exception) as exc:  # fall back to the LLM path
        logger.warning(f"Jev importance check failed for {email.id}: {exc}")
        return None


async def classify_email_importance(email: ProcessedEmail) -> Optional[str]:
    """Return summary text when email should be surfaced; otherwise None.

    With Jev enabled, a typed yes/no decision gates the call: only emails Jev scores at or
    above ``OPENPOKE_JEV_EMAIL_THRESHOLD`` reach the chat model, which then writes the
    summary. Without Jev (or on Jev failure) the chat model makes both decisions as upstream.
    """

    settings = get_settings()
    api_key = settings.openrouter_api_key

    if not api_key:
        logger.warning("Skipping importance check; OpenRouter API key missing")
        return None

    probability = await jev_importance_probability(email)
    if probability is not None and probability < _jev_threshold():
        logger.debug(f"Jev dismissed {email.id} (p={probability:.2f})")
        return None
    jev_decided = probability is not None  # Jev said important: the LLM only summarises

    user_payload = _format_email_payload(email)
    messages = [{"role": "user", "content": user_payload}]

    try:
        response = await llm_complete(
            role=ROLE_EMAIL_CLASSIFIER,
            messages=messages,
            system=_SUMMARY_ONLY_SYSTEM_PROMPT if jev_decided else _SYSTEM_PROMPT,
            api_key=api_key,
            tools=[_TOOL_SCHEMA],
            tag=email.id or "",
        )
    except OpenRouterError as exc:
        logger.error(
            "Importance classification failed",
            extra={"message_id": email.id, "error": str(exc)},
        )
        return None
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception(
            "Unexpected error during importance classification",
            extra={"message_id": email.id},
        )
        return None

    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    tool_calls = message.get("tool_calls") or []

    for tool_call in tool_calls:
        function_block = tool_call.get("function") or {}
        if function_block.get("name") != _TOOL_NAME:
            continue

        raw_arguments = function_block.get("arguments")
        arguments = _coerce_arguments(raw_arguments)
        if arguments is None:
            logger.warning(
                "Importance tool returned invalid arguments",
                extra={"message_id": email.id},
            )
            return None

        important = bool(arguments.get("important")) or jev_decided
        summary = arguments.get("summary")

        if not important:
            return None

        if not isinstance(summary, str) or not summary.strip():
            logger.warning(
                "Importance tool marked email important without summary",
                extra={"message_id": email.id},
            )
            # Jev already decided this one matters: never drop it for want of prose.
            return _fallback_summary(email) if jev_decided else None

        return summary.strip()

    logger.debug(
        "Importance classification produced no tool call",
        extra={"message_id": email.id},
    )
    return _fallback_summary(email) if jev_decided else None


def _fallback_summary(email: ProcessedEmail) -> str:
    """A serviceable notification when the summary model returned nothing usable."""
    sender = (email.sender or "an unknown sender").strip()
    subject = (email.subject or "(no subject)").strip()
    body = " ".join((email.clean_text or "").split())
    snippet = (body[:157] + "…") if len(body) > 160 else body
    return f"{sender}: {subject}." + (f" {snippet}" if snippet else "")


def _coerce_arguments(raw: Any) -> Optional[Dict[str, Any]]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        if not raw.strip():
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None
    return None


__all__ = ["classify_email_importance"]
