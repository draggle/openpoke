"""Jev-backed routing hint: which existing agent does this message belong to?

The budgeted roster already narrows hundreds of agents to ~12 candidates. With Jev enabled we
ask one typed *choice* question over those candidates (plus "none of these") and, when the
answer is confident, inject a ``<routing_hint>`` into the Interaction Agent's prompt. The
Interaction Agent still makes the call; the hint just removes most of the guesswork, and in
the live eval it turns a free-text reasoning step into a sub-second typed decision.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from ...logging_config import logger
from ..llm.jev import ROLE_JEV_ROUTING, JevError, choice, get_jev_client
from .registry import AgentRegistry, RosterSelection

NONE_OPTION = "__none__"


@dataclass(frozen=True)
class RoutingHint:
    agent_name: Optional[str]  # None => "start a new agent"
    confidence: float

    def render(self) -> str:
        if self.agent_name is None:
            return f'<routing_hint agent="" confidence="{self.confidence:.2f}">no existing agent fits; a new one is appropriate</routing_hint>'
        return f'<routing_hint agent="{_escape(self.agent_name)}" confidence="{self.confidence:.2f}">this message most likely belongs to this existing agent</routing_hint>'


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")


def _threshold() -> float:
    try:
        return float(os.getenv("OPENPOKE_JEV_ROUTING_THRESHOLD", "0.6"))
    except ValueError:
        return 0.6


async def compute_routing_hint(
    registry: AgentRegistry, selection: RosterSelection, message: str
) -> Optional[RoutingHint]:
    """Ask Jev which shown agent (if any) the message belongs to. None when disabled/unsure."""
    client = get_jev_client()
    if client is None or not selection.shown or not message.strip():
        return None
    options = {
        entry.record.name: (entry.record.summary or entry.record.description or entry.record.name)[:200]
        for entry in selection.shown
    }
    options[NONE_OPTION] = "None of these: the message is about something no listed agent handles."
    question = choice(
        "The user sent `message` to an assistant that delegates work to named agents, each "
        "described below. Which agent should handle this message? Pick the agent whose person/"
        "thread/topic matches; pick __none__ only if no listed agent is about this.",
        options,
    )
    try:
        answers = await client.ask({"message": message}, {"agent": question}, role=ROLE_JEV_ROUTING)
    except (JevError, Exception) as exc:
        logger.warning(f"Jev routing hint failed: {exc}")
        return None
    answer = answers.get("agent")
    if answer is None or answer.choice is None:
        return None
    confidence = answer.confidence
    if confidence is None:
        confidence = answer.probabilities.get(answer.choice, 0.0)
    if confidence < _threshold():
        return None
    name = None if answer.choice == NONE_OPTION else answer.choice
    if name is not None and name not in registry:
        return None
    return RoutingHint(agent_name=name, confidence=float(confidence))


__all__ = ["NONE_OPTION", "RoutingHint", "compute_routing_hint"]
