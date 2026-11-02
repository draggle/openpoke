"""Tool definitions for interaction agent."""

import asyncio
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Optional

from ...logging_config import logger
from ...services.conversation import get_conversation_log
from ...services.send_gate import authorise_send
from ...services.execution import (
    get_agent_memory,
    get_agent_registry,
    get_agent_roster,
    get_execution_agent_logs,
)
from ..execution_agent.batch_manager import ExecutionBatchManager


@dataclass
class ToolResult:
    """Standardized payload returned by interaction-agent tools."""

    success: bool
    payload: Any = None
    user_message: Optional[str] = None
    recorded_reply: bool = False

# Tool schemas for OpenRouter
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "send_message_to_agent",
            "description": "Deliver instructions to a specific execution agent. Creates a new agent if the name doesn't exist in the roster, or reuses an existing one.",
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "description": "Human-readable agent name describing its purpose (e.g., 'Vercel Job Offer', 'Email to Sharanjeet'). This name will be used to identify and potentially reuse the agent."
                    },
                    "instructions": {"type": "string", "description": "Instructions for the agent to execute."},
                },
                "required": ["agent_name", "instructions"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_message_to_user",
            "description": "Deliver a natural-language response directly to the user. Use this for updates, confirmations, or any assistant response the user should see immediately.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Plain-text message that will be shown to the user and recorded in the conversation log.",
                    },
                },
                "required": ["message"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_draft",
            "description": "Record an email draft so the user can review the exact text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Recipient email for the draft.",
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject for the draft.",
                    },
                    "body": {
                        "type": "string",
                        "description": "Email body content (plain text).",
                    },
                },
                "required": ["to", "subject", "body"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_agents",
            "description": (
                "Search existing execution agents by topic, person, or keyword. The roster shown "
                "in <active_agents> is only the most recent and most relevant agents; call this "
                "before creating a new agent when the task may belong to an older agent that is "
                "not listed (e.g. a follow-up about a person or thread handled days ago)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keywords describing the person, thread, or task to look for.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of matches to return (default 5).",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait",
            "description": "Wait silently when a message is already in conversation history to avoid duplicating responses. Adds a <wait> log entry that is not visible to the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Brief explanation of why waiting (e.g., 'Message already sent', 'Draft already created').",
                    },
                },
                "required": ["reason"],
                "additionalProperties": False,
            },
        },
    },
]

_EXECUTION_BATCH_MANAGER = ExecutionBatchManager()

# One challenge per (agent, interaction turn): the model's second attempt in the same turn,
# repeated or rephrased, is honoured, so the guard can never loop the tool-iteration budget.
_GUARD_SEEN: set = set()
_TURN = 0


_TURN_MESSAGE = ""
_TURN_DISPATCHED: list = []  # agents dispatched this turn; two or more make it a fan-out
_PREVIOUS_USER_FAN_OUT: list = []  # agents the previous *user* turn fanned out to, else empty
_LAST_REPLY_WAS_DRAFT = False  # the assistant's last reply to the user was a draft to approve


def begin_turn(user_message: str = "") -> int:
    """Called by the runtime at the start of each user or agent turn."""
    global _TURN, _TURN_MESSAGE, _PREVIOUS_USER_FAN_OUT
    _TURN += 1
    if user_message:
        # Agent-result turns sit between two user turns; only a user turn closes the window.
        _PREVIOUS_USER_FAN_OUT = list(_TURN_DISPATCHED) if len(_TURN_DISPATCHED) > 1 else []
        _TURN_DISPATCHED.clear()
    _TURN_MESSAGE = user_message or ""
    _GUARD_SEEN.clear()
    return _TURN


def current_turn() -> int:
    return _TURN


def current_turn_message() -> str:
    """What the user said this turn ("" on agent-result turns)."""
    return _TURN_MESSAGE


def _guard_enabled() -> bool:
    return os.getenv("OPENPOKE_ROUTING_GUARD", "1") != "0"


_DANGLING_PRONOUN_RE = re.compile(r"\b(it|that|this|those|them)\b", re.I)


def _stem(token: str) -> str:
    return token[:-1] if len(token) > 3 and token.endswith("s") else token


def _ambiguous_after_fan_out(agent_name: str, message: str) -> Optional[list]:
    """The previous user turn asked several agents (a roster-wide question such as "what
    reminders do I have?"). If this message acts on "it" / "that" and names no thread by a
    word from any of those agents' names or descriptions, the referent is the user's choice,
    not the model's, and the marker for the thread touched last does not cover it. Returns the
    agents the user must choose between, or None. A follow-up without a dangling pronoun ("and
    then?", "what reminders do I have now?") is left to the ordinary rules."""
    if len(_PREVIOUS_USER_FAN_OUT) < 2 or not message or not _DANGLING_PRONOUN_RE.search(message):
        return None
    from ...services.execution.registry import tokenize

    words = {_stem(t) for t in tokenize(message)}
    registry = get_agent_registry()
    for name in set(_PREVIOUS_USER_FAN_OUT) | {agent_name}:
        record = registry.get(name)
        about = {_stem(t) for t in tokenize(name)}
        if record is not None:
            about |= {_stem(t) for t in tokenize(record.description)}
        if words & about:
            return None  # the message points at a thread by name or topic
    return _PREVIOUS_USER_FAN_OUT


# Create or reuse execution agent and dispatch instructions asynchronously
def send_message_to_agent(agent_name: str, instructions: str) -> ToolResult:
    """Send instructions to an execution agent."""
    registry = get_agent_registry()
    registry.load()
    duplicate = registry.find_duplicate(agent_name, instructions)
    if duplicate is not None:
        logger.info(f"Consolidated new agent '{agent_name}' into existing '{duplicate.name}'")
        agent_name = duplicate.name
    if _guard_enabled() and ("fan_out", _TURN) not in _GUARD_SEEN:
        choices = _ambiguous_after_fan_out(agent_name, current_turn_message())
        if choices:
            _GUARD_SEEN.add(("fan_out", _TURN))
            logger.info(f"Fan-out guard: bare follow-up after asking {choices}; asking the user which")
            return ToolResult(
                success=False,
                payload={
                    "error": (
                        "The previous message was answered by several agents ("
                        + "; ".join(choices)
                        + ") and this follow-up names none of them, so it is ambiguous. Ask the "
                        "user which one they mean, in a few words, instead of guessing. Repeat "
                        "this exact call to override."
                    ),
                    "ambiguous_after_fan_out": list(choices),
                },
            )
    if _guard_enabled() and agent_name in registry:
        key = (agent_name, _TURN)
        if key not in _GUARD_SEEN:
            anchor = current_turn_message() or None
            mismatch = (registry.new_agent_mismatch(agent_name, instructions, anchor_text=anchor)
                        if registry.get(agent_name) is None
                        else registry.routing_mismatch(agent_name, instructions, anchor_text=anchor))
            if mismatch is not None:
                _GUARD_SEEN.add(key)
                logger.info(f"Routing guard: '{agent_name}' is not about {mismatch.tokens}; "
                            f"suggesting {mismatch.candidates}")
                return ToolResult(
                    success=False,
                    payload={"error": mismatch.render(), "did_you_mean": mismatch.candidates},
                )
    is_new = registry.register(agent_name, instructions, reason="user")
    if agent_name not in _TURN_DISPATCHED:
        _TURN_DISPATCHED.append(agent_name)
    if len(_TURN_DISPATCHED) > 1:
        # A roster-wide question asked of several agents touches none of their threads.
        registry.mark_fan_out(_TURN_DISPATCHED)

    # Keep the legacy roster.json in sync so older tooling keeps working.
    roster = get_agent_roster()
    roster.load()
    if agent_name not in set(roster.get_agents()):
        roster.add_agent(agent_name)

    get_execution_agent_logs().record_request(agent_name, instructions)
    try:
        # Reminders and rules in the request are kept verbatim, outside compaction.
        get_agent_memory().note_request(agent_name, instructions)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"could not record commitments for {agent_name}: {exc}")

    action = "Created" if is_new else "Reused"
    logger.info(f"{action} agent: {agent_name}")
    # Outbound email is allowed in this run only if the user asked for it this turn.
    authorisation = authorise_send(current_turn_message(), last_reply_was_draft=_LAST_REPLY_WAS_DRAFT)

    async def _execute_async() -> None:
        try:
            result = await _EXECUTION_BATCH_MANAGER.execute_agent(agent_name, instructions, send_authorisation=authorisation)
            status = "SUCCESS" if result.success else "FAILED"
            logger.info(f"Agent '{agent_name}' completed: {status}")
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Agent '{agent_name}' failed: {str(exc)}")

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.error("No running event loop available for async execution")
        return ToolResult(success=False, payload={"error": "No event loop available"})

    loop.create_task(_execute_async())

    payload = {"status": "submitted", "agent_name": agent_name, "new_agent_created": is_new}
    if duplicate is not None:
        payload["consolidated_into_existing"] = True
    return ToolResult(success=True, payload=payload)


# Send immediate message to user and record in conversation history
def send_message_to_user(message: str) -> ToolResult:
    """Record a user-visible reply in the conversation log."""
    global _LAST_REPLY_WAS_DRAFT
    _LAST_REPLY_WAS_DRAFT = False
    log = get_conversation_log()
    log.record_reply(message)

    return ToolResult(
        success=True,
        payload={"status": "delivered"},
        user_message=message,
        recorded_reply=True,
    )


# Format and record email draft for user review
def send_draft(
    to: str,
    subject: str,
    body: str,
) -> ToolResult:
    """Record a draft update in the conversation log for the interaction agent."""
    global _LAST_REPLY_WAS_DRAFT
    _LAST_REPLY_WAS_DRAFT = True
    log = get_conversation_log()

    message = f"To: {to}\nSubject: {subject}\n\n{body}"

    log.record_reply(message)
    logger.info(f"Draft recorded for: {to}")

    return ToolResult(
        success=True,
        payload={
            "status": "draft_recorded",
            "to": to,
            "subject": subject,
        },
        recorded_reply=True,
    )


# Search the registry for agents that are not in the current budgeted roster
def search_agents(query: str, limit: int = 5) -> ToolResult:
    """Return existing agents matching the query, including dormant/archived ones."""
    registry = get_agent_registry()
    registry.load()
    try:
        limit = max(1, min(int(limit), 20))
    except (TypeError, ValueError):
        limit = 5
    results = registry.search(query, limit=limit, include_archived=True)
    payload = registry.render_search_results(results)
    logger.info(f"search_agents('{query}') -> {len(payload)} matches")
    return ToolResult(
        success=True,
        payload={"query": query, "matches": payload, "total_agents": len(registry)},
    )


# Record silent wait state to avoid duplicate responses
def wait(reason: str) -> ToolResult:
    """Wait silently and add a wait log entry that is not visible to the user."""
    log = get_conversation_log()
    
    # Record a dedicated wait entry so the UI knows to ignore it
    log.record_wait(reason)
    

    return ToolResult(
        success=True,
        payload={
            "status": "waiting",
            "reason": reason,
        },
        recorded_reply=True,
    )


# Return predefined tool schemas for LLM function calling
def get_tool_schemas():
    """Return OpenAI-compatible tool schemas."""
    return TOOL_SCHEMAS


# Route tool calls to appropriate handlers with argument validation and error handling
def handle_tool_call(name: str, arguments: Any) -> ToolResult:
    """Handle tool calls from interaction agent."""
    try:
        if isinstance(arguments, str):
            args = json.loads(arguments) if arguments.strip() else {}
        elif isinstance(arguments, dict):
            args = arguments
        else:
            return ToolResult(success=False, payload={"error": "Invalid arguments format"})

        if name == "send_message_to_agent":
            return send_message_to_agent(**args)
        if name == "send_message_to_user":
            return send_message_to_user(**args)
        if name == "send_draft":
            return send_draft(**args)
        if name == "search_agents":
            return search_agents(**args)
        if name == "wait":
            return wait(**args)

        logger.warning("unexpected tool", extra={"tool": name})
        return ToolResult(success=False, payload={"error": f"Unknown tool: {name}"})
    except json.JSONDecodeError:
        return ToolResult(success=False, payload={"error": "Invalid JSON"})
    except TypeError as exc:
        return ToolResult(success=False, payload={"error": f"Missing required arguments: {exc}"})
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("tool call failed", extra={"tool": name, "error": str(exc)})
        return ToolResult(success=False, payload={"error": "Failed to execute"})
