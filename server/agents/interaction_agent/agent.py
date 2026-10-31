"""Interaction agent helpers for prompt construction."""

from html import escape
from pathlib import Path
from typing import Dict, List, Optional

from ...services.execution import get_agent_registry
from ...services.execution.registry import RosterSelection

_prompt_path = Path(__file__).parent / "system_prompt.md"
SYSTEM_PROMPT = _prompt_path.read_text(encoding="utf-8").strip()


# Load and return the pre-defined system prompt from markdown file
def build_system_prompt() -> str:
    """Return the static system prompt for the interaction agent."""
    return SYSTEM_PROMPT


# Build structured message with conversation history, active agents, and current turn
def prepare_message_with_history(
    latest_text: str,
    transcript: str,
    message_type: str = "user",
    routing_hint: Optional[str] = None,
    selection: Optional[RosterSelection] = None,
) -> List[Dict[str, str]]:
    """Compose a message that bundles history, roster, and the latest turn.

    ``selection`` is the roster already chosen for this turn (see ``select_roster``); when
    omitted it is computed here, so older callers and tests keep working.
    """
    sections: List[str] = []

    sections.append(_render_conversation_history(transcript))
    sections.append(f"<active_agents>\n{_render_active_agents(latest_text, selection)}\n</active_agents>")
    if routing_hint:
        sections.append(routing_hint)
    sections.append(_render_current_turn(latest_text, message_type))

    content = "\n\n".join(sections)
    return [{"role": "user", "content": content}]


# Format conversation transcript into XML tags for LLM context
def _render_conversation_history(transcript: str) -> str:
    history = transcript.strip()
    if not history:
        history = "None"
    return f"<conversation_history>\n{history}\n</conversation_history>"


# Choose the roster for this turn: one registry load + selection, reused by the prompt and
# by the routing hint.
def select_roster(latest_text: str, transcript: str = "") -> RosterSelection:
    registry = get_agent_registry()
    registry.load()
    return registry.select(latest_text)


# Render the budgeted roster of execution agents relevant to this turn.
def _render_active_agents(query: str = "", selection: Optional[RosterSelection] = None) -> str:
    """Select and render execution agents for the Interaction Agent.

    Upstream rendered every agent name ever created. The registry now picks a bounded set
    (recent + relevant to ``query``) and annotates each with status and a one-line summary.
    """
    registry = get_agent_registry()
    if selection is None:
        selection = select_roster(query)
    return registry.render(selection)


# Ask Jev (when enabled) which shown agent the user's message belongs to.
async def build_routing_hint(latest_text: str, selection: Optional[RosterSelection] = None) -> Optional[str]:
    from ...services.execution.routing_hint import compute_routing_hint

    registry = get_agent_registry()
    if selection is None:
        selection = select_roster(latest_text)
    if selection.strategy != "budgeted":
        return None
    hint = await compute_routing_hint(registry, selection, latest_text)
    return hint.render() if hint else None


# Wrap the current message in appropriate XML tags based on sender type
def _render_current_turn(latest_text: str, message_type: str) -> str:
    tag = "new_agent_message" if message_type == "agent" else "new_user_message"
    body = latest_text.strip()
    return f"<{tag}>\n{body}\n</{tag}>"
