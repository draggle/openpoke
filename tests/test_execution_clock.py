"""The execution agent is told what day it is.

Hand use found "Friday at 8am" stored as Saturday and "Monday at noon" as Tuesday: the agent
received the instruction with no current date, so it guessed the weekday arithmetic.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from server.agents.execution_agent import agent as agent_module


@pytest.fixture
def frozen_clock(monkeypatch):
    fixed = datetime(2026, 11, 2, 1, 15, tzinfo=ZoneInfo("America/Toronto"))
    monkeypatch.setattr(agent_module, "now_in_user_timezone", lambda fmt=None, default="UTC": fixed if fmt is None else fixed.strftime(fmt))
    monkeypatch.setattr(agent_module, "get_user_timezone_name", lambda default="UTC": "America/Toronto")
    return fixed


def test_clock_line_names_weekday_date_time_and_zone(frozen_clock):
    line = agent_module.clock_line()
    assert line.startswith("Current date and time: Monday 2026-11-02 01:15 EST (America/Toronto)")
    assert "weekday" in line.lower() and "ISO 8601" in line


def test_instruction_message_starts_with_the_clock(frozen_clock):
    agent = agent_module.ExecutionAgent(name="Dentist reminder")
    messages = agent.build_messages_for_llm("Remind me Friday at 8am to call the dentist")
    assert messages[0]["role"] == "user"
    assert messages[0]["content"].startswith("Current date and time: Monday 2026-11-02")
    assert messages[0]["content"].endswith("Remind me Friday at 8am to call the dentist")


@pytest.mark.asyncio
async def test_runtime_sends_the_clock_with_the_instruction(monkeypatch, frozen_clock, tmp_path):
    from server.agents.execution_agent import runtime as rt
    from server.openrouter_client import client as oc

    seen = {}

    async def scripted(*, model, messages, system=None, api_key=None, tools=None, base_url=None):
        seen["messages"] = messages
        return {"choices": [{"message": {"content": "done <user_summary>done</user_summary>"}}]}

    monkeypatch.setattr(oc, "request_chat_completion", scripted)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    runtime = rt.ExecutionAgentRuntime(agent_name="Dentist reminder")
    await runtime.execute("Remind me Friday at 8am to call the dentist")
    assert seen["messages"][0]["content"].startswith("Current date and time: Monday 2026-11-02 01:15 EST")


def test_execution_prompt_says_a_new_reminder_creates_rather_than_moves():
    prompt = agent_module.ExecutionAgent(name="x").build_system_prompt()
    line = next(l for l in prompt.splitlines() if "new reminder" in l.lower())
    assert "create" in line.lower() and "do not move" in line.lower()


def test_trigger_tool_schema_asks_for_an_offset_and_points_at_the_clock():
    from server.agents.execution_agent.tools import triggers

    schemas = triggers.get_schemas()
    create = next(s for s in schemas if s["function"]["name"] == "createTrigger")
    desc = create["function"]["parameters"]["properties"]["start_time"]["description"]
    assert "offset" in desc and "current date" in desc.lower()
