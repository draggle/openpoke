"""Conversion between OpenAI-style requests/responses and Anthropic's Messages API."""

from __future__ import annotations

import json

from server.openrouter_client.anthropic_native import (
    convert_messages, convert_response, convert_system, convert_tools,
)


def test_tools_convert_to_input_schema():
    tools = [{"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object", "properties": {"x": {"type": "string"}}}}}]
    [t] = convert_tools(tools)
    assert t == {"name": "f", "description": "d", "input_schema": {"type": "object", "properties": {"x": {"type": "string"}}}}
    assert convert_tools(None) is None


def test_system_passthrough_and_cache_block():
    assert convert_system("hi") == "hi"
    blocks = convert_system([{"type": "text", "text": "s", "cache_control": {"type": "ephemeral"}}])
    assert blocks == [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral"}}]


def test_messages_tool_loop_round_trip():
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "search_agents", "arguments": json.dumps({"query": "priya"})}},
            {"id": "c2", "type": "function", "function": {"name": "wait", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": "{\"matches\": []}"},
        {"role": "tool", "tool_call_id": "c2", "content": "ok"},
        {"role": "assistant", "content": "done"},
    ]
    out = convert_messages(msgs)
    assert [m["role"] for m in out] == ["user", "assistant", "user", "assistant"]
    assert out[1]["content"][0] == {"type": "tool_use", "id": "c1", "name": "search_agents", "input": {"query": "priya"}}
    assert out[1]["content"][1]["input"] == {}
    # two tool results merged into one user turn
    assert [b["type"] for b in out[2]["content"]] == ["tool_result", "tool_result"]
    assert out[2]["content"][0]["tool_use_id"] == "c1"


def test_response_conversion_with_cache_usage():
    data = {
        "id": "msg_1", "model": "claude-sonnet-4-5", "stop_reason": "tool_use",
        "content": [{"type": "text", "text": "Let me check."},
                    {"type": "tool_use", "id": "toolu_1", "name": "send_message_to_agent", "input": {"agent_name": "A", "instructions": "x"}}],
        "usage": {"input_tokens": 500, "output_tokens": 40, "cache_read_input_tokens": 2800, "cache_creation_input_tokens": 0},
    }
    out = convert_response(data)
    msg = out["choices"][0]["message"]
    assert msg["content"] == "Let me check."
    assert msg["tool_calls"][0]["function"]["name"] == "send_message_to_agent"
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"])["agent_name"] == "A"
    assert out["usage"]["prompt_tokens"] == 3300
    assert out["usage"]["prompt_tokens_details"]["cached_tokens"] == 2800


async def test_client_dispatches_to_native_when_selected(monkeypatch):
    from server.openrouter_client import client as oc
    from server.openrouter_client import anthropic_native as native
    seen = {}

    async def fake(**kwargs):
        seen.update(kwargs)
        return {"choices": [{"message": {"content": "native"}}]}

    monkeypatch.setattr(native, "request_messages", fake)
    monkeypatch.setenv("OPENPOKE_LLM_API", "anthropic")
    out = await oc.request_chat_completion(model="m", messages=[{"role": "user", "content": "hi"}], system="s", api_key="k")
    assert out["choices"][0]["message"]["content"] == "native" and seen["model"] == "m"


async def test_native_prefers_anthropic_key_env(monkeypatch):
    """Callers pass the OpenRouter key; in native mode ANTHROPIC_API_KEY must win."""
    import httpx
    from server.openrouter_client import anthropic_native as native

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-real")
    captured = {}

    class FakeResponse:
        def raise_for_status(self): pass
        def json(self): return {"content": [{"type": "text", "text": "ok"}], "usage": {"input_tokens": 1, "output_tokens": 1}}

    class FakeClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, *, headers, json, timeout):
            captured["headers"] = headers
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", lambda: FakeClient())
    await native.request_messages(model="m", messages=[{"role": "user", "content": "hi"}], api_key="sk-or-openrouter")
    assert captured["headers"]["x-api-key"] == "sk-ant-real"
