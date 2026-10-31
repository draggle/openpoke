"""Native Anthropic Messages API adapter with the same call/return shape as the OpenAI path.

Anthropic's OpenAI-compatible endpoint ignores ``cache_control`` (prompt caching is only on
the native API). This adapter converts OpenAI-style messages/tools into Messages-API form,
sends the request, and converts the response back into the ``{"choices": [...], "usage": ...}``
shape the runtimes already parse — so the rest of the code base does not change.

Selected by ``OPENPOKE_LLM_API=anthropic`` (set automatically by ``evals/run_live.sh`` when an
Anthropic key is used). Usage reports ``cache_read_input_tokens`` so the ledger can bill cache
reads at 10% and the eval can confirm caching actually happened.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import httpx

ANTHROPIC_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 4096


class AnthropicError(RuntimeError):
    pass


# --------------------------------------------------------------------------------------
# Request conversion
# --------------------------------------------------------------------------------------


def convert_tools(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    if not tools:
        return None
    out = []
    for tool in tools:
        fn = tool.get("function") or {}
        out.append(
            {
                "name": fn.get("name"),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
            }
        )
    return out


def convert_system(system: Any) -> Any:
    """str -> str; OpenAI-style text blocks (possibly with cache_control) pass through."""
    if not system:
        return None
    if isinstance(system, str):
        return system
    blocks = []
    for block in system:
        if isinstance(block, dict) and block.get("type") == "text":
            item: Dict[str, Any] = {"type": "text", "text": block.get("text", "")}
            if block.get("cache_control"):
                item["cache_control"] = block["cache_control"]
            blocks.append(item)
    return blocks or None


def convert_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """OpenAI chat messages -> Anthropic messages.

    - assistant ``tool_calls`` become ``tool_use`` blocks
    - ``tool`` role messages become ``tool_result`` blocks inside a user message; consecutive
      tool results are merged into one user message (Anthropic requires alternation)
    - system messages are expected to be passed separately; any found here are hoisted as text
    """
    out: List[Dict[str, Any]] = []

    def push(role: str, blocks: List[Dict[str, Any]]) -> None:
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": list(blocks)})

    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role == "tool":
            push("user", [{
                "type": "tool_result",
                "tool_use_id": message.get("tool_call_id") or "tool",
                "content": content if isinstance(content, str) else json.dumps(content, default=str),
            }])
            continue
        if role == "assistant":
            blocks: List[Dict[str, Any]] = []
            if isinstance(content, str) and content.strip():
                blocks.append({"type": "text", "text": content})
            for call in message.get("tool_calls") or []:
                fn = call.get("function") or {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except json.JSONDecodeError:
                        args = {"_raw": args}
                blocks.append({"type": "tool_use", "id": call.get("id") or "call", "name": fn.get("name"), "input": args or {}})
            if blocks:
                push("assistant", blocks)
            continue
        # user (and any stray system) messages
        text = content if isinstance(content, str) else json.dumps(content, default=str)
        push("user", [{"type": "text", "text": text}])

    # Anthropic requires the first message to be from the user.
    if out and out[0]["role"] != "user":
        out.insert(0, {"role": "user", "content": [{"type": "text", "text": "(continue)"}]})
    return out


# --------------------------------------------------------------------------------------
# Response conversion
# --------------------------------------------------------------------------------------


def convert_response(data: Dict[str, Any]) -> Dict[str, Any]:
    text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    for block in data.get("content") or []:
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            tool_calls.append({
                "id": block.get("id"),
                "type": "function",
                "function": {"name": block.get("name"), "arguments": json.dumps(block.get("input") or {})},
            })
    message: Dict[str, Any] = {"role": "assistant", "content": "".join(text_parts)}
    if tool_calls:
        message["tool_calls"] = tool_calls

    usage = data.get("usage") or {}
    cache_read = int(usage.get("cache_read_input_tokens") or 0)
    cache_write = int(usage.get("cache_creation_input_tokens") or 0)
    # Anthropic's input_tokens excludes cached tokens; OpenAI's prompt_tokens is the total.
    prompt_tokens = int(usage.get("input_tokens") or 0) + cache_read + cache_write
    return {
        "id": data.get("id"),
        "model": data.get("model"),
        "choices": [{"index": 0, "message": message, "finish_reason": data.get("stop_reason")}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": int(usage.get("output_tokens") or 0),
            "cache_read_input_tokens": cache_read,
            "cache_creation_input_tokens": cache_write,
            "prompt_tokens_details": {"cached_tokens": cache_read},
        },
    }


# --------------------------------------------------------------------------------------
# Call
# --------------------------------------------------------------------------------------


async def request_messages(
    *,
    model: str,
    messages: List[Dict[str, Any]],
    system: Any = None,
    api_key: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    base_url: Optional[str] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Dict[str, Any]:
    # Callers pass settings.openrouter_api_key; in native mode the Anthropic key must win,
    # otherwise a .env with both keys sends the OpenRouter key to Anthropic and gets 401s.
    key = (os.getenv("ANTHROPIC_API_KEY") or api_key or os.getenv("OPENROUTER_API_KEY") or "").strip()
    if not key:
        raise AnthropicError("Missing Anthropic API key")
    payload: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": convert_messages(messages),
    }
    converted_system = convert_system(system)
    if converted_system:
        payload["system"] = converted_system
    converted_tools = convert_tools(tools)
    if converted_tools:
        payload["tools"] = converted_tools

    url = f"{(base_url or os.getenv('ANTHROPIC_BASE_URL') or ANTHROPIC_BASE_URL).rstrip('/')}/v1/messages"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                url,
                headers={"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION, "content-type": "application/json"},
                json=payload,
                timeout=60.0,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise AnthropicError(f"Anthropic request failed ({exc.response.status_code}): {exc.response.text[:300]}") from exc
        except httpx.HTTPError as exc:
            raise AnthropicError(f"Anthropic request failed: {exc}") from exc
    return convert_response(response.json())


__all__ = ["AnthropicError", "convert_messages", "convert_response", "convert_system", "convert_tools", "request_messages"]
