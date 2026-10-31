"""Single entry point for LLM calls: resolves the model by role, records usage.

Call sites pass a *role* instead of a model. The router picks the model, the call goes
through ``openrouter_client.request_chat_completion`` (the same function upstream used, so
tests and evals can monkeypatch it in one place), and the usage ledger records tokens and
estimated cost.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Optional

from ...logging_config import logger
from ...openrouter_client import client as _openrouter
from .router import get_model_router
from .usage import UsageRecord, estimate_message_tokens, estimate_tokens, get_usage_ledger

# Prompt caching: the Interaction Agent's ~3k-token system prompt and the tool schemas are
# identical on every call, so we mark the system prompt as a cacheable block (Anthropic
# ``cache_control``, also honoured by OpenRouter). If the endpoint rejects block-formatted
# system content we fall back to a plain string and stop trying for the process lifetime.
_CACHE_SUPPORTED = True
_MIN_CACHE_CHARS = 2048  # Anthropic ignores cache markers on prefixes shorter than ~1k tokens


def _cache_enabled(model: str, system: Optional[str]) -> bool:
    if os.getenv("OPENPOKE_PROMPT_CACHE", "1") == "0" or not _CACHE_SUPPORTED:
        return False
    return bool(system) and len(system) >= _MIN_CACHE_CHARS and "claude" in model.lower()


def _cached_system(system: str) -> List[Dict[str, Any]]:
    return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]


async def complete(
    *,
    role: str,
    messages: List[Dict[str, Any]],
    system: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    tag: str = "",
) -> Dict[str, Any]:
    """Run a chat completion for ``role``; ``model`` overrides routing when given."""

    router = get_model_router()
    resolved_model = model or router.model_for(role)

    # Resolve the function at call time so monkeypatching ``request_chat_completion`` on
    # the client module (what tests/evals do) is honoured.
    global _CACHE_SUPPORTED
    system_payload: Any = _cached_system(system) if _cache_enabled(resolved_model, system) else system
    try:
        response = await _openrouter.request_chat_completion(
            model=resolved_model,
            messages=messages,
            system=system_payload,
            api_key=api_key,
            tools=tools,
        )
    except Exception as exc:
        if system_payload is not system and _is_cache_rejection(exc):
            logger.warning(f"prompt caching not supported by endpoint; disabling ({str(exc)[:120]})")
            _CACHE_SUPPORTED = False
            response = await _openrouter.request_chat_completion(
                model=resolved_model, messages=messages, system=system, api_key=api_key, tools=tools
            )
        else:
            raise

    _record_usage(role, resolved_model, messages, system, tools, response, tag)
    return response


_CACHE_REJECTION_RE = re.compile(
    r"cache_control|system.*(content|array|list|block)|content\[\d+\]|invalid.*system", re.I
)


def _is_cache_rejection(exc: Exception) -> bool:
    """True only for an error about the block-formatted system prompt, not any 400."""
    text = str(exc)
    return "400" in text and bool(_CACHE_REJECTION_RE.search(text))


def _record_usage(
    role: str,
    model: str,
    messages: List[Dict[str, Any]],
    system: Optional[str],
    tools: Optional[List[Dict[str, Any]]],
    response: Dict[str, Any],
    tag: str,
) -> None:
    usage = response.get("usage") if isinstance(response, dict) else None
    estimated = False
    cached_tokens = 0
    cache_write_tokens = 0
    if isinstance(usage, dict) and usage.get("prompt_tokens") is not None:
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        cached_tokens = _cached_tokens(usage)
        cache_write_tokens = int(usage.get("cache_creation_input_tokens") or 0)
    else:
        estimated = True
        prompt_tokens = estimate_message_tokens(messages, system)
        if tools:
            import json

            prompt_tokens += estimate_tokens(json.dumps(tools))
        completion_tokens = _estimate_completion(response)

    price = get_model_router().price_for(model)
    get_usage_ledger().record(
        UsageRecord(
            timestamp=time.time(),
            role=role,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=price.cost(prompt_tokens, completion_tokens, cached_tokens, cache_write_tokens),
            estimated=estimated,
            tag=tag,
            cached_tokens=cached_tokens,
            cache_write_tokens=cache_write_tokens,
        )
    )


def _cached_tokens(usage: Dict[str, Any]) -> int:
    """Cache-read tokens in either OpenAI (prompt_tokens_details.cached_tokens) or Anthropic
    (cache_read_input_tokens) spelling."""
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict) and details.get("cached_tokens") is not None:
        return int(details.get("cached_tokens") or 0)
    return int(usage.get("cache_read_input_tokens") or 0)


def _estimate_completion(response: Dict[str, Any]) -> int:
    try:
        message = (response.get("choices") or [{}])[0].get("message") or {}
    except Exception:
        return 1
    total = estimate_tokens(message.get("content") or "")
    tool_calls = message.get("tool_calls")
    if tool_calls:
        import json

        total += estimate_tokens(json.dumps(tool_calls, default=str))
    return total


__all__ = ["complete"]
