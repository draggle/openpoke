"""Single entry point for LLM calls: resolves the model by role, records usage.

Call sites pass a *role* instead of a model. The router picks the model, the call goes
through ``openrouter_client.request_chat_completion`` (the same function upstream used, so
tests and evals can monkeypatch it in one place), and the usage ledger records tokens and
estimated cost.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from ...openrouter_client import client as _openrouter
from .router import get_model_router
from .usage import UsageRecord, estimate_message_tokens, estimate_tokens, get_usage_ledger


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
    response = await _openrouter.request_chat_completion(
        model=resolved_model,
        messages=messages,
        system=system,
        api_key=api_key,
        tools=tools,
    )

    _record_usage(role, resolved_model, messages, system, tools, response, tag)
    return response


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
    if isinstance(usage, dict) and usage.get("prompt_tokens") is not None:
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
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
            cost_usd=price.cost(prompt_tokens, completion_tokens),
            estimated=estimated,
            tag=tag,
        )
    )


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
