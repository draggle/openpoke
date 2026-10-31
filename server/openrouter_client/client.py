from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import os

import httpx

from ..config import get_settings

# Any OpenAI-compatible chat-completions endpoint works here (OpenRouter by default;
# e.g. OPENROUTER_BASE_URL=https://ai-gateway.vercel.sh/v1 for Vercel AI Gateway).
OpenRouterBaseURL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")


class OpenRouterError(RuntimeError):
    """Raised when the OpenRouter API returns an error response."""


def _headers(*, api_key: Optional[str] = None) -> Dict[str, str]:
    settings = get_settings()
    key = (api_key or settings.openrouter_api_key or "").strip()
    if not key:
        raise OpenRouterError("Missing OpenRouter API key")

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    return headers


def _build_messages(messages: List[Dict[str, Any]], system: Any) -> List[Dict[str, Any]]:
    # ``system`` may be a plain string or a list of content blocks (used for prompt caching).
    if system:
        return [{"role": "system", "content": system}, *messages]
    return messages


def _handle_response_error(exc: httpx.HTTPStatusError) -> None:
    response = exc.response
    detail: str
    try:
        payload = response.json()
        detail = payload.get("error") or payload.get("message") or json.dumps(payload)
    except Exception:
        detail = response.text
    raise OpenRouterError(f"OpenRouter request failed ({response.status_code}): {detail}") from exc


async def request_chat_completion(
    *,
    model: str,
    messages: List[Dict[str, Any]],
    system: Any = None,
    api_key: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    base_url: str = OpenRouterBaseURL,
) -> Dict[str, Any]:
    """Request a chat completion and return the raw JSON payload."""

    if os.getenv("OPENPOKE_LLM_API", "openai").lower() == "anthropic":
        # Native Messages API (prompt caching, exact cache accounting). Same return shape.
        from .anthropic_native import request_messages

        return await request_messages(
            model=model, messages=messages, system=system, api_key=api_key, tools=tools
        )

    payload: Dict[str, object] = {
        "model": model,
        "messages": _build_messages(messages, system),
        "stream": False,
    }
    if tools:
        payload["tools"] = tools

    url = f"{base_url.rstrip('/')}/chat/completions"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                url,
                headers=_headers(api_key=api_key),
                json=payload,
                timeout=60.0,  # Set reasonable timeout instead of None
            )
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                _handle_response_error(exc)
            return response.json()
        except httpx.HTTPStatusError as exc:  # pragma: no cover - handled above
            _handle_response_error(exc)
        except httpx.HTTPError as exc:
            raise OpenRouterError(f"OpenRouter request failed: {exc}") from exc

    raise OpenRouterError("OpenRouter request failed: unknown error")


__all__ = ["OpenRouterError", "request_chat_completion", "OpenRouterBaseURL"]
