"""Jev (TypeSafe "System One") client: typed decisions instead of generated text.

Several decisions in OpenPoke are classification, not generation - "is this email worth
interrupting the user for?", "which existing agent does this message belong to?". Today each
one costs a full chat-model call. Jev answers a map of typed questions about one ``state`` in
a single request and returns probabilities, which is both faster (parallel decoding) and lets
us threshold.

Request:   POST {base_url}/v1/systemone   Authorization: Bearer $TYPESAFE_API_KEY
           {"model": "jev-latest", "state": <str|obj>, "questions": {id: {type, instructions, criteria?}}}
Response:  {"answers": {id: {"noul": p} | {"choice": label, "probabilities": {...}, "confidence": c} | ...}}

The wire format was taken from TypeSafe's SDK docs; ``_parse_answer`` is deliberately tolerant
(it accepts a few plausible spellings) and is the one place to adjust if the live API differs.

Enable with ``OPENPOKE_JEV=1`` and ``TYPESAFE_API_KEY``. Tests and offline evals swap in a
``FakeJev`` via ``set_jev_client``. Every call is recorded in the usage ledger under its role.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Protocol, Union

import httpx

from ...logging_config import logger
from .usage import UsageRecord, estimate_tokens, get_usage_ledger

DEFAULT_BASE_URL = "https://api.typesafe.ai"
DEFAULT_MODEL = "jev-latest"
MAX_CHOICE_OPTIONS = 255
# USD per 1M input tokens (output is free). Illustrative list price; override with
# OPENPOKE_JEV_PRICE_PER_M.
DEFAULT_PRICE_PER_M = 0.0578

ROLE_JEV_EMAIL = "jev_email_importance"
ROLE_JEV_ROUTING = "jev_agent_routing"


class JevError(RuntimeError):
    pass


@dataclass
class Answer:
    kind: str  # noul | choice | score
    probability: Optional[float] = None  # noul: P(yes)
    choice: Optional[str] = None
    probabilities: Dict[str, float] = field(default_factory=dict)
    confidence: Optional[float] = None
    score: Optional[float] = None
    raw: Any = None


class JevLike(Protocol):
    async def ask(
        self, state: Union[str, Mapping[str, Any]], questions: Mapping[str, Mapping[str, Any]], *, role: str, tag: str = ""
    ) -> Dict[str, Answer]: ...


# ---- question builders -----------------------------------------------------------------


def noul(instructions: str, *, true_means: str = "", false_means: str = "") -> Dict[str, Any]:
    q: Dict[str, Any] = {"type": "noul", "instructions": instructions}
    if true_means or false_means:
        q["criteria"] = {"true": true_means, "false": false_means}
    return q


def choice(instructions: str, options: Mapping[str, str]) -> Dict[str, Any]:
    if not 1 <= len(options) <= MAX_CHOICE_OPTIONS:
        raise JevError(f"choice needs 1..{MAX_CHOICE_OPTIONS} options, got {len(options)}")
    return {"type": "choice", "instructions": instructions, "criteria": dict(options)}


# ---- parsing -----------------------------------------------------------------------------


def _parse_answer(raw: Any) -> Answer:
    if not isinstance(raw, dict):
        raise JevError(f"unexpected answer payload: {raw!r}")
    if "noul" in raw or "probability" in raw:
        p = raw.get("noul", raw.get("probability"))
        return Answer(kind="noul", probability=float(p), raw=raw)
    if "choice" in raw or "selected" in raw:
        label = raw.get("choice", raw.get("selected"))
        probs = raw.get("probabilities") or raw.get("distribution") or {}
        if isinstance(probs, list):  # tolerate [{"label":..,"probability":..}]
            probs = {str(p.get("label")): float(p.get("probability", 0)) for p in probs if isinstance(p, dict)}
        return Answer(kind="choice", choice=str(label), probabilities={str(k): float(v) for k, v in probs.items()},
                      confidence=_opt_float(raw.get("confidence")), raw=raw)
    if "score" in raw:
        return Answer(kind="score", score=float(raw["score"]), confidence=_opt_float(raw.get("confidence")), raw=raw)
    raise JevError(f"unrecognised answer shape: {list(raw)}")


def _opt_float(value: Any) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


# ---- client ------------------------------------------------------------------------------


class JevClient:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = 10.0,
        price_per_million: float = DEFAULT_PRICE_PER_M,
    ) -> None:
        self.api_key = (api_key or os.getenv("TYPESAFE_API_KEY") or "").strip()
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.price_per_million = price_per_million
        if not self.api_key:
            raise JevError("Missing TYPESAFE_API_KEY")

    async def ask(
        self, state: Union[str, Mapping[str, Any]], questions: Mapping[str, Mapping[str, Any]], *, role: str, tag: str = ""
    ) -> Dict[str, Answer]:
        payload = {"model": self.model, "state": state, "questions": dict(questions)}
        started = time.perf_counter()
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/v1/systemone",
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise JevError(f"Jev request failed ({exc.response.status_code}): {exc.response.text[:200]}") from exc
            except httpx.HTTPError as exc:
                raise JevError(f"Jev request failed: {exc}") from exc
        data = response.json()
        answers_raw = data.get("answers") if isinstance(data, dict) else None
        if not isinstance(answers_raw, dict):
            raise JevError(f"Jev response missing answers: {str(data)[:200]}")
        answers = {qid: _parse_answer(raw) for qid, raw in answers_raw.items()}
        self._record(role, payload, data, tag, time.perf_counter() - started)
        return answers

    def _record(self, role: str, payload: Dict[str, Any], data: Dict[str, Any], tag: str, elapsed: float) -> None:
        usage = data.get("usage") if isinstance(data, dict) else None
        if isinstance(usage, dict) and usage.get("input_tokens") is not None:
            tokens = int(usage["input_tokens"])
            estimated = False
        else:
            tokens = estimate_tokens(json.dumps(payload, default=str))
            estimated = True
        get_usage_ledger().record(
            UsageRecord(
                timestamp=time.time(), role=role, model=f"typesafe/{self.model}",
                prompt_tokens=tokens, completion_tokens=0,
                cost_usd=tokens * self.price_per_million / 1_000_000,
                estimated=estimated, tag=tag or f"{elapsed:.3f}s",
            )
        )


class FakeJev:
    """Scripted stand-in for tests/evals: ``answers_for(state, questions) -> {qid: Answer}``."""

    def __init__(self, answers_for) -> None:
        self._answers_for = answers_for
        self.calls: List[Dict[str, Any]] = []

    async def ask(self, state, questions, *, role: str, tag: str = "") -> Dict[str, Answer]:
        self.calls.append({"state": state, "questions": dict(questions), "role": role, "tag": tag})
        answers = self._answers_for(state, questions)
        tokens = estimate_tokens(json.dumps({"state": state, "questions": questions}, default=str))
        get_usage_ledger().record(
            UsageRecord(timestamp=time.time(), role=role, model="typesafe/jev-fake", prompt_tokens=tokens,
                        completion_tokens=0, cost_usd=tokens * DEFAULT_PRICE_PER_M / 1_000_000, estimated=True, tag=tag)
        )
        return answers


# ---- wiring ------------------------------------------------------------------------------

_client: Optional[JevLike] = None
_client_resolved = False


def jev_enabled() -> bool:
    return os.getenv("OPENPOKE_JEV", "0") == "1"


def get_jev_client() -> Optional[JevLike]:
    """Return the configured client, or None when Jev is disabled/unconfigured."""
    global _client, _client_resolved
    if _client is not None:
        return _client
    if _client_resolved or not jev_enabled():
        return None
    _client_resolved = True
    try:
        _client = JevClient(
            base_url=os.getenv("TYPESAFE_BASE_URL", DEFAULT_BASE_URL),
            model=os.getenv("TYPESAFE_MODEL", DEFAULT_MODEL),
            price_per_million=float(os.getenv("OPENPOKE_JEV_PRICE_PER_M", DEFAULT_PRICE_PER_M)),
        )
    except JevError as exc:
        logger.warning(f"Jev disabled: {exc}")
        _client = None
    return _client


def set_jev_client(client: Optional[JevLike]) -> None:
    global _client, _client_resolved
    _client = client
    _client_resolved = client is not None


__all__ = [
    "Answer", "FakeJev", "JevClient", "JevError", "JevLike", "ROLE_JEV_EMAIL", "ROLE_JEV_ROUTING",
    "choice", "get_jev_client", "jev_enabled", "noul", "set_jev_client",
]
