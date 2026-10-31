"""Role-based model routing.

Upstream sends every call - the user-facing Interaction Agent, the execution agents, the
email classifier, the summariser - to the same strong model. Most of those calls are
mechanical (classify an email, compress a log) and do fine on a cheap model.

Rather than a learned router (RouteLLM-style), we route by *call role*, which is known at
the call site and is the dominant predictor of difficulty here. Each role maps to a tier,
each tier maps to a model, and both are overridable via environment variables so the eval
can compare configurations without code changes.

Environment:
    OPENPOKE_STRONG_MODEL            default anthropic/claude-sonnet-4
    OPENPOKE_CHEAP_MODEL             default anthropic/claude-3.5-haiku
    OPENPOKE_ROUTING                 "tiered" (default) or "strong_only" (upstream behaviour)
    OPENPOKE_TIER_<ROLE>             override a single role: "strong" or "cheap"
    OPENPOKE_MODEL_PRICES            JSON {model: [usd_per_1M_input, usd_per_1M_output]}
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Mapping, Optional

# Roles, one per call site. Keep in sync with gateway callers.
ROLE_INTERACTION = "interaction"
ROLE_EXECUTION = "execution"
ROLE_EXECUTION_SEARCH = "execution_search"
ROLE_SUMMARIZER = "summarizer"
ROLE_EMAIL_CLASSIFIER = "email_classifier"
ROLE_MEMORY_COMPACTOR = "memory_compactor"
ROLE_EXECUTION_TRIGGER = "execution_trigger"  # execution agent woken by a scheduled trigger

ALL_ROLES = (
    ROLE_INTERACTION,
    ROLE_EXECUTION,
    ROLE_EXECUTION_SEARCH,
    ROLE_SUMMARIZER,
    ROLE_EMAIL_CLASSIFIER,
    ROLE_MEMORY_COMPACTOR,
    ROLE_EXECUTION_TRIGGER,
)

# Default tier per role. Rationale:
#   interaction      - user-facing voice + delegation decisions: strong.
#   execution        - drafts emails the user will send under their name: strong.
#   execution_search - query generation + aggregation over inbox: cheap works well.
#   summarizer       - compress a transcript: cheap.
#   email_classifier - binary importance call on every inbound email, high volume: cheap.
#   memory_compactor - structured extraction from an agent log: cheap.
#   execution_trigger - agent woken by a reminder; defaults strong (it may draft mail), set
#                       OPENPOKE_TIER_EXECUTION_TRIGGER=cheap to route reminder wake-ups cheaply.
DEFAULT_TIERS: Dict[str, str] = {
    ROLE_INTERACTION: "strong",
    ROLE_EXECUTION: "strong",
    ROLE_EXECUTION_SEARCH: "cheap",
    ROLE_SUMMARIZER: "cheap",
    ROLE_EMAIL_CLASSIFIER: "cheap",
    ROLE_MEMORY_COMPACTOR: "cheap",
    ROLE_EXECUTION_TRIGGER: "strong",
}

DEFAULT_STRONG_MODEL = "anthropic/claude-sonnet-4"
DEFAULT_CHEAP_MODEL = "anthropic/claude-3.5-haiku"

# USD per 1M tokens (input, output). Illustrative list prices; override with
# OPENPOKE_MODEL_PRICES for your account. Unknown models fall back to the strong price so
# cost estimates err on the pessimistic side.
DEFAULT_PRICES: Dict[str, tuple[float, float]] = {
    "anthropic/claude-sonnet-4": (3.00, 15.00),
    "anthropic/claude-3.5-haiku": (0.80, 4.00),
    "anthropic/claude-3-haiku": (0.25, 1.25),
}


@dataclass(frozen=True)
class ModelPrice:
    input_per_million: float
    output_per_million: float

    CACHE_READ_DISCOUNT = 0.10  # Anthropic bills cache reads at 10% of the input price

    def cost(self, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0) -> float:
        cached = min(cached_tokens, prompt_tokens)
        uncached = prompt_tokens - cached
        return (
            uncached * self.input_per_million
            + cached * self.input_per_million * self.CACHE_READ_DISCOUNT
            + completion_tokens * self.output_per_million
        ) / 1_000_000


class ModelRouter:
    """Resolve a call role to a concrete model and its price."""

    def __init__(
        self,
        *,
        strong_model: str = DEFAULT_STRONG_MODEL,
        cheap_model: str = DEFAULT_CHEAP_MODEL,
        tiers: Optional[Mapping[str, str]] = None,
        mode: str = "tiered",
        prices: Optional[Mapping[str, tuple[float, float]]] = None,
    ) -> None:
        self.strong_model = strong_model
        self.cheap_model = cheap_model
        self.mode = mode if mode in {"tiered", "strong_only"} else "tiered"
        self.tiers: Dict[str, str] = dict(DEFAULT_TIERS)
        if tiers:
            for role, tier in tiers.items():
                if tier in {"strong", "cheap"}:
                    self.tiers[role] = tier
        self.prices: Dict[str, tuple[float, float]] = dict(DEFAULT_PRICES)
        if prices:
            self.prices.update(prices)

    @classmethod
    def from_env(cls) -> "ModelRouter":
        tiers: Dict[str, str] = {}
        for role in ALL_ROLES:
            value = os.getenv(f"OPENPOKE_TIER_{role.upper()}")
            if value:
                tiers[role] = value.strip().lower()
        prices: Dict[str, tuple[float, float]] = {}
        raw_prices = os.getenv("OPENPOKE_MODEL_PRICES")
        if raw_prices:
            try:
                for model, pair in json.loads(raw_prices).items():
                    prices[str(model)] = (float(pair[0]), float(pair[1]))
            except Exception:
                pass
        return cls(
            strong_model=os.getenv("OPENPOKE_STRONG_MODEL", DEFAULT_STRONG_MODEL),
            cheap_model=os.getenv("OPENPOKE_CHEAP_MODEL", DEFAULT_CHEAP_MODEL),
            tiers=tiers,
            mode=os.getenv("OPENPOKE_ROUTING", "tiered").strip().lower(),
            prices=prices,
        )

    def tier_for(self, role: str) -> str:
        if self.mode == "strong_only":
            return "strong"
        return self.tiers.get(role, "strong")

    def model_for(self, role: str) -> str:
        return self.strong_model if self.tier_for(role) == "strong" else self.cheap_model

    def price_for(self, model: str) -> ModelPrice:
        pair = self.prices.get(model) or self.prices.get(self.strong_model) or (3.0, 15.0)
        return ModelPrice(*pair)


_router: Optional[ModelRouter] = None


def get_model_router() -> ModelRouter:
    global _router
    if _router is None:
        _router = ModelRouter.from_env()
    return _router


def set_model_router(router: Optional[ModelRouter]) -> None:
    """Swap the process-wide router (tests / evals)."""
    global _router
    _router = router


__all__ = [
    "ALL_ROLES",
    "DEFAULT_TIERS",
    "ModelPrice",
    "ModelRouter",
    "ROLE_EMAIL_CLASSIFIER",
    "ROLE_EXECUTION",
    "ROLE_EXECUTION_SEARCH",
    "ROLE_INTERACTION",
    "ROLE_MEMORY_COMPACTOR",
    "ROLE_EXECUTION_TRIGGER",
    "ROLE_SUMMARIZER",
    "get_model_router",
    "set_model_router",
]
