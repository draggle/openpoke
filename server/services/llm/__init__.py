"""LLM access: role-based routing, usage accounting, single call gateway."""

from .gateway import complete
from .jev import FakeJev, JevClient, JevError, get_jev_client, jev_enabled, set_jev_client
from .router import (
    ALL_ROLES,
    ROLE_EMAIL_CLASSIFIER,
    ROLE_EXECUTION,
    ROLE_EXECUTION_SEARCH,
    ROLE_INTERACTION,
    ROLE_MEMORY_COMPACTOR,
    ROLE_SUMMARIZER,
    ModelRouter,
    get_model_router,
    set_model_router,
)
from .usage import (
    UsageLedger,
    UsageRecord,
    estimate_message_tokens,
    estimate_tokens,
    get_usage_ledger,
    set_usage_ledger,
)

__all__ = [
    "ALL_ROLES",
    "ModelRouter",
    "ROLE_EMAIL_CLASSIFIER",
    "ROLE_EXECUTION",
    "ROLE_EXECUTION_SEARCH",
    "ROLE_INTERACTION",
    "ROLE_MEMORY_COMPACTOR",
    "ROLE_SUMMARIZER",
    "UsageLedger",
    "UsageRecord",
    "complete",
    "FakeJev",
    "JevClient",
    "JevError",
    "get_jev_client",
    "jev_enabled",
    "set_jev_client",
    "estimate_message_tokens",
    "estimate_tokens",
    "get_model_router",
    "get_usage_ledger",
    "set_model_router",
    "set_usage_ledger",
]
