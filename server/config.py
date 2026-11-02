"""Simplified configuration management."""

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field


def _load_env_file() -> None:
    """Load .env from root directory if present."""
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.is_file():
        return
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                key, value = stripped.split("=", 1)
                key, value = key.strip(), value.strip().strip("'\"")
                if key and value and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        pass


ANTHROPIC_NATIVE_BASE_URL = "https://api.anthropic.com/v1"
ANTHROPIC_NATIVE_STRONG_MODEL = "claude-sonnet-4-5"
ANTHROPIC_NATIVE_CHEAP_MODEL = "claude-haiku-4-5"


def apply_provider_defaults() -> None:
    """An ``ANTHROPIC_API_KEY`` on its own is enough: route to the native Messages API.

    Upstream requires ``OPENROUTER_API_KEY`` before any model call and reports its absence as a
    configuration error, so a ``.env`` holding only an Anthropic key failed at the first message.
    When no OpenAI-compatible key is set, the Anthropic key stands in for it and the base URL,
    API flavour and model ids default to Anthropic's own. Anything set explicitly is kept.
    """
    if os.environ.get("OPENROUTER_API_KEY") or not os.environ.get("ANTHROPIC_API_KEY"):
        return
    os.environ["OPENROUTER_API_KEY"] = os.environ["ANTHROPIC_API_KEY"]
    os.environ.setdefault("OPENROUTER_BASE_URL", ANTHROPIC_NATIVE_BASE_URL)
    os.environ.setdefault("OPENPOKE_LLM_API", "anthropic")
    os.environ.setdefault("OPENPOKE_STRONG_MODEL", ANTHROPIC_NATIVE_STRONG_MODEL)
    os.environ.setdefault("OPENPOKE_CHEAP_MODEL", ANTHROPIC_NATIVE_CHEAP_MODEL)


_load_env_file()
apply_provider_defaults()


DEFAULT_APP_NAME = "OpenPoke Server"
DEFAULT_APP_VERSION = "0.3.0"


def _env_int(name: str, fallback: int) -> int:
    try:
        return int(os.getenv(name, str(fallback)))
    except (TypeError, ValueError):
        return fallback


class Settings(BaseModel):
    """Application settings with lightweight env fallbacks."""

    # App metadata
    app_name: str = Field(default=DEFAULT_APP_NAME)
    app_version: str = Field(default=DEFAULT_APP_VERSION)

    # Server runtime
    server_host: str = Field(default=os.getenv("OPENPOKE_HOST", "0.0.0.0"))
    server_port: int = Field(default=_env_int("OPENPOKE_PORT", 8001))

    # LLM model selection now lives in services/llm/router.py (role -> tier -> model), driven by
    # OPENPOKE_STRONG_MODEL / OPENPOKE_CHEAP_MODEL / OPENPOKE_TIER_<ROLE>. The fields below are
    # kept only for backward compatibility with upstream call sites and are not consulted.
    interaction_agent_model: str = Field(default="anthropic/claude-sonnet-4")
    execution_agent_model: str = Field(default="anthropic/claude-sonnet-4")
    execution_agent_search_model: str = Field(default="anthropic/claude-sonnet-4")
    summarizer_model: str = Field(default="anthropic/claude-sonnet-4")
    email_classifier_model: str = Field(default="anthropic/claude-sonnet-4")

    # Credentials / integrations
    openrouter_api_key: Optional[str] = Field(default=os.getenv("OPENROUTER_API_KEY"))
    composio_gmail_auth_config_id: Optional[str] = Field(default=os.getenv("COMPOSIO_GMAIL_AUTH_CONFIG_ID"))
    composio_api_key: Optional[str] = Field(default=os.getenv("COMPOSIO_API_KEY"))

    # HTTP behaviour
    cors_allow_origins_raw: str = Field(default=os.getenv("OPENPOKE_CORS_ALLOW_ORIGINS", "*"))
    enable_docs: bool = Field(default=os.getenv("OPENPOKE_ENABLE_DOCS", "1") != "0")
    docs_url: Optional[str] = Field(default=os.getenv("OPENPOKE_DOCS_URL", "/docs"))

    # Summarisation controls
    conversation_summary_threshold: int = Field(default=100)
    conversation_summary_tail_size: int = Field(default=10)

    @property
    def cors_allow_origins(self) -> List[str]:
        """Parse CORS origins from comma-separated string."""
        if self.cors_allow_origins_raw.strip() in {"", "*"}:
            return ["*"]
        return [origin.strip() for origin in self.cors_allow_origins_raw.split(",") if origin.strip()]

    @property
    def resolved_docs_url(self) -> Optional[str]:
        """Return documentation URL when docs are enabled."""
        return (self.docs_url or "/docs") if self.enable_docs else None

    @property
    def summarization_enabled(self) -> bool:
        """Flag indicating conversation summarisation is active."""
        return self.conversation_summary_threshold > 0


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
