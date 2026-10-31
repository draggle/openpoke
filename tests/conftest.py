"""Shared pytest configuration.

Tests never talk to OpenRouter or Gmail. A dummy API key keeps the runtimes constructible,
and every LLM call is mocked at the single chokepoint ``request_chat_completion``.
"""

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("OPENROUTER_API_KEY", "test-key-not-used")
for _key in [k for k in os.environ if k.startswith("OPENPOKE_")]:
    os.environ.pop(_key)  # tests assume defaults regardless of the developer's .env

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _isolated_usage_ledger():
    """Never append test usage to server/data/usage.jsonl."""
    from server.services.llm import UsageLedger, set_usage_ledger

    set_usage_ledger(UsageLedger())
    yield
    set_usage_ledger(None)
