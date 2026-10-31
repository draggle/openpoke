"""Shared pytest configuration.

Tests never talk to OpenRouter or Gmail. A dummy API key keeps the runtimes constructible,
and every LLM call is mocked at the single chokepoint ``request_chat_completion``.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("OPENROUTER_API_KEY", "test-key-not-used")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
