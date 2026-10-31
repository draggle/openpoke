"""Where the server keeps its state.

``OPENPOKE_DATA_DIR`` overrides the default ``server/data`` so evaluators that drive the real
app in-process (``evals/session_probe.py``) never touch a working installation's conversation,
agents, memory or reminders.
"""

from __future__ import annotations

import os
from pathlib import Path

_DEFAULT = Path(__file__).resolve().parent / "data"


def data_dir() -> Path:
    override = os.getenv("OPENPOKE_DATA_DIR")
    return Path(override).expanduser().resolve() if override else _DEFAULT


__all__ = ["data_dir"]
