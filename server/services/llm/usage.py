"""Per-call usage ledger: tokens and estimated cost by role and model.

Every call through the gateway appends one record. The ledger is kept in memory and, when a
path is configured, mirrored to a JSONL file so cost can be inspected after a run
(``GET /meta/usage`` exposes the in-memory totals).
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token for English). Used only when the provider
    omits ``usage``; evals use it consistently for both baseline and treatment."""
    return max(1, (len(text) + 3) // 4)


def estimate_message_tokens(messages: List[Mapping[str, Any]], system: Optional[str]) -> int:
    total = estimate_tokens(system or "")
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif content is not None:
            total += estimate_tokens(json.dumps(content, default=str))
        tool_calls = message.get("tool_calls")
        if tool_calls:
            total += estimate_tokens(json.dumps(tool_calls, default=str))
    return total


@dataclass
class UsageRecord:
    timestamp: float
    role: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    estimated: bool  # True when tokens were estimated rather than reported by the provider
    tag: str = ""  # free-form, e.g. agent name


class UsageLedger:
    MAX_RECORDS = 5000  # ring buffer; totals are kept separately so nothing is lost

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = path
        self._records: List[UsageRecord] = []
        self._lock = threading.Lock()
        self._by_role: Dict[str, Dict[str, float]] = {}
        self._by_model: Dict[str, Dict[str, float]] = {}
        self._total = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}

    @staticmethod
    def _bump(bucket: Dict[str, float], rec: "UsageRecord") -> None:
        bucket["calls"] += 1
        bucket["prompt_tokens"] += rec.prompt_tokens
        bucket["completion_tokens"] += rec.completion_tokens
        bucket["cost_usd"] += rec.cost_usd

    def record(self, record: UsageRecord) -> None:
        with self._lock:
            self._records.append(record)
            if len(self._records) > self.MAX_RECORDS:
                del self._records[: len(self._records) - self.MAX_RECORDS]
            for buckets, key in ((self._by_role, record.role), (self._by_model, record.model)):
                self._bump(buckets.setdefault(key, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}), record)
            self._bump(self._total, record)
            if self._path is not None:
                try:
                    self._path.parent.mkdir(parents=True, exist_ok=True)
                    with self._path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(asdict(record)) + "\n")
                except Exception:
                    pass

    def records(self) -> List[UsageRecord]:
        with self._lock:
            return list(self._records)

    def reset(self) -> None:
        with self._lock:
            self._records.clear()
            self._by_role.clear()
            self._by_model.clear()
            self._total = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}

    def totals(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total": dict(self._total),
                "by_role": {k: dict(v) for k, v in self._by_role.items()},
                "by_model": {k: dict(v) for k, v in self._by_model.items()},
            }


_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_ledger: Optional[UsageLedger] = None


def get_usage_ledger() -> UsageLedger:
    global _ledger
    if _ledger is None:
        _ledger = UsageLedger(_DATA_DIR / "usage.jsonl")
    return _ledger


def set_usage_ledger(ledger: Optional[UsageLedger]) -> None:
    global _ledger
    _ledger = ledger


def now() -> float:
    return time.time()


__all__ = [
    "UsageLedger",
    "UsageRecord",
    "estimate_message_tokens",
    "estimate_tokens",
    "get_usage_ledger",
    "set_usage_ledger",
]
