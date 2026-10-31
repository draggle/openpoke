"""Tiered memory for execution agents.

Upstream embeds an execution agent's **entire** append-only log in its system prompt on every
run. The main conversation gets summarised; execution agents never do. After weeks, an
agent that has handled one person's email carries hundreds of raw tool payloads.

This module keeps the property the author wanted (an agent can answer "did Alice reply?"
with full context) while bounding the prompt:

    ┌──────────────────────────────────────────────────────────────┐
    │ # Memory                     ← structured, compact, durable   │
    │   facts        key → value, with temporal invalidation        │
    │   open threads what we're waiting on                          │
    │   timeline     one-line dated episodes                        │
    │ # Recent History             ← last N raw log entries verbatim│
    └──────────────────────────────────────────────────────────────┘

Compaction runs when the raw transcript exceeds a character budget. An *extractor* turns the
older entries into facts / threads / episodes. Two extractors ship:

- ``LLMExtractor`` (default at runtime): one cheap-model call with a strict JSON schema.
- ``HeuristicExtractor`` (fallback + offline evals/tests): deterministic, no network.

Facts are the one knowledge-graph idea worth stealing from Zep/Graphiti: when a new value
arrives for an existing key, the old fact is *invalidated*, not deleted, so "Alice moved
lunch from Friday to Thursday" is answerable and auditable.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

from ...logging_config import logger
from .log_store import ExecutionAgentLogStore, _slugify, get_execution_agent_logs

LogEntry = Tuple[str, str, str]  # (tag, timestamp, payload)


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------


def _env_int(name: str, fallback: int) -> int:
    try:
        return int(os.getenv(name, str(fallback)))
    except (TypeError, ValueError):
        return fallback


@dataclass(frozen=True)
class MemoryConfig:
    enabled: bool = True
    budget_chars: int = 12_000  # ~3k tokens of raw history before compaction kicks in
    tail_entries: int = 12  # raw entries always kept verbatim
    max_episodes: int = 40
    max_facts: int = 60
    max_threads: int = 15

    @classmethod
    def from_env(cls) -> "MemoryConfig":
        return cls(
            enabled=os.getenv("OPENPOKE_AGENT_MEMORY", "1") != "0",
            budget_chars=_env_int("OPENPOKE_AGENT_MEMORY_BUDGET_CHARS", 12_000),
            tail_entries=_env_int("OPENPOKE_AGENT_MEMORY_TAIL", 12),
            max_episodes=_env_int("OPENPOKE_AGENT_MEMORY_MAX_EPISODES", 40),
        )


# --------------------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------------------


@dataclass
class Fact:
    key: str
    value: str
    recorded_at: str
    invalidated_at: Optional[str] = None
    superseded_by: Optional[str] = None  # the value that replaced this one

    @property
    def active(self) -> bool:
        return self.invalidated_at is None


@dataclass
class Thread:
    subject: str
    last_action: str
    waiting_on: str = ""
    updated_at: str = ""


@dataclass
class Episode:
    timestamp: str
    text: str


@dataclass
class MemoryState:
    facts: List[Fact] = field(default_factory=list)
    threads: List[Thread] = field(default_factory=list)
    episodes: List[Episode] = field(default_factory=list)
    compacted_through: int = 0  # number of log entries absorbed into memory
    compactions: int = 0

    def active_facts(self) -> List[Fact]:
        return [f for f in self.facts if f.active]

    def to_json(self) -> Dict[str, Any]:
        return {
            "facts": [asdict(f) for f in self.facts],
            "threads": [asdict(t) for t in self.threads],
            "episodes": [asdict(e) for e in self.episodes],
            "compacted_through": self.compacted_through,
            "compactions": self.compactions,
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "MemoryState":
        return cls(
            facts=[Fact(**f) for f in data.get("facts", [])],
            threads=[Thread(**t) for t in data.get("threads", [])],
            episodes=[Episode(**e) for e in data.get("episodes", [])],
            compacted_through=int(data.get("compacted_through", 0)),
            compactions=int(data.get("compactions", 0)),
        )


@dataclass
class Extraction:
    """What an extractor returns for a batch of old log entries."""

    facts: List[Tuple[str, str]] = field(default_factory=list)  # (key, value)
    threads: List[Thread] = field(default_factory=list)
    resolved_threads: List[str] = field(default_factory=list)  # subjects now closed
    episodes: List[str] = field(default_factory=list)


class Extractor(Protocol):
    async def __call__(self, entries: Sequence[LogEntry], state: MemoryState) -> Extraction: ...


# --------------------------------------------------------------------------------------
# Extractors
# --------------------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_CAPITALISED_RE = re.compile(r"\b([A-Z][a-z]{1,30})\b")
_NOT_A_NAME = frozenset(
    "Calling Round Email To From Subject Cc Bcc Draft Drafted Sent Reply Ask Asked Find Found "
    "Please Remind Reminder Send Check The And For With About Result Results Mr Ms Dr".split()
)


def _name_before(text: str, end: int, window: int = 48) -> Optional[str]:
    """Nearest capitalised word before ``end`` that looks like a person's name."""
    names = [m.group(1) for m in _CAPITALISED_RE.finditer(text, max(0, end - window), end)]
    for candidate in reversed(names):
        if candidate not in _NOT_A_NAME:
            return candidate
    return None


class HeuristicExtractor:
    """Deterministic extractor: no LLM. Used offline and as a fallback.

    - every ``agent_request`` becomes a timeline episode ("Asked: ...")
    - every ``agent_response`` becomes a timeline episode ("Result: ...")
    - "Name ... email@x" patterns become facts ("Name email" -> address)
    - a response mentioning "waiting"/"awaiting"/"no reply" opens a thread
    """

    def __init__(self, snippet_chars: int = 160) -> None:
        self.snippet_chars = snippet_chars

    async def __call__(self, entries: Sequence[LogEntry], state: MemoryState) -> Extraction:
        out = Extraction()
        for tag, _ts, payload in entries:
            text = " ".join(payload.split())
            if tag == "agent_request":
                out.episodes.append(f"Asked: {text[: self.snippet_chars]}")
            elif tag == "agent_response":
                out.episodes.append(f"Result: {text[: self.snippet_chars]}")
                lowered = text.lower()
                if any(k in lowered for k in ("waiting", "awaiting", "no reply", "not yet replied")):
                    out.threads.append(
                        Thread(subject=text[:60], last_action=text[: self.snippet_chars],
                               waiting_on="reply")
                    )
                if any(k in lowered for k in ("replied", "confirmed", "sent successfully", "done")):
                    out.resolved_threads.extend(t.subject for t in state.threads)
            for match in _EMAIL_RE.finditer(payload):
                name = _name_before(payload, match.start())
                if name:
                    out.facts.append((f"{name} email", match.group(0)))
        return out


_LLM_SYSTEM_PROMPT = """You compress an execution agent's activity log into durable memory.
Return ONLY a JSON object with keys:
  "facts": [{"key": "...", "value": "..."}]   stable facts (addresses, dates, decisions, preferences). Keys are short noun phrases; if a fact changed, give the NEW value.
  "threads": [{"subject": "...", "last_action": "...", "waiting_on": "..."}]   things still pending.
  "resolved_threads": ["subject", ...]   previously-open threads that are now closed.
  "episodes": ["...", ...]   one line per notable event, chronological, past tense, <=140 chars.
Be concrete. Preserve names, email addresses, dates, amounts, and any commitment or reminder the user asked for verbatim. Omit tool-call mechanics."""


class LLMExtractor:
    """Extract memory with one cheap-model call; falls back to the heuristic on any error."""

    def __init__(self, fallback: Optional[Extractor] = None) -> None:
        self.fallback = fallback or HeuristicExtractor()

    async def __call__(self, entries: Sequence[LogEntry], state: MemoryState) -> Extraction:
        from ..llm import ROLE_MEMORY_COMPACTOR, complete  # local import avoids cycles

        rendered = "\n".join(f"[{ts}] <{tag}> {payload}" for tag, ts, payload in entries)
        existing = {
            "facts": [{"key": f.key, "value": f.value} for f in state.active_facts()],
            "threads": [asdict(t) for t in state.threads],
        }
        user = (
            "Existing memory (for context; only report changes and new items):\n"
            f"{json.dumps(existing, ensure_ascii=False)}\n\n"
            f"New log entries to compress:\n{rendered}"
        )
        try:
            response = await complete(
                role=ROLE_MEMORY_COMPACTOR,
                messages=[{"role": "user", "content": user}],
                system=_LLM_SYSTEM_PROMPT,
                tag="memory_compaction",
            )
            content = ((response.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            data = _parse_json_object(content)
            return Extraction(
                facts=[(str(f.get("key", "")).strip(), str(f.get("value", "")).strip())
                       for f in data.get("facts", []) if f.get("key") and f.get("value")],
                threads=[Thread(subject=str(t.get("subject", ""))[:120],
                                last_action=str(t.get("last_action", ""))[:240],
                                waiting_on=str(t.get("waiting_on", ""))[:120])
                         for t in data.get("threads", []) if t.get("subject")],
                resolved_threads=[str(s) for s in data.get("resolved_threads", [])],
                episodes=[str(e)[:200] for e in data.get("episodes", [])],
            )
        except Exception as exc:
            logger.warning(f"LLM memory extraction failed; using heuristic: {exc}")
            return await self.fallback(entries, state)


def _parse_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in extractor output")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("extractor output is not an object")
    return data


# --------------------------------------------------------------------------------------
# Memory manager
# --------------------------------------------------------------------------------------


class AgentMemory:
    def __init__(
        self,
        base_dir: Path,
        *,
        log_store: Optional[ExecutionAgentLogStore] = None,
        config: Optional[MemoryConfig] = None,
        extractor: Optional[Extractor] = None,
        on_summary: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        self._base_dir = base_dir
        self._log_store = log_store or get_execution_agent_logs()
        self.config = config or MemoryConfig.from_env()
        self.extractor: Extractor = extractor or LLMExtractor()
        self._on_summary = on_summary  # e.g. registry.update_summary

    # ---- persistence -----------------------------------------------------------------

    def _path(self, agent_name: str) -> Path:
        return self._base_dir / f"{_slugify(agent_name)}.memory.json"

    def load(self, agent_name: str) -> MemoryState:
        path = self._path(agent_name)
        if not path.exists():
            return MemoryState()
        try:
            return MemoryState.from_json(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            logger.warning(f"Failed to load memory for {agent_name}: {exc}")
            return MemoryState()

    def save(self, agent_name: str, state: MemoryState) -> None:
        try:
            self._base_dir.mkdir(parents=True, exist_ok=True)
            self._path(agent_name).write_text(
                json.dumps(state.to_json(), indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning(f"Failed to save memory for {agent_name}: {exc}")

    def clear_all(self) -> None:
        for path in self._base_dir.glob("*.memory.json"):
            try:
                path.unlink()
            except Exception:
                pass

    # ---- compaction ------------------------------------------------------------------

    def transcript_chars(self, agent_name: str) -> int:
        return len(self._log_store.load_transcript(agent_name))

    def needs_compaction(self, agent_name: str, state: Optional[MemoryState] = None) -> bool:
        if not self.config.enabled:
            return False
        state = state or self.load(agent_name)
        entries = list(self._log_store.iter_entries(agent_name))
        pending = entries[state.compacted_through : max(state.compacted_through, len(entries) - self.config.tail_entries)]
        if not pending:
            return False
        raw_tail_chars = sum(len(p) for _, _, p in entries[state.compacted_through :])
        return raw_tail_chars > self.config.budget_chars

    async def compact(self, agent_name: str, *, force: bool = False) -> Optional[MemoryState]:
        """Absorb older log entries into structured memory. Returns new state or None."""
        state = self.load(agent_name)
        if not force and not self.needs_compaction(agent_name, state):
            return None
        entries = list(self._log_store.iter_entries(agent_name))
        cutoff = max(state.compacted_through, len(entries) - self.config.tail_entries)
        batch = entries[state.compacted_through : cutoff]
        if not batch:
            return None

        extraction = await self.extractor(batch, state)
        now_ts = batch[-1][1] or ""
        self._merge(state, extraction, now_ts)
        state.compacted_through = cutoff
        state.compactions += 1
        self.save(agent_name, state)

        if self._on_summary:
            try:
                self._on_summary(agent_name, self.one_line(state))
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(f"summary callback failed for {agent_name}: {exc}")
        logger.info(
            f"[{agent_name}] memory compacted: {len(batch)} entries -> "
            f"{len(state.active_facts())} facts, {len(state.threads)} threads, {len(state.episodes)} episodes"
        )
        return state

    def _merge(self, state: MemoryState, extraction: Extraction, now_ts: str) -> None:
        # facts: temporal invalidation on key collision
        for key, value in extraction.facts:
            key_norm = key.strip().lower()
            if not key_norm or not value:
                continue
            current = next((f for f in state.facts if f.active and f.key.lower() == key_norm), None)
            if current is not None:
                if current.value.strip().lower() == value.strip().lower():
                    continue
                current.invalidated_at = now_ts
                current.superseded_by = value
            state.facts.append(Fact(key=key.strip(), value=value.strip(), recorded_at=now_ts))
        # cap facts: drop oldest *invalidated* first, never active ones
        if len(state.facts) > self.config.max_facts:
            inactive = [f for f in state.facts if not f.active]
            overflow = len(state.facts) - self.config.max_facts
            for fact in inactive[:overflow]:
                state.facts.remove(fact)

        # threads
        resolved = {s.strip().lower() for s in extraction.resolved_threads}
        state.threads = [t for t in state.threads if t.subject.strip().lower() not in resolved]
        for thread in extraction.threads:
            subject_norm = thread.subject.strip().lower()
            if not subject_norm or subject_norm in resolved:
                continue
            thread.updated_at = thread.updated_at or now_ts
            existing = next((t for t in state.threads if t.subject.strip().lower() == subject_norm), None)
            if existing:
                existing.last_action = thread.last_action
                existing.waiting_on = thread.waiting_on
                existing.updated_at = now_ts
            else:
                state.threads.append(thread)
        state.threads = state.threads[-self.config.max_threads :]

        # episodes
        for text in extraction.episodes:
            if text.strip():
                state.episodes.append(Episode(timestamp=now_ts, text=text.strip()))
        if len(state.episodes) > self.config.max_episodes:
            # keep the first few (origin story) and the most recent
            keep_head = 5
            state.episodes = state.episodes[:keep_head] + state.episodes[-(self.config.max_episodes - keep_head) :]

    # ---- rendering -------------------------------------------------------------------

    def one_line(self, state: MemoryState) -> str:
        if state.threads:
            t = state.threads[-1]
            waiting = f" (waiting on {t.waiting_on})" if t.waiting_on else ""
            return f"{t.subject}{waiting}"
        if state.episodes:
            return state.episodes[-1].text
        if state.active_facts():
            f = state.active_facts()[-1]
            return f"{f.key}: {f.value}"
        return ""

    def render_history(self, agent_name: str) -> str:
        """History block for the execution agent's system prompt.

        Falls back to the upstream full transcript when memory is disabled or nothing has
        been compacted yet, so behaviour is unchanged for young agents.
        """
        if not self.config.enabled:
            return self._log_store.load_transcript(agent_name)
        state = self.load(agent_name)
        entries = list(self._log_store.iter_entries(agent_name))
        if state.compacted_through == 0:
            return self._log_store.load_transcript(agent_name)

        parts: List[str] = ["# Memory (compressed from earlier activity)"]
        facts = state.active_facts()
        if facts:
            parts.append("## Facts")
            parts.extend(f"- {f.key}: {f.value} (as of {f.recorded_at})" for f in facts)
            superseded = [f for f in state.facts if not f.active][-5:]
            if superseded:
                parts.append("## Changed facts (old → new)")
                parts.extend(
                    f"- {f.key}: {f.value} → {f.superseded_by} (changed {f.invalidated_at})"
                    for f in superseded
                )
        if state.threads:
            parts.append("## Open threads")
            for t in state.threads:
                waiting = f"; waiting on {t.waiting_on}" if t.waiting_on else ""
                parts.append(f"- {t.subject}: {t.last_action}{waiting}")
        if state.episodes:
            parts.append("## Timeline")
            parts.extend(f"- [{e.timestamp}] {e.text}" for e in state.episodes)

        recent = entries[state.compacted_through :]
        parts.append("")
        parts.append(f"# Recent History ({len(recent)} most recent entries, verbatim)")
        from html import escape

        for tag, ts, payload in recent:
            body = escape(payload, quote=False)
            parts.append(f'<{tag} timestamp="{ts}">{body}</{tag}>' if ts else f"<{tag}>{body}</{tag}>")
        return "\n".join(parts)


# --------------------------------------------------------------------------------------
# Singleton wiring
# --------------------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_MEMORY_DIR = _DATA_DIR / "execution_agents"
_memory: Optional[AgentMemory] = None


def get_agent_memory() -> AgentMemory:
    global _memory
    if _memory is None:
        from .registry import get_agent_registry

        _memory = AgentMemory(
            _MEMORY_DIR,
            on_summary=lambda name, summary: get_agent_registry().update_summary(name, summary),
        )
    return _memory


__all__ = [
    "AgentMemory",
    "Episode",
    "Extraction",
    "Fact",
    "HeuristicExtractor",
    "LLMExtractor",
    "MemoryConfig",
    "MemoryState",
    "Thread",
    "get_agent_memory",
]
