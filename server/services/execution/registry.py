"""Agent registry: a budgeted, relevance-ranked view of execution agents.

Upstream OpenPoke keeps a flat ``roster.json`` list of agent *names* and injects every
name into every Interaction Agent prompt. This module replaces that with a registry that
records per-agent metadata (recency, usage, a one-line summary) and, on each turn, selects a
small roster using three signals:

1. **Lifecycle** - agents idle for longer than ``active_ttl`` become *dormant*; longer than
   ``archive_ttl`` become *archived* and are excluded from the prompt (but stay searchable).
   Agents with pending triggers are pinned and never archived.
2. **Hot cache** - the most recently touched agents are always shown, because follow-ups
   ("did she reply?") rarely contain retrievable keywords.
3. **Retrieval** - BM25 over name + description + rolling summary + recent requests,
   so "what did Alice say about the invoice" surfaces the right agent out of hundreds.

The result is capped at ``max_roster`` entries. A ``<roster_note>`` tells the model how many
agents are hidden and that it can call ``search_agents`` before spawning a new one.

Set ``OPENPOKE_ROSTER_STRATEGY=all`` to reproduce upstream behaviour (used as the eval baseline).
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from html import escape
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from rank_bm25 import BM25Okapi

from ...logging_config import logger


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------


def _env_int(name: str, fallback: int) -> int:
    try:
        return int(os.getenv(name, str(fallback)))
    except (TypeError, ValueError):
        return fallback


def _env_float(name: str, fallback: float) -> float:
    try:
        return float(os.getenv(name, str(fallback)))
    except (TypeError, ValueError):
        return fallback


@dataclass(frozen=True)
class RegistryConfig:
    """Tunable knobs. Defaults are deliberately conservative."""

    strategy: str = "budgeted"  # "budgeted" | "all"
    max_roster: int = 12
    hot_cache_size: int = 4
    retrieval_k: int = 8
    min_score: float = 0.0  # BM25 score strictly greater than this to count as a hit
    active_ttl_s: float = 48 * 3600  # idle longer than this -> dormant
    archive_ttl_s: float = 14 * 24 * 3600  # idle longer than this -> archived
    description_chars: int = 160
    recent_requests_kept: int = 3
    recency_weight: float = 0.6  # weight of the recency prior in retrieval scores
    recency_half_life_s: float = 3 * 24 * 3600  # recency prior halves every 3 days idle

    @classmethod
    def from_env(cls) -> "RegistryConfig":
        return cls(
            strategy=os.getenv("OPENPOKE_ROSTER_STRATEGY", "budgeted").strip().lower() or "budgeted",
            max_roster=_env_int("OPENPOKE_ROSTER_MAX", 12),
            hot_cache_size=_env_int("OPENPOKE_ROSTER_HOT", 4),
            retrieval_k=_env_int("OPENPOKE_ROSTER_RETRIEVAL_K", 8),
            min_score=_env_float("OPENPOKE_ROSTER_MIN_SCORE", 0.0),
            recency_weight=_env_float("OPENPOKE_ROSTER_RECENCY_WEIGHT", 0.6),
            active_ttl_s=_env_float("OPENPOKE_AGENT_ACTIVE_TTL_S", 48 * 3600),
            archive_ttl_s=_env_float("OPENPOKE_AGENT_ARCHIVE_TTL_S", 14 * 24 * 3600),
        )


class AgentStatus(str, Enum):
    ACTIVE = "active"
    DORMANT = "dormant"
    ARCHIVED = "archived"


# --------------------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------------------


@dataclass
class AgentRecord:
    name: str
    created_at: float
    last_used_at: float
    use_count: int = 0
    description: str = ""  # first instruction, truncated
    summary: str = ""  # one-line rolling summary (maintained by AgentMemory)
    recent_requests: List[str] = field(default_factory=list)
    last_reason: str = "created"  # "user" | "trigger" | "created" | "migrated"

    def to_json(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: Dict) -> "AgentRecord":
        return cls(
            name=str(data.get("name", "")),
            created_at=float(data.get("created_at", 0.0)),
            last_used_at=float(data.get("last_used_at", 0.0)),
            use_count=int(data.get("use_count", 0)),
            description=str(data.get("description", "")),
            summary=str(data.get("summary", "")),
            recent_requests=[str(x) for x in data.get("recent_requests", [])][-5:],
            last_reason=str(data.get("last_reason", "created")),
        )

    def retrieval_text(self) -> str:
        parts = [self.name, self.name, self.description, self.summary, *self.recent_requests]
        return " ".join(p for p in parts if p)


@dataclass
class RosterEntry:
    record: AgentRecord
    status: AgentStatus
    pinned: bool
    reason: str  # "hot" | "retrieved" | "all"
    score: float = 0.0


@dataclass
class RosterSelection:
    shown: List[RosterEntry]
    total: int
    archived: int
    strategy: str

    @property
    def hidden(self) -> int:
        return max(0, self.total - self.archived - len(self.shown))

    def names(self) -> List[str]:
        return [entry.record.name for entry in self.shown]


# --------------------------------------------------------------------------------------
# Tokeniser (tiny, dependency-free)
# --------------------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    """a an the and or but if then of to in on at for from by with about as into like through
    after over between out against during without before under around among is are was were be
    been being have has had do does did will would shall should can could may might must i me my
    we our you your he him his she her it its they them their what which who whom this that these
    those am did do does doing yes no not so too very just also please let lets hey hi ok okay
    email emails agent agents message messages send sent reply replied""".split()
)


def tokenize(text: str) -> List[str]:
    return [tok for tok in _TOKEN_RE.findall(text.lower()) if tok not in _STOPWORDS]


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------


class AgentRegistry:
    """Persistent registry of execution agents with budgeted roster selection."""

    def __init__(
        self,
        path: Path,
        *,
        config: Optional[RegistryConfig] = None,
        clock: Callable[[], float] = time.time,
        has_pending_triggers: Optional[Callable[[str], bool]] = None,
        legacy_roster_path: Optional[Path] = None,
    ) -> None:
        self._path = path
        self.config = config or RegistryConfig.from_env()
        self._clock = clock
        self._has_pending_triggers = has_pending_triggers or (lambda _name: False)
        self._legacy_roster_path = legacy_roster_path
        self._records: Dict[str, AgentRecord] = {}
        self.load()

    # ---- persistence -----------------------------------------------------------------

    def load(self) -> None:
        self._records = {}
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                for raw in data.get("agents", []):
                    record = AgentRecord.from_json(raw)
                    if record.name:
                        self._records[record.name] = record
            except Exception as exc:
                logger.warning(f"Failed to load registry.json: {exc}")
        self._migrate_legacy_roster()

    def _migrate_legacy_roster(self) -> None:
        """Adopt names from upstream roster.json that the registry has never seen."""
        if not self._legacy_roster_path or not self._legacy_roster_path.exists():
            return
        try:
            names = json.loads(self._legacy_roster_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(names, list):
            return
        changed = False
        now = self._clock()
        for name in names:
            name = str(name)
            if name and name not in self._records:
                self._records[name] = AgentRecord(
                    name=name, created_at=now, last_used_at=now, last_reason="migrated"
                )
                changed = True
        if changed:
            self.save()

    def save(self) -> None:
        payload = {"version": 1, "agents": [r.to_json() for r in self._records.values()]}
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # Atomic replace: readers never see a partially written file.
            tmp = self._path.with_suffix(f".{os.getpid()}.tmp")
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            os.replace(tmp, self._path)
        except Exception as exc:
            logger.warning(f"Failed to save registry.json: {exc}")

    def clear(self) -> None:
        self._records = {}
        try:
            if self._path.exists():
                self._path.unlink()
        except Exception as exc:
            logger.warning(f"Failed to clear registry.json: {exc}")

    # ---- mutation --------------------------------------------------------------------

    def register(self, name: str, instructions: str = "", *, reason: str = "user") -> bool:
        """Create or touch an agent. Returns True if the agent is new."""
        now = self._clock()
        record = self._records.get(name)
        is_new = record is None
        if record is None:
            record = AgentRecord(name=name, created_at=now, last_used_at=now)
            self._records[name] = record
        record.last_used_at = now
        record.use_count += 1
        record.last_reason = reason
        snippet = " ".join(instructions.split())[: self.config.description_chars]
        if snippet:
            if not record.description:
                record.description = snippet
            record.recent_requests = (record.recent_requests + [snippet])[
                -self.config.recent_requests_kept :
            ]
        self.save()
        return is_new

    def touch(self, name: str, *, reason: str = "user") -> None:
        """Mark an agent as used now (e.g. a trigger fired for it) without a new request."""
        record = self._records.get(name)
        now = self._clock()
        if record is None:
            self._records[name] = AgentRecord(
                name=name, created_at=now, last_used_at=now, use_count=1, last_reason=reason
            )
        else:
            record.last_used_at = now
            record.last_reason = reason
        self.save()

    def update_summary(self, name: str, summary: str) -> None:
        record = self._records.get(name)
        if record is None:
            return
        record.summary = " ".join(summary.split())[:240]
        self.save()

    # ---- queries ---------------------------------------------------------------------

    def get(self, name: str) -> Optional[AgentRecord]:
        return self._records.get(name)

    def names(self) -> List[str]:
        return list(self._records.keys())

    def records(self) -> List[AgentRecord]:
        return list(self._records.values())

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, name: object) -> bool:
        return name in self._records

    def is_pinned(self, name: str) -> bool:
        try:
            return bool(self._has_pending_triggers(name))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"pending-trigger lookup failed for {name}: {exc}")
            return False

    def status_of(self, record: AgentRecord, now: Optional[float] = None) -> AgentStatus:
        now = self._clock() if now is None else now
        idle = max(0.0, now - record.last_used_at)
        if idle <= self.config.active_ttl_s:
            return AgentStatus.ACTIVE
        if idle <= self.config.archive_ttl_s or self.is_pinned(record.name):
            return AgentStatus.DORMANT
        return AgentStatus.ARCHIVED

    def search(
        self,
        query: str,
        *,
        limit: int = 8,
        include_archived: bool = True,
        statuses: Optional[Dict[str, AgentStatus]] = None,
    ) -> List[Tuple[AgentRecord, float]]:
        """BM25 search over agent metadata. Returns (record, score) with score > min_score."""
        if include_archived:
            candidates = list(self._records.values())
        else:
            statuses = statuses or {r.name: self.status_of(r) for r in self._records.values()}
            candidates = [r for r in self._records.values() if statuses[r.name] != AgentStatus.ARCHIVED]
        query_tokens = tokenize(query)
        if not candidates or not query_tokens:
            return []
        corpus = [tokenize(r.retrieval_text()) or ["_"] for r in candidates]
        bm25 = BM25Okapi(corpus)
        bm25_scores = bm25.get_scores(query_tokens)

        # BM25's IDF collapses to ~0 on tiny corpora (a term in 1 of 2 docs scores nothing),
        # so blend in plain query-token overlap. A hit must share at least one token.
        # Scores are normalised to [0, 1] and a recency prior is added so that, among
        # agents that match equally well ("any update on the visa?" with three visa agents),
        # the one the user most recently worked with wins.
        unique_query = set(query_tokens)
        max_bm = max((float(b) for b in bm25_scores), default=0.0)
        now = self._clock()
        scored: List[Tuple[AgentRecord, float]] = []
        for record, doc_tokens, bm in zip(candidates, corpus, bm25_scores):
            overlap = len(unique_query & set(doc_tokens))
            if overlap == 0:
                continue
            bm_norm = float(bm) / max_bm if max_bm > 0 else 0.0
            idle = max(0.0, now - record.last_used_at)
            recency = 0.5 ** (idle / self.config.recency_half_life_s)
            score = bm_norm + overlap / len(unique_query) + self.config.recency_weight * recency
            if score > self.config.min_score:
                scored.append((record, score))
        scored.sort(key=lambda pair: (-pair[1], -pair[0].last_used_at))
        return scored[:limit]

    def select(self, query: str = "", *, now: Optional[float] = None) -> RosterSelection:
        """Pick the roster to show the Interaction Agent for this turn."""
        now = self._clock() if now is None else now
        records = list(self._records.values())
        total = len(records)

        if self.config.strategy == "all":
            # Upstream order (roster.json insertion order); no status/trigger lookups.
            shown = [RosterEntry(r, AgentStatus.ACTIVE, False, "all") for r in records]
            return RosterSelection(shown=shown, total=total, archived=0, strategy="all")

        # One status pass per turn. status_of() only consults the trigger store for agents
        # past the archive TTL, and pinned-ness is cached from that pass below.
        statuses = {r.name: self.status_of(r, now) for r in records}
        pinned_cache: Dict[str, bool] = {}

        def pinned(name: str) -> bool:
            if name not in pinned_cache:
                pinned_cache[name] = self.is_pinned(name)
            return pinned_cache[name]

        archived = sum(1 for s in statuses.values() if s == AgentStatus.ARCHIVED)
        live = [r for r in records if statuses[r.name] != AgentStatus.ARCHIVED]

        chosen: Dict[str, RosterEntry] = {}

        # 1) hot cache: most recently used live agents
        for record in sorted(live, key=lambda r: -r.last_used_at)[: self.config.hot_cache_size]:
            chosen[record.name] = RosterEntry(record, statuses[record.name], pinned(record.name), "hot")

        # 2) retrieval: relevance to the current message
        if query.strip():
            for record, score in self.search(
                query, limit=self.config.retrieval_k, include_archived=False, statuses=statuses
            ):
                if record.name in chosen:
                    chosen[record.name].score = max(chosen[record.name].score, score)
                    continue
                if len(chosen) >= self.config.max_roster:
                    break
                chosen[record.name] = RosterEntry(
                    record, statuses[record.name], pinned(record.name), "retrieved", score
                )

        # 3) fill any remaining budget with the next most recent live agents
        if len(chosen) < self.config.max_roster:
            for record in sorted(live, key=lambda r: -r.last_used_at):
                if len(chosen) >= self.config.max_roster:
                    break
                if record.name not in chosen:
                    chosen[record.name] = RosterEntry(
                        record, statuses[record.name], pinned(record.name), "recent"
                    )

        shown = sorted(
            chosen.values(), key=lambda e: (-e.score, -e.record.last_used_at)
        )[: self.config.max_roster]
        return RosterSelection(shown=shown, total=total, archived=archived, strategy="budgeted")

    # ---- rendering -------------------------------------------------------------------

    def render(self, selection: RosterSelection, *, now: Optional[float] = None) -> str:
        """Render a selection as the <active_agents> body."""
        if selection.total == 0:
            return "None"
        now = self._clock() if now is None else now

        if selection.strategy == "all":
            # Byte-for-byte the upstream format so the baseline eval is honest.
            return "\n".join(
                f'<agent name="{escape(e.record.name or "agent", quote=True)}" />'
                for e in selection.shown
            )

        lines: List[str] = []
        for entry in selection.shown:
            record = entry.record
            attrs = [
                f'name="{escape(record.name or "agent", quote=True)}"',
                f'status="{entry.status.value}"',
                f'last_used="{_relative_time(now - record.last_used_at)}"',
            ]
            if entry.pinned:
                attrs.append('has_reminders="true"')
            body = record.summary or record.description
            if body:
                lines.append(f"<agent {' '.join(attrs)}>{escape(body, quote=False)}</agent>")
            else:
                lines.append(f"<agent {' '.join(attrs)} />")

        note = (
            f"Showing {len(selection.shown)} of {selection.total} agents "
            f"(most recent + most relevant to the current message); "
            f"{selection.hidden} others hidden, {selection.archived} archived. "
            "Before creating a new agent for a topic that may already have one, "
            "call search_agents."
        )
        lines.append(f"<roster_note>{note}</roster_note>")
        return "\n".join(lines)

    def render_search_results(self, results: Sequence[Tuple[AgentRecord, float]]) -> List[Dict]:
        now = self._clock()
        payload = []
        for record, score in results:
            payload.append(
                {
                    "agent_name": record.name,
                    "status": self.status_of(record, now).value,
                    "last_used": _relative_time(now - record.last_used_at),
                    "summary": record.summary or record.description,
                    "score": round(score, 3),
                }
            )
        return payload


def _relative_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 60:
        return "just now"
    minutes = seconds / 60
    if minutes < 60:
        return f"{int(minutes)}m ago"
    hours = minutes / 60
    if hours < 48:
        return f"{int(hours)}h ago"
    days = hours / 24
    return f"{int(days)}d ago"


# --------------------------------------------------------------------------------------
# Singleton wiring
# --------------------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_REGISTRY_PATH = _DATA_DIR / "execution_agents" / "registry.json"
_LEGACY_ROSTER_PATH = _DATA_DIR / "execution_agents" / "roster.json"


def _agent_has_pending_triggers(agent_name: str) -> bool:
    """True when the agent owns an active trigger with a future fire time."""
    from ..triggers import get_trigger_service  # local import: avoid cycles at module load

    try:
        for trigger in get_trigger_service().list_triggers(agent_name=agent_name):
            if trigger.status == "active" and trigger.next_trigger:
                return True
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"trigger lookup failed for {agent_name}: {exc}")
    return False


_registry: Optional[AgentRegistry] = None


def get_agent_registry() -> AgentRegistry:
    global _registry
    if _registry is None:
        _registry = AgentRegistry(
            _REGISTRY_PATH,
            has_pending_triggers=_agent_has_pending_triggers,
            legacy_roster_path=_LEGACY_ROSTER_PATH,
        )
    return _registry


__all__ = [
    "AgentRecord",
    "AgentRegistry",
    "AgentStatus",
    "RegistryConfig",
    "RosterEntry",
    "RosterSelection",
    "get_agent_registry",
    "tokenize",
]
