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
    transcript_boost: float = 1.0  # added to the score of agents named in the recent conversation
    pinned_slots: int = 3  # roster slots reserved for the most recent agents that own reminders
    fuzzy: bool = False  # trigram fallback when no token matches (misspelt names); off by default
    fuzzy_min_similarity: float = 0.4
    recency_half_life_s: float = 3 * 24 * 3600  # recency prior halves every 3 days idle
    consolidate: bool = True  # redirect new agents that duplicate a live one
    consolidate_min_overlap: float = 0.75  # name-token overlap required in *both* directions
    consolidate_instruction_overlap: float = 0.5  # or: shared name token + this much instruction overlap
    consolidate_max_unexplained: float = 0.5  # existing-name tokens the new request never mentions

    @classmethod
    def from_env(cls) -> "RegistryConfig":
        return cls(
            strategy=os.getenv("OPENPOKE_ROSTER_STRATEGY", "budgeted").strip().lower() or "budgeted",
            max_roster=_env_int("OPENPOKE_ROSTER_MAX", 12),
            hot_cache_size=_env_int("OPENPOKE_ROSTER_HOT", 4),
            retrieval_k=_env_int("OPENPOKE_ROSTER_RETRIEVAL_K", 8),
            min_score=_env_float("OPENPOKE_ROSTER_MIN_SCORE", 0.0),
            recency_weight=_env_float("OPENPOKE_ROSTER_RECENCY_WEIGHT", 0.6),
            transcript_boost=_env_float("OPENPOKE_ROSTER_TRANSCRIPT_BOOST", 1.0),
            pinned_slots=_env_int("OPENPOKE_ROSTER_PINNED_SLOTS", 3),
            fuzzy=os.getenv("OPENPOKE_ROSTER_FUZZY", "0") == "1",
            consolidate=os.getenv("OPENPOKE_AGENT_CONSOLIDATE", "1") != "0",
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
# Words that carry no identity in an agent *name* ("Email to Alice about lunch" -> alice, lunch).
_NAME_NOISE = frozenset(
    "email emails follow up with reminder reminders thread task about re fwd draft send check ask find".split()
)
_STOPWORDS = frozenset(
    """a an the and or but if then of to in on at for from by with about as into like through
    after over between out against during without before under around among is are was were be
    been being have has had do does did will would shall should can could may might must i me my
    we our you your he him his she her it its they them their what which who whom this that these
    those am did do does doing yes no not so too very just also please let lets hey hi ok okay
    email emails agent agents message messages send sent reply replied""".split()
)


_POSSESSIVE_RE = re.compile(r"(?<=\w)[\u2019']s\b")
_CAPITALISED_RE = re.compile(r"\b[A-Z][a-z]+")
_PERIOD_TOKENS = frozenset(
    "january february march april may june july august september october november december "
    "jan feb mar apr jun jul aug sep sept oct nov dec q1 q2 q3 q4 monday tuesday wednesday thursday "
    "friday saturday sunday".split()
)


def tokenize(text: str) -> List[str]:
    text = _POSSESSIVE_RE.sub("", text)
    return [tok for tok in _TOKEN_RE.findall(text.lower()) if tok not in _STOPWORDS]


def _entities(text: str) -> set:
    """Capitalised words that do not start the text: in instructions these are people and dates."""
    text = _POSSESSIVE_RE.sub("", text)
    return {
        m.group(0).lower() for m in _CAPITALISED_RE.finditer(text) if m.start() > 0
    } - _STOPWORDS


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
        log_dir: Optional[Path] = None,
    ) -> None:
        self._path = path
        self.config = config or RegistryConfig.from_env()
        self._clock = clock
        self._has_pending_triggers = has_pending_triggers or (lambda _name: False)
        self._legacy_roster_path = legacy_roster_path
        self._log_dir = log_dir  # per-agent log files; their mtime dates a migrated agent
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
                last_used = self._legacy_last_used(name, now)
                self._records[name] = AgentRecord(
                    name=name, created_at=last_used, last_used_at=last_used, last_reason="migrated"
                )
                changed = True
        if changed:
            self.save()

    def _legacy_last_used(self, name: str, fallback: float) -> float:
        """Date a migrated agent by its log file's mtime, so upgrade keeps real recency."""
        if self._log_dir is None:
            return fallback
        try:
            from .log_store import _slugify  # local import: avoid cycles at module load

            log = self._log_dir / f"{_slugify(name)}.log"
            if log.exists():
                return float(log.stat().st_mtime)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"could not date legacy agent {name}: {exc}")
        return fallback

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

    def find_duplicate(self, name: str, instructions: str = "") -> Optional[AgentRecord]:
        """Return a live agent that ``name`` would duplicate, if any.

        The Interaction Agent names agents freely ("Email to Alice about lunch" today,
        "Alice Lunch" tomorrow). A new agent is redirected to an existing non-archived one when
        either the names agree in both directions, or they share a token and the *instructions*
        overlap the existing agent's recorded work. Three guards stop wrong merges: the names
        name different periods ("October" vs "November"), the instructions name different
        people ("Alice" vs "Bob"), or the existing name carries topic words the new request never
        mentions ("Alice lunch" must not land on "Alice lunch invoice reconciliation").
        """
        if not self.config.consolidate or name in self._records:
            return None
        new_tokens = set(tokenize(name)) - _NAME_NOISE
        if len(new_tokens) < 2:
            return None  # too little signal to merge on
        instr_tokens = set(tokenize(instructions)) - _NAME_NOISE
        new_entities = _entities(instructions)
        now = self._clock()
        best: Optional[Tuple[float, AgentRecord]] = None
        for record in self._records.values():
            if self.status_of(record, now) == AgentStatus.ARCHIVED:
                continue
            existing_tokens = set(tokenize(record.name)) - _NAME_NOISE
            if not existing_tokens:
                continue
            shared = new_tokens & existing_tokens
            if not shared:
                continue
            # Guard 1: different periods in the names ("October invoice" vs "November invoice").
            if (new_tokens & _PERIOD_TOKENS) and (existing_tokens & _PERIOD_TOKENS) and not (
                new_tokens & existing_tokens & _PERIOD_TOKENS
            ):
                continue
            # Guard 2: the requests name different people/dates and none in common.
            existing_text = " ".join([record.description, *record.recent_requests])
            existing_entities = _entities(existing_text)
            if new_entities and existing_entities and not (new_entities & existing_entities):
                continue
            # Guard 3: the existing agent is about something the new request never mentions.
            unexplained = existing_tokens - new_tokens - instr_tokens
            if len(unexplained) / len(existing_tokens) >= self.config.consolidate_max_unexplained:
                continue
            overlap = len(shared) / len(new_tokens)
            reverse = len(shared) / len(existing_tokens)
            symmetric = (
                overlap >= self.config.consolidate_min_overlap and reverse >= self.config.consolidate_min_overlap
            )
            existing_all = set(tokenize(record.retrieval_text()))
            instr_overlap = (len(instr_tokens & existing_all) / len(instr_tokens)) if instr_tokens else 0.0
            if symmetric or instr_overlap >= self.config.consolidate_instruction_overlap:
                score = overlap + reverse + instr_overlap + 0.01 * (record.last_used_at / max(now, 1.0))
                if best is None or score > best[0]:
                    best = (score, record)
        return best[1] if best else None

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
        recent_agent_names: Sequence[str] = (),
    ) -> List[Tuple[AgentRecord, float]]:
        """BM25 search over agent metadata. Returns (record, score) with score > min_score.

        ``recent_agent_names`` are agents the conversation itself just mentioned; they get a
        fixed boost so "nudge Tara again" resolves to the Tara thread on screen, not the one
        BM25 happens to rank first.
        """
        recent = set(recent_agent_names)
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
            if record.name in recent:
                score += self.config.transcript_boost
            if score > self.config.min_score:
                scored.append((record, score))
        if not scored and self.config.fuzzy:
            scored = self._fuzzy_search(candidates, query_tokens, now)
        scored.sort(key=lambda pair: (-pair[1], -pair[0].last_used_at))
        return scored[:limit]

    def _fuzzy_search(
        self, candidates: List[AgentRecord], query_tokens: List[str], now: float
    ) -> List[Tuple[AgentRecord, float]]:
        """Character-trigram fallback: 'Priyaa' still finds Priya. Only runs when BM25 found nothing."""
        scored: List[Tuple[AgentRecord, float]] = []
        long_query = [t for t in query_tokens if len(t) >= 4]
        if not long_query:
            return scored
        for record in candidates:
            name_tokens = [t for t in tokenize(record.name) if len(t) >= 3]
            best = max(
                (_trigram_similarity(q, n) for q in long_query for n in name_tokens), default=0.0
            )
            if best >= self.config.fuzzy_min_similarity:
                idle = max(0.0, now - record.last_used_at)
                recency = 0.5 ** (idle / self.config.recency_half_life_s)
                scored.append((record, best + self.config.recency_weight * recency))
        return scored

    def select(
        self, query: str = "", *, now: Optional[float] = None, recent_agent_names: Sequence[str] = ()
    ) -> RosterSelection:
        """Pick the roster to show the Interaction Agent for this turn.

        ``recent_agent_names``: agents named in the last few conversation entries, most recent
        first. They are always shown (this is the signal upstream relied on for "did she
        reply?") and boosted in retrieval.
        """
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

        # 0) agents that own pending reminders: always reachable, up to half the budget, most
        #    recent first. Hiding one leads to a duplicate agent *and* a duplicate reminder.
        pinned_budget = max(0, min(self.config.pinned_slots, self.config.max_roster // 2))
        pinned_live = [r for r in sorted(live, key=lambda r: -r.last_used_at) if pinned(r.name)]
        for record in pinned_live[:pinned_budget]:
            chosen[record.name] = RosterEntry(record, statuses[record.name], True, "pinned")

        # 1) agents the conversation just mentioned, most recent mention first
        live_by_name = {r.name: r for r in live}
        for name in recent_agent_names:
            record = live_by_name.get(name)
            if record is None or name in chosen or len(chosen) >= self.config.max_roster:
                continue
            chosen[name] = RosterEntry(record, statuses[name], pinned(name), "transcript")

        # 2) hot cache: most recently used live agents
        for record in sorted(live, key=lambda r: -r.last_used_at)[: self.config.hot_cache_size]:
            if record.name not in chosen and len(chosen) < self.config.max_roster:
                chosen[record.name] = RosterEntry(record, statuses[record.name], pinned(record.name), "hot")

        # 3) retrieval: relevance to the current message
        if query.strip():
            for record, score in self.search(
                query, limit=self.config.retrieval_k, include_archived=False, statuses=statuses,
                recent_agent_names=recent_agent_names,
            ):
                if record.name in chosen:
                    chosen[record.name].score = max(chosen[record.name].score, score)
                    continue
                if len(chosen) >= self.config.max_roster:
                    break
                chosen[record.name] = RosterEntry(
                    record, statuses[record.name], pinned(record.name), "retrieved", score
                )

        # 4) fill any remaining budget with the next most recent live agents
        if len(chosen) < self.config.max_roster:
            for record in sorted(live, key=lambda r: -r.last_used_at):
                if len(chosen) >= self.config.max_roster:
                    break
                if record.name not in chosen:
                    chosen[record.name] = RosterEntry(
                        record, statuses[record.name], pinned(record.name), "recent"
                    )

        # Most recently used first. Relevance decided *which* agents are shown; recency decides
        # the order, because on ambiguous follow-ups ("nudge Tara again") the user almost
        # always means the thread they touched last, and models weight list order.
        shown = sorted(chosen.values(), key=lambda e: -e.record.last_used_at)[: self.config.max_roster]
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
            f"Showing {len(selection.shown)} of {selection.total} agents, most recently used first "
            f"(selected by recency + relevance to the current message); "
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


def _trigrams(token: str) -> set:
    padded = f"  {token} "
    return {padded[i : i + 3] for i in range(len(padded) - 2)}


def _trigram_similarity(a: str, b: str) -> float:
    ta, tb = _trigrams(a), _trigrams(b)
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


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
            log_dir=_DATA_DIR / "execution_agents",
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
