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

import asyncio
import copy
import hashlib
import json
import os
import re
from decimal import Decimal, InvalidOperation
import time
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
    max_commitments: int = 20  # verbatim user commitments/constraints kept outside compaction
    commitment_chars: int = 300

    @classmethod
    def from_env(cls) -> "MemoryConfig":
        return cls(
            enabled=os.getenv("OPENPOKE_AGENT_MEMORY", "1") != "0",
            budget_chars=_env_int("OPENPOKE_AGENT_MEMORY_BUDGET_CHARS", 12_000),
            tail_entries=_env_int("OPENPOKE_AGENT_MEMORY_TAIL", 12),
            max_episodes=_env_int("OPENPOKE_AGENT_MEMORY_MAX_EPISODES", 40),
            max_commitments=_env_int("OPENPOKE_AGENT_MEMORY_MAX_COMMITMENTS", 20),
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
class Commitment:
    """A sentence the user asked for verbatim: a reminder, a rule, a constraint.

    Stored at write time by regex, never passed through an extractor, never dropped by
    compaction. Compaction studies show these are the first things summaries lose.
    """

    text: str
    recorded_at: str


@dataclass
class MemoryState:
    facts: List[Fact] = field(default_factory=list)
    threads: List[Thread] = field(default_factory=list)
    episodes: List[Episode] = field(default_factory=list)
    commitments: List[Commitment] = field(default_factory=list)
    compacted_through: int = 0  # number of log entries absorbed into memory
    compactions: int = 0

    def active_facts(self) -> List[Fact]:
        return [f for f in self.facts if f.active]

    def to_json(self) -> Dict[str, Any]:
        return {
            "facts": [asdict(f) for f in self.facts],
            "threads": [asdict(t) for t in self.threads],
            "episodes": [asdict(e) for e in self.episodes],
            "commitments": [asdict(c) for c in self.commitments],
            "compacted_through": self.compacted_through,
            "compactions": self.compactions,
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "MemoryState":
        return cls(
            facts=[Fact(**f) for f in data.get("facts", [])],
            threads=[Thread(**t) for t in data.get("threads", [])],
            episodes=[Episode(**e) for e in data.get("episodes", [])],
            commitments=[Commitment(**c) for c in data.get("commitments", [])],
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
    # Round three. Key normalisation only supersedes a fact when the extractor reuses the key;
    # these two carry the extractor's own knowledge of what changed:
    corrections: List[Tuple[str, str]] = field(default_factory=list)  # (old value, new value)
    correction_context: Dict[Tuple[str, str], str] = field(default_factory=dict)  # pair -> the sentence that stated it
    replaces: Dict[str, str] = field(default_factory=dict)  # new key -> existing key it updates


class Extractor(Protocol):
    async def __call__(self, entries: Sequence[LogEntry], state: MemoryState) -> Extraction: ...


# Sentence boundary: punctuation + space. Titles ("Dr. Chen") are masked first so they do not split.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_TITLE_RE = re.compile(r"\b(Dr|Mr|Mrs|Ms|Prof|St|Jr|Sr)\.(?=\s)")
# Durable phrasing: a standing rule or a reminder, not a one-off scoping instruction.
_DURABLE_RE = re.compile(
    r"\b(remind|reminder|reminders|every (day|week|month|morning|evening|monday|tuesday|wednesday|"
    r"thursday|friday|saturday|sunday)|each (day|week|month)|never|always|from now on|going forward|"
    r"don'?t ever|do not ever|under no circumstances|in future)\b",
    re.I,
)
# "do not"/"don't" count only as an imperative (sentence start, or after please/and/also/but).
_IMPERATIVE_NEG_RE = re.compile(r"^(?:(?:please|and|also|but|then)[, ]+)?(?:do not|don'?t)\b", re.I)
# One-off markers that make a "do not" about this task, not a standing rule.
_TRANSIENT_RE = re.compile(r"\b(yet|for now|this time|right now|just|first|today|tonight|the draft|until)\b", re.I)


def extract_commitments(text: str) -> List[str]:
    """Sentences that read as a standing rule or a reminder, verbatim.

    Kept deliberately narrow: the Interaction Agent writes "only", "make sure", "do not … yet"
    for one-off scoping, and recording those as permanent rules would be worse than missing a
    rule. The rendered block tells the model the current request wins on conflict.
    """
    out: List[str] = []
    masked = _TITLE_RE.sub(lambda m: m.group(1) + "\u2024", " ".join(text.split()))  # "Dr." -> "Dr․"
    for sentence in _SENTENCE_SPLIT_RE.split(masked):
        sentence = sentence.replace("\u2024", ".").strip()
        if not sentence:
            continue
        durable = bool(_DURABLE_RE.search(sentence))
        imperative_neg = bool(_IMPERATIVE_NEG_RE.search(sentence)) and not _TRANSIENT_RE.search(sentence)
        if durable or imperative_neg:
            out.append(sentence)
    return out


_KEY_POSSESSIVE_RE = re.compile(r"(?<=\w)[\u2019']s\b")
_KEY_STOPWORDS = frozenset({"the", "a", "an", "of", "for", "s"})


def normalise_key(key: str) -> str:
    """Canonical form for fact-key matching: "Alice's Email Address" == "alice email address".

    The LLM extractor writes free-form keys; without this, a changed value lands as a second
    active fact instead of superseding the old one.
    """
    text = _KEY_POSSESSIVE_RE.sub("", key.lower())
    words = [w for w in re.findall(r"[a-z0-9]+", text) if w not in _KEY_STOPWORDS]
    return " ".join(words)


# --------------------------------------------------------------------------------------
# Extractors
# --------------------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# "it is $700, not $500" / "$900 (was $850)" / "$900, was $850" -> (old, new)
_AMOUNT = r"(\$\d(?:[\d,]*\d)?(?:\.\d+)?)"
_CORRECTION_NOT_RE = re.compile(_AMOUNT + r"\s*,?\s*not\s+" + _AMOUNT, re.I)
_CORRECTION_WAS_RE = re.compile(_AMOUNT + r"\s*(?:\(|,)?\s*(?:was|previously|formerly|up from|down from)\s+" + _AMOUNT, re.I)


_AMOUNT_RE = re.compile(_AMOUNT)


def _amount_value(value: str) -> Optional[Decimal]:
    try:
        return Decimal(value.strip().replace("$", "").replace(",", ""))
    except InvalidOperation:
        return None


def _contains_amount(text: str, value: str) -> bool:
    """``value`` occurs in ``text`` as a whole amount, compared by value: "$500" matches
    "$500.00" and "$1,200" matches "$1200", but never a prefix ("$500" is not in "$5000")."""
    target = _amount_value(value)
    if target is None:
        return re.search(r"(?<![\w$.,])" + re.escape(value) + r"(?![\w.,]?\d)", text) is not None
    return any(_amount_value(found) == target for found in _AMOUNT_RE.findall(text))


def extract_corrections(text: str) -> List[Tuple[str, str]]:
    """(old, new) value pairs stated as corrections in free text."""
    return [(old, new) for old, new, _sentence in extract_corrections_with_context(text)]


def extract_corrections_with_context(text: str) -> List[Tuple[str, str, str]]:
    """(old, new, sentence) for each correction; the sentence anchors which facts it may retire."""
    out: List[Tuple[str, str, str]] = []
    seen = set()
    masked = _TITLE_RE.sub(lambda m: m.group(1) + "\u2024", " ".join(text.split()))
    for sentence in _SENTENCE_SPLIT_RE.split(masked):
        sentence = sentence.replace("\u2024", ".").strip()
        pairs = [(old, new) for new, old in _CORRECTION_NOT_RE.findall(sentence)]
        pairs += [(old, new) for new, old in _CORRECTION_WAS_RE.findall(sentence)]
        for old, new in pairs:
            if (old, new) not in seen:
                seen.add((old, new))
                out.append((old, new, sentence))
    return out


# Corrections are taken from what the user and the agent said, never from tool results: an
# email body or newsletter saying "now $49 (was $99)" is content, not a correction of memory.
_CORRECTION_TAGS = frozenset({"agent_request", "agent_response"})
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
    - reminder requests are promoted to facts so capping the timeline never drops them
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
                # Commitments (reminders) are promoted to facts so they are never dropped
                # when the timeline is capped.
                if re.search(r"\bremind(?:er|ers)?\b", text, re.I):
                    digest = hashlib.sha1(text.lower().encode()).hexdigest()[:6]
                    out.facts.append((f"reminder {digest}", text[: self.snippet_chars]))
            elif tag == "agent_response":
                out.episodes.append(f"Result: {text[: self.snippet_chars]}")
                lowered = text.lower()
                if any(k in lowered for k in ("waiting", "awaiting", "no reply", "not yet replied")):
                    out.threads.append(
                        Thread(subject=text[:60], last_action=text[: self.snippet_chars],
                               waiting_on="reply")
                    )
                if any(k in lowered for k in ("replied", "confirmed", "sent successfully")):
                    # Resolve only threads this response plausibly refers to (shared tokens).
                    words = set(re.findall(r"[a-z0-9]{3,}", lowered))
                    for thread in state.threads:
                        subject_words = set(re.findall(r"[a-z0-9]{3,}", thread.subject.lower()))
                        if len(words & subject_words) >= 2:
                            out.resolved_threads.append(thread.subject)
            for match in _EMAIL_RE.finditer(payload):
                name = _name_before(payload, match.start())
                if name:
                    out.facts.append((f"{name} email", match.group(0)))
            if tag in _CORRECTION_TAGS:
                for old, new, sentence in extract_corrections_with_context(text):
                    if (old, new) not in out.corrections:
                        out.corrections.append((old, new))
                        out.correction_context[(old, new)] = sentence
        return out


_LLM_SYSTEM_PROMPT = """You compress an execution agent's activity log into durable memory.
Return ONLY a JSON object with keys:
  "facts": [{"key": "...", "value": "...", "replaces": "..."}]   stable facts (addresses, dates, amounts, decisions, preferences).
      Keys are "<entity> <attribute>" ("Priya invoice amount", "Alice email", "dentist appointment date").
      If a fact in the existing memory changed, REUSE ITS KEY EXACTLY and give the new value. If you must use a
      different key for a fact that updates an existing one, set "replaces" to the existing key. Omit "replaces" otherwise.
  "corrections": [{"old": "$500", "new": "$700"}]   every value the log explicitly corrects ("it is $700, not $500", "$900 (was $850)").
  "threads": [{"subject": "...", "last_action": "...", "waiting_on": "..."}]   things still pending.
  "resolved_threads": ["subject", ...]   previously-open threads that are now closed.
  "episodes": ["...", ...]   one line per notable event, chronological, past tense, <=140 chars.
Be concrete. Preserve names, email addresses, dates, amounts, and any commitment or reminder the user asked for verbatim. Never list a superseded value as a current fact. Omit tool-call mechanics. (Reminders and rules the user stated are also kept verbatim in a separate commitments list; you need not repeat them, but do not contradict them.)"""


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
            raw_facts = [f for f in data.get("facts", []) if isinstance(f, dict) and f.get("key") and f.get("value")]
            replaces = {str(f["key"]).strip(): str(f["replaces"]).strip()
                        for f in raw_facts if f.get("replaces") and str(f["replaces"]).strip()}
            corrections = [(str(c.get("old", "")).strip(), str(c.get("new", "")).strip())
                           for c in data.get("corrections", []) if isinstance(c, dict) and c.get("old") and c.get("new")]
            # Corrections stated by the user or the agent are a regex fact, not a judgement
            # call: union the deterministic ones so a correction never depends on the model
            # noticing it. Tool results (email bodies) are never a source.
            context: Dict[Tuple[str, str], str] = {}
            for tag, _ts, payload in entries:
                if tag in _CORRECTION_TAGS:
                    for old, new, sentence in extract_corrections_with_context(" ".join(payload.split())):
                        if (old, new) not in corrections:
                            corrections.append((old, new))
                        context.setdefault((old, new), sentence)
            return Extraction(
                facts=[(str(f["key"]).strip(), str(f["value"]).strip()) for f in raw_facts],
                replaces=replaces,
                corrections=corrections,
                correction_context=context,
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


def _is_empty(extraction: Extraction) -> bool:
    return not (extraction.facts or extraction.threads or extraction.resolved_threads or extraction.episodes)


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
        self._locks: Dict[str, asyncio.Lock] = {}
        self._tasks: set = set()  # background compactions; strong refs so the loop cannot drop them

    def _lock(self, agent_name: str) -> asyncio.Lock:
        return self._locks.setdefault(_slugify(agent_name), asyncio.Lock())

    # ---- persistence -----------------------------------------------------------------

    def _path(self, agent_name: str) -> Path:
        return self._base_dir / f"{_slugify(agent_name)}.memory.json"

    def load(self, agent_name: str) -> MemoryState:
        path = self._path(agent_name)
        if not path.exists():
            return MemoryState()
        try:
            state = MemoryState.from_json(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            logger.warning(f"Failed to load memory for {agent_name}: {exc}")
            return MemoryState()
        # The log was cleared/truncated underneath us (e.g. DELETE /history): start over.
        if state.compacted_through > len(list(self._log_store.iter_entries(agent_name))):
            return MemoryState()
        return state

    def save(self, agent_name: str, state: MemoryState) -> None:
        try:
            self._base_dir.mkdir(parents=True, exist_ok=True)
            path = self._path(agent_name)
            # Atomic replace: a crash mid-write never leaves a truncated file behind.
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(state.to_json(), indent=2, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)
        except Exception as exc:
            logger.warning(f"Failed to save memory for {agent_name}: {exc}")

    def clear_all(self) -> None:
        for path in self._base_dir.glob("*.memory.json"):
            try:
                path.unlink()
            except Exception:
                pass

    # ---- commitments -----------------------------------------------------------------

    def note_request(self, agent_name: str, instructions: str) -> None:
        """Record any reminder/rule sentences in ``instructions`` verbatim, at write time."""
        if not self.config.enabled:
            return
        found = extract_commitments(instructions)
        if not found:
            return
        state = self.load(agent_name)
        known = {c.text.lower() for c in state.commitments}
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        for sentence in found:
            text = sentence[: self.config.commitment_chars]
            if text.lower() in known:
                continue
            state.commitments.append(Commitment(text=text, recorded_at=stamp))
            known.add(text.lower())
        state.commitments = state.commitments[-self.config.max_commitments :]
        self.save(agent_name, state)

    def _render_commitments(self, state: MemoryState) -> List[str]:
        if not state.commitments:
            return []
        from html import escape

        lines = ["## Commitments (verbatim; rules the user stated earlier — the current request wins on conflict)"]
        lines.extend(f"- {escape(c.text, quote=False)} (asked {c.recorded_at})" for c in state.commitments)
        return lines

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
        async with self._lock(agent_name):  # one compaction per agent at a time
            state = self.load(agent_name)
            if not force and not self.needs_compaction(agent_name, state):
                return None
            entries = list(self._log_store.iter_entries(agent_name))
            cutoff = max(state.compacted_through, len(entries) - self.config.tail_entries)
            batch = entries[state.compacted_through : cutoff]
            if not batch:
                return None

            extraction = await self.extractor(batch, state)
            # The extractor call awaited; note_request() may have added commitments meanwhile.
            # Commitments are owned by note_request, so take them from disk, never from the
            # state loaded before the await.
            state.commitments = self.load(agent_name).commitments
            if _is_empty(extraction):
                # A well-formed but empty answer ("nothing notable") would otherwise advance the
                # cursor and hide these entries from every future prompt. Try the heuristic once;
                # if that is empty too, leave the cursor where it is and try again next time.
                extraction = await HeuristicExtractor()(batch, state)
                if _is_empty(extraction):
                    logger.warning(f"[{agent_name}] memory extraction returned nothing; cursor not advanced")
                    return None
            now_ts = batch[-1][1] or ""
            self._merge(state, extraction, now_ts)
            # A correction that arrived after the batch (still in the verbatim tail) must not
            # leave the fact block asserting the old value right above the entry that retracts it.
            self._apply_tail_corrections(state, entries[cutoff:], now_ts)
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

    def schedule_compaction(self, agent_name: str) -> Optional["asyncio.Task[Optional[MemoryState]]"]:
        """Run compaction in the background if the raw log is over budget.

        Called by the execution runtime *after* its result is produced, so compaction (one
        cheap-model call) never sits on the user's request path or inside the batch timeout.
        """
        try:
            if not self.needs_compaction(agent_name):
                return None
            task = asyncio.get_running_loop().create_task(self.compact(agent_name))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            return task
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"[{agent_name}] could not schedule compaction: {exc}")
            return None

    def pending_tasks(self) -> set:
        return set(self._tasks)

    async def shutdown(self) -> None:
        """Cancel background compactions (saves are atomic, so cancelling loses nothing on disk)."""
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._tasks.clear()

    @staticmethod
    def _tail_corrections(tail: Sequence[LogEntry]) -> Extraction:
        extraction = Extraction()
        for tag, _ts, payload in tail:
            if tag not in _CORRECTION_TAGS:
                continue
            for old, new, sentence in extract_corrections_with_context(" ".join(payload.split())):
                if (old, new) not in extraction.corrections:
                    extraction.corrections.append((old, new))
                    extraction.correction_context[(old, new)] = sentence
        return extraction

    def _apply_tail_corrections(self, state: MemoryState, tail: Sequence[LogEntry], now_ts: str) -> None:
        extraction = self._tail_corrections(tail)
        if extraction.corrections:
            self._merge(state, extraction, now_ts)

    def _merge(self, state: MemoryState, extraction: Extraction, now_ts: str) -> None:
        # facts: temporal invalidation on key collision
        for key, value in extraction.facts:
            key_norm = normalise_key(key)
            if not key_norm or not value:
                continue
            current = next((f for f in state.facts if f.active and normalise_key(f.key) == key_norm), None)
            if current is not None:
                if current.value.strip().lower() == value.strip().lower():
                    continue
                current.invalidated_at = now_ts
                current.superseded_by = value
            # The extractor named the existing key this fact updates (a different spelling).
            replaced_norm = normalise_key(extraction.replaces.get(key, ""))
            if replaced_norm and replaced_norm != key_norm:
                for fact in state.facts:
                    if fact.active and normalise_key(fact.key) == replaced_norm:
                        fact.invalidated_at = now_ts
                        fact.superseded_by = value
            state.facts.append(Fact(key=key.strip(), value=value.strip(), recorded_at=now_ts))
        # corrections: "it is $700, not $500" invalidates any active fact still asserting $500,
        # provided its key shares a token with the fact that carries the new value (so an
        # unrelated $500 elsewhere is left alone).
        # The old value may sit in the key ("Hannah refund $3579: confirmed by Nov 4"), which is
        # how the live extractor writes amounts, so key and value are both searched. A correction
        # may only retire a fact it is *about*: one whose key shares a word with a fact carrying
        # the new value, or with the sentence that stated the correction ("Priya ... invoice").
        # With nothing to anchor on, nothing is retired.
        for old_value, new_value in extraction.corrections:
            old_norm = old_value.strip().lower()
            new_norm = new_value.strip().lower()
            if not old_norm:
                continue
            carriers = [set(normalise_key(k).split()) for k, v in extraction.facts
                        if new_norm and _contains_amount(f"{k} {v}".lower(), new_norm)]
            carriers += [set(normalise_key(f.key).split()) for f in state.facts
                         if f.active and new_norm and _contains_amount(f"{f.key} {f.value}".lower(), new_norm)]
            sentence = extraction.correction_context.get((old_value, new_value), "")
            if sentence:
                carriers.append({t for t in normalise_key(sentence).split() if not t.isdigit()})
            carriers = [c for c in carriers if c]
            if not carriers:
                continue
            for fact in state.facts:
                text = f"{fact.key} {fact.value}".lower()
                if not fact.active or not _contains_amount(text, old_norm) or (new_norm and _contains_amount(text, new_norm)):
                    continue
                fact_tokens = set(normalise_key(fact.key).split())
                if any(fact_tokens & c for c in carriers):
                    fact.invalidated_at = now_ts
                    fact.superseded_by = new_value
        # cap facts: drop oldest *invalidated* first; only then the oldest active ones
        if len(state.facts) > self.config.max_facts:
            inactive = [f for f in state.facts if not f.active]
            overflow = len(state.facts) - self.config.max_facts
            for fact in inactive[:overflow]:
                state.facts.remove(fact)
        if len(state.facts) > self.config.max_facts:
            state.facts = state.facts[len(state.facts) - self.config.max_facts :]

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
            transcript = self._log_store.load_transcript(agent_name)
            commitments = self._render_commitments(state)
            return "\n".join([*commitments, "", transcript]) if commitments else transcript

        # Display-time pass: a correction that landed in the tail since the last compaction
        # retires the fact it contradicts (not persisted; the next compaction persists it).
        recent = entries[state.compacted_through :]
        tail_extraction = self._tail_corrections(recent)
        if tail_extraction.corrections:
            state = copy.deepcopy(state)
            self._merge(state, tail_extraction, recent[-1][1] if recent else "")

        parts: List[str] = [*self._render_commitments(state), "# Memory (compressed from earlier activity)"]
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

from ...paths import data_dir

_DATA_DIR = data_dir()
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
    "Commitment",
    "Episode",
    "Extraction",
    "Fact",
    "HeuristicExtractor",
    "LLMExtractor",
    "MemoryConfig",
    "MemoryState",
    "Thread",
    "extract_commitments",
    "extract_corrections",
    "extract_corrections_with_context",
    "get_agent_memory",
    "normalise_key",
]
