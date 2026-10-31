# OpenPoke — agent overload, fixed and measured

Fork of [shlokkhemani/openpoke](https://github.com/shlokkhemani/openpoke) (an open-source
reconstruction of Poke's multi-agent architecture; see the
[explainer](https://www.shloked.com/writing/openpoke)). Upstream setup instructions are in
[README.upstream.md](README.upstream.md).

The author names **execution-agent overload** as the system's main open problem: every
request passes the full roster of agents to the Interaction Agent, and after days of use
hundreds compete for attention. This fork treats that as *two* coupled problems (the roster,
and each agent's unbounded history), fixes both, adds the cheap wins around them, and ships
the tests and evaluators to show what changed.

This is the second round. The first round's claims were reviewed adversarially; the review
found three correctness bugs, a handicapped baseline, and headline cells that rested on one
query. This round fixes those, re-measures with a fair baseline and confidence intervals, and
adopts three results from the 2026 agent-memory and routing literature where the evaluators
here could verify them. Every number below names the file it comes from.

## Results at a glance

### Live routing: does the real Interaction model pick the right agent?

Claude Sonnet 4.5 via Anthropic's native Messages API (`bash evals/run_live.sh`). 150 agents,
3 workload seeds × 40 follow-up queries, equal numbers of each query kind, and **both arms see
the same conversation history** (the last 6 interactions). The first round stripped the history,
which handicapped upstream on exactly the queries it relies on history for; that is corrected.
Source: [`docs/live/round2/live_routing_results_history.json`](docs/live/round2/live_routing_results_history.json),
per-query log in [`live_routing_log.txt`](docs/live/round2/live_routing_log.txt).

| n = 120 per arm | Upstream (all 150 names) | This fork (budgeted roster + `search_agents`) |
|---|---|---|
| **Correct existing agent chosen** | **49% (95% CI 40–58)** | **74% (95% CI 66–81)** |
| …person + topic ("what did Bob say about the flight") | 90% (74–96) | 87% (70–95) |
| …person only ("nudge Hannah again") | 37% (22–55) | **93% (79–98)** |
| …topic only ("any update on the tax documents?") | 40% (25–58) | **83% (66–93)** |
| …keyword-free ("ok send it", "and then?") | 30% (17–48) | 33% (19–51) |
| Wrong agent chosen | 42 / 120 | **9 / 120** |
| Did not delegate (asked the user instead) | 19 | 22 |
| New agent spawned for an existing thread | 2 | 1 |
| Prompt tokens per query (cached) | 7,307 (4,019) | **5,262 (3,779)** |
| Cost of 120 queries | $1.56 | **$0.91** |

What the keyword-free row means: on "ok send it", "any news?" and "did she reply?" the fork
delegates to the right thread (9/9 when it delegates; upstream sent "any news?" to the wrong
agent 4 times). On "and then?", "yes do that", "go ahead" and "what did they say" **both** models
ask the user which thread they mean. With only a synthetic transcript to go on that is the
reasonable answer, and the fork never guesses wrong (0 wrong picks on this row vs 4 upstream);
the metric counts the question as a miss.

Where the fork still misses: 6 of 120 targets were outside the 12-entry roster; the model called
`search_agents` for 2 of them (both found) and did not for the other 4. The search tool works;
the model under-uses it. The person-only misses are all one person with several live threads
and a summary that reads as closed ("Rosa confirmed the lunch; nothing pending").

### Live memory: what survives compaction (LLM extractor, Claude Haiku 4.5)

Source: [`docs/live/round2/memory_results_live.json`](docs/live/round2/memory_results_live.json).
Runtime budget (12k chars of raw log before compaction); the two shorter logs never compact.
Tool results are written the way the runtime writes them: compacted to 500 chars in the log,
in full in the raw journal that `search_history` reads.

| 136-entry agent log | Upstream | This fork |
|---|---|---|
| History in the system prompt | 6,220 tokens | **2,548 tokens (41%)** |
| Surface facts / reminders in the prompt | 100% / 100% | **100% / 100%** |
| "Do not …" constraints the user stated | 100% | **100% (verbatim commitments tier)** |
| Corrected amounts: new value present | 100% | **100%** |
| Corrected amounts: old value still asserted as a fact | n/a (no facts block) | **29%** (2 of 7; see limitations) |
| Facts buried deep in a tool payload: in the prompt | 0% (500-char cap) | 0% (500-char cap) |
| …recoverable with `search_history` | n/a | **100%** |

The first round measured "buried" needles on payloads stored uncompacted, which the runtime
never does; that row is now measured on the real write path, and the answer for the prompt is
the same as upstream's. The raw journal is what makes the difference.

### Live email (prefilter + Claude Haiku 4.5 classifier, 46 labelled emails)

Source: [`docs/live/round2/email_results.json`](docs/live/round2/email_results.json),
[`jev_bench_live.json`](docs/live/round2/jev_bench_live.json).

| | Value |
|---|---|
| Pre-filter skips without a model call | 12 / 46 (26%), **0 important emails dropped** |
| Classifier precision / recall / F1 on what reaches it | 1.00 / 0.81 / 0.89 |
| Classifier misses | a human's lunch reply, a bank balance alert, a recruiter message, a visa status update |
| Jev typed yes/no instead of the LLM: recall, p50 latency, $/120 emails | 0.86, 0.16 s, $0.003 (LLM: 0.90, 1.1 s, $0.16) |
| Jev gate + LLM summary (what ships with `OPENPOKE_JEV=1`) | 0.81, 1.1 s, $0.09 |
| Jev as agent router (40 queries, one seed) | 42.5% vs LLM 87.5% — **off by default** |

The first round's prefilter would have dismissed the bank alert, the recruiter message and the
visa update with no model call at all (plus a Google Groups outage and a Slack sign-in code,
which the classifier now catches). Those are now the classifier's misses, visible and tunable.

### Offline (deterministic, no key — `python -m evals.run_all`)

Sources: [`docs/live/round2/roster_results.json`](docs/live/round2/roster_results.json),
[`roster_paraphrase_results.json`](docs/live/round2/roster_paraphrase_results.json),
[`memory_results.json`](docs/live/round2/memory_results.json), [`cost_sensitivity.json`](docs/live/round2/cost_sensitivity.json).

| | Upstream | This fork |
|---|---|---|
| Roster tokens per turn, 400 agents | 4,476 and growing linearly | **507, flat** (12 entries) |
| Correct agent in the roster, 400 agents (300 queries) | 100% | 93%, **99% with `search_agents`** |
| 32 hand-written paraphrases ("yuki car thing", "Priyaa invoice?") | — | 91% in roster, **100% with search** |
| History in prompt, 274 entries (heuristic extractor, no LLM) | 12.6k tokens | **3.7k (29%)**; commitments and constraints 100%, surface facts 63%, corrected values current 43%, buried facts 0% in prompt but 100% recoverable via `search_history` |
| Modelled cost, 200 agents, 40 turns/day, long (136-entry, nearest measured) histories | $5.34/day | **$3.21/day (−40%)** |

![roster tokens vs agents](docs/roster_tokens_vs_agents.png)
![memory tokens vs history](docs/memory_tokens_vs_history.png)

The cost model's headline moves with two assumptions, so the sensitivity table is part of the
result: at 1.2 execution runs per turn the saving is 16% with short (34-entry, uncompacted)
histories and 40% with long (136-entry) ones; at 0.5 runs per turn, 22% and 37%. Each row uses
the nearest measured history size. The roster term dominates for short-lived agents, the
history term for long-lived ones. Neither flips sign.

## The problem, located in code

Read-through notes with line references: [ARCHITECTURE.md](ARCHITECTURE.md). The short version:

1. **Horizontal overload** — `services/execution/roster.py` is a JSON list of agent *names*.
   `_render_active_agents()` emits all of them, every turn, with no description, recency or
   pruning. Cost is O(#agents) per turn and reuse decisions degrade as names collide.
2. **Vertical overload** — `ExecutionAgent.build_system_prompt_with_history()` embeds the
   agent's **entire** append-only log (raw tool payloads included) in the system prompt.
   The main conversation is summarised; execution agents never are.
3. Everything runs on the strong model, including email classification on every inbound mail.
4. Triggers wake agents *by name*, so any pruning scheme that forgets an agent orphans its reminders.
5. There were no tests.

## What I built

### 1. `AgentRegistry` — budgeted roster (`server/services/execution/registry.py`)

Replaces the flat roster with per-agent metadata (created/last-used, use count, first
instruction, rolling one-line summary) and a per-turn selection:

- **Lifecycle**: idle > 48h → *dormant*; idle > 14d → *archived* (out of the prompt, still searchable).
  Agents that own a pending trigger are never archived, and the 3 most recent of them are
  always in the roster (a hidden reminder owner means a duplicate agent *and* a duplicate
  reminder). Three slots, not more: six cost 8 points of roster recall at 400 agents.
- **Transcript-aware**: agents named in the last 20 conversation entries are always shown and
  boosted in retrieval. This is the signal upstream used for "did she reply?"; the first round's
  roster ignored it.
- **Hot cache**: the 4 most recently used agents are always shown.
- **Retrieval**: BM25 + token overlap over name/description/summary/recent requests, blended with
  a recency prior (half-life 3 days). An optional character-trigram fallback for misspellings
  (`OPENPOKE_ROSTER_FUZZY=1`) is off by default: the paraphrase eval shows it adds nothing.
- **Budget**: at most 12 entries, most recent first, each with status, last-used and summary,
  followed by a `<roster_note>` saying how many are hidden.
- **`search_agents` tool** for the Interaction Agent: the self-correction step when the agent
  it needs is not listed.
- **Duplicate consolidation**: a new agent is redirected to an existing one only on evidence:
  symmetric name agreement, or a shared name token plus overlap between the new instructions
  and the existing agent's recorded work; never across different months/people, and never when
  the existing name carries topic words the new request does not mention. The first round's
  name-only rule merged "Alice lunch" into "Alice lunch invoice reconciliation" (its own
  comment's counter-example) and missed "Liam Flight Cancellation" vs "Flight Confirmation with
  Liam"; both are now table tests.
- Legacy `roster.json` is migrated with each agent dated by its log file's mtime, so an
  upgrade keeps real recency instead of marking everything "used just now".

`OPENPOKE_ROSTER_STRATEGY=all` reproduces upstream's `<active_agents>` block byte-for-byte.
(The system prompt and tool list are the fork's in both modes; the eval baseline is
"upstream roster, fork everything else".)

*Why BM25 and not embeddings?* Agent descriptions are short and name-heavy; lexical overlap on
a person's name is the dominant signal. The paraphrase set (abbreviations, synonyms, typos)
reaches 91% in roster and 100% with search, so there is no measured gap for a dense index to
close. Tool-to-Agent Retrieval (arXiv 2511.01854) reports +19% Recall@5 from hybrid retrieval on
MCP tool sets; that is the experiment to run if a real inbox shows paraphrase misses.

### 2. `AgentMemory` — tiered per-agent memory (`server/services/execution/memory.py`)

```
## Commitments (verbatim)   reminders and rules the user stated; never summarised
# Memory (compressed)
## Facts                    key → value, with temporal invalidation (old value kept as "changed")
## Open threads             what we're waiting on
## Timeline                 one-line dated episodes
# Recent History            last 12 log entries, verbatim
```

- **Commitments tier.** Sentences that read as a standing rule or a reminder ("remind me…",
  "never cc…", "always…", "from now on…", an imperative "do not…" without a one-off marker like
  "yet" or "for now") are captured by regex when the request is dispatched and rendered first,
  above the compressed memory, for young and old agents alike, with a note that the current
  request wins on conflict. They never pass through an extractor, and a request that arrives
  while a compaction is running cannot be overwritten by it.
  Source: "Benchmarks Are Surprisingly Compaction-Friendly" (arXiv 2610.31778): behavioural
  constraints are the content summaries drop first, and the loss compounds per compaction. The
  live eval shows 100% retention of planted "do not" rules through compaction.
- **Compaction** runs in the background after an agent's result is delivered (the first round
  awaited it inside the 90-second execution timeout, so a slow compactor could turn a finished
  task into "timed out"). An extraction that returns nothing falls back to the heuristic; if that
  is empty too the cursor does not advance, so entries are never silently lost. Memory files are
  written atomically.
- **Facts** supersede on a normalised key ("Alice's email address" and "alice email" are the
  same fact). The extractor is one cheap-model call with a strict JSON contract; a deterministic
  heuristic is the fallback and powers the offline evals.
- **`search_history` tool** for execution agents: BM25 plus exact-match over the agent's log
  *and* a raw journal of full tool results (the log keeps the 500-char compacted form because
  it is rendered into prompts), returning a snippet windowed on the match. Anything compaction
  or the write-time cap dropped is one call away. Source: AMA-Bench (arXiv 2602.22769): lossy
  memory with similarity retrieval loses to keeping the trajectory retrievable, and retrieval
  exposed as a tool the agent queries itself beats fixed top-k. Measured: 100% of facts buried
  in tool payloads are recoverable, in both the live and offline runs.
- Each compaction pushes a one-line summary into the registry, so the roster shows *what an
  agent is doing*.

### 3. Model router, usage ledger, prompt caching (`server/services/llm/`)

All LLM call sites go through `complete(role=...)`. Roles map to a strong or cheap tier
(user-facing interaction and execution stay strong; email classification, summarisation, email
search and memory compaction go cheap), overridable per role via env. Every call is recorded
with tokens, cache reads (billed at 10%), cache writes (125%) and estimated cost;
`GET /api/v1/meta/usage` exposes totals by role and model, `GET /api/v1/meta/agents` the registry.
`OPENPOKE_ROUTING=strong_only` reproduces upstream.

*Why not a learned router?* Two 2026 studies measured them: LLMRouterBench (arXiv 2601.07206)
finds most routers, including commercial ones, do not beat the best single model, and "Dynamic
LLM Routers are Often Misguided" (arXiv 2610.02762) finds none of six commercial routers beats
random choice between two well-chosen models at matched cost. Routing by call role, where the
quality bar is known at the call site, is the defensible version of the idea.

**Prompt caching.** The system prompt and tool schemas are marked as a cacheable prefix (native
Messages API via `OPENPOKE_LLM_API=anthropic`; Anthropic's OpenAI-compatible endpoint ignores the
markers). The execution agent's prompt is split into a static block (cached) and the history
block (not), because a cache block that contains the history is a cache write on every run.
Measured on a three-turn live session ([`docs/live/round2/cache_by_role_probe.json`](docs/live/round2/cache_by_role_probe.json)):
interaction 83% of prompt tokens served from cache, execution 61%. The live routing run above
served 72% of the fork's prompt tokens from cache. The native adapter retries dropped connections and overloaded responses with
backoff; a 400 only disables caching if it is about the cache markers.

### 4. Email pre-filter (`server/services/gmail/prefilter.py`)

A routing layer, not a classifier: only Gmail-labelled promotions with an unsubscribe footer,
and spam, skip the model. Everything else, including bulk-looking senders and forum/social
categories, reaches the cheap-model classifier. The first round's rules dismissed Google Groups
outage threads, bank alerts, sign-in codes and recruiter mail on a footer match; six such emails
are now in the fixture set and the property "no important email is ever dropped by the rules"
is checked against all 46. Classification runs concurrently (bounded, default 4).

### 5. Jev typed decisions (`server/services/llm/jev.py`) — opt-in, benchmarked

Two of OpenPoke's decisions are classification, not generation. [Jev](https://docs.typesafe.ai)
answers typed questions (yes/no with a probability, or a choice) in ~0.16 s. With
`OPENPOKE_JEV=1` and `TYPESAFE_API_KEY`, a yes/no gates the email classifier (Jev decides; the
cheap model only writes the summary). The agent-routing hint is off by default: measured live,
Jev picks the right agent 42.5% of the time (abstaining on 17 of 40) against the LLM's 87.5%.
Email: Jev alone reaches recall 0.86 at $0.003 per 120 emails vs the LLM's 0.90 at $0.16; the
shipped gate-plus-summary pipeline is 0.81 at $0.09, so the honest saving is 40%, not the 150×
of the decision alone.

## Tests and evaluators

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r server/requirements.txt pytest pytest-asyncio matplotlib

python -m pytest                 # 165 tests, ~1s, no network
python -m evals.run_all          # all offline evals -> evals/out/*.json + charts
bash evals/run_live.sh           # every live evaluator (needs a key; ~$8)
```

| Suite | What it checks |
|---|---|
| `tests/test_registry.py` | lifecycle, trigger pinning and reserved slots, trigger touch, budget, hot cache, retrieval reserved ahead of the hot cache, transcript boost with a cap, archived-but-searchable, upstream-identical `all` rendering, migration from log mtime, an 11-row consolidation table incl. full names and capitalised instructions, trigram fallback |
| `tests/test_memory.py` | upstream-identical before first compaction, prompt bound + verbatim tail, incremental compaction, fact supersession with normalised keys, threads, reminder and constraint commitments surviving 6 compactions and a concurrent compaction, regex precision on one-off instructions, empty-extraction fallback and cursor safety, atomic save, real background scheduling and shutdown, LLM extractor parsing |
| `tests/test_history_tool.py` | `search_history` finds a reference that only the raw journal holds, windows the snippet on the match, ranks lexical matches, handles empty queries, is registered |
| `tests/test_router_and_gateway.py` | tiering, overrides, usage from provider or estimated, cache block only on cacheable blocks, cache-write pricing, cache disable only on a cache_control rejection, family prices for dated model ids |
| `tests/test_anthropic_native.py` | message/tool/system conversion both ways, key precedence, retry on transport errors and 529, no retry on 400 |
| `tests/test_email_prefilter.py` | every important fixture reaches the model (incl. the six adversarial ones); only spam and labelled promotions are hard skips |
| `tests/test_jev.py` | answer parsing, gate skips/permits, fallback, routing hint on/off, offline oracle bench is live |
| `tests/test_interaction_e2e.py` | the real `InteractionAgentRuntime` loop with a scripted LLM: bounded roster, target genuinely absent from the first prompt, `search_agents` round-trip, one roster selection per turn, transcript name extraction |
| `tests/test_payloads_and_misc.py` | payload compaction keeps thread ids and per-email fields, the real watcher classifies with bounded concurrency, compaction never blocks the result, static/history prompt split |
| `tests/test_evals.py` | stratified workloads, transcript replay, Wilson intervals, search-call counting, stale-fact scoring, paraphrase resolution, cost sensitivity |

| Evaluator | Measures | Needs key |
|---|---|---|
| `evals.roster_eval` | roster tokens/turn and recall across 10–400 agents, upstream vs budgeted, by query kind; `--paraphrase` scores 32 hand-written follow-ups with and without the trigram fallback | no |
| `evals.memory_eval` | prompt size and retention of surface, buried, reminder, constraint and *corrected* facts (new value present, old value not asserted), plus recovery via `search_history` | no (`--live` for the LLM extractor) |
| `evals.email_eval` | pre-filter skip rate/precision; `--live` pipeline P/R/F1 with the ids it missed | no / yes |
| `evals.cost_eval` | $/user/day under four configurations plus a sensitivity table over history length and runs per turn | no |
| `evals.jev_bench` | email gating and routing hint, offline (oracle) or live | no / yes + TypeSafe |
| `evals.live_routing_eval` | the real Interaction model choosing an agent: `--history` for both arms, `--stratified` kinds, `--seeds`, Wilson CIs, `search_agents` call/need/hit counts, the model's words on every non-delegation | yes |

The synthetic workload (`evals/synthetic.py`) is seeded and adversarial: the same person recurs
across topics, the same topic across people, names follow the five patterns the Interaction
Agent uses, and a quarter of queries carry no retrievable words.

### How to read the numbers, and what they do not show

- **The 49% → 74% routing gap is the only headline that clears the noise**, and the two
  intervals do not overlap. Per-kind rows have n = 30 and ±15-point intervals; treat them as
  direction, not magnitude. Two runs of identical code on the first round's 40 queries disagreed
  on 4 queries, which is why this round pools 120.
- **Keyword-free follow-ups are hard for both models even with history.** The fork's gain there
  is "never wrong" rather than "always right".
- **The model under-uses `search_agents`** (2 calls in 6 cases that needed one). The tool is
  proven; prompting the model to use it is the open item.
- **Stale facts.** With the LLM extractor, 2 of 7 corrected amounts kept the old value as a
  separate active fact. Key normalisation helps only when the extractor reuses the key; the
  next step is an entity-plus-attribute key schema in the extractor contract. The metric was
  tightened twice during this round (it first counted the timeline's "was $500" and the fact's
  own "(corrected from $500)" as stale, scoring 86%); the 29% is from a fresh live run with the
  final metric.
- **The 100% commitment and constraint rows measure the mechanism**: the planted needles use
  the regex's own trigger words. How often the regex misses a real rule, or records a one-off
  as a rule, is not measured; the regex is deliberately narrow and the block tells the model
  the current request wins.
- **The email classifier misses 4 of 21 important emails**, including a bank alert. That is a
  prompt problem in `importance_classifier.py`, now visible because the rules no longer hide it.
- **The cost model is a model.** Its two driving assumptions are printed and swept.
- **Consolidation is unit-tested, not live-measured**: the live eval never creates agents.
- **Not everything turns off with the upstream flags.** `search_history` and its prompt line
  are always present, and consolidation has its own switch (`OPENPOKE_AGENT_CONSOLIDATE=0`).
  The trigram fallback is off and, on the paraphrase set, inert.
- The live workload is synthetic. The code paths are real, the inbox is not.

## Running the server

As upstream (`README.upstream.md`). New knobs are documented in `.env.example`; defaults turn
everything on. Set `OPENPOKE_ROSTER_STRATEGY=all`, `OPENPOKE_AGENT_MEMORY=0` and
`OPENPOKE_ROUTING=strong_only` to get upstream behaviour for A/B comparison on a real inbox.
`OPENPOKE_LLM_API=anthropic` with `ANTHROPIC_API_KEY` uses the native API (prompt caching works there).

## Gaps I saw beyond the brief

Fixed in this fork:
- **Orphaned reminders** — the registry never archives trigger owners, shows the most recent
  three, and trigger fires re-activate their agent.
- **No per-agent memory** — see above; it dominates cost for long-lived agents.
- **Classifier on every email, and then rules that dropped alerts** — a prefilter that routes
  instead of dismissing.
- **Constraints lost to summarisation** — the verbatim commitments tier.
- **Nothing recoverable after compaction** — `search_history`.
- **Zero tests** — 165 now, plus seven evaluators.

Flagged, not built (what I'd do next, in order):
1. **Calls per turn.** A three-turn session costs 11 LLM calls; this work cuts tokens per call
   and leaves the announce/delegate/process/announce loop alone. Merging announce + delegate and
   streaming the first execution result are the largest remaining cost and latency levers.
2. **Make the model search.** The roster note already says to call `search_agents`; a hard
   rule ("before creating an agent, search") plus a "did you mean" error from
   `send_message_to_agent` when a near-duplicate exists.
3. **Extractor key schema** so corrections always supersede; re-run `memory_eval --live`.
4. **Classifier prompt** for alerts and recruiter mail, measured on the 46-email set.
5. **Live A/B on a real inbox** with `/meta/usage` before/after.
6. **Memory merge on consolidation** for threads that already split before this change.

## Files changed

```
ARCHITECTURE.md                                   code-path notes with line refs
docs/superpowers/specs, docs/superpowers/plans    round-two design and plan
docs/live/                                        round-one live results (kept for the record)
docs/live/round2/                                 round-two live results and logs
server/services/execution/registry.py             AgentRegistry
server/services/execution/memory.py               AgentMemory (+ commitments tier)
server/services/execution/payloads.py             tool-payload compaction
server/services/execution/routing_hint.py         Jev routing hint (off by default)
server/services/llm/{router,usage,gateway}.py     routing, ledger, cache blocks
server/services/llm/jev.py                        Jev typed-decision client
server/services/gmail/prefilter.py                email routing layer
server/openrouter_client/anthropic_native.py      native Messages API adapter with retry
server/agents/interaction_agent/{agent,tools}.py  budgeted + transcript-aware roster, search_agents
server/agents/execution_agent/{agent,runtime}.py  memory-backed history, cache-stable prompt, background compaction
server/agents/execution_agent/tools/history.py    search_history tool
server/services/trigger_scheduler.py              touch agent on fire
server/services/gmail/importance_watcher.py       prefilter + concurrent classification
server/routes/meta.py                             /api/v1/meta/usage, /api/v1/meta/agents
tests/                                            165 tests + 46 labelled emails
evals/                                            7 evaluators, synthetic workload, paraphrase set, stats
```
