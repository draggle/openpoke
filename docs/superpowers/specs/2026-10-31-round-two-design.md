# Round two: make the overload fixes correct, measured, and defensible

Date: 2026-10-31. Builds on the submission at `3de3839` and the adversarial review in
`../../../../openpoke-review-findings.md` (outside the repo).

## Intent

The first round located the problem and shipped bounded fixes. The review found three
correctness bugs, a set of eval claims that do not survive replay, and research that points at
cheap, measurable improvements. This round makes the code trustworthy, makes every headline
number reproducible with a confidence interval, and adopts the three research results that the
existing evaluators can verify. The deliverable is a repo a reviewer can clone, run, and
grill without finding a number that does not hold.

## Goals

1. No silent loss: important email is never dismissed without a model; memory compaction never
   drops entries it did not absorb; commitments and constraints survive compaction verbatim.
2. No surprise on the user's critical path: compaction runs after the result is delivered.
3. Every README number is produced by one named run, with n and a 95% interval, and the
   upstream baseline sees the same conversation history the fork sees.
4. Research adopted only where an evaluator in this repo shows the gain.

## Non-goals

- Learned model routing. Two independent 2026 studies (LLMRouterBench; "Dynamic LLM Routers are
  Often Misguided") find learned routers rarely beat a well-chosen two-model pair at matched
  cost. Role-based tiers stay; the README defends them on that evidence.
- Embedding retrieval. Not adopted unless the new paraphrase eval shows lexical retrieval
  failing; a trigram fallback is the cheap first step.
- Reducing LLM calls per user turn (merging announce + delegate). Architectural; documented as
  the next lever.

## Anti-goals

- A number in the README that comes from a different run than the number beside it.
- A test that passes with a no-op implementation.
- Any change that alters behaviour when every flag is at its upstream value.

## Constraints

- Python 3.10+, no new network dependencies for offline evals.
- Live evals cost real money; budget about $10 for this round.
- All new behaviour behind existing flags or always-on only where it cannot regress upstream.

## The work

### A. Correctness fixes (each lands with a failing test first)

A1. **Prefilter routes, it does not drop.** Hard skip only for `SPAM` and for
`CATEGORY_PROMOTIONS` with an unsubscribe footer. Forums, Social, no-reply and `updates@` rules
are removed. Keep-signals scan the whole body; `sign in`/`sign-in`/`signin` all match. Six
adversarial fixtures (bank alert, Google Groups outage, Slack sign-in code, LinkedIn recruiter,
visa status, marketing digest) join `tests/fixtures/emails.json`. The skip-rate metric will fall;
the README reports the new number.

A2. **Compaction off the critical path.** `ExecutionAgentRuntime.execute` returns, then schedules
`memory.compact` as a background task (per-agent lock already serialises it). An extraction with
zero facts, threads and episodes from a non-empty batch is treated as a failure (fallback
extractor runs; if that also yields nothing, the cursor does not advance). Memory file writes are
atomic (`os.replace`).

A3. **Consolidation requires evidence from instructions.** Name-token overlap must be symmetric
(≥0.75 both ways) OR the names share a person token and the new instructions' content tokens
overlap the existing agent's retrieval text at ≥0.5. Only non-archived agents, as before. The
unused `instructions` parameter is now used. Adversarial pairs from the review become tests in
both directions.

A4. **Cache auto-disable matches the real error.** Only an error mentioning `cache_control` or
invalid `system` content disables caching; any other 400 propagates without retry.

A5. **Native adapter key and pricing.** When `OPENPOKE_LLM_API=anthropic`, `ANTHROPIC_API_KEY`
wins over the OpenRouter key passed by callers. Cache-creation tokens are passed to the ledger
and billed at 1.25x. `.env.example` documents `OPENPOKE_LLM_API`.

A6. **Registry: pinned agents are always shown; migration keeps real recency.** `select()`
reserves slots for agents with pending triggers (bounded by `max_roster`). Legacy migration sets
`last_used_at` from the per-agent log file's mtime when the log exists.

A7. **Payload compaction keeps `threadId` and projects list items.** `threadId` leaves the noise
set; list items are projected to (id, thread_id/threadId, sender/from, subject, 120-char
snippet) before the 500-char cap, so the second and third emails survive.

A8. **Fact keys are normalised** (lowercase, possessives and punctuation stripped, stopwords
removed) before supersession, so "Alice's email address" supersedes "alice email".

A9. **Offline Jev routing bench runs.** `routing_offline` and `run_live.sh` set
`OPENPOKE_JEV_ROUTING=1` for the bench only.

A10. **One roster selection per turn.** The routing-hint path reuses the selection computed
for the prompt.

### B. Research-backed additions (each with a test and an evaluator row)

B1. **Commitments tier, never re-summarised.** At write time, `record_request` detects
reminder/commitment and "do not"/"always"/"never" constraint language with a regex and stores
the sentence verbatim in a per-agent `commitments` list inside `MemoryState`. `render_history`
emits a `## Commitments (verbatim)` block ahead of facts. Compaction never touches it. Source:
"Benchmarks Are Surprisingly Compaction-Friendly" (2610.31778): constraints are the content
compaction drops most, and loss compounds per compaction. Measured by new `constraint` needles
in `memory_eval` with a value that changes mid-log.

B2. **`search_history` tool for execution agents.** BM25 over the agent's own raw log entries;
returns matching entries verbatim. Memory stays lossy but nothing is unrecoverable. Source:
AMA-Bench (2602.22769): expose retrieval as a tool the agent calls with its own query rather
than fixed top-k. Measured by a `buried` needle retrieval row in `memory_eval` (needle absent
from compacted prompt but recoverable via the tool).

B3. **Transcript-aware roster boost.** `select()` accepts `recent_agent_names`; the interaction
agent extracts agent names mentioned in the last 20 conversation entries (`[STATUS] Name:` and
user text) and those agents get a fixed boost and guaranteed inclusion. This is the signal
upstream relied on and the eval stripped. Measured by `live_routing_eval --history`.

B4. **Cache-stable execution prompt.** The execution agent's static instructions become the
cacheable system block; the history block is a second, uncached system block. Measured by
`cached_tokens` by role in the ledger during the live run.

B5. **Trigram fallback in retrieval** (flag `OPENPOKE_ROSTER_FUZZY=1`, default off): when BM25
returns nothing, score by character-trigram Jaccard over names. Measured by the new paraphrase
query set in `roster_eval`; adopted as default only if it moves recall.

### C. Evaluation rework

C1. `live_routing_eval`: `--history` replays a synthetic transcript (the last 6 interactions,
naming the agents involved, generated from the workload) into both arms; `--stratified` draws
equal numbers per query kind; `--seeds 7,8,9`; output includes per-kind n, Wilson 95% CI,
`search_agents` call count and hit rate, and the assistant text for every `None`.
C2. `memory_eval`: needles with changing values (score active-fact correctness, penalise stale
values), constraint needles, and a `search_history` recovery row. Default `budget_chars` matches
the runtime (12,000).
C3. `roster_eval`: a hand-written paraphrase query set (30 queries) alongside the templated one.
C4. `cost_eval`: sensitivity table over history length (0/40/80/160) and runs per turn; uses the
LLM extractor's measured sizes.
C5. Tests: replace the asyncio-only watcher test with one that drives the watcher; e2e test
asserts the hidden agent is absent from the first prompt; a no-op-extractor test for A2; a
consolidation table test for A3; a prefilter adversarial table for A1; cache-error matching
for A4; key precedence for A5; pinned-shown and mtime-migration for A6; payload projection for
A7; key normalisation for A8; commitments survive N compactions for B1; `search_history` for
B2; boost for B3; prompt split for B4.

### D. Live re-measurement and README

Run `live_routing_eval` with history, stratified, 3 seeds × 40 queries × 2 arms (about 240
decisions, ~$6), `memory_eval --live` with the new needles, `email_eval --live` with the new
fixtures, and the Jev bench with routing enabled. Rewrite the README results section so each
cell names its run file, n, and CI; remove the n=1 cell; replace the miss narratives with the
replayed roster lines; add the "Jev gate + LLM summary" pipeline row; add the cost sensitivity
table; cite the three papers where they drove a decision.

## Left to the builder

Exact regexes, boost magnitudes, paraphrase wording, and test fixture text. Keep all magnitudes
as named constants in `RegistryConfig`/`MemoryConfig`.
