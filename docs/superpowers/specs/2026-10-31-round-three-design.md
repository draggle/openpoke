# Round three: push every measured metric where round two showed the loss

Date: 2026-10-31. Builds on branch `round-two` (28 commits over `3de3839`). The user's brief:
"try everything, then run live tests, tests, etc.; get this as good as it can get."

## Intent

Round two made the numbers honest. Each row of its results table now says where the loss is:
keyword-free follow-ups are 33%, the model under-uses `search_agents`, four important emails
slip past the classifier, 2 of 7 corrected amounts keep a stale fact, the cost of a user turn
is dominated by call count rather than tokens, and two mechanisms (`search_history`, Jev gating)
ship without a measurement of how the model actually uses them. This round attacks each loss at
its measured cause, adds the two missing measurements, and runs one offline experiment (a learned
roster reranker) to settle a question a reviewer will ask. Every change lands test-first and
every claim gets a fresh live run.

## Goals (each is a README row that must move or be newly measured)

1. Keyword-free follow-ups: raise the fork's delegation rate without raising wrong picks.
2. Wrong-person picks: stop the model dispatching to an agent about a different person when the
   right agent exists; make the model search when the person it names is not on the roster.
3. Email recall: surface human replies, account alerts, recruiter mail and government status
   changes; keep precision at 1.0; report on a held-out set written before the prompt changed.
4. Stale facts: corrected amounts invalidate the old fact (target 0 of 7 stale, live).
5. LLM calls per user turn: cut the follow-up call that produces nothing after the model has
   both told the user and dispatched the agent; measure calls per turn before and after.
6. Measure `search_history` usage: how often the real execution model calls it when the answer
   is only in the raw journal, and how often it calls it when it need not.
7. Jev gate: a two-sided gate (confident-no skips the LLM, confident-yes goes straight to
   summary, the middle band gets the full classifier) tuned on recorded probabilities; fix the
   summary-only path that lost a recall point.
8. Trigger wake-ups for the same agent due in the same poll coalesce into one execution.
9. Learned roster reranker: train a logistic reranker on synthetic seeds, evaluate held-out
   seeds against the hand-tuned `select()`. Report the result either way; wire it only if it wins
   by more than the interval.

## Non-goals

- Agent hibernation (snapshot dormant logs to cold storage). The prompt is already bounded by
  tiered memory; disk is not the problem the brief names.
- Real-inbox replay. Needs the user's mail; documented as the next step, not built.
- Self-correcting routing (treating "I have no record of that" as a routing miss). Unmeasurable
  without a real inbox; the mismatch guard in goal 2 covers the detectable half.
- Embedding retrieval. Paraphrase recall is 0.91/1.00; nothing to gain at this size.

## Anti-goals

- Any change that alters behaviour with every flag at its upstream value. New behaviour is on by
  default in the fork but switchable: `OPENPOKE_INTERACTION_EARLY_EXIT=0`,
  `OPENPOKE_ROUTING_GUARD=0`, `OPENPOKE_JEV_EMAIL_LOW/HIGH`, `OPENPOKE_TRIGGER_COALESCE=0`.
- A prompt tuned on the test set without saying so. The 46-email fixture is the dev set; a new
  14-email held-out set is written before the classifier prompt is touched and reported apart.
- A metric moved by changing the metric. The synthetic recency queries change (goal 1), so both
  arms are re-run on the new workload and the README says the workload changed and why.

## Constraints

- Python 3.10+, no new runtime dependencies. `numpy` (already present via matplotlib) may be
  used by the offline reranker experiment only.
- Live budget about $12: routing both arms (3 seeds × 40), memory live, email live (dev +
  held-out), Jev bench with per-email probabilities, the `search_history` usage eval, the
  session probe.
- Upstream arm rendering stays byte-identical to upstream.

## The work

### A. Routing

A1. **Coherent keyword-free queries.** `evals/synthetic.py` draws a recency query from a set
conditioned on the most recently used agent's summary state: a thread "awaiting reply" or
"waiting on" gets "did she reply?", "did he reply?", "any news?", "what did they say", "nudge
them again"; a thread whose summary begins "Sent draft" additionally gets "ok send it", "go
ahead", "yes do that"; a thread with "nothing pending" gets "what did they say", "any news?",
"and then?"; a thread "Reminder scheduled" gets "and then?", "cancel it", "move it to 10am",
"yes do that". `Query` gains `state: str`. Seeded and deterministic as before.

A2. **`last_touched` marker.** `AgentRegistry.render` adds `last_touched="true"` to the first
(most recently used) entry of a budgeted roster. The Interaction system prompt gains one rule:
a bare follow-up with no name or topic refers to the agent marked `last_touched`; delegate to it
and do not ask which thread unless the history makes the referent genuinely ambiguous. It also
gains: if the message names a person and no listed agent is about that person, call
`search_agents` before picking another person's agent or creating one.

A3. **Routing mismatch guard.** `AgentRegistry.routing_mismatch(agent_name, message) ->
Optional[RoutingMismatch]` returns candidates when (a) the chosen agent exists, (b) the message
contains at least one token that appears in some other live agent's *name* and in none of the
chosen agent's name or first instruction, (c) all such tokens are absent from the chosen agent,
and (d) a search for the message returns at least one live agent, other than the chosen one,
whose name contains every such token. `send_message_to_agent` consults it on the reuse path:
the first call for a given (agent, message) pair returns a structured error naming up to three
candidates ("did you mean"); a repeated identical call is honoured (the model may override).
Flag `OPENPOKE_ROUTING_GUARD` (default `1`). The live eval applies the same guard and feeds the
error back as a tool result, counting `guard_fired`, `guard_helped` (final pick correct after a
fired guard) and `guard_hurt` (first pick was correct, final pick wrong).

### B. Interaction loop

B1. **Early exit.** `_run_interaction_loop(system_prompt, messages, message_type)` ends the loop
after an iteration in which, for a user turn, `send_message_to_agent` ran and a user-visible
message has been sent in this turn (this or an earlier iteration), and no `search_agents` ran in
the iteration; for an agent turn, after an iteration in which `send_message_to_user`, `send_draft`
or `wait` ran and neither `send_message_to_agent` nor `search_agents` did. An iteration that only
sends a user message on a user turn does not exit (the model may delegate next). Flag
`OPENPOKE_INTERACTION_EARLY_EXIT` (default `1`).

B2. **Session probe.** `evals/session_probe.py --live` boots the FastAPI app in-process (ASGI
transport), runs the round-two three-turn reminder session, and reports LLM calls, tokens, cached
share and cost by role, with early exit on and off. Output `evals/out/session_probe.json`.

### C. Email

C1. **Held-out set.** 14 new labelled rows (`h01`–`h14`) in `tests/fixtures/emails_heldout.json`,
written before C2: human replies with casual wording, a card-declined alert, a payroll notice,
an immigration appointment, a recruiter via a job board, plus noise that shares surface features
(an account newsletter with "your account" wording, a social notification, a promotional
"your application" for a credit card, a forum digest). `email_eval` reports dev and held-out
separately; `--live` runs both.

C2. **Classifier prompt.** `_SYSTEM_PROMPT` lists what to surface (a person replying to the
user, scheduling and confirmations included; account and money alerts: balance, declined, fraud,
failed payment, payroll; job and recruiter outreach; government, immigration, legal or medical
status changes; security codes and sign-in notices; deadlines, decisions, meeting changes) and
what not to (receipts, shipping, marketing, newsletters, social digests, product updates,
surveys), and states that an unsubscribe or preferences footer does not make a message
promotional when the body is about the user's own account, application, money or a person
writing to them. Two short illustrative examples that are not fixture rows.

C3. **Summary-only fallback.** When Jev decided and the LLM returns no usable summary, surface
the email with a generated fallback summary (sender and subject) instead of dropping it.

C4. **Two-sided Jev gate.** `classify_email_importance`: `p < low` dismisses without the LLM;
`p >= high` goes to the summary-only prompt; otherwise the full classifier prompt decides.
`OPENPOKE_JEV_EMAIL_LOW` (default `0.5`) and `OPENPOKE_JEV_EMAIL_HIGH` (default `0.5`) keep
round-two behaviour; `OPENPOKE_JEV_EMAIL_THRESHOLD` remains an alias for `LOW`. `jev_bench
--live` records per-email probability and LLM verdict, and `gate_sweep(rows)` reports recall,
precision and LLM-call count for a grid of (low, high); the README recommends the pair with
recall ≥ the LLM-only recall at the lowest LLM-call count, and states it was chosen on the dev set
with the held-out set as the check.

### D. Memory

D1. **Corrections in the extractor contract.** `Extraction` gains `corrections: List[Tuple[str,
str]]` (old value, new value) and `replaces: Dict[str, str]` (new key → existing key). The LLM
prompt asks for `"replaces"` on a fact whose key differs from the existing key it updates, and
for `"corrections": [{"old": "$500", "new": "$700"}]` when a value is explicitly corrected; keys
are `<entity> <attribute>`. `HeuristicExtractor` emits corrections from `is $N, not $M` and
`(was $M)`. `_merge` invalidates the fact named by `replaces`, and any active fact whose value
contains a corrected old value and whose key shares a token with the correcting fact's key.

D2. **`search_history` usage eval.** `evals/recall_eval.py --live`: after building and compacting
the 40-round memory agent, ask the real execution model eight questions with the execution tool
schemas (only `search_history` executable): four whose answer is only in the raw journal (the
`REF-` codes), four whose answer is in the compacted prompt (amounts). Report tool-call rate and
answer accuracy per class. Output `evals/out/recall_results_live.json`.

### E. Triggers

E1. **Coalesce.** `TriggerScheduler._poll_once` groups due triggers by agent; a group of two or
more becomes one execution whose instructions list every trigger's payload and id; success and
failure bookkeeping runs per trigger. `OPENPOKE_TRIGGER_COALESCE` (default `1`).

### F. Experiment

F1. **Learned reranker.** `evals/learned_selector.py`: features per (query, live agent):
normalised BM25 score, name-token overlap fraction, recency rank / n, log age in days, use
count, pinned, named in transcript tail, summary-says-waiting. Logistic regression by gradient
descent (numpy) on seeds 1–5 (300 queries each), evaluated as recall@12 on seeds 7–9 against
`select()`'s recall@12 on the same workloads. Output `evals/out/learned_selector.json`. Not wired
into the runtime unless it wins by more than the pooled Wilson interval width; the README
reports the number either way.

### G. Measurement and docs

- `run_live.sh` adds `recall_eval --live`, `session_probe --live`, held-out email.
- Results archived in `docs/live/round3/`; README results tables cite round three files, with
  round-two numbers beside them where the workload is unchanged and a note where it changed.
- the rehearsal notes get a round-three section (outside the repo).

## Left to the builder

Exact token rules in the mismatch guard; the wording of prompt lines; the grid for the gate
sweep; the number of training queries for the reranker.
