# Round Three Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move every round-two metric at its measured cause, add the two missing measurements, run one reranker experiment, re-measure live.

**Architecture:** Same modules as round two. Routing changes live in `registry.py` (render marker, mismatch guard), `interaction_agent/tools.py` and `system_prompt.md`; the loop change in `interaction_agent/runtime.py`; email in `importance_classifier.py`; memory in `memory.py`; triggers in `trigger_scheduler.py`. New evals: `session_probe.py`, `recall_eval.py`, `learned_selector.py`. Fixture `emails_heldout.json`.

**Tech Stack:** Python 3.10+, pytest + pytest-asyncio, rank_bm25, numpy (experiment only), httpx ASGI transport for the probe.

**Spec:** `docs/superpowers/specs/2026-10-31-round-three-design.md`

## Global Constraints

- Upstream-flag behaviour unchanged: `OPENPOKE_ROSTER_STRATEGY=all` rendering byte-identical; new switches default on in the fork, off restores round-two behaviour.
- Every production change: failing test first, then suite green (`.venv/bin/python -m pytest -q`).
- Held-out email set written (Task 6) before the classifier prompt changes (Task 7).
- Test command for every task: `.venv/bin/python -m pytest -q`.

## Review Focus

1. Mismatch guard false positive: "ask Tomasz whether Bob is coming" to a Tomasz agent must not fire (chosen agent shares a name token). Test in Task 3.
2. Early exit must not swallow a sequential announce-then-delegate (iteration with only `send_message_to_user` on a user turn keeps looping). Test in Task 4.
3. Two-sided gate with `low == high` must reproduce round-two behaviour exactly. Test in Task 8.
4. Corrections must not invalidate an unrelated fact that happens to contain the old amount (key shares no token). Test in Task 9.
5. Coalesced trigger failure must record a failure on every trigger in the group and reschedule recurring ones. Test in Task 10.

---

### Task 1: State-conditioned recency queries

**Files:** Modify `evals/synthetic.py`; Test `tests/test_evals.py`.
**Interfaces:** Produces `Query.state: str`; `RECENCY_BY_STATE: Dict[str, List[str]]`; `recency_state(summary) -> str` in {"awaiting", "draft", "done", "reminder"}.

- [ ] Test `test_recency_queries_match_last_thread_state`: for seeds 7,8,9 stratified 40, every recency query text is in `RECENCY_BY_STATE[q.state]` and `q.state == recency_state(workload.by_name()[q.target].summary)`; the set of distinct recency texts across seeds is ≥ 6. Also `test_recency_query_coherent_examples`: a summary starting "Sent draft" yields state "draft" and "ok send it" is allowed; "nothing pending" yields "done" and "ok send it" is not allowed.
- [ ] Run, expect FAIL (no `state`). Implement. Run, expect PASS. Commit `feat(evals): keyword-free follow-ups conditioned on the last thread's state`.

### Task 2: `last_touched` marker and prompt rules

**Files:** Modify `server/services/execution/registry.py` (`render`), `server/agents/interaction_agent/system_prompt.md`; Test `tests/test_registry.py`, `tests/test_interaction_e2e.py`.

- [ ] Test `test_render_marks_most_recent_agent_last_touched`: budgeted render has exactly one `last_touched="true"`, on the first line, and the `all` strategy render has none.
- [ ] Test `test_system_prompt_names_last_touched_rule`: `build_system_prompt()` contains `last_touched` and "search_agents" within the same paragraph as "names a person".
- [ ] Run → FAIL. Implement. Run → PASS. Commit `feat(roster): last_touched marker; prompt rules for bare follow-ups and unlisted people`.

### Task 3: Routing mismatch guard

**Files:** Modify `registry.py`, `interaction_agent/tools.py`, `evals/live_routing_eval.py`; Test `tests/test_registry.py`, `tests/test_interaction_e2e.py`, `tests/test_evals.py`.
**Interfaces:** `RoutingMismatch(tokens: List[str], candidates: List[str])`; `AgentRegistry.routing_mismatch(agent_name: str, message: str) -> Optional[RoutingMismatch]`; `tools.send_message_to_agent` returns `ToolResult(success=False, payload={"error": ..., "did_you_mean": [...]})` once per (agent, message) when `OPENPOKE_ROUTING_GUARD != "0"`; `Decision` gains `guard_fired: bool`, `first_pick: Optional[str]`.

- [ ] Registry table test `test_routing_mismatch`: rows — ("what did bob say about the flight confirmation?", "Follow up with Tomasz - flight") → candidates include "Flight Confirmation with Bob"; ("ask Tomasz whether Bob is coming", "Follow up with Tomasz - flight") → None; ("any update on the flight?", "Follow up with Tomasz - flight") → None (no person token); ("did bob reply?", "Flight Confirmation with Bob") → None; unknown agent → None.
- [ ] Tools test `test_send_message_guard_fires_once`: first call returns error with `did_you_mean`, second identical call dispatches; with `OPENPOKE_ROUTING_GUARD=0` the first call dispatches.
- [ ] Eval test `test_decide_applies_guard` with a scripted `complete`: first pick wrong-person, guard error fed back, second pick correct → `ok`, `guard_fired`, `first_pick` set.
- [ ] Run → FAIL. Implement. Run → PASS. Commit `feat(routing): did-you-mean guard when the chosen agent is about a different person`.

### Task 4: Interaction early exit

**Files:** Modify `interaction_agent/runtime.py`; Test `tests/test_interaction_e2e.py`.

- [ ] Tests: `test_early_exit_after_announce_and_delegate` (iteration 1 = user msg + agent msg → exactly 1 LLM call; response is the user message); `test_no_early_exit_after_announce_only` (iteration 1 = user msg only; iteration 2 = agent msg; iteration 3 empty → 3 calls... minus exit: expect 2 calls since iteration 2 qualifies); `test_agent_turn_exits_after_reply` (handle_agent_message, iteration 1 = user msg → 1 call); `test_early_exit_flag_off` restores 2 calls for the first case.
- [ ] Run → FAIL. Implement (pass `message_type` into the loop). Run → PASS. Commit `perf(interaction): end the turn once the user is told and the agent dispatched`.

### Task 5: Session probe

**Files:** Create `evals/session_probe.py`; Test `tests/test_evals.py` (`test_session_probe_report_shape` using a fake ledger).
**Interfaces:** `report(ledger_totals_by_role, early_exit: bool) -> Dict`; `main()` with `--live`, boots app via `httpx.AsyncClient(transport=ASGITransport(app))`, posts the three turns to the chat endpoint, reads `/api/v1/meta/usage`, repeats with the flag off, writes `evals/out/session_probe.json`.

- [ ] Test → FAIL. Implement. PASS. Commit `feat(evals): in-process session probe for calls, cache share and cost by role`.

### Task 6: Held-out email set

**Files:** Create `tests/fixtures/emails_heldout.json` (14 rows h01–h14); Modify `evals/email_eval.py` (`load_fixtures(name="emails")`, `--set dev|heldout|both`), `tests/test_email_prefilter.py` (held-out important rows reach the model).

- [ ] Test `test_heldout_important_reach_model` → FAIL (file missing). Write the fixture. PASS. Commit `test(email): held-out labelled set, written before the classifier prompt changes`.

### Task 7: Classifier prompt and summary fallback

**Files:** Modify `importance_classifier.py`; Test `tests/test_payloads_and_misc.py` or new `tests/test_classifier.py`.

- [ ] Tests: `test_prompt_lists_categories` (prompt mentions "balance", "recruiter", "immigration", "replying", "unsubscribe"); `test_summary_only_fallback_when_llm_gives_no_summary` (FakeJev p=0.9, scripted LLM returns important=false no summary → result is a non-empty string containing the subject).
- [ ] FAIL → implement → PASS. Commit `fix(email): classifier prompt names the categories it missed; Jev-decided mail is never dropped for want of a summary`.

### Task 8: Two-sided Jev gate and sweep

**Files:** Modify `importance_classifier.py`, `evals/jev_bench.py`, `.env.example`; Test `tests/test_jev.py`, `tests/test_evals.py`.
**Interfaces:** `gate_thresholds() -> Tuple[float, float]`; `gate_sweep(rows: List[Dict]) -> List[Dict]` where each row has `p`, `llm_important`, `truth`; result rows have `low, high, recall, precision, llm_calls`.

- [ ] Tests: `test_gate_low_equals_high_matches_round_two` (p=0.4 dismissed, p=0.6 summary-only); `test_gate_middle_band_runs_full_classifier` (low 0.2, high 0.8, p=0.5 → full prompt used and LLM verdict respected); `test_gate_sweep_counts` on a 6-row synthetic table.
- [ ] FAIL → implement → PASS. Commit `feat(email): two-sided Jev gate with a recorded-probability sweep`.

### Task 9: Corrections in the extractor contract

**Files:** Modify `memory.py`; Test `tests/test_memory.py`.

- [ ] Tests: `test_heuristic_extracts_corrections` ("the invoice was wrong: it is $700, not $500" → corrections [("$500","$700")]); `test_merge_correction_invalidates_matching_fact` (active fact "Priya invoice amount: $500" invalidated by correction with new key "Priya invoice"; unrelated "Bob deposit: $500" untouched); `test_llm_extractor_parses_replaces_and_corrections`; `test_merge_replaces_invalidates_named_key`.
- [ ] FAIL → implement → PASS. Run `python -m evals.memory_eval --no-plot` and note stale. Commit `fix(memory): corrections and key replacement in the extractor contract`.

### Task 10: Trigger coalescing

**Files:** Modify `server/services/trigger_scheduler.py`; Test new `tests/test_trigger_scheduler.py`.

- [ ] Tests: `test_due_triggers_for_same_agent_coalesce` (two due → one `execute_agent` call whose instructions contain both payloads and ids; both completed); `test_coalesced_failure_marks_every_trigger`; `test_coalesce_flag_off_dispatches_separately`.
- [ ] FAIL → implement → PASS. Commit `perf(triggers): one execution per agent per poll`.

### Task 11: `search_history` usage eval

**Files:** Create `evals/recall_eval.py`; Test `tests/test_evals.py` (`test_recall_eval_scoring`).
**Interfaces:** `questions_for(needles) -> List[Dict]` (4 buried, 4 surface); `score(rows) -> Dict` with `buried_call_rate, buried_accuracy, surface_call_rate, surface_accuracy`.

- [ ] FAIL → implement (live path reuses `memory_eval.build_log`, compacts with `LLMExtractor`, calls `complete` with `ROLE_EXECUTION`, executes only `search_history`) → PASS. Commit `feat(evals): measure whether the execution model reaches for search_history`.

### Task 12: Learned reranker experiment

**Files:** Create `evals/learned_selector.py`; Test `tests/test_evals.py` (`test_learned_selector_features_and_fit` on a tiny workload: fit converges, recall computed).
**Interfaces:** `features(query, agent, ctx) -> List[float]`; `fit(X, y, epochs=300, lr=0.1) -> weights`; `evaluate(train_seeds, test_seeds, n_agents=150) -> Dict`.

- [ ] FAIL → implement → PASS. Run it, record the result. Commit `experiment(evals): learned roster reranker vs hand-tuned select()`.

### Task 13: Wire live runner, run everything, document

**Files:** Modify `evals/run_live.sh`, `evals/README.md`, `README.md`, `.env.example`; create `docs/live/round3/`.

- [ ] Offline: `python -m evals.run_all`, `roster_eval --paraphrase`, `learned_selector`; copy outputs to `docs/live/round3/`.
- [ ] Live: `bash evals/run_live.sh` (routing both arms, memory, email dev+heldout, Jev with sweep), then `recall_eval --live`, `session_probe --live`. Archive.
- [ ] README: results tables cite round-three files; note the workload change for keyword-free; new rows for guard counters, calls per turn, `search_history` usage, gate sweep, held-out email, reranker result. Limitations updated.
- [ ] Commit `docs: round-three results with provenance`.
