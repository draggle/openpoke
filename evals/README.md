# Evaluators

See the "Tests and evaluators" section of the top-level README. Quick start:

    python -m evals.run_all                      # offline: roster, memory, email, cost, jev (oracle)
    python -m evals.roster_eval --paraphrase     # + the hand-written paraphrase set
    python -m evals.memory_eval --live           # LLM extractor (needs a key)
    python -m evals.email_eval --live            # prefilter + classifier P/R with missed ids
    python -m evals.live_routing_eval --history --stratified --seeds 7,8,9   # real Interaction model
    python -m evals.recall_eval --live           # does the execution model call search_history?
    python -m evals.session_probe --live        # calls/cost per turn through the app: old loop, early exit, + direct relay (temporary data dir)
    python -m evals.followup_probe --live       # bare follow-ups that are not ambiguous must act; new reminder creates, not moves; triggers land on the named weekday (shipping config, temporary data dir)
    python -m evals.trigger_probe --live        # a reminder fires under the real scheduler; the interaction agent must deliver it
    python -m evals.hand_session --live --script evals/hand_scripts/gmail.json --scheduler --fake-gmail   # scripted hand use through the real app; transcript for review
    python -m evals.guard_check                  # offline: did-you-mean guard false fires / catches
    python -m evals.lifecycle_eval               # offline: sixty simulated days; roster cost flat, reminders never archived
    python -m evals.stats <results.json>         # offline: McNemar + paired bootstrap for a two-arm routing result
    python -m evals.live_routing_eval --fixture evals/heldout_routing.json --history   # live: the hand-written held-out set
    python -m evals.heldout_routing <results.json>   # grade a held-out run against its pre-written bars
    python -m evals.injection_eval --live        # live: injected emails vs the send gate (attempted / executed sends)
    python -m evals.learned_selector             # offline: learned reranker vs select()
    bash evals/run_live.sh [path/to/.env]        # everything live, one command

`run_live.sh` accepts an Anthropic, OpenRouter, OpenCode Zen or Vercel AI Gateway key, discovers
model IDs, and by default runs routing with `--history --stratified --seeds 7,8,9`
(`ROUTING_EXTRA_ARGS=""` for the round-one style single-seed run).

Outputs land in `evals/out/` (gitignored). The results cited in the README are archived under
`docs/live/round3/`, and the re-run after the hardening pass in `docs/live/hardening/`; round-two results stay in `docs/live/round2/` and round-one in `docs/live/`
for the record. `email_eval` reports the 46-row dev set and the 14-row held-out set separately.
