# Evaluators

See the "Tests and evaluators" section of the top-level README. Quick start:

    python -m evals.run_all                      # offline: roster, memory, email, cost, jev (oracle)
    python -m evals.roster_eval --paraphrase     # + the hand-written paraphrase set
    python -m evals.memory_eval --live           # LLM extractor (needs a key)
    python -m evals.email_eval --live            # prefilter + classifier P/R with missed ids
    python -m evals.live_routing_eval --history --stratified --seeds 7,8,9   # real Interaction model
    python -m evals.recall_eval --live           # does the execution model call search_history?
    python -m evals.session_probe --live --clear-state   # calls/cost per turn through the app (clears chat state)
    python -m evals.learned_selector             # offline: learned reranker vs select()
    bash evals/run_live.sh [path/to/.env]        # everything live, one command

`run_live.sh` accepts an Anthropic, OpenRouter, OpenCode Zen or Vercel AI Gateway key, discovers
model IDs, and by default runs routing with `--history --stratified --seeds 7,8,9`
(`ROUTING_EXTRA_ARGS=""` for the round-one style single-seed run).

Outputs land in `evals/out/` (gitignored). The results cited in the README are archived under
`docs/live/round3/`; round-two results stay in `docs/live/round2/` and round-one in `docs/live/`
for the record. `email_eval` reports the 46-row dev set and the 14-row held-out set separately.
