# Evaluators

See the "Tests and evaluators" section of the top-level README. Quick start:

    python -m evals.run_all                      # offline: roster, memory, email, cost, jev (oracle)
    python -m evals.roster_eval --paraphrase     # + the hand-written paraphrase set
    python -m evals.memory_eval --live           # LLM extractor (needs a key)
    python -m evals.email_eval --live            # prefilter + classifier P/R with missed ids
    python -m evals.live_routing_eval --history --stratified --seeds 7,8,9   # real Interaction model
    bash evals/run_live.sh [path/to/.env]        # everything live, one command

`run_live.sh` accepts an Anthropic, OpenRouter, OpenCode Zen or Vercel AI Gateway key, discovers
model IDs, and by default runs routing with `--history --stratified --seeds 7,8,9`
(`ROUTING_EXTRA_ARGS=""` for the round-one style single-seed run).

Outputs land in `evals/out/` (gitignored). The results cited in the README are archived under
`docs/live/round2/`; round-one results stay in `docs/live/` for the record.
