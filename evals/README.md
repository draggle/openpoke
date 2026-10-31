# Evaluators

See the "Tests and evaluators" section of the top-level README. Quick start:

    python -m evals.run_all                      # offline: roster, memory, email, cost
    OPENROUTER_API_KEY=... python -m evals.live_routing_eval --queries 40
    OPENROUTER_API_KEY=... python -m evals.memory_eval --live
    OPENROUTER_API_KEY=... python -m evals.email_eval --live

Outputs land in `evals/out/` (gitignored). Charts used in the README are copied to `docs/`.
