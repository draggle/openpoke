#!/usr/bin/env bash
# Run every live (API-backed) evaluator and write results to evals/out/.
#
# Usage:  bash evals/run_live.sh [path/to/.env]
#
# The .env needs ONE of:
#   AI_GATEWAY_API_KEY=...     (Vercel AI Gateway: serves both Claude and Jev)
# or
#   OPENROUTER_API_KEY=...     (+ optionally TYPESAFE_API_KEY=... for Jev)
set -euo pipefail
cd "$(dirname "$0")/.."

ENV_FILE="${1:-.env}"
if [ -f "$ENV_FILE" ]; then
  set -a; . "$ENV_FILE"; set +a
fi

# --- map a Vercel AI Gateway key onto both providers ---------------------------------
if [ -n "${AI_GATEWAY_API_KEY:-}" ]; then
  export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-$AI_GATEWAY_API_KEY}"
  export OPENROUTER_BASE_URL="${OPENROUTER_BASE_URL:-https://ai-gateway.vercel.sh/v1}"
  export TYPESAFE_API_KEY="${TYPESAFE_API_KEY:-$AI_GATEWAY_API_KEY}"
  export TYPESAFE_BASE_URL="${TYPESAFE_BASE_URL:-https://ai-gateway.vercel.sh/typesafe}"
  export TYPESAFE_MODEL="${TYPESAFE_MODEL:-typesafe-ai/jev}"
fi
: "${OPENROUTER_API_KEY:?need AI_GATEWAY_API_KEY or OPENROUTER_API_KEY in $ENV_FILE}"

# Model names on the OpenAI-compatible endpoint. Override if your gateway spells them differently.
export OPENPOKE_STRONG_MODEL="${OPENPOKE_STRONG_MODEL:-anthropic/claude-sonnet-4}"
export OPENPOKE_CHEAP_MODEL="${OPENPOKE_CHEAP_MODEL:-anthropic/claude-3.5-haiku}"
export OPENPOKE_JEV=1

QUERIES="${QUERIES:-40}"
mkdir -p evals/out

if [ ! -d .venv ]; then python3 -m venv .venv; fi
# shellcheck disable=SC1091
. .venv/bin/activate
pip install -q -r server/requirements.txt pytest pytest-asyncio matplotlib

echo "== preflight: can we reach the chat endpoint and Jev? =="
python - <<'PY'
import asyncio, os, sys
from server.openrouter_client.client import request_chat_completion
from server.services.llm.jev import JevClient, noul

async def main():
    r = await request_chat_completion(model=os.environ["OPENPOKE_CHEAP_MODEL"],
                                      messages=[{"role": "user", "content": "Reply with the single word: ok"}])
    print("chat ok:", (r.get("choices") or [{}])[0].get("message", {}).get("content", "")[:40].strip())
    try:
        a = await JevClient().ask("The invoice is 10 days overdue.", {"urgent": noul("Is this urgent?")}, role="preflight")
        print("jev ok: P(urgent)=%.2f" % a["urgent"].probability)
    except Exception as exc:
        print("jev FAILED (continuing without it):", exc)
        os.environ["OPENPOKE_JEV"] = "0"
        open("evals/out/.jev_unavailable", "w").close()
asyncio.run(main())
PY
if [ -f evals/out/.jev_unavailable ]; then export OPENPOKE_JEV=0; rm -f evals/out/.jev_unavailable; fi

run() { echo; echo "================ $* ================"; "$@" 2>&1 | grep -v " - INFO - " | tee -a evals/out/live_log.txt; }

: > evals/out/live_log.txt
run python -m evals.live_routing_eval --agents 150 --queries "$QUERIES"
run python -m evals.memory_eval --live --rounds 10,20,40
run python -m evals.email_eval --live
if [ "${OPENPOKE_JEV}" = "1" ]; then
  run python -m evals.jev_bench --live --agents 150 --queries "$QUERIES"
fi

echo
echo "done. results: evals/out/*.json and evals/out/live_log.txt"
