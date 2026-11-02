#!/usr/bin/env bash
# Run every live (API-backed) evaluator and write results to evals/out/.
#
# Usage:  bash evals/run_live.sh [path/to/.env]
#
# The .env needs ONE of:
#   ANTHROPIC_API_KEY=...      (Anthropic directly, for the Claude side; add TYPESAFE_API_KEY for Jev)
#   OPENCODE_API_KEY=...       (OpenCode Zen: serves Claude and Jev)
#   AI_GATEWAY_API_KEY=...     (Vercel AI Gateway)
#   OPENROUTER_API_KEY=...     (+ optionally TYPESAFE_API_KEY=... for Jev)
# Optional: OPENPOKE_STRONG_MODEL / OPENPOKE_CHEAP_MODEL / TYPESAFE_MODEL to pin model IDs.
#           ONLY_JEV=1 bash evals/run_live.sh   -> run just the Jev bench (Claude evals already done)
#           ROUTING_STRATEGIES=budgeted ONLY_ROUTING=1 bash evals/run_live.sh -> re-run routing only
#           ROUTING_EXTRA_ARGS="" bash evals/run_live.sh -> round-one style (no history, sampled kinds, one seed)
# Round three adds evals.recall_eval (search_history usage) and evals.session_probe (calls per turn
# through the in-process app, in an isolated temporary data dir) and the held-out email set.
set -euo pipefail
cd "$(dirname "$0")/.."

ENV_FILE="${1:-.env}"
if [ -f "$ENV_FILE" ]; then
  set -a; . "$ENV_FILE"; set +a
fi

# --- Jev via OpenRouter, independent of which provider serves the chat side ----------
#   JEV_OPENROUTER_KEY=sk-or-...   (OpenRouter proxies TypeSafe's native schema as typesafe/jev-latest)
if [ -n "${JEV_OPENROUTER_KEY:-}" ]; then
  export TYPESAFE_API_KEY="${TYPESAFE_API_KEY:-$JEV_OPENROUTER_KEY}"
  export TYPESAFE_MODEL="${TYPESAFE_MODEL:-typesafe/jev-latest}"
  export JEV_BASE_CANDIDATES="${TYPESAFE_BASE_URL:-https://openrouter.ai/api https://openrouter.ai/api/v1 https://openrouter.ai/api/v1/typesafe}"
fi
# --- Anthropic directly (OpenAI-compatible endpoint) ---------------------------------
if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
  export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-$ANTHROPIC_API_KEY}"
  export OPENROUTER_BASE_URL="${OPENROUTER_BASE_URL:-https://api.anthropic.com/v1}"
  # Native Messages API: the OpenAI-compatible shim ignores prompt caching.
  export OPENPOKE_LLM_API="${OPENPOKE_LLM_API:-anthropic}"
fi
# --- map an OpenCode Zen key onto both providers -------------------------------------
if [ -n "${OPENCODE_API_KEY:-}" ]; then
  export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-$OPENCODE_API_KEY}"
  export OPENROUTER_BASE_URL="${OPENROUTER_BASE_URL:-https://opencode.ai/zen/v1}"
  export TYPESAFE_API_KEY="${TYPESAFE_API_KEY:-$OPENCODE_API_KEY}"
  export TYPESAFE_BASE_URL="${TYPESAFE_BASE_URL:-https://opencode.ai/zen}"
  export TYPESAFE_MODEL="${TYPESAFE_MODEL:-jev-latest}"
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

export OPENPOKE_JEV=1

QUERIES="${QUERIES:-40}"
# Round two: give both arms the conversation history, draw equal query kinds, pool 3 seeds.
ROUTING_EXTRA_ARGS="${ROUTING_EXTRA_ARGS:---history --stratified --seeds 7,8,9}"
mkdir -p evals/out

if [ ! -d .venv ]; then python3 -m venv .venv; fi
# shellcheck disable=SC1091
. .venv/bin/activate
pip install -q -r server/requirements.txt pytest pytest-asyncio matplotlib

echo "== discovering model IDs on ${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1} =="
python - <<'PY'
import json, os, re, sys
import httpx  # uses certifi, unlike urllib on a python.org build without system certs
base = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
key = os.environ["OPENROUTER_API_KEY"]
try:
    data = httpx.get(f"{base}/models", headers={"Authorization": f"Bearer {key}", "x-api-key": key, "anthropic-version": "2023-06-01"}, timeout=20).json()
    ids = [m.get("id", "") for m in (data.get("data") or data.get("models") or [])]
except Exception as exc:
    print("could not list models:", exc); ids = []
def pick(patterns, fallback):
    for pat in patterns:
        for i in ids:
            if re.search(pat, i): return i
    return fallback
strong = os.environ.get("OPENPOKE_STRONG_MODEL") or pick([r"claude-sonnet-4-5", r"claude-sonnet-4(?!.*haiku)", r"claude-3-7-sonnet", r"claude.*sonnet"], "claude-sonnet-4-5" if "anthropic.com" in base else "anthropic/claude-sonnet-4")
cheap  = os.environ.get("OPENPOKE_CHEAP_MODEL")  or pick([r"claude-haiku-4-5", r"claude-3-5-haiku", r"claude.*haiku"], "claude-haiku-4-5" if "anthropic.com" in base else strong)
jev    = os.environ.get("TYPESAFE_MODEL") or pick([r"^jev-latest$", r"jev", r"typesafe"], "jev-latest")
print(f"strong={strong}\ncheap={cheap}\njev={jev}")
print("claude/jev models seen:", [i for i in ids if re.search(r"claude|jev|typesafe", i)][:20])
open("evals/out/.models", "w").write(f"{strong}\n{cheap}\n{jev}\n")
PY
{ read -r STRONG; read -r CHEAP; read -r JEVM; } < evals/out/.models
export OPENPOKE_STRONG_MODEL="$STRONG" OPENPOKE_CHEAP_MODEL="$CHEAP" TYPESAFE_MODEL="$JEVM"

echo "== preflight: can we reach the chat endpoint and Jev? =="
python - <<'PY'
import asyncio, os, sys
from server.openrouter_client.client import request_chat_completion
from server.services.llm.jev import JevClient, noul

async def main():
    r = await request_chat_completion(model=os.environ["OPENPOKE_CHEAP_MODEL"],
                                      messages=[{"role": "user", "content": "Reply with the single word: ok"}])
    print("chat ok:", (r.get("choices") or [{}])[0].get("message", {}).get("content", "")[:40].strip())
    if not os.environ.get("TYPESAFE_API_KEY"):
        print("jev: no TYPESAFE_API_KEY, skipping Jev evals"); open("evals/out/.jev_unavailable", "w").close(); return
    bases = os.environ.get("JEV_BASE_CANDIDATES", os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai")).split()
    for base in bases:
        try:
            a = await JevClient(base_url=base, model=os.environ.get("TYPESAFE_MODEL", "jev-latest")).ask(
                "The invoice is 10 days overdue.", {"urgent": noul("Is this urgent?")}, role="preflight")
            print(f"jev ok via {base}: P(urgent)={a['urgent'].probability:.2f}")
            open("evals/out/.jev_base", "w").write(base)
            return
        except Exception as exc:
            print(f"jev via {base} failed: {str(exc)[:160]}")
    print("jev FAILED on every base URL (continuing without it)")
    open("evals/out/.jev_unavailable", "w").close()
asyncio.run(main())
PY
if [ -f evals/out/.jev_unavailable ]; then export OPENPOKE_JEV=0; rm -f evals/out/.jev_unavailable; fi
if [ -f evals/out/.jev_base ]; then export TYPESAFE_BASE_URL="$(cat evals/out/.jev_base)"; rm -f evals/out/.jev_base; fi

run() { echo; echo "================ $* ================"; PYTHONUNBUFFERED=1 "$@" 2>&1 | grep --line-buffered -v " - INFO - " | tee -a evals/out/live_log.txt; }

: > evals/out/live_log.txt
if [ "${ONLY_ROUTING:-0}" = "1" ]; then
  # shellcheck disable=SC2086
  run python -m evals.live_routing_eval --agents 150 --queries "$QUERIES" --strategies "${ROUTING_STRATEGIES:-budgeted}" $ROUTING_EXTRA_ARGS
  echo; echo "done (routing only). results: evals/out/live_routing_results*.json"; exit 0
fi
if [ "${ONLY_JEV:-0}" != "1" ]; then
  # shellcheck disable=SC2086
  run python -m evals.live_routing_eval --agents 150 --queries "$QUERIES" --strategies "${ROUTING_STRATEGIES:-all,budgeted}" $ROUTING_EXTRA_ARGS
  run python -m evals.memory_eval --live --rounds 10,20,40
  run python -m evals.email_eval --live
fi
if [ "${OPENPOKE_JEV}" = "1" ]; then
  OPENPOKE_JEV_ROUTING=1 run python -m evals.jev_bench --live --agents 150 --queries "$QUERIES"
fi
if [ "${ONLY_JEV:-0}" != "1" ]; then
  # Round three: does the execution model use search_history; calls/cost per turn through the app.
  run python -m evals.recall_eval --live --rounds 40
  run python -m evals.session_probe --live
  run python -m evals.followup_probe --live
  run python -m evals.hand_session --live --script evals/hand_scripts/reminders.json --scheduler
  run python -m evals.hand_session --live --script evals/hand_scripts/edges.json
  run python -m evals.hand_session --live --script evals/hand_scripts/gmail.json --scheduler --fake-gmail
  run python -m evals.trigger_probe --live
  run python -m evals.injection_eval --live
fi

echo
echo "done. results: evals/out/*.json and evals/out/live_log.txt"
