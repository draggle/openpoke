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
set -euo pipefail
cd "$(dirname "$0")/.."

ENV_FILE="${1:-.env}"
if [ -f "$ENV_FILE" ]; then
  set -a; . "$ENV_FILE"; set +a
fi

# --- Anthropic directly (OpenAI-compatible endpoint) ---------------------------------
if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
  export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-$ANTHROPIC_API_KEY}"
  export OPENROUTER_BASE_URL="${OPENROUTER_BASE_URL:-https://api.anthropic.com/v1}"
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
mkdir -p evals/out

if [ ! -d .venv ]; then python3 -m venv .venv; fi
# shellcheck disable=SC1091
. .venv/bin/activate
pip install -q -r server/requirements.txt pytest pytest-asyncio matplotlib

echo "== discovering model IDs on ${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1} =="
python - <<'PY'
import json, os, re, sys, urllib.request
base = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
key = os.environ["OPENROUTER_API_KEY"]
req = urllib.request.Request(f"{base}/models", headers={"Authorization": f"Bearer {key}", "x-api-key": key, "anthropic-version": "2023-06-01"})
try:
    data = json.load(urllib.request.urlopen(req, timeout=20))
    ids = [m.get("id", "") for m in (data.get("data") or data.get("models") or [])]
except Exception as exc:
    print("could not list models:", exc); ids = []
def pick(patterns, fallback):
    for pat in patterns:
        for i in ids:
            if re.search(pat, i): return i
    return fallback
strong = os.environ.get("OPENPOKE_STRONG_MODEL") or pick([r"claude-sonnet-4-5", r"claude-sonnet-4(?!.*haiku)", r"claude-3-7-sonnet", r"claude.*sonnet"], "claude-sonnet-4-5" if "anthropic.com" in base else "anthropic/claude-sonnet-4")
cheap  = os.environ.get("OPENPOKE_CHEAP_MODEL")  or pick([r"claude-haiku-4-5", r"claude-3-5-haiku", r"claude.*haiku"], strong)
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

run() { echo; echo "================ $* ================"; PYTHONUNBUFFERED=1 "$@" 2>&1 | grep --line-buffered -v " - INFO - " | tee -a evals/out/live_log.txt; }

: > evals/out/live_log.txt
run python -m evals.live_routing_eval --agents 150 --queries "$QUERIES"
run python -m evals.memory_eval --live --rounds 10,20,40
run python -m evals.email_eval --live
if [ "${OPENPOKE_JEV}" = "1" ]; then
  run python -m evals.jev_bench --live --agents 150 --queries "$QUERIES"
fi

echo
echo "done. results: evals/out/*.json and evals/out/live_log.txt"
