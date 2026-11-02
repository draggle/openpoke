"""Does the execution model actually reach for ``search_history``?

Round two showed that facts buried in a tool payload are 100% *recoverable* through the tool
and 0% present in the compacted prompt. That measures the mechanism. This eval measures the
behaviour: after the memory eval's 40-round agent is compacted, the real execution model is
asked eight questions with the execution system prompt and the ``search_history`` tool:

- ``buried``  (4): the internal reference code that only the raw journal holds. The right
  behaviour is to call the tool and answer from it.
- ``surface`` (4): an amount that is in the compacted prompt. The right behaviour is to answer
  without a tool call (a call here is wasted cost, not an error).

Reported per class: tool-call rate and answer accuracy. Live only (needs a key).

Usage:
    python -m evals.recall_eval --live [--rounds 40]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Sequence

from server.agents.execution_agent.agent import SYSTEM_PROMPT_TEMPLATE
from server.agents.execution_agent.tools import history as history_tool
from server.services.execution.log_store import ExecutionAgentLogStore
from server.services.execution.memory import AgentMemory, LLMExtractor, MemoryConfig

from .memory_eval import Needle, build_log
from .synthetic import PEOPLE

_REQUEST_RE = re.compile(r"Ask (\w+) whether the (\w+) for (\$\d+) can be settled by (\w+ \d+)")
_ITEMS = ["invoice", "deposit", "refund"]


def questions_for(needles: Sequence[Needle], *, person: str, n_each: int = 4,
                  requests: Sequence[str] = ()) -> List[Dict]:
    """Eight questions with known answers: ``n_each`` buried (reference codes) and surface (amounts)."""
    dates = {}  # amount -> date, from the original requests when available
    for text in requests:
        m = _REQUEST_RE.search(text)
        if m:
            dates[m.group(3)] = m.group(4)
    amount_by_round = {n.round: n.text for n in needles if n.kind == "surface"}
    buried = [n for n in needles if n.kind == "buried" and n.round in amount_by_round]
    surface = [n for n in needles if n.kind == "surface"]

    def spread(items: List[Needle]) -> List[Needle]:
        if len(items) <= n_each:
            return list(items)
        step = len(items) / n_each
        return [items[int(i * step)] for i in range(n_each)]

    out: List[Dict] = []
    for n in spread(buried):
        amount, item = amount_by_round[n.round], _ITEMS[n.round % 3]
        out.append({"kind": "buried", "answer": n.text,
                    "question": f"What internal reference number did {person} give in their email about the {item} for {amount}? Reply with the reference only."})
    for n in spread(surface):
        item = _ITEMS[n.round % 3]
        when = f" to be settled by {dates[n.text]}" if n.text in dates else ""
        out.append({"kind": "surface", "answer": n.text,
                    "question": f"What was the amount of the {item} {person} was asked about{when}? Reply with the amount only."})
    return out


def score(rows: Sequence[Dict]) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for kind in ("buried", "surface"):
        sub = [r for r in rows if r["kind"] == kind]
        if not sub:
            continue
        out[kind] = {"n": len(sub),
                     "call_rate": round(sum(r["called_search"] for r in sub) / len(sub), 3),
                     "accuracy": round(sum(r["correct"] for r in sub) / len(sub), 3)}
        if any("answer_returned_by_search" in r for r in sub):
            out[kind]["search_returned_answer"] = round(sum(bool(r.get("answer_returned_by_search")) for r in sub) / len(sub), 3)
    return out


async def _ask(agent: str, store: ExecutionAgentLogStore, system_blocks: List[Dict], question: str) -> Dict:
    from server.services.llm import ROLE_EXECUTION, complete

    messages = [{"role": "user", "content": question}]
    called = False
    text = ""
    queries: List[str] = []
    returned: List[str] = []
    for _ in range(4):  # a search, a refined search, an answer, one spare
        response = await complete(role=ROLE_EXECUTION, messages=messages, system=system_blocks,
                                  tools=history_tool.get_schemas(), tag="recall_eval")
        message = (response.get("choices") or [{}])[0].get("message") or {}
        text = message.get("content") or text
        calls = message.get("tool_calls") or []
        messages.append({"role": "assistant", "content": message.get("content") or "",
                         **({"tool_calls": calls} if calls else {})})
        if not calls:
            break
        for call in calls:
            fn = call.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            if fn.get("name") == "search_history":
                called = True
                queries.append(str(args.get("query", "")))
                result = history_tool.search_history(agent, args.get("query", ""), int(args.get("limit", 5) or 5), log_store=store)
                returned.extend(m.get("text", "") for m in result.get("matches", []))
            else:
                result = {"error": f"tool {fn.get('name')} is not available in this eval"}
            messages.append({"role": "tool", "tool_call_id": call.get("id") or "t",
                             "content": json.dumps({"tool": fn.get("name"), "status": "success", "result": result})})
    return {"called_search": called, "text": (text or "")[:300], "queries": queries, "returned": returned}


async def evaluate(rounds: int, out_dir: Path, seed: int = 11, budget_chars: int = 12_000) -> Dict:
    rng = random.Random(seed + rounds)
    work = out_dir / "tmp" / f"recall_{rounds}"
    if work.exists():
        for f in work.rglob("*"):
            if f.is_file():
                f.unlink()
    store = ExecutionAgentLogStore(work / "logs")
    agent = f"Email to {rng.choice(PEOPLE)}"
    person = agent.split()[-1]
    needles = build_log(store, agent, rounds, rng)
    memory = AgentMemory(work / "mem", log_store=store, extractor=LLMExtractor(),
                         config=MemoryConfig(budget_chars=budget_chars, tail_entries=12))
    for tag, _ts, payload in store.iter_entries(agent):
        if tag == "agent_request":
            memory.note_request(agent, payload)
    while memory.needs_compaction(agent):
        if await memory.compact(agent) is None:
            break
    history = memory.render_history(agent)
    static = SYSTEM_PROMPT_TEMPLATE.format(agent_name=agent, agent_purpose=f"Handle tasks related to: {agent}")
    blocks = [{"type": "text", "text": static, "cacheable": True},
              {"type": "text", "text": f"# Execution History\n\n{history}"}]
    requests = [p for t, _ts, p in store.iter_entries(agent) if t == "agent_request"]
    questions = questions_for(needles, person=person, requests=requests)
    rows: List[Dict] = []
    for q in questions:
        got = await _ask(agent, store, blocks, q["question"])
        correct = q["answer"].lower() in got["text"].lower()
        answer_returned = any(q["answer"] in snippet for snippet in got.pop("returned"))
        rows.append({**q, **got, "correct": correct, "in_prompt": q["answer"] in history,
                     "answer_returned_by_search": answer_returned})
        print(f"[{q['kind']:<7}] search={'Y' if got['called_search'] else 'n'} returned={'Y' if answer_returned else 'n'} "
              f"{'OK ' if correct else 'MISS'} want {q['answer']:<10} got {got['text'][:50]!r} q={got['queries'][:2]}", flush=True)
    result = {"rounds": rounds, "entries": sum(1 for _ in store.iter_entries(agent)),
              "history_chars": len(history), "scores": score(rows), "rows": rows}
    (out_dir / "recall_results_live.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--out", default="evals/out")
    args = parser.parse_args()
    if not args.live:
        raise SystemExit("recall_eval only has a live mode (it asks the real execution model); pass --live")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = asyncio.run(evaluate(args.rounds, out_dir))
    print(json.dumps(result["scores"], indent=1))


if __name__ == "__main__":
    main()
