"""Experiment: a learned roster reranker vs the hand-tuned ``select()``.

Question a reviewer will ask: "why hand-tune the roster selection instead of learning it?"
This answers it with a number. A logistic reranker is trained on synthetic workloads (seeds
1-5) to score (query, agent) pairs, and its recall@12 on held-out seeds (7-9) is compared with
``AgentRegistry.select()`` on the same workloads. Features are the signals ``select()`` already
uses, so the experiment asks whether *learned weights* beat the hand-chosen ones, not whether
new signals exist. numpy only; nothing here runs at runtime.

The caveat is printed with the result: the labels are synthetic. A real deployment would train
on its own delegation log.

Usage:
    python -m evals.learned_selector [--agents 150] [--train 1,2,3,4,5] [--test 7,8,9]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from server.agents.interaction_agent.agent import agent_names_in_transcript
from server.services.execution.registry import AgentRegistry, RegistryConfig, tokenize

from .roster_eval import build_registry
from .synthetic import DAY, Workload, generate, transcript_for

FEATURE_NAMES = ["bm25", "name_overlap", "recency_rank", "age", "use_count", "pinned", "in_transcript", "waiting"]


def _context(registry: AgentRegistry, workload: Workload) -> Dict:
    records = [registry.get(a.name) for a in workload.agents]
    records = [r for r in records if r is not None]
    corpus = [tokenize(r.retrieval_text()) or ["_"] for r in records]
    order = sorted(range(len(records)), key=lambda i: -records[i].last_used_at)
    rank = {records[i].name: pos for pos, i in enumerate(order)}
    pinned = {a.name for a in workload.agents if a.has_trigger}
    transcript = transcript_for(workload)
    recent = set(agent_names_in_transcript(transcript, registry.names()))
    return {"records": records, "bm25": BM25Okapi(corpus), "rank": rank, "pinned": pinned,
            "recent": recent, "now": workload.now, "n": len(records)}


def features(query: str, index: int, ctx: Dict, bm25_scores: np.ndarray) -> List[float]:
    record = ctx["records"][index]
    q = set(tokenize(query))
    name_tokens = set(tokenize(record.name))
    max_bm = float(bm25_scores.max()) if bm25_scores.size else 0.0
    return [
        float(bm25_scores[index]) / max_bm if max_bm > 0 else 0.0,
        len(q & name_tokens) / len(q) if q else 0.0,
        ctx["rank"][record.name] / max(1, ctx["n"]),
        min(1.0, max(0.0, ctx["now"] - record.last_used_at) / (21 * DAY)),
        min(1.0, record.use_count / 6.0),
        1.0 if record.name in ctx["pinned"] else 0.0,
        1.0 if record.name in ctx["recent"] else 0.0,
        1.0 if ("waiting" in (record.summary or "").lower() or "awaiting" in (record.summary or "").lower()) else 0.0,
    ]


def _pairs(workload: Workload, registry: AgentRegistry) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """One row per (query, agent). Returns X, y and (query index, agent index) for grouping."""
    ctx = _context(registry, workload)
    X: List[List[float]] = []
    y: List[int] = []
    groups: List[Tuple[int, int]] = []
    for qi, query in enumerate(workload.queries):
        scores = ctx["bm25"].get_scores(tokenize(query.text) or ["_"])
        for ai, record in enumerate(ctx["records"]):
            X.append(features(query.text, ai, ctx, scores))
            y.append(1 if record.name == query.target else 0)
            groups.append((qi, ai))
    return np.array(X), np.array(y), groups


def fit(X: np.ndarray, y: np.ndarray, *, epochs: int = 300, lr: float = 0.1) -> Tuple[np.ndarray, float]:
    """Logistic regression by full-batch gradient descent, positives up-weighted to balance."""
    w = np.zeros(X.shape[1])
    b = 0.0
    pos_weight = (len(y) - y.sum()) / max(1, y.sum())
    sample_w = np.where(y == 1, pos_weight, 1.0)
    for _ in range(epochs):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        grad = (p - y) * sample_w
        w -= lr * (X.T @ grad) / len(y)
        b -= lr * grad.mean()
    return w, b


def _recall_at_k(workload: Workload, registry: AgentRegistry, w: np.ndarray, b: float, k: int) -> float:
    X, _y, groups = _pairs(workload, registry)
    scores = X @ w + b
    by_query: Dict[int, List[Tuple[float, int]]] = {}
    for (qi, ai), s in zip(groups, scores):
        by_query.setdefault(qi, []).append((float(s), ai))
    ctx_records = [registry.get(a.name) for a in workload.agents]
    ctx_records = [r for r in ctx_records if r is not None]
    hits = 0
    for qi, query in enumerate(workload.queries):
        top = sorted(by_query[qi], key=lambda t: -t[0])[:k]
        if any(ctx_records[ai].name == query.target for _s, ai in top):
            hits += 1
    return hits / len(workload.queries)


def _select_recall(workload: Workload, registry: AgentRegistry) -> float:
    transcript = transcript_for(workload)
    recent = agent_names_in_transcript(transcript, registry.names())
    hits = 0
    for query in workload.queries:
        shown = registry.select(query.text, now=workload.now, recent_agent_names=recent).names()
        hits += query.target in shown
    return hits / len(workload.queries)


def evaluate(*, train_seeds: Sequence[int], test_seeds: Sequence[int], n_agents: int = 150,
             n_queries: int = 300, tmp_dir: Path = Path("evals/out/tmp"), k: int = 12) -> Dict:
    Xs, ys = [], []
    for seed in train_seeds:
        w_ = generate(n_agents, seed=seed, n_queries=n_queries)
        reg = build_registry(w_, tmp_dir / f"learn_{seed}", RegistryConfig())
        X, y, _ = _pairs(w_, reg)
        Xs.append(X)
        ys.append(y)
    X, y = np.vstack(Xs), np.concatenate(ys)
    w, b = fit(X, y)
    learned, hand, n_test = [], [], 0
    for seed in test_seeds:
        w_ = generate(n_agents, seed=seed, n_queries=n_queries, stratified=True)
        reg = build_registry(w_, tmp_dir / f"learn_{seed}", RegistryConfig())
        learned.append(_recall_at_k(w_, reg, w, b, k))
        hand.append(_select_recall(w_, reg))
        n_test += len(w_.queries)
    return {
        "feature_names": FEATURE_NAMES,
        "weights": [round(float(x), 3) for x in w],
        "bias": round(float(b), 3),
        "k": k,
        "n_train_pairs": int(len(y)),
        "n_test": n_test,
        "learned_recall": round(sum(learned) / len(learned), 3),
        "select_recall": round(sum(hand) / len(hand), 3),
        "note": "trained and tested on synthetic workloads; a real deployment would train on its own delegation log",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agents", type=int, default=150)
    parser.add_argument("--train", default="1,2,3,4,5")
    parser.add_argument("--test", default="7,8,9")
    parser.add_argument("--out", default="evals/out")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result = evaluate(train_seeds=[int(s) for s in args.train.split(",")], test_seeds=[int(s) for s in args.test.split(",")],
                      n_agents=args.agents, tmp_dir=out / "tmp")
    (out / "learned_selector.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
