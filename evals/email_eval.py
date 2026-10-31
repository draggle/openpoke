"""Email importance eval against the labelled fixture set.

Offline (default): measures the rule-based pre-filter alone.
    - skip precision   P(not important | prefilter said skip)   -> must be 1.0 (never drop a real one)
    - skip rate        fraction of emails dismissed without a model call  -> cost saving
Live (``--live`` with OPENROUTER_API_KEY): runs prefilter + LLM classifier and reports
precision / recall / F1 of the full pipeline against the labels.

Usage:
    python -m evals.email_eval [--live]
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from server.services.gmail.prefilter import prefilter
from server.services.gmail.processing import ProcessedEmail

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "emails.json"


def load_fixtures() -> List[Dict]:
    return json.loads(FIXTURES.read_text())["emails"]


def to_processed(row: Dict) -> ProcessedEmail:
    return ProcessedEmail(
        id=row["id"], thread_id=row["id"], query="", subject=row["subject"], sender=row["sender"],
        recipient="me@example.com", timestamp=datetime(2026, 10, 30, 9, 0, tzinfo=timezone.utc),
        label_ids=row.get("labels", []), clean_text=row["body"], has_attachments=False,
        attachment_count=0, attachment_filenames=[],
    )


def eval_prefilter(rows: List[Dict]) -> Dict[str, float]:
    skipped = [r for r in rows if prefilter(to_processed(r)).skip]
    wrongly_skipped = [r["id"] for r in skipped if r["important"]]
    noise = [r for r in rows if not r["important"]]
    noise_skipped = [r for r in skipped if not r["important"]]
    return {
        "emails": len(rows),
        "skipped": len(skipped),
        "skip_rate": len(skipped) / len(rows),
        "skip_precision": 1.0 - (len(wrongly_skipped) / len(skipped) if skipped else 0.0),
        "noise_caught_by_rules": len(noise_skipped) / len(noise),
        "wrongly_skipped_ids": wrongly_skipped,
    }


async def eval_live(rows: List[Dict]) -> Dict[str, float]:
    from server.services.gmail.importance_classifier import classify_email_importance

    tp = fp = fn = tn = 0
    model_calls = 0
    for row in rows:
        email = to_processed(row)
        if prefilter(email).skip:
            predicted = False
        else:
            model_calls += 1
            predicted = (await classify_email_importance(email)) is not None
        truth = row["important"]
        tp += predicted and truth
        fp += predicted and not truth
        fn += (not predicted) and truth
        tn += (not predicted) and not truth
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"model_calls": model_calls, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--out", default="evals/out")
    args = parser.parse_args()
    rows = load_fixtures()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pre = eval_prefilter(rows)
    print("prefilter (rules only):")
    for k, v in pre.items():
        print(f"  {k:<24} {v:.2f}" if isinstance(v, float) else f"  {k:<24} {v}")
    results = {"prefilter": pre}
    if args.live:
        live = asyncio.run(eval_live(rows))
        print("\nprefilter + LLM classifier:")
        for k, v in live.items():
            print(f"  {k:<24} {v:.2f}" if isinstance(v, float) else f"  {k:<24} {v}")
        results["live"] = live
    (out / "email_results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
