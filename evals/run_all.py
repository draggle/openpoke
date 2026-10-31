"""Run every offline evaluator and write results + charts to evals/out/.

    python -m evals.run_all
"""

from __future__ import annotations

import subprocess
import sys

STEPS = [
    [sys.executable, "-m", "evals.roster_eval"],
    [sys.executable, "-m", "evals.memory_eval"],
    [sys.executable, "-m", "evals.email_eval"],
    [sys.executable, "-m", "evals.cost_eval"],
]


def main() -> None:
    for step in STEPS:
        print("\n" + "=" * 80 + f"\n$ {' '.join(step[1:])}\n" + "=" * 80)
        subprocess.run(step, check=True)


if __name__ == "__main__":
    main()
