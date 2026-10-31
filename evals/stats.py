"""Small statistics helpers for the evaluators (no scipy)."""

from __future__ import annotations

from math import sqrt
from typing import Tuple


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score 95% interval for k successes in n trials."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


__all__ = ["wilson"]
