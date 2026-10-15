"""Small statistics helpers."""

from __future__ import annotations

import math


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def upper_bound_zero_failures(n: int, conf: float = 0.95) -> float:
    """Exact upper confidence bound on a failure rate when 0 of n sampled items failed."""
    return 1.0 - (1.0 - conf) ** (1.0 / n)
