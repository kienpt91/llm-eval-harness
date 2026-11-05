"""Small statistics: percentiles, spread, an exact test for two pass rates, a confidence interval."""

from __future__ import annotations

import math
from collections.abc import Sequence


def mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values) if values else 0.0


def variance(values: Sequence[float]) -> float:
    """Sample variance (n - 1); zero for fewer than two values."""
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.fsum((v - m) ** 2 for v in values) / (len(values) - 1)


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile (``q`` in 0..100) of the values; zero for no values. Deterministic, no interpolation."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def fisher_exact(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided Fisher exact p-value for k1/n1 successes against k2/n2 (sum of tables no more likely than observed)."""
    total, successes = n1 + n2, k1 + k2
    if n1 == 0 or n2 == 0:
        return 1.0
    denominator = math.comb(total, n1)

    def probability(x: int) -> int:
        return math.comb(successes, x) * math.comb(total - successes, n1 - x)

    lo, hi = max(0, n1 - (total - successes)), min(n1, successes)
    observed = probability(k1)
    mass = sum(probability(x) for x in range(lo, hi + 1) if probability(x) <= observed)
    return min(1.0, mass / denominator)


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion; (0, 1) when there are no trials."""
    if trials == 0:
        return 0.0, 1.0
    p = successes / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)
