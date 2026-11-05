"""A small deterministic PRNG (SplitMix64). The domain may not import ``random``: this is what seeded sampling uses."""

from __future__ import annotations

from collections.abc import Sequence

_MASK = (1 << 64) - 1


class SplitMix64:
    """64-bit generator with a one-integer state. Same seed, same sequence, on every platform and version."""

    def __init__(self, seed: int) -> None:
        self._state = seed & _MASK

    def next_u64(self) -> int:
        self._state = (self._state + 0x9E3779B97F4A7C15) & _MASK
        z = self._state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
        return z ^ (z >> 31)

    def below(self, bound: int) -> int:
        """Uniform integer in ``[0, bound)`` (rejection sampling, no modulo bias)."""
        if bound <= 0:
            raise ValueError("bound must be positive")
        limit = _MASK - (_MASK + 1) % bound
        while True:
            value = self.next_u64()
            if value <= limit:
                return value % bound


def sample[T](items: Sequence[T], k: int, seed: int) -> list[T]:
    """``k`` items chosen without replacement, in their original order. ``k`` >= len returns all of them."""
    n = len(items)
    if k >= n:
        return list(items)
    rng = SplitMix64(seed)
    indices = list(range(n))
    for i in range(k):  # partial Fisher-Yates
        j = i + rng.below(n - i)
        indices[i], indices[j] = indices[j], indices[i]
    return [items[i] for i in sorted(indices[:k])]
