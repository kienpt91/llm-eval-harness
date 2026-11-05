"""A seeded PRNG for property tests: every run of the suite explores the same inputs, and a failure names its seed."""

from __future__ import annotations

import random
import string

ALPHABET = string.ascii_letters + string.digits + " .,;:!?-_/\\'\"()[]{}<>@#$%^&*+=~`|\n\t"
HOSTILE = [
    "\x1b[31mred\x1b[0m",
    "\x1b]0;title\x07",
    "\x1b]8;;http://x.invalid\x1b\\link\x1b]8;;\x1b\\",
    "\u202eevil\u202c",
    "\u2066x\u2069",
    "a\u200bb",
    "\x00\x01\x02\x7f\x9b",
    "\r\x08\x0c",
    "\ud800",
    "\U0001f600",
    "\u2028\u2029",
    "e\u0301",
]


class Prng:
    """Thin wrapper over a seeded ``random.Random`` with the generators the tests need."""

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self._r = random.Random(seed)

    def integer(self, lo: int, hi: int) -> int:
        return self._r.randint(lo, hi)

    def real(self, lo: float, hi: float) -> float:
        return self._r.uniform(lo, hi)

    def choice[T](self, items: list[T]) -> T:
        return self._r.choice(items)

    def text(self, max_len: int = 60, *, hostile: bool = False) -> str:
        pieces = [self._r.choice(ALPHABET) for _ in range(self._r.randint(0, max_len))]
        if hostile:
            for _ in range(self._r.randint(0, 3)):
                pieces.insert(self._r.randint(0, len(pieces)), self._r.choice(HOSTILE))
        return "".join(pieces)

    def word(self) -> str:
        return "".join(self._r.choice(string.ascii_lowercase) for _ in range(self._r.randint(1, 8)))

    def shuffled[T](self, items: list[T]) -> list[T]:
        out = list(items)
        self._r.shuffle(out)
        return out
