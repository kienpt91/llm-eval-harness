"""Checks of complexity that do not depend on how fast or how busy the machine is.

A wall-clock bound ("finishes in 30 seconds") says more about the runner than about the code. Two things are
robust instead (the one wall-clock limit of the suite is pytest's ``faulthandler_timeout`` in pyproject.toml, which
only stops a test that is stuck):

* *work counters*: the pattern engine counts its own steps, which depend only on the pattern and the text;
* *scaling*: CPU time (not wall time) of the same operation on an input and on one ``factor`` times larger. Linear
  code costs about ``factor`` times more, quadratic code ``factor`` squared times more; a generous limit in between
  tells them apart whatever the speed of the machine, because both measurements slow down together.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable

FLOOR_S = 0.03  # a measurement shorter than this is mostly noise: the input is grown until it is not
MAX_DOUBLINGS = 8


def cpu_seconds(fn: Callable[[], object], repeats: int = 3) -> float:
    """The least CPU time (``time.process_time``) of ``repeats`` runs: the run least disturbed by other work."""
    best = float("inf")
    for _ in range(repeats):
        started = time.process_time()
        fn()
        best = min(best, time.process_time() - started)
    return best


def assert_scales_linearly(
    run: Callable[[int], object], size: int, *, factor: int = 4, limit: float | None = None
) -> float:
    """Assert that ``run(size * factor)`` costs well under ``factor ** 2`` times what ``run(size)`` does.

    ``size`` grows (by doubling) until one run takes at least ``FLOOR_S`` of CPU time. The default ``limit`` sits
    between linear (``factor``) and quadratic (``factor ** 2``). Returns the measured ratio.
    """
    for _ in range(MAX_DOUBLINGS):
        small = cpu_seconds(functools.partial(run, size))
        if small >= FLOOR_S:
            break
        size *= 2
    large = cpu_seconds(functools.partial(run, size * factor))
    ratio = large / max(small, 1e-9)
    bound = (
        limit if limit is not None else (factor + factor * factor) / 2 + 1
    )  # 11 for factor 4: linear 4, quadratic 16
    assert ratio < bound, (
        f"cost grew {ratio:.1f}x for an input {factor}x larger (linear: {factor}x, quadratic: {factor**2}x)"
    )
    return ratio
