"""The scaling check must tell linear from quadratic, or the tests that rely on it prove nothing."""

from __future__ import annotations

import pytest

from support.scaling import assert_scales_linearly


def test_linear_work_passes() -> None:
    ratio = assert_scales_linearly(lambda n: sum(range(n)), 100_000)  # the helper's own limit is the assertion
    assert ratio > 0


def test_quadratic_work_is_caught() -> None:
    def quadratic(n: int) -> int:
        total = 0
        for i in range(n):
            for _ in range(i):
                total += 1
        return total

    with pytest.raises(AssertionError, match="cost grew"):
        assert_scales_linearly(quadratic, 300)
