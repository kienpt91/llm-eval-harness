"""Formatting helpers shared by the terminal and Markdown renderers. Every untrusted string passes ``safe``."""

from __future__ import annotations

from llm_eval_harness.domain.compare import Delta
from llm_eval_harness.domain.text import one_line


def safe(value: str, max_chars: int = 120) -> str:
    """Single-line, escape-free rendering of an untrusted string."""
    return one_line(value, max_chars=max_chars)


def money(value: float | None, currency: str) -> str:
    if value is None:
        return "unknown"
    digits = 4 if abs(value) >= 0.01 or value == 0 else 6 if abs(value) >= 0.0001 else 8
    return f"{value:.{digits}f} {safe(currency, 8)}"


def millis(value: float) -> str:
    return f"{value:.0f} ms" if value >= 10 else f"{value:.1f} ms"


def rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def signed_percent(delta: Delta) -> str:
    pct = delta.percent
    if pct is None:
        return "n/a"
    return f"{pct:+.1f}%"


def signed(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:+.{digits}f}"


def plural(n: int, noun: str) -> str:
    return f"{n} {noun}" + ("" if n == 1 else "s")
