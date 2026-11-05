"""numeric: compare a number found in the answer with an expected value within a tolerance."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass

from llm_eval_harness.domain.errors import Problem
from llm_eval_harness.domain.grading.params import Params
from llm_eval_harness.domain.jsonio import JsonValue
from llm_eval_harness.domain.model import GradeContext, Outcome

# optional sign, digits with optional thousands separators or none, optional fraction, optional exponent
_NUMBER = re.compile(
    r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3}){1,10}+|\d{1,30}+)(?:\.\d{1,30}+)?(?:[eE][-+]?\d{1,4}+)?(?![\w])"
)


@dataclass(frozen=True, slots=True)
class NumericConfig:
    expected: float
    abs_tol: float
    rel_tol: float
    which: str


def parse_numeric(raw: Mapping[str, JsonValue], path: str, problems: list[Problem]) -> NumericConfig:
    p = Params(raw, path, problems, {"expected", "abs_tol", "rel_tol", "which"})
    return NumericConfig(
        p.number("expected", required=True) or 0.0,
        p.number("abs_tol", default=0.0, minimum=0.0) or 0.0,
        p.number("rel_tol", default=0.0, minimum=0.0) or 0.0,
        p.choice("which", ("first", "last", "any"), default="first"),
    )


def extract_numbers(text: str, limit: int = 1000) -> list[float]:
    """Numbers in a text, in order. Thousands separators are removed; a decimal comma is not understood."""
    out: list[float] = []
    for match in _NUMBER.finditer(text):
        try:
            value = float(match.group().replace(",", ""))
        except ValueError:
            continue
        if math.isfinite(value):
            out.append(value)
        if len(out) >= limit:
            break
    return out


def grade_numeric(config: NumericConfig, ctx: GradeContext) -> Outcome:
    """Passes when the first (or last, or any) number in the answer is within tolerance of the expected value.

    Within tolerance means ``|got - expected| <= max(abs_tol, rel_tol * |expected|)``. Failure modes: the
    number is taken from free text, so the first number may be a date or a quantity ("3 items cost 12.50"
    with ``which: first`` reads 3); use ``last`` or ``any``, or ask the pipeline for structured output.
    Decimal commas and spelled-out numbers are not understood; thousands separators are.
    """
    numbers = extract_numbers(ctx.output.text)
    if not numbers:
        return Outcome(False, 0.0, "no number found in the answer")
    allowed = max(config.abs_tol, config.rel_tol * abs(config.expected))
    candidates = {"first": numbers[:1], "last": numbers[-1:], "any": numbers}[config.which]
    best = min(candidates, key=lambda n: abs(n - config.expected))
    delta = abs(best - config.expected)
    if delta <= allowed:
        return Outcome(True, 1.0, f"{best:g} is within {allowed:g} of {config.expected:g}")
    return Outcome(False, 0.0, f"{best:g} differs from {config.expected:g} by {delta:g} (allowed {allowed:g})")
