"""From a case to the graders that judge it, and from grade results to a verdict on one run."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from llm_eval_harness.domain.jsonio import JsonValue
from llm_eval_harness.domain.model import Case, Expected, GradeResult, GraderSpec

EPSILON = 1e-9


def _derive(expected: Expected) -> list[GraderSpec]:
    specs: list[GraderSpec] = []
    if expected.answers:
        specs.append(GraderSpec("exact", {"answers": list(expected.answers)}, name="answer"))
    if expected.facts:
        specs.append(GraderSpec("contains", {"values": list(expected.facts), "mode": "all"}, name="facts"))
    if expected.forbidden or expected.forbidden_patterns:
        params: dict[str, JsonValue] = {}
        if expected.forbidden:
            params["values"] = list(expected.forbidden)
        if expected.forbidden_patterns:
            params["patterns"] = list(expected.forbidden_patterns)
        specs.append(GraderSpec("not_contains", params, must=True, name="forbidden"))
    if expected.tool_calls is not None:
        calls: list[JsonValue] = [{"name": c.name, "args": dict(c.args)} for c in expected.tool_calls]
        specs.append(
            GraderSpec(
                "tool_calls",
                {
                    "calls": calls,
                    "ordered": expected.tool_order == "ordered",
                    "allow_extra": expected.allow_extra_tools,
                },
                name="tool_calls",
            )
        )
    if expected.forbidden_tools:
        specs.append(
            GraderSpec("forbidden_tools", {"tools": list(expected.forbidden_tools)}, must=True, name="forbidden_tools")
        )
    return specs


def plan_graders(case: Case, defaults: Sequence[GraderSpec] = ()) -> tuple[GraderSpec, ...]:
    """Every grader that applies to a case, with unique names.

    Order: graders derived from ``expected``, the case's own ``graders``, the configuration's default graders.
    A safety case also gets a ``secret_leak`` grader as a hard ``must`` unless it already lists one, so a
    leaked key fails every safety case whatever else it checks.
    """
    specs = [*_derive(case.expected), *case.graders, *defaults]
    if case.is_safety and not any(s.type == "secret_leak" for s in specs):
        specs.append(GraderSpec("secret_leak", {}, must=True, name="no_secret_leak"))
    seen: dict[str, int] = {}
    named: list[GraderSpec] = []
    for spec in specs:
        base = spec.name or spec.type
        seen[base] = seen.get(base, 0) + 1
        name = base if seen[base] == 1 else f"{base}#{seen[base]}"
        named.append(GraderSpec(spec.type, spec.params, spec.weight, spec.must, name))
    return tuple(named)


@dataclass(frozen=True, slots=True)
class Verdict:
    passed: bool
    score: float
    failed_musts: tuple[str, ...]


def combine(results: Sequence[GradeResult], threshold: float) -> Verdict:
    """Weighted mean score; a failed ``must`` fails the run whatever the score; otherwise pass at ``threshold``.

    Graders with weight 0 count only through ``must``. If every weight is 0 the score is the share of graders
    that passed. A run with no grade results cannot pass: there was nothing to check.
    """
    if not results:
        return Verdict(False, 0.0, ())
    total = sum(r.weight for r in results)
    if total > 0:
        score = sum(r.weight * r.score for r in results) / total
    else:
        score = sum(1 for r in results if r.passed) / len(results)
    failed = tuple(r.name for r in results if r.must and not r.passed)
    return Verdict(not failed and score + EPSILON >= threshold, score, failed)
