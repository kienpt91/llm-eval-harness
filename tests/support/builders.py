"""Small constructors for domain objects, so a test states only what it cares about."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import cast

from llm_eval_harness.domain.errors import Problem
from llm_eval_harness.domain.jsonio import JsonValue
from llm_eval_harness.domain.model import (
    Case,
    ContextDocument,
    Expected,
    GradeContext,
    GraderSpec,
    Outcome,
    ProviderOutput,
    ToolCall,
    Usage,
)


def case(
    case_id: str = "c1",
    *,
    category: str = "quality",
    tags: tuple[str, ...] = (),
    input: JsonValue = "question",
    context: tuple[ContextDocument, ...] = (),
    expected: Expected | None = None,
    graders: tuple[GraderSpec, ...] = (),
    threshold: float = 1.0,
    hash: str = "h",
) -> Case:
    return Case(
        id=case_id,
        category=category,
        tags=tags,
        input=input,
        context=context,
        expected=expected or Expected(),
        graders=graders,
        threshold=threshold,
        metadata={},
        hash=hash,
    )


def output(
    text: str = "",
    *,
    tool_calls: tuple[ToolCall, ...] = (),
    citations: tuple[str, ...] = (),
    retrieved: tuple[str, ...] = (),
    usage: Usage | None = None,
    latency_ms: float = 0.0,
    structured: JsonValue = None,
    has_structured: bool = False,
) -> ProviderOutput:
    return ProviderOutput(
        text=text,
        structured=structured,
        has_structured=has_structured,
        tool_calls=tool_calls,
        citations=citations,
        retrieved=retrieved,
        usage=usage,
        latency_ms=latency_ms,
    )


def ctx(
    out: ProviderOutput | None = None, *, the_case: Case | None = None, cost: Decimal | None = None
) -> GradeContext:
    return GradeContext(case=the_case or case(), output=out or output(), cost_usd=cost)


def js(value: object) -> JsonValue:
    """Tell the type checker that a literal built in a test is JSON."""
    return cast(JsonValue, value)


def spec(
    grader_type: str, params: Mapping[str, object] | None = None, *, weight: float = 1.0, must: bool = False
) -> GraderSpec:
    return GraderSpec(type=grader_type, params=cast(dict[str, JsonValue], dict(params or {})), weight=weight, must=must)


def validate(grader_type: str, params: Mapping[str, object], path: str = "g") -> list[Problem]:
    from llm_eval_harness.domain.grading.registry import validate_params

    return validate_params(grader_type, cast(dict[str, JsonValue], dict(params)), path)


def grade(
    grader_type: str,
    params: Mapping[str, object],
    out: ProviderOutput,
    *,
    the_case: Case | None = None,
    cost: Decimal | None = None,
) -> Outcome:
    from llm_eval_harness.domain.grading.registry import evaluate

    problems = validate(grader_type, params, grader_type)
    assert problems == [], [p.render() for p in problems]
    return evaluate(spec(grader_type, params), ctx(out, the_case=the_case, cost=cost))
