"""Grading one answer: run every planned grader, record each result, combine them into a verdict."""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal

from llm_eval_harness.application.ports import Clock, JudgeRunner
from llm_eval_harness.application.redaction import Redactor
from llm_eval_harness.domain.errors import ProviderError
from llm_eval_harness.domain.grading.judge import JudgeConfig, judge_outcome, parse_verdict
from llm_eval_harness.domain.grading.registry import GRADERS, evaluate, parsed_config
from llm_eval_harness.domain.limits import MAX_MESSAGE_CHARS
from llm_eval_harness.domain.model import Case, GradeContext, GradeResult, GraderSpec, Outcome, ProviderOutput
from llm_eval_harness.domain.protocol import build_judge_request
from llm_eval_harness.domain.text import truncate


def grade_output(
    case: Case,
    output: ProviderOutput,
    cost: Decimal | None,
    specs: Sequence[GraderSpec],
    judge: JudgeRunner | None,
    redactor: Redactor,
    clock: Clock,
    retries: int,
) -> list[GradeResult]:
    """One ``GradeResult`` per grader. A grader that crashes FAILS (closed) instead of taking the run down.

    Judge graders run through the ``JudgeRunner`` port; an infrastructure failure there raises ``ProviderError``
    after ``retries`` retries, and the caller records the run as an error rather than a grade.
    """
    ctx = GradeContext(case=case, output=output, cost_usd=cost)
    results: list[GradeResult] = []
    for spec in specs:
        if spec.type == "judge":
            results.append(_judge(spec, case, output, judge, redactor, clock, retries))
            continue
        try:
            outcome = evaluate(spec, ctx)
        except Exception as error:  # a bug in a grader must fail the grade, never pass it or stop the run
            outcome = Outcome(False, 0.0, f"grader crashed ({type(error).__name__}); counted as failed")
        results.append(_result(spec, outcome, redactor, deterministic=GRADERS[spec.type].deterministic))
    return results


def _result(
    spec: GraderSpec,
    outcome: Outcome,
    redactor: Redactor,
    *,
    deterministic: bool,
    confidence: float | None = None,
    verdict: str | None = None,
) -> GradeResult:
    return GradeResult(
        name=spec.name or spec.type,
        type=spec.type,
        passed=outcome.passed,
        score=min(1.0, max(0.0, outcome.score)),
        weight=spec.weight,
        must=spec.must,
        detail=truncate(redactor.text(outcome.detail), MAX_MESSAGE_CHARS),  # redact first, then cut
        deterministic=deterministic,
        confidence=confidence,
        verdict=verdict,
    )


def _judge(
    spec: GraderSpec,
    case: Case,
    output: ProviderOutput,
    judge: JudgeRunner | None,
    redactor: Redactor,
    clock: Clock,
    retries: int,
) -> GradeResult:
    if judge is None:
        raise ProviderError("refused", "a judge grader needs a judge runner", retryable=False)
    config: JudgeConfig = parsed_config(spec)
    request = build_judge_request(case, output, config.rubric)
    attempt = 0
    while True:
        attempt += 1
        try:
            raw = judge.run(config, request)
            break
        except ProviderError as error:
            if not error.retryable or attempt > retries:
                raise
            clock.sleep(0.2 * attempt)
    try:
        verdict = parse_verdict(raw)
    except ValueError as error:
        raise ProviderError("protocol", f"judge program: {error}", retryable=False) from None
    outcome = judge_outcome(config, verdict)
    return _result(spec, outcome, redactor, deterministic=False, confidence=verdict.confidence, verdict=verdict.verdict)
