"""The runner: every (case, repeat) pair through the provider, graded, with bounded concurrency and honest retries.

Retries exist for infrastructure only. A provider call that fails with a retryable error (timeout, spawn,
network, non-zero exit) is repeated up to ``retries`` times; a call that returns an answer is graded once and
that grade stands. A failed grade is never retried, so retries cannot hide a regression.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from decimal import Decimal

from llm_eval_harness.application.grading import grade_output
from llm_eval_harness.application.ports import Clock, JudgeRunner, Provider
from llm_eval_harness.application.redaction import Redactor, bounded_json
from llm_eval_harness.domain.errors import ProviderError
from llm_eval_harness.domain.jsonio import JsonValue, canonical
from llm_eval_harness.domain.limits import MAX_MESSAGE_CHARS, STORED_OUTPUT_CHARS
from llm_eval_harness.domain.model import Case, GraderSpec, ProviderOutput, ToolCall, Usage
from llm_eval_harness.domain.plan import combine, plan_graders
from llm_eval_harness.domain.pricing import EMPTY_TABLE, PriceTable, cost_of
from llm_eval_harness.domain.protocol import build_request
from llm_eval_harness.domain.results import ERROR, FAILED, PASSED, CaseResult, ErrorRecord, OutputRecord, RunRecord
from llm_eval_harness.domain.text import one_line, truncate

INPUT_PREVIEW_CHARS = 2000
TOOL_ARGUMENT_CHARS = 4000


@dataclass(frozen=True, slots=True)
class RunOptions:
    repeat: int = 1
    concurrency: int = 4
    timeout_s: float = 30.0
    retries: int = 1
    seed: int | None = None
    default_graders: tuple[GraderSpec, ...] = ()
    prices: PriceTable = field(default_factory=lambda: EMPTY_TABLE)


def run_cases(
    cases: Sequence[Case],
    provider: Provider,
    judge: JudgeRunner | None,
    redactor: Redactor,
    clock: Clock,
    options: RunOptions,
) -> list[CaseResult]:
    """Run ``options.repeat`` times every case. The result order is by case id and repeat, never by completion."""
    tasks = [(case, repeat) for case in sorted(cases, key=lambda c: c.id) for repeat in range(options.repeat)]

    def execute(task: tuple[Case, int]) -> RunRecord:
        case, repeat = task
        try:
            return _run_once(case, repeat, provider, judge, redactor, clock, options)
        except Exception as error:  # an unforeseen bug becomes an error run: the gate then exits 3, not 0
            return _error_run(
                repeat, 1, ProviderError("internal", f"{type(error).__name__}", retryable=False), redactor
            )

    with ThreadPoolExecutor(max_workers=max(1, options.concurrency)) as pool:
        records = list(pool.map(execute, tasks))
    by_case: dict[str, list[RunRecord]] = {}
    for (case, _), record in zip(tasks, records, strict=True):
        by_case.setdefault(case.id, []).append(record)
    return [
        CaseResult(
            id=case.id,
            category=case.category,
            tags=case.tags,
            case_hash=case.hash,
            input_preview=truncate(redactor.text(canonical(case.input)), INPUT_PREVIEW_CHARS),
            runs=tuple(by_case[case.id]),
        )
        for case in sorted(cases, key=lambda c: c.id)
    ]


def _run_once(
    case: Case,
    repeat: int,
    provider: Provider,
    judge: JudgeRunner | None,
    redactor: Redactor,
    clock: Clock,
    options: RunOptions,
) -> RunRecord:
    attempt = 0
    while True:
        attempt += 1
        request = build_request(case, repeat=repeat, attempt=attempt, seed=options.seed)
        try:
            output = provider.generate(request, options.timeout_s)
            break
        except ProviderError as error:
            if not error.retryable or attempt > options.retries:
                return _error_run(repeat, attempt, error, redactor)
            clock.sleep(0.2 * attempt)
    cost = cost_of(output.usage, options.prices)
    specs = plan_graders(case, options.default_graders)
    try:
        grades = grade_output(case, output, cost, specs, judge, redactor, clock, options.retries)
    except ProviderError as error:  # the judge could not be reached: there is no grade, so this is an error run
        return _error_run(repeat, attempt, error, redactor)
    verdict = combine(grades, case.threshold)
    return RunRecord(
        repeat=repeat,
        attempts=attempt,
        status=PASSED if verdict.passed else FAILED,
        score=verdict.score,
        failed_musts=verdict.failed_musts,
        grades=tuple(grades),
        output=_store(output, cost, redactor),
        error=None,
    )


def _error_run(repeat: int, attempts: int, error: ProviderError, redactor: Redactor) -> RunRecord:
    message = redactor.text(one_line(str(error), max_chars=MAX_MESSAGE_CHARS))
    return RunRecord(repeat, attempts, ERROR, 0.0, (), (), None, ErrorRecord(error.kind, message))


def _store(output: ProviderOutput, cost: Decimal | None, redactor: Redactor) -> OutputRecord:
    text = redactor.text(output.text)
    calls = tuple(
        ToolCall(
            redactor.text(c.name),
            bounded_json(_dict(redactor.json(dict(c.arguments))), TOOL_ARGUMENT_CHARS),
        )
        for c in output.tool_calls
    )
    return OutputRecord(
        text=truncate(text, STORED_OUTPUT_CHARS),
        truncated=len(text) > STORED_OUTPUT_CHARS,
        tool_calls=calls,
        citations=tuple(redactor.text(c) for c in output.citations),
        retrieved=tuple(redactor.text(c) for c in output.retrieved),
        usage=None
        if output.usage is None
        else Usage(
            None if output.usage.model is None else redactor.text(output.usage.model),
            output.usage.input_tokens,
            output.usage.output_tokens,
        ),
        latency_ms=output.latency_ms,
        cost_usd=cost,
    )


def _dict(value: JsonValue) -> dict[str, JsonValue]:
    assert isinstance(value, dict)
    return value
