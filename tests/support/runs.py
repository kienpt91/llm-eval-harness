"""Constructors for run records, case results and reports."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

from llm_eval_harness.domain.model import GradeResult, Usage
from llm_eval_harness.domain.report import ReportMeta, RunSettings, build_report
from llm_eval_harness.domain.reportdoc import ReportDoc, parse_report
from llm_eval_harness.domain.results import CaseResult, ErrorRecord, OutputRecord, RunRecord

DEFAULT_USAGE = Usage("m", 10, 5)


def run_record(
    status: str = "passed",
    *,
    repeat: int = 0,
    score: float | None = None,
    text: str = "answer",
    latency_ms: float = 10.0,
    cost: str | None = "0.001",
    usage: Usage | None = DEFAULT_USAGE,
    failed_musts: tuple[str, ...] = (),
    error_kind: str = "timeout",
) -> RunRecord:
    if status == "error":
        return RunRecord(repeat, 2, "error", 0.0, (), (), None, ErrorRecord(error_kind, "the provider failed"))
    value = score if score is not None else (1.0 if status == "passed" else 0.0)
    grade_result = GradeResult("facts", "contains", status == "passed", value, 1.0, bool(failed_musts), "detail")
    return RunRecord(
        repeat,
        1,
        status,
        value,
        failed_musts,
        (grade_result,),
        OutputRecord(text, False, (), (), (), usage, latency_ms, None if cost is None else Decimal(cost)),
        None,
    )


def case_result(
    case_id: str,
    statuses: tuple[str, ...] = ("passed",),
    *,
    category: str = "quality",
    case_hash: str = "h",
    **kwargs: object,
) -> CaseResult:
    runs = tuple(run_record(s, repeat=i, **kwargs) for i, s in enumerate(statuses))  # type: ignore[arg-type]
    return CaseResult(case_id, category, ("t",), case_hash, "the input", runs)


def report_meta(**overrides: object) -> ReportMeta:
    settings = RunSettings(
        repeat=int(overrides.pop("repeat", 1)),  # type: ignore[call-overload]
        seed=None,
        sample=None,
        tags=(),
        categories=(),
        concurrency=2,
        retries=1,
        redacted=True,
    )
    meta = ReportMeta(
        tool_version="0.1.0",
        started_at="2026-01-01T00:00:00Z",
        finished_at="2026-01-01T00:00:01Z",
        dataset_source="datasets/x.jsonl",
        dataset_hash="sha256:abc",
        total_cases=3,
        provider={"type": "fake"},
        currency="USD",
        settings=settings,
    )
    return replace(meta, **overrides)  # type: ignore[arg-type]


def report_doc(results: list[CaseResult], **overrides: object) -> ReportDoc:
    return parse_report(build_report(report_meta(**overrides), results), "report.json")
