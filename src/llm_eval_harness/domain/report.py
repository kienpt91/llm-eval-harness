"""The machine-readable run report (schema version 1) and the summary numbers behind it."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from llm_eval_harness.domain.jsonio import JsonValue
from llm_eval_harness.domain.model import GradeResult
from llm_eval_harness.domain.results import CaseResult, CaseStat, OutputRecord, RunRecord, stat_of
from llm_eval_harness.domain.stats import mean, percentile

SCHEMA_VERSION = 1
REPORT_KIND = "llm-eval-report"
TOOL_NAME = "llm-eval-harness"


@dataclass(frozen=True, slots=True)
class RunSettings:
    repeat: int
    seed: int | None
    sample: int | None
    tags: tuple[str, ...]
    categories: tuple[str, ...]
    concurrency: int
    retries: int
    redacted: bool


@dataclass(frozen=True, slots=True)
class ReportMeta:
    tool_version: str
    started_at: str
    finished_at: str
    dataset_source: str
    dataset_hash: str
    total_cases: int
    provider: dict[str, JsonValue]
    currency: str
    settings: RunSettings


@dataclass(frozen=True, slots=True)
class Summary:
    cases: int
    runs: int
    graded_runs: int
    errored_runs: int
    quality_cases: int
    quality_mean_pass_rate: float | None
    quality_fully_passing: int
    quality_mean_score: float | None
    safety_cases: int
    safety_failing_cases: int
    safety_failing_runs: int
    flaky_cases: int
    input_tokens: int
    output_tokens: int
    total_cost: float | None
    mean_cost_per_run: float | None
    unpriced_runs: int
    latency_p50: float
    latency_p95: float
    latency_max: float
    latency_mean: float


def _outputs(results: Sequence[CaseResult]) -> list[OutputRecord]:
    return [r.output for c in results for r in c.runs if r.output is not None]


def summarize(results: Sequence[CaseResult]) -> Summary:
    stats: list[CaseStat] = [stat_of(c) for c in results]
    quality = [s for s in stats if s.category == "quality"]
    safety_results = [c for c in results if c.category == "safety"]
    safety_stats = [stat_of(c) for c in safety_results]
    rates = [s.pass_rate for s in quality if s.pass_rate is not None]
    outputs = _outputs(results)
    costs = [o.cost_usd for o in outputs if o.cost_usd is not None]
    unpriced = sum(1 for o in outputs if o.cost_usd is None)
    latencies = [o.latency_ms for o in outputs]
    total_cost = float(round(sum(costs, Decimal(0)), 8)) if costs and unpriced == 0 else None
    return Summary(
        cases=len(results),
        runs=sum(len(c.runs) for c in results),
        graded_runs=sum(s.graded_runs for s in stats),
        errored_runs=sum(s.errors for s in stats),
        quality_cases=len(quality),
        quality_mean_pass_rate=round(mean(rates), 6) if rates else None,
        quality_fully_passing=sum(1 for s in quality if s.graded_runs and s.failures == 0 and s.errors == 0),
        quality_mean_score=round(mean([s.mean_score for s in quality if s.graded_runs]), 6)
        if any(s.graded_runs for s in quality)
        else None,
        safety_cases=len(safety_stats),
        safety_failing_cases=sum(1 for s in safety_stats if s.failures),
        safety_failing_runs=sum(s.failures for s in safety_stats),
        flaky_cases=sum(1 for s in stats if s.flaky),
        input_tokens=sum(o.usage.input_tokens for o in outputs if o.usage),
        output_tokens=sum(o.usage.output_tokens for o in outputs if o.usage),
        total_cost=total_cost,
        mean_cost_per_run=round(total_cost / len(outputs), 8) if total_cost is not None and outputs else None,
        unpriced_runs=unpriced,
        latency_p50=round(percentile(latencies, 50), 3),
        latency_p95=round(percentile(latencies, 95), 3),
        latency_max=round(max(latencies, default=0.0), 3),
        latency_mean=round(mean(latencies), 3),
    )


def _grade(g: GradeResult) -> JsonValue:
    return {
        "name": g.name,
        "type": g.type,
        "passed": g.passed,
        "score": round(g.score, 6),
        "weight": g.weight,
        "must": g.must,
        "deterministic": g.deterministic,
        "confidence": g.confidence,
        "verdict": g.verdict,
        "detail": g.detail,
    }


def _output(o: OutputRecord | None) -> JsonValue:
    if o is None:
        return None
    return {
        "text": o.text,
        "truncated": o.truncated,
        "tool_calls": [{"name": c.name, "arguments": dict(c.arguments)} for c in o.tool_calls],
        "citations": list(o.citations),
        "retrieved": list(o.retrieved),
        "usage": None
        if o.usage is None
        else {"model": o.usage.model, "input_tokens": o.usage.input_tokens, "output_tokens": o.usage.output_tokens},
        "latency_ms": round(o.latency_ms, 3),
        "cost": None if o.cost_usd is None else float(round(o.cost_usd, 8)),
    }


def _run(r: RunRecord) -> JsonValue:
    return {
        "repeat": r.repeat,
        "attempts": r.attempts,
        "status": r.status,
        "score": round(r.score, 6),
        "failed_musts": list(r.failed_musts),
        "grades": [_grade(g) for g in r.grades],
        "output": _output(r.output),
        "error": None if r.error is None else {"kind": r.error.kind, "message": r.error.message},
    }


def _case(c: CaseResult) -> JsonValue:
    s = stat_of(c)
    return {
        "id": c.id,
        "category": c.category,
        "tags": list(c.tags),
        "case_hash": c.case_hash,
        "input": c.input_preview,
        "status": s.status,
        "graded_runs": s.graded_runs,
        "passes": s.passes,
        "errors": s.errors,
        "pass_rate": None if s.pass_rate is None else round(s.pass_rate, 6),
        "mean_score": round(s.mean_score, 6),
        "score_variance": round(s.score_variance, 6),
        "flaky": s.flaky,
        "runs": [_run(r) for r in sorted(c.runs, key=lambda r: r.repeat)],
    }


def summary_json(s: Summary, currency: str) -> JsonValue:
    return {
        "cases": s.cases,
        "runs": s.runs,
        "graded_runs": s.graded_runs,
        "errored_runs": s.errored_runs,
        "quality": {
            "cases": s.quality_cases,
            "mean_pass_rate": s.quality_mean_pass_rate,
            "fully_passing": s.quality_fully_passing,
            "mean_score": s.quality_mean_score,
        },
        "safety": {
            "cases": s.safety_cases,
            "failing_cases": s.safety_failing_cases,
            "failing_runs": s.safety_failing_runs,
        },
        "flaky_cases": s.flaky_cases,
        "tokens": {"input": s.input_tokens, "output": s.output_tokens},
        "cost": {
            "currency": currency,
            "total": s.total_cost,
            "mean_per_run": s.mean_cost_per_run,
            "unpriced_runs": s.unpriced_runs,
        },
        "latency_ms": {"p50": s.latency_p50, "p95": s.latency_p95, "max": s.latency_max, "mean": s.latency_mean},
    }


def build_report(meta: ReportMeta, results: Sequence[CaseResult]) -> dict[str, JsonValue]:
    """The report as a JSON-able object. Cases are sorted by id, runs by repeat: the order never depends on timing."""
    ordered = sorted(results, key=lambda c: c.id)
    s = meta.settings
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "tool": {"name": TOOL_NAME, "version": meta.tool_version},
        "started_at": meta.started_at,
        "finished_at": meta.finished_at,
        "dataset": {
            "source": meta.dataset_source,
            "hash": meta.dataset_hash,
            "total_cases": meta.total_cases,
            "selected_cases": len(ordered),
        },
        "run": {
            "repeat": s.repeat,
            "seed": s.seed,
            "sample": s.sample,
            "tags": list(s.tags),
            "categories": list(s.categories),
            "concurrency": s.concurrency,
            "retries": s.retries,
            "redacted": s.redacted,
        },
        "provider": meta.provider,
        "summary": summary_json(summarize(ordered), meta.currency),
        "cases": [_case(c) for c in ordered],
    }
