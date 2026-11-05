"""Reading a report or baseline file back: strict validation of untrusted JSON into typed documents."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from llm_eval_harness.domain.errors import Problem, ValidationError
from llm_eval_harness.domain.identifiers import ID_PATTERN
from llm_eval_harness.domain.jsonio import JsonValue, canonical, kind_of
from llm_eval_harness.domain.limits import MAX_REPEAT, MAX_SAFETY_LISTED
from llm_eval_harness.domain.model import CATEGORIES
from llm_eval_harness.domain.report import REPORT_KIND, SCHEMA_VERSION, Summary
from llm_eval_harness.domain.results import ERROR, FAILED, PASSED, CaseStat
from llm_eval_harness.domain.stats import mean, percentile

BASELINE_KIND = "llm-eval-baseline"
MAX_CASES_READ = 100_000


class Reader:
    """Typed access to a JSON document; the first violation raises a ``ValidationError`` naming its path."""

    def __init__(self, source: str) -> None:
        self.source = source

    def fail(self, path: str, message: str) -> ValidationError:
        return ValidationError(self.source, [Problem(path, message)])

    def obj(self, value: JsonValue, path: str) -> dict[str, JsonValue]:
        if not isinstance(value, dict):
            raise self.fail(path, f"must be an object (got {kind_of(value)})")
        return value

    def lst(self, value: JsonValue, path: str) -> list[JsonValue]:
        if not isinstance(value, list):
            raise self.fail(path, f"must be an array (got {kind_of(value)})")
        return value

    def child(self, d: Mapping[str, JsonValue], key: str, path: str) -> JsonValue:
        if key not in d:
            raise self.fail(f"{path}.{key}" if path else key, "is missing")
        return d[key]

    def s(self, d: Mapping[str, JsonValue], key: str, path: str, *, nullable: bool = False) -> str | None:
        value = self.child(d, key, path)
        where = f"{path}.{key}" if path else key
        if value is None and nullable:
            return None
        if not isinstance(value, str):
            raise self.fail(where, f"must be a string (got {kind_of(value)})")
        return value

    def req_s(self, d: Mapping[str, JsonValue], key: str, path: str) -> str:
        value = self.s(d, key, path)
        assert value is not None
        return value

    def n(self, d: Mapping[str, JsonValue], key: str, path: str, *, nullable: bool = False) -> float | None:
        value = self.child(d, key, path)
        where = f"{path}.{key}" if path else key
        if value is None and nullable:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise self.fail(where, f"must be a number (got {kind_of(value)})")
        return float(value)

    def req_n(self, d: Mapping[str, JsonValue], key: str, path: str) -> float:
        value = self.n(d, key, path)
        assert value is not None
        return value

    def i(self, d: Mapping[str, JsonValue], key: str, path: str) -> int:
        value = self.child(d, key, path)
        where = f"{path}.{key}" if path else key
        if isinstance(value, bool) or not isinstance(value, int):
            raise self.fail(where, f"must be a whole number (got {kind_of(value)})")
        return value

    def b(self, d: Mapping[str, JsonValue], key: str, path: str) -> bool:
        value = self.child(d, key, path)
        if not isinstance(value, bool):
            raise self.fail(f"{path}.{key}" if path else key, f"must be true or false (got {kind_of(value)})")
        return value

    def strs(self, d: Mapping[str, JsonValue], key: str, path: str) -> tuple[str, ...]:
        items = self.lst(self.child(d, key, path), f"{path}.{key}" if path else key)
        out: list[str] = []
        for index, item in enumerate(items):
            if not isinstance(item, str):
                raise self.fail(
                    f"{path}.{key}[{index}]" if path else f"{key}[{index}]", f"must be a string (got {kind_of(item)})"
                )
            out.append(item)
        return tuple(out)

    def header(self, doc: JsonValue, kind: str) -> dict[str, JsonValue]:
        root = self.obj(doc, "")
        if self.s(root, "kind", "") != kind:
            raise self.fail("kind", f"must be {kind!r}: this is not a {kind} file")
        version = self.i(root, "schema_version", "")
        if version != SCHEMA_VERSION:
            raise self.fail("schema_version", f"is {version}; this tool reads version {SCHEMA_VERSION}")
        return root


@dataclass(frozen=True, slots=True)
class GradeDoc:
    name: str
    type: str
    passed: bool
    score: float
    must: bool
    detail: str
    deterministic: bool
    confidence: float | None
    verdict: str | None


@dataclass(frozen=True, slots=True)
class RunDoc:
    repeat: int
    attempts: int
    status: str
    score: float
    failed_musts: tuple[str, ...]
    grades: tuple[GradeDoc, ...]
    text: str | None
    truncated: bool
    tool_calls: tuple[tuple[str, str], ...]
    citations: tuple[str, ...]
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost: float | None
    error: tuple[str, str] | None


@dataclass(frozen=True, slots=True)
class CaseDoc:
    stat: CaseStat
    tags: tuple[str, ...]
    input: str
    status: str
    runs: tuple[RunDoc, ...]


@dataclass(frozen=True, slots=True)
class ReportDoc:
    tool_version: str
    started_at: str
    finished_at: str
    dataset_source: str
    dataset_hash: str
    total_cases: int
    repeat: int
    seed: int | None
    sample: int | None
    concurrency: int
    retries: int
    redacted: bool
    provider: dict[str, JsonValue]
    currency: str
    summary: Summary
    cases: tuple[CaseDoc, ...]
    safety_not_run: tuple[str, ...] = ()
    safety_not_run_count: int = 0


def _grade(r: Reader, value: JsonValue, path: str) -> GradeDoc:
    g = r.obj(value, path)
    return GradeDoc(
        r.req_s(g, "name", path),
        r.req_s(g, "type", path),
        r.b(g, "passed", path),
        r.req_n(g, "score", path),
        r.b(g, "must", path),
        r.req_s(g, "detail", path),
        r.b(g, "deterministic", path),
        r.n(g, "confidence", path, nullable=True),
        r.s(g, "verdict", path, nullable=True),
    )


def _run(r: Reader, value: JsonValue, path: str) -> RunDoc:
    d = r.obj(value, path)
    out_raw = r.child(d, "output", path)
    err_raw = r.child(d, "error", path)
    text: str | None = None
    truncated = False
    calls: list[tuple[str, str]] = []
    citations: tuple[str, ...] = ()
    tokens_in = tokens_out = 0
    latency = 0.0
    cost: float | None = None
    if out_raw is not None:
        o = r.obj(out_raw, f"{path}.output")
        text = r.req_s(o, "text", f"{path}.output")
        truncated = r.b(o, "truncated", f"{path}.output")
        for index, call in enumerate(r.lst(r.child(o, "tool_calls", f"{path}.output"), f"{path}.output.tool_calls")):
            c = r.obj(call, f"{path}.output.tool_calls[{index}]")
            calls.append(
                (r.req_s(c, "name", f"{path}.output.tool_calls[{index}]"), canonical(r.child(c, "arguments", "")))
            )
        citations = r.strs(o, "citations", f"{path}.output")
        usage = r.child(o, "usage", f"{path}.output")
        if usage is not None:
            u = r.obj(usage, f"{path}.output.usage")
            tokens_in, tokens_out = (
                r.i(u, "input_tokens", f"{path}.output.usage"),
                r.i(u, "output_tokens", f"{path}.output.usage"),
            )
        latency = r.req_n(o, "latency_ms", f"{path}.output")
        cost = r.n(o, "cost", f"{path}.output", nullable=True)
    error: tuple[str, str] | None = None
    if err_raw is not None:
        e = r.obj(err_raw, f"{path}.error")
        error = (r.req_s(e, "kind", f"{path}.error"), r.req_s(e, "message", f"{path}.error"))
    grades = tuple(
        _grade(r, g, f"{path}.grades[{i}]") for i, g in enumerate(r.lst(r.child(d, "grades", path), f"{path}.grades"))
    )
    status = r.req_s(d, "status", path)
    if status not in (PASSED, FAILED, ERROR):
        raise r.fail(f"{path}.status", f"must be {PASSED}, {FAILED} or {ERROR} (got {status!r})")
    if (status == ERROR) != (error is not None) or (status == ERROR) != (out_raw is None):
        raise r.fail(path, "has an error without status error, or status error without an error (or an output)")
    score = r.req_n(d, "score", path)
    if not 0.0 <= score <= 1.0:
        raise r.fail(f"{path}.score", f"must be between 0 and 1 (got {score})")
    failed_musts = r.strs(d, "failed_musts", path)
    if failed_musts and status != FAILED:
        raise r.fail(f"{path}.failed_musts", "can only be set on a failed run")
    return RunDoc(
        r.i(d, "repeat", path),
        r.i(d, "attempts", path),
        status,
        score,
        failed_musts,
        grades,
        text,
        truncated,
        tuple(calls),
        citations,
        tokens_in,
        tokens_out,
        latency,
        cost,
        error,
    )


def _case(r: Reader, value: JsonValue, path: str) -> CaseDoc:
    c = r.obj(value, path)
    stat = CaseStat(
        id=r.req_s(c, "id", path),
        category=r.req_s(c, "category", path),
        case_hash=r.req_s(c, "case_hash", path),
        graded_runs=r.i(c, "graded_runs", path),
        passes=r.i(c, "passes", path),
        errors=r.i(c, "errors", path),
        mean_score=r.req_n(c, "mean_score", path),
        score_variance=r.req_n(c, "score_variance", path),
    )
    if not ID_PATTERN.fullmatch(stat.id) or len(stat.id) > 128:
        raise r.fail(f"{path}.id", "must use letters, digits and . _ : / - only, and be at most 128 characters")
    if stat.category not in CATEGORIES:
        raise r.fail(f"{path}.category", f"must be one of {', '.join(CATEGORIES)}")
    runs_raw = r.lst(r.child(c, "runs", path), f"{path}.runs")
    if len(runs_raw) > MAX_REPEAT:
        raise r.fail(f"{path}.runs", f"has more than {MAX_REPEAT} runs")
    runs = tuple(_run(r, v, f"{path}.runs[{i}]") for i, v in enumerate(runs_raw))
    # the counts are redundant with the runs: they must agree, so that nobody has to trust which one is edited
    real = CaseStat(
        stat.id,
        stat.category,
        stat.case_hash,
        graded_runs=sum(1 for run in runs if run.status != ERROR),
        passes=sum(1 for run in runs if run.status == PASSED),
        errors=sum(1 for run in runs if run.status == ERROR),
        mean_score=stat.mean_score,
    )
    for name in ("graded_runs", "passes", "errors"):
        if getattr(stat, name) != getattr(real, name):
            raise r.fail(f"{path}.{name}", f"is {getattr(stat, name)} but its runs show {getattr(real, name)}")
    status = r.req_s(c, "status", path)
    if status != real.status:
        raise r.fail(f"{path}.status", f"is {status!r} but its runs show {real.status!r}")
    return CaseDoc(stat, r.strs(c, "tags", path), r.req_s(c, "input", path), status, runs)


def _summary(r: Reader, value: JsonValue) -> Summary:
    s = r.obj(value, "summary")
    q = r.obj(r.child(s, "quality", "summary"), "summary.quality")
    sa = r.obj(r.child(s, "safety", "summary"), "summary.safety")
    tk = r.obj(r.child(s, "tokens", "summary"), "summary.tokens")
    co = r.obj(r.child(s, "cost", "summary"), "summary.cost")
    la = r.obj(r.child(s, "latency_ms", "summary"), "summary.latency_ms")
    return Summary(
        cases=r.i(s, "cases", "summary"),
        runs=r.i(s, "runs", "summary"),
        graded_runs=r.i(s, "graded_runs", "summary"),
        errored_runs=r.i(s, "errored_runs", "summary"),
        quality_cases=r.i(q, "cases", "summary.quality"),
        quality_mean_pass_rate=r.n(q, "mean_pass_rate", "summary.quality", nullable=True),
        quality_fully_passing=r.i(q, "fully_passing", "summary.quality"),
        quality_mean_score=r.n(q, "mean_score", "summary.quality", nullable=True),
        safety_cases=r.i(sa, "cases", "summary.safety"),
        safety_failing_cases=r.i(sa, "failing_cases", "summary.safety"),
        safety_failing_runs=r.i(sa, "failing_runs", "summary.safety"),
        flaky_cases=r.i(s, "flaky_cases", "summary"),
        input_tokens=r.i(tk, "input", "summary.tokens"),
        output_tokens=r.i(tk, "output", "summary.tokens"),
        total_cost=r.n(co, "total", "summary.cost", nullable=True),
        mean_cost_per_run=r.n(co, "mean_per_run", "summary.cost", nullable=True),
        unpriced_runs=r.i(co, "unpriced_runs", "summary.cost"),
        latency_p50=r.req_n(la, "p50", "summary.latency_ms"),
        latency_p95=r.req_n(la, "p95", "summary.latency_ms"),
        latency_max=r.req_n(la, "max", "summary.latency_ms"),
        latency_mean=r.req_n(la, "mean", "summary.latency_ms"),
    )


def parse_report(doc: JsonValue, source: str) -> ReportDoc:
    """Validate a report document (as written by ``build_report``) and return it typed."""
    r = Reader(source)
    root = r.header(doc, REPORT_KIND)
    tool = r.obj(r.child(root, "tool", ""), "tool")
    dataset = r.obj(r.child(root, "dataset", ""), "dataset")
    run = r.obj(r.child(root, "run", ""), "run")
    cases_raw = r.lst(r.child(root, "cases", ""), "cases")
    if len(cases_raw) > MAX_CASES_READ:
        raise r.fail("cases", f"has more than {MAX_CASES_READ} cases")
    seed = r.n(run, "seed", "run", nullable=True)
    sample = r.n(run, "sample", "run", nullable=True)
    summary_raw = r.child(root, "summary", "")
    not_run_raw = r.lst(r.child(dataset, "safety_not_run", "dataset"), "dataset.safety_not_run")
    not_run = r.strs(dataset, "safety_not_run", "dataset")
    for index, case_id in enumerate(not_run):
        if not ID_PATTERN.fullmatch(case_id) or len(case_id) > 128:
            raise r.fail(f"dataset.safety_not_run[{index}]", "is not a valid case id")
    not_run_count = r.i(dataset, "safety_not_run_count", "dataset")
    if len(not_run) > MAX_SAFETY_LISTED or not len(not_run_raw) <= not_run_count:
        raise r.fail("dataset.safety_not_run_count", f"is {not_run_count}, less than the {len(not_run_raw)} ids listed")
    report = ReportDoc(
        tool_version=r.req_s(tool, "version", "tool"),
        started_at=r.req_s(root, "started_at", ""),
        finished_at=r.req_s(root, "finished_at", ""),
        dataset_source=r.req_s(dataset, "source", "dataset"),
        dataset_hash=r.req_s(dataset, "hash", "dataset"),
        total_cases=r.i(dataset, "total_cases", "dataset"),
        repeat=r.i(run, "repeat", "run"),
        seed=None if seed is None else int(seed),
        sample=None if sample is None else int(sample),
        concurrency=r.i(run, "concurrency", "run"),
        retries=r.i(run, "retries", "run"),
        redacted=r.b(run, "redacted", "run"),
        provider=r.obj(r.child(root, "provider", ""), "provider"),
        currency=r.req_s(
            r.obj(r.child(r.obj(summary_raw, "summary"), "cost", "summary"), "summary.cost"), "currency", "summary.cost"
        ),
        summary=_summary(r, summary_raw),
        cases=tuple(_case(r, c, f"cases[{i}]") for i, c in enumerate(cases_raw)),
        safety_not_run=not_run,
        safety_not_run_count=not_run_count,
    )
    _check_consistency(r, report, r.i(dataset, "selected_cases", "dataset"))
    return report


def _close(a: float | None, b: float | None, tolerance: float = 1e-6) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=1e-6, abs_tol=tolerance)


def _check_consistency(r: Reader, doc: ReportDoc, selected: int) -> None:
    """The report says the same thing twice (cases and runs, then the summary); the two must agree."""
    seen: set[str] = set()
    for index, case in enumerate(doc.cases):
        if case.stat.id in seen:
            raise r.fail(f"cases[{index}].id", f"duplicate case id {case.stat.id!r}")
        seen.add(case.stat.id)
    if selected != len(doc.cases):
        raise r.fail("dataset.selected_cases", f"is {selected} but the report has {len(doc.cases)} cases")
    s = doc.summary
    stats = [c.stat for c in doc.cases]
    runs = [run for c in doc.cases for run in c.runs]
    outputs = [run for run in runs if run.text is not None]
    quality = [x for x in stats if x.category == "quality"]
    safety = [x for x in stats if x.category == "safety"]
    rates = [x.pass_rate for x in quality if x.pass_rate is not None]
    costs = [run.cost for run in outputs if run.cost is not None]
    unpriced = sum(1 for run in outputs if run.cost is None)
    latencies = [run.latency_ms for run in outputs]
    total_cost = sum(costs) if costs and unpriced == 0 else None
    expected: list[tuple[str, object, object]] = [
        ("summary.cases", s.cases, len(doc.cases)),
        ("summary.runs", s.runs, len(runs)),
        ("summary.graded_runs", s.graded_runs, sum(x.graded_runs for x in stats)),
        ("summary.errored_runs", s.errored_runs, sum(x.errors for x in stats)),
        ("summary.quality.cases", s.quality_cases, len(quality)),
        ("summary.safety.cases", s.safety_cases, len(safety)),
        ("summary.safety.failing_cases", s.safety_failing_cases, sum(1 for x in safety if x.failures)),
        ("summary.safety.failing_runs", s.safety_failing_runs, sum(x.failures for x in safety)),
        ("summary.flaky_cases", s.flaky_cases, sum(1 for x in stats if x.flaky)),
        ("summary.tokens.input", s.input_tokens, sum(run.input_tokens for run in outputs)),
        ("summary.tokens.output", s.output_tokens, sum(run.output_tokens for run in outputs)),
        ("summary.cost.unpriced_runs", s.unpriced_runs, unpriced),
        (
            "summary.quality.fully_passing",
            s.quality_fully_passing,
            sum(1 for x in quality if x.graded_runs and not x.failures and not x.errors),
        ),
    ]
    for name, stored, real in expected:
        if stored != real:
            raise r.fail(name, f"is {stored} but the report has {real}")
    approximate: list[tuple[str, float | None, float | None]] = [
        ("summary.quality.mean_pass_rate", s.quality_mean_pass_rate, mean(rates) if rates else None),
        ("summary.cost.total", s.total_cost, total_cost),
        ("summary.latency_ms.p50", s.latency_p50, percentile(latencies, 50)),
        ("summary.latency_ms.p95", s.latency_p95, percentile(latencies, 95)),
        ("summary.latency_ms.max", s.latency_max, max(latencies, default=0.0)),
        ("summary.latency_ms.mean", s.latency_mean, mean(latencies)),
    ]
    for name, stored_value, real_value in approximate:
        if not _close(stored_value, real_value, 2e-3 if "latency" in name else 1e-5):
            raise r.fail(name, f"is {stored_value} but the runs give {real_value}")
