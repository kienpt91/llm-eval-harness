"""Reading a report or baseline file back: strict validation of untrusted JSON into typed documents."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from llm_eval_harness.domain.errors import Problem, ValidationError
from llm_eval_harness.domain.jsonio import JsonValue, canonical, kind_of
from llm_eval_harness.domain.report import REPORT_KIND, SCHEMA_VERSION, Summary
from llm_eval_harness.domain.results import CaseStat

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
    return RunDoc(
        r.i(d, "repeat", path),
        r.i(d, "attempts", path),
        r.req_s(d, "status", path),
        r.req_n(d, "score", path),
        r.strs(d, "failed_musts", path),
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
    runs = tuple(_run(r, v, f"{path}.runs[{i}]") for i, v in enumerate(r.lst(r.child(c, "runs", path), f"{path}.runs")))
    return CaseDoc(stat, r.strs(c, "tags", path), r.req_s(c, "input", path), r.req_s(c, "status", path), runs)


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
    return ReportDoc(
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
    )
