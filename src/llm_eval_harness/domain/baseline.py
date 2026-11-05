"""The baseline file: what is committed to git and compared against. Stable by construction.

It holds per-case pass counts and the aggregate quality, cost and latency, never outputs, inputs or timestamps,
so it is small, reviewable in a diff, and cannot leak model output. Keys are sorted and the file ends in a
newline; the same report always produces the same bytes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from llm_eval_harness.domain.identifiers import ID_PATTERN
from llm_eval_harness.domain.jsonio import JsonValue
from llm_eval_harness.domain.limits import MAX_REPEAT
from llm_eval_harness.domain.model import CATEGORIES
from llm_eval_harness.domain.report import SCHEMA_VERSION, TOOL_NAME
from llm_eval_harness.domain.reportdoc import BASELINE_KIND, Reader, ReportDoc
from llm_eval_harness.domain.results import CaseStat


@dataclass(frozen=True, slots=True)
class Snapshot:
    """The comparable essence of a report or a baseline."""

    dataset_hash: str
    repeat: int
    cases: dict[str, CaseStat]
    quality_mean_pass_rate: float | None
    runs: int
    mean_cost_per_run: float | None
    total_cost: float | None
    unpriced_runs: int
    mean_tokens_per_run: float | None
    latency_p50: float
    latency_p95: float
    currency: str


def snapshot_of_report(doc: ReportDoc) -> Snapshot:
    s = doc.summary
    outputs = sum(len([r for r in c.runs if r.text is not None]) for c in doc.cases)
    return Snapshot(
        dataset_hash=doc.dataset_hash,
        repeat=doc.repeat,
        cases={c.stat.id: c.stat for c in doc.cases},
        quality_mean_pass_rate=s.quality_mean_pass_rate,
        runs=s.runs,
        mean_cost_per_run=s.mean_cost_per_run,
        total_cost=s.total_cost,
        unpriced_runs=s.unpriced_runs,
        mean_tokens_per_run=(s.input_tokens + s.output_tokens) / outputs if outputs else None,
        latency_p50=s.latency_p50,
        latency_p95=s.latency_p95,
        currency=doc.currency,
    )


def baseline_json(doc: ReportDoc) -> dict[str, JsonValue]:
    snap = snapshot_of_report(doc)
    cases: dict[str, JsonValue] = {
        cid: {
            "category": s.category,
            "case_hash": s.case_hash,
            "graded_runs": s.graded_runs,
            "passes": s.passes,
            "errors": s.errors,
            "mean_score": round(s.mean_score, 6),
        }
        for cid, s in sorted(snap.cases.items())
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": BASELINE_KIND,
        "tool": {"name": TOOL_NAME, "version": doc.tool_version},
        "dataset": {"source": doc.dataset_source, "hash": doc.dataset_hash},
        "run": {"repeat": doc.repeat, "seed": doc.seed, "sample": doc.sample},
        "summary": {
            "runs": snap.runs,
            "quality_mean_pass_rate": snap.quality_mean_pass_rate,
            "currency": snap.currency,
            "total_cost": snap.total_cost,
            "mean_cost_per_run": snap.mean_cost_per_run,
            "unpriced_runs": snap.unpriced_runs,
            "mean_tokens_per_run": None if snap.mean_tokens_per_run is None else round(snap.mean_tokens_per_run, 3),
            "latency_p50_ms": snap.latency_p50,
            "latency_p95_ms": snap.latency_p95,
        },
        "cases": cases,
    }


def baseline_text(doc: ReportDoc) -> str:
    """The baseline file content: sorted keys, two-space indent, trailing newline."""
    return json.dumps(baseline_json(doc), indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"


def parse_baseline(value: JsonValue, source: str) -> Snapshot:
    r = Reader(source)
    root = r.header(value, BASELINE_KIND)
    dataset = r.obj(r.child(root, "dataset", ""), "dataset")
    run = r.obj(r.child(root, "run", ""), "run")
    summary = r.obj(r.child(root, "summary", ""), "summary")
    cases: dict[str, CaseStat] = {}
    for cid, raw in r.obj(r.child(root, "cases", ""), "cases").items():
        path = f"cases.{cid}"
        if not ID_PATTERN.fullmatch(cid) or len(cid) > 128:
            raise r.fail("cases", f"has an id that is not a valid case id: {cid[:40]!r}")
        c = r.obj(raw, path)
        stat = CaseStat(
            id=cid,
            category=r.req_s(c, "category", path),
            case_hash=r.req_s(c, "case_hash", path),
            graded_runs=r.i(c, "graded_runs", path),
            passes=r.i(c, "passes", path),
            errors=r.i(c, "errors", path),
            mean_score=r.req_n(c, "mean_score", path),
        )
        if stat.category not in CATEGORIES:
            raise r.fail(f"{path}.category", f"must be one of {', '.join(CATEGORIES)}")
        for name in ("graded_runs", "errors"):
            if getattr(stat, name) > MAX_REPEAT:
                raise r.fail(f"{path}.{name}", f"is more than {MAX_REPEAT}, the most runs one case can have")
        if min(stat.passes, stat.graded_runs, stat.errors) < 0 or stat.passes > stat.graded_runs:
            raise r.fail(path, "has inconsistent run counts")
        if stat.graded_runs + stat.errors > MAX_REPEAT:
            raise r.fail(path, f"has more than {MAX_REPEAT} runs")
        cases[cid] = stat
    return Snapshot(
        dataset_hash=r.req_s(dataset, "hash", "dataset"),
        repeat=r.i(run, "repeat", "run"),
        cases=cases,
        quality_mean_pass_rate=r.n(summary, "quality_mean_pass_rate", "summary", nullable=True),
        runs=r.i(summary, "runs", "summary"),
        mean_cost_per_run=r.n(summary, "mean_cost_per_run", "summary", nullable=True),
        total_cost=r.n(summary, "total_cost", "summary", nullable=True),
        unpriced_runs=r.i(summary, "unpriced_runs", "summary"),
        mean_tokens_per_run=r.n(summary, "mean_tokens_per_run", "summary", nullable=True),
        latency_p50=r.req_n(summary, "latency_p50_ms", "summary"),
        latency_p95=r.req_n(summary, "latency_p95_ms", "summary"),
        currency=r.req_s(summary, "currency", "summary"),
    )
