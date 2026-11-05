from __future__ import annotations

import json

import pytest

from llm_eval_harness.application.config import parse_config
from llm_eval_harness.application.redaction import Redactor
from llm_eval_harness.application.selection import Selection
from llm_eval_harness.application.usecases import (
    RunOutcome,
    RunRequest,
    compare_with_baseline,
    init_safety_suite,
    read_baseline,
    read_report,
    require,
    run_evaluation,
    write_baseline,
)
from llm_eval_harness.domain.compare import CompareOptions
from llm_eval_harness.domain.errors import DatasetError, UsageError
from llm_eval_harness.domain.pricing import EMPTY_TABLE
from llm_eval_harness.domain.safety_suite import starter_cases
from support.builders import js, output
from support.fakes import FakeClock, MemoryFiles, ScriptedProvider

LINES = [
    {"id": "q1", "input": "returns?", "expected": {"facts": ["30 days"]}},
    {"id": "q2", "input": "shipping?", "expected": {"facts": ["2 days"]}},
    {"id": "s1", "category": "safety", "input": "attack", "expected": {"forbidden_tools": ["issue_refund"]}},
]


def files(lines: list[dict[str, object]] | None = None) -> MemoryFiles:
    return MemoryFiles({"cases.jsonl": "".join(json.dumps(x) + "\n" for x in (lines or LINES))})


def request(**overrides: object) -> RunRequest:
    values: dict[str, object] = {
        "dataset_path": "cases.jsonl",
        "selection": Selection(),
        "repeat": 2,
        "concurrency": 2,
        "timeout_s": 5.0,
        "retries": 1,
        "seed": 3,
        "out_dir": "out",
        "version": "0.1.0",
    }
    values.update(overrides)
    return RunRequest(**values)  # type: ignore[arg-type]


PROVIDER = {"q1": [output("30 days")], "q2": [output("2 days")], "s1": [output("no")]}


def run(fs: MemoryFiles, provider: ScriptedProvider | None = None, **overrides: object) -> RunOutcome:
    return run_evaluation(
        fs,
        parse_config(js({}), "c"),
        provider or ScriptedProvider(PROVIDER),
        None,
        FakeClock(),
        request(**overrides),
        EMPTY_TABLE,
        Redactor(),
    )


def test_a_run_writes_the_json_and_markdown_reports_and_returns_what_it_read_back() -> None:
    fs = files()
    outcome = run(fs)
    assert outcome.written == ("out/report.json", "out/report.md")
    assert set(fs.files) == {"cases.jsonl", "out/report.json", "out/report.md"}
    assert read_report(fs, "out/report.json") == outcome.doc
    doc = outcome.doc
    assert doc.summary.cases == 3 and doc.summary.runs == 6 and doc.repeat == 2 and doc.seed == 3
    assert doc.dataset_source == "cases.jsonl" and doc.dataset_hash.startswith("sha256:") and doc.total_cases == 3
    assert doc.started_at == "2026-01-01T00:00:01Z" and doc.finished_at == "2026-01-01T00:00:02Z"
    assert doc.provider == {"type": "scripted"} and doc.redacted
    assert fs.files["out/report.md"].startswith("## LLM eval report")


def test_the_same_run_writes_the_same_report_bytes() -> None:
    a, b = files(), files()
    run(a)
    run(b)
    assert a.files["out/report.json"] == b.files["out/report.json"]


def test_the_report_is_pure_ascii_json() -> None:
    fs = files()
    run(fs, ScriptedProvider({**PROVIDER, "q1": [output("30 days \u202e \u0000 \ud800 \u00e9")]}))
    text = fs.files["out/report.json"]
    assert text.isascii() and json.loads(text)["cases"][0]["runs"][0]["output"]["text"].startswith("30 days")


def test_selection_filters_and_sampling_shape_the_report() -> None:
    fs = files()
    doc = run(fs, selection=Selection(categories=("quality",))).doc
    assert [c.stat.id for c in doc.cases] == ["q1", "q2"] and doc.total_cases == 3 and doc.summary.cases == 2
    doc = run(fs, selection=Selection(sample=1, seed=5), seed=5).doc
    assert doc.sample == 1 and doc.summary.cases == 1


def test_a_bad_dataset_stops_before_anything_is_run() -> None:
    provider = ScriptedProvider(PROVIDER)
    with pytest.raises(DatasetError):
        run(files([{"id": "x"}]), provider)
    assert not provider.requests


def test_baseline_round_trip_and_overwrite_protection() -> None:
    fs = files()
    doc = run(fs).doc
    path = write_baseline(fs, doc, "baseline.json", overwrite=False)
    assert path == "baseline.json"
    with pytest.raises(UsageError, match="already exists"):
        write_baseline(fs, doc, "baseline.json", overwrite=False)
    write_baseline(fs, doc, "baseline.json", overwrite=True)
    snapshot = read_baseline(fs, "baseline.json")
    assert set(snapshot.cases) == {"q1", "q2", "s1"} and snapshot.dataset_hash == doc.dataset_hash
    comparison = compare_with_baseline(snapshot, doc, CompareOptions())
    assert comparison.verdict == "ok" and not comparison.dataset_changed


def test_compare_detects_a_regression_against_a_committed_baseline() -> None:
    fs = files()
    good = run(fs).doc
    write_baseline(fs, good, "baseline.json", overwrite=True)
    worse = run(fs, ScriptedProvider({**PROVIDER, "q1": [output("no idea")]}), repeat=2).doc
    comparison = compare_with_baseline(read_baseline(fs, "baseline.json"), worse, CompareOptions(deterministic=True))
    assert comparison.verdict == "regression" and [c.id for c in comparison.of("regressed")] == ["q1"]


def test_init_safety_writes_a_valid_starter_dataset_once() -> None:
    fs = MemoryFiles()
    path, count = init_safety_suite(fs, "datasets/safety.jsonl", overwrite=False)
    assert path == "datasets/safety.jsonl" and count == len(starter_cases())
    with pytest.raises(UsageError, match="already exists"):
        init_safety_suite(fs, "datasets/safety.jsonl", overwrite=False)
    init_safety_suite(fs, "datasets/safety.jsonl", overwrite=True)
    from llm_eval_harness.application.dataset_loader import load_dataset

    assert len(load_dataset(fs, "datasets/safety.jsonl").cases) == count


def test_require() -> None:
    assert require("x", "dataset") == "x"
    with pytest.raises(UsageError, match="dataset is required"):
        require(None, "dataset")
