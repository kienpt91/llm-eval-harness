from __future__ import annotations

import json

import pytest

from llm_eval_harness.domain.baseline import baseline_json, baseline_text, parse_baseline, snapshot_of_report
from llm_eval_harness.domain.errors import ValidationError
from llm_eval_harness.domain.jsonio import JsonValue, canonical, parse_json
from llm_eval_harness.domain.report import SCHEMA_VERSION, build_report, summarize
from llm_eval_harness.domain.reportdoc import parse_report
from llm_eval_harness.domain.results import CaseResult, stat_of
from support.builders import js
from support.prng import Prng
from support.runs import case_result, report_doc, report_meta, run_record


def results() -> list[CaseResult]:
    return [
        case_result("q2", ("passed", "failed", "passed"), latency_ms=30.0, cost="0.002"),
        case_result("q1", ("passed",), latency_ms=10.0, cost="0.001"),
        case_result("s1", ("passed", "passed"), category="safety", latency_ms=20.0, cost="0.001"),
        case_result("e1", ("passed", "error"), latency_ms=40.0, cost="0.001"),
    ]


def test_case_statistics() -> None:
    stat = stat_of(case_result("q", ("passed", "failed", "passed", "error")))
    assert (stat.graded_runs, stat.passes, stat.failures, stat.errors) == (3, 2, 1, 1)
    assert stat.pass_rate == pytest.approx(2 / 3) and stat.flaky and stat.status == "failed"
    assert stat.mean_score == pytest.approx(2 / 3) and stat.score_variance == pytest.approx(1 / 3)
    assert stat_of(case_result("q", ("passed", "error"))).status == "error"
    assert stat_of(case_result("q", ("passed", "passed"))).status == "passed"
    none = stat_of(case_result("q", ("error",)))
    assert none.pass_rate is None and not none.flaky and none.graded_runs == 0


def test_summary_numbers() -> None:
    s = summarize(results())
    assert (s.cases, s.runs, s.graded_runs, s.errored_runs) == (4, 8, 7, 1)
    assert s.quality_cases == 3 and s.quality_fully_passing == 1  # q1 only: q2 failed once, e1 errored
    assert s.quality_mean_pass_rate == pytest.approx((2 / 3 + 1 + 1) / 3, abs=1e-6)
    assert s.safety_cases == 1 and s.safety_failing_cases == 0 and s.flaky_cases == 1
    assert s.input_tokens == 70 and s.output_tokens == 35  # 7 runs with an output
    assert s.total_cost == pytest.approx(0.010, abs=1e-9) and s.mean_cost_per_run == pytest.approx(0.010 / 7, abs=1e-8)
    assert s.latency_max == 40.0 and s.latency_p50 == 30.0 and s.latency_p95 == 40.0


def test_summary_of_nothing_is_well_defined() -> None:
    s = summarize([])
    assert s.cases == 0 and s.quality_mean_pass_rate is None and s.total_cost is None and s.latency_p95 == 0.0


def test_unpriced_runs_make_the_total_unknown_instead_of_wrong() -> None:
    s = summarize([case_result("a", ("passed", "passed"), cost=None), case_result("b", ("passed",), cost="1")])
    assert s.total_cost is None and s.unpriced_runs == 2


def test_report_json_shape_and_ordering() -> None:
    doc = build_report(report_meta(), results())
    assert doc["schema_version"] == SCHEMA_VERSION == 1 and doc["kind"] == "llm-eval-report"
    cases = doc["cases"]
    assert isinstance(cases, list)
    assert [c["id"] for c in cases] == ["e1", "q1", "q2", "s1"]  # type: ignore[index,call-overload]
    shuffled = build_report(report_meta(), list(reversed(results())))
    assert canonical(doc) == canonical(shuffled)  # ordering never depends on completion order
    json.dumps(doc, allow_nan=False)


def test_report_round_trips_through_the_reader() -> None:
    doc = report_doc(results(), repeat=3)
    assert doc.repeat == 3 and doc.dataset_hash == "sha256:abc" and doc.redacted
    assert [c.stat.id for c in doc.cases] == ["e1", "q1", "q2", "s1"]
    q2 = doc.cases[2]
    assert q2.stat.passes == 2 and q2.runs[1].status == "failed" and q2.runs[1].grades[0].name == "facts"
    assert q2.runs[0].text == "answer" and q2.runs[0].input_tokens == 10 and q2.runs[0].cost == 0.002
    assert doc.cases[0].runs[1].error == ("timeout", "the provider failed")
    assert doc.summary == summarize(results())


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [
        (lambda d: d.update(kind="llm-eval-baseline"), "must be 'llm-eval-report'"),
        (lambda d: d.update(schema_version=2), "schema_version: is 2; this tool reads version 1"),
        (lambda d: d.pop("cases"), "cases: is missing"),
        (lambda d: d.update(cases={}), "cases: must be an array (got object)"),
        (lambda d: d["cases"][0].update(passes="x"), "cases[0].passes: must be a whole number (got string)"),  # type: ignore[index]
        (lambda d: d["cases"][0]["runs"][0].pop("grades"), "cases[0].runs[0].grades: is missing"),  # type: ignore[index]
        (lambda d: d["summary"].pop("cost"), "summary.cost: is missing"),  # type: ignore[index]
        (lambda d: d["dataset"].update(hash=5), "dataset.hash: must be a string (got number)"),  # type: ignore[index]
    ],
)
def test_reader_names_the_broken_field(mutate: object, fragment: str) -> None:
    doc = build_report(report_meta(), results())
    mutate(doc)  # type: ignore[operator]
    with pytest.raises(ValidationError) as error:
        parse_report(doc, "r.json")
    assert fragment in str(error.value)


def test_reader_refuses_non_objects() -> None:
    with pytest.raises(ValidationError):
        parse_report(js([]), "r.json")


def test_baseline_is_stable_small_and_free_of_outputs() -> None:
    doc = report_doc(results())
    text = baseline_text(doc)
    assert text.endswith("}\n") and text == baseline_text(report_doc(list(reversed(results()))))
    assert "answer" not in text and "input" not in text and "started_at" not in text
    parsed = json.loads(text)
    assert list(parsed["cases"]) == ["e1", "q1", "q2", "s1"]
    assert json.dumps(parsed, indent=2, sort_keys=True, ensure_ascii=False) + "\n" == text


def test_baseline_round_trips_to_the_same_snapshot_as_the_report() -> None:
    doc = report_doc(results(), repeat=3)
    from_report = snapshot_of_report(doc)
    from_baseline = parse_baseline(parse_json(baseline_text(doc)), "baseline.json")
    assert from_report.dataset_hash == from_baseline.dataset_hash and from_report.repeat == from_baseline.repeat == 3
    assert from_report.quality_mean_pass_rate == from_baseline.quality_mean_pass_rate
    for cid, stat in from_report.cases.items():
        other = from_baseline.cases[cid]
        assert (stat.category, stat.case_hash, stat.graded_runs, stat.passes, stat.errors) == (
            other.category,
            other.case_hash,
            other.graded_runs,
            other.passes,
            other.errors,
        )
    assert from_baseline.latency_p95 == from_report.latency_p95 and from_baseline.currency == "USD"
    assert from_baseline.mean_tokens_per_run == pytest.approx(from_report.mean_tokens_per_run, abs=1e-3)  # type: ignore[arg-type]


def test_baseline_reader_rejects_nonsense() -> None:
    good = baseline_json(report_doc(results()))

    def broken(mutator: object) -> JsonValue:
        copy = parse_json(canonical(good))
        mutator(copy)  # type: ignore[operator]
        return copy

    with pytest.raises(ValidationError, match=r"must be .llm-eval-baseline."):
        parse_baseline(js(build_report(report_meta(), results())), "b.json")
    with pytest.raises(ValidationError, match="inconsistent run counts"):
        parse_baseline(broken(lambda d: d["cases"]["q1"].update(passes=9)), "b.json")  # type: ignore[index]
    with pytest.raises(ValidationError, match=r"cases\.q1\.category: is missing"):
        parse_baseline(broken(lambda d: d["cases"]["q1"].pop("category")), "b.json")  # type: ignore[index]


def test_property_baseline_bytes_depend_only_on_the_results() -> None:
    for seed in range(50):
        rng = Prng(seed)
        rs = [
            case_result(f"c{i}", tuple(rng.choice(["passed", "failed", "error"]) for _ in range(rng.integer(1, 4))))
            for i in range(rng.integer(1, 6))
        ]
        assert baseline_text(report_doc(rs)) == baseline_text(report_doc(rng.shuffled(rs))), seed


def test_run_record_helper_builds_consistent_failures() -> None:
    r = run_record("failed", failed_musts=("forbidden",))
    assert r.status == "failed" and r.failed_musts == ("forbidden",) and r.grades[0].must
