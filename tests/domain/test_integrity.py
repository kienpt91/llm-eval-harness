"""Verdict integrity: nothing a hostile output, dataset or file can do may hide a failure or average a safety case away."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import pytest

from llm_eval_harness.domain import exitcodes
from llm_eval_harness.domain.baseline import baseline_json, parse_baseline, snapshot_of_report
from llm_eval_harness.domain.compare import REGRESSION, CompareOptions, compare
from llm_eval_harness.domain.dataset import parse_dataset
from llm_eval_harness.domain.errors import DatasetError, ValidationError
from llm_eval_harness.domain.gate import GateDecision, GateOptions, decide
from llm_eval_harness.domain.jsonio import JsonValue, canonical, parse_json
from llm_eval_harness.domain.report import build_report
from llm_eval_harness.domain.reportdoc import parse_report
from llm_eval_harness.domain.results import CaseResult
from support.builders import js
from support.runs import case_result, report_doc, report_meta


def gate(
    results: list[CaseResult],
    *,
    baseline: list[CaseResult] | None = None,
    options: GateOptions | None = None,
    compare_options: CompareOptions | None = None,
    repeat: int = 1,
    **meta: object,
) -> GateDecision:
    doc = report_doc(results, repeat=repeat, **meta)
    comparison = None
    if baseline is not None:
        comparison = compare(
            snapshot_of_report(report_doc(baseline, repeat=repeat)),
            snapshot_of_report(doc),
            compare_options or CompareOptions(deterministic=True),
        )
    return decide(doc, comparison, options or GateOptions())


# --- safety cases that were not run ------------------------------------------------------------------------------


def test_a_run_that_left_safety_cases_out_cannot_pass_the_gate() -> None:
    decision = gate([case_result("q", ("passed",))], safety_not_run=("s-injection", "s-refund"))
    assert decision.exit_code == exitcodes.SAFETY == 4
    assert "safety coverage is incomplete" in decision.reasons[0] and "s-injection" in decision.reasons[0]


def test_the_explicit_option_accepts_a_partial_run_and_says_so() -> None:
    decision = gate(
        [case_result("q", ("passed",))], safety_not_run=("s1",), options=GateOptions(allow_partial_safety=True)
    )
    assert decision.exit_code == 0
    partial = report_doc([case_result("q")], safety_not_run=("s1",))
    assert partial.safety_not_run == ("s1",)


def test_a_safety_case_that_vanished_from_the_dataset_or_was_recategorised_is_a_safety_failure() -> None:
    baseline = [case_result("q"), case_result("s1", category="safety"), case_result("s2", category="safety")]
    removed = gate([case_result("q"), case_result("s1", category="safety")], baseline=baseline)
    assert removed.exit_code == 4 and "s2" in removed.reasons[0]
    recategorised = gate(
        [
            case_result("q"),
            case_result("s1", category="safety"),
            case_result("s2", category="quality", case_hash="other"),
        ],
        baseline=baseline,
    )
    assert recategorised.exit_code == 4 and "s2" in recategorised.reasons[0]
    assert (
        gate(
            [case_result("q"), case_result("s1", category="safety")],
            baseline=baseline,
            options=GateOptions(allow_partial_safety=True),
        ).exit_code
        == 0
    )


def test_a_failed_safety_case_and_missing_coverage_are_reported_together() -> None:
    decision = gate([case_result("s1", ("failed",), category="safety")], safety_not_run=("s2",))
    assert decision.exit_code == 4 and len(decision.reasons) == 2


def test_coverage_is_judged_before_infrastructure_errors() -> None:
    decision = gate([case_result("e", ("error",))], safety_not_run=("s1",))
    assert decision.exit_code == 4


# --- new and changed quality cases ---------------------------------------------------------------------------------


def test_a_failing_new_or_changed_quality_case_fails_the_gate_even_with_a_baseline() -> None:
    baseline = [case_result("a", ("passed",)), case_result("b", ("passed",), case_hash="v1")]
    fresh = gate(
        [case_result("a"), case_result("b", ("failed",), case_hash="v2"), case_result("c", ("failed",))],
        baseline=baseline,
    )
    assert (
        fresh.exit_code == 1
        and "2 new or changed quality case(s) fail" in fresh.reasons[0]
        and "b, c" in fresh.reasons[0]
    )
    fine = gate([case_result("a"), case_result("b", ("passed",), case_hash="v2"), case_result("c")], baseline=baseline)
    assert fine.exit_code == 0


# --- cost ------------------------------------------------------------------------------------------------------------


def test_a_cost_limit_cannot_be_dodged_by_not_reporting_usage() -> None:
    baseline = [case_result("a", cost="0.001")]
    hidden = [case_result("a", cost=None)]
    comparison = compare(
        snapshot_of_report(report_doc(baseline)),
        snapshot_of_report(report_doc(hidden)),
        CompareOptions(max_cost_increase_pct=10),
    )
    assert comparison.verdict == REGRESSION and "cost is unknown" in comparison.reasons[0]
    assert gate(hidden, baseline=baseline, compare_options=CompareOptions(max_cost_increase_pct=10)).exit_code == 1
    off = compare(snapshot_of_report(report_doc(baseline)), snapshot_of_report(report_doc(hidden)), CompareOptions())
    assert off.verdict == "ok"  # no limit set: nothing to enforce


# --- dataset rules for safety cases -------------------------------------------------------------------------------


def dataset_problems(record: JsonValue) -> list[str]:
    with pytest.raises(DatasetError) as error:
        parse_dataset([("line 1", record)], "d.jsonl")
    return [p.render() for p in error.value.problems]


def test_a_safety_case_cannot_lower_its_threshold_or_carry_a_grader_that_never_counts() -> None:
    base = {"id": "s", "category": "safety", "input": "x", "expected": {"forbidden": ["x"]}}
    assert any("threshold" in p and "safety" in p for p in dataset_problems(js({**base, "threshold": 0.5})))
    zero = {**base, "graders": [{"type": "latency_budget", "max_ms": 5, "weight": 0}]}
    assert any("graders[0]" in p and "weight 0" in p for p in dataset_problems(js(zero)))
    parse_dataset([("line 1", js({**base, "threshold": 1}))], "d.jsonl")
    parse_dataset(
        [("line 1", js({**base, "graders": [{"type": "latency_budget", "max_ms": 5, "weight": 0, "must": True}]}))],
        "d.jsonl",
    )
    parse_dataset(
        [
            (
                "line 1",
                js(
                    {
                        "id": "q",
                        "input": "x",
                        "expected": {"facts": ["x"]},
                        "threshold": 0.5,
                        "graders": [{"type": "regex", "pattern": "x", "weight": 0}],
                    }
                ),
            )
        ],
        "d.jsonl",
    )


# --- reports and baselines are validated, not trusted ------------------------------------------------------------------


def report_json(results: list[CaseResult] | None = None, **meta: object) -> dict[str, JsonValue]:
    return build_report(
        report_meta(**meta),
        results or [case_result("q", ("passed", "failed")), case_result("s", ("passed",), category="safety")],
    )


def edited(mutator: Callable[[Any], object]) -> dict[str, JsonValue]:
    document = parse_json(canonical(report_json()))
    mutator(document)
    assert isinstance(document, dict)
    return document


@pytest.mark.parametrize(
    ("mutator", "fragment"),
    [
        (lambda d: d["cases"][1].update(passes=5), "cases[1].passes: is 5 but its runs show 1"),
        (lambda d: d["cases"][1].update(passes=-4, graded_runs=1), "cases[1].passes"),
        (lambda d: d["cases"][1]["runs"][0].update(status="failed"), "cases[1]"),
        (lambda d: d["cases"][0].update(graded_runs=7), "cases[0].graded_runs: is 7 but its runs show 2"),
        (lambda d: d["cases"][0].update(status="passed"), "cases[0].status"),
        (lambda d: d["cases"][0]["runs"][0].update(status="maybe"), "status"),
        (lambda d: d["cases"][1].update(id="q"), "duplicate case id 'q'"),
        (lambda d: d["cases"][0].update(id="bad id\x1b[31m"), "cases[0].id"),
        (lambda d: d["cases"][0].update(category="safety "), "cases[0].category"),
        (lambda d: d["summary"].update(cases=99), "summary.cases: is 99 but the report has 2"),
        (
            lambda d: d["summary"]["safety"].update(failing_cases=5),
            "summary.safety.failing_cases: is 5 but the report has 0",
        ),
        (lambda d: d["summary"].update(errored_runs=3), "summary.errored_runs"),
        (lambda d: d["cases"][0]["runs"][0].update(score=7), "score"),
        (lambda d: d["dataset"].update(selected_cases=5), "dataset.selected_cases"),
        (lambda d: d["dataset"].update(safety_not_run=["bad id"]), "dataset.safety_not_run"),
        (lambda d: d["cases"][0].update(runs=d["cases"][0]["runs"] * 600), "more than 1000 runs"),
    ],
)
def test_a_report_that_contradicts_itself_is_refused_with_the_field_named(
    mutator: Callable[[Any], object], fragment: str
) -> None:
    document = edited(mutator)
    with pytest.raises(ValidationError) as error:
        parse_report(document, "report.json")
    assert fragment in str(error.value), str(error.value)


def test_a_consistent_report_still_reads() -> None:
    parse_report(report_json(), "report.json")
    parse_report(report_json(safety_not_run=("x1",)), "report.json")


def baseline_document() -> dict[str, JsonValue]:
    return copy.deepcopy(
        baseline_json(report_doc([case_result("q", ("passed", "failed")), case_result("s", category="safety")]))
    )


@pytest.mark.parametrize(
    ("mutator", "fragment"),
    [
        (lambda d: d["cases"]["q"].update(graded_runs=10**9, passes=5 * 10**8), "graded_runs"),
        (lambda d: d["cases"]["q"].update(passes=-1), "inconsistent run counts"),
        (lambda d: d["cases"]["q"].update(errors=-1), "inconsistent run counts"),
        (lambda d: d["cases"]["q"].update(errors=5000), "errors"),
        (lambda d: d["cases"]["q"].update(category="weird"), "category"),
        (lambda d: d["cases"].update({"bad id\x1b": d["cases"]["q"]}), "id"),
    ],
)
def test_a_baseline_with_absurd_counts_is_refused_before_any_statistics_run(mutator: object, fragment: str) -> None:
    document = baseline_document()
    mutator(document)  # type: ignore[operator]
    with pytest.raises(ValidationError) as error:
        parse_baseline(document, "baseline.json")
    assert fragment in str(error.value)


def test_a_valid_baseline_still_reads() -> None:
    assert set(parse_baseline(baseline_document(), "b.json").cases) == {"q", "s"}
