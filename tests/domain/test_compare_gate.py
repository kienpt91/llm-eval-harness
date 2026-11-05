from __future__ import annotations

import pytest

from llm_eval_harness.domain import exitcodes
from llm_eval_harness.domain.baseline import Snapshot, snapshot_of_report
from llm_eval_harness.domain.compare import (
    CHANGED,
    IMPROVED,
    INCONCLUSIVE,
    INCONCLUSIVE_OVERALL,
    NEW,
    OK,
    REGRESSED,
    REGRESSION,
    REMOVED,
    UNCHANGED,
    CompareOptions,
    Delta,
    compare,
    compare_case,
    worst_regressions,
)
from llm_eval_harness.domain.gate import GateDecision, GateOptions, decide
from llm_eval_harness.domain.results import CaseResult, CaseStat
from support.prng import Prng
from support.runs import case_result, report_doc


def stat(passes: int, runs: int, *, case_hash: str = "h", errors: int = 0, category: str = "quality") -> CaseStat:
    return CaseStat("c", category, case_hash, runs, passes, errors, passes / runs if runs else 0.0)


def verdict(base: CaseStat | None, cur: CaseStat | None, **options: object) -> str:
    return compare_case(base, cur, CompareOptions(**options), "c").verdict  # type: ignore[arg-type]


def test_single_runs_cannot_support_a_verdict_unless_the_provider_is_deterministic() -> None:
    assert verdict(stat(1, 1), stat(0, 1)) == INCONCLUSIVE
    assert verdict(stat(0, 1), stat(1, 1)) == INCONCLUSIVE
    assert verdict(stat(1, 1), stat(0, 1), deterministic=True) == REGRESSED
    assert verdict(stat(0, 1), stat(1, 1), deterministic=True) == IMPROVED
    assert verdict(stat(1, 1), stat(1, 1)) == UNCHANGED


def test_enough_repeats_make_a_drop_significant() -> None:
    assert verdict(stat(10, 10), stat(2, 10)) == REGRESSED
    assert verdict(stat(2, 10), stat(10, 10)) == IMPROVED
    assert verdict(stat(5, 5), stat(0, 5)) == REGRESSED
    assert verdict(stat(10, 10), stat(8, 10)) == INCONCLUSIVE  # p ~ 0.47: could be chance


def test_noise_within_tolerance_is_not_flagged_at_all() -> None:
    assert verdict(stat(10, 10), stat(9, 10), tolerance=0.1) == UNCHANGED
    assert verdict(stat(10, 10), stat(8, 10), tolerance=0.1) == INCONCLUSIVE
    assert verdict(stat(10, 10), stat(1, 10), tolerance=0.1) == REGRESSED
    assert verdict(stat(3, 10), stat(5, 10), tolerance=0.25) == UNCHANGED


def test_alpha_is_configurable() -> None:
    assert verdict(stat(8, 10), stat(2, 10), alpha=0.001) == INCONCLUSIVE
    assert verdict(stat(8, 10), stat(2, 10), alpha=0.05) == REGRESSED


def test_case_set_differences_and_changed_cases() -> None:
    assert verdict(None, stat(1, 1)) == NEW
    assert verdict(stat(1, 1), None) == REMOVED
    assert verdict(stat(1, 1, case_hash="old"), stat(0, 1, case_hash="new"), deterministic=True) == CHANGED
    assert verdict(stat(0, 0), stat(1, 1)) == INCONCLUSIVE
    assert verdict(stat(1, 1), stat(0, 0, errors=2)) == INCONCLUSIVE


def test_reasons_explain_the_evidence() -> None:
    c = compare_case(stat(10, 10), stat(2, 10), CompareOptions(), "c")
    assert "pass rate 1.00 (10/10) to 0.20 (2/10); p=0.0" in c.reason and c.delta == pytest.approx(-0.8)
    flat = compare_case(stat(1, 1), stat(0, 1), CompareOptions(), "c")
    assert "cannot be told from chance" in flat.reason and "--repeat" in flat.reason
    assert (
        "declared deterministic" in compare_case(stat(1, 1), stat(0, 1), CompareOptions(deterministic=True), "c").reason
    )


def test_property_comparing_a_run_with_itself_never_flags_anything() -> None:
    for seed in range(200):
        rng = Prng(seed)
        n = rng.integer(1, 12)
        s = stat(rng.integer(0, n), n)
        for deterministic in (False, True):
            assert verdict(s, s, deterministic=deterministic) == UNCHANGED, seed


def test_property_a_flip_is_never_called_a_regression_without_evidence() -> None:
    for seed in range(300):
        rng = Prng(seed)
        n1, n2 = rng.integer(1, 4), rng.integer(1, 4)
        base, cur = stat(n1, n1), stat(rng.integer(0, n2), n2)
        outcome = verdict(base, cur)
        if outcome == REGRESSED:
            from llm_eval_harness.domain.stats import fisher_exact

            assert fisher_exact(cur.passes, cur.graded_runs, base.passes, base.graded_runs) <= 0.05, seed


def snap(results: list[CaseResult], **kw: object) -> Snapshot:
    return snapshot_of_report(report_doc(results, **kw))


def test_overall_verdicts() -> None:
    base = snap([case_result("a", ("passed",) * 5), case_result("b", ("passed",) * 5)], repeat=5)
    same = compare(
        base, snap([case_result("a", ("passed",) * 5), case_result("b", ("passed",) * 5)], repeat=5), CompareOptions()
    )
    assert same.verdict == OK and not same.reasons and same.accuracy.absolute == 0.0
    worse = compare(
        base, snap([case_result("a", ("failed",) * 5), case_result("b", ("passed",) * 5)], repeat=5), CompareOptions()
    )
    assert worse.verdict == REGRESSION and "a" in worse.reasons[0] and worse.accuracy.absolute == pytest.approx(-0.5)
    assert [c.id for c in worst_regressions(worse.cases)] == ["a"]
    allowed = compare(
        base,
        snap([case_result("a", ("failed",) * 5), case_result("b", ("passed",) * 5)], repeat=5),
        CompareOptions(max_regressed_cases=1),
    )
    assert allowed.verdict == OK


def test_inconclusive_overall_and_notes() -> None:
    base = snap([case_result("a", ("passed",)), case_result("gone", ("passed",))])
    cur = snap([case_result("a", ("failed",)), case_result("fresh", ("passed",))])
    result = compare(base, cur, CompareOptions())
    assert result.verdict == INCONCLUSIVE_OVERALL
    joined = " ".join(result.notes)
    assert "cannot support a verdict: a" in joined and "were not run" in joined and "are new" in joined
    assert result.count(NEW) == 1 and result.count(REMOVED) == 1 and [c.id for c in result.of(INCONCLUSIVE)] == ["a"]


def test_changed_cases_are_noted_not_compared() -> None:
    base = snap([case_result("a", ("passed",), case_hash="v1")])
    cur = snap([case_result("a", ("failed",), case_hash="v2")])
    result = compare(base, cur, CompareOptions(deterministic=True))
    assert (
        result.verdict == OK
        and result.count(CHANGED) == 1
        and any("changed since the baseline" in n for n in result.notes)
    )


def test_dataset_change_is_reported() -> None:
    base = snap([case_result("a", ("passed",))])
    other = snapshot_of_report(report_doc([case_result("a", ("passed",))], dataset_hash="sha256:other"))
    assert compare(base, other, CompareOptions()).dataset_changed
    assert not compare(base, base, CompareOptions()).dataset_changed


def test_cost_and_latency_limits() -> None:
    base = snap([case_result("a", ("passed",), cost="0.001", latency_ms=10)])
    dearer = snap([case_result("a", ("passed",), cost="0.0015", latency_ms=25)])
    off = compare(base, dearer, CompareOptions())
    assert (
        off.verdict == OK
        and off.cost_per_run.percent == pytest.approx(50.0)
        and off.latency_p95.percent == pytest.approx(150.0)
    )
    cost = compare(base, dearer, CompareOptions(max_cost_increase_pct=20))
    assert cost.verdict == REGRESSION and "rose 50.0%" in cost.reasons[0]
    assert compare(base, dearer, CompareOptions(max_cost_increase_pct=50)).verdict == OK
    lat = compare(base, dearer, CompareOptions(max_p95_latency_increase_pct=100))
    assert lat.verdict == REGRESSION and "p95 latency rose 150.0%" in lat.reasons[0]


def test_unknown_cost_is_never_compared_and_cannot_dodge_a_cost_limit() -> None:
    base = snap([case_result("a", ("passed",), cost="0.001")])
    unpriced = snap([case_result("a", ("passed",), cost=None)])
    off = compare(base, unpriced, CompareOptions())
    assert off.cost_per_run == Delta(None, None) and off.verdict == OK
    limited = compare(base, unpriced, CompareOptions(max_cost_increase_pct=1))
    assert limited.cost_per_run == Delta(None, None) and limited.verdict == REGRESSION
    assert "cost is unknown" in limited.reasons[0]


def test_delta_arithmetic() -> None:
    assert Delta(2.0, 3.0).absolute == 1.0 and Delta(2.0, 3.0).percent == 50.0
    assert Delta(None, 3.0).absolute is None and Delta(0.0, 3.0).percent is None and Delta(2.0, None).percent is None


# --- the gate -------------------------------------------------------------------------------------------------


def gate(
    results: list[CaseResult],
    *,
    baseline: list[CaseResult] | None = None,
    options: GateOptions | None = None,
    compare_options: CompareOptions | None = None,
    repeat: int = 1,
) -> GateDecision:
    options = options or GateOptions()
    compare_options = compare_options or CompareOptions()
    doc = report_doc(results, repeat=repeat)
    comparison = (
        None
        if baseline is None
        else compare(snapshot_of_report(report_doc(baseline, repeat=repeat)), snapshot_of_report(doc), compare_options)
    )
    return decide(doc, comparison, options)


GOOD = [case_result("q", ("passed",)), case_result("s", ("passed",), category="safety")]


def test_clean_run_passes_with_exit_0() -> None:
    d = gate(GOOD)
    assert d.exit_code == exitcodes.OK == 0 and d.passed and not d.reasons and d.verdict == "pass"


def test_a_failing_safety_case_is_fatal_whatever_else_is_true() -> None:
    results = [
        *[case_result(f"q{i}", ("passed",) * 10) for i in range(50)],
        case_result("s", ("passed",) * 9 + ("failed",), category="safety"),
    ]
    d = gate(results)
    assert d.exit_code == exitcodes.SAFETY == 4 and "s" in d.reasons[0] and "1 run(s)" in d.reasons[0]
    # not excused by a baseline that failed too, nor by options
    d2 = gate(results, baseline=results, options=GateOptions(min_pass_rate=0.0, on_inconclusive="pass"))
    assert d2.exit_code == 4


def test_safety_outranks_infrastructure_errors_which_outrank_quality() -> None:
    both = [
        case_result("q", ("failed",)),
        case_result("e", ("error",)),
        case_result("s", ("failed",), category="safety"),
    ]
    assert gate(both).exit_code == 4
    assert gate([case_result("q", ("failed",)), case_result("e", ("error",))]).exit_code == 3
    assert gate([case_result("q", ("failed",))]).exit_code == 1


def test_infrastructure_errors_name_the_kinds_and_cases() -> None:
    d = gate(
        [case_result("a", ("passed", "error"), error_kind="timeout"), case_result("b", ("error",), error_kind="spawn")]
    )
    assert d.exit_code == 3 and "spawn, timeout" in d.reasons[0] and "a, b" in d.reasons[0]


def test_a_safety_case_that_errored_is_an_infrastructure_failure_not_a_pass() -> None:
    assert gate([case_result("s", ("error",), category="safety")]).exit_code == 3


def test_without_a_baseline_every_quality_case_must_pass_by_default() -> None:
    d = gate([case_result("a", ("passed",)), case_result("b", ("failed",))])
    assert d.exit_code == 1 and "0.500 is below the minimum 1" in d.reasons[0] and "failing: b" in d.reasons[0]
    assert (
        gate(
            [case_result("a", ("passed",)), case_result("b", ("failed",))], options=GateOptions(min_pass_rate=0.5)
        ).exit_code
        == 0
    )
    assert (
        gate([case_result("a", ("passed",) * 9 + ("failed",))], options=GateOptions(min_pass_rate=0.9)).exit_code == 0
    )


def test_with_a_baseline_only_regressions_fail_by_default() -> None:
    base = [case_result("a", ("passed",) * 5), case_result("b", ("failed",) * 5)]
    same = gate(base, baseline=base, repeat=5)
    assert same.exit_code == 0  # b was already failing: not a regression
    worse = gate([case_result("a", ("failed",) * 5), case_result("b", ("failed",) * 5)], baseline=base, repeat=5)
    assert worse.exit_code == 1 and "regressed" in worse.reasons[0]


def test_inconclusive_comparison_follows_the_configured_policy() -> None:
    base, cur = [case_result("a", ("passed",))], [case_result("a", ("failed",))]
    strict = gate(cur, baseline=base)
    assert strict.exit_code == 1 and "inconclusive" in strict.reasons[0]
    lenient = gate(cur, baseline=base, options=GateOptions(on_inconclusive="pass"))
    assert lenient.exit_code == 0 and any("cannot support a verdict" in n for n in lenient.notes)
    deterministic = gate(cur, baseline=base, compare_options=CompareOptions(deterministic=True))
    assert deterministic.exit_code == 1 and "regressed" in deterministic.reasons[0]


def test_removed_cases_fail_only_when_asked() -> None:
    base = [case_result("a", ("passed",)), case_result("b", ("passed",))]
    cur = [case_result("a", ("passed",))]
    assert gate(cur, baseline=base).exit_code == 0
    d = gate(cur, baseline=base, options=GateOptions(fail_on_removed=True))
    assert d.exit_code == 1 and "were not run: b" in d.reasons[0]


def test_cost_regression_fails_the_gate() -> None:
    base = [case_result("a", ("passed",), cost="0.001")]
    cur = [case_result("a", ("passed",), cost="0.01")]
    d = gate(cur, baseline=base, compare_options=CompareOptions(max_cost_increase_pct=50))
    assert d.exit_code == 1 and "cost" in d.reasons[0]


def test_no_quality_cases_is_not_a_failure() -> None:
    assert gate([case_result("s", ("passed",), category="safety")]).exit_code == 0


def test_property_any_failed_safety_run_gives_exit_4() -> None:
    for seed in range(200):
        rng = Prng(seed)
        statuses = tuple(rng.choice(["passed", "failed"]) for _ in range(rng.integer(1, 6)))
        results = [case_result("q", ("passed",)), case_result("s", statuses, category="safety")]
        expected = 4 if "failed" in statuses else 0
        assert gate(results, options=GateOptions(min_pass_rate=0.0)).exit_code == expected, seed
