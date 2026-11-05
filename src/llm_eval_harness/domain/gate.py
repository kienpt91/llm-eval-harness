"""The gate: one decision and one exit code from a run (and optionally its comparison with a baseline).

Order of precedence, strongest first:

4  a safety case failed in any run. Never averaged away, never excused by a baseline.
3  a run could not be graded (provider or infrastructure error after retries): the answer is unknown.
1  quality regression: a case regressed against the baseline, a cost or latency limit was exceeded, the pass
   rate is below the minimum, or the comparison is inconclusive and ``on_inconclusive`` is ``fail``.
0  pass.

Usage errors (2) are raised before a run exists and never reach this module.
"""

from __future__ import annotations

from dataclasses import dataclass

from llm_eval_harness.domain import exitcodes
from llm_eval_harness.domain.compare import INCONCLUSIVE_OVERALL, REGRESSION, REMOVED, Comparison
from llm_eval_harness.domain.reportdoc import ReportDoc


@dataclass(frozen=True, slots=True)
class GateOptions:
    min_pass_rate: float | None = None
    on_inconclusive: str = "fail"
    fail_on_removed: bool = False


@dataclass(frozen=True, slots=True)
class GateDecision:
    exit_code: int
    reasons: tuple[str, ...]
    notes: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.exit_code == exitcodes.OK

    @property
    def verdict(self) -> str:
        return exitcodes.MEANING[self.exit_code]


def _names(ids: list[str]) -> str:
    return ", ".join(ids[:10]) + (f" and {len(ids) - 10} more" if len(ids) > 10 else "")


def decide(report: ReportDoc, comparison: Comparison | None, options: GateOptions) -> GateDecision:
    safety_failed = sorted(c.stat.id for c in report.cases if c.stat.category == "safety" and c.stat.failures > 0)
    if safety_failed:
        runs = report.summary.safety_failing_runs
        return GateDecision(
            exitcodes.SAFETY,
            (f"safety case(s) failed in {runs} run(s): {_names(safety_failed)}",),
            _notes(comparison),
        )
    errored = sorted(c.stat.id for c in report.cases if c.stat.errors > 0)
    if errored:
        kinds = sorted({r.error[0] for c in report.cases for r in c.runs if r.error is not None})
        return GateDecision(
            exitcodes.INFRASTRUCTURE,
            (f"{report.summary.errored_runs} run(s) could not be graded ({', '.join(kinds)}) in: {_names(errored)}",),
            _notes(comparison),
        )
    reasons: list[str] = []
    minimum = options.min_pass_rate if options.min_pass_rate is not None else (1.0 if comparison is None else None)
    rate = report.summary.quality_mean_pass_rate
    if minimum is not None and rate is not None and rate + 1e-9 < minimum:
        failing = sorted(c.stat.id for c in report.cases if c.stat.category == "quality" and c.stat.failures > 0)
        reasons.append(f"quality pass rate {rate:.3f} is below the minimum {minimum:g}; failing: {_names(failing)}")
    if comparison is not None:
        if comparison.verdict == REGRESSION:
            reasons.extend(comparison.reasons)
        elif comparison.verdict == INCONCLUSIVE_OVERALL and options.on_inconclusive == "fail":
            reasons.append(
                "the comparison with the baseline is inconclusive and --on-inconclusive is fail (add --repeat)"
            )
        if options.fail_on_removed:
            removed = sorted(c.id for c in comparison.cases if c.verdict == REMOVED)
            if removed:
                reasons.append(f"{len(removed)} baseline case(s) were not run: {_names(removed)}")
    if reasons:
        return GateDecision(exitcodes.QUALITY_REGRESSION, tuple(reasons), _notes(comparison))
    return GateDecision(exitcodes.OK, (), _notes(comparison))


def _notes(comparison: Comparison | None) -> tuple[str, ...]:
    return comparison.notes if comparison is not None else ()
