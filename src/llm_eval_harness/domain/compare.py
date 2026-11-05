"""Comparing a run with a baseline without mistaking noise for change.

Per case, the evidence is a pass rate over ``n`` graded runs on each side. A difference larger than the
tolerance is called a regression or an improvement only when the data supports it: Fisher's exact test must
reject "same pass rate" at ``alpha`` (or the provider is declared deterministic, in which case one run is a
fact). Otherwise the case is ``inconclusive`` and the report says how little data it has.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from llm_eval_harness.domain.baseline import Snapshot
from llm_eval_harness.domain.results import CaseStat
from llm_eval_harness.domain.stats import fisher_exact

UNCHANGED, REGRESSED, IMPROVED, INCONCLUSIVE, NEW, REMOVED, CHANGED = (
    "unchanged",
    "regressed",
    "improved",
    "inconclusive",
    "new",
    "removed",
    "changed",
)
OK, REGRESSION, INCONCLUSIVE_OVERALL = "ok", "regression", "inconclusive"
EPSILON = 1e-9


@dataclass(frozen=True, slots=True)
class CompareOptions:
    tolerance: float = 0.0
    alpha: float = 0.05
    deterministic: bool = False
    max_regressed_cases: int = 0
    max_cost_increase_pct: float | None = None
    max_p95_latency_increase_pct: float | None = None


@dataclass(frozen=True, slots=True)
class CaseComparison:
    id: str
    category: str
    verdict: str
    base: CaseStat | None
    current: CaseStat | None
    delta: float | None
    p_value: float | None
    reason: str


@dataclass(frozen=True, slots=True)
class Delta:
    baseline: float | None
    current: float | None

    @property
    def absolute(self) -> float | None:
        if self.baseline is None or self.current is None:
            return None
        return self.current - self.baseline

    @property
    def percent(self) -> float | None:
        if self.baseline is None or self.current is None or self.baseline <= 0:
            return None
        return (self.current - self.baseline) / self.baseline * 100


@dataclass(frozen=True, slots=True)
class Comparison:
    cases: tuple[CaseComparison, ...]
    accuracy: Delta
    cost_per_run: Delta
    tokens_per_run: Delta
    latency_p50: Delta
    latency_p95: Delta
    currency: str
    dataset_changed: bool
    verdict: str
    reasons: tuple[str, ...]
    notes: tuple[str, ...]

    def count(self, verdict: str) -> int:
        return sum(1 for c in self.cases if c.verdict == verdict)

    def of(self, verdict: str) -> list[CaseComparison]:
        return [c for c in self.cases if c.verdict == verdict]


def compare_case(
    base: CaseStat | None, current: CaseStat | None, options: CompareOptions, case_id: str
) -> CaseComparison:
    if base is None and current is not None:
        return CaseComparison(case_id, current.category, NEW, None, current, None, None, "not in the baseline")
    if current is None and base is not None:
        return CaseComparison(case_id, base.category, REMOVED, base, None, None, None, "not in this run")
    assert base is not None and current is not None
    category = current.category
    if base.case_hash != current.case_hash:
        return CaseComparison(
            case_id,
            category,
            CHANGED,
            base,
            current,
            None,
            None,
            "the case itself changed since the baseline; not comparable",
        )
    if base.graded_runs == 0 or current.graded_runs == 0:
        return CaseComparison(case_id, category, INCONCLUSIVE, base, current, None, None, "no graded runs on one side")
    assert base.pass_rate is not None and current.pass_rate is not None
    delta = current.pass_rate - base.pass_rate
    if abs(delta) <= options.tolerance + EPSILON:
        return CaseComparison(case_id, category, UNCHANGED, base, current, delta, None, "within tolerance")
    p = fisher_exact(current.passes, current.graded_runs, base.passes, base.graded_runs)
    summary = f"pass rate {base.pass_rate:.2f} ({base.passes}/{base.graded_runs}) to {current.pass_rate:.2f} ({current.passes}/{current.graded_runs})"
    if options.deterministic or p <= options.alpha:
        verdict = REGRESSED if delta < 0 else IMPROVED
        basis = "provider declared deterministic" if options.deterministic else f"p={p:.3f}"
        return CaseComparison(case_id, category, verdict, base, current, delta, p, f"{summary}; {basis}")
    return CaseComparison(
        case_id,
        category,
        INCONCLUSIVE,
        base,
        current,
        delta,
        p,
        f"{summary} cannot be told from chance (p={p:.3f} > alpha {options.alpha}); run with more --repeat",
    )


def compare(baseline: Snapshot, current: Snapshot, options: CompareOptions) -> Comparison:
    ids = sorted(set(baseline.cases) | set(current.cases))
    cases = tuple(compare_case(baseline.cases.get(i), current.cases.get(i), options, i) for i in ids)
    cost_comparable = (
        baseline.mean_cost_per_run is not None
        and current.mean_cost_per_run is not None
        and baseline.unpriced_runs == 0
        and current.unpriced_runs == 0
    )
    cost = Delta(baseline.mean_cost_per_run, current.mean_cost_per_run) if cost_comparable else Delta(None, None)
    result = Comparison(
        cases=cases,
        accuracy=Delta(baseline.quality_mean_pass_rate, current.quality_mean_pass_rate),
        cost_per_run=cost,
        tokens_per_run=Delta(baseline.mean_tokens_per_run, current.mean_tokens_per_run),
        latency_p50=Delta(baseline.latency_p50, current.latency_p50),
        latency_p95=Delta(baseline.latency_p95, current.latency_p95),
        currency=current.currency,
        dataset_changed=baseline.dataset_hash != current.dataset_hash,
        verdict=OK,
        reasons=(),
        notes=(),
    )
    return _judge(result, options, cost_comparable)


def _judge(result: Comparison, options: CompareOptions, cost_comparable: bool) -> Comparison:
    reasons: list[str] = []
    notes: list[str] = []
    regressed = result.of(REGRESSED)
    if len(regressed) > options.max_regressed_cases:
        names = ", ".join(c.id for c in regressed[:10]) + (" ..." if len(regressed) > 10 else "")
        reasons.append(f"{len(regressed)} case(s) regressed (allowed {options.max_regressed_cases}): {names}")
    pct = result.cost_per_run.percent
    if options.max_cost_increase_pct is not None:
        if not cost_comparable:
            notes.append("cost increase not checked: cost is unknown on one side")
        elif pct is not None and pct > options.max_cost_increase_pct + EPSILON:
            reasons.append(f"mean cost per run rose {pct:.1f}% (allowed {options.max_cost_increase_pct:g}%)")
    lat = result.latency_p95.percent
    if (
        options.max_p95_latency_increase_pct is not None
        and lat is not None
        and lat > options.max_p95_latency_increase_pct + EPSILON
    ):
        reasons.append(f"p95 latency rose {lat:.1f}% (allowed {options.max_p95_latency_increase_pct:g}%)")
    inconclusive = result.of(INCONCLUSIVE)
    if inconclusive:
        names = ", ".join(c.id for c in inconclusive[:10]) + (" ..." if len(inconclusive) > 10 else "")
        notes.append(
            f"{len(inconclusive)} case(s) differ by more than the tolerance but the data cannot support a verdict: {names}"
        )
    if result.count(CHANGED):
        notes.append(f"{result.count(CHANGED)} case(s) changed since the baseline and were not compared")
    if result.count(REMOVED):
        notes.append(f"{result.count(REMOVED)} baseline case(s) were not run (removed, filtered or sampled out)")
    if result.count(NEW):
        notes.append(f"{result.count(NEW)} case(s) are new since the baseline")
    verdict = REGRESSION if reasons else INCONCLUSIVE_OVERALL if inconclusive else OK
    return Comparison(
        result.cases,
        result.accuracy,
        result.cost_per_run,
        result.tokens_per_run,
        result.latency_p50,
        result.latency_p95,
        result.currency,
        result.dataset_changed,
        verdict,
        tuple(reasons),
        tuple(notes),
    )


def worst_regressions(comparisons: Sequence[CaseComparison]) -> list[CaseComparison]:
    """Regressed cases, biggest drop first."""
    return sorted((c for c in comparisons if c.verdict == REGRESSED), key=lambda c: (c.delta or 0.0, c.id))
