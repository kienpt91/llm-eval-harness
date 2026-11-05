"""The terminal summary. Plain text, no colour, no escape sequences: everything untrusted is sanitised."""

from __future__ import annotations

from llm_eval_harness.application.report.format import (
    millis,
    money,
    plural,
    rate,
    safe,
    short_hash,
    signed,
    signed_percent,
)
from llm_eval_harness.domain.compare import IMPROVED, INCONCLUSIVE, Comparison, worst_regressions
from llm_eval_harness.domain.gate import GateDecision
from llm_eval_harness.domain.reportdoc import CaseDoc, ReportDoc

MAX_LISTED = 15


def _provider(doc: ReportDoc) -> str:
    parts = [f"{k}={v}" for k, v in sorted(doc.provider.items()) if isinstance(v, (str, int, float, bool))]
    return safe(", ".join(parts), 200) if parts else "unknown"


def render_terminal(doc: ReportDoc, comparison: Comparison | None = None, decision: GateDecision | None = None) -> str:
    s = doc.summary
    lines = [
        "llm-eval report",
        f"  dataset   {safe(doc.dataset_source)}  {short_hash(doc.dataset_hash)}  ({s.cases} of {doc.total_cases} cases)",
        f"  provider  {_provider(doc)}",
        f"  run       repeat {doc.repeat}, concurrency {doc.concurrency}, retries {doc.retries}"
        + (f", seed {doc.seed}" if doc.seed is not None else "")
        + (f", sample {doc.sample}" if doc.sample is not None else "")
        + ("" if doc.redacted else ", redaction OFF"),
        "",
        f"  quality   mean pass rate {rate(s.quality_mean_pass_rate)} over {plural(s.quality_cases, 'case')}"
        f" ({s.quality_fully_passing} fully passing)",
        f"  safety    {plural(s.safety_cases, 'case')}, {s.safety_failing_cases} failing"
        + (f" ({plural(s.safety_failing_runs, 'failed run')})" if s.safety_failing_runs else ""),
        f"  runs      {s.graded_runs} graded, {s.errored_runs} errored, {plural(s.flaky_cases, 'flaky case')}",
        f"  tokens    in {s.input_tokens:,}  out {s.output_tokens:,}",
        f"  cost      {_cost(doc)}",
        f"  latency   p50 {millis(s.latency_p50)}  p95 {millis(s.latency_p95)}  max {millis(s.latency_max)}",
    ]
    problems = [c for c in doc.cases if c.stat.failures or c.stat.errors]
    if problems:
        lines += ["", "  cases that did not pass"]
        for case in sorted(problems, key=_order)[:MAX_LISTED]:
            lines += _case_lines(case)
        if len(problems) > MAX_LISTED:
            lines.append(f"    ... and {len(problems) - MAX_LISTED} more (see the JSON report)")
    flaky = [c for c in doc.cases if c.stat.flaky]
    if flaky:
        lines += ["", "  flaky cases (passed some runs, failed others)"]
        for case in flaky[:MAX_LISTED]:
            stat = case.stat
            lines.append(
                f"    {safe(stat.id)}  pass rate {rate(stat.pass_rate)} ({stat.passes}/{stat.graded_runs}), score variance {stat.score_variance:.3f}"
            )
    if comparison is not None:
        lines += ["", *render_comparison_text(comparison)]
    if decision is not None:
        lines += ["", *_decision(decision)]
    return "\n".join(lines) + "\n"


def _cost(doc: ReportDoc) -> str:
    s = doc.summary
    if s.total_cost is None:
        why = f" ({s.unpriced_runs} run(s) without usage or a price)" if s.unpriced_runs else " (no runs)"
        return "unknown" + why
    return f"{money(s.total_cost, doc.currency)} total, {money(s.mean_cost_per_run, doc.currency)} per run"


def _order(case: CaseDoc) -> tuple[int, str]:
    return (0 if case.stat.category == "safety" else 1, case.stat.id)


def _case_lines(case: CaseDoc) -> list[str]:
    stat = case.stat
    label = "ERROR" if stat.failures == 0 else "FAIL "
    head = f"    {label} {safe(stat.category, 20):<7} {safe(stat.id)}  {stat.passes}/{stat.graded_runs} runs passed"
    if stat.errors:
        head += f", {stat.errors} errored"
    lines = [head]
    shown = False
    for run in case.runs:
        if run.status == "failed" and not shown:
            for grade in run.grades:
                if not grade.passed:
                    lines.append(
                        f"          {'must ' if grade.must else ''}{safe(grade.name, 40)}: {safe(grade.detail, 160)}"
                    )
            shown = True
        elif run.status == "error" and run.error is not None:
            lines.append(f"          provider error ({safe(run.error[0], 20)}): {safe(run.error[1], 160)}")
            break
    return lines


def render_comparison_text(c: Comparison) -> list[str]:
    lines = [
        "  comparison with the baseline",
        f"    quality   {rate(c.accuracy.baseline)} -> {rate(c.accuracy.current)}  ({signed(c.accuracy.absolute)})",
        f"    cost/run  {money(c.cost_per_run.baseline, c.currency)} -> {money(c.cost_per_run.current, c.currency)}  ({signed_percent(c.cost_per_run)})",
        f"    p50       {millis(c.latency_p50.baseline or 0)} -> {millis(c.latency_p50.current or 0)}  ({signed_percent(c.latency_p50)})",
        f"    p95       {millis(c.latency_p95.baseline or 0)} -> {millis(c.latency_p95.current or 0)}  ({signed_percent(c.latency_p95)})",
        f"    verdict   {c.verdict.upper()}",
    ]
    if c.dataset_changed:
        lines.append("    note      the dataset hash differs from the baseline's")
    for case in worst_regressions(c.cases)[:MAX_LISTED]:
        lines.append(f"    regressed    {safe(case.id)}: {safe(case.reason, 200)}")
    for case in c.of(IMPROVED)[:MAX_LISTED]:
        lines.append(f"    improved     {safe(case.id)}: {safe(case.reason, 200)}")
    for case in c.of(INCONCLUSIVE)[:MAX_LISTED]:
        lines.append(f"    inconclusive {safe(case.id)}: {safe(case.reason, 200)}")
    lines.extend(f"    note      {safe(note, 300)}" for note in c.notes)
    return lines


def _decision(d: GateDecision) -> list[str]:
    lines = [f"  gate      {'PASS' if d.passed else 'FAIL'} (exit {d.exit_code}): {d.verdict}"]
    lines.extend(f"            - {safe(reason, 400)}" for reason in d.reasons)
    return lines
