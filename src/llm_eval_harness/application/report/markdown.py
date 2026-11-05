"""The Markdown report for a pull request: what regressed, what improved, the worst cases, cost and latency deltas."""

from __future__ import annotations

from llm_eval_harness.application.report.format import millis, money, rate, safe, signed, signed_percent
from llm_eval_harness.domain.compare import IMPROVED, INCONCLUSIVE, Comparison, worst_regressions
from llm_eval_harness.domain.gate import GateDecision
from llm_eval_harness.domain.reportdoc import CaseDoc, ReportDoc, RunDoc
from llm_eval_harness.domain.text import fence_for, sanitize

WORST_CASES = 5
LISTED = 15
EXCERPT = 700


def _cell(value: str, max_chars: int = 160) -> str:
    return safe(value, max_chars).replace("|", "\\|").replace("`", "'")


def _code(value: str) -> str:
    return f"`{_cell(value, 100)}`"


def _block(value: str, max_chars: int = EXCERPT) -> list[str]:
    clean = sanitize(value, max_chars=max_chars)
    fence = fence_for(clean)
    return [fence, clean, fence]


def render_markdown(doc: ReportDoc, comparison: Comparison | None = None, decision: GateDecision | None = None) -> str:
    out: list[str] = []
    if decision is not None:
        status = "PASSED" if decision.passed else "FAILED"
        out.append(f"## LLM eval: {status} (exit {decision.exit_code})")
        out.append("")
        out.append(_cell(decision.verdict, 300))
        out.extend(f"- {_cell(reason, 400)}" for reason in decision.reasons)
    elif comparison is not None:
        out.append(f"## LLM eval: {comparison.verdict}")
    else:
        out.append("## LLM eval report")
    out.append("")
    out += _summary_table(doc, comparison)
    safety = [c for c in doc.cases if c.stat.category == "safety" and c.stat.failures]
    if safety:
        out += ["", "### Safety failures (always fatal)", ""]
        for case in safety:
            out += _case_section(case)
    if comparison is not None:
        out += _comparison_sections(comparison)
    failing = sorted(
        (c for c in doc.cases if c.stat.category == "quality" and (c.stat.failures or c.stat.errors)),
        key=lambda c: (c.stat.pass_rate if c.stat.pass_rate is not None else -1.0, c.stat.id),
    )
    if failing:
        out += [
            "",
            f"### Worst quality cases ({min(len(failing), WORST_CASES)} of {len(failing)} not fully passing)",
            "",
        ]
        for case in failing[:WORST_CASES]:
            out += _case_section(case)
    flaky = [c for c in doc.cases if c.stat.flaky]
    if flaky:
        out += ["", "### Flaky cases", "", "| case | pass rate | runs | score variance |", "| --- | --- | --- | --- |"]
        out.extend(
            f"| {_code(c.stat.id)} | {rate(c.stat.pass_rate)} | {c.stat.passes}/{c.stat.graded_runs} | {c.stat.score_variance:.3f} |"
            for c in flaky[:LISTED]
        )
    out += [
        "",
        "<sub>"
        + f"dataset {_cell(doc.dataset_source)} {_cell(doc.dataset_hash, 24)}; {doc.summary.cases} of {doc.total_cases} cases; "
        + f"repeat {doc.repeat}, concurrency {doc.concurrency}, retries {doc.retries}"
        + (f", seed {doc.seed}" if doc.seed is not None else "")
        + f"; harness {_cell(doc.tool_version, 20)}</sub>",
    ]
    return "\n".join(out) + "\n"


def _summary_table(doc: ReportDoc, c: Comparison | None) -> list[str]:
    s = doc.summary
    cost = money(s.mean_cost_per_run, doc.currency)
    rows: list[tuple[str, str, str, str]] = [
        (
            "Quality (mean pass rate)",
            rate(c.accuracy.baseline) if c else "",
            rate(s.quality_mean_pass_rate),
            signed(c.accuracy.absolute) if c else "",
        ),
        ("Safety cases failing", "", f"{s.safety_failing_cases} of {s.safety_cases}", ""),
        ("Errored runs", "", str(s.errored_runs), ""),
        (
            "Cost per run",
            money(c.cost_per_run.baseline, c.currency) if c else "",
            cost,
            signed_percent(c.cost_per_run) if c else "",
        ),
        (
            "Tokens per run",
            f"{c.tokens_per_run.baseline:.0f}" if c and c.tokens_per_run.baseline is not None else "",
            f"{(s.input_tokens + s.output_tokens) / max(1, s.graded_runs):.0f}",
            signed_percent(c.tokens_per_run) if c else "",
        ),
        (
            "Latency p50",
            millis(c.latency_p50.baseline or 0) if c else "",
            millis(s.latency_p50),
            signed_percent(c.latency_p50) if c else "",
        ),
        (
            "Latency p95",
            millis(c.latency_p95.baseline or 0) if c else "",
            millis(s.latency_p95),
            signed_percent(c.latency_p95) if c else "",
        ),
    ]
    lines = ["| metric | baseline | current | change |", "| --- | --- | --- | --- |"]
    lines.extend(f"| {name} | {base or '-'} | {now} | {change or '-'} |" for name, base, now, change in rows)
    return lines


def _comparison_sections(c: Comparison) -> list[str]:
    out: list[str] = []
    for title, cases in (
        ("Regressed", worst_regressions(c.cases)),
        ("Improved", c.of(IMPROVED)),
        ("Inconclusive (not enough data for a verdict)", c.of(INCONCLUSIVE)),
    ):
        if not cases:
            continue
        out += ["", f"### {title}", "", "| case | category | evidence |", "| --- | --- | --- |"]
        out.extend(f"| {_code(x.id)} | {_cell(x.category)} | {_cell(x.reason, 240)} |" for x in cases[:LISTED])
        if len(cases) > LISTED:
            out.append(f"| ... | | {len(cases) - LISTED} more |")
    if c.notes or c.dataset_changed:
        out += [""]
        out.extend(f"> {_cell(n, 300)}" for n in c.notes)
        if c.dataset_changed:
            out.append("> The dataset hash differs from the baseline's.")
    return out


def _case_section(case: CaseDoc) -> list[str]:
    stat = case.stat
    out = [f"#### {_code(stat.id)} ({_cell(stat.category)}): {stat.passes}/{stat.graded_runs} runs passed", ""]
    run = _worst_run(case)
    out.append("Input:")
    out += _block(case.input)
    if run is not None:
        if run.error is not None:
            out.append(f"Provider error ({_cell(run.error[0], 20)}): {_cell(run.error[1], 300)}")
        else:
            out.append("Output:")
            out += _block(run.text or "")
            if run.tool_calls:
                out.append("Tool calls: " + ", ".join(_code(f"{n}({a})") for n, a in run.tool_calls[:8]))
        failed = [g for g in run.grades if not g.passed]
        if failed:
            out.append("")
            out.extend(f"- {'**must** ' if g.must else ''}{_code(g.name)}: {_cell(g.detail, 300)}" for g in failed[:8])
    out.append("")
    return out


def _worst_run(case: CaseDoc) -> RunDoc | None:
    candidates = [r for r in case.runs if r.status == "failed"] or [r for r in case.runs if r.status == "error"]
    return min(candidates, key=lambda r: (r.score, r.repeat)) if candidates else None
