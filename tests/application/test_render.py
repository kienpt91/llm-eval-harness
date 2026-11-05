from __future__ import annotations

import re

from llm_eval_harness.application.report.markdown import render_markdown
from llm_eval_harness.application.report.terminal import render_terminal
from llm_eval_harness.domain.baseline import snapshot_of_report
from llm_eval_harness.domain.compare import CompareOptions, compare
from llm_eval_harness.domain.gate import GateOptions, decide
from llm_eval_harness.domain.model import ToolCall
from llm_eval_harness.domain.reportdoc import ReportDoc
from llm_eval_harness.domain.results import CaseResult, OutputRecord, RunRecord
from support.runs import case_result, report_doc, run_record

CONTROL = re.compile("[\x00-\x08\x0b-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069\u200b-\u200f]")


def hostile_results() -> list[CaseResult]:
    evil = "\x1b[31mred\x1b]0;pwned\x07\u202eevil\u2069 | `ticks` ``` \x00\r\n# not a heading"
    bad = run_record("failed", text=evil, failed_musts=("forbidden",))
    out = bad.output
    assert out is not None
    tool = RunRecord(
        bad.repeat, bad.attempts, bad.status, bad.score, bad.failed_musts, bad.grades, _with_tools(out), None
    )
    return [
        CaseResult("s-evil", "safety", ("t",), "h", evil, (tool,)),
        case_result("q-ok", ("passed",) * 3),
        case_result("q-flaky", ("passed", "failed", "passed")),
        case_result("q-broken", ("error",)),
    ]


def _with_tools(out: OutputRecord) -> OutputRecord:
    from dataclasses import replace

    return replace(out, tool_calls=(ToolCall("issue_refund", {"amount": 500, "note": "\x1b[1m"}),))


def doc(results: list[CaseResult] | None = None) -> ReportDoc:
    return report_doc(results or hostile_results(), repeat=3)


def test_terminal_output_has_no_control_or_bidi_characters_from_untrusted_text() -> None:
    d = doc()
    text = render_terminal(d, None, decide(d, None, GateOptions()))
    assert not CONTROL.search(text.replace("\n", ""))
    assert "\x1b" not in text and "pwned" not in text  # the title payload of the OSC sequence goes with it
    assert "red" in text and "evil" in text  # visible text stays


def test_markdown_output_is_safe_and_cannot_break_out_of_its_blocks() -> None:
    d = doc()
    text = render_markdown(d, None, decide(d, None, GateOptions()))
    assert not CONTROL.search(text.replace("\n", ""))
    assert "\x1b" not in text
    # the hostile output holds a run of three backticks: its fence must be longer
    assert "````" in text
    # the ids are placed in table cells and code spans: pipes and backticks cannot split them
    assert "#### `s-evil" in text


def test_terminal_summary_content() -> None:
    d = doc([case_result("a", ("passed",) * 2), case_result("b", ("passed", "failed"))])
    text = render_terminal(d)
    for expected in [
        "llm-eval report",
        "dataset   datasets/x.jsonl  sha256:abc",
        "(2 of 3 cases)",
        "quality   mean pass rate 0.750 over 2 cases (1 fully passing)",
        "safety    0 cases, 0 failing",
        "runs      4 graded, 0 errored, 1 flaky case",
        "tokens    in 40  out 20",
        "cost      0.004000 USD total, 0.001000 USD per run",
        "latency   p50 10 ms",
        "FAIL  quality b  1/2 runs passed",
        "facts: detail",
        "flaky cases",
    ]:
        assert expected in text, expected


def test_terminal_lists_errors_and_unknown_cost() -> None:
    d = doc([case_result("e", ("error",), error_kind="spawn"), case_result("n", ("passed",), cost=None)])
    text = render_terminal(d)
    assert "ERROR quality e" in text and "provider error (spawn): the provider failed" in text
    assert "cost      unknown (1 run(s) without usage or a price)" in text


def test_terminal_with_comparison_and_decision() -> None:
    base = report_doc([case_result("a", ("passed",) * 5), case_result("b", ("passed",) * 5)], repeat=5)
    cur = report_doc(
        [case_result("a", ("failed",) * 5), case_result("b", ("passed",) * 5), case_result("n", ("passed",))],
        repeat=5,
        dataset_hash="sha256:new",
    )
    comparison = compare(snapshot_of_report(base), snapshot_of_report(cur), CompareOptions())
    decision = decide(cur, comparison, GateOptions())
    text = render_terminal(cur, comparison, decision)
    for expected in [
        "comparison with the baseline",
        "quality   1.000 -> 0.667",
        "verdict   REGRESSION",
        "regressed    a:",
        "dataset hash differs",
        "gate      FAIL (exit 1)",
        "- 1 case(s) regressed",
    ]:
        assert expected in text, expected


def test_markdown_for_a_pull_request() -> None:
    base = report_doc(
        [case_result("a", ("passed",) * 5, cost="0.001"), case_result("b", ("passed",) * 5, cost="0.001")], repeat=5
    )
    cur = report_doc(
        [
            case_result("a", ("failed",) * 5, text="a wrong answer", cost="0.002", latency_ms=50),
            case_result("b", ("passed",) * 5, cost="0.002", latency_ms=50),
            case_result("c", ("passed", "failed", "passed", "passed", "failed")),
        ],
        repeat=5,
    )
    comparison = compare(snapshot_of_report(base), snapshot_of_report(cur), CompareOptions())
    decision = decide(cur, comparison, GateOptions())
    md = render_markdown(cur, comparison, decision)
    for expected in [
        "## LLM eval: FAILED (exit 1)",
        "| Quality (mean pass rate) | 1.000 | 0.533 |",
        "### Regressed",
        "| `a` | `quality` | `pass rate 1.00 (5/5) to 0.00 (0/5); p=0.008` |",
        "### Worst quality cases",
        "#### `a` (`quality`): 0/5 runs passed",
        "a wrong answer",
        "Input:",
        "`facts`: `detail`",
        "### Flaky cases",
        "| `c` | 0.600 | 3/5 |",
        "| Cost per run |",
        "| Latency p95 |",
    ]:
        assert expected in md, expected


def test_markdown_lists_safety_failures_first_and_marks_them_fatal() -> None:
    d = doc()
    md = render_markdown(d, None, decide(d, None, GateOptions()))
    assert md.index("### Safety failures (always fatal)") < md.index("### Worst quality cases")
    assert "## LLM eval: FAILED (exit 4)" in md and "issue_refund" in md


def test_markdown_without_comparison_or_decision() -> None:
    assert render_markdown(doc([case_result("a")])).startswith("## LLM eval report\n")
    base = doc([case_result("a")])
    cmp = compare(snapshot_of_report(base), snapshot_of_report(base), CompareOptions())
    assert render_markdown(base, cmp).startswith("## LLM eval: ok\n")


def test_long_lists_are_truncated_with_a_count() -> None:
    results = [case_result(f"c{i:02d}", ("failed",)) for i in range(40)]
    d = report_doc(results)
    assert "... and 25 more" in render_terminal(d)
    assert "5 of 40 not fully passing" in render_markdown(d)
