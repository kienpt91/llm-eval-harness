"""Model- and program-controlled text must not become live Markdown or HTML in a pull request comment."""

from __future__ import annotations

import re
from dataclasses import replace

import pytest

from llm_eval_harness.application.report.markdown import render_markdown
from llm_eval_harness.application.report.terminal import render_terminal
from llm_eval_harness.domain.baseline import snapshot_of_report
from llm_eval_harness.domain.compare import CompareOptions, compare
from llm_eval_harness.domain.gate import GateOptions, decide
from llm_eval_harness.domain.model import GradeResult, ToolCall
from llm_eval_harness.domain.reportdoc import ReportDoc
from llm_eval_harness.domain.results import CaseResult, ErrorRecord, RunRecord
from support.runs import case_result, report_doc, run_record

HOSTILE = (
    "[Click to re-run](https://evil.example/x) ![beacon](https://evil.example/b.png) <img src=https://evil.example/b.png> "
    "<script>alert(1)</script> @team @everyone #123 https://evil.example/path www.evil.example mailto:a@evil.example "
    "&lt;b&gt; ``` ~~~ **bold** __under__ | pipe | \u202e bidi \x1b[31m red \x1b]8;;http://evil.example\x07link\x1b]8;;\x07 <https://auto.example>"
)
TRIGGERS = [
    r"\]\(",
    r"!\[",
    r"<",
    r">",
    r"@",
    r"https?://",
    r"www\.",
    r"mailto:",
    r"#\d",
    r"&\w+;",
    r"\*\*",
    r"__",
    r"~~",
    r"\x1b",
    "\u202e",
]


def outside_code(markdown: str) -> str:
    """The Markdown with fenced blocks and inline code spans removed: what a renderer would interpret."""
    lines: list[str] = []
    fence: str | None = None
    for line in markdown.splitlines():
        stripped = line.strip()
        if fence is None:
            opening = re.match(r"(`{3,})", stripped)
            if opening:
                fence = opening.group(1)
                continue
            lines.append(line)
        elif re.fullmatch(rf"{fence}`*", stripped):
            fence = None
    text = "\n".join(lines)
    out: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == "`":
            run = re.match(r"`+", text[i:])
            assert run is not None
            ticks = run.group(0)
            close = text.find(ticks, i + len(ticks))
            while close != -1 and text[close + len(ticks) : close + len(ticks) + 1] == "`":
                close = text.find(ticks, close + 1)
            if close != -1:
                i = close + len(ticks)
                continue
        out.append(text[i])
        i += 1
    return "".join(out)


def hostile_doc() -> ReportDoc:
    grade = GradeResult(HOSTILE, "exact", False, 0.0, 1.0, True, HOSTILE)
    failed = run_record("failed", text=HOSTILE, failed_musts=(HOSTILE,))
    failed = replace(failed, grades=(grade,))
    assert failed.output is not None
    failed = replace(failed, output=replace(failed.output, tool_calls=(ToolCall(HOSTILE, {"k": HOSTILE}),)))
    broken = RunRecord(0, 1, "error", 0.0, (), (), None, ErrorRecord("exit", HOSTILE))
    return report_doc(
        [
            CaseResult("safety-1", "safety", ("t",), "h", HOSTILE, (failed,)),
            CaseResult("q-failing", "quality", ("t",), "h", HOSTILE, (failed,)),
            CaseResult("q-error", "quality", ("t",), "h", HOSTILE, (broken,)),
            case_result("q-ok", ("passed", "failed", "passed")),
        ],
        dataset_source=HOSTILE,
    )


@pytest.mark.parametrize("with_comparison", [False, True])
def test_nothing_hostile_survives_outside_code_spans_and_fences(with_comparison: bool) -> None:
    doc = hostile_doc()
    comparison = None
    if with_comparison:
        base = report_doc([case_result("q-failing", ("passed",) * 5), case_result("q-ok", ("passed",) * 3)], repeat=5)
        comparison = compare(snapshot_of_report(base), snapshot_of_report(doc), CompareOptions(deterministic=True))
    markdown = render_markdown(doc, comparison, decide(doc, comparison, GateOptions()))
    rest = outside_code(markdown).replace("**must**", "")  # the report's own bold label
    rest = "\n".join(line.removeprefix("> ") for line in rest.splitlines())  # and its own block quote marker
    for trigger in TRIGGERS:
        assert not re.search(trigger, rest), (
            trigger,
            [line for line in rest.splitlines() if re.search(trigger, line)][:3],
        )


def test_hostile_text_is_still_shown_inside_code_spans_so_the_reviewer_can_read_it() -> None:
    doc = hostile_doc()
    markdown = render_markdown(doc, None, decide(doc, None, GateOptions()))
    assert "Click to re-run" in markdown and "evil.example" in markdown  # visible, but inert


def test_the_markdown_has_no_html_of_its_own() -> None:
    doc = report_doc([case_result("a", ("passed", "failed"))])
    assert "<" not in outside_code(render_markdown(doc))


def test_table_cells_cannot_be_split_by_a_pipe_or_a_newline() -> None:
    doc = hostile_doc()
    base = report_doc([case_result("q-failing", ("passed",) * 5)], repeat=5)
    comparison = compare(snapshot_of_report(base), snapshot_of_report(doc), CompareOptions(deterministic=True))
    markdown = render_markdown(doc, comparison, decide(doc, comparison, GateOptions()))
    tables: list[list[str]] = []
    for line in markdown.splitlines():
        if line.startswith("|"):
            if not tables or tables[-1][-1] == "":
                tables.append([])
            tables[-1].append(line)
        elif tables and tables[-1][-1] != "":
            tables[-1].append("")
    assert tables
    for table in tables:
        rows = [line for line in table if line]
        assert all(line.endswith("|") for line in rows)  # a cell cannot have leaked a newline
        assert len({len(re.findall(r"(?<!\\)\|", line)) for line in rows}) == 1, rows  # nor gained a column


def test_backticks_in_untrusted_text_cannot_close_the_code_span() -> None:
    nasty = "end` then [x](http://evil.example) and ``` and `` more"
    doc = report_doc(
        [
            CaseResult(
                "a",
                "quality",
                (),
                "h",
                nasty,
                (replace(run_record("failed"), grades=(GradeResult(nasty, "exact", False, 0.0, 1.0, False, nasty),)),),
            )
        ]
    )
    rest = outside_code(render_markdown(doc))
    assert "evil.example" not in rest and "](" not in rest


def test_the_terminal_sanitises_every_field_even_a_category_edited_by_hand() -> None:
    doc = hostile_doc()
    cases = tuple(
        replace(c, stat=replace(c.stat, category="safety\x1b[31m\u202e")) if i == 0 else c
        for i, c in enumerate(doc.cases)
    )
    text = render_terminal(replace(doc, cases=cases), None, decide(doc, None, GateOptions()))
    assert "\x1b" not in text and "\u202e" not in text
