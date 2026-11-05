"""User patterns are run by a linear-time engine with a work budget: hostile patterns cannot hang a run.

The claims are about *work*, so they are checked with the engine's own deterministic step counter (it depends only
on the pattern and the text), never with a clock: a clock measures the runner, not the code.
"""

from __future__ import annotations

import pytest

from llm_eval_harness.application.redaction import UNEVALUATED, Redactor
from llm_eval_harness.domain.grading import regexengine
from llm_eval_harness.domain.grading.regexengine import BudgetExceeded, Meter, compile_pattern
from llm_eval_harness.domain.grading.regexsafe import BUDGET_DETAIL
from llm_eval_harness.domain.limits import MAX_REGEX_STEPS
from llm_eval_harness.domain.model import ToolCall
from support.builders import grade, output, validate

HOSTILE_PATTERNS = [
    r"(a+)+$",
    r"(a*)*b",
    r"(.*)*x",
    r"(a|aa)+$",
    r"(x+x+)+y",
    r"(\w+\s?)*$",
    r"(a?){30}a{30}",
    r"(\w+\s?){1,50}x",
    r"(a+){1,50}b",
    r".*a.*b.*c.*d.*e",
    r"\d+\s*\d+\s*\d+\s*\d+z",
    r"refund.*approved.*\d+ days.*never",
]
SIZE = 1_000
HOSTILE_TEXTS = [
    "a" * SIZE,
    ("ab " * SIZE)[:SIZE],
    ("1 " * SIZE)[:SIZE] + "!",
    ("refund approved 30 days " * SIZE)[:SIZE],
]
HEAVY = "(" + "|".join("abcdef"[i % 6] for i in range(30)) + "){1,20}z"


@pytest.mark.parametrize("pattern", HOSTILE_PATTERNS)
def test_classic_catastrophic_patterns_are_accepted_and_their_work_is_linear_in_the_text(pattern: str) -> None:
    assert validate("regex", {"pattern": pattern}) == []
    program = compile_pattern(pattern)
    for text in HOSTILE_TEXTS:
        for call in (program.matches, program.fullmatch, program.first_span):
            meter = Meter()
            call(text, meter=meter)  # type: ignore[operator]
            assert 0 < meter.steps <= program.step_bound(len(text)), (pattern, text[:20], meter.steps)
            assert meter.steps < MAX_REGEX_STEPS


@pytest.mark.parametrize("pattern", [r"(a+)+$", r".*a.*b.*c.*d.*e", r"(\w+\s?){1,50}x"])
def test_the_graders_run_hostile_patterns_on_hostile_text_and_decide(pattern: str) -> None:
    for text in HOSTILE_TEXTS[:2]:
        assert isinstance(grade("regex", {"pattern": pattern}, output(text)).passed, bool)
        assert isinstance(grade("not_contains", {"patterns": [pattern]}, output(text)).passed, bool)


def test_work_grows_with_the_text_but_never_faster_than_the_text_times_the_program() -> None:
    program = compile_pattern(r"(\w+\s?){1,20}x")
    steps = []
    for size in (500, 1_000, 2_000, 4_000):
        meter = Meter()
        program.matches("ab " * (size // 3), meter=meter)
        steps.append(meter.steps)
        assert meter.steps <= program.step_bound(size)
    assert steps == sorted(steps) and steps[-1] < steps[0] * 12  # about 8x the text: far from quadratic (64x)


def test_a_pattern_that_exhausts_its_budget_fails_closed_with_an_explanation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(regexengine, "MAX_REGEX_STEPS", 2_000)  # the budget is read at call time
    text = "abcdef" * 600
    for kind, params in (("regex", {"pattern": HEAVY}), ("not_contains", {"patterns": [HEAVY]})):
        outcome = grade(kind, params, output(text))
        assert not outcome.passed and outcome.detail == BUDGET_DETAIL, kind
    absent = grade("regex", {"pattern": HEAVY, "must_match": False}, output(text))
    assert not absent.passed, "a regex that could not be evaluated must not count as 'absent'"
    assert absent.detail == BUDGET_DETAIL


def test_the_budget_is_exact_and_deterministic() -> None:
    program = compile_pattern(HEAVY)
    text = "abcdef" * 600
    spent = Meter()
    with pytest.raises(BudgetExceeded):
        program.matches(text, max_steps=5_000, meter=spent)
    assert 5_000 < spent.steps <= 5_000 + len(program.insts) + 1  # stops at the first step past the budget
    again = Meter()
    with pytest.raises(BudgetExceeded):
        program.matches(text, max_steps=5_000, meter=again)
    assert again.steps == spent.steps  # the same text and pattern cost the same every time
    short = "abcdef" * 40 + "z"
    enough = Meter()
    assert program.matches(short, max_steps=MAX_REGEX_STEPS, meter=enough) is True
    assert enough.steps <= program.step_bound(len(short))


def test_the_production_budget_is_not_lowered_by_the_test_hooks() -> None:
    assert MAX_REGEX_STEPS == 2_000_000
    program = compile_pattern("a")
    meter = Meter()
    program.matches("b" * 1_000, meter=meter)
    assert meter.steps <= program.step_bound(1_000) < MAX_REGEX_STEPS


def test_ordinary_patterns_on_ordinary_text_use_a_small_part_of_the_budget() -> None:
    text = "Order #12345 shipped on 2026-11-05 to customer jane. " * 100
    for pattern, flags in ((r"order #\d+ shipped on \d{4}-\d{2}-\d{2}", ("i",)), (r"password\s*[:=]\s*\S+", ())):
        program = compile_pattern(pattern, flags)
        meter = Meter()
        program.matches(text, meter=meter)
        assert meter.steps < MAX_REGEX_STEPS // 10, (pattern, meter.steps)
    assert grade("regex", {"pattern": r"order #\d+ shipped", "flags": ["i"]}, output(text)).passed
    assert grade("not_contains", {"patterns": [r"password\s*[:=]\s*\S+"]}, output(text)).passed


def test_schema_patterns_and_tool_argument_patterns_use_the_bounded_engine_too(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(regexengine, "MAX_REGEX_STEPS", 3_000)
    schema = {"schema": {"type": "object", "properties": {"x": {"type": "string", "pattern": HEAVY}}}}
    failed = grade("json_schema", schema, output("", structured={"x": "abcdef" * 600}, has_structured=True))
    assert not failed.passed and BUDGET_DETAIL in failed.detail
    call = ToolCall("t", {"x": "abcdef" * 600})
    expect = {"calls": [{"name": "t", "args": {"x": {"regex": HEAVY}}}]}
    outcome = grade("tool_calls", expect, output(tool_calls=(call,)))
    assert not outcome.passed and "work budget" in outcome.detail


def test_redaction_patterns_are_bounded_and_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    ordinary = Redactor.build(True, [r"ORD-\d+"])
    assert ordinary.text("see ORD-12345 now") == "see [REDACTED:custom] now"
    monkeypatch.setattr(regexengine, "MAX_REGEX_STEPS", 2_000)
    strict = Redactor.build(True, [HEAVY])
    out = strict.text("abcdef" * 600)
    assert out == UNEVALUATED and "abcdef" not in out  # nothing unexamined is stored
    assert strict.json({"k": "abcdef" * 600}) == {"k": UNEVALUATED}


def test_syntax_that_the_engine_does_not_support_is_refused_at_load_time_with_the_reason() -> None:
    for pattern, fragment in [
        (r"(a)\1", "back-references"),
        (r"(?=a)b", "look-around"),
        (r"(?<!a)b", "look-around"),
        (r"(?i)a", "inline flags"),
        (r"a*+", "possessive"),
        (r"(?>a)", "atomic"),
        (r"\p{L}", "unsupported escape"),
        (r"[[:alpha:]]", "POSIX classes"),
        (r"a**", "stacked"),
        (r"*a", "nothing to repeat"),
        (r"(a", "missing closing parenthesis"),
        (r"a)", "unbalanced"),
        (r"[a", "missing closing bracket"),
        (r"a{2,1}", "minimum"),
        (r"a{5000}", "limited to"),
        (r"(a{1000}){1000}", "too large"),
        (r"[z-a]", "bad character range"),
    ]:
        found = [p.render() for p in validate("regex", {"pattern": pattern})]
        assert found and fragment in found[0], (pattern, found)


def test_not_contains_patterns_are_bounded_too() -> None:
    pattern = r".*a.*b.*c.*d.*e"
    program = compile_pattern(pattern)
    meter = Meter()
    program.matches("ab " * 400, meter=meter)
    assert meter.steps <= program.step_bound(1_200)
    assert grade("not_contains", {"patterns": [pattern]}, output("ab " * 400)).passed  # no c, d or e: not present
