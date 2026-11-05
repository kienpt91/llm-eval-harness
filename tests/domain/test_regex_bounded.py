"""User patterns are run by a linear-time engine with a work budget: hostile patterns cannot hang a run."""

from __future__ import annotations

import time

import pytest

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
HOSTILE_TEXTS = ["a" * 32_768, ("ab " * 11_000)[:32_768]]


@pytest.mark.parametrize("pattern", HOSTILE_PATTERNS)
def test_classic_catastrophic_patterns_are_accepted_and_finish_quickly_on_hostile_text(pattern: str) -> None:
    assert validate("regex", {"pattern": pattern}) == []
    started = time.monotonic()
    for text in HOSTILE_TEXTS:
        outcome = grade("regex", {"pattern": pattern}, output(text))
        assert isinstance(outcome.passed, bool)
    assert time.monotonic() - started < 30  # at worst the work budget (about two seconds a call)


def test_not_contains_patterns_are_bounded_too() -> None:
    started = time.monotonic()
    for pattern in (r"(a+)+$", r".*a.*b.*c.*d.*e"):
        grade("not_contains", {"patterns": [pattern]}, output("ab " * 10_900))
    assert time.monotonic() - started < 30


def test_a_pattern_that_exhausts_its_budget_fails_closed_with_an_explanation() -> None:
    heavy = "(" + "|".join("abcdef"[i % 6] for i in range(30)) + "){1,20}z"
    text = ("abcdef" * 6000)[:32_000]
    for kind, params in (("regex", {"pattern": heavy}), ("not_contains", {"patterns": [heavy]})):
        outcome = grade(kind, params, output(text))
        assert not outcome.passed, kind
    must_not_match = grade("regex", {"pattern": heavy, "must_match": False}, output(text))
    assert not must_not_match.passed, "a regex that could not be evaluated must not count as 'absent'"
    assert "work budget" in must_not_match.detail or "too large" in must_not_match.detail


def test_the_budget_does_not_fire_for_ordinary_patterns_on_ordinary_text() -> None:
    text = "Order #12345 shipped on 2026-11-05 to customer jane. " * 600
    assert grade("regex", {"pattern": r"order #\d+ shipped on \d{4}-\d{2}-\d{2}", "flags": ["i"]}, output(text)).passed
    assert grade("not_contains", {"patterns": [r"password\s*[:=]\s*\S+"]}, output(text)).passed


def test_schema_patterns_and_tool_argument_patterns_use_the_bounded_engine_too() -> None:
    schema = {"schema": {"type": "object", "properties": {"x": {"type": "string", "pattern": "(a+)+$"}}}}
    assert (
        grade("json_schema", schema, output("", structured={"x": "a" * 5000 + "!"}, has_structured=True)).passed
        is False
    )
    call = ToolCall("t", {"x": "a" * 20_000 + "!"})
    expect = {"calls": [{"name": "t", "args": {"x": {"regex": "(a+)+$"}}}]}
    started = time.monotonic()
    assert not grade("tool_calls", expect, output(tool_calls=(call,))).passed
    assert time.monotonic() - started < 30


def test_redaction_patterns_are_bounded_and_fail_closed() -> None:
    from llm_eval_harness.application.redaction import Redactor

    redactor = Redactor.build(True, [r"(a+)+$", r".*a.*b.*c.*d.*e"])
    started = time.monotonic()
    stored = redactor.text("a" * 30_000 + "!")
    assert time.monotonic() - started < 60
    assert len(stored) < 40_000
    heavy = "(" + "|".join("abcdef"[i % 6] for i in range(30)) + "){1,20}z"
    strict = Redactor.build(True, [heavy])
    out = strict.text(("abcdef" * 6000)[:32_000])
    assert "abcdef" not in out and "could not be evaluated" in out  # nothing unexamined is stored


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
