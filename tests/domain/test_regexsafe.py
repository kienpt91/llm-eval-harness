from __future__ import annotations

import re
import time

import pytest

from llm_eval_harness.domain.grading.regexsafe import check_pattern, compile_flags


@pytest.mark.parametrize(
    "pattern",
    [r"(a+)+$", r"(a*)*b", r"(.*)*", r"(a|aa)+", r"(x+x+)+y", r"(?:ab*)+", r"(\w+\s?)*$", r"((a+)b)+", r"(a{2,})+"],
)
def test_catastrophic_shapes_are_refused(pattern: str) -> None:
    reason = check_pattern(pattern)
    assert reason is not None and "backtracking" in reason


@pytest.mark.parametrize("pattern", [r"(a)\1", r"(?P<x>a)(?P=x)", r"(?=a)b", r"(?!a)b", r"(?<=a)b", r"(?<!a)b"])
def test_backreferences_and_lookaround_are_refused(pattern: str) -> None:
    assert check_pattern(pattern) is not None


@pytest.mark.parametrize(
    "pattern",
    [
        r"\d{3}-\d{4}",
        r"^order-[0-9a-f]{8}$",
        r"(ab){1,5}",
        r"[a-z]+@[a-z]+\.com",
        r"colou?r",
        r"\(a+\)+",
        r"[(a+)+]",
        r"a+b*c?",
        r"(?i)hello",
        r"x{2,5}",
    ],
)
def test_ordinary_patterns_are_accepted(pattern: str) -> None:
    assert check_pattern(pattern) is None


def test_invalid_empty_and_long_patterns() -> None:
    assert "valid regular expression" in (check_pattern("(") or "")
    assert check_pattern("") == "must not be empty"
    assert "too long" in (check_pattern("a" * 513) or "")


def test_character_class_edge_cases_do_not_confuse_the_screen() -> None:
    assert check_pattern(r"[]a+]+") is None
    assert check_pattern(r"[^]a]+x") is None
    assert check_pattern(r"[\]a]+") is None


def test_long_text_is_refused_instead_of_scanned() -> None:
    from llm_eval_harness.domain.grading.regexsafe import too_long_for_regex
    from llm_eval_harness.domain.limits import MAX_REGEX_INPUT_CHARS

    assert not too_long_for_regex("a" * MAX_REGEX_INPUT_CHARS)
    assert too_long_for_regex("a" * (MAX_REGEX_INPUT_CHARS + 1))


def test_accepted_quadratic_pattern_is_bounded_by_the_input_cap() -> None:
    from llm_eval_harness.domain.limits import MAX_REGEX_INPUT_CHARS

    hostile = "a" * MAX_REGEX_INPUT_CHARS
    started = time.monotonic()
    assert check_pattern(r"a+b") is None
    re.search(r"a+b", hostile)
    assert time.monotonic() - started < 10


def test_compile_flags() -> None:
    assert compile_flags(("i", "m")) == re.IGNORECASE | re.MULTILINE
    assert compile_flags(()) == 0
