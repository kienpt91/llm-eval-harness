"""The regular expression front door: loading, caching, the input cap, the flags."""

from __future__ import annotations

import pytest

from llm_eval_harness.domain.grading.regexsafe import (
    check_pattern,
    compiled,
    pattern_matches,
    refusal_detail,
    too_long_for_regex,
)
from llm_eval_harness.domain.limits import MAX_REGEX_INPUT_CHARS, MAX_REGEX_LENGTH


@pytest.mark.parametrize(
    "pattern",
    [
        r"\d{3}-\d{4}",
        r"^order-[0-9a-f]{8}$",
        r"(ab){1,5}",
        r"[a-z]+@[a-z]+\.com",
        r"colou?r",
        r"\(a+\)+",
        r"(a+)+$",
        r"x{2,5}",
    ],
)
def test_ordinary_and_formerly_dangerous_patterns_are_accepted(pattern: str) -> None:
    assert check_pattern(pattern) is None


@pytest.mark.parametrize(
    ("pattern", "fragment"),
    [
        ("", "must not be empty"),
        ("(", "missing closing parenthesis"),
        ("a" * (MAX_REGEX_LENGTH + 1), "too long"),
        (r"(a)\1", "back-references"),
    ],
)
def test_refusals_say_why(pattern: str, fragment: str) -> None:
    assert fragment in (check_pattern(pattern) or "")


def test_the_reason_includes_the_position() -> None:
    assert "(at position " in (check_pattern("abc*?*") or "")


def test_unknown_flags_are_refused() -> None:
    assert "unknown flag" in (check_pattern("a", ("x",)) or "")


def test_compiled_patterns_are_cached_and_flag_sensitive() -> None:
    assert compiled("abc") is compiled("abc")
    assert compiled("abc", ("i",)) is not compiled("abc")
    assert pattern_matches("abc", ("i",), "xABCx") and not pattern_matches("abc", (), "xABCx")


def test_long_text_is_refused_instead_of_scanned() -> None:
    assert not too_long_for_regex("a" * MAX_REGEX_INPUT_CHARS)
    assert too_long_for_regex("a" * (MAX_REGEX_INPUT_CHARS + 1))
    assert str(MAX_REGEX_INPUT_CHARS) in refusal_detail("answer")
