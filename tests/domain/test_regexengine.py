"""The pattern engine against ``re`` (seeded random patterns and texts) and on its own edge cases."""

from __future__ import annotations

import re
import time

import pytest

from llm_eval_harness.domain.grading.regexengine import BudgetExceeded, RegexError, compile_pattern
from llm_eval_harness.domain.limits import MAX_REGEX_STEPS
from support.prng import Prng

FLAG_BITS = {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL}


def py(pattern: str, flags: tuple[str, ...]) -> re.Pattern[str]:
    value = 0
    for f in flags:
        value |= FLAG_BITS[f]
    return re.compile(pattern, value)


def agree(pattern: str, text: str, flags: tuple[str, ...] = (), *, spans: bool = True) -> None:
    ours, theirs = compile_pattern(pattern, flags), py(pattern, flags)
    expected = theirs.search(text)
    assert ours.matches(text) == (expected is not None), (pattern, text, flags)
    assert ours.fullmatch(text) == (theirs.fullmatch(text) is not None), (pattern, text, flags, "fullmatch")
    if spans:
        assert ours.first_span(text) == (expected.span() if expected else None), (pattern, text, flags, "span")


@pytest.mark.parametrize(
    ("pattern", "text"),
    [
        ("abc", "xxabcxx"),
        ("a.c", "a\nc"),
        ("^a", "ba"),
        ("a$", "a\n"),
        ("a$", "a\nb"),
        (r"\bfoo\b", "a foo b"),
        (r"\Bfoo", "afoo"),
        (r"\Aab\Z", "ab"),
        ("a*", "baaa"),
        ("a+?", "aaa"),
        ("a{2,3}", "aaaa"),
        ("a{2,3}?", "aaaa"),
        ("(ab|a)(c|bcd)", "abcd"),
        ("(a|ab)(c|bcd)", "abcd"),
        ("a|ab", "ab"),
        ("x*", ""),
        ("[^a-c]+", "abxyzc"),
        (r"[\d\s]+", "ab 12 3c"),
        (r"[a\-z]+", "a-z"),
        (r"[]a]+", "]a]"),
        (r"[^]a]+", "bc]"),
        (r"\w+@\w+\.com", "mail me@x.com now"),
        ("(?:ab)+c", "ababc"),
        ("(?P<n>a)b", "ab"),
        ("a{,2}b", "aab"),
        ("a{2}", "aaa"),
        ("a{", "a{"),
        ("a{1", "a{1"),
        ("a{1,2", "a{1,2"),
        (r"\x41\u0042", "AB"),
        (r"a\.b", "a.b"),
        ("é+", "ééé"),
        (r"\d+", "٣٤"),
        ("(a*)*b", "aaab"),
        ("(a*)+", "aaa"),
        ("(a|)+b", "aab"),
        ("^$", ""),
        ("^$", "\n"),
        ("(?:a|b)*?c", "abac"),
    ],
)
def test_known_cases_agree_with_re(pattern: str, text: str) -> None:
    agree(pattern, text, spans=pattern not in {"(a*)*b", "(a*)+", "(a|)+b"})
    agree(pattern, text, ("i", "m", "s"), spans=False)


def random_pattern(rng: Prng, depth: int = 0) -> tuple[str, bool]:
    """A pattern of the supported syntax; the flag says whether a quantifier ever applies to a group."""
    pieces: list[str] = []
    quantified_group = False
    for _ in range(rng.integer(1, 4)):
        kind = rng.integer(0, 9)
        if kind <= 3:
            atom = rng.choice(["a", "b", "c", ".", "[ab]", "[^a]", r"\d", r"\w", r"\s", "[a-c]", r"\."])
        elif kind == 4:
            atom = rng.choice([r"\b", "^", "$", r"\B"])
            pieces.append(atom)
            continue
        elif depth < 2:
            inner, q = random_pattern(rng, depth + 1)
            quantified_group = quantified_group or q
            atom = "(" + rng.choice(["", "?:"]) + inner + ")"
            if rng.integer(0, 2) == 0:
                atom += rng.choice(["*", "+", "?", "{1,2}", "{2}"])
                quantified_group = True
            pieces.append(atom)
            continue
        else:
            atom = "a"
        pieces.append(
            atom + rng.choice(["", "", "", "*", "+", "?", "*?", "+?", "??", "{2}", "{1,3}", "{0,2}?", "{2,}"])
        )
    pattern = "".join(pieces)
    if depth < 2 and rng.integer(0, 4) == 0:
        other, q = random_pattern(rng, depth + 1)
        pattern += "|" + other
        quantified_group = quantified_group or q
    return pattern, quantified_group


def random_text(rng: Prng) -> str:
    return "".join(rng.choice(list("aabbc1 \n._x")) for _ in range(rng.integer(0, 14)))


def test_property_the_engine_agrees_with_re_on_random_patterns_and_texts() -> None:
    checked = spans_checked = 0
    for seed in range(1500):
        rng = Prng(seed)
        pattern, grouped_quantifier = random_pattern(rng)
        flags = tuple(f for f in "ims" if rng.integer(0, 3) == 0)
        try:
            theirs = py(pattern, flags)
        except re.error:
            with pytest.raises(RegexError):
                compile_pattern(pattern, flags)
            continue
        ours = compile_pattern(pattern, flags)
        for _ in range(6):
            text = random_text(rng)
            expected = theirs.search(text)
            assert ours.matches(text) == (expected is not None), (seed, pattern, flags, text)
            assert ours.fullmatch(text) == (theirs.fullmatch(text) is not None), (seed, pattern, flags, text)
            checked += 1
            if not grouped_quantifier:  # empty iterations of a repeated group are where engines legitimately differ
                assert ours.first_span(text) == (expected.span() if expected else None), (seed, pattern, flags, text)
                spans_checked += 1
    assert checked > 3000 and spans_checked > 1000


def test_property_spans_are_non_overlapping_non_empty_and_inside_the_text() -> None:
    for seed in range(300):
        rng = Prng(seed)
        pattern, _ = random_pattern(rng)
        try:
            program = compile_pattern(pattern)
        except RegexError:
            continue
        text = random_text(rng) * 3
        previous_end = 0
        for start, end in program.spans(text):
            assert previous_end <= start < end <= len(text), (pattern, text)
            previous_end = end


def test_spans_replace_like_re_finditer_for_simple_patterns() -> None:
    program = compile_pattern(r"ORD-\d+")
    assert program.spans("a ORD-1 b ORD-22 ORD-") == [(2, 7), (10, 16)]
    assert compile_pattern("a*").spans("baaac") == [(1, 4)]  # empty matches are skipped
    assert compile_pattern("x").spans("") == []


def test_work_is_linear_not_exponential() -> None:
    started = time.monotonic()
    for pattern, text in [
        (r"(a+)+$", "a" * 30_000 + "!"),
        (r"(a?){30}a{30}", "a" * 30),
        (r"(.*)*x", "y" * 20_000),
        (r".*a.*b.*c", "ab" * 10_000),
    ]:
        compile_pattern(pattern).matches(text)
    assert time.monotonic() - started < 40


def test_the_step_budget_is_a_hard_bound() -> None:
    program = compile_pattern("(" + "|".join("abcd"[i % 4] for i in range(30)) + "){1,20}z")
    started = time.monotonic()
    with pytest.raises(BudgetExceeded):
        program.matches("abcd" * 8000)
    assert time.monotonic() - started < 60
    assert MAX_REGEX_STEPS == 2_000_000


def test_the_budget_is_per_call_and_a_fresh_call_starts_fresh() -> None:
    program = compile_pattern("a")
    for _ in range(30):
        assert program.matches("b" * 30_000 + "a")


def test_unicode_classes_and_case_insensitivity() -> None:
    assert compile_pattern(r"\w+").fullmatch("héllo_٣")
    assert compile_pattern("straße", ("i",)).matches("STRASSE") is False  # no full Unicode case folding: documented
    assert compile_pattern("É", ("i",)).matches("é")
    assert compile_pattern("[a-c]+", ("i",)).fullmatch("ABc")


def test_the_nesting_and_size_limits_are_enforced_with_a_message() -> None:
    with pytest.raises(RegexError, match="nested deeper"):
        compile_pattern("(" * 40 + "a" + ")" * 40)
    with pytest.raises(RegexError, match="too large"):
        compile_pattern("(a{700}b{700}c{700})")
    compile_pattern("a{1000}")


@pytest.mark.parametrize(
    "bad", ["\\", "[a-", "a{1,2}{3}", "(?P<1x>a)", "(?P<n", r"\x4", r"\u00", r"\U00110000", "[a-\\d]"]
)
def test_malformed_patterns_raise_regex_error_never_anything_else(bad: str) -> None:
    with pytest.raises(RegexError):
        compile_pattern(bad)


def test_property_arbitrary_text_as_a_pattern_never_crashes_the_compiler() -> None:
    for seed in range(1500):
        rng = Prng(seed)
        pattern = "".join(rng.choice(list("ab()[]{}*+?|\\.^$-,:<>=!0123dws")) for _ in range(rng.integer(1, 12)))
        try:
            compile_pattern(pattern)
        except RegexError:
            continue
