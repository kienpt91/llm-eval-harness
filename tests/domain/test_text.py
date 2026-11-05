from __future__ import annotations

import pytest

from llm_eval_harness.domain.text import collapse_whitespace, fence_for, fold, one_line, sanitize, truncate
from support.prng import HOSTILE, Prng


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("\x1b[31mred\x1b[0m", "red"),
        ("a\x1b[2Jb", "ab"),
        ("\x1b]0;window title\x07text", "text"),
        ("\x1b]8;;http://x.invalid\x1b\\link\x1b]8;;\x1b\\", "link"),
        ("\x1bPdcs data\x1b\\after", "after"),
        ("\x9b31mx", "x"),
        ("a\x1bcb", "ab"),
        ("plain\nline\ttab", "plain\nline\ttab"),
        ("bell\x07", "bell\\x07"),
        ("cr\rlf", "cr\\x0dlf"),
        ("nul\x00", "nul\\x00"),
        ("del\x7f", "del\\x7f"),
        ("\u202eevil\u202c", "<U+202E>evil<U+202C>"),
        ("a\u200bb", "a<U+200B>b"),
        ("x\u2028y", "x<U+2028>y"),
        ("\ufeffbom", "<U+FEFF>bom"),
        ("lone\ud800", "lone<U+D800>"),
        ("private\ue000", "private<U+E000>"),
        ("naïve café 日本語 \U0001f600", "naïve café 日本語 \U0001f600"),
    ],
)
def test_sanitize(raw: str, expected: str) -> None:
    assert sanitize(raw) == expected


def test_unterminated_escape_is_escaped_not_dropped_silently() -> None:
    assert sanitize("a\x1b[31") == "a\\x1b[31"
    assert sanitize("a\x1b]0;never ends") == "a\\x1b]0;never ends"


def test_sanitize_truncates_and_says_so() -> None:
    assert sanitize("x" * 10, max_chars=4) == "xxxx…[6 more characters]"
    assert one_line("a\nb\tc", max_chars=50) == "a\\nb\\tc"


def test_property_sanitised_text_has_no_control_or_bidi_characters() -> None:
    for seed in range(300):
        rng = Prng(seed)
        out = sanitize(rng.text(80, hostile=True) + HOSTILE[seed % len(HOSTILE)])
        for ch in out:
            assert ch in "\n\t" or (ord(ch) >= 0x20 and ord(ch) != 0x7F and not 0x80 <= ord(ch) <= 0x9F), (
                seed,
                repr(ch),
            )
            assert ch not in "\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069\u200e\u200f\u061c\u200b\ufeff", (
                seed,
                repr(ch),
            )
        assert "\x1b" not in out.replace("\\x1b", ""), seed


def test_sanitize_is_linear_on_hostile_input() -> None:
    import time

    started = time.monotonic()
    sanitize("\x1b[" * 200_000)
    sanitize("\x1b]" + "a" * 500_000)
    sanitize("\x1bP" + "b" * 500_000)
    assert time.monotonic() - started < 5


def test_fold_defeats_cheap_obfuscation() -> None:
    assert fold("PASS\u200bWORD") == "password"
    assert fold("ｐａｓｓｗｏｒｄ") == "password"
    assert fold("Straße") == fold("STRASSE")
    assert fold("a\u202eb") == "ab"


def test_truncate_and_whitespace_and_fence() -> None:
    assert truncate("abc", 5) == "abc"
    assert truncate("abcdef", 3) == "abc…[3 more characters]"
    assert collapse_whitespace("  a \n\t b  ") == "a b"
    assert fence_for("no ticks") == "```"
    assert fence_for("has ```` four") == "`````"
