"""Loading and running user-supplied regular expressions through the bounded engine.

User patterns are never handed to Python's ``re``: see ``regexengine`` for why, and for the supported syntax. Two
bounds apply on top of the engine's linear time: text longer than ``MAX_REGEX_INPUT_CHARS`` is not scanned at all
and a call that spends its step budget raises ``BudgetExceeded``. Callers FAIL CLOSED in both cases (a grader that
could not decide does not pass, whatever the pattern was meant to prove), via ``refusal_detail`` / ``BUDGET_DETAIL``.
"""

from __future__ import annotations

from functools import lru_cache

from llm_eval_harness.domain.grading.regexengine import BudgetExceeded, Program, RegexError, compile_pattern
from llm_eval_harness.domain.limits import MAX_REGEX_INPUT_CHARS, MAX_REGEX_LENGTH

FLAG_NAMES = ("i", "m", "s")
BUDGET_DETAIL = "the pattern used up its work budget on this text; counted as not decided (failing closed)"

__all__ = [
    "BUDGET_DETAIL",
    "FLAG_NAMES",
    "BudgetExceeded",
    "check_pattern",
    "compiled",
    "pattern_matches",
    "refusal_detail",
    "too_long_for_regex",
]


def check_pattern(pattern: str, flags: tuple[str, ...] = ()) -> str | None:
    """The reason a pattern is refused, or ``None`` when the engine accepts it."""
    if len(pattern) > MAX_REGEX_LENGTH:
        return f"is too long ({len(pattern)} characters, limit {MAX_REGEX_LENGTH})"
    try:
        compiled(pattern, flags)
    except RegexError as error:
        return str(error)
    return None


@lru_cache(maxsize=512)
def compiled(pattern: str, flags: tuple[str, ...] = ()) -> Program:
    """The compiled program of an already validated pattern (cached: patterns repeat for every case)."""
    return compile_pattern(pattern, flags)


def pattern_matches(pattern: str, flags: tuple[str, ...], text: str, *, full: bool = False) -> bool:
    """Whether the pattern matches in (or, with ``full``, all of) ``text``. May raise ``BudgetExceeded``."""
    program = compiled(pattern, flags)
    return program.fullmatch(text) if full else program.matches(text)


def too_long_for_regex(text: str) -> bool:
    """True when ``text`` is longer than a user-supplied pattern may be run on (see ``MAX_REGEX_INPUT_CHARS``)."""
    return len(text) > MAX_REGEX_INPUT_CHARS


def refusal_detail(subject: str) -> str:
    return f"{subject} is longer than {MAX_REGEX_INPUT_CHARS} characters; patterns are not run on it (failing closed)"
