"""Compile user-supplied regular expressions defensively.

Python's ``re`` cannot be interrupted, so a pattern with catastrophic backtracking would hang the harness.
``check_pattern`` is a conservative static screen, not a proof: it rejects over-long patterns, back-references,
look-around and a quantified group that itself contains an unbounded quantifier (``(a+)+``, ``(.*)*``,
``(a|aa)*``-style alternation under a star). Text that is matched is already bounded by the output cap.
"""

from __future__ import annotations

import re

from llm_eval_harness.domain.limits import MAX_REGEX_INPUT_CHARS, MAX_REGEX_LENGTH

FLAGS = {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL}


def check_pattern(pattern: str) -> str | None:
    """Return the reason a pattern is refused, or ``None`` when it is acceptable and compiles."""
    if not pattern:
        return "must not be empty"
    if len(pattern) > MAX_REGEX_LENGTH:
        return f"is too long ({len(pattern)} characters, limit {MAX_REGEX_LENGTH})"
    reason = _screen(pattern)
    if reason is not None:
        return reason
    try:
        re.compile(pattern)
    except re.error as error:
        return f"is not a valid regular expression ({error})"
    return None


def _is_unbounded_quantifier(pattern: str, i: int) -> int:
    """Length of an unbounded quantifier at ``i`` (``*``, ``+``, ``{n,}``), else 0."""
    if pattern[i] in "*+":
        return 1
    if pattern[i] == "{":
        end = pattern.find("}", i)
        if end != -1 and re.fullmatch(r"\{\d*,\}", pattern[i : end + 1]):
            return end + 1 - i
    return 0


def _screen(pattern: str) -> str | None:
    # stack of groups: True when the group (so far) holds an unbounded quantifier or an alternation
    stack: list[list[bool]] = []
    i = 0
    in_class = False
    while i < len(pattern):
        ch = pattern[i]
        if ch == "\\":
            nxt = pattern[i + 1 : i + 2]
            if nxt.isdigit() and nxt != "0" and not in_class:
                return "uses a back-reference, which can backtrack without limit"
            i += 2
            continue
        if in_class:
            if ch == "]":
                in_class = False
            i += 1
            continue
        if ch == "[":
            in_class = True
            i += 1
            if i < len(pattern) and pattern[i] == "^":
                i += 1
            if i < len(pattern) and pattern[i] == "]":
                i += 1
            continue
        if ch == "(":
            if pattern.startswith(("(?=", "(?!", "(?<=", "(?<!"), i):
                return "uses look-around, which is not supported"
            if pattern.startswith("(?P=", i):
                return "uses a back-reference, which can backtrack without limit"
            stack.append([False, False])  # [has unbounded quantifier, has alternation]
        elif ch == ")":
            inner = stack.pop() if stack else [False, False]
            j = i + 1
            length = _is_unbounded_quantifier(pattern, j) if j < len(pattern) else 0
            if length and (inner[0] or inner[1]):
                return "repeats a group that itself repeats or alternates without a bound (catastrophic backtracking)"
            if stack and (inner[0] or length):
                stack[-1][0] = True
            if stack and inner[1]:
                stack[-1][1] = True
        elif ch == "|":
            if stack:
                stack[-1][1] = True
        else:
            length = _is_unbounded_quantifier(pattern, i)
            if length and stack:
                stack[-1][0] = True
                i += length
                continue
        i += 1
    return None


def compile_flags(names: tuple[str, ...]) -> int:
    value = 0
    for name in names:
        value |= FLAGS[name]
    return value


def too_long_for_regex(text: str) -> bool:
    """True when ``text`` is longer than a user-supplied pattern may be run on (see ``MAX_REGEX_INPUT_CHARS``)."""
    return len(text) > MAX_REGEX_INPUT_CHARS


def refusal_detail(subject: str) -> str:
    return f"{subject} is longer than {MAX_REGEX_INPUT_CHARS} characters; patterns are not run on it (failing closed)"
