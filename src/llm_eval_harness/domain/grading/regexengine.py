"""A small regular expression engine for user-supplied patterns: linear time, bounded work, restricted syntax.

Why not ``re``: Python's engine backtracks, cannot be interrupted, and holds the interpreter lock, so one hostile
pattern (``(a+)+$``, ``.*a.*b.*c``, ``(a?){30}a{30}``) on hostile text hangs the whole run. Screening patterns is
not a proof. This engine is a Pike VM: it advances every possible match in lock step, so a search costs at most
``len(text) * len(program)`` steps whatever the pattern, and a step budget (``MAX_REGEX_STEPS`` per call) turns even
that into a hard bound: when it is spent, ``BudgetExceeded`` is raised and the caller fails closed.

Supported syntax (anything else is rejected when the pattern is loaded, with the position):

    literals and escaped punctuation     a  \\.  \\-            escapes   \\n \\t \\r \\f \\v \\a \\0 \\xHH \\uHHHH \\UHHHHHHHH
    any character                        .   (not a newline unless flag s)
    classes                              [abc] [^abc] [a-z] [\\d\\w\\s] and the negated \\D \\W \\S, also inside [...]
    anchors and assertions               ^ $ \\A \\Z \\b \\B
    groups                               (...) (?:...) (?P<name>...)      groups never capture
    alternation                          a|b
    repetition                           * + ? {n} {n,} {n,m} {,m}, and the lazy forms *? +? ?? {n,m}?   (n, m <= 1000)

Not supported: back-references, look-around, possessive and atomic forms, conditionals, inline flags such as
``(?i)`` (use the ``flags`` of the grader: ``i``, ``m``, ``s``), named or numbered escapes (``\\p{..}``, ``\\N{..}``),
POSIX classes, comments, and patterns that compile to more than ``MAX_PROGRAM`` instructions. Case-insensitive
matching compares the lower-case and upper-case forms of a character (no full Unicode case folding).

Semantics are Perl-style leftmost-first, like ``re``; the test suite compares the two on seeded random patterns and
texts. ``$`` matches at the end and before a final newline, as in ``re``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llm_eval_harness.domain.limits import MAX_REGEX_STEPS

MAX_PROGRAM = 2000
MAX_REPEAT_COUNT = 1000
MAX_NESTING = 24

_CHAR, _ANY, _SET, _SPLIT, _JMP, _ASSERT, _MATCH = range(7)
_BOL, _EOL, _BOS, _EOS, _WORDB, _NWORDB = range(6)

type _Range = tuple[int, int]


class RegexError(ValueError):
    """The pattern is invalid or uses syntax this engine does not support."""


class BudgetExceeded(Exception):
    """The match used up its step budget. The caller treats this as 'could not decide' and fails closed."""


@dataclass(frozen=True, slots=True)
class _CharClass:
    ranges: tuple[_Range, ...]
    classes: str  # letters of \d \w \s \D \W \S
    negated: bool


@dataclass(slots=True)
class _Budget:
    left: int = MAX_REGEX_STEPS


# --- parsing ---------------------------------------------------------------------------------------------------

_SIMPLE_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "f": "\f", "v": "\v", "a": "\a", "0": "\0"}
_ASSERT_ESCAPES = {"b": _WORDB, "B": _NWORDB, "A": _BOS, "Z": _EOS}
_CLASS_ESCAPES = "dDwWsS"
_HEX = "0123456789abcdefABCDEF"

type _Node = tuple[Any, ...]  # tagged tuples: ("char", c), ("cat", [..]), ("rep", node, lo, hi, lazy), ...


class _Parser:
    def __init__(self, pattern: str) -> None:
        self.p = pattern
        self.i = 0
        self.depth = 0

    def fail(self, message: str) -> RegexError:
        return RegexError(f"{message} (at position {self.i})")

    def parse(self) -> _Node:
        node = self.alternation()
        if self.i < len(self.p):
            raise self.fail("unbalanced parenthesis")
        return node

    def peek(self) -> str:
        return self.p[self.i] if self.i < len(self.p) else ""

    def alternation(self) -> _Node:
        self.depth += 1
        if self.depth > MAX_NESTING:
            raise self.fail(f"groups are nested deeper than {MAX_NESTING} levels")
        branches = [self.concatenation()]
        while self.peek() == "|":
            self.i += 1
            branches.append(self.concatenation())
        self.depth -= 1
        return branches[0] if len(branches) == 1 else ("alt", branches)

    def concatenation(self) -> _Node:
        items: list[_Node] = []
        while self.i < len(self.p) and self.p[self.i] not in "|)":
            items.append(self.repetition())
        return ("cat", items)

    def quantifier(self) -> tuple[int, int | None] | None:
        c = self.peek()
        if c == "*":
            self.i += 1
            return 0, None
        if c == "+":
            self.i += 1
            return 1, None
        if c == "?":
            self.i += 1
            return 0, 1
        if c == "{":
            return self.braces()
        return None

    def braces(self) -> tuple[int, int | None] | None:
        """``{n}``, ``{n,}``, ``{n,m}``, ``{,m}``; anything else after a ``{`` is a literal brace, as in ``re``."""
        j = self.i + 1
        lo_text = ""
        while j < len(self.p) and self.p[j].isdecimal() and self.p[j].isascii():
            lo_text += self.p[j]
            j += 1
        comma = j < len(self.p) and self.p[j] == ","
        hi_text = ""
        if comma:
            j += 1
            while j < len(self.p) and self.p[j].isdecimal() and self.p[j].isascii():
                hi_text += self.p[j]
                j += 1
        if j >= len(self.p) or self.p[j] != "}" or not (lo_text or hi_text):
            return None
        lo = int(lo_text) if lo_text else 0
        hi: int | None = (int(hi_text) if hi_text else None) if comma else lo
        if lo > MAX_REPEAT_COUNT or (hi is not None and hi > MAX_REPEAT_COUNT):
            raise self.fail(f"repetition counts are limited to {MAX_REPEAT_COUNT}")
        if hi is not None and lo > hi:
            raise self.fail("the minimum of a repetition is larger than its maximum")
        self.i = j + 1
        return lo, hi

    def repetition(self) -> _Node:
        atom = self.atom()
        start = self.i
        quant = self.quantifier()
        if quant is None:
            return atom
        if atom[0] == "assert":
            self.i = start
            raise self.fail("an anchor cannot be repeated")
        lazy = False
        if self.peek() == "?":
            lazy = True
            self.i += 1
        if self.peek() in ("*", "+", "?") or (self.peek() == "{" and self.braces() is not None):
            raise self.fail("stacked or possessive quantifiers are not supported")
        return ("rep", atom, quant[0], quant[1], lazy)

    def atom(self) -> _Node:
        c = self.p[self.i]
        if c == "(":
            return self.group()
        if c == "[":
            return self.char_class()
        if c == ".":
            self.i += 1
            return ("any",)
        if c == "^":
            self.i += 1
            return ("assert", _BOL)
        if c == "$":
            self.i += 1
            return ("assert", _EOL)
        if c == "\\":
            return self.escape()
        if c in "*+?":
            raise self.fail("nothing to repeat")
        if c == "{":
            save = self.i
            if self.braces() is not None:
                self.i = save
                raise self.fail("nothing to repeat")
            self.i = save
        self.i += 1
        return ("char", c)

    def group(self) -> _Node:
        self.i += 1  # (
        if self.peek() == "?":
            if self.p.startswith("?:", self.i):
                self.i += 2
            elif self.p.startswith("?P<", self.i):
                end = self.p.find(">", self.i)
                name = self.p[self.i + 3 : end] if end != -1 else ""
                if not name.isidentifier() or not name.isascii():
                    raise self.fail("invalid group name")
                self.i = end + 1
            else:
                raise self.fail(
                    "unsupported group syntax: look-around, atomic groups, conditionals and inline flags are not supported"
                )
        node = self.alternation()
        if self.peek() != ")":
            raise self.fail("missing closing parenthesis")
        self.i += 1
        return node

    def hex_escape(self, digits: int) -> str:
        chunk = self.p[self.i + 1 : self.i + 1 + digits]
        if len(chunk) != digits or any(ch not in _HEX for ch in chunk):
            raise self.fail("incomplete hexadecimal escape")
        value = int(chunk, 16)
        if value > 0x10FFFF:
            raise self.fail("escape is not a character")
        self.i += 1 + digits
        return chr(value)

    def escape_char(self) -> str:
        """After a backslash: the literal character it stands for. Letters and digits are errors unless defined."""
        self.i += 1
        if self.i >= len(self.p):
            raise self.fail("pattern ends with a backslash")
        c = self.p[self.i]
        if c in _SIMPLE_ESCAPES:
            self.i += 1
            return _SIMPLE_ESCAPES[c]
        if c == "x":
            return self.hex_escape(2)
        if c == "u":
            return self.hex_escape(4)
        if c == "U":
            return self.hex_escape(8)
        if c.isdecimal():
            raise self.fail("back-references are not supported")
        if c.isalpha() and c.isascii():
            raise self.fail(f"unsupported escape \\{c}")
        self.i += 1
        return c

    def escape(self) -> _Node:
        c = self.p[self.i + 1 : self.i + 2]
        if c in _CLASS_ESCAPES and c:
            self.i += 2
            return ("set", _CharClass((), c, False))
        if c in _ASSERT_ESCAPES and c:
            self.i += 2
            return ("assert", _ASSERT_ESCAPES[c])
        return ("char", self.escape_char())

    def class_item(self) -> str | tuple[str]:
        """A single character, or a one-element tuple holding a class letter (``d``, ``W``...)."""
        c = self.p[self.i]
        if c == "\\":
            nxt = self.p[self.i + 1 : self.i + 2]
            if nxt and nxt in _CLASS_ESCAPES:
                self.i += 2
                return (nxt,)
            if nxt == "b":
                raise self.fail("\\b inside a character class (backspace) is not supported")
            return self.escape_char()
        self.i += 1
        return c

    def char_class(self) -> _Node:
        self.i += 1  # [
        negated = self.peek() == "^"
        if negated:
            self.i += 1
        ranges: list[_Range] = []
        classes = ""
        first = True
        while True:
            if self.i >= len(self.p):
                raise self.fail("missing closing bracket")
            if self.p[self.i] == "]" and not first:
                self.i += 1
                break
            first = False
            if self.p.startswith("[", self.i) and self.p.startswith(("[:", "[.", "[="), self.i):
                raise self.fail("POSIX classes are not supported")
            item = self.class_item()
            if isinstance(item, tuple):
                classes += item[0]
                continue
            if self.peek() == "-" and self.p[self.i + 1 : self.i + 2] not in ("]", ""):
                self.i += 1
                end = self.class_item()
                if isinstance(end, tuple):
                    raise self.fail("a range cannot end with a class")
                if ord(end) < ord(item):
                    raise self.fail("bad character range")
                ranges.append((ord(item), ord(end)))
            else:
                ranges.append((ord(item), ord(item)))
        return ("set", _CharClass(tuple(ranges), classes, negated))


# --- compiling -------------------------------------------------------------------------------------------------


def _emit(node: _Node, out: list[list[object]]) -> None:
    if len(out) > MAX_PROGRAM:
        raise RegexError(f"pattern is too large (more than {MAX_PROGRAM} instructions after expanding repetitions)")
    kind = node[0]
    if kind == "char":
        out.append([_CHAR, node[1]])
    elif kind == "any":
        out.append([_ANY])
    elif kind == "set":
        out.append([_SET, node[1]])
    elif kind == "assert":
        out.append([_ASSERT, node[1]])
    elif kind == "cat":
        for item in node[1]:
            _emit(item, out)
    elif kind == "alt":
        branches = node[1]
        jumps: list[int] = []
        for index, branch in enumerate(branches):
            if index < len(branches) - 1:
                split = len(out)
                out.append([_SPLIT, split + 1, None])
                _emit(branch, out)
                jumps.append(len(out))
                out.append([_JMP, None])
                out[split][2] = len(out)
            else:
                _emit(branch, out)
        for jump in jumps:
            out[jump][1] = len(out)
    else:  # rep
        _, body, lo, hi, lazy = node
        for _ in range(lo):
            _emit(body, out)
        if hi is None:
            loop = len(out)
            out.append([_SPLIT, None, None])
            _emit(body, out)
            out.append([_JMP, loop])
            end = len(out)
            out[loop][1:] = [end, loop + 1] if lazy else [loop + 1, end]
        else:
            splits: list[int] = []
            for _ in range(hi - lo):
                splits.append(len(out))
                out.append([_SPLIT, None, None])
                _emit(body, out)
            end = len(out)
            for split in splits:
                out[split][1:] = [end, split + 1] if lazy else [split + 1, end]


# --- matching --------------------------------------------------------------------------------------------------


def _is_word(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


@dataclass(frozen=True, slots=True)
class Program:
    """A compiled pattern. Immutable and safe to share between threads."""

    source: str
    insts: tuple[tuple[object, ...], ...] = field(repr=False)
    ignorecase: bool
    multiline: bool
    dotall: bool

    # -- single character tests --

    def _class_match(self, spec: _CharClass, ch: str) -> bool:
        candidates = (ch, ch.lower(), ch.upper()) if self.ignorecase else (ch,)
        hit = False
        for cand in candidates:
            code = ord(cand)
            if any(lo <= code <= hi for lo, hi in spec.ranges):
                hit = True
                break
        if not hit:
            for letter in spec.classes:
                if (
                    (letter == "d" and ch.isdecimal())
                    or (letter == "D" and not ch.isdecimal())
                    or (letter == "w" and _is_word(ch))
                    or (letter == "W" and not _is_word(ch))
                    or (letter == "s" and ch.isspace())
                    or (letter == "S" and not ch.isspace())
                ):
                    hit = True
                    break
        return hit != spec.negated

    def _consumes(self, inst: tuple[object, ...], ch: str) -> bool:
        op = inst[0]
        if op == _CHAR:
            literal = inst[1]
            assert isinstance(literal, str)
            return ch == literal or (self.ignorecase and ch.lower() == literal.lower())
        if op == _ANY:
            return self.dotall or ch != "\n"
        spec = inst[1]
        assert isinstance(spec, _CharClass)
        return self._class_match(spec, ch)

    def _assertion(self, kind: object, text: str, pos: int) -> bool:
        n = len(text)
        if kind == _BOL:
            return pos == 0 or (self.multiline and text[pos - 1] == "\n")
        if kind == _EOL:
            return pos == n or (pos == n - 1 and text[pos] == "\n") or (self.multiline and text[pos] == "\n")
        if kind == _BOS:
            return pos == 0
        if kind == _EOS:
            return pos == n
        before = pos > 0 and _is_word(text[pos - 1])
        after = pos < n and _is_word(text[pos])
        if kind == _WORDB:
            return before != after
        return n > 0 and before == after  # like re, \B never matches the empty string

    def _closure(self, pc: int, start: int, pos: int, text: str, seen: set[int], out: list[tuple[int, int]]) -> None:
        stack = [pc]
        insts = self.insts
        while stack:
            pc = stack.pop()
            if pc in seen:
                continue
            seen.add(pc)
            inst = insts[pc]
            op = inst[0]
            if op == _JMP:
                stack.append(inst[1])  # type: ignore[arg-type]
            elif op == _SPLIT:
                stack.append(inst[2])  # type: ignore[arg-type]
                stack.append(inst[1])  # type: ignore[arg-type]
            elif op == _ASSERT:
                if self._assertion(inst[1], text, pos):
                    stack.append(pc + 1)
            else:
                out.append((pc, start))

    def _scan(self, text: str, begin: int, budget: _Budget, *, full: bool, first_only: bool) -> tuple[int, int] | None:
        """Leftmost-first match starting at or after ``begin`` (only at ``begin`` when ``full``)."""
        n = len(text)
        insts = self.insts
        seen: set[int] = set()
        current: list[tuple[int, int]] = []
        self._closure(0, begin, begin, text, seen, current)
        best: tuple[int, int] | None = None
        pos = begin
        while True:
            budget.left -= 1 + len(current)
            if budget.left < 0:
                raise BudgetExceeded
            following: list[tuple[int, int]] = []
            following_seen: set[int] = set()
            for pc, start in current:
                inst = insts[pc]
                if inst[0] == _MATCH:
                    if full and pos != n:
                        continue
                    best = (start, pos)
                    if first_only:
                        return best
                    break  # every thread after this one has lower priority: drop them
                if pos < n and self._consumes(inst, text[pos]):
                    self._closure(pc + 1, start, pos + 1, text, following_seen, following)
            if pos >= n:
                return best
            pos += 1
            if best is None and not full:
                self._closure(0, pos, pos, text, following_seen, following)
            if not following and (best is not None or full):
                return best
            current = following

    # -- public API --

    def matches(self, text: str) -> bool:
        """Whether the pattern matches anywhere in ``text``."""
        return self._scan(text, 0, _Budget(), full=False, first_only=True) is not None

    def fullmatch(self, text: str) -> bool:
        """Whether the pattern matches the whole of ``text``."""
        return self._scan(text, 0, _Budget(), full=True, first_only=True) is not None

    def first_span(self, text: str) -> tuple[int, int] | None:
        """The leftmost-first match, as ``re.search(...).span()`` would give it."""
        return self._scan(text, 0, _Budget(), full=False, first_only=False)

    def spans(self, text: str) -> list[tuple[int, int]]:
        """Non-overlapping, non-empty matches from left to right, with one step budget for the whole call."""
        budget = _Budget()
        found: list[tuple[int, int]] = []
        position = 0
        while position <= len(text):
            match = self._scan(text, position, budget, full=False, first_only=False)
            if match is None:
                break
            start, end = match
            if end > start:
                found.append(match)
                position = end
            else:
                position = start + 1
        return found


def compile_pattern(pattern: str, flags: tuple[str, ...] = ()) -> Program:
    """Compile ``pattern`` (flags: ``i``, ``m``, ``s``). Raises ``RegexError`` with the reason."""
    if not pattern:
        raise RegexError("must not be empty")
    for flag in flags:
        if flag not in ("i", "m", "s"):
            raise RegexError(f"unknown flag {flag!r} (allowed: i, m, s)")
    tree = _Parser(pattern).parse()
    out: list[list[object]] = []
    _emit(tree, out)
    out.append([_MATCH])
    if len(out) > MAX_PROGRAM + 1:
        raise RegexError(f"pattern is too large (more than {MAX_PROGRAM} instructions after expanding repetitions)")
    return Program(pattern, tuple(tuple(i) for i in out), "i" in flags, "m" in flags, "s" in flags)
