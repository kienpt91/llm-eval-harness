"""Redaction of what is stored: secret-shaped strings in outputs, inputs, tool arguments and messages."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from llm_eval_harness.domain.errors import UsageError
from llm_eval_harness.domain.grading.regexsafe import BudgetExceeded, check_pattern, compiled
from llm_eval_harness.domain.jsonio import JsonValue, canonical
from llm_eval_harness.domain.limits import MAX_REDACT_CHARS
from llm_eval_harness.domain.shapes import redact
from llm_eval_harness.domain.text import truncate

MAX_REDACT_DEPTH = 64
UNEVALUATED = "[REDACTED: a redaction pattern could not be evaluated within its work budget]"

type _Finder = Callable[[str], list[tuple[int, int]]]


@dataclass(frozen=True, slots=True)
class Redactor:
    """Replaces secret-shaped strings (and the configured extra patterns) before anything is stored."""

    enabled: bool = True
    patterns: tuple[str, ...] = field(default_factory=tuple)

    @staticmethod
    def build(enabled: bool, patterns: Sequence[str]) -> Redactor:
        for pattern in patterns:
            reason = check_pattern(pattern)
            if reason:
                raise UsageError(f"redact pattern {pattern[:40]!r} {reason}")
        return Redactor(enabled, tuple(patterns))

    def _finders(self) -> list[_Finder]:
        return [compiled(pattern).spans for pattern in self.patterns]

    def text(self, value: str) -> str:
        """Redacted text. Only the first ``MAX_REDACT_CHARS`` characters are looked at and kept; the rest is
        replaced by a count, so no pattern runs on unbounded text and nothing unexamined is stored. A custom pattern
        that uses up its work budget makes the whole string unusable: it is replaced, never stored as is."""
        if not self.enabled:
            return value
        head = value if len(value) <= MAX_REDACT_CHARS else value[:MAX_REDACT_CHARS]
        try:
            out = redact(head, self._finders())
        except BudgetExceeded:
            return UNEVALUATED
        return out if head is value else out + f"…[{len(value) - MAX_REDACT_CHARS} more characters]"

    def json(self, value: JsonValue, depth: int = 0) -> JsonValue:
        if not self.enabled:
            return value
        if depth > MAX_REDACT_DEPTH:
            return "[TRUNCATED]"
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, list):
            return [self.json(v, depth + 1) for v in value]
        if isinstance(value, dict):
            return {self.text(k): self.json(v, depth + 1) for k, v in value.items()}
        return value


def bounded_json(value: dict[str, JsonValue], max_chars: int) -> dict[str, JsonValue]:
    """The value itself when its JSON form is short enough, otherwise a marker with a truncated preview."""
    text = canonical(value)
    if len(text) <= max_chars:
        return value
    return {"_truncated": truncate(text, max_chars)}
