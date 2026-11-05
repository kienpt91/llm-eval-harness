"""Redaction of what is stored: secret-shaped strings in outputs, inputs, tool arguments and messages."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from llm_eval_harness.domain.errors import UsageError
from llm_eval_harness.domain.grading.regexsafe import check_pattern
from llm_eval_harness.domain.jsonio import JsonValue, canonical
from llm_eval_harness.domain.limits import MAX_REDACT_CHARS
from llm_eval_harness.domain.shapes import redact
from llm_eval_harness.domain.text import truncate

MAX_REDACT_DEPTH = 64


@dataclass(frozen=True, slots=True)
class Redactor:
    """Replaces secret-shaped strings (and the configured extra patterns) before anything is stored."""

    enabled: bool = True
    patterns: tuple[re.Pattern[str], ...] = field(default_factory=tuple)

    @staticmethod
    def build(enabled: bool, patterns: Sequence[str]) -> Redactor:
        compiled: list[re.Pattern[str]] = []
        for pattern in patterns:
            reason = check_pattern(pattern)
            if reason:
                raise UsageError(f"redact pattern {pattern[:40]!r} {reason}")
            compiled.append(re.compile(pattern))
        return Redactor(enabled, tuple(compiled))

    def text(self, value: str) -> str:
        """Redacted text. Only the first ``MAX_REDACT_CHARS`` characters are looked at and kept; the rest is
        replaced by a count, so no pattern ever runs on unbounded text and nothing unexamined is stored."""
        if not self.enabled:
            return value
        if len(value) <= MAX_REDACT_CHARS:
            return redact(value, self.patterns)
        return redact(value[:MAX_REDACT_CHARS], self.patterns) + f"…[{len(value) - MAX_REDACT_CHARS} more characters]"

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
