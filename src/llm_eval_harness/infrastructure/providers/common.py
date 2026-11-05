"""What every adapter does with the bytes it received from the user's pipeline."""

from __future__ import annotations

import dataclasses

from llm_eval_harness.domain.errors import ProviderError
from llm_eval_harness.domain.jsonio import JsonParseError, JsonValue, parse_json
from llm_eval_harness.domain.model import ProviderOutput
from llm_eval_harness.domain.protocol import parse_response
from llm_eval_harness.domain.text import one_line


def decode_json(raw: bytes, *, source: str) -> JsonValue:
    """UTF-8 decode and strictly parse. The error never echoes the content (it is untrusted)."""
    if not raw.strip():
        raise ProviderError("protocol", f"{source} produced no output", retryable=False)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ProviderError("protocol", f"{source} output is not UTF-8 text (binary data?)", retryable=False) from None
    try:
        return parse_json(text)
    except JsonParseError as error:
        raise ProviderError("protocol", f"{source} output is not valid JSON: {error}", retryable=False) from None


def decode_answer(raw: bytes, *, source: str, latency_ms: float) -> ProviderOutput:
    """Bytes in, validated ``ProviderOutput`` out, with the measured latency attached."""
    output = parse_response(decode_json(raw, source=source))
    return dataclasses.replace(output, latency_ms=latency_ms)


def stderr_excerpt(raw: bytes, max_chars: int = 300) -> str:
    """The tail of a program's stderr as one safe line, for an error message."""
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        return "(no stderr)"
    tail = text[-max_chars:]
    return one_line(tail, max_chars=max_chars)
