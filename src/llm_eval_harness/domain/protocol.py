"""The wire protocol between the harness and a user's pipeline (``command`` and ``http`` providers).

Request (what the harness sends; JSON on stdin or in the POST body). Expectations and graders are NOT sent:
a pipeline must not be able to read the answer key.

    {"protocol": 1, "case": {"id", "category", "tags", "input", "context": [{"id", "text", "metadata"}], "metadata"},
     "repeat": 0, "attempt": 1, "seed": 7}

Response (what the pipeline writes; one JSON object):

    {"text": "...",                                   required
     "structured": <any JSON>,                         optional: parsed output for json_schema / set_overlap
     "tool_calls": [{"name": "...", "arguments": {}}], optional: every side effect the pipeline performed or requested
     "citations": ["doc-id"], "retrieved": ["doc-id"], optional
     "usage": {"model": "name", "input_tokens": 12, "output_tokens": 30},   optional
     "metadata": {}}                                   optional, ignored
"""

from __future__ import annotations

from llm_eval_harness.domain.errors import ProviderError
from llm_eval_harness.domain.jsonio import JsonValue, kind_of
from llm_eval_harness.domain.model import Case, ProviderOutput, ToolCall, Usage
from llm_eval_harness.domain.text import one_line

PROTOCOL_VERSION = 1
RESPONSE_KEYS = ("text", "structured", "tool_calls", "citations", "retrieved", "usage", "metadata")
MAX_TOOL_CALLS = 256
MAX_IDS = 1024
MAX_TOKENS_REPORTED = 10**12


def build_request(case: Case, *, repeat: int, attempt: int, seed: int | None) -> dict[str, JsonValue]:
    """The request for one run of a case. Contains the input and context only, never the expectations."""
    context: list[JsonValue] = [{"id": d.id, "text": d.text, "metadata": dict(d.metadata)} for d in case.context]
    body: dict[str, JsonValue] = {
        "id": case.id,
        "category": case.category,
        "tags": list(case.tags),
        "input": case.input,
        "context": context,
        "metadata": dict(case.metadata),
    }
    return {"protocol": PROTOCOL_VERSION, "case": body, "repeat": repeat, "attempt": attempt, "seed": seed}


def _bad(message: str) -> ProviderError:
    return ProviderError("protocol", message, retryable=False)


def _id_list(obj: dict[str, JsonValue], key: str) -> tuple[str, ...]:
    value = obj.get(key, [])
    if not isinstance(value, list):
        raise _bad(f'"{key}" must be an array of strings (got {kind_of(value)})')
    if len(value) > MAX_IDS:
        raise _bad(f'"{key}" has more than {MAX_IDS} entries')
    out: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or len(item) > 256:
            raise _bad(f'"{key}[{index}]" must be a string of at most 256 characters')
        out.append(item)
    return tuple(out)


def _tool_calls(obj: dict[str, JsonValue]) -> tuple[ToolCall, ...]:
    value = obj.get("tool_calls", [])
    if not isinstance(value, list):
        raise _bad(f'"tool_calls" must be an array (got {kind_of(value)})')
    if len(value) > MAX_TOOL_CALLS:
        raise _bad(f'"tool_calls" has more than {MAX_TOOL_CALLS} entries')
    out: list[ToolCall] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise _bad(f'"tool_calls[{index}]" must be an object with name and arguments (got {kind_of(item)})')
        name = item.get("name")
        if not isinstance(name, str) or not name.strip() or len(name) > 200:
            raise _bad(f'"tool_calls[{index}].name" must be a non-empty string of at most 200 characters')
        arguments = item.get("arguments", {})
        if not isinstance(arguments, dict):
            raise _bad(f'"tool_calls[{index}].arguments" must be an object (got {kind_of(arguments)})')
        extra = sorted(set(item) - {"name", "arguments"})
        if extra:
            raise _bad(
                f'"tool_calls[{index}]" has unknown key {one_line(extra[0], max_chars=40)!r} (use name and arguments)'
            )
        out.append(ToolCall(name, arguments))
    return tuple(out)


def _count(usage: dict[str, JsonValue], key: str) -> int:
    value = usage.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_TOKENS_REPORTED:
        raise _bad(f'"usage.{key}" must be a non-negative whole number')
    return value


def _usage(obj: dict[str, JsonValue]) -> Usage | None:
    value = obj.get("usage")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise _bad(f'"usage" must be an object (got {kind_of(value)})')
    extra = sorted(set(value) - {"model", "input_tokens", "output_tokens"})
    if extra:
        raise _bad(f'"usage" has unknown key {one_line(extra[0], max_chars=40)!r}')
    model = value.get("model")
    if model is not None and (not isinstance(model, str) or not model or len(model) > 128):
        raise _bad('"usage.model" must be a non-empty string of at most 128 characters')
    return Usage(model, _count(value, "input_tokens"), _count(value, "output_tokens"))


def parse_response(obj: JsonValue) -> ProviderOutput:
    """Validate a pipeline's answer and turn it into a ``ProviderOutput`` (latency is added by the adapter)."""
    if not isinstance(obj, dict):
        raise _bad(f"the response must be a JSON object (got {kind_of(obj)})")
    unknown = sorted(set(obj) - set(RESPONSE_KEYS))
    if unknown:
        raise _bad(
            f"unknown key {one_line(unknown[0], max_chars=40)!r} in the response (allowed: {', '.join(RESPONSE_KEYS)})"
        )
    text = obj.get("text")
    if not isinstance(text, str):
        raise _bad(f'the response needs "text" as a string (got {kind_of(text)})')
    return ProviderOutput(
        text=text,
        structured=obj.get("structured"),
        has_structured="structured" in obj,
        tool_calls=_tool_calls(obj),
        citations=_id_list(obj, "citations"),
        retrieved=_id_list(obj, "retrieved"),
        usage=_usage(obj),
    )


def build_judge_request(case: Case, output: ProviderOutput, rubric: str) -> dict[str, JsonValue]:
    """The request to a judge program: the case input, the answer, the facts it should contain and a rubric."""
    calls: list[JsonValue] = [{"name": c.name, "arguments": dict(c.arguments)} for c in output.tool_calls]
    return {
        "protocol": PROTOCOL_VERSION,
        "kind": "judge",
        "case": {"id": case.id, "category": case.category, "input": case.input},
        "expected": {"answers": list(case.expected.answers), "facts": list(case.expected.facts)},
        "output": {"text": output.text, "tool_calls": calls, "citations": list(output.citations)},
        "rubric": rubric,
    }
