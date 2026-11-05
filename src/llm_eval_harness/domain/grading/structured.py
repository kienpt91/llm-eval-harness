"""json_schema: validate structured output against a small, documented subset of JSON Schema."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from llm_eval_harness.domain.errors import Problem
from llm_eval_harness.domain.grading.params import Params
from llm_eval_harness.domain.grading.regexsafe import (
    BUDGET_DETAIL,
    BudgetExceeded,
    check_pattern,
    pattern_matches,
    refusal_detail,
    too_long_for_regex,
)
from llm_eval_harness.domain.jsonio import JsonParseError, JsonValue, contains_json, json_equal, kind_of, parse_json
from llm_eval_harness.domain.model import GradeContext, Outcome
from llm_eval_harness.domain.text import one_line

TYPES = ("string", "number", "integer", "boolean", "null", "object", "array")
KEYWORDS = frozenset(
    {
        "type", "enum", "const", "properties", "required", "additionalProperties", "items", "minItems", "maxItems",
        "minLength", "maxLength", "minimum", "maximum", "pattern", "anyOf", "description", "title",
    }
)  # fmt: skip
MAX_SCHEMA_DEPTH = 16
MAX_ANY_OF = 8


def check_schema(schema: JsonValue, path: str, problems: list[Problem], depth: int = 0) -> None:
    """Validate the schema itself: known keywords only, right value types, bounded depth."""
    if depth > MAX_SCHEMA_DEPTH:
        problems.append(Problem(path, f"schema is nested deeper than {MAX_SCHEMA_DEPTH} levels"))
        return
    if schema is True or schema is False:
        return
    if not isinstance(schema, dict):
        problems.append(Problem(path, f"schema must be an object or a boolean (got {kind_of(schema)})"))
        return
    p = Params(schema, path, problems, KEYWORDS)
    type_value = schema.get("type")
    if type_value is not None:
        names = [type_value] if isinstance(type_value, str) else type_value
        if not isinstance(names, list) or not names or not all(isinstance(n, str) and n in TYPES for n in names):
            p.fail("type", f"must be one of {', '.join(TYPES)} or a list of them")
    if "enum" in schema and (not isinstance(schema["enum"], list) or not schema["enum"]):
        p.fail("enum", "must be a non-empty array")
    p.number("minItems", minimum=0, integer=True)
    p.number("maxItems", minimum=0, integer=True)
    p.number("minLength", minimum=0, integer=True)
    p.number("maxLength", minimum=0, integer=True)
    p.number("minimum")
    p.number("maximum")
    p.strings("required")
    pattern = p.string("pattern", max_len=512)
    if pattern is not None:
        reason = check_pattern(pattern)
        if reason:
            p.fail("pattern", reason)
    properties = p.obj("properties")
    for name, sub in (properties or {}).items():
        check_schema(sub, f"{path}.properties.{name}", problems, depth + 1)
    additional = schema.get("additionalProperties")
    if additional is not None and not isinstance(additional, bool):
        p.fail("additionalProperties", "must be true or false")
    if "items" in schema:
        check_schema(schema["items"], f"{path}.items", problems, depth + 1)
    any_of = p.array("anyOf")
    if any_of is not None:
        if not any_of or len(any_of) > MAX_ANY_OF:
            p.fail("anyOf", f"needs between 1 and {MAX_ANY_OF} schemas")
        for index, sub in enumerate(any_of):
            check_schema(sub, f"{path}.anyOf[{index}]", problems, depth + 1)


def _type_ok(value: JsonValue, name: str) -> bool:
    if name == "string":
        return isinstance(value, str)
    if name == "boolean":
        return isinstance(value, bool)
    if name == "null":
        return value is None
    if name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if name == "integer":
        return not isinstance(value, bool) and (
            isinstance(value, int) or (isinstance(value, float) and value.is_integer())
        )
    if name == "object":
        return isinstance(value, dict)
    return isinstance(value, list)


def violations(value: JsonValue, schema: JsonValue, path: str = "$", limit: int = 20) -> list[str]:
    """Every way ``value`` breaks ``schema`` (at most ``limit``), each with a JSON-path-like location."""
    out: list[str] = []
    _walk(value, schema, path, out, limit)
    return out


def _walk(value: JsonValue, schema: JsonValue, path: str, out: list[str], limit: int) -> None:
    if len(out) >= limit or schema is True:
        return
    if schema is False:
        out.append(f"{path}: no value is allowed here")
        return
    if not isinstance(schema, dict):
        return

    def bad(message: str) -> None:
        if len(out) < limit:
            out.append(f"{path}: {message}")

    if "type" in schema:
        names = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(isinstance(n, str) and _type_ok(value, n) for n in names):
            bad(f"expected {' or '.join(str(n) for n in names)}, got {kind_of(value)}")
            return
    if "enum" in schema and isinstance(schema["enum"], list) and not contains_json(schema["enum"], value):
        bad(f"value is not one of the {len(schema['enum'])} allowed values")
    if "const" in schema and not json_equal(value, schema["const"]):
        bad("value differs from the required constant")
    if "anyOf" in schema and isinstance(schema["anyOf"], list):
        if not any(not violations(value, sub, path, limit=1) for sub in schema["anyOf"]):
            bad("matches none of the allowed alternatives")
    if isinstance(value, str):
        _check_string(value, schema, bad)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        lo, hi = schema.get("minimum"), schema.get("maximum")
        if isinstance(lo, (int, float)) and value < lo:
            bad(f"{value} is below the minimum {lo}")
        if isinstance(hi, (int, float)) and value > hi:
            bad(f"{value} is above the maximum {hi}")
    if isinstance(value, list):
        _check_array(value, schema, path, out, limit, bad)
    if isinstance(value, dict):
        _check_object(value, schema, path, out, limit, bad)


def _check_string(value: str, schema: Mapping[str, JsonValue], bad: Callable[[str], None]) -> None:
    lo, hi, pattern = schema.get("minLength"), schema.get("maxLength"), schema.get("pattern")
    if isinstance(lo, int) and len(value) < lo:
        bad(f"string is shorter than {lo}")
    if isinstance(hi, int) and len(value) > hi:
        bad(f"string is longer than {hi}")
    if isinstance(pattern, str):
        if too_long_for_regex(value):
            bad(refusal_detail("string"))
        else:
            try:
                if not pattern_matches(pattern, (), value):
                    bad("string does not match the required pattern")
            except BudgetExceeded:
                bad(BUDGET_DETAIL)


def _check_array(
    value: list[JsonValue],
    schema: Mapping[str, JsonValue],
    path: str,
    out: list[str],
    limit: int,
    bad: Callable[[str], None],
) -> None:
    lo, hi = schema.get("minItems"), schema.get("maxItems")
    if isinstance(lo, int) and len(value) < lo:
        bad(f"array has fewer than {lo} items")
    if isinstance(hi, int) and len(value) > hi:
        bad(f"array has more than {hi} items")
    if "items" in schema:
        for index, item in enumerate(value):
            if len(out) >= limit:
                break
            _walk(item, schema["items"], f"{path}[{index}]", out, limit)


def _check_object(
    value: dict[str, JsonValue],
    schema: Mapping[str, JsonValue],
    path: str,
    out: list[str],
    limit: int,
    bad: Callable[[str], None],
) -> None:
    required = schema.get("required")
    if isinstance(required, list):
        for name in required:
            if isinstance(name, str) and name not in value:
                bad(f"missing required property {name!r}")
    properties = schema.get("properties")
    props = properties if isinstance(properties, dict) else {}
    for name, sub in props.items():
        if name in value:
            _walk(value[name], sub, f"{path}.{name}", out, limit)
    if schema.get("additionalProperties") is False:
        extra = sorted(k for k in value if k not in props)
        if extra:
            bad(f"unexpected properties: {', '.join(repr(one_line(e, max_chars=30)) for e in extra[:5])}")


@dataclass(frozen=True, slots=True)
class SchemaConfig:
    schema: JsonValue
    source: str


def parse_json_schema(raw: Mapping[str, JsonValue], path: str, problems: list[Problem]) -> SchemaConfig:
    p = Params(raw, path, problems, {"schema", "source"})
    schema = p.obj("schema", required=True)
    if schema is not None:
        check_schema(schema, f"{path}.schema", problems)
    return SchemaConfig(
        schema if schema is not None else {}, p.choice("source", ("auto", "structured", "text"), default="auto")
    )


def grade_json_schema(config: SchemaConfig, ctx: GradeContext) -> Outcome:
    """Passes when the structured output (or the answer text parsed as JSON) satisfies the schema.

    Supported keywords: type, enum, const, properties, required, additionalProperties (boolean), items,
    minItems, maxItems, minLength, maxLength, minimum, maximum, pattern, anyOf. Failure modes: it checks shape,
    not truth (a well-formed wrong answer passes); ``$ref``, ``oneOf``, ``format`` and the rest of JSON Schema
    are refused at load time rather than ignored. Text that is not valid JSON, or is nested deeper than the
    parser limit, fails.
    """
    value: JsonValue
    if config.source == "structured" or (config.source == "auto" and ctx.output.has_structured):
        if not ctx.output.has_structured:
            return Outcome(False, 0.0, "provider returned no structured output")
        value = ctx.output.structured
    else:
        try:
            value = parse_json(ctx.output.text)
        except JsonParseError as error:
            return Outcome(False, 0.0, f"answer is not valid JSON: {error}")
    found = violations(value, config.schema)
    if not found:
        return Outcome(True, 1.0, "output satisfies the schema")
    return Outcome(False, 0.0, f"{len(found)} schema violation(s): " + "; ".join(found[:3]))
