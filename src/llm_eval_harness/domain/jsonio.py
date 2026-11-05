"""Strict, bounded JSON parsing of untrusted text. Pure: operates on strings only."""

from __future__ import annotations

import json
import math

from llm_eval_harness.domain.limits import MAX_JSON_DEPTH

type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]


class JsonParseError(Exception):
    def __init__(self, message: str, line: int = 1, column: int = 1) -> None:
        self.message = message
        self.line = line
        self.column = column
        super().__init__(message)

    def __str__(self) -> str:
        return f"{self.message} (line {self.line}, column {self.column})"


def kind_of(value: object) -> str:
    """The JSON type name of a value, used in error messages."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def nesting_depth(text: str) -> int:
    """Deepest nesting of arrays and objects in JSON text, found without recursion (strings are skipped)."""
    depth = deepest = 0
    in_string = escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "[{":
            depth += 1
            deepest = max(deepest, depth)
        elif ch in "]}":
            depth -= 1
    return deepest


def _no_constants(name: str) -> float:
    raise ValueError(f"{name} is not valid JSON")


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("number is out of range")
    return value


def _unique_pairs(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {key!r}")
        result[key] = value
    return result


def parse_json(text: str, *, max_depth: int = MAX_JSON_DEPTH) -> JsonValue:
    """Parse JSON text, rejecting what other readers would disagree about.

    Rejected: nesting deeper than ``max_depth``, duplicate object keys, NaN/Infinity, numbers outside the
    float range and integers with absurdly many digits. Anything else the standard parser accepts.
    """
    if nesting_depth(text) > max_depth:
        raise JsonParseError(f"nesting is deeper than {max_depth} levels")
    try:
        value: JsonValue = json.loads(
            text,
            parse_constant=_no_constants,
            parse_float=_finite_float,
            object_pairs_hook=_unique_pairs,
        )
    except json.JSONDecodeError as error:
        raise JsonParseError(error.msg, error.lineno, error.colno) from None
    except RecursionError:
        raise JsonParseError(f"nesting is deeper than {max_depth} levels") from None
    except ValueError as error:
        raise JsonParseError(str(error)) from None
    return value


def canonical(value: JsonValue) -> str:
    """The canonical serialisation used for hashing: sorted keys, no whitespace, ASCII only."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def json_equal(a: JsonValue, b: JsonValue) -> bool:
    """Equality by JSON type: ``true`` is not ``1``; ``1`` equals ``1.0``; objects ignore key order."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(json_equal(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(json_equal(v, b[k]) for k, v in a.items())
    return type(a) is type(b) and a == b


def contains_json(items: list[JsonValue], value: JsonValue) -> bool:
    return any(json_equal(item, value) for item in items)
