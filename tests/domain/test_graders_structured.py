from __future__ import annotations

import json

import pytest

from llm_eval_harness.domain.errors import Problem
from llm_eval_harness.domain.grading.structured import check_schema, violations
from support.builders import grade, js, output, validate

SCHEMA = {
    "type": "object",
    "required": ["name", "age"],
    "additionalProperties": False,
    "properties": {
        "name": {"type": "string", "minLength": 1, "maxLength": 5},
        "age": {"type": "integer", "minimum": 0, "maximum": 150},
        "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 2},
        "kind": {"enum": ["a", "b", 1]},
        "code": {"type": "string", "pattern": "^[A-Z]{2}$"},
        "any": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    },
}


def test_valid_document_passes() -> None:
    doc = {"name": "Kim", "age": 30, "tags": ["x"], "kind": "a", "code": "AB", "any": None}
    assert violations(js(doc), js(SCHEMA)) == []


@pytest.mark.parametrize(
    ("doc", "fragment"),
    [
        ({"age": 3}, "missing required property 'name'"),
        ({"name": "Kim", "age": "3"}, "$.age: expected integer, got string"),
        ({"name": "Kim", "age": 3.5}, "$.age: expected integer, got number"),
        ({"name": "Kim", "age": True}, "$.age: expected integer, got boolean"),
        ({"name": "Kimberly", "age": 3}, "string is longer than 5"),
        ({"name": "", "age": 3}, "string is shorter than 1"),
        ({"name": "Kim", "age": -1}, "below the minimum"),
        ({"name": "Kim", "age": 200}, "above the maximum"),
        ({"name": "Kim", "age": 3, "tags": []}, "fewer than 1 items"),
        ({"name": "Kim", "age": 3, "tags": ["a", "b", "c"]}, "more than 2 items"),
        ({"name": "Kim", "age": 3, "tags": [1]}, "$.tags[0]: expected string"),
        ({"name": "Kim", "age": 3, "kind": True}, "not one of the 3 allowed"),
        ({"name": "Kim", "age": 3, "code": "abc"}, "does not match the required pattern"),
        ({"name": "Kim", "age": 3, "any": 5}, "matches none of the allowed alternatives"),
        ({"name": "Kim", "age": 3, "extra": 1}, "unexpected properties: 'extra'"),
        ([], "$: expected object, got array"),
    ],
)
def test_violations_say_where_and_why(doc: object, fragment: str) -> None:
    found = violations(js(doc), js(SCHEMA))
    assert any(fragment in line for line in found), found


def test_boolean_schemas_const_and_float_integers() -> None:
    assert violations(1, True) == []
    assert violations(1, False) == ["$: no value is allowed here"]
    assert violations(2.0, {"type": "integer"}) == []
    assert violations("a", {"const": "b"}) == ["$: value differs from the required constant"]
    assert violations(1, {"type": ["string", "number"]}) == []
    assert violations(None, {"type": ["string", "number"]})[0].startswith("$: expected string or number")


def test_violation_list_is_bounded() -> None:
    assert len(violations(list(range(1000)), {"items": {"type": "string"}}, limit=20)) == 20


def test_grade_uses_structured_output_when_present_else_parses_text() -> None:
    params = {"schema": {"type": "object", "required": ["a"]}}
    assert grade("json_schema", params, output("ignored", structured={"a": 1}, has_structured=True)).passed
    assert grade("json_schema", params, output(json.dumps({"a": 1}))).passed
    bad = grade("json_schema", params, output("not json"))
    assert not bad.passed and "not valid JSON" in bad.detail
    assert not grade("json_schema", {**params, "source": "structured"}, output("{}")).passed
    assert not grade(
        "json_schema", {**params, "source": "text"}, output("x", structured={"a": 1}, has_structured=True)
    ).passed


def test_deeply_nested_or_hostile_text_fails_instead_of_crashing() -> None:
    params = {"schema": {"type": "array"}}
    assert not grade("json_schema", params, output("[" * 100_000 + "]" * 100_000)).passed
    assert not grade("json_schema", params, output('{"a": 1, "a": 2}')).passed
    assert not grade("json_schema", params, output("NaN")).passed


def test_schema_itself_is_validated_with_paths() -> None:
    def problems(schema: object) -> list[str]:
        out: list[Problem] = []
        check_schema(schema, "s", out)  # type: ignore[arg-type]
        return [p.render() for p in out]

    assert problems({"type": "strng"}) == [
        "s.type: must be one of string, number, integer, boolean, null, object, array or a list of them"
    ]
    assert any("unknown key" in p for p in problems({"$ref": "#/x"}))
    assert any("unknown key" in p for p in problems({"oneOf": []}))
    assert problems({"properties": {"a": {"minLength": -1}}}) == [
        "s.properties.a.minLength: must be at least 0 (got -1)"
    ]
    assert problems({"pattern": "(a+)+"})[0].startswith("s.pattern: repeats a group")
    assert problems({"anyOf": []}) == ["s.anyOf: needs between 1 and 8 schemas"]
    assert problems({"additionalProperties": {}}) == ["s.additionalProperties: must be true or false"]
    assert problems(5) == ["s: schema must be an object or a boolean (got number)"]
    assert problems({"enum": []}) == ["s.enum: must be a non-empty array"]
    deep: dict[str, object] = {"type": "object"}
    for _ in range(20):
        deep = {"items": deep}
    assert any("deeper than" in p for p in problems(deep))
    assert validate("json_schema", {}, "g")[0].render() == "g.schema: is required"
