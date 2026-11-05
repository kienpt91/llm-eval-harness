from __future__ import annotations

import pytest

from llm_eval_harness.domain.jsonio import (
    JsonParseError,
    canonical,
    contains_json,
    json_equal,
    kind_of,
    nesting_depth,
    parse_json,
)
from support.prng import Prng


def test_parses_plain_json() -> None:
    assert parse_json('{"a": [1, 2.5, "x", null, true]}') == {"a": [1, 2.5, "x", None, True]}


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ('{"a": 1, "a": 2}', "duplicate key"),
        ("NaN", "not valid JSON"),
        ("Infinity", "not valid JSON"),
        ("-Infinity", "not valid JSON"),
        ("1e999", "out of range"),
        ('{"a": ', "Expecting value"),
        ("[1,]", "line 1"),
        ("1" * 5000, "Exceeds the limit"),
    ],
)
def test_rejects_what_readers_disagree_about(text: str, fragment: str) -> None:
    with pytest.raises(JsonParseError) as error:
        parse_json(text)
    assert fragment.lower() in str(error.value).lower()


def test_error_carries_position() -> None:
    with pytest.raises(JsonParseError) as error:
        parse_json('{\n  "a": oops\n}')
    assert (error.value.line, error.value.column) == (2, 8)
    assert "line 2, column 8" in str(error.value)


def test_deep_nesting_is_refused_before_the_parser_recurses() -> None:
    text = "[" * 5000 + "]" * 5000
    with pytest.raises(JsonParseError, match="deeper than 32"):
        parse_json(text)
    parse_json("[" * 32 + "]" * 32)
    with pytest.raises(JsonParseError):
        parse_json("[" * 33 + "]" * 33)


def test_depth_ignores_brackets_inside_strings() -> None:
    assert nesting_depth('["[[[[", "]]]]", "\\"[["]') == 1
    assert nesting_depth('{"a": {"b": [1]}}') == 3


def test_prototype_pollution_keys_are_plain_data() -> None:
    value = parse_json('{"__proto__": {"admin": true}, "constructor": 1, "prototype": 2}')
    assert isinstance(value, dict)
    assert set(value) == {"__proto__", "constructor", "prototype"}
    assert value["__proto__"] == {"admin": True}


def test_canonical_is_stable_under_key_order() -> None:
    assert canonical({"b": 1, "a": [1, {"d": 1, "c": 2}]}) == canonical({"a": [1, {"c": 2, "d": 1}], "b": 1})
    assert canonical({"a": "é"}) == '{"a":"\\u00e9"}'


def test_kind_of() -> None:
    assert [kind_of(v) for v in (None, True, 1, 1.5, "s", [], {})] == [
        "null",
        "boolean",
        "number",
        "number",
        "string",
        "array",
        "object",
    ]
    assert kind_of(object()) == "object"


def test_json_equal_respects_json_types() -> None:
    assert json_equal(1, 1.0)
    assert not json_equal(True, 1)
    assert not json_equal(0, False)
    assert not json_equal("1", 1)
    assert json_equal({"a": [1, 2]}, {"a": [1.0, 2]})
    assert not json_equal({"a": 1}, {"a": 1, "b": 2})
    assert not json_equal([1], [1, 2])
    assert not json_equal([True], [1])
    assert contains_json([1, "a", {"x": 1}], {"x": 1})
    assert not contains_json([1], True)


def test_property_canonical_roundtrip_and_equality() -> None:
    for seed in range(200):
        rng = Prng(seed)
        value = {
            rng.word(): [rng.integer(-5, 5), rng.text(10), {rng.word(): rng.real(-1, 1)}]
            for _ in range(rng.integer(0, 4))
        }
        assert json_equal(parse_json(canonical(value)), value), seed
        assert canonical(parse_json(canonical(value))) == canonical(value), seed
