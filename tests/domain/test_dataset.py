from __future__ import annotations

import json

import pytest

from llm_eval_harness.domain.dataset import case_hash, dataset_hash, parse_dataset
from llm_eval_harness.domain.errors import DatasetError
from llm_eval_harness.domain.jsonio import JsonValue, parse_json
from llm_eval_harness.domain.safety_suite import starter_cases, starter_dataset_text
from support.builders import js
from support.prng import Prng


def ds(*records: object):  # type: ignore[no-untyped-def]
    return parse_dataset([(f"line {i + 1}", js(r)) for i, r in enumerate(records)], "cases.jsonl")


def problems_of(*records: object) -> list[str]:
    with pytest.raises(DatasetError) as error:
        ds(*records)
    return [p.render() for p in error.value.problems]


PINNED_HASH = "sha256:6a92802f1e17d8a23a3fd554e0d8c4d4e4c8f9e303887776fb2a40b06d734c0a"

GOOD = {
    "id": "refund-window",
    "tags": ["policy"],
    "input": {"question": "How long do I have to return headphones?"},
    "context": [{"id": "returns", "text": "30 days", "metadata": {"page": 2}}],
    "expected": {
        "facts": ["30 days"],
        "forbidden": ["CANARY"],
        "tool_calls": [{"name": "lookup", "args": {"id": "x"}}],
    },
    "graders": [{"type": "regex", "pattern": "days", "weight": 2, "must": True, "name": "has-unit"}],
    "threshold": 0.5,
    "metadata": {"owner": "support"},
}


def test_a_full_case_is_read_into_the_model() -> None:
    dataset = ds(GOOD)
    [case] = dataset.cases
    assert case.id == "refund-window" and case.category == "quality" and case.tags == ("policy",)
    assert case.context[0].id == "returns" and case.context[0].metadata == {"page": 2}
    assert case.expected.facts == ("30 days",)
    assert case.expected.tool_calls is not None and case.expected.tool_calls[0].name == "lookup"
    assert case.graders[0].type == "regex" and case.graders[0].weight == 2.0 and case.graders[0].must
    assert case.graders[0].name == "has-unit" and case.threshold == 0.5
    assert dataset.hash.startswith("sha256:") and case.hash == case_hash(GOOD)  # type: ignore[arg-type]


def test_cases_are_sorted_by_id_and_hash_ignores_order() -> None:
    a = {"id": "a", "input": "x", "expected": {"answers": ["1"]}}
    b = {"id": "b", "input": "y", "expected": {"answers": ["2"]}}
    assert [c.id for c in ds(b, a).cases] == ["a", "b"]
    assert ds(a, b).hash == ds(b, a).hash


def test_hash_changes_with_any_change_to_any_case() -> None:
    base = {"id": "a", "input": "x", "expected": {"answers": ["1"]}}
    other = {"id": "b", "input": "y", "expected": {"answers": ["2"]}}
    h = ds(base, other).hash
    assert ds(base, {**other, "input": "z"}).hash != h
    assert ds(base, {**other, "tags": ["t"]}).hash != h
    assert ds(base, {**other, "expected": {"answers": ["3"]}}).hash != h
    assert ds(base).hash != h
    assert ds(base, {**other, "id": "c"}).hash != h
    assert ds(base, other).hash == h


def test_property_hash_is_deterministic_and_order_free() -> None:
    for seed in range(100):
        rng = Prng(seed)
        records = [
            {"id": f"c{i}", "input": rng.text(20), "expected": {"facts": [rng.word()]}}
            for i in range(rng.integer(1, 8))
        ]
        assert ds(*records).hash == ds(*rng.shuffled(records)).hash, seed
        assert dataset_hash(ds(*records).cases) == ds(*records).hash, seed


def test_hash_is_stable_across_runs_and_key_order() -> None:
    one = ds({"id": "a", "input": "x", "expected": {"answers": ["1"]}}).hash
    assert one == ds({"expected": {"answers": ["1"]}, "input": "x", "id": "a"}).hash
    assert one == PINNED_HASH


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        (5, "line 1: each case must be a JSON object (got number)"),
        ({}, "line 1.id: is required"),
        ({"id": "a"}, "line 1.input: is required"),
        ({"id": "a", "input": "x"}, "line 1: has nothing to grade: add expected fields or graders"),
        ({"id": "has space", "input": "x", "expected": {"facts": ["a"]}}, "line 1.id: may use letters"),
        ({"id": "", "input": "x", "expected": {"facts": ["a"]}}, "line 1.id: must not be empty"),
        ({"id": 5, "input": "x"}, "line 1.id: must be a string (got number)"),
        (
            {"id": "a", "input": "x", "extra": 1, "expected": {"facts": ["a"]}},
            "line 1.extra: unknown key (allowed: category, context",
        ),
        (
            {"id": "a", "input": "x", "category": "security", "expected": {"facts": ["a"]}},
            "line 1.category: must be one of quality, safety (got 'security')",
        ),
        ({"id": "a", "input": "x", "tags": ["ok", "ok"], "expected": {"facts": ["a"]}}, "line 1.tags: has duplicates"),
        (
            {"id": "a", "input": "x", "tags": ["bad tag"], "expected": {"facts": ["a"]}},
            "line 1.tags[0]: may use letters",
        ),
        (
            {"id": "a", "input": "x", "tags": "t", "expected": {"facts": ["a"]}},
            "line 1.tags: must be an array of strings (got string)",
        ),
        ({"id": "a", "input": "x", "expected": []}, "line 1.expected: must be an object (got array)"),
        (
            {"id": "a", "input": "x", "expected": {"facts": "a"}},
            "line 1.expected.facts: must be an array of strings (got string)",
        ),
        ({"id": "a", "input": "x", "expected": {"fact": ["a"]}}, "line 1.expected.fact: unknown key"),
        (
            {"id": "a", "input": "x", "expected": {"forbidden_patterns": ["(a)\\1"]}},
            "line 1.expected.forbidden_patterns[0]: back-references are not supported",
        ),
        (
            {"id": "a", "input": "x", "expected": {"tool_calls": [{"args": {}}]}},
            "line 1.expected.tool_calls[0].name: is required",
        ),
        (
            {"id": "a", "input": "x", "expected": {"tool_calls": [{"name": "t", "args": {"x": {"min": "a"}}}]}},
            "args.x.min: must be a number (got string)",
        ),
        (
            {"id": "a", "input": "x", "expected": {"tool_order": "sorted", "facts": ["a"]}},
            "line 1.expected.tool_order: must be one of any, ordered",
        ),
        (
            {"id": "a", "input": "x", "expected": {"allow_extra_tools": "no", "facts": ["a"]}},
            "allow_extra_tools: must be true or false (got string)",
        ),
        (
            {"id": "a", "input": "x", "context": [{"id": "d"}], "expected": {"facts": ["a"]}},
            "line 1.context[0].text: is required",
        ),
        (
            {
                "id": "a",
                "input": "x",
                "context": [{"id": "d", "text": "t"}, {"id": "d", "text": "u"}],
                "expected": {"facts": ["a"]},
            },
            "line 1.context[1].id: duplicate document id 'd'",
        ),
        (
            {"id": "a", "input": "x", "context": ["doc"], "expected": {"facts": ["a"]}},
            "line 1.context[0]: must be an object with id and text (got string)",
        ),
        (
            {"id": "a", "input": "x", "threshold": 2, "expected": {"facts": ["a"]}},
            "line 1.threshold: must be at most 1.0 (got 2)",
        ),
        (
            {"id": "a", "input": "x", "graders": [{"type": "magic"}]},
            "line 1.graders[0].type: unknown grader 'magic' (known: contains",
        ),
        ({"id": "a", "input": "x", "graders": [{}]}, "line 1.graders[0].type: is required and must be a string"),
        (
            {"id": "a", "input": "x", "graders": ["exact"]},
            "line 1.graders[0]: must be an object with a type (got string)",
        ),
        ({"id": "a", "input": "x", "graders": [{"type": "exact"}]}, "line 1.graders[0].answers: is required"),
        (
            {"id": "a", "input": "x", "graders": [{"type": "exact", "answers": ["a"], "weight": -1}]},
            "line 1.graders[0].weight: must be at least 0.0 (got -1)",
        ),
        (
            {"id": "a", "input": "x", "graders": [{"type": "regex", "pattern": "a", "must": "yes"}]},
            "line 1.graders[0].must: must be true or false (got string)",
        ),
        (
            {"id": "a", "input": "x", "graders": [{"type": "regex", "pattern": "("}]},
            "line 1.graders[0].pattern: missing closing parenthesis",
        ),
        (
            {"id": "a", "input": "x", "metadata": 5, "expected": {"facts": ["a"]}},
            "line 1.metadata: must be an object (got number)",
        ),
    ],
)
def test_error_messages_say_which_record_and_which_field(record: object, expected: str) -> None:
    found = problems_of(record)
    assert any(expected in line for line in found), found


def test_duplicate_ids_name_both_lines() -> None:
    a = {"id": "a", "input": "x", "expected": {"facts": ["1"]}}
    assert problems_of(a, a) == ["line 2: duplicate case id 'a' (first used at line 1)"]


def test_every_problem_is_reported_not_just_the_first() -> None:
    found = problems_of({"id": "a b", "input": "x", "tags": "t"}, {"id": "c"})
    assert len(found) >= 3
    assert (
        any("line 1.id" in f for f in found)
        and any("line 1.tags" in f for f in found)
        and any("line 2.input" in f for f in found)
    )


def test_error_text_lists_the_source_and_count() -> None:
    with pytest.raises(DatasetError) as error:
        ds({"id": "a"})
    assert str(error.value).startswith("cases.jsonl: 1 problem(s)\n  line 1.input: is required")


def test_empty_dataset_and_too_many_cases_are_refused() -> None:
    with pytest.raises(DatasetError, match="contains no cases"):
        parse_dataset([], "x.jsonl")
    records = [(f"line {i}", js({"id": f"c{i}", "input": "x", "expected": {"facts": ["a"]}})) for i in range(10_001)]
    with pytest.raises(DatasetError, match="the limit is 10000"):
        parse_dataset(records, "x.jsonl")


def test_problem_list_is_capped() -> None:
    records = [(f"line {i}", js({"id": "bad id"})) for i in range(500)]
    with pytest.raises(DatasetError) as error:
        parse_dataset(records, "x.jsonl")
    assert len(error.value.problems) <= 51 and "stopped after 50 problems" in error.value.problems[-1].message


def test_oversized_case_is_refused() -> None:
    found = problems_of({"id": "a", "input": "x" * 300_000, "expected": {"facts": ["a"]}})
    assert "the limit is 262144" in found[0]


def test_hostile_values_are_plain_data() -> None:
    record = parse_json(
        '{"id": "a", "input": {"__proto__": {"polluted": true}, "constructor": "x"}, "metadata": {"__proto__": 1},'
        ' "expected": {"facts": ["\\u202e\\u0000\\u001b[31m"]}, "tags": ["constructor"]}'
    )
    dataset = parse_dataset([("line 1", record)], "x.jsonl")
    assert dataset.cases[0].input == {"__proto__": {"polluted": True}, "constructor": "x"}
    assert dataset.cases[0].metadata == {"__proto__": 1}
    assert ({}).__class__ is dict and not hasattr({}, "polluted")


def test_the_starter_safety_dataset_is_itself_valid() -> None:
    records: list[tuple[str, JsonValue]] = [
        (f"line {i + 1}", parse_json(line)) for i, line in enumerate(starter_dataset_text().splitlines())
    ]
    dataset = parse_dataset(records, "safety.jsonl")
    assert len(dataset.cases) == len(starter_cases()) >= 8
    assert all(c.category == "safety" and "safety" in c.tags for c in dataset.cases)
    tags = {t for c in dataset.cases for t in c.tags}
    assert {"prompt-injection", "data-exfiltration", "unauthorised-tool"} <= tags
    assert starter_dataset_text() == starter_dataset_text()
    assert all(json.loads(line) for line in starter_dataset_text().splitlines())
