from __future__ import annotations

import pytest

from llm_eval_harness.domain.errors import ProviderError
from llm_eval_harness.domain.model import ContextDocument, Expected, ToolCall, Usage
from llm_eval_harness.domain.protocol import build_judge_request, build_request, parse_response
from support.builders import case, js, output


def test_request_carries_input_and_context_but_never_the_answer_key() -> None:
    c = case(
        "q1",
        category="safety",
        tags=("a",),
        input={"question": "hi"},
        context=(ContextDocument("d1", "doc text", {"p": 1}),),
        expected=Expected(answers=("SECRET-ANSWER",), facts=("SECRET-FACT",), forbidden=("SECRET-BAN",)),
    )
    request = build_request(c, repeat=2, attempt=1, seed=9)
    assert request == {
        "protocol": 1,
        "case": {
            "id": "q1",
            "category": "safety",
            "tags": ["a"],
            "input": {"question": "hi"},
            "context": [{"id": "d1", "text": "doc text", "metadata": {"p": 1}}],
            "metadata": {},
        },
        "repeat": 2,
        "attempt": 1,
        "seed": 9,
    }
    assert "SECRET" not in repr(request)


def test_a_minimal_and_a_full_response_are_read() -> None:
    assert parse_response(js({"text": "hi"})).text == "hi"
    full = parse_response(
        js(
            {
                "text": "t",
                "structured": {"a": 1},
                "tool_calls": [{"name": "lookup", "arguments": {"id": 1}}, {"name": "noargs"}],
                "citations": ["d1"],
                "retrieved": ["d1", "d2"],
                "usage": {"model": "m", "input_tokens": 3, "output_tokens": 4},
                "metadata": {"anything": [1]},
            }
        )
    )
    assert full.has_structured and full.structured == {"a": 1}
    assert full.tool_calls == (ToolCall("lookup", {"id": 1}), ToolCall("noargs", {}))
    assert full.citations == ("d1",) and full.retrieved == ("d1", "d2") and full.usage == Usage("m", 3, 4)
    assert parse_response(js({"text": "t", "structured": None})).has_structured


@pytest.mark.parametrize(
    ("response", "fragment"),
    [
        ([], "must be a JSON object (got array)"),
        ({}, 'needs "text" as a string'),
        ({"text": 5}, 'needs "text" as a string (got number)'),
        ({"text": "t", "tool_call": []}, "unknown key 'tool_call'"),
        ({"text": "t", "tool_calls": {}}, '"tool_calls" must be an array'),
        ({"text": "t", "tool_calls": [1]}, '"tool_calls[0]" must be an object'),
        ({"text": "t", "tool_calls": [{"arguments": {}}]}, '"tool_calls[0].name" must be a non-empty string'),
        ({"text": "t", "tool_calls": [{"name": "a", "arguments": []}]}, '"tool_calls[0].arguments" must be an object'),
        ({"text": "t", "tool_calls": [{"name": "a", "args": {}}]}, "unknown key 'args' (use name and arguments)"),
        ({"text": "t", "tool_calls": [{"name": "a"}] * 300}, "more than 256 entries"),
        ({"text": "t", "citations": "d1"}, '"citations" must be an array of strings'),
        ({"text": "t", "citations": [1]}, '"citations[0]" must be a string'),
        ({"text": "t", "retrieved": ["x" * 300]}, "at most 256 characters"),
        ({"text": "t", "citations": ["a"] * 2000}, "more than 1024 entries"),
        ({"text": "t", "usage": 5}, '"usage" must be an object'),
        ({"text": "t", "usage": {"input_tokens": -1}}, '"usage.input_tokens" must be a non-negative whole number'),
        ({"text": "t", "usage": {"input_tokens": 1.5}}, '"usage.input_tokens"'),
        ({"text": "t", "usage": {"output_tokens": True}}, '"usage.output_tokens"'),
        ({"text": "t", "usage": {"input_tokens": 10**13}}, '"usage.input_tokens"'),
        ({"text": "t", "usage": {"model": ""}}, '"usage.model" must be a non-empty string'),
        ({"text": "t", "usage": {"cost": 1}}, "unknown key 'cost'"),
    ],
)
def test_malformed_responses_are_protocol_errors_that_are_not_retried(response: object, fragment: str) -> None:
    with pytest.raises(ProviderError) as error:
        parse_response(js(response))
    assert error.value.kind == "protocol" and not error.value.retryable
    assert fragment in str(error.value)


def test_unknown_key_names_are_shown_safely() -> None:
    with pytest.raises(ProviderError) as error:
        parse_response(js({"text": "t", "\x1b[31mevil": 1}))
    assert "\x1b" not in str(error.value)


def test_judge_request_has_the_output_and_expectations_for_the_rubric() -> None:
    c = case("q", input="why?", expected=Expected(answers=("a",), facts=("f",), forbidden=("never-sent",)))
    request = build_judge_request(
        c, output("the answer", tool_calls=(ToolCall("t", {"a": 1}),), citations=("d",)), "be strict"
    )
    assert request["kind"] == "judge" and request["rubric"] == "be strict"
    assert request["expected"] == {"answers": ["a"], "facts": ["f"]}
    assert request["output"] == {
        "text": "the answer",
        "tool_calls": [{"name": "t", "arguments": {"a": 1}}],
        "citations": ["d"],
    }
    assert "never-sent" not in repr(request)
