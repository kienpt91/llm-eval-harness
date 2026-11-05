from __future__ import annotations

import pytest

from llm_eval_harness.application.config import FakeProviderConfig
from llm_eval_harness.domain.errors import ProviderError, UsageError
from llm_eval_harness.domain.jsonio import JsonValue
from llm_eval_harness.domain.protocol import build_request
from llm_eval_harness.infrastructure.providers.fake import FakeProvider
from support.builders import case, js


def fake(spec: object) -> FakeProvider:
    return FakeProvider(FakeProviderConfig(js(spec)))  # type: ignore[arg-type]


def req(case_id: str = "a", *, input: JsonValue = "hello", repeat: int = 0, attempt: int = 1) -> dict[str, JsonValue]:
    return build_request(case(case_id, input=input), repeat=repeat, attempt=attempt, seed=None)


def test_unscripted_cases_are_echoed_deterministically() -> None:
    p = fake({})
    assert p.generate(req(input="hello"), 1.0).text == "hello"
    assert p.generate(req(input={"question": "why?"}), 1.0).text == "why?"
    assert p.generate(req(input={"n": 1}), 1.0).text == '{"n":1}'
    assert p.generate(req(), 1.0) == p.generate(req(), 1.0)
    assert p.describe() == {"type": "fake", "model": "fake-model"}


def test_scripted_answers_with_protocol_fields_and_estimated_usage() -> None:
    p = fake(
        {
            "model": "tiny",
            "latency_ms": 12,
            "script": {
                "a": {"text": "x" * 40, "tool_calls": [{"name": "t", "arguments": {"k": 1}}], "citations": ["d"]}
            },
        }
    )
    out = p.generate(req("a", input="abcd" * 5), 1.0)
    assert out.text == "x" * 40 and out.tool_calls[0].name == "t" and out.citations == ("d",)
    assert out.usage is not None and (out.usage.model, out.usage.input_tokens, out.usage.output_tokens) == (
        "tiny",
        6,
        10,
    )
    assert out.latency_ms == 12.0


def test_an_answer_may_carry_its_own_usage_and_latency() -> None:
    p = fake(
        {
            "script": {
                "a": {"text": "x", "usage": {"model": "m", "input_tokens": 1, "output_tokens": 2}, "latency_ms": 99}
            }
        }
    )
    out = p.generate(req("a"), 1.0)
    assert out.usage is not None and out.usage.total_tokens == 3 and out.latency_ms == 99.0


def test_lists_vary_by_repeat_which_is_how_flakiness_is_scripted() -> None:
    p = fake({"script": {"a": [{"text": "one"}, {"text": "two"}, {"text": "three"}]}})
    assert [p.generate(req("a", repeat=r), 1.0).text for r in range(5)] == ["one", "two", "three", "one", "two"]


def test_default_answer_for_unscripted_cases() -> None:
    p = fake({"default": {"text": "default"}, "script": {"a": {"text": "mine"}}})
    assert p.generate(req("a"), 1.0).text == "mine" and p.generate(req("b"), 1.0).text == "default"


def test_scripted_failures_depend_only_on_the_attempt_number() -> None:
    p = fake(
        {
            "script": {
                "a": {"error": {"kind": "timeout", "message": "slow", "fail_attempts": 2, "then": {"text": "finally"}}},
                "b": {"error": {"kind": "protocol", "message": "bad"}},
            }
        }
    )
    for attempt in (1, 2):
        with pytest.raises(ProviderError) as error:
            p.generate(req("a", attempt=attempt), 1.0)
        assert error.value.kind == "timeout" and error.value.retryable
    assert p.generate(req("a", attempt=3), 1.0).text == "finally"
    with pytest.raises(ProviderError) as never:
        p.generate(req("b", attempt=50), 1.0)
    assert never.value.kind == "protocol" and not never.value.retryable


@pytest.mark.parametrize(
    ("spec", "fragment"),
    [
        ({"vendor": 1}, "provider.vendor: unknown key"),
        ({"model": 5}, "provider.model must be a string"),
        ({"latency_ms": -1}, "non-negative"),
        ({"script": []}, "provider.script: must be an object"),
        ({"script": {"a": 5}}, "entries are objects or lists"),
        ({"script": {"a": {}}}, 'needs "text"'),
        ({"script": {"a": {"text": "x", "bogus": 1}}}, "unknown key 'bogus'"),
        ({"script": {"a": []}}, "needs at least one entry"),
        ({"script": {"a": [[{"text": "x"}]]}}, "cannot nest"),
        ({"script": {"a": {"error": {"kind": "weird"}}}}, "needs a kind"),
        ({"script": {"a": {"error": 1}}}, "needs a kind"),
        ({"script": {"a": {"error": {"kind": "timeout", "then": {"nope": 1}}}}}, "unknown key 'nope'"),
        ({"default": 5}, "entries are objects or lists"),
    ],
)
def test_bad_specs_fail_at_construction_not_mid_run(spec: object, fragment: str) -> None:
    with pytest.raises(UsageError) as error:
        fake(spec)
    assert fragment in str(error.value)
