from __future__ import annotations

import threading
from decimal import Decimal
from typing import Any

import pytest

from llm_eval_harness.application.redaction import Redactor
from llm_eval_harness.application.runner import RunOptions, run_cases
from llm_eval_harness.domain.errors import ProviderError
from llm_eval_harness.domain.model import Case, ContextDocument, Expected, ToolCall, Usage
from llm_eval_harness.domain.pricing import PriceTable, parse_price_table
from llm_eval_harness.domain.results import CaseResult, stat_of
from support import keys
from support.builders import case, js, output, spec
from support.fakes import FakeClock, ScriptedJudge, ScriptedProvider

FACT = Expected(facts=("30 days",))
PASS = output("returns within 30 days")
FAIL = output("no idea")


def run(
    cases: list[Case],
    provider: ScriptedProvider,
    *,
    judge: ScriptedJudge | None = None,
    redactor: Redactor | None = None,
    **options: Any,
) -> list[CaseResult]:
    return run_cases(cases, provider, judge, redactor or Redactor(), FakeClock(), RunOptions(**options))


def test_one_run_per_case_and_repeat_ordered_by_id_then_repeat() -> None:
    provider = ScriptedProvider({}, default=PASS)
    results = run([case("b", expected=FACT), case("a", expected=FACT)], provider, repeat=3)
    assert [r.id for r in results] == ["a", "b"]
    assert [[run_.repeat for run_ in r.runs] for r in results] == [[0, 1, 2], [0, 1, 2]]
    assert sum(provider.calls.values()) == 6
    assert all(run_.status == "passed" and run_.attempts == 1 for r in results for run_ in r.runs)


def test_the_request_carries_repeat_attempt_and_seed_but_no_expectations() -> None:
    provider = ScriptedProvider({}, default=PASS)
    run([case("a", expected=Expected(facts=("TOP-SECRET-FACT",)))], provider, repeat=2, seed=11, timeout_s=7.5)
    requests = sorted(provider.requests, key=lambda r: r["repeat"])  # type: ignore[arg-type, return-value]
    assert [(r["repeat"], r["attempt"], r["seed"]) for r in requests] == [(0, 1, 11), (1, 1, 11)]
    assert "TOP-SECRET-FACT" not in repr(provider.requests) and provider.timeouts == [7.5, 7.5]


def test_failed_grades_are_never_retried() -> None:
    provider = ScriptedProvider({"a": [FAIL, PASS]})
    [result] = run([case("a", expected=FACT)], provider, retries=5)
    assert provider.calls["a"] == 1
    assert result.runs[0].status == "failed" and result.runs[0].attempts == 1


def test_retryable_infrastructure_errors_are_retried_with_backoff() -> None:
    boom = ProviderError("timeout", "slow", retryable=True)
    provider = ScriptedProvider({"a": [boom, boom, PASS]})
    clock = FakeClock()
    [result] = run_cases([case("a", expected=FACT)], provider, None, Redactor(), clock, RunOptions(retries=2))
    assert result.runs[0].status == "passed" and result.runs[0].attempts == 3
    assert clock.sleeps == [0.2, 0.4] and provider.calls["a"] == 3
    attempts = sorted(int(r["attempt"]) for r in provider.requests)  # type: ignore[arg-type,call-overload]
    assert attempts == [1, 2, 3]


def test_retries_run_out_and_the_error_is_recorded_not_hidden() -> None:
    boom = ProviderError("timeout", "slow", retryable=True)
    provider = ScriptedProvider({"a": [boom]})
    [result] = run([case("a", expected=FACT)], provider, retries=2)
    run_ = result.runs[0]
    assert run_.status == "error" and run_.attempts == 3 and run_.error is not None
    assert run_.error.kind == "timeout" and run_.output is None and run_.grades == ()
    assert stat_of(result).graded_runs == 0 and stat_of(result).errors == 1


def test_non_retryable_errors_are_not_retried() -> None:
    provider = ScriptedProvider({"a": [ProviderError("protocol", "bad json", retryable=False)]})
    [result] = run([case("a", expected=FACT)], provider, retries=5)
    assert provider.calls["a"] == 1 and result.runs[0].attempts == 1 and result.runs[0].error is not None


def test_unforeseen_exceptions_become_error_runs_and_do_not_stop_the_others() -> None:
    def explode(_: dict[str, object]) -> object:
        raise RuntimeError("secret detail that must not leak")

    provider = ScriptedProvider({"a": [explode]}, default=PASS)  # type: ignore[list-item]
    results = run([case("a", expected=FACT), case("b", expected=FACT)], provider)
    a, b = results
    assert a.runs[0].status == "error" and a.runs[0].error is not None and a.runs[0].error.kind == "internal"
    assert "secret detail" not in a.runs[0].error.message
    assert b.runs[0].status == "passed"


def test_concurrency_is_bounded() -> None:
    provider = ScriptedProvider({}, default=PASS)
    provider.delay = threading.Event()
    run([case(f"c{i}", expected=FACT) for i in range(12)], provider, concurrency=3)
    assert 1 < provider.peak <= 3
    sequential = ScriptedProvider({}, default=PASS)
    sequential.delay = threading.Event()
    run([case(f"c{i}", expected=FACT) for i in range(6)], sequential, concurrency=1)
    assert sequential.peak == 1


def test_results_do_not_depend_on_completion_order() -> None:
    cases = [case(f"c{i}", expected=FACT) for i in range(10)]
    a = run(cases, ScriptedProvider({}, default=PASS), concurrency=1, repeat=2)
    b = run(list(reversed(cases)), ScriptedProvider({}, default=PASS), concurrency=8, repeat=2)
    assert a == b


def test_cost_comes_from_reported_usage_and_the_price_table() -> None:
    prices = parse_price_table(js({"models": {"m": {"input_per_mtok": 1000, "output_per_mtok": 2000}}}), "p")
    provider = ScriptedProvider(
        {"a": [output("30 days", usage=Usage("m", 1000, 500))], "b": [output("30 days", usage=Usage("other", 1, 1))]}
    )
    a, b = run([case("a", expected=FACT), case("b", expected=FACT)], provider, prices=prices)
    assert a.runs[0].output is not None and a.runs[0].output.cost_usd == Decimal(2)
    assert b.runs[0].output is not None and b.runs[0].output.cost_usd is None
    budget = spec("cost_budget", {"max_usd": 5})
    [c] = run(
        [case("c", graders=(budget,))],
        ScriptedProvider({"c": [output("x", usage=Usage("m", 1000, 500))]}),
        prices=prices,
    )
    assert c.runs[0].status == "passed"
    [d] = run([case("d", graders=(budget,))], ScriptedProvider({"d": [output("x")]}), prices=PriceTable("USD", {}))
    assert d.runs[0].status == "failed" and "unknown" in d.runs[0].grades[0].detail


def test_default_graders_apply_to_every_case() -> None:
    provider = ScriptedProvider({}, default=output("30 days", latency_ms=500))
    default = (spec("latency_budget", {"max_ms": 100}),)
    [result] = run([case("a", expected=FACT)], provider, default_graders=default)
    assert result.runs[0].status == "failed" and [g.name for g in result.runs[0].grades] == ["facts", "latency_budget"]


def test_a_failed_must_fails_the_run_whatever_the_score() -> None:
    expected = Expected(facts=("30 days",), forbidden=("CANARY",))
    [result] = run([case("a", expected=expected)], ScriptedProvider({}, default=output("30 days CANARY")))
    run_ = result.runs[0]
    assert run_.status == "failed" and run_.failed_musts == ("forbidden",) and run_.score == 0.5


def test_stored_output_is_redacted_but_grading_saw_the_original() -> None:
    token = keys.github_token()
    provider = ScriptedProvider(
        {}, default=output(f"here {token}", tool_calls=(ToolCall("send", {"body": f"k={token}"}),), citations=(token,))
    )
    [result] = run([case("a", graders=(spec("secret_leak"),), input={"q": token})], provider)
    run_ = result.runs[0]
    assert run_.status == "failed" and "github-token" in run_.grades[0].detail  # the grader saw the secret
    stored = run_.output
    assert stored is not None
    assert token not in stored.text and "[REDACTED:github-token]" in stored.text
    assert token not in repr(stored.tool_calls) and token not in repr(stored.citations)
    assert token not in result.input_preview
    assert token not in repr(run_.grades)


def test_redaction_can_be_switched_off_and_extended() -> None:
    token = keys.github_token()
    provider = ScriptedProvider({}, default=output(f"{token} order ORD-77"))
    [off] = run([case("a", expected=Expected(facts=("ORD",)))], provider, redactor=Redactor.build(False, []))
    assert off.runs[0].output is not None and token in off.runs[0].output.text
    [custom] = run(
        [case("a", expected=Expected(facts=("ORD",)))], provider, redactor=Redactor.build(True, [r"ORD-\d+"])
    )
    assert custom.runs[0].output is not None and "ORD-77" not in custom.runs[0].output.text
    assert custom.runs[0].status == "passed"  # grading looked at the raw text, which contains ORD


def test_huge_outputs_are_stored_truncated_but_graded_whole() -> None:
    text = "a" * 100_000 + " 30 days"
    [result] = run([case("a", expected=FACT)], ScriptedProvider({}, default=output(text)))
    stored = result.runs[0].output
    assert result.runs[0].status == "passed"  # the fact at the very end was found
    assert stored is not None and stored.truncated and len(stored.text) < 21_000


def test_huge_tool_arguments_are_bounded_in_storage() -> None:
    provider = ScriptedProvider({}, default=output("x", tool_calls=(ToolCall("t", {"blob": "z" * 50_000}),)))
    [result] = run([case("a", graders=(spec("regex", {"pattern": "x"}),))], provider)
    stored = result.runs[0].output
    assert stored is not None and "_truncated" in stored.tool_calls[0].arguments


def test_a_crashing_grader_fails_closed_without_stopping_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    from llm_eval_harness.application import grading

    def broken(*_: object) -> object:
        raise RuntimeError("bug")

    monkeypatch.setattr(grading, "evaluate", broken)
    [result] = run([case("a", expected=FACT)], ScriptedProvider({}, default=PASS))
    run_ = result.runs[0]
    assert run_.status == "failed" and "grader crashed (RuntimeError)" in run_.grades[0].detail


def test_judge_verdicts_are_recorded_as_non_deterministic_with_confidence() -> None:
    judge = ScriptedJudge([js({"verdict": "pass", "confidence": 0.9, "reason": f"fine {keys.github_token()}"})])
    judged = case("a", graders=(spec("judge", {"argv": ["judge"], "rubric": "r"}),), expected=FACT)
    [result] = run([judged], ScriptedProvider({}, default=PASS), judge=judge)
    grade = result.runs[0].grades[1]
    assert (
        result.runs[0].status == "passed"
        and not grade.deterministic
        and grade.confidence == 0.9
        and grade.verdict == "pass"
    )
    assert "REDACTED:github-token" in grade.detail
    assert judge.requests[0]["kind"] == "judge"


def test_judge_failures_are_error_runs_not_grades_and_are_retried_when_retryable() -> None:
    judged = case("a", graders=(spec("judge", {"argv": ["judge"]}),))
    flaky = ScriptedJudge([ProviderError("timeout", "slow", retryable=True), js({"verdict": "pass", "confidence": 1})])
    [ok] = run([judged], ScriptedProvider({}, default=PASS), judge=flaky, retries=1)
    assert ok.runs[0].status == "passed" and flaky.calls == 2
    dead = ScriptedJudge([ProviderError("timeout", "slow", retryable=True)])
    [bad] = run([judged], ScriptedProvider({}, default=PASS), judge=dead, retries=1)
    assert bad.runs[0].status == "error" and bad.runs[0].error is not None and bad.runs[0].error.kind == "timeout"
    garbled = ScriptedJudge([js({"verdict": "maybe"})])
    [odd] = run([judged], ScriptedProvider({}, default=PASS), judge=garbled)
    assert odd.runs[0].status == "error" and odd.runs[0].error is not None and odd.runs[0].error.kind == "protocol"
    [none] = run([judged], ScriptedProvider({}, default=PASS), judge=None)
    assert none.runs[0].status == "error" and none.runs[0].error is not None and none.runs[0].error.kind == "refused"


def test_context_documents_are_passed_to_the_provider() -> None:
    provider = ScriptedProvider({}, default=PASS)
    run([case("a", expected=FACT, context=(ContextDocument("d1", "text"),))], provider)
    sent = provider.requests[0]["case"]
    assert isinstance(sent, dict) and sent["context"] == [{"id": "d1", "text": "text", "metadata": {}}]


def test_error_messages_are_single_line_and_redacted() -> None:
    token = keys.github_token()
    boom = ProviderError("exit", f"exit status 1: line one\nline two {token}\x1b[31m", retryable=False)
    [result] = run([case("a", expected=FACT)], ScriptedProvider({"a": [boom]}))
    message = result.runs[0].error.message  # type: ignore[union-attr]
    assert "\n" not in message and "\x1b" not in message and token not in message


def test_grader_details_never_carry_secret_shaped_text_into_the_report() -> None:
    token = keys.github_token()
    exact = case("a", graders=(spec("exact", {"answers": ["nope"]}),))
    [result] = run([exact], ScriptedProvider({}, default=output(token)))
    detail = result.runs[0].grades[0].detail
    assert result.runs[0].status == "failed" and token not in detail and "[REDACTED:github-token]" in detail
    grounded = case("g", graders=(spec("grounding", {}),))
    [g] = run([grounded], ScriptedProvider({}, default=output("x", citations=(token,))))
    assert token not in repr(g.runs[0].grades)


def test_the_reported_model_name_is_redacted_but_cost_still_uses_the_real_name() -> None:
    token = keys.github_token()
    prices = parse_price_table(js({"models": {token: {"input_per_mtok": 1000, "output_per_mtok": 1000}}}), "p")
    [result] = run(
        [case("a", expected=FACT)],
        ScriptedProvider({}, default=output("30 days", usage=Usage(token, 1000, 0))),
        prices=prices,
    )
    stored = result.runs[0].output
    assert stored is not None and stored.usage is not None and token not in (stored.usage.model or "")
    assert stored.cost_usd == Decimal(1)


def test_redaction_looks_at_most_at_the_first_32k_characters_of_any_string_and_says_so() -> None:
    from llm_eval_harness.application.redaction import Redactor

    r = Redactor()
    token = keys.github_token()
    out = r.text("a" * 40_000 + token)
    assert token not in out and len(out) < 33_000 and "more characters" in out
    stored = r.json({"k": "b" * 40_000})
    assert isinstance(stored, dict) and str(stored["k"]).endswith("more characters]")
