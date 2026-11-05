from __future__ import annotations

from decimal import Decimal

import pytest

from llm_eval_harness.domain.grading.judge import (
    JudgeConfig,
    JudgeVerdict,
    judge_outcome,
    parse_judge,
    parse_verdict,
)
from llm_eval_harness.domain.grading.registry import GRADERS
from llm_eval_harness.domain.model import ToolCall, Usage
from support import keys
from support.builders import grade, output, validate


def test_secret_leak_in_text_and_tool_arguments_without_echoing_the_secret() -> None:
    secret = keys.github_token()
    out = grade("secret_leak", {}, output(f"your token is {secret}"))
    assert not out.passed and secret not in out.detail and "github-token" in out.detail
    nested = ToolCall("send_email", {"body": {"parts": ["hello", keys.aws_key_id()]}})
    out2 = grade("secret_leak", {}, output("clean text", tool_calls=(nested,)))
    assert not out2.passed and "aws-access-key-id" in out2.detail
    assert grade("secret_leak", {}, output("nothing secret")).passed


def test_secret_leak_kinds_and_allowlist() -> None:
    secret = keys.github_token()
    assert grade("secret_leak", {"kinds": ["aws-access-key-id"]}, output(secret)).passed
    assert grade("secret_leak", {"allow": [secret]}, output(f"placeholder {secret}")).passed
    assert not grade("secret_leak", {"kinds": ["github-token"]}, output(secret)).passed
    found = [p.render() for p in validate("secret_leak", {"kinds": ["nope"]}, "g")]
    assert found[0].startswith("g.kinds[0]: unknown kind 'nope'")


def test_pii_leak() -> None:
    out = grade("pii_leak", {}, output("write to jane@example.com or 078-05-1120"))
    assert not out.passed and "email" in out.detail and "us-ssn" in out.detail and "jane" not in out.detail
    assert grade("pii_leak", {"allow": ["support@example.com"]}, output("mail support@example.com")).passed
    assert grade("pii_leak", {"kinds": ["us-ssn"]}, output("jane@example.com")).passed
    assert not grade("pii_leak", {}, output("", tool_calls=(ToolCall("t", {"to": "a@b.example"}),))).passed
    assert grade("pii_leak", {}, output("no personal data")).passed


def test_latency_budget() -> None:
    assert grade("latency_budget", {"max_ms": 100}, output(latency_ms=99.5)).passed
    out = grade("latency_budget", {"max_ms": 100}, output(latency_ms=150))
    assert not out.passed and "150 ms exceeds" in out.detail


def test_cost_budget_fails_closed_when_cost_is_unknown() -> None:
    assert grade("cost_budget", {"max_usd": 0.01}, output(), cost=Decimal("0.0099")).passed
    assert not grade("cost_budget", {"max_usd": 0.01}, output(), cost=Decimal("0.02")).passed
    unknown = grade("cost_budget", {"max_usd": 1}, output(), cost=None)
    assert not unknown.passed and "unknown" in unknown.detail


def test_token_budget() -> None:
    assert grade("token_budget", {"max_total": 100}, output(usage=Usage("m", 60, 40))).passed
    assert not grade("token_budget", {"max_total": 99}, output(usage=Usage("m", 60, 40))).passed
    assert "no usage" in grade("token_budget", {"max_total": 99}, output()).detail


def test_judge_parameters_and_verdicts() -> None:
    problems: list = []  # type: ignore[type-arg]
    config = parse_judge({"argv": ["python3", "judge.py"], "rubric": "be nice"}, "g", problems)
    assert problems == []
    assert config.timeout_s == 60.0 and config.min_confidence == 0.7 and config.low_confidence == "fail"
    assert [p.render() for p in validate("judge", {}, "g")] == ["g.argv: is required"]
    assert not GRADERS["judge"].deterministic and GRADERS["judge"].grade is None


def test_judge_outcome_applies_min_confidence() -> None:
    config = JudgeConfig(("x",), 1.0, (), "", 0.7, "fail")
    ok = judge_outcome(config, JudgeVerdict("pass", 1.0, 0.9, "good"))
    assert ok.passed and "confidence 0.90" in ok.detail and "good" in ok.detail
    assert not judge_outcome(config, JudgeVerdict("fail", 0.2, 0.9, "")).passed
    low = judge_outcome(config, JudgeVerdict("pass", 1.0, 0.3, ""))
    assert not low.passed and "below min_confidence" in low.detail
    lenient = JudgeConfig(("x",), 1.0, (), "", 0.7, "pass")
    assert judge_outcome(lenient, JudgeVerdict("fail", 0.0, 0.3, "")).passed


@pytest.mark.parametrize(
    ("obj", "fragment"),
    [
        ([], "must be a JSON object"),
        ({"verdict": "maybe", "confidence": 1}, '"verdict"'),
        ({"verdict": "pass"}, '"confidence"'),
        ({"verdict": "pass", "confidence": 2}, '"confidence"'),
        ({"verdict": "pass", "confidence": True}, '"confidence"'),
        ({"verdict": "pass", "confidence": 1, "score": 5}, '"score"'),
        ({"verdict": "pass", "confidence": 1, "reason": 5}, '"reason"'),
    ],
)
def test_malformed_judge_answers_are_rejected(obj: object, fragment: str) -> None:
    with pytest.raises(ValueError) as error:
        parse_verdict(obj)  # type: ignore[arg-type]
    assert fragment in str(error.value)


def test_well_formed_judge_answer() -> None:
    verdict = parse_verdict({"verdict": "fail", "confidence": 0.5, "reason": "x" * 5000})
    assert verdict.verdict == "fail" and verdict.score == 0.0 and len(verdict.reason) < 2100
