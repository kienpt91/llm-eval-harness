"""The worked example (examples/support-agent) through the command line: the demo's claims, checked."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

from support.cli import invoke

pytestmark = pytest.mark.slow
EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "support-agent"


def texts(report: dict[str, Any]) -> list[list[str]]:
    return [[run["output"]["text"] for run in c["runs"]] for c in report["cases"]]


@pytest.fixture
def example(tmp_path: Path) -> Path:
    work = tmp_path / "support-agent"
    shutil.copytree(EXAMPLE, work, ignore=shutil.ignore_patterns("out", "out-*", "__pycache__"))
    for name in ("llm-eval.json", "llm-eval.v2.json"):
        path = work / name
        config = json.loads(path.read_text())
        config["provider"]["argv"][0] = sys.executable
        path.write_text(json.dumps(config))
    return work


def test_the_committed_baseline_matches_the_dataset_and_the_unchanged_pipeline_passes(example: Path) -> None:
    result = invoke(["gate"], example)
    assert result.code == 0, result.out + result.err
    assert "verdict   OK" in result.out and "gate      PASS (exit 0)" in result.out
    assert "q-damaged-on-arrival" in result.out  # the one quality case v1 does not pass is visible, not hidden


def test_the_change_improves_quality_and_the_gate_still_exits_4(example: Path) -> None:
    result = invoke(["gate", "--config", "llm-eval.v2.json", "--markdown", "pr.md"], example)
    assert result.code == 4, result.out + result.err
    assert "quality   mean pass rate 1.000 over 12 cases (12 fully passing)" in result.out
    assert "safety    9 cases, 1 failing" in result.out
    assert "forbidden tool called: 'issue_refund'" in result.out
    assert (
        "regressed    safety-injection-in-document" in result.out and "improved     q-damaged-on-arrival" in result.out
    )
    decision = json.loads((example / "out-v2" / "gate.json").read_text())
    assert decision["exit_code"] == 4 and "safety-injection-in-document" in decision["reasons"][0]
    assert decision["comparison"]["accuracy"]["absolute"] > 0
    markdown = (example / "pr.md").read_text()
    assert "## LLM eval: FAILED (exit 4)" in markdown and "### Safety failures (always fatal)" in markdown
    assert "ORD-1001" in markdown  # the unauthorised call is shown with its arguments


def test_quality_improvement_cannot_buy_back_a_safety_failure(example: Path) -> None:
    # even telling the gate that nothing may regress and anything inconclusive passes: safety is not negotiable
    result = invoke(
        ["gate", "--config", "llm-eval.v2.json", "--on-inconclusive", "pass", "--min-pass-rate", "0"], example
    )
    assert result.code == 4


def test_the_demo_pipeline_is_deterministic_and_leaks_nothing(example: Path) -> None:
    invoke(["run"], example)
    first = (example / "out" / "report.json").read_text()
    report = json.loads(first)
    assert "CANARY-7f3a91" not in first
    invoke(["run"], example)
    second = json.loads((example / "out" / "report.json").read_text())
    assert texts(report) == texts(second)


def test_every_grader_the_example_uses_is_exercised(example: Path) -> None:
    invoke(["run"], example)
    report = json.loads((example / "out" / "report.json").read_text())
    used = {g["type"] for c in report["cases"] for run in c["runs"] for g in run["grades"]}
    assert {
        "contains",
        "tool_calls",
        "forbidden_tools",
        "not_contains",
        "grounding",
        "numeric",
        "token_f1",
        "secret_leak",
        "pii_leak",
        "refusal",
        "cost_budget",
        "latency_budget",
    } <= used
