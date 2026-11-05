"""The command line, end to end in process: real files in a temporary project, the deterministic fake provider."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from llm_eval_harness import __version__
from support import keys
from support.cli import invoke, write_dataset, write_json

QUALITY = {"id": "q1", "input": "returns?", "expected": {"facts": ["30 days"]}}
SAFETY = {"id": "s1", "category": "safety", "input": "attack", "expected": {"forbidden_tools": ["issue_refund"]}}


def config(provider: Mapping[str, object] | None = None, **extra: object) -> dict[str, object]:
    base: dict[str, object] = {
        "dataset": "cases.jsonl",
        "provider": provider or {"type": "fake", "script": {"q1": {"text": "30 days"}, "s1": {"text": "no"}}},
    }
    base.update(extra)
    return base


@pytest.fixture
def project(tmp_path: Path) -> Path:
    write_dataset(tmp_path / "cases.jsonl", QUALITY, SAFETY)
    write_json(tmp_path / "llm-eval.json", config())
    return tmp_path


def test_version_and_help(tmp_path: Path) -> None:
    assert invoke(["--version"], tmp_path).out == f"llm-eval {__version__}\n"
    top = invoke(["--help"], tmp_path)
    assert top.code == 0 and "exit codes" in top.out and "init-safety" in top.out
    for command in ("run", "compare", "gate", "baseline", "validate", "init-safety"):
        result = invoke([command, "--help"], tmp_path)
        assert result.code == 0 and result.out.startswith(f"usage: llm-eval {command}"), command
    assert invoke([], tmp_path).code == 2


@pytest.mark.parametrize(
    ("argv", "fragment"),
    [
        (["bogus"], "invalid choice"),
        (["run", "--repeat", "x"], "not a whole number"),
        (["run", "--repeat", "0"], "--repeat must be positive"),
        (["run", "--timeout", "nan"], "must be finite"),
        (["run", "--retries", "-1"], "--retries must not be negative"),
        (["run", "--sample", "0"], "--sample must be at least 1"),
        (["run", "--unknown"], "unrecognized arguments"),
        (["gate", "--on-inconclusive", "maybe"], "invalid choice"),
        (["--root", "/definitely/not/there", "validate"], "is not a directory"),
        (["validate", "--dataset", "nope.jsonl"], "no such file"),
        (["validate", "--dataset", "../outside.jsonl"], "outside the project"),
    ],
)
def test_usage_errors_exit_2_with_a_message_on_stderr(project: Path, argv: list[str], fragment: str) -> None:
    (project.parent / "outside.jsonl").write_text("{}\n")
    result = invoke(argv, project)
    assert (
        result.code == 2 and fragment in result.err and result.err.startswith("llm-eval: error: ") and result.out == ""
    )


def test_validate_prints_counts_hash_and_tags(project: Path) -> None:
    result = invoke(["validate"], project)
    assert result.code == 0
    assert (
        "cases.jsonl: ok" in result.out and "cases    2 (1 quality, 1 safety)" in result.out and "sha256:" in result.out
    )


def test_validate_reports_every_dataset_problem_with_its_line(tmp_path: Path) -> None:
    (tmp_path / "bad.jsonl").write_text('{"id": "a"}\n{"id": "b", "input": 1, "expected": {"fact": []}}\nnot json\n')
    result = invoke(["validate", "--dataset", "bad.jsonl"], tmp_path)
    assert result.code == 2
    for fragment in (
        "line 1.input: is required",
        "line 2.expected.fact: unknown key",
        "line 3, column 1: invalid JSON",
    ):
        assert fragment in result.err, result.err


def test_run_writes_reports_prints_a_summary_and_exits_0(project: Path) -> None:
    result = invoke(["run"], project)
    assert result.code == 0, result.err
    assert "quality   mean pass rate 1.000 over 1 case" in result.out and "wrote out/report.json" in result.out
    report = result.json(project / "out" / "report.json")
    assert report["schema_version"] == 1 and report["kind"] == "llm-eval-report"
    assert (project / "out" / "report.md").read_text().startswith("## LLM eval report")


def test_run_exits_4_on_a_safety_failure_and_3_on_infrastructure_errors(project: Path) -> None:
    write_json(
        project / "bad-safety.json",
        config(
            {
                "type": "fake",
                "script": {
                    "q1": {"text": "30 days"},
                    "s1": {"text": "x", "tool_calls": [{"name": "issue_refund", "arguments": {}}]},
                },
            }
        ),
    )
    assert invoke(["run", "--config", "bad-safety.json"], project).code == 4
    write_json(
        project / "infra.json",
        config(
            {"type": "fake", "script": {"q1": {"error": {"kind": "timeout", "message": "slow"}}, "s1": {"text": "no"}}}
        ),
    )
    result = invoke(["run", "--config", "infra.json", "--retries", "0"], project)
    assert result.code == 3 and "provider error (timeout)" in result.out


def test_run_does_not_fail_on_quality_alone_but_gate_does(project: Path) -> None:
    write_json(
        project / "weak.json", config({"type": "fake", "script": {"q1": {"text": "no idea"}, "s1": {"text": "no"}}})
    )
    assert invoke(["run", "--config", "weak.json"], project).code == 0
    gate = invoke(["gate", "--config", "weak.json"], project)
    assert gate.code == 1 and "gate      FAIL (exit 1)" in gate.out and "below the minimum 1" in gate.out


def test_gate_runs_and_passes_and_writes_the_decision(project: Path) -> None:
    result = invoke(["gate"], project)
    assert result.code == 0 and "gate      PASS (exit 0)" in result.out
    decision = result.json(project / "out" / "gate.json")
    assert decision["kind"] == "llm-eval-gate" and decision["exit_code"] == 0 and decision["passed"] is True


def test_baseline_compare_and_gate_flow(project: Path) -> None:
    assert invoke(["run"], project).code == 0
    result = invoke(["baseline"], project)
    assert result.code == 0 and "wrote baseline.json" in result.out
    assert invoke(["baseline"], project).code == 2  # refuses to overwrite
    assert invoke(["baseline", "--force"], project).code == 0
    text = (project / "baseline.json").read_text()
    assert text.endswith("}\n") and '"q1"' in text and "30 days" not in text  # no outputs in the baseline
    write_json(project / "llm-eval.json", config(baseline="baseline.json", gate={"deterministic": True}))
    same = invoke(["compare"], project)
    assert same.code == 0 and "verdict   OK" in same.out
    write_json(
        project / "worse.json",
        config(
            {"type": "fake", "script": {"q1": {"text": "no idea"}, "s1": {"text": "no"}}},
            baseline="baseline.json",
            gate={"deterministic": True},
        ),
    )
    gate = invoke(["gate", "--config", "worse.json"], project)
    assert gate.code == 1 and "regressed    q1" in gate.out and "1 case(s) regressed" in gate.out
    decision = json.loads((project / "out" / "gate.json").read_text())
    assert decision["comparison"]["verdict"] == "regression" and decision["comparison"]["cases"][0]["id"] == "q1"


def test_without_deterministic_a_single_run_flip_is_inconclusive(project: Path) -> None:
    invoke(["run", "--write-baseline", "baseline.json"], project)
    write_json(
        project / "worse.json",
        config({"type": "fake", "script": {"q1": {"text": "no idea"}, "s1": {"text": "no"}}}, baseline="baseline.json"),
    )
    strict = invoke(["gate", "--config", "worse.json"], project)
    assert strict.code == 1 and "inconclusive" in strict.out and "--repeat" in strict.out
    lenient = invoke(["gate", "--config", "worse.json", "--on-inconclusive", "pass"], project)
    assert lenient.code == 0 and "cannot support a verdict" in lenient.out
    flagged = invoke(["gate", "--config", "worse.json", "--deterministic"], project)
    assert flagged.code == 1 and "regressed" in flagged.out


def test_repeats_give_flakiness_statistics_and_real_evidence(project: Path) -> None:
    flaky = {
        "type": "fake",
        "script": {"q1": [{"text": "30 days"}, {"text": "no"}, {"text": "30 days"}], "s1": {"text": "no"}},
    }
    write_json(project / "flaky.json", config(flaky))
    result = invoke(["run", "--config", "flaky.json", "--repeat", "3"], project)
    assert (
        "1 flaky case" in result.out
        and "flaky cases (passed some runs" in result.out
        and "pass rate 0.667 (2/3)" in result.out
    )
    report = result.json(project / "out" / "report.json")
    q1 = next(c for c in report["cases"] if c["id"] == "q1")  # type: ignore[attr-defined]
    assert q1["flaky"] is True and q1["passes"] == 2 and q1["graded_runs"] == 3 and q1["score_variance"] > 0


def test_gate_with_a_report_does_not_run_and_rejects_run_options(project: Path) -> None:
    invoke(["run"], project)
    ok = invoke(["gate", "--report", "out/report.json"], project)
    assert ok.code == 0
    clash = invoke(["gate", "--report", "out/report.json", "--repeat", "3"], project)
    assert clash.code == 2 and "cannot be combined" in clash.err


def test_filters_sampling_and_unknown_ids(project: Path) -> None:
    result = invoke(["run", "--category", "safety"], project)
    report = result.json(project / "out" / "report.json")
    assert [c["id"] for c in report["cases"]] == ["s1"]  # type: ignore[attr-defined]
    assert invoke(["run", "--case", "nope"], project).code == 2
    assert invoke(["run", "--tag", "nothing"], project).code == 2
    sampled = invoke(["run", "--sample", "1", "--seed", "3"], project)
    assert sampled.code == 0 and "(1 of 2 cases)" in sampled.out


def test_markdown_output_for_a_pull_request(project: Path) -> None:
    result = invoke(["gate", "--markdown", "pr.md"], project)
    assert result.code == 0 and "wrote pr.md" in result.out
    text = (project / "pr.md").read_text()
    assert text.startswith("## LLM eval: PASSED (exit 0)")


def test_init_safety_writes_a_dataset_that_validates_and_refuses_to_overwrite(tmp_path: Path) -> None:
    first = invoke(["init-safety"], tmp_path)
    assert first.code == 0 and "wrote datasets/safety.jsonl (10 safety cases)" in first.out
    assert invoke(["validate", "--dataset", "datasets/safety.jsonl"], tmp_path).code == 0
    assert invoke(["init-safety"], tmp_path).code == 2
    assert invoke(["init-safety", "--force", "--out", "elsewhere/s.jsonl"], tmp_path).code == 0


def test_no_provider_or_dataset_is_a_usage_error(tmp_path: Path) -> None:
    write_dataset(tmp_path / "cases.jsonl", QUALITY)
    no_provider = invoke(["run", "--dataset", "cases.jsonl"], tmp_path)
    assert no_provider.code == 2 and "no provider configured" in no_provider.err
    write_json(tmp_path / "llm-eval.json", {"provider": {"type": "fake"}})
    assert "dataset" in invoke(["run"], tmp_path).err


def test_a_broken_configuration_names_the_key(tmp_path: Path) -> None:
    write_json(tmp_path / "llm-eval.json", {"provider": {"type": "command"}, "gate": {"alpha": 7}})
    result = invoke(["run"], tmp_path)
    assert result.code == 2 and "provider.argv: is required" in result.err and "gate.alpha" in result.err


def test_output_and_inputs_cannot_leave_the_project(project: Path) -> None:
    assert invoke(["run", "--out", "../escaped"], project).code == 2
    assert invoke(["run", "--out", "/tmp/llm-eval-escape-test"], project).code == 2
    assert invoke(["gate", "--markdown", "../pr.md"], project).code == 2
    assert not (project.parent / "escaped").exists()


def test_secret_shaped_strings_are_redacted_in_stored_reports_unless_switched_off(project: Path) -> None:
    token = keys.github_token()
    write_json(
        project / "leaky.json",
        config({"type": "fake", "script": {"q1": {"text": f"30 days {token}"}, "s1": {"text": "no"}}}),
    )
    invoke(["run", "--config", "leaky.json"], project)
    stored = (project / "out" / "report.json").read_text() + (project / "out" / "report.md").read_text()
    assert token not in stored and "REDACTED:github-token" in stored
    invoke(["run", "--config", "leaky.json", "--no-redact"], project)
    assert token in (project / "out" / "report.json").read_text()
    write_json(
        project / "off.json",
        config(
            {"type": "fake", "script": {"q1": {"text": f"30 days {token}"}, "s1": {"text": "no"}}},
            redact={"enabled": False},
        ),
    )
    result = invoke(["run", "--config", "off.json"], project)
    assert "redaction OFF" in result.out


def test_terminal_and_error_output_never_carry_control_sequences(project: Path) -> None:
    evil = "30 days \x1b[31mred\x1b]0;pwned\x07 \u202e"
    write_json(
        project / "evil.json",
        config(
            {
                "type": "fake",
                "script": {
                    "q1": {"text": evil, "tool_calls": [{"name": "t\x1b[1m", "arguments": {}}]},
                    "s1": {"text": "\x1b[2J no"},
                },
            }
        ),
    )
    result = invoke(["run", "--config", "evil.json"], project)
    assert "\x1b" not in result.out + result.err and "\u202e" not in result.out
    (project / "weird.jsonl").write_text('{"id": "a\\u001b[31m", "input": 1}\n')
    bad = invoke(["validate", "--dataset", "weird.jsonl"], project)
    assert "\x1b" not in bad.err


def test_command_provider_through_the_cli(project: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    import sys

    script = project / "pipeline.py"
    script.write_text(
        "import json,sys\nr=json.load(sys.stdin)\nprint(json.dumps({'text': '30 days', 'usage': {'model': 'm', 'input_tokens': 5, 'output_tokens': 3}}))\n"
    )
    write_json(project / "prices.json", {"models": {"m": {"input_per_mtok": 1000, "output_per_mtok": 2000}}})
    write_json(
        project / "cmd.json",
        config(
            {"type": "command", "argv": [sys.executable, "pipeline.py"]},
            price_table="prices.json",
            graders=[{"type": "cost_budget", "max_usd": 0.1}],
        ),
    )
    result = invoke(["run", "--config", "cmd.json", "--category", "quality"], project)
    assert result.code == 0 and "cost      0.0110 USD total" in result.out, result.out
