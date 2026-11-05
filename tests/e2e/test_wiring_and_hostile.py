"""The wiring examples of docs/wiring.md, and hostile pipelines through the whole command line."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from support import programs
from support.cli import invoke, write_dataset, write_json
from support.programs import write_program

pytestmark = pytest.mark.slow
WIRING = Path(__file__).resolve().parents[2] / "examples" / "wiring"
REQUEST = {
    "protocol": 1,
    "case": {"id": "x", "input": {"question": "hello there"}, "context": [{"id": "d", "text": "t"}]},
    "repeat": 0,
    "attempt": 1,
    "seed": None,
}
EXPECTED = {
    "text": "You asked: hello there",
    "tool_calls": [],
    "citations": ["d"],
    "usage": {"model": "my-model", "input_tokens": 2, "output_tokens": 5},
}


def test_the_python_wiring_example_answers_the_protocol() -> None:
    done = subprocess.run(
        [sys.executable, str(WIRING / "pipeline.py")],
        input=json.dumps(REQUEST),
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(done.stdout) == EXPECTED


@pytest.mark.skipif(
    shutil.which("node") is None, reason="Node is not installed here; the example was run in a Node container"
)
def test_the_node_wiring_example_answers_the_protocol() -> None:
    done = subprocess.run(
        ["node", str(WIRING / "pipeline.mjs")], input=json.dumps(REQUEST), capture_output=True, text=True, check=True
    )
    assert json.loads(done.stdout) == EXPECTED


def test_the_wiring_project_runs_end_to_end_including_the_judge(tmp_path: Path) -> None:
    work = tmp_path / "wiring"
    shutil.copytree(WIRING, work)
    for name in ("pipeline.py", "judge.py"):
        assert (work / name).exists()
    config = json.loads((work / "llm-eval.json").read_text())
    config["provider"]["argv"][0] = sys.executable
    (work / "llm-eval.json").write_text(json.dumps(config))
    cases = [json.loads(line) for line in (work / "cases.jsonl").read_text().splitlines()]
    for case in cases:
        for grader in case.get("graders", []):
            if grader["type"] == "judge":
                grader["argv"][0] = sys.executable
    write_dataset(work / "cases.jsonl", *cases)
    result = invoke(["gate"], work)
    assert result.code == 0, result.out + result.err
    report = json.loads((work / "out" / "report.json").read_text())
    judged = [g for c in report["cases"] for r in c["runs"] for g in r["grades"] if g["type"] == "judge"]
    assert (
        judged
        and judged[0]["deterministic"] is False
        and judged[0]["confidence"] == 0.9
        and judged[0]["verdict"] == "pass"
    )


def hostile_project(tmp_path: Path, source: str, *, timeout: float = 20.0, **provider: object) -> Path:
    write_dataset(tmp_path / "cases.jsonl", {"id": "q", "input": "hello", "expected": {"facts": ["hello"]}})
    argv = write_program(tmp_path, "pipeline", source)
    write_json(
        tmp_path / "llm-eval.json",
        {
            "dataset": "cases.jsonl",
            "provider": {"type": "command", "argv": argv, **provider},
            "defaults": {"retries": 0, "timeout_s": timeout},
        },
    )
    return tmp_path


@pytest.mark.parametrize(
    ("source", "kind"),
    [
        (programs.INVALID_JSON, "protocol"),
        (programs.NOT_AN_OBJECT, "protocol"),
        (programs.BINARY, "protocol"),
        (programs.DEEP_JSON, "protocol"),
        (programs.EXIT_NONZERO, "exit"),
        (programs.SELF_KILL, "exit"),
        (programs.UNKNOWN_KEY, "protocol"),
    ],
)
def test_a_broken_pipeline_is_exit_3_with_the_reason_in_the_report(tmp_path: Path, source: str, kind: str) -> None:
    project = hostile_project(tmp_path, source)
    result = invoke(["gate"], project)
    assert result.code == 3, result.out + result.err
    assert f"provider error ({kind})" in result.out and "\x1b" not in result.out
    report = json.loads((project / "out" / "report.json").read_text())
    assert report["cases"][0]["runs"][0]["error"]["kind"] == kind and report["summary"]["errored_runs"] == 1


def test_a_hung_pipeline_is_killed_and_the_gate_still_answers(tmp_path: Path) -> None:
    project = hostile_project(tmp_path, programs.HANG, timeout=0.5)
    result = invoke(["gate"], project)  # the program would sleep for ten minutes
    assert result.code == 3 and "provider error (timeout)" in result.out


def test_an_endless_pipeline_is_cut_at_the_cap(tmp_path: Path) -> None:
    project = hostile_project(tmp_path, programs.HUGE_STDOUT, max_output_bytes=4096)
    result = invoke(["gate"], project)  # the program never stops writing by itself
    assert result.code == 3 and "more than 4096 bytes" in result.out


def test_hostile_text_in_an_answer_is_graded_stored_safely_and_printed_safely(tmp_path: Path) -> None:
    source = (
        "import json\nprint(json.dumps({'text': 'hello \\x1b[31mred\\x1b]0;pwned\\x07 \\u202e\\x00 end',"
        " 'structured': {'__proto__': {'admin': True}}}))\n"
    )
    project = hostile_project(tmp_path, source)
    result = invoke(["gate"], project)
    assert result.code == 0, result.out + result.err
    assert "\x1b" not in result.out and "\u202e" not in result.out
    stored = (project / "out" / "report.json").read_text()
    assert stored.isascii() and "\\u202e" in stored


def test_a_pipeline_that_tries_to_leak_a_secret_it_was_not_given_cannot_see_it(tmp_path: Path) -> None:
    source = (
        "import json, os\nprint(json.dumps({'text': 'hello ' + os.environ.get('CI_SECRET_SENTINEL', 'not-visible')}))\n"
    )
    project = hostile_project(tmp_path, source)
    result = invoke(["run"], project, environ={"PATH": "/usr/bin:/bin", "CI_SECRET_SENTINEL": "super-secret"})
    assert result.code == 0
    report = (project / "out" / "report.json").read_text()
    assert "not-visible" in report and "super-secret" not in report
