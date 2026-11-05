from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from llm_eval_harness.domain.errors import ProviderError
from llm_eval_harness.infrastructure.process import ProcessResult, run_process
from support import programs
from support.programs import is_alive, write_program

pytestmark = pytest.mark.slow
ENV = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}


def run(
    argv: list[str],
    stdin: bytes = b"{}",
    *,
    timeout_s: float = 20.0,
    max_stdout: int = 1_000_000,
    max_stderr: int = 10_000,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
) -> ProcessResult:
    return run_process(
        argv,
        stdin,
        env=ENV if env is None else env,
        cwd=cwd,
        timeout_s=timeout_s,
        max_stdout=max_stdout,
        max_stderr=max_stderr,
    )


def wait_dead(pid: int, seconds: float = 5.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not is_alive(pid):
            return True
        time.sleep(0.05)
    return not is_alive(pid)


def test_stdin_to_stdout_round_trip(tmp_path: Path) -> None:
    result = run(write_program(tmp_path, "echo", programs.ECHO), json.dumps({"case": {"input": "hi"}}).encode())
    assert result.returncode == 0 and not result.timed_out and not result.stdout_overflow
    assert json.loads(result.stdout)["text"] == "echo: hi" and result.seconds > 0


def test_the_environment_is_exactly_what_was_passed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SENTINEL_SECRET_TOKEN", "do-not-leak-me")
    monkeypatch.setenv("HOME", "/home/someone")
    argv = write_program(tmp_path, "env", programs.ENV_DUMP)
    result = run(argv, env={"PATH": ENV["PATH"], "ONLY_THIS": "1"})
    seen = json.loads(json.loads(result.stdout)["text"])
    assert "SENTINEL_SECRET_TOKEN" not in seen and "HOME" not in seen and "do-not-leak-me" not in result.stdout.decode()
    assert seen["ONLY_THIS"] == "1" and set(seen) <= {"PATH", "ONLY_THIS", "LC_CTYPE", "PWD", "SHLVL", "_"}


def test_no_shell_arguments_are_passed_literally(tmp_path: Path) -> None:
    hostile = ["; echo pwned", "$(touch /tmp/pwned-by-test)", "`id`", "a b", "*", "-n"]
    result = run([*write_program(tmp_path, "argv", programs.ARGV_DUMP), *hostile])
    assert json.loads(json.loads(result.stdout)["text"]) == hostile
    assert not Path("/tmp/pwned-by-test").exists()


def test_no_inherited_file_descriptors(tmp_path: Path) -> None:
    leak = os.open(__file__, os.O_RDONLY)
    try:
        os.set_inheritable(leak, True)
        result = run(write_program(tmp_path, "fds", programs.FD_DUMP))
    finally:
        os.close(leak)
    fds = json.loads(json.loads(result.stdout)["text"])
    assert leak not in fds and set(fds) <= {0, 1, 2, 3}  # 3 is the directory handle of the listing itself


def test_working_directory(tmp_path: Path) -> None:
    sub = tmp_path / "work"
    sub.mkdir()
    result = run(write_program(tmp_path, "cwd", programs.CWD_DUMP), cwd=str(sub))
    assert os.path.realpath(json.loads(result.stdout)["text"]) == os.path.realpath(sub)


def test_endless_stdout_is_cut_at_the_cap_and_the_child_killed(tmp_path: Path) -> None:
    started = time.monotonic()
    result = run(write_program(tmp_path, "huge", programs.HUGE_STDOUT), max_stdout=100_000)
    assert result.stdout_overflow and len(result.stdout) == 100_000 and not result.timed_out
    assert time.monotonic() - started < 10


def test_huge_stderr_is_capped_too(tmp_path: Path) -> None:
    result = run(write_program(tmp_path, "err", programs.HUGE_STDERR), max_stderr=1000)
    assert result.returncode == 3 and len(result.stderr) == 1000 and not result.stdout_overflow


def test_a_hung_program_is_killed_at_the_deadline(tmp_path: Path) -> None:
    started = time.monotonic()
    result = run(write_program(tmp_path, "hang", programs.HANG), timeout_s=0.5)
    assert result.timed_out and result.returncode < 0
    assert 0.4 < time.monotonic() - started < 5


def test_a_program_that_never_reads_a_huge_request_cannot_wedge_the_harness(tmp_path: Path) -> None:
    started = time.monotonic()
    result = run(write_program(tmp_path, "deaf", programs.NEVER_READS_STDIN), b"x" * 5_000_000, timeout_s=0.5)
    assert result.timed_out and time.monotonic() - started < 8


def test_a_program_that_closes_stdin_early_is_fine(tmp_path: Path) -> None:
    result = run(write_program(tmp_path, "closer", programs.CLOSES_STDIN_AND_ANSWERS), b"y" * 2_000_000)
    assert result.returncode == 0 and json.loads(result.stdout)["text"] == "answered without reading"


def test_exit_status_and_signal_are_reported(tmp_path: Path) -> None:
    assert run(write_program(tmp_path, "fail", programs.EXIT_NONZERO)).returncode == 7
    killed = run(write_program(tmp_path, "kill", programs.SELF_KILL))
    assert killed.returncode == -9 and not killed.timed_out


def test_a_grandchild_left_behind_is_killed_with_the_group(tmp_path: Path) -> None:
    pid_file = tmp_path / "pid"
    result = run([*write_program(tmp_path, "orphan", programs.ORPHAN_MAKER), str(pid_file)])
    assert result.returncode == 0
    pid = int(pid_file.read_text())
    assert wait_dead(pid), "the background child outlived the call"


def test_a_timeout_takes_the_whole_process_tree(tmp_path: Path) -> None:
    pid_file = tmp_path / "pid"
    result = run([*write_program(tmp_path, "tree", programs.HANG_WITH_CHILD), str(pid_file)], timeout_s=1.0)
    assert result.timed_out
    assert wait_dead(int(pid_file.read_text())), "the grandchild survived the timeout"


def test_a_missing_or_unrunnable_program_is_a_spawn_error(tmp_path: Path) -> None:
    with pytest.raises(ProviderError) as missing:
        run(["/no/such/program-xyz"])
    assert missing.value.kind == "spawn" and not missing.value.retryable
    plain = tmp_path / "not-executable"
    plain.write_text("#!/bin/sh\n")
    plain.chmod(0o644)
    with pytest.raises(ProviderError) as denied:
        run([str(plain)])
    assert denied.value.kind == "spawn"
    with pytest.raises(ProviderError):
        run(["/no/such/dir/prog"], cwd=None)
    with pytest.raises(ProviderError):
        run(["python3-does-not-exist"])


def zombie_children() -> list[int]:
    found = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            fields = Path(f"/proc/{entry}/stat").read_text().rsplit(")", 1)[1].split()
        except OSError:
            continue
        if fields[0] == "Z" and int(fields[1]) == os.getpid():
            found.append(int(entry))
    return found


def test_no_zombies_are_left_after_many_runs(tmp_path: Path) -> None:
    argv = write_program(tmp_path, "echo", programs.ECHO)
    for _ in range(15):
        run(argv, json.dumps({"case": {"input": "x"}}).encode())
    run([*write_program(tmp_path, "orphan", programs.ORPHAN_MAKER), str(tmp_path / "pid")])
    assert zombie_children() == []
