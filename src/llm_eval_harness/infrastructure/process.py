"""Running a user's program safely: the only module that creates processes.

What it guarantees, for any program (including hostile ones):

* the argument vector is used as given, never through a shell;
* the child gets exactly the environment it is handed, nothing inherited;
* it runs in its own session, so a timeout or an overflow kills its whole process group, children included,
  and nothing is left running or waiting to be reaped;
* stdout and stderr are read incrementally and capped: a program that prints without end is killed at the cap
  and uses at most the cap in memory;
* a program that never reads stdin or never answers is killed at the deadline.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import BufferedReader

from llm_eval_harness.domain.errors import ProviderError

CHUNK = 65536
JOIN_GRACE_S = 2.0


@dataclass(frozen=True, slots=True)
class ProcessResult:
    returncode: int
    stdout: bytes
    stderr: bytes
    timed_out: bool
    stdout_overflow: bool
    seconds: float


class _Capped:
    """Reads a pipe in a thread, keeping at most ``limit`` bytes and flagging when more arrived."""

    def __init__(self, stream: BufferedReader, limit: int, on_overflow: threading.Event | None) -> None:
        self._stream = stream
        self.limit = limit
        self.data = bytearray()
        self.overflow = False
        self._on_overflow = on_overflow
        self.thread = threading.Thread(target=self._pump, daemon=True)
        self.thread.start()

    def _pump(self) -> None:
        read = getattr(self._stream, "read1", None) or self._stream.read
        try:
            while True:
                chunk: bytes = read(CHUNK)
                if not chunk:
                    return
                room = self.limit - len(self.data)
                if len(chunk) > room:
                    self.data += chunk[: max(room, 0)]
                    self.overflow = True
                    if self._on_overflow is not None:
                        self._on_overflow.set()
                        return
                    continue  # nobody is asked to stop: keep draining and discard, so the child never blocks on a full pipe
                self.data += chunk
        except (OSError, ValueError):
            return


def _kill_group(proc: subprocess.Popen[bytes]) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGKILL)
    with contextlib.suppress(ProcessLookupError):
        proc.kill()


def run_process(
    argv: Sequence[str],
    stdin: bytes,
    *,
    env: Mapping[str, str],
    cwd: str | None,
    timeout_s: float,
    max_stdout: int,
    max_stderr: int,
) -> ProcessResult:
    """Run ``argv`` with ``stdin`` and return what it produced. Raises ``ProviderError('spawn')`` if it cannot start."""
    started = time.monotonic()
    try:
        proc = subprocess.Popen(  # noqa: S603 - argv is a list, shell is off, the program is the user's own
            list(argv),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=dict(env),
            cwd=cwd,
            shell=False,
            close_fds=True,
            start_new_session=True,
        )
    except OSError as error:
        raise ProviderError(
            "spawn",
            f"cannot start {os.path.basename(argv[0])!r}: {error.strerror or type(error).__name__}",
            retryable=False,
        ) from None
    assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
    overflow = threading.Event()
    assert isinstance(proc.stdout, BufferedReader) and isinstance(proc.stderr, BufferedReader)
    out = _Capped(proc.stdout, max_stdout, overflow)
    err = _Capped(proc.stderr, max_stderr, None)

    def feed() -> None:
        try:
            proc.stdin.write(stdin)  # type: ignore[union-attr]
        except (BrokenPipeError, OSError, ValueError):
            pass
        finally:
            with contextlib.suppress(OSError, ValueError):
                proc.stdin.close()  # type: ignore[union-attr]

    writer = threading.Thread(target=feed, daemon=True)
    writer.start()
    deadline = started + timeout_s
    timed_out = False
    try:
        while proc.poll() is None:
            if overflow.is_set():
                break
            if time.monotonic() >= deadline:
                timed_out = True
                break
            overflow.wait(0.01)
    finally:
        # whatever happened, nothing of the child's process group may outlive this call
        _kill_group(proc)
        proc.wait()
        for thread in (writer, out.thread, err.thread):
            thread.join(JOIN_GRACE_S)
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            with contextlib.suppress(OSError, ValueError):
                stream.close()
    return ProcessResult(
        proc.returncode, bytes(out.data), bytes(err.data), timed_out, out.overflow, time.monotonic() - started
    )
