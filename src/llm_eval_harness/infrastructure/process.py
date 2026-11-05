"""Running a user's program safely: the only module that creates processes.

What it guarantees, for any program (including hostile ones):

* the argument vector is used as given, never through a shell;
* the child gets exactly the environment it is handed, nothing inherited;
* it runs in its own session, so a timeout, an overflow or its exit kills the whole process group, children
  included, and nothing in that group is left running or waiting to be reaped;
* the group is killed *before* the child is reaped (``waitid`` with ``WNOWAIT``, where the platform has it), so the
  pid cannot be recycled by an unrelated process in between;
* there are no threads and no blocking calls: stdin is written, and stdout and stderr are read, with
  non-blocking ``os.read``/``os.write`` driven by ``selectors`` against one deadline. Output is capped while it is
  read (the program is killed at the cap), so memory stays bounded however much it prints, and a program that never
  reads its input or never answers is killed at the deadline. The call cannot block on a pipe;
* file descriptors are closed without waiting for anyone.

What it cannot do: a process that **leaves the session** (``setsid``, a double fork with ``setsid``) is outside the
group and is not killed. The harness still returns on time (it does not wait for the pipes to reach end of file once
the program has exited, and it never waits for a process it did not start), but such a helper keeps running until
it ends by itself. Run pipelines you do not trust in a container. See SECURITY.md.
"""

from __future__ import annotations

import contextlib
import os
import selectors
import signal
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from llm_eval_harness.domain.errors import ProviderError

CHUNK = 65536
POLL_S = 0.05
REAP_GRACE_S = 5.0


@dataclass(frozen=True, slots=True)
class ProcessResult:
    returncode: int
    stdout: bytes
    stderr: bytes
    timed_out: bool
    stdout_overflow: bool
    seconds: float


def _exited(proc: subprocess.Popen[bytes]) -> bool:
    """Whether the child has ended, without reaping it where possible (so its pid stays reserved)."""
    waitid = getattr(os, "waitid", None)
    if waitid is None:  # macOS: no WNOWAIT; poll() reaps, and the pid-reuse window is the tiny one documented
        return proc.poll() is not None
    try:
        return waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None
    except ChildProcessError:
        return True


def _kill_group(proc: subprocess.Popen[bytes]) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGKILL)
    with contextlib.suppress(ProcessLookupError):
        proc.kill()


def _close(proc: subprocess.Popen[bytes]) -> None:
    for stream in (proc.stdin, proc.stdout, proc.stderr):
        if stream is not None:
            with contextlib.suppress(OSError, ValueError):
                stream.close()


class _Pump:
    """Moves bytes between the child's pipes and memory, never blocking."""

    def __init__(self, proc: subprocess.Popen[bytes], stdin: bytes, max_stdout: int, max_stderr: int) -> None:
        assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
        self._stdin_stream = proc.stdin
        self.selector = selectors.DefaultSelector()
        self.out = bytearray()
        self.err = bytearray()
        self.limits = {"out": max_stdout, "err": max_stderr}
        self.overflow = False
        self._stdin_fd = proc.stdin.fileno()
        self._out_fd = proc.stdout.fileno()
        self._err_fd = proc.stderr.fileno()
        self._pending = memoryview(stdin)
        for fd in (self._stdin_fd, self._out_fd, self._err_fd):
            os.set_blocking(fd, False)
        self._open: set[int] = set()
        if stdin:
            self._register(self._stdin_fd, selectors.EVENT_WRITE)
        else:
            self._close_stdin()
        self._register(self._out_fd, selectors.EVENT_READ)
        self._register(self._err_fd, selectors.EVENT_READ)

    def _register(self, fd: int, events: int) -> None:
        self.selector.register(fd, events)
        self._open.add(fd)

    def _unregister(self, fd: int) -> None:
        if fd in self._open:
            self._open.discard(fd)
            with contextlib.suppress(KeyError, ValueError, OSError):
                self.selector.unregister(fd)

    def _close_stdin(self) -> None:
        self._unregister(self._stdin_fd)
        # close through the stream object, never `os.close(fd)`: the object would close the same number again later,
        # by which time another thread may own it
        with contextlib.suppress(OSError, ValueError):
            self._stdin_stream.close()
        self._stdin_fd = -1

    def step(self, timeout: float) -> None:
        for key, _ in self.selector.select(timeout):
            if key.fd == self._stdin_fd:
                self._write()
            else:
                self._read(key.fd)

    def drain(self) -> None:
        """Read what is already in the pipes (the program has exited); stop at once if nothing is."""
        for fd in (self._out_fd, self._err_fd):
            while fd in self._open and not self.overflow and self._read(fd):
                pass

    def _write(self) -> None:
        try:
            written = os.write(self._stdin_fd, self._pending[:CHUNK])
        except BlockingIOError:
            return
        except OSError:  # the program closed its end (BrokenPipeError) or died: stop feeding it
            self._close_stdin()
            return
        self._pending = self._pending[written:]
        if not self._pending:
            self._close_stdin()

    def _read(self, fd: int) -> bool:
        """Read one chunk. Returns True if data arrived (more may follow), False on end of file or no data."""
        try:
            chunk = os.read(fd, CHUNK)
        except BlockingIOError:
            return False
        except OSError:
            self._unregister(fd)
            return False
        if not chunk:
            self._unregister(fd)
            return False
        into, limit = (self.out, self.limits["out"]) if fd == self._out_fd else (self.err, self.limits["err"])
        room = limit - len(into)
        if len(chunk) > room:
            into += chunk[: max(room, 0)]
            if fd == self._out_fd:
                self.overflow = True
            # stderr beyond its cap is discarded but still drained, so the program never blocks on a full pipe
        else:
            into += chunk
        return True

    def close(self) -> None:
        self.selector.close()


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
            bufsize=0,
        )
    except OSError as error:
        name = os.path.basename(argv[0])
        raise ProviderError(
            "spawn", f"cannot start {name!r}: {error.strerror or type(error).__name__}", retryable=False
        ) from None
    pump: _Pump | None = None
    timed_out = False
    try:
        pump = _Pump(proc, stdin, max_stdout, max_stderr)
        deadline = started + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            pump.step(min(remaining, POLL_S))
            if pump.overflow:
                break
            if _exited(proc):
                pump.drain()
                break
    finally:
        # whatever happened (also an exception above): nothing in the child's process group outlives this call
        _kill_group(proc)
        if pump is not None:
            pump.close()
        _close(proc)
        with contextlib.suppress(subprocess.TimeoutExpired):  # a process in uninterruptible sleep: leave it, never hang
            proc.wait(REAP_GRACE_S)
    returncode = proc.returncode if proc.returncode is not None else -signal.SIGKILL
    return ProcessResult(
        returncode, bytes(pump.out), bytes(pump.err), timed_out, pump.overflow, time.monotonic() - started
    )
