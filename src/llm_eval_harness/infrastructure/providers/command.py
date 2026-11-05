"""The ``command`` provider: any program, in any language, that reads one JSON request on stdin and writes one
JSON answer on stdout. The harness needs no SDK and no knowledge of what is inside the program.
"""

from __future__ import annotations

import json
import os
import signal
from collections.abc import Mapping
from pathlib import Path

from llm_eval_harness.application.config import CommandProviderConfig
from llm_eval_harness.domain.errors import ProviderError
from llm_eval_harness.domain.grading.judge import DEFAULT_ENV_ALLOW
from llm_eval_harness.domain.jsonio import JsonValue
from llm_eval_harness.domain.limits import MAX_STDERR_BYTES
from llm_eval_harness.domain.model import ProviderOutput
from llm_eval_harness.infrastructure.fs import ProjectFilesystem
from llm_eval_harness.infrastructure.process import ProcessResult, run_process
from llm_eval_harness.infrastructure.providers.common import decode_answer, stderr_excerpt


def build_env(environ: Mapping[str, str], allow: tuple[str, ...], fixed: tuple[tuple[str, str], ...]) -> dict[str, str]:
    """The child's environment: the default allow-list plus the variables the user named, plus fixed values.

    Nothing else is passed on. ``environ`` is a snapshot handed in by the composition layer.
    """
    env = {name: environ[name] for name in (*DEFAULT_ENV_ALLOW, *allow) if name in environ}
    env.update(dict(fixed))
    env["LLM_EVAL"] = "1"
    return env


def describe_exit(result: ProcessResult) -> str:
    if result.returncode < 0:
        try:
            name = signal.Signals(-result.returncode).name
        except ValueError:
            name = str(-result.returncode)
        return f"killed by signal {name}"
    return f"exited with status {result.returncode}"


def interpret(result: ProcessResult, *, timeout_s: float, max_bytes: int, source: str) -> bytes:
    """The stdout of a finished run, or the ``ProviderError`` that its outcome amounts to."""
    if result.timed_out:
        raise ProviderError("timeout", f"{source} gave no answer within {timeout_s:g}s", retryable=True)
    if result.stdout_overflow:
        raise ProviderError("protocol", f"{source} wrote more than {max_bytes} bytes to stdout", retryable=False)
    if result.returncode != 0:
        raise ProviderError(
            "exit", f"{source} {describe_exit(result)}: {stderr_excerpt(result.stderr)}", retryable=True
        )
    return result.stdout


class CommandProvider:
    def __init__(self, config: CommandProviderConfig, root: Path, environ: Mapping[str, str]) -> None:
        self._argv = config.argv
        self._max = config.max_output_bytes
        self._env = build_env(environ, config.env_allow, config.env_set)
        self._cwd = str(ProjectFilesystem(root).resolve_directory(config.cwd)) if config.cwd else str(root.resolve())

    def describe(self) -> dict[str, JsonValue]:
        return {"type": "command", "program": os.path.basename(self._argv[0]), "arguments": len(self._argv) - 1}

    def generate(self, request: dict[str, JsonValue], timeout_s: float) -> ProviderOutput:
        body = json.dumps(request, ensure_ascii=True, allow_nan=False).encode("ascii")
        result = run_process(
            self._argv,
            body,
            env=self._env,
            cwd=self._cwd,
            timeout_s=timeout_s,
            max_stdout=self._max,
            max_stderr=MAX_STDERR_BYTES,
        )
        raw = interpret(result, timeout_s=timeout_s, max_bytes=self._max, source="the provider program")
        return decode_answer(raw, source="the provider program", latency_ms=result.seconds * 1000.0)
