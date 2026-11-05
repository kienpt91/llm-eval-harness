"""The judge runner: an external program decides a ``judge`` grader. Same process guarantees as the provider."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path

from llm_eval_harness.domain.grading.judge import JudgeConfig
from llm_eval_harness.domain.jsonio import JsonValue
from llm_eval_harness.domain.limits import MAX_STDERR_BYTES
from llm_eval_harness.infrastructure.process import run_process
from llm_eval_harness.infrastructure.providers.command import build_env, interpret
from llm_eval_harness.infrastructure.providers.common import decode_json

MAX_JUDGE_OUTPUT = 256 * 1024


class CommandJudge:
    def __init__(self, root: Path, environ: Mapping[str, str]) -> None:
        self._cwd = str(root.resolve())
        self._environ = dict(environ)

    def run(self, config: JudgeConfig, request: dict[str, JsonValue]) -> JsonValue:
        env = build_env(self._environ, config.env_allow, ())
        body = json.dumps(request, ensure_ascii=True, allow_nan=False).encode("ascii")
        result = run_process(
            config.argv,
            body,
            env=env,
            cwd=self._cwd,
            timeout_s=config.timeout_s,
            max_stdout=MAX_JUDGE_OUTPUT,
            max_stderr=MAX_STDERR_BYTES,
        )
        source = f"the judge program {os.path.basename(config.argv[0])!r}"
        raw = interpret(result, timeout_s=config.timeout_s, max_bytes=MAX_JUDGE_OUTPUT, source=source)
        return decode_json(raw, source=source)
