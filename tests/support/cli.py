"""Run the command line in process, with its own stdout, stderr and environment."""

from __future__ import annotations

import io
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from llm_eval_harness.composition.main import main


@dataclass(frozen=True, slots=True)
class Result:
    code: int
    out: str
    err: str

    def json(self, path: Path) -> dict[str, object]:
        data = json.loads(path.read_text())
        assert isinstance(data, dict)
        return data


def invoke(argv: list[str], cwd: Path, environ: dict[str, str] | None = None) -> Result:
    out, err = io.StringIO(), io.StringIO()
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")} if environ is None else environ
    code = main(argv, environ=env, cwd=str(cwd), stdout=out, stderr=err)
    return Result(code, out.getvalue(), err.getvalue())


def write_dataset(path: Path, *cases: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(c) + "\n" for c in cases))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
