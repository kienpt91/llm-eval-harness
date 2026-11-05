"""The built distribution: build a wheel from a copy of the sources, install it, run the real ``llm-eval`` script."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def installed(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    work = tmp_path_factory.mktemp("wheel")
    source = work / "src"
    source.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy(ROOT / name, source / name)
    shutil.copytree(ROOT / "src", source / "src", ignore=shutil.ignore_patterns("__pycache__"))
    dist = work / "dist"
    env = {**os.environ, "SOURCE_DATE_EPOCH": "1700000000"}
    subprocess.run(
        [sys.executable, "-m", "build", "--no-isolation", "--wheel", "--outdir", str(dist), str(source)],
        check=True,
        capture_output=True,
        env=env,
    )
    [wheel] = dist.glob("*.whl")
    target = work / "site"
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--no-index", "--target", str(target), str(wheel)],
        check=True,
        capture_output=True,
    )
    return wheel, target


def test_the_wheel_ships_the_package_the_licence_and_nothing_else(installed: tuple[Path, Path]) -> None:
    wheel, _ = installed
    names = zipfile.ZipFile(wheel).namelist()
    assert any(n.endswith("llm_eval_harness/py.typed") for n in names)
    assert any(n.endswith("domain/grading/registry.py") for n in names)
    assert any("licenses/LICENSE" in n or n.endswith("LICENSE") for n in names)
    assert not [n for n in names if n.startswith(("tests", "examples", "scripts", "docs"))]
    metadata = next(n for n in names if n.endswith("METADATA"))
    text = zipfile.ZipFile(wheel).read(metadata).decode()
    assert "Requires-Python: >=3.12" in text and "Requires-Dist" not in text  # zero runtime dependencies
    assert "License-Expression: MIT" in text and "Name: llm-eval-harness" in text


def test_the_installed_console_script_runs(installed: tuple[Path, Path], tmp_path: Path) -> None:
    _, target = installed
    script = target / "bin" / "llm-eval"
    assert script.exists()
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": str(target)}
    version = subprocess.run([str(script), "--version"], capture_output=True, text=True, env=env, check=False)
    assert version.returncode == 0 and version.stdout.startswith("llm-eval 0.")
    init = subprocess.run(
        [str(script), "init-safety"], cwd=tmp_path, capture_output=True, text=True, env=env, check=False
    )
    assert init.returncode == 0 and (tmp_path / "datasets" / "safety.jsonl").exists()
    bad = subprocess.run(
        [str(script), "validate", "--dataset", "../nope.jsonl"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert bad.returncode == 2 and bad.stdout == ""


def test_python_dash_m_works_and_the_exit_code_is_the_gate_code(installed: tuple[Path, Path], tmp_path: Path) -> None:
    _, target = installed
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": str(target)}
    (tmp_path / "cases.jsonl").write_text(
        '{"id": "s", "category": "safety", "input": "x", "expected": {"forbidden_tools": ["bad"]}}\n'
    )
    (tmp_path / "llm-eval.json").write_text(
        '{"dataset": "cases.jsonl", "provider": {"type": "fake", "default": {"text": "x", "tool_calls": [{"name": "bad", "arguments": {}}]}}}'
    )
    result = subprocess.run(
        [sys.executable, "-m", "llm_eval_harness", "gate"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 4 and "gate      FAIL (exit 4)" in result.stdout
