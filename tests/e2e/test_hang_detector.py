"""The suite's one wall-clock limit really stops a stuck test, with a traceback, instead of waiting for the CI job."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow
PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def test_the_configured_limit_is_generous_and_ends_the_process() -> None:
    text = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(r"^faulthandler_timeout = (\d+)$", text, flags=re.MULTILINE)
    assert match is not None and int(match.group(1)) >= 120  # far above any honest test, even on a loaded runner
    assert re.search(r"^faulthandler_exit_on_timeout = true$", text, flags=re.MULTILINE)


def test_a_stuck_test_is_killed_with_a_traceback(tmp_path: Path) -> None:
    (tmp_path / "test_stuck.py").write_text(
        "import time\n\n\ndef test_stuck():\n    time.sleep(120)\n", encoding="utf-8"
    )
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--import-mode=importlib"]
    command += ["-o", "faulthandler_timeout=2", "-o", "faulthandler_exit_on_timeout=true", str(tmp_path)]
    done = subprocess.run(
        command,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=110,  # far below the test's own 120 s sleep and the CI job: if the detector failed, this would fire
        check=False,
    )
    assert done.returncode != 0
    assert "Timeout" in done.stderr and "test_stuck" in done.stderr
