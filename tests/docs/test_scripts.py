"""The logic of the small scripts that are not part of the package: lock generation, the age check, coverage layers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("wheel", "interpreter", "arch", "expected"),
    [
        ("pkg-1.0-py3-none-any.whl", "cp313", "x86_64", True),
        ("pkg-1.0-py2.py3-none-any.whl", "cp312", "aarch64", True),
        ("pkg-1.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl", "cp313", "x86_64", True),
        ("pkg-1.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl", "cp313", "aarch64", False),
        ("pkg-1.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl", "cp312", "x86_64", False),
        ("pkg-1.0-cp39-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl", "cp312", "aarch64", True),
        ("pkg-1.0-cp313-abi3-manylinux_2_17_aarch64.whl", "cp312", "aarch64", False),
        ("pkg-1.0-py3-none-manylinux_2_17_x86_64.whl", "cp313", "x86_64", True),
        ("pkg-1.0-py3-none-macosx_11_0_arm64.whl", "cp313", "aarch64", False),
        ("pkg-1.0-cp313-cp313-musllinux_1_2_aarch64.whl", "cp313", "aarch64", True),
        ("pkg-1.0-cp313-cp313t-manylinux_2_17_x86_64.whl", "cp313", "x86_64", False),
        ("pkg-1.0.tar.gz", "cp313", "x86_64", False),
    ],
)
def test_wheel_selection_for_the_hash_lock(wheel: str, interpreter: str, arch: str, expected: bool) -> None:
    assert load("lock-dev").covers(wheel, interpreter, arch) is expected


def test_the_age_check_reads_pins_and_hashes() -> None:
    check = load("check-dep-age")
    digest = "a" * 64
    text = f"# comment\nfoo==1.0 \\\n    --hash=sha256:{digest} \\\n    --hash=sha256:{'b' * 64}\nbar-baz==2.1.0 \\\n    --hash=sha256:{digest}\n"
    pins = check.parse(text)
    assert pins == {("foo", "1.0"): [digest, "b" * 64], ("bar-baz", "2.1.0"): [digest]}
    with pytest.raises(ValueError, match="cannot read line"):
        check.parse("foo>=1.0\n")


def test_the_real_lock_parses_and_every_pin_has_hashes() -> None:
    check = load("check-dep-age")
    pins = check.parse((SCRIPTS.parent / "requirements-dev.txt").read_text())
    assert len(pins) >= 10 and all(hashes for hashes in pins.values())


def test_coverage_layers_are_taken_from_the_path() -> None:
    coverage = load("coverage_check")
    assert coverage.layer_of("src/llm_eval_harness/domain/text.py") == "domain"
    assert coverage.layer_of("/work/src/llm_eval_harness/infrastructure/providers/http.py") == "infrastructure"
    assert coverage.layer_of("src/llm_eval_harness/__main__.py") == "entry"
    assert coverage.layer_of("elsewhere/x.py") == "other"
    assert coverage.THRESHOLDS == {"domain": 95.0, "application": 90.0, "overall": 85.0}
