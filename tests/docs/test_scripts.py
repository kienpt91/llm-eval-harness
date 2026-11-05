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


# --- scripts/check-dep-age.py against a fake package index on loopback ----------------------------------------------------


class FakeIndex:
    """Serves ``/pypi/NAME/VERSION/json`` from a dict, like the real index's JSON API."""

    def __init__(self, releases: dict[tuple[str, str], list[dict[str, object]]]) -> None:
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        outer = releases

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                return

            def do_GET(self) -> None:
                parts = self.path.strip("/").split("/")
                files = outer.get((parts[1], parts[2])) if len(parts) == 4 and parts[3] == "json" else None
                if files is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                import json as _json

                body = _json.dumps({"urls": files}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)


def release_file(digest: str, uploaded: str, *, yanked: bool = False) -> dict[str, object]:
    return {
        "filename": f"x-{digest[:4]}.whl",
        "digests": {"sha256": digest},
        "upload_time_iso_8601": uploaded,
        "yanked": yanked,
    }


def lock_files(tmp_path: Path, pins: dict[str, str], hashes: dict[str, list[str]]) -> tuple[Path, Path]:
    lock = tmp_path / "lock.txt"
    lock.write_text(
        "".join(
            f"{name}=={version} \\\n" + " \\\n".join(f"    --hash=sha256:{h}" for h in hashes[name]) + "\n"
            for name, version in pins.items()
        )
    )
    declared = tmp_path / "in.txt"
    declared.write_text("".join(f"{name}=={version}\n" for name, version in pins.items()))
    return lock, declared


H1, H2, H3 = "1" * 64, "2" * 64, "3" * 64


def run_check(
    tmp_path: Path, index: FakeIndex, pins: dict[str, str], hashes: dict[str, list[str]], cutoff: str = "2026-10-29"
) -> int:
    lock, declared = lock_files(tmp_path, pins, hashes)
    check = load("check-dep-age")
    return int(
        check.main(["check-dep-age.py", cutoff, "--index", index.url, "--lock", str(lock), "--input", str(declared)])
    )


def test_the_age_check_accepts_old_files_and_names_the_ones_that_are_too_new(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    index = FakeIndex(
        {
            ("a", "1.0"): [release_file(H1, "2026-10-01T00:00:00Z")],
            ("b", "2.0"): [release_file(H2, "2026-10-28T00:00:00Z"), release_file(H3, "2026-10-30T00:00:00Z")],
        }
    )
    try:
        assert run_check(tmp_path, index, {"a": "1.0"}, {"a": [H1]}) == 0
        assert "No file is newer than the cutoff" in capsys.readouterr().out
        assert run_check(tmp_path, index, {"b": "2.0"}, {"b": [H2, H3]}) == 1
        assert "uploaded 2026-10-30" in capsys.readouterr().out
        assert (
            run_check(tmp_path, index, {"b": "2.0"}, {"b": [H2]}, cutoff="2026-10-27") == 1
        )  # now H2 is too new as well
    finally:
        index.close()


def test_the_age_check_refuses_unknown_hashes_yanked_files_and_missing_releases(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    index = FakeIndex(
        {
            ("a", "1.0"): [release_file(H1, "2026-10-01T00:00:00Z", yanked=True)],
            ("c", "1.0"): [release_file(H2, "2026-10-01T00:00:00Z")],
        }
    )
    try:
        assert run_check(tmp_path, index, {"a": "1.0"}, {"a": [H1]}) == 1
        assert "yanked" in capsys.readouterr().out
        assert (
            run_check(tmp_path, index, {"c": "1.0"}, {"c": [H3]}) == 1
        )  # a hash the index does not list for this release
        assert "is not a file of this release" in capsys.readouterr().out
        assert (
            run_check(tmp_path, index, {"gone": "9.9"}, {"gone": [H1]}) == 2
        )  # the index has no such release: cannot check
    finally:
        index.close()


def test_the_age_check_fails_when_the_lock_and_its_input_disagree_or_a_pin_has_no_hash(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    index = FakeIndex({("a", "1.0"): [release_file(H1, "2026-10-01T00:00:00Z")]})
    try:
        lock, declared = lock_files(tmp_path, {"a": "1.0"}, {"a": [H1]})
        declared.write_text("a==1.1\n")
        check = load("check-dep-age")
        argv = ["check-dep-age.py", "2026-10-29", "--index", index.url, "--lock", str(lock), "--input", str(declared)]
        assert check.main(argv) == 1 and "disagree" in capsys.readouterr().err
        declared.write_text("a==1.0\n")
        lock.write_text("a==1.0\n")
        assert check.main(argv) == 1 and "no hash pinned" in capsys.readouterr().out
        assert check.main(["check-dep-age.py", "not-a-date", "--index", index.url]) == 2
        assert check.main(["check-dep-age.py", "--index", "ftp://example.org"]) == 2
    finally:
        index.close()


def test_the_age_check_reports_an_unreachable_index_as_an_error_not_a_pass(tmp_path: Path) -> None:
    lock, declared = lock_files(tmp_path, {"a": "1.0"}, {"a": [H1]})
    check = load("check-dep-age")
    assert (
        check.main(
            [
                "check-dep-age.py",
                "2026-10-29",
                "--index",
                "http://127.0.0.1:9",
                "--lock",
                str(lock),
                "--input",
                str(declared),
            ]
        )
        == 2
    )
