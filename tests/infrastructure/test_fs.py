from __future__ import annotations

import os
from pathlib import Path

import pytest

from llm_eval_harness.domain.errors import UsageError
from llm_eval_harness.infrastructure.fs import ProjectFilesystem


@pytest.fixture
def project(tmp_path: Path) -> tuple[ProjectFilesystem, Path, Path]:
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("outside data")
    return ProjectFilesystem(root), root, outside


def test_reads_inside_the_project(project: tuple[ProjectFilesystem, Path, Path]) -> None:
    fs, root, _ = project
    (root / "d").mkdir()
    (root / "d" / "a.txt").write_text("héllo")
    assert fs.read_text("d/a.txt", 100) == "héllo"
    assert fs.read_text(str(root / "d" / "a.txt"), 100) == "héllo"
    assert fs.read_text("d/../d/a.txt", 100) == "héllo"


@pytest.mark.parametrize("path", ["../outside/secret.txt", "d/../../outside/secret.txt"])
def test_dotdot_escapes_are_refused(project: tuple[ProjectFilesystem, Path, Path], path: str) -> None:
    fs, root, _ = project
    (root / "d").mkdir()
    with pytest.raises(UsageError, match="outside the project"):
        fs.read_text(path, 100)


def test_absolute_paths_outside_are_refused(project: tuple[ProjectFilesystem, Path, Path]) -> None:
    fs, _, outside = project
    with pytest.raises(UsageError, match="outside the project"):
        fs.read_text(str(outside / "secret.txt"), 100)
    with pytest.raises(UsageError, match="outside the project"):
        fs.read_text("/etc/passwd", 100)


def test_symlinks_that_leave_the_project_are_refused_but_inside_ones_work(
    project: tuple[ProjectFilesystem, Path, Path],
) -> None:
    fs, root, outside = project
    (root / "real.txt").write_text("inside")
    os.symlink(outside / "secret.txt", root / "leak.txt")
    os.symlink(outside, root / "leakdir")
    os.symlink(root / "real.txt", root / "ok.txt")
    with pytest.raises(UsageError, match="outside the project"):
        fs.read_text("leak.txt", 100)
    with pytest.raises(UsageError, match="outside the project"):
        fs.read_text("leakdir/secret.txt", 100)
    assert fs.read_text("ok.txt", 100) == "inside"


def test_unreadable_things_are_refused_with_a_reason(project: tuple[ProjectFilesystem, Path, Path]) -> None:
    fs, root, _ = project
    (root / "big.txt").write_text("x" * 50)
    (root / "bin.dat").write_bytes(b"\xff\xfe\x00")
    os.mkfifo(root / "pipe")
    (root / "dir").mkdir()
    for path, fragment in [
        ("missing.txt", "no such file"),
        ("big.txt", "the limit is 10"),
        ("bin.dat", "not valid UTF-8"),
        ("pipe", "not a regular file"),
        ("dir", "not a regular file"),
        ("", "empty or invalid path"),
        ("a\x00b", "empty or invalid path"),
    ]:
        with pytest.raises(UsageError, match=fragment):
            fs.read_text(path, 10)


def test_symlink_loops_do_not_hang(project: tuple[ProjectFilesystem, Path, Path]) -> None:
    fs, root, _ = project
    os.symlink(root / "b", root / "a")
    os.symlink(root / "a", root / "b")
    with pytest.raises(UsageError):
        fs.read_text("a", 100)


def test_writes_create_directories_and_return_the_relative_path(project: tuple[ProjectFilesystem, Path, Path]) -> None:
    fs, root, _ = project
    assert fs.write_text("out/deep/report.json", "data", overwrite=False) == "out/deep/report.json"
    assert (root / "out" / "deep" / "report.json").read_text() == "data"
    assert oct((root / "out" / "deep" / "report.json").stat().st_mode & 0o777) == "0o600"
    assert [p.name for p in (root / "out" / "deep").iterdir()] == ["report.json"]  # no temporary file left behind


def test_overwrite_is_explicit(project: tuple[ProjectFilesystem, Path, Path]) -> None:
    fs, root, _ = project
    fs.write_text("a.txt", "one", overwrite=False)
    with pytest.raises(UsageError, match="already exists"):
        fs.write_text("a.txt", "two", overwrite=False)
    assert (root / "a.txt").read_text() == "one"
    fs.write_text("a.txt", "two", overwrite=True)
    assert (root / "a.txt").read_text() == "two"


@pytest.mark.parametrize("path", ["../escape.txt", "d/../../escape.txt", "/tmp/llm-eval-escape.txt"])
def test_writes_outside_the_project_are_refused(project: tuple[ProjectFilesystem, Path, Path], path: str) -> None:
    fs, root, _ = project
    (root / "d").mkdir()
    with pytest.raises(UsageError, match="outside the project"):
        fs.write_text(path, "x", overwrite=True)
    assert not (root.parent / "escape.txt").exists() and not Path("/tmp/llm-eval-escape.txt").exists()


def test_writes_never_follow_symlinks(project: tuple[ProjectFilesystem, Path, Path]) -> None:
    fs, root, outside = project
    os.symlink(outside / "secret.txt", root / "link.txt")
    os.symlink(root / "real.txt", root / "inside-link.txt")
    os.symlink(outside, root / "linkdir")
    with pytest.raises(UsageError, match="symbolic link"):
        fs.write_text("link.txt", "pwned", overwrite=True)
    with pytest.raises(UsageError, match="symbolic link"):
        fs.write_text("inside-link.txt", "x", overwrite=True)
    with pytest.raises(UsageError, match="outside the project"):
        fs.write_text("linkdir/new.txt", "pwned", overwrite=True)
    assert (outside / "secret.txt").read_text() == "outside data" and not (outside / "new.txt").exists()


def test_writing_to_a_directory_or_unwritable_place_is_a_usage_error(
    project: tuple[ProjectFilesystem, Path, Path],
) -> None:
    fs, root, _ = project
    (root / "d").mkdir()
    with pytest.raises(UsageError, match="is a directory"):
        fs.write_text("d", "x", overwrite=True)
    (root / "f").write_text("a file")
    with pytest.raises(UsageError, match="cannot be written"):
        fs.write_text("f/child.txt", "x", overwrite=True)


def test_resolve_directory(project: tuple[ProjectFilesystem, Path, Path]) -> None:
    fs, root, _ = project
    (root / "d").mkdir()
    (root / "f").write_text("x")
    assert fs.resolve_directory("d") == (root / "d").resolve()
    with pytest.raises(UsageError, match="not a directory"):
        fs.resolve_directory("f")
    with pytest.raises(UsageError, match="outside"):
        fs.resolve_directory("..")
