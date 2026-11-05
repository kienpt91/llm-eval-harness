"""Files inside the project and nowhere else.

Reads and writes go through ``ProjectFilesystem``. A path that resolves outside the project root (``..``, an
absolute path, or a symbolic link that leaves it) is refused. Writes never follow a symbolic link, are atomic
(temporary file in the same directory, then rename) and refuse to replace an existing file unless told to.
"""

from __future__ import annotations

import contextlib
import os
import stat
import tempfile
from pathlib import Path

from llm_eval_harness.domain.errors import UsageError


class ProjectFilesystem:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve(strict=True)

    # -- resolution ---------------------------------------------------------------------------------------

    def _candidate(self, path: str) -> Path:
        if not path or "\x00" in path:
            raise UsageError("empty or invalid path")
        raw = Path(path)
        return raw if raw.is_absolute() else self.root / raw

    def _inside(self, resolved: Path, shown: str) -> None:
        if resolved != self.root and not resolved.is_relative_to(self.root):
            raise UsageError(f"{shown}: outside the project directory (symbolic links that leave it are refused too)")

    def resolve_existing(self, path: str) -> Path:
        candidate = self._candidate(path)
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError:
            raise UsageError(f"{path}: no such file") from None
        except (OSError, RuntimeError) as error:
            raise UsageError(f"{path}: cannot be resolved ({type(error).__name__})") from None
        self._inside(resolved, path)
        return resolved

    def resolve_directory(self, path: str) -> Path:
        resolved = self.resolve_existing(path)
        if not resolved.is_dir():
            raise UsageError(f"{path}: not a directory")
        return resolved

    # -- port ---------------------------------------------------------------------------------------------

    def read_text(self, path: str, max_bytes: int) -> str:
        resolved = self.resolve_existing(path)
        try:
            info = resolved.stat()
        except OSError as error:
            raise UsageError(f"{path}: cannot be read ({type(error).__name__})") from None
        if not stat.S_ISREG(info.st_mode):
            raise UsageError(f"{path}: not a regular file")
        if info.st_size > max_bytes:
            raise UsageError(f"{path}: {info.st_size} bytes, the limit is {max_bytes}")
        try:
            with resolved.open("rb") as handle:
                data = handle.read(max_bytes + 1)
        except OSError as error:
            raise UsageError(f"{path}: cannot be read ({type(error).__name__})") from None
        if len(data) > max_bytes:
            raise UsageError(f"{path}: larger than the limit of {max_bytes} bytes")
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            raise UsageError(f"{path}: not valid UTF-8 text") from None

    def write_text(self, path: str, text: str, *, overwrite: bool) -> str:
        target = self._candidate(path)
        target = Path(os.path.normpath(target))  # collapses '..' lexically, then the real location is checked below
        self._inside(target, path)
        # every existing ancestor must really be inside the project
        ancestor = target.parent
        while not ancestor.exists() and ancestor != self.root:
            ancestor = ancestor.parent
        self._inside(ancestor.resolve(strict=True), path)
        if target.is_symlink():
            raise UsageError(f"{path}: is a symbolic link; refusing to write through it")
        if target.exists() and not overwrite:
            raise UsageError(f"{path}: already exists (use --force to replace it)")
        if target.is_dir():
            raise UsageError(f"{path}: is a directory")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            self._inside(target.parent.resolve(strict=True), path)
            handle, temporary = tempfile.mkstemp(dir=target.parent, prefix=".llm-eval-", suffix=".tmp")
            try:
                with os.fdopen(handle, "wb") as out:
                    out.write(text.encode("utf-8"))
                os.replace(temporary, target)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(temporary)
                raise
        except OSError as error:
            raise UsageError(f"{path}: cannot be written ({error.strerror or type(error).__name__})") from None
        return str(target.relative_to(self.root))
