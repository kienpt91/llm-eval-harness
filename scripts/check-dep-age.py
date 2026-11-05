#!/usr/bin/env python3
"""Dependency age policy: every file pinned in requirements-dev.txt must be at least seven days old on PyPI.

    python scripts/check-dep-age.py [CUTOFF_ISO_DATE]

Without an argument the cutoff is seven days before now. For each `name==version` and each `--hash=sha256:...` it
asks the PyPI JSON API for the release files, finds the file with that hash and compares its upload time with the
cutoff. Exit status: 0 all old enough, 1 a file is too new or unknown to PyPI, 2 usage or network error.
Standard library only; needs the network (it is the one script that does).
"""
from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

PIN = re.compile(r"^(?P<name>[A-Za-z0-9._-]+)==(?P<version>[A-Za-z0-9.!+_-]+)\s*\\?$")
HASH = re.compile(r"--hash=sha256:(?P<digest>[0-9a-f]{64})")


def parse(text: str) -> dict[tuple[str, str], list[str]]:
    pins: dict[tuple[str, str], list[str]] = {}
    current: list[str] | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        pin = PIN.match(stripped)
        if pin is not None:
            current = pins.setdefault((pin["name"], pin["version"]), [])
            continue
        digest = HASH.search(stripped)
        if digest is not None and current is not None:
            current.append(digest["digest"])
            continue
        raise ValueError(f"cannot read line: {stripped!r}")
    return pins


def main(argv: list[str]) -> int:
    try:
        cutoff = datetime.fromisoformat(argv[1]) if len(argv) > 1 else datetime.now(timezone.utc) - timedelta(days=7)
    except ValueError:
        print("invalid cutoff date", file=sys.stderr)
        return 2
    if cutoff.tzinfo is None:
        cutoff = cutoff.replace(tzinfo=timezone.utc)
    pins = parse(Path(__file__).resolve().parent.parent.joinpath("requirements-dev.txt").read_text())
    if not pins:
        print("requirements-dev.txt pins nothing", file=sys.stderr)
        return 2
    too_new: list[str] = []
    files_checked = 0
    for (name, version), digests in sorted(pins.items()):
        if not digests:
            too_new.append(f"{name}=={version}: no hash pinned")
            continue
        try:
            with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout=30) as response:  # noqa: S310
                release = json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            print(f"{name}=={version}: PyPI lookup failed: {error}", file=sys.stderr)
            return 2
        by_hash = {f["digests"]["sha256"]: f for f in release["urls"]}
        for digest in digests:
            found = by_hash.get(digest)
            if found is None:
                too_new.append(f"{name}=={version}: hash {digest[:12]} is not a file of this release on PyPI")
                continue
            files_checked += 1
            uploaded = datetime.fromisoformat(found["upload_time_iso_8601"].replace("Z", "+00:00"))
            if uploaded > cutoff:
                too_new.append(f"{found['filename']}: uploaded {uploaded.isoformat()}")
    print(f"Checked {files_checked} files of {len(pins)} pinned packages against cutoff {cutoff.isoformat()}")
    if too_new:
        print("Not acceptable:")
        for line in too_new:
            print(f"  {line}")
        return 1
    print("No file is newer than the cutoff.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
