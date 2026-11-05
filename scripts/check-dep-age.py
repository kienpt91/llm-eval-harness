#!/usr/bin/env python3
"""Dependency age policy: every file pinned in requirements-dev.txt must be at least seven days old on PyPI.

    python scripts/check-dep-age.py [CUTOFF_ISO_DATE] [--index URL] [--lock FILE] [--input FILE]

Without a cutoff it is seven days before now. For each ``name==version`` of the lock and each
``--hash=sha256:...`` it asks the index's JSON API for the files of that exact release, finds the file with that
hash and compares its upload time with the cutoff. It also fails when

* the lock and its input (``requirements-dev.in``) do not pin exactly the same packages and versions,
* a pin has no hash,
* a hash is not a file of that release, or the file was yanked.

Exit status: 0 all old enough, 1 a file is too new, unknown or yanked (or lock and input disagree), 2 usage or network
error (also: the index has no such release, so nothing could be checked). Standard library only; needs the network
(it is the one script that does). ``--index`` exists for the tests: it must be https, or http on loopback.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INDEX = "https://pypi.org"
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


def normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def index_is_acceptable(index: str) -> bool:
    parts = urlsplit(index)
    host = parts.hostname or ""
    return parts.scheme == "https" or (parts.scheme == "http" and host in ("127.0.0.1", "localhost", "::1"))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="check-dep-age.py", add_help=True)
    parser.add_argument("cutoff", nargs="?")
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--lock", default=str(ROOT / "requirements-dev.txt"))
    parser.add_argument("--input", default=str(ROOT / "requirements-dev.in"))
    try:
        args = parser.parse_args(argv[1:])
    except SystemExit as error:
        return 2 if error.code else 0
    try:
        cutoff = datetime.fromisoformat(args.cutoff) if args.cutoff else datetime.now(UTC) - timedelta(days=7)
    except ValueError:
        print("invalid cutoff date", file=sys.stderr)
        return 2
    if cutoff.tzinfo is None:
        cutoff = cutoff.replace(tzinfo=UTC)
    if not index_is_acceptable(args.index):
        print("the index must be an https URL (or http on loopback)", file=sys.stderr)
        return 2
    pins = parse(Path(args.lock).read_text())
    if not pins:
        print("the lock pins nothing", file=sys.stderr)
        return 2
    declared = {
        (normalise(name), version)
        for name, version in (
            line.strip().split("==", 1)
            for line in Path(args.input).read_text().splitlines()
            if line.strip() and not line.startswith("#")
        )
    }
    if declared != {(normalise(name), version) for name, version in pins}:
        print("the input and the lock disagree: run scripts/lock-dev.py", file=sys.stderr)
        return 1
    too_new: list[str] = []
    files_checked = 0
    for (name, version), digests in sorted(pins.items()):
        if not digests:
            too_new.append(f"{name}=={version}: no hash pinned")
            continue
        try:
            with urllib.request.urlopen(f"{args.index.rstrip('/')}/pypi/{name}/{version}/json", timeout=30) as response:
                release = json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as error:
            print(f"{name}=={version}: lookup failed: {error}", file=sys.stderr)
            return 2
        by_hash = {f["digests"]["sha256"]: f for f in release["urls"]}
        for digest in digests:
            found = by_hash.get(digest)
            if found is None:
                too_new.append(f"{name}=={version}: hash {digest[:12]} is not a file of this release on the index")
                continue
            files_checked += 1
            if found.get("yanked"):
                too_new.append(f"{found['filename']}: yanked")
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
