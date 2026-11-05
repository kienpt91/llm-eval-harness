#!/bin/sh
# Build the sdist and the wheel in the pinned container, reproducibly, into out/dist.
# usage: scripts/build-dist.sh
#
# The sources are copied to a scratch directory inside the container (the project is mounted read-only) and built
# with `python -m build --no-isolation`: the build backend is the exact, hash-checked setuptools of requirements-dev.txt,
# nothing is downloaded at build time. SOURCE_DATE_EPOCH is the time of the last commit, so the same commit gives
# the same files.
set -eu
here=$(cd "$(dirname "$0")/.." && pwd)
epoch=$(git -C "$here" log -1 --format=%ct 2> /dev/null || echo 0)
rm -rf "$here/out/dist"
sh "$here/scripts/py.sh" "set -e
mkdir /tmp/build && cp -r pyproject.toml README.md LICENSE src /tmp/build/ && cd /tmp/build
SOURCE_DATE_EPOCH=$epoch python -m build --no-isolation --outdir /work/out/dist .
cd /work/out/dist && ls -l"
