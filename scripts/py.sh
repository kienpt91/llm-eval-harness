#!/bin/sh
# Run a command inside a throwaway Python container with the project mounted READ-ONLY.
# The only writable places are /work/out (a bind of ./out: coverage data, caches, dist/), /tmp (tmpfs) and the
# venv volume. The container has no access to $HOME or host env, drops all capabilities, runs as a non-root
# user, has no network except for `install`, and is limited to 4 CPUs and 6 GiB.
#
# usage: scripts/py.sh install            create the venv and install requirements-dev.txt (--require-hashes)
#        scripts/py.sh <shell command>    run a command with the venv on PATH, no network
#   PY_VERSION=3.13 (default) or 3.12
#   PY_NET=bridge    allow the network for one command (only scripts/lock-dev.py and scripts/check-dep-age.py need it)
#
# Images are the official `python` slim-trixie images, pinned by the digest of the multi-arch index.
#   3.13.17-slim-trixie  index pushed 2026-10-24
#   3.12.15-slim-trixie  index pushed 2026-10-22
# Both are older than the seven day dependency age policy at the time of pinning (2026-11-05).
set -eu
here=$(cd "$(dirname "$0")/.." && pwd)
ver=${PY_VERSION:-3.13}
case "$ver" in
  3.13) default_image=python@sha256:d98cc29d3266e535dd9da1099688d9cb3ca4753813571e071763d17725460962 ;; # 3.13.17-slim-trixie, 2026-10-24
  3.12) default_image=python@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704 ;; # 3.12.15-slim-trixie, 2026-10-22
  *) echo "unsupported PY_VERSION (3.12 or 3.13)" >&2; exit 2 ;;
esac
image=${PY_IMAGE:-$default_image}
[ $# -ge 1 ] || { echo "usage: py.sh install | py.sh <shell command>" >&2; exit 2; }
name=$(basename "$here")
vol="py-venv-$name-$ver"
mkdir -p "$here/out"
docker volume create "$vol" > /dev/null
# the volume must be owned by the unprivileged user
docker run --rm -u 0 -v "$vol:/venv" "$image" chown "$(id -u):$(id -g)" /venv > /dev/null

if [ "$1" = install ]; then
  net=bridge
  cmd='[ -x /venv/bin/python ] || python -m venv /venv; /venv/bin/python -m pip install --disable-pip-version-check --no-deps --require-hashes -r requirements-dev.txt'
else
  net=${PY_NET:-none}
  cmd="$*"
fi
exec docker run --rm \
  --user "$(id -u):$(id -g)" \
  --cap-drop ALL --security-opt no-new-privileges \
  --cpus 4 --memory 6g --pids-limit 1024 \
  --network "$net" \
  --read-only --tmpfs /tmp:rw,exec,size=1g \
  -e HOME=/tmp/home -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 \
  -e PIP_CACHE_DIR=/venv/.pip-cache -e MYPY_CACHE_DIR=/work/out/.mypy_cache -e RUFF_CACHE_DIR=/work/out/.ruff_cache \
  -e COVERAGE_FILE=/work/out/.coverage -e PATH=/venv/bin:/usr/local/bin:/usr/bin:/bin \
  -v "$here:/work:ro" -v "$here/out:/work/out" -v "$vol:/venv" -w /work \
  "$image" sh -c "$cmd"
