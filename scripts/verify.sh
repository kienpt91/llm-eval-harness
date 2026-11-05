#!/bin/sh
# The whole local gate in the pinned container: format check, lint, strict type check, tests with coverage thresholds.
# usage: scripts/verify.sh [fast]   (fast skips coverage)
set -eu
here=$(dirname "$0")
if [ "${1:-}" = fast ]; then
  exec sh "$here/py.sh" 'ruff format --check . && ruff check . && python -m mypy && python -m pytest -q'
fi
exec sh "$here/py.sh" 'ruff format --check . && ruff check . && python -m mypy && python scripts/coverage_check.py'
