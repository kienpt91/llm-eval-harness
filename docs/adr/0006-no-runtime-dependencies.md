# 6. No runtime dependencies; pinned, hash-checked development tools

Status: accepted

## Context

The tool runs user programs and sits in CI with secrets nearby. Every dependency is code that runs there.

## Decision

- The package depends on nothing but the Python 3.12+ standard library (`dependencies = []`; an AST test fails on any
  third-party import).
- Development tools (`pytest`, `pytest-cov`, `coverage`, `mypy`, `ruff`, `build`, `setuptools` and their
  dependencies) are pinned to exact versions in `requirements-dev.in` and installed from `requirements-dev.txt` with
  `pip install --require-hashes`. The hashes are those of the wheels for CPython 3.12 and 3.13 on Linux x86-64 and
  aarch64, written by `scripts/lock-dev.py` from the PyPI JSON API. It refuses any file uploaded to PyPI less than
  seven days before the cutoff, and `scripts/check-dep-age.py` repeats the check in CI.
- Everything runs in a container pinned by **digest** (`scripts/py.sh`): non-root, all capabilities dropped,
  no network except to install, 4 CPUs, 6 GiB, source mounted read-only.
- GitHub Actions are pinned by commit SHA and workflows default to read-only permissions; the release workflow
  builds in the same pinned image, attests provenance, and never publishes to a package registry.

## Consequences

- `pip-tools` is not used: generating the hashes with a stdlib script keeps the toolchain smaller than the thing it
  would lock.
- Updating a tool is deliberate: edit `requirements-dev.in`, regenerate, review.
