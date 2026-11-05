# 1. Layers and the dependency rule

Status: accepted

## Context

The tool is mostly decisions (is this answer good, did this run regress, what exit code) wrapped around a little
input and output (processes, sockets, files). The decisions must be testable without any of that, and the
input and output must be unable to leak into the decisions.

## Decision

Four layers inside `src/llm_eval_harness/`, dependencies pointing inward only:

- `domain`: the model, the graders, dataset validation and hashing, statistics, comparison, the gate. **Pure**:
  it imports only standard modules that compute (`re`, `json`, `hashlib`, `dataclasses`, `decimal`,
  `statistics`-style math, `ipaddress`, ...). No files, network, processes, clock, randomness, threads or
  environment. Seeded sampling therefore uses its own small PRNG (SplitMix64) instead of `random`.
- `application`: use cases and the runner over **ports** (`Provider`, `JudgeRunner`, `Clock`, `ProjectFiles`,
  protocols in `ports.py`). It may use thread primitives for the concurrency-limited runner and nothing that
  touches the world.
- `infrastructure`: the adapters: process runner, file system guard, command/http/fake providers, argument
  parsing. Process creation lives in one module (`process.py`) and raw sockets in one (`providers/http.py`).
- `composition`: wiring and the command line. The only layer that reads the environment.

The rule is **enforced mechanically**: `tests/architecture/test_layers.py` parses every source file with `ast` and
fails on a forbidden import (by layer, and by module allow-list for the pure layers), on `eval`/`exec`/
`compile`/`__import__`, on `shell=` other than the literal `False`, on `os.system`/`os.popen`, on wildcard
imports, on environment access outside composition, on third-party imports, and on `subprocess` or network modules
outside their one module. The checker itself is tested against a snippet for each rule.

## Consequences

- The domain and application layers are tested with plain objects and in-memory fakes (domain coverage over 95%,
  application over 90%, see the README).
- Adding a provider means a new adapter and a line in `composition/compose.py`, nothing else.
- A convenient `import requests` or `import os` in a pure layer is a failing test, not a review comment.
