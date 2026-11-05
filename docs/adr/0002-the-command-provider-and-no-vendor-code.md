# 2. The command provider, and no vendor code in the core

Status: accepted

## Context

Teams evaluate pipelines written in different languages on different model services. A harness that bundles
clients for them ties itself to their release cycles, needs their credentials, adds dependencies, and makes the
tests depend on the outside world.

## Decision

The core calls no hosted model and contains no vendor client, name or price. A pipeline is reached through:

- the **`command` provider**: your program, fixed argv, one JSON request on stdin, one JSON answer on stdout, with a
  deadline, size caps and an allow-listed environment. Any language, any model client, no SDK in the harness;
- the **`http` provider** for services that already run, restricted to loopback or explicitly listed hosts;
- the **`fake` provider**, deterministic, for tests and the demo.

LLM-as-judge is the same idea: a `judge` grader runs *your* program and records its verdict and confidence,
marked non-deterministic. Prices come from a table the user supplies; there are no built-in prices.

## Consequences

- Zero runtime dependencies; the test suite needs no network except to loopback.
- The protocol is the contract; wiring is a dozen lines in any language ([wiring.md](../wiring.md)).
- The expectations are never sent to the pipeline, so a pipeline cannot read the answer key.
- Costs and tokens are only as good as the `usage` the pipeline reports.
