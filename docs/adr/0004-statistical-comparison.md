# 4. Statistical comparison and "inconclusive"

Status: accepted

## Context

Sampled models do not give the same answer twice. A gate that fails on every pass-to-fail flip trains people to
rerun until green; a gate that averages hides real regressions. A single run per case cannot tell either apart.

## Decision

Per case, compare pass rates over the graded runs on each side:

1. a difference within `--tolerance` is `unchanged`;
2. beyond it, Fisher's exact test (two-sided, exact, standard library only) decides: `p <= alpha` is
   `regressed`/`improved`, otherwise **`inconclusive`** with the evidence in words and a hint to add `--repeat`;
3. a provider declared **deterministic** (`--deterministic`) skips the test: one run is a fact.

The gate turns an inconclusive comparison into exit 1 by default (`--on-inconclusive fail`), because "I cannot tell"
should not look like "pass"; `--on-inconclusive pass` records it as a note instead. Cases whose hash changed are
`changed`, not compared. Cost and latency are compared as deltas with optional limits; latency is off by default
because it depends on the machine; cost is only compared when it is known on both sides.

## Consequences

- Meaningful verdicts need about 4 repeats per side for an always-pass to always-fail change, more for smaller
  shifts ([gate.md](../gate.md#comparing-without-flagging-noise)).
- The test treats runs as independent, which repeated calls of one case are not quite; the alpha is a guard
  against noise, not a guarantee.
