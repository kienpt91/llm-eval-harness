# Running, comparing and gating

```
llm-eval run      run the dataset through the provider, grade, write out/report.json and out/report.md
llm-eval baseline turn a report into baseline.json (commit it)
llm-eval compare  current report against the baseline (informational)
llm-eval gate     run (or read a report), compare, decide: the exit code is for CI
```

## Exit codes

| code | meaning | when |
| --- | --- | --- |
| 0 | pass | nothing wrong |
| 1 | quality regression | a case regressed against the baseline, a new or changed quality case fails, a cost or latency limit was exceeded (or cost is unknown while a cost limit is set), the pass rate is below the minimum, or the comparison is inconclusive and `--on-inconclusive fail` (the default) |
| 2 | usage error | bad command line, configuration or dataset; a path outside the project; no provider or dataset configured. Nothing was run. |
| 3 | provider or infrastructure error | at least one run could not be graded after its retries (timeout, crash, bad protocol, unreachable endpoint, judge unavailable), or the harness hit an internal error |
| 4 | safety failure | any run of any `safety` case failed, **or a safety case of the dataset or the baseline was not run** (filtered, sampled out, removed, recategorised) |

When several apply, the strongest wins: **4, then 3, then 1, then 0**.

- A failing safety case is **never averaged away**: one failed run among a thousand quality passes is exit 4. A
  baseline cannot excuse it (a safety case that already failed in the baseline fails again), a quality
  improvement cannot buy it back, and no threshold or pass-rate option relaxes it. With `--repeat`, one failing
  repeat of a safety case is a failure. Safety cases cannot lower their `threshold` and cannot carry a grader that
  could never fail them (weight 0 without `must`): the dataset is refused.
- A safety case that **was not run** is not a passing one. `--category quality`, `--tag`, `--case` and `--sample`
  are fine for `run`, but a `gate` whose selection leaves out any safety case of the dataset exits 4 and names the
  ids; so does a baseline safety case that is no longer in the dataset or is no longer a safety case (deleting or
  recategorising a safety test is a visible failure, not a quiet pass). `--allow-partial-safety` (or
  `"allow_partial_safety": true` under `gate`) accepts that deliberately, for development; it never accepts a safety
  case that ran and failed.
- Exit 3 outranks 1 because a run that could not be graded says nothing about quality: the honest answer is "I
  could not tell", never "pass". A **safety case that errored** is exit 3, not a pass.
- `run` uses the same ladder without a baseline or a pass-rate minimum: quality failures alone do not fail `run`
  (it measures), safety failures (4) and ungradable runs (3) do. `compare` always exits 0. `validate`,
  `baseline` and `init-safety` exit 0 or 2.

## What `gate` does

1. Without `--report` it runs the dataset first (same options as `run`). With `--report FILE` it gates that report
   and refuses run options (`--repeat`, `--dataset`, ...) instead of ignoring them. A report is validated and
   cross-checked (see [report-format.md](report-format.md#validation)) and refused as **stale** (exit 2) when the
   configured dataset no longer has the hash the report was made with; `compare` and `baseline` do the same. Reports
   are not signed: in CI, let `gate` run the pipeline itself.
2. With a baseline (`--baseline FILE`, or `"baseline"` in the configuration) it compares the run with it. Without
   one, every quality case must pass: the minimum pass rate defaults to 1 (change it with `--min-pass-rate`).
3. It prints the summary, the comparison and the decision with its reasons, and writes `out/gate.json` (machine
   readable, schema version 1) and, with `--markdown FILE`, the pull request text.

## Comparing without flagging noise

Model behaviour varies, so one case flipping from pass to fail is not always a regression. For each case the
harness has a pass rate over *n* graded runs on each side (`--repeat N`, `n` ≥ 1). Errored runs are not graded
runs; they are counted separately.

| verdict | meaning |
| --- | --- |
| `unchanged` | the pass-rate difference is within `--tolerance` (default 0) |
| `regressed` / `improved` | the difference exceeds the tolerance **and** the data supports it |
| `inconclusive` | the difference exceeds the tolerance but the data cannot support a verdict |
| `changed` | the case itself changed since the baseline (its hash differs): not comparable with its old self, so it is judged on its own: **a new or changed quality case that fails is exit 1** (the baseline cannot excuse a result it never saw) |
| `new` / `removed` | in only one of the two. `removed` fails the gate only with `--fail-on-removed` (a removed *safety* case always fails it, see above) |

*The data supports a verdict* when Fisher's exact test (two-sided) rejects "the two pass rates are the same" at
`--alpha` (default 0.05), **or** when you declare the provider deterministic (`--deterministic`, or
`"deterministic": true` under `gate` in the configuration): then one run per case is a fact, and any difference
beyond the tolerance counts. Use that for a scripted pipeline or a model at temperature 0 with a fixed seed;
otherwise repeat.

How much data is enough? For a case that went from always passing to always failing:

| runs per side | p-value | verdict at alpha 0.05 |
| --- | --- | --- |
| 1 | 1.000 | inconclusive |
| 3 | 0.100 | inconclusive |
| 4 | 0.029 | regressed |
| 5 | 0.008 | regressed |

Partial changes need more: 8 of 10 against 10 of 10 is p = 0.474, inconclusive. The report says so in words
("cannot be told from chance (p=0.474 > alpha 0.05); run with more --repeat").

The overall comparison is `regression` when more cases regressed than `--max-regressed-cases` allows (default 0),
or `--max-cost-increase-pct` / `--max-p95-latency-increase-pct` is exceeded; else `inconclusive` if any case is;
else `ok`. `--on-inconclusive pass` lets an inconclusive comparison pass (it is still printed as a note); the
default `fail` turns it into exit 1, meaning "I cannot tell: add `--repeat`".

Cost and latency are compared as deltas of the run's mean cost per run and p50/p95 latency. Latency depends on
the machine, so its limit is off by default. The cost delta is only computed when every run on both sides has a
known cost. When `--max-cost-increase-pct` is set and cost is unknown on either side (a run without usage, or a model
that is not in the price table), the comparison is a **regression**: a limit that cannot be checked is not met, the same
way a `cost_budget` fails on unknown cost. Token counts and model names are reported by the pipeline, so a pipeline can
still misreport them: see [SECURITY.md](../SECURITY.md#not-protected-read-this-before-relying-on-it).

## Repeats and flakiness

`--repeat N` runs every case N times (independent provider calls; the request carries `repeat` and `attempt`).
The report gives each case a pass rate and a score variance; a case that passed some runs and failed others is
**flaky** and is listed in the terminal summary and the Markdown. Retries (`--retries`, default 1) are only for
infrastructure: a call that failed with a timeout, a crash, a non-zero exit or a network error is repeated; a
call that produced an answer is graded once and that grade stands. A failed grade is never retried, so retries
cannot hide a regression, and the report shows how many attempts a run needed.

## Sampling and filters

`--sample K --seed S` runs a seeded sample of K cases (after the filters), the same K for the same seed on every
platform; `--tag`, `--category` and `--case` filter. The report records the selection, and cases that were not
run show up as `removed` against a full baseline.

## The baseline lifecycle

1. `llm-eval run --write-baseline baseline.json` (or `llm-eval baseline`) on the main branch; commit the file.
2. Every pull request runs `llm-eval gate`.
3. When a change is *meant* to move quality, cost or latency, regenerate the baseline in the same pull request, so
   the new numbers are reviewed like code. A baseline never silences a safety failure.

The baseline holds per-case counts and the aggregates, no inputs, no outputs and no timestamps; keys are sorted,
so the same run always produces the same bytes and a diff shows exactly what moved. See
[report-format.md](report-format.md).

## Ranges

Flags are held to the ranges the configuration file enforces: `--repeat` 1 to 1000, `--concurrency` 1 to 64,
`--retries` 0 to 10, `--timeout` 0.1 to 3600 seconds, `--sample` 1 to 10000, `--seed` 0 or more, `--tolerance` 0 to 1,
`--alpha` 1e-6 to 0.5, `--min-pass-rate` 0 to 1, `--max-regressed-cases` 0 to 10000, the percentage limits 0 to 1e6. A
value outside is a usage error (exit 2) naming the range.
