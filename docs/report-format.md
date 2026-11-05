# Report, baseline and gate file formats

All three are JSON with `schema_version` (currently **1**) and a `kind`. A reader refuses a file with another
kind or version with a message naming the field; a future change that is not backward compatible bumps the
version. Files are ASCII-only JSON (non-ASCII text is `\uXXXX`-escaped), so raw control characters and
bidirectional marks from model output can never reach a file.

## `out/report.json` (`"kind": "llm-eval-report"`)

Everything about one run. Cases are sorted by id and runs by repeat, so the order never depends on timing.

```
schema_version  1
kind            "llm-eval-report"
tool            {name, version}
started_at, finished_at   UTC ISO 8601
dataset         {source, hash, total_cases, selected_cases}
run             {repeat, seed, sample, tags, categories, concurrency, retries, redacted}
provider        a short, secret-free description: {type, program, arguments} | {type, host, scheme, headers} | {type, model}
summary         {cases, runs, graded_runs, errored_runs,
                 quality {cases, mean_pass_rate, fully_passing, mean_score},
                 safety  {cases, failing_cases, failing_runs},
                 flaky_cases, tokens {input, output},
                 cost {currency, total, mean_per_run, unpriced_runs},
                 latency_ms {p50, p95, max, mean}}
cases[]         {id, category, tags, case_hash, input, status, graded_runs, passes, errors,
                 pass_rate, mean_score, score_variance, flaky,
                 runs[] {repeat, attempts, status, score, failed_musts,
                         grades[] {name, type, passed, score, weight, must, deterministic, confidence, verdict, detail},
                         output {text, truncated, tool_calls, citations, retrieved, usage, latency_ms, cost} | null,
                         error {kind, message} | null}}
```

- `status` of a run is `passed`, `failed` (a grade) or `error` (the run could not be graded: no output, no grades).
  A case is `failed` if any graded run failed, else `error` if any run errored, else `passed`.
- `pass_rate` is passes over *graded* runs, `null` when there are none. `flaky` means some graded runs passed and
  some failed.
- Latency percentiles are nearest-rank over runs that produced an output. `cost.total` is `null` (never a partial
  sum) when any run's cost is unknown; `unpriced_runs` counts them.
- `output.text` is stored redacted (unless redaction is off) and cut at 20,000 characters (`truncated`); grading
  always saw the whole answer. Tool arguments are redacted and bounded. `input` is the case input as compact JSON,
  redacted, cut at 2,000 characters.
- Judge grades have `deterministic: false` plus the judge's `confidence` and `verdict`.

## `baseline.json` (`"kind": "llm-eval-baseline"`)

What is committed to git. **No inputs, no outputs, no timestamps**: small, reviewable, and it cannot leak model
output. Keys are sorted, two-space indent, trailing newline; the same report always produces the same bytes.

```
schema_version, kind, tool {name, version}
dataset {source, hash}
run     {repeat, seed, sample}
summary {runs, quality_mean_pass_rate, currency, total_cost, mean_cost_per_run, unpriced_runs,
         mean_tokens_per_run, latency_p50_ms, latency_p95_ms}
cases   {"<id>": {category, case_hash, graded_runs, passes, errors, mean_score}, ...}
```

## `out/gate.json` (`"kind": "llm-eval-gate"`)

The decision, for dashboards and scripts:
`{exit_code, passed, verdict, reasons[], notes[], comparison}` where `comparison` is `null` without a baseline,
else `{verdict, dataset_changed, reasons, notes, accuracy, cost_per_run, tokens_per_run, latency_p50_ms,
latency_p95_ms (each {baseline, current, absolute, percent}), cases[] {id, category, verdict, delta, p_value,
reason}}`.

## `out/report.md`

The report as Markdown for a pull request (also written by `--markdown FILE` with the comparison and decision): the
verdict and reasons, a summary table with baseline, current and change for quality, safety, cost per run, tokens per
run and p50/p95 latency, the safety failures (always listed first, with input, output and tool calls), regressed,
improved and inconclusive cases, the worst quality cases with input and output, and flaky cases. Untrusted text
is sanitised and placed in code fences longer than any run of backticks inside it.
