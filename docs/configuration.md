# Configuration

`llm-eval.json` in the project directory (or `--config FILE`) holds what does not change from run to run. Every
setting can be overridden on the command line; the command line wins, then the file, then the default. All paths
are relative to the project directory (`--root`, default: the current directory) and must stay inside it.

```json
{
  "dataset": "datasets/support.jsonl",
  "provider": {"type": "command", "argv": ["python3", "pipeline.py"], "env_allow": ["MY_API_KEY"]},
  "price_table": "prices.json",
  "baseline": "baseline.json",
  "out_dir": "out",
  "defaults": {"repeat": 1, "concurrency": 4, "timeout_s": 30, "retries": 1, "seed": 7},
  "graders": [{"type": "latency_budget", "max_ms": 5000, "name": "latency"}],
  "redact": {"enabled": true, "patterns": ["ORD-\\d{4,}"]},
  "gate": {"min_pass_rate": 0.9, "on_inconclusive": "fail", "tolerance": 0.0, "alpha": 0.05,
           "deterministic": false, "max_regressed_cases": 0, "max_cost_increase_pct": 20,
           "fail_on_removed": false}
}
```

Unknown keys are errors. Every problem in the file is reported with its path (`gate.alpha: must be at most 0.5`).

| key | default | meaning |
| --- | --- | --- |
| `dataset` | none | the dataset file (`--dataset`) |
| `provider` | none | see below (required to run) |
| `price_table` | none | price table file for cost accounting (`--price-table`) |
| `baseline` | none | the baseline `compare` and `gate` use (`--baseline`) |
| `out_dir` | `out` | where `report.json`, `report.md` and `gate.json` go (`--out`); must be inside the project |
| `defaults.repeat` | 1 | runs per case (`--repeat`), 1 to 1000 |
| `defaults.concurrency` | 4 | parallel provider calls (`--concurrency`), 1 to 64 |
| `defaults.timeout_s` | 30 | deadline per provider call (`--timeout`) |
| `defaults.retries` | 1 | retries of **infrastructure** errors only (`--retries`), 0 to 10 |
| `defaults.seed` | none | passed to the provider and used by `--sample` (`--seed`) |
| `graders` | none | graders added to **every** case (same syntax as a case's `graders`) |
| `redact.enabled` | true | redact secret-shaped strings in stored reports (`--no-redact` switches it off) |
| `redact.patterns` | none | extra regular expressions to redact (screened like all patterns) |
| `gate.*` | see [gate.md](gate.md) | defaults of the gate and compare options |

## `provider`

**`command`**: `argv` (list, required, at most 64 items), `cwd` (inside the project), `env_allow` (names of
variables to pass on, in addition to `PATH`, `LANG`, `LC_ALL`), `env_set` (fixed values), `max_output_bytes`
(1,024 to 16 MiB, default 1 MiB). See [wiring.md](wiring.md#what-the-command-provider-guarantees).

**`http`**: `url` (required), `allow_hosts`, `headers` (static values, no line breaks), `headers_env`
(`{"Header-Name": "ENV_VARIABLE"}`), `max_output_bytes`. See [wiring.md](wiring.md#the-http-provider).

**`fake`**: see below.

### The `fake` provider

A deterministic built-in for tests and demos; no process, no network, no clock, no randomness.

```json
{"type": "fake", "model": "fake-model", "latency_ms": 2,
 "default": {"text": "I cannot help with that."},
 "script": {
   "case-id": {"text": "30 days", "tool_calls": [{"name": "lookup", "arguments": {"id": 1}}]},
   "flaky-case": [{"text": "right"}, {"text": "wrong"}],
   "unreliable-case": {"error": {"kind": "timeout", "message": "slow", "fail_attempts": 1, "then": {"text": "ok"}}}
 }}
```

An answer has the keys of the [wire protocol](wiring.md#the-protocol) and may carry `latency_ms`. A list gives one
answer per repeat (`repeat % length`), which is how flakiness is scripted. An error entry fails the first
`fail_attempts` attempts with the given `kind` (`timeout`, `spawn`, `exit`, `network`, `protocol`, `refused`) and then
answers with `then`. Without a script entry and a `default`, the case input is echoed. Token usage is estimated
from text length (four characters per token) unless the answer gives its own.

## Price table

There are **no built-in prices**: a price table is a file you supply, so a number in a report is a number you
chose.

```json
{"currency": "USD", "models": {"my-model": {"input_per_mtok": 0.15, "output_per_mtok": 0.60}}}
```

Prices are per million tokens. The cost of a run is `input_tokens * input_per_mtok + output_tokens *
output_per_mtok`, divided by 1,000,000, in exact decimal arithmetic, from the `usage` the provider reports. A run
whose usage or model is not in the table has an **unknown** cost: it is never counted as zero, the report shows
the cost as unknown instead of a partial total, and a [`cost_budget`](graders.md#cost_budget) fails on it.

## Secrets and the environment

The harness reads the process environment in one place: to build the allow-listed environments of your programs and
to read the variables named in `headers_env`. Secrets are never printed, never stored in a report, and never
passed to a program that did not name them in `env_allow`.
