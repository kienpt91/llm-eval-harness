# Changelog

## 0.1.0

First release.

- Datasets: JSON Lines or JSON cases with id, category (`quality` or `safety`), tags, input, context documents,
  expected facts and answers, forbidden content, expected and forbidden tool calls, metadata; a validator with
  precise, path-addressed errors that reports syntax and validation problems in one pass; order-independent
  dataset hashing.
- Providers: `command` (any program: fixed argv, JSON on stdin and stdout, deadline, size caps, allow-listed
  environment, process-group kill), `http` (loopback or explicitly allow-listed https hosts, no redirects, one
  deadline, size cap) and a deterministic `fake`. No hosted model, vendor client or price in the core.
- Eighteen graders: exact, contains, not_contains, regex, refusal, json_schema (a documented subset), numeric,
  set_overlap, token_f1, tool_calls, forbidden_tools, grounding, secret_leak, pii_leak, latency_budget,
  cost_budget, token_budget and a non-deterministic `judge` that runs an external program and records its verdict and
  confidence. Weights, `must`, per-case thresholds, default graders from the configuration.
- Runner: bounded concurrency, per-call timeout, retries for infrastructure errors only, `--repeat` with pass
  rate and variance per case, seeded sampling, tag/category/case filters, deterministic result order, token and
  cost accounting from a user-supplied price table, latency percentiles.
- Reports: terminal summary, versioned JSON (schema 1), Markdown for pull requests, a stable baseline file with no
  outputs, `out/gate.json`.
- `compare`: per-case Fisher exact test with tolerance and alpha, `inconclusive` when the data cannot support a
  verdict, `--deterministic` for scripted pipelines, cost and latency deltas with optional limits.
- `gate`: exit 0 pass, 1 quality regression, 2 usage error, 3 provider or infrastructure error, 4 safety failure
  (always fatal, never averaged away).
- `init-safety`: a starter dataset of prompt injection, data exfiltration and unauthorised tool call cases.
- Security: output sanitised for terminals and Markdown, ASCII-only JSON files, secret-shaped strings redacted
  before storing, bounded reads, screened regular expressions, no path or symlink escapes, no network except the
  `http` provider, explicit environment allow-list, nothing ever executed from model output.
- Example `examples/support-agent`: a retrieval pipeline with tools, quality and safety cases, a baseline, a CI
  workflow; the demo shows a change that improves quality while making one safety case fail, and the gate exiting 4.
- Zero runtime dependencies. Python 3.12 and 3.13. Development tools pinned by hash, container pinned by digest.
