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
- Security: output sanitised for terminals; the pull request Markdown puts every untrusted value in an inert code span;
  ASCII-only JSON files created with mode 0600; secret-shaped strings redacted at every stored sink, through Unicode
  obfuscation, whole PEM blocks and long tokens; leak graders scan every reported field without a size cap; bounded
  reads; user regular expressions on a linear-time engine with a work budget that fails closed; child pipes handled
  without threads or blocking calls, the group killed before reaping, a deadline for the whole http exchange; no path or
  symlink escapes; no network except the `http` provider; explicit environment allow-list; nothing ever executed from
  model output.
- Gate integrity: a safety case that was not run (filtered, sampled out, removed, recategorised) is exit 4 unless
  `--allow-partial-safety`; safety cases cannot lower their threshold; failing new or changed quality cases fail the
  gate; a cost limit with unknown cost fails; reports and baselines are validated and cross-checked and a report for a
  different dataset is refused as stale; flags are held to the configuration's ranges.
- Example `examples/support-agent`: a retrieval pipeline with tools, quality and safety cases, a baseline, a CI
  workflow; the demo shows a change that improves quality while making one safety case fail, and the gate exiting 4.
- Zero runtime dependencies. Python 3.12 and 3.13. Development tools pinned by hash, container pinned by digest.
