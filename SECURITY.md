# Security

llm-eval-harness runs programs you point it at and reads text it cannot trust: model output, tool arguments,
datasets, report files. This document is the threat model: what is protected, from whom, how, how that is tested,
and what is *not* protected. Where a defence has limits, the limits are stated next to it.

## Reporting a vulnerability

Open a private security advisory on the repository (Security, Report a vulnerability). Please include the command
line, the dataset or program that triggers it, and the version (`llm-eval --version`). There is no bounty; reports
are answered and credited in the changelog.

## What is protected

| asset | why it matters |
| --- | --- |
| the machine and CI environment that run the harness | a pipeline under test is code; the harness must not widen its reach |
| secrets in the environment (API keys, CI tokens) | the programs under test and the judge must not see more than they are given |
| files outside the project directory | datasets, reports and baselines must not read or overwrite them |
| the terminal and the pull request | output of a model must not be able to rewrite what a reviewer sees |
| the verdict | a hostile output, dataset, program or file must not make a failure look like a pass, or hang the gate |

## Actors and trust

| actor | trusted? |
| --- | --- |
| the person running the harness, and the configuration file, price table and **dataset** they commit | trusted, but their files are validated strictly (typos and malformed data are errors, not silent) |
| the **program under test** (`command` provider), the **endpoint** (`http`), the **judge program** | *code you chose to run*; their **output** is untrusted. The harness limits their reach (below) but is **not a sandbox** |
| **model output**, tool call arguments, citations, the reported model name and token counts | untrusted: it may carry escape sequences, bidirectional marks, gigabytes, deep JSON, prototype-pollution keys, secrets, Markdown aimed at a human reviewer, prompt injection |
| report and baseline files read by `compare`, `gate` and `baseline` | untrusted structure: validated field by field and cross-checked (see "Reports are validated, not trusted") |

## Threats and mitigations

Each row names the test that exercises it, so a reviewer can run it (`scripts/py.sh 'python -m pytest tests -k NAME'`).

| threat | mitigation | tests |
| --- | --- | --- |
| model output executed | no code path evaluates output: no `eval`, `exec`, `compile`, `pickle`, `shell=True` anywhere. An AST **lint** (see its limits below) fails the build on those, on aliases of them, on computed `getattr`, and on interpreter-internals attributes; output is data in a report | `test_layers.py::test_checker_flags_breaking_snippet`, `test_command_provider.py::test_model_output_is_never_executed`, `test_process.py::test_no_shell_arguments_are_passed_literally` |
| terminal injection by model output (ANSI, OSC 8 links, other control characters, bidirectional marks, zero-width characters, lone surrogates) | everything printed passes `sanitize`: complete escape sequences are removed, any other control, format, private-use, unassigned or separator character is shown as a visible escape (`\x07`, `<U+202E>`); this includes ids, categories and every message | `test_text.py`, `test_render.py::test_terminal_output_has_no_control_or_bidi_characters_from_untrusted_text`, `test_markdown_injection.py::test_the_terminal_sanitises_every_field_even_a_category_edited_by_hand`, `test_cli.py::test_terminal_and_error_output_never_carry_control_sequences` |
| Markdown, HTML, links, images, `@mentions`, `#123` references or autolinks injected into a pull request comment | the Markdown report never puts an untrusted value in plain text: every value from a model, a program, a dataset, a configuration or a report file is an inline code span (outputs: a fenced block) whose delimiter is longer than any backtick run inside it, after being made single-line and sanitised; table pipes are escaped; the report has no HTML and no links of its own. Inside a code span a renderer makes none of those things live | `test_markdown_injection.py` (hostile text in grader details, error messages, tool names, ids, categories, sources; asserts nothing markup-capable survives outside code) |
| model output smuggled into files | report and baseline JSON are written ASCII-only (`\uXXXX` escapes): raw control or bidi characters cannot reach a file; files are created with mode 0600 | `test_usecases.py::test_the_report_is_pure_ascii_json`, `test_cli.py::test_reports_are_private_files` |
| unbounded memory from a huge answer | the `command` provider reads stdout in chunks and kills the program at `max_output_bytes`; stderr is capped (and drained so the program cannot block); the `http` provider reads at most the cap; datasets, configs and reports have size limits; stored output is truncated; the redactor looks at no more than 32,768 characters of any string | `test_process.py::test_endless_stdout_is_cut_at_the_cap_and_the_child_killed`, `test_http_provider.py::test_oversized_and_endless_bodies_are_cut_at_the_cap`, `test_runner.py::test_redaction_looks_at_most_at_the_first_32k_characters_of_any_string_and_says_so` |
| deep JSON, duplicate keys, `NaN`, huge integers | strict parser: nesting limit 32 (checked without recursion before parsing), duplicate keys and non-finite numbers rejected, integer digit limit of the runtime | `test_jsonio.py`, `test_command_provider.py::test_hostile_programs_become_precise_provider_errors` |
| prototype-pollution keys (`__proto__`, `constructor`) | plain data everywhere; nothing assigns attributes from keys; the harness does not merge objects | `test_jsonio.py::test_prototype_pollution_keys_are_plain_data`, `test_dataset.py::test_hostile_values_are_plain_data` |
| the program under test reads the harness's secrets | the child environment is an **explicit allow-list**: `PATH`, `LANG`, `LC_ALL`, names in `env_allow`, fixed `env_set`, `LLM_EVAL=1`; never the whole environment. The judge program gets the same treatment. The answer key is never sent (expectations and graders are not in the request) | `test_command_provider.py::test_the_child_environment_is_an_explicit_allow_list`, `test_process.py::test_the_environment_is_exactly_what_was_passed`, `test_protocol.py::test_request_carries_input_and_context_but_never_the_answer_key` |
| a program that hangs, floods, spawns children or leaves zombies, and the harness hanging with it | no threads and no blocking calls: stdin is written and stdout/stderr read with non-blocking `os.read`/`os.write` driven by `selectors` against one deadline, so the call cannot block on a pipe; the program runs in its own session and, on timeout, overflow or exit, the whole process group is killed **before** the child is reaped (`waitid` with `WNOWAIT`: the pid cannot be recycled in between, on platforms that have it); inherited descriptors are closed; descriptors are closed through their stream objects exactly once. **Limit:** a process that leaves the session (`setsid`, a double fork with `setsid`) is not in the group and is not killed; the harness still returns on time (it does not wait for pipes to reach end of file after the program has exited, and never waits for a process it did not start) | `test_process.py` (hang, deaf program, orphan and process-tree kills, an escaped `setsid` helper that holds the pipes, group killed before reaping, no zombies, no inherited descriptors, 48 concurrent runs) |
| write outside the project | every path is resolved and must be inside the project root; `..`, absolute paths and symbolic links that leave it are refused for reads and writes (also the default configuration file); writes never follow a symlink, are atomic (temporary file, rename) and do not overwrite without `--force` | `test_fs.py`, `test_cli.py::test_output_and_inputs_cannot_leave_the_project`, `test_cli.py::test_a_default_config_that_is_a_symlink_out_of_the_project_is_refused_not_ignored` |
| the harness used as a network client (SSRF) | no network in the core; the `http` provider is the only module that can open a socket (an AST lint enforces it) and only to loopback or hosts you list (exact match, https required, no URL credentials, no redirects). One deadline covers connect, request, response headers and body, enforced by a watchdog that shuts the socket down. **Limit:** name resolution cannot be interrupted (loopback names and the hosts you listed only) | `test_layers.py`, `test_http_provider.py` (host policy, redirect, credentials, suffix tricks, header and body dripping, oversized bodies) |
| secrets in stored reports | every string that is stored passes the redactor: outputs, grader details, tool names and arguments (keys too), citations, ids of retrieved documents, the reported model name, inputs, error messages, judge reasons. Secret-shaped strings become `[REDACTED:kind]`; extra patterns are configurable; `--no-redact` and the config switch say so loudly in the summary. Only the first 32,768 characters of a string are examined and kept, the rest is replaced by a count. Header values are never stored, only header names. Grading runs on the original, redaction on what is stored. See "What the secret and personal-data shapes recognise" for what is and is not found | `test_runner.py::test_grader_details_never_carry_secret_shaped_text_into_the_report`, `test_runner.py::test_stored_output_is_redacted_but_grading_saw_the_original`, `test_shapes_hardening.py`, `test_cli.py::test_secret_shaped_strings_are_redacted_in_stored_reports_unless_switched_off` |
| a leak reported by the harness itself | the leak graders name kinds and counts, never the text | `test_graders_leaks_budgets_judge.py`, `test_leaks_hardening.py::test_the_report_of_a_leak_names_kinds_and_counts_only` |
| a detector fails open at a bound | the leak graders scan every reported string without a size cap (each string on its own); patterns and text over a bound fail closed; an exhausted work budget fails closed | `test_leaks_hardening.py`, `test_regex_bounded.py` |
| a grader crash or a judge outage turned into a pass | an exception in a grader fails the grade (closed); a judge that cannot be reached makes the run an error (exit 3); unknown cost fails a cost budget and a cost limit; a safety case that errored is exit 3, never a pass | `test_runner.py::test_a_crashing_grader_fails_closed_without_stopping_the_run`, `test_integrity.py`, `test_compare_gate.py` |
| retries hiding a regression | retries apply to infrastructure errors only; a graded answer is never re-run | `test_runner.py::test_failed_grades_are_never_retried` |
| a safety case skipped, sampled out, removed, recategorised or softened | the gate exits 4 when any safety case of the dataset (or of the baseline) was not run, unless you pass `--allow-partial-safety`; safety cases cannot lower their threshold or carry a grader that could never fail them | `test_integrity.py`, `test_cli.py::test_a_gate_that_leaves_the_safety_suite_out_does_not_pass`, `test_cli.py::test_removing_a_safety_case_from_the_dataset_is_caught_against_the_baseline` |
| dependency compromise | zero runtime dependencies; development tools pinned by version and **sha256** (`requirements-dev.txt`), none uploaded less than seven days before the cutoff and none yanked, checked by `scripts/check-dep-age.py` in CI (itself tested against a fake index); the container image is pinned by digest; GitHub Actions are pinned by commit SHA with read-only permissions | `test_layers.py::test_package_has_no_runtime_dependencies_declared`, `test_scripts.py` |

### Reports are validated, not trusted

`compare`, `gate --report` and `baseline` read files that may have been edited. The readers check every field and
cross-check the redundancy: per case the counts (`graded_runs`, `passes`, `errors`) and `status` must equal what its
runs show, the summary (cases, runs, safety failures, tokens, cost, latency percentiles, pass rate) must equal what
the cases give, case ids must be valid and unique, categories known, run counts at most 1,000, and the baseline's
counts bounded before any statistic runs. A report that contradicts itself is refused with the field named, so a
negative `passes` cannot hide a safety failure and an enormous `graded_runs` cannot make the Fisher test run for
hours. A report is also refused as **stale** when the configured dataset no longer has the hash it was made with.

**This is consistency, not authenticity.** A report is not signed: someone who can write a consistent report (all
runs edited) can write any verdict. Run `gate` without `--report` in CI (it runs the pipeline itself), and treat
`out/` as an artifact of that run.

### What the secret and personal-data shapes recognise

Shapes, not secrets (details in [docs/graders.md](docs/graders.md#secret_leak)):

- **Found:** private key blocks (the `BEGIN` line through the `END` line, or 16 KiB when there is none), cloud access
  key ids, source-host tokens (classic and fine-grained), payment keys, webhook secrets, chat tokens, API keys of Google
  style, package registry tokens, `sk-` style keys, JWTs, URLs with a password, quoted `api_key = "..."`
  assignments, 64-digit hex keys, bearer tokens; emails, phone numbers with separators, US SSN shapes, Luhn-valid card
  numbers. Text is NFKC-normalised and invisible characters are dropped before matching, so full-width letters and a
  zero-width character inside a key do not hide it; spans map back to the original, so redaction removes the original
  characters. A token glued to punctuation (`my_<token>`, `key=<token>`) is found; one glued to a letter or digit is not.
  A match is extended over the token characters that follow it, so a token longer than the pattern's cap is redacted whole.
- **Not found:** a password or token of no particular shape; a key split over lines or strings; base64 or hex of a
  key; homoglyphs from other scripts; a JWT with a segment over 4,096 characters; names, street addresses,
  free-text health data; phone numbers without separators. For a secret you know, plant a canary and forbid it.

### Regular expressions

User patterns (datasets, configuration, redaction lists) never reach Python's `re`, which backtracks, cannot be
interrupted and holds the interpreter lock: a pattern accepted by any screen (`.*a.*b.*c.*d` on 33 KiB of text ran
past twenty seconds in a check) can hang the whole run. They run on a small engine in `domain/grading/regexengine.py`:

- a Pike VM, linear in `len(text) * len(program)` for **any** pattern (nested quantifiers are safe), restricted
  syntax (no back-references, look-around, possessive or atomic forms, inline flags, `\p{..}`; programs over 2,000
  instructions are refused, repetition counts at most 1,000), refused at load time with the position;
- a step budget per call (2,000,000, about two seconds of work); when it is spent the grader **fails closed**, also
  for `must_match: false` and `not_contains`;
- user patterns only run on text up to 32,768 characters; longer text fails the grader (closed);
- the redactor applies the same engine to at most 32,768 characters per string, and a redaction pattern that spends its
  budget replaces the whole string by a marker instead of storing it unexamined;
- the engine is compared with `re` on seeded random patterns and texts (leftmost-first spans, full matches, flags).

**Limits:** the engine is pure Python, so a hostile pattern costs up to the budget (seconds) per call; a dataset full of
such patterns makes a slow run, not a hung one. Case-insensitive matching uses simple lower/upper mapping, not full
Unicode case folding. The built-in secret shapes still use `re`; they are fixed, anchored, bounded patterns tested for
linear time on hostile megabytes.

### What the architecture test is

`tests/architecture/test_layers.py` parses every source file with `ast` and fails the build on forbidden imports (by
layer and, for the pure layers, by allow-list), on third-party imports, on process creation or sockets outside their
one module, on `eval`/`exec`/`compile`/`__import__` and any *reference* to them (aliases), on `getattr`/`setattr`/
`delattr` with a computed name or a forbidden one, on attributes that reach the interpreter's internals (`__builtins__`,
`__globals__`, `__subclasses__`, `__dict__`, `sys.modules`, ...), on `os.system`/`popen`, on `os.exec*`/`spawn*`/`fork`/
`kill*`/`wait*` outside the process module, on environment access outside composition (through any alias of `os` or
`from os import ...`), on `shell=` other than the literal `False`, and on wildcard imports. Each rule has a snippet
that must fail and one that must pass.

**It is a lint, not a sandbox.** It stops honest mistakes and the obvious evasions, not a determined contributor: code
built from strings and handed to something it cannot see (a function fetched from a container built at run time, a
module reached through a path it does not know) passes. Code review is the control for that.

## Not protected (read this before relying on it)

- **The program under test is not sandboxed.** It runs with your privileges, with the working directory you chose.
  A malicious or compromised pipeline can read your files and use the network. The environment allow-list stops
  *accidental* secret exposure, not a hostile program. A helper it starts in a new session survives the harness. Run
  unknown pipelines in a container with no secrets and no network, as the development scripts do (`scripts/py.sh`).
- The harness does not limit CPU or memory of the program under test; only wall-clock time and output size.
- **Token counts, model names, and so cost, are reported by the pipeline.** A pipeline can under-report usage or name a
  cheap model; the harness measures latency itself but cannot check usage. A cost *limit* fails when cost is unknown, a
  cost *budget* fails when it is unknown, but a believable false number passes.
- POSIX only (Linux, macOS). Process-group handling is not implemented for Windows. Without `os.waitid` (macOS) the
  process group is killed after the child has been reaped, which leaves a window of microseconds in which the pid
  could in theory be reused.
- A time-of-check/time-of-use race exists in the file guard (a symlink swapped between resolution and open); it
  requires an attacker who can write inside your project at the same moment.
- The graders are heuristics. A `secret_leak` pass does not mean no secret was leaked. A judge is non-deterministic
  and can be fooled by the text it grades.
- `http` over `https` uses the platform's default certificate store and verification.
- Redaction is by shape, and a string over 32,768 characters is cut: what lies beyond is not stored. The baseline holds
  no text at all.
- Reports are consistent, not authentic (see above).

## Verification status

What the tests establish and what they do not is stated in the README ("What is verified, and what is not").
