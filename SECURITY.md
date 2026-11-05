# Security

llm-eval-harness runs programs you point it at and reads text it cannot trust: model output, tool arguments,
datasets, report files. This document is the threat model: what is protected, from whom, how, how that is tested,
and what is *not* protected.

## Reporting a vulnerability

Open a private security advisory on the repository (Security, Report a vulnerability). Please include the command line, the dataset or program that triggers it, and the
version (`llm-eval --version`). There is no bounty; reports are answered and credited in the changelog.

## What is protected

| asset | why it matters |
| --- | --- |
| the machine and CI environment that run the harness | a pipeline under test is code; the harness must not widen its reach |
| secrets in the environment (API keys, CI tokens) | the programs under test and the judge must not see more than they are given |
| files outside the project directory | datasets, reports and baselines must not read or overwrite them |
| the terminal and the pull request | output of a model must not be able to rewrite what a reviewer sees |
| the verdict | a hostile output must not make a failure look like a pass, or hang the gate |

## Actors and trust

| actor | trusted? |
| --- | --- |
| the person running the harness, and the configuration file, price table and **dataset** they commit | trusted, but their files are validated strictly (typos and malformed data are errors, not silent) |
| the **program under test** (`command` provider), the **endpoint** (`http`), the **judge program** | *code you chose to run*; their **output** is untrusted. The harness limits their reach (below) but is **not a sandbox** |
| **model output**, tool call arguments, citations | untrusted: it may carry escape sequences, bidirectional marks, gigabytes, deep JSON, prototype-pollution keys, secrets, prompt injection aimed at a human reviewer |
| report and baseline files read by `compare` and `gate` | untrusted structure: validated field by field |

## Threats and mitigations

Each row names the test that exercises it, so a reviewer can run it (`scripts/py.sh 'python -m pytest tests -k NAME'`).

| threat | mitigation | tests |
| --- | --- | --- |
| model output executed | no code path evaluates output: no `eval`, `exec`, `compile`, `pickle`, `shell=True` anywhere (an AST test fails the build if one appears); output is data in a report | `test_layers.py::test_checker_flags_breaking_snippet`, `test_command_provider.py::test_model_output_is_never_executed`, `test_process.py::test_no_shell_arguments_are_passed_literally` |
| terminal or Markdown injection by model output (ANSI, OSC 8 links, other control characters, bidirectional marks, zero-width characters, lone surrogates) | everything printed passes `sanitize`: complete escape sequences are removed, any other control, format, private-use, unassigned or separator character is shown as a visible escape (`\x07`, `<U+202E>`); Markdown puts untrusted text in code fences longer than any backtick run inside it, escapes table cells and code spans | `test_text.py`, `test_render.py::test_terminal_output_has_no_control_or_bidi_characters_from_untrusted_text`, `test_cli.py::test_terminal_and_error_output_never_carry_control_sequences` |
| model output smuggled into files | report and baseline JSON are written ASCII-only (`\uXXXX` escapes): raw control or bidi characters cannot reach a file | `test_usecases.py::test_the_report_is_pure_ascii_json` |
| unbounded memory from a huge answer | the `command` provider reads stdout in chunks and kills the program at `max_output_bytes`; stderr is capped (and drained so the program cannot block); the `http` provider reads at most the cap; datasets, configs and reports have size limits; stored output is truncated, grading sees the whole (bounded) answer | `test_process.py::test_endless_stdout_is_cut_at_the_cap_and_the_child_killed`, `test_http_provider.py::test_oversized_and_endless_bodies_are_cut_at_the_cap`, `test_fs.py` |
| deep JSON, duplicate keys, `NaN`, huge integers | strict parser: nesting limit 32 (checked without recursion before parsing), duplicate keys and non-finite numbers rejected, integer digit limit of the runtime | `test_jsonio.py`, `test_command_provider.py::test_hostile_programs_become_precise_provider_errors` |
| prototype-pollution keys (`__proto__`, `constructor`) | plain data everywhere; nothing assigns attributes from keys; reports are JSON, which downstream JavaScript can parse safely only if it also avoids unsafe merges: the harness does not merge | `test_jsonio.py::test_prototype_pollution_keys_are_plain_data`, `test_dataset.py::test_hostile_values_are_plain_data` |
| the program under test reads the harness's secrets | the child environment is an **explicit allow-list**: `PATH`, `LANG`, `LC_ALL`, names in `env_allow`, fixed `env_set`, `LLM_EVAL=1`; never the whole environment. The judge program gets the same treatment. The answer key is never sent (expectations and graders are not in the request) | `test_command_provider.py::test_the_child_environment_is_an_explicit_allow_list`, `test_process.py::test_the_environment_is_exactly_what_was_passed`, `test_protocol.py::test_request_carries_input_and_context_but_never_the_answer_key` |
| a program that hangs, floods, spawns children or leaves zombies | own session; on timeout, overflow or exit the whole process group is killed and reaped; stdin is written from a thread (a program that never reads cannot wedge the harness); inherited file descriptors are closed | `test_process.py` (hang, deaf program, orphan and process-tree kills, no zombies, no inherited descriptors) |
| write outside the project | every path is resolved and must be inside the project root; `..`, absolute paths and symbolic links that leave it are refused for reads and writes; writes never follow a symlink, are atomic (temporary file, rename) and do not overwrite without `--force` | `test_fs.py`, `test_cli.py::test_output_and_inputs_cannot_leave_the_project` |
| the harness used as a network client (SSRF) | no network in the core; the `http` provider is the only module that can open a socket (an AST test enforces it) and only to loopback or hosts you list (exact match, https required, no URL credentials, no redirects, one deadline) | `test_layers.py`, `test_http_provider.py` (host policy, redirect, credentials, suffix tricks) |
| secrets in stored reports | secret-shaped strings (the shapes of the repository's own secret scanner) are replaced by `[REDACTED:kind]` in outputs, inputs, tool arguments, messages and judge reasons before anything is stored; configurable patterns; `--no-redact` and the config switch say so loudly in the summary. Header values are never stored, only header names | `test_runner.py::test_stored_output_is_redacted_but_grading_saw_the_original`, `test_cli.py::test_secret_shaped_strings_are_redacted_in_stored_reports_unless_switched_off`, `test_http_provider.py::test_headers_static_and_from_the_environment_and_never_described` |
| a leak reported by the harness itself | the leak graders name kinds and counts, never the text | `test_graders_leaks_budgets_judge.py` |
| a grader crash or a judge outage turned into a pass | an exception in a grader fails the grade (closed); a judge that cannot be reached makes the run an error (exit 3); unknown cost fails a cost budget; a safety case that errored is exit 3, never a pass | `test_runner.py::test_a_crashing_grader_fails_closed_without_stopping_the_run`, `test_compare_gate.py` |
| retries hiding a regression | retries apply to infrastructure errors only; a graded answer is never re-run | `test_runner.py::test_failed_grades_are_never_retried` |
| a tampered baseline or report | read with the same strict validation as every other input; a baseline carries no output text; a baseline can never excuse a safety failure | `test_report_baseline.py`, `test_compare_gate.py::test_a_failing_safety_case_is_fatal_whatever_else_is_true` |
| dependency compromise | zero runtime dependencies; development tools pinned by version and **sha256** (`requirements-dev.txt`), none uploaded less than seven days before the cutoff, checked by `scripts/check-dep-age.py` in CI; the container image is pinned by digest; GitHub Actions are pinned by commit SHA with read-only permissions | `test_layers.py::test_package_has_no_runtime_dependencies_declared` |

## Regular expressions

Python's `re` cannot be interrupted, so a hostile pattern or hostile text could hang a run. Defences:

- Patterns in datasets, configuration and redaction lists are **screened at load time**: at most 512 characters,
  no back-references, no look-around, and no quantified group that itself repeats or alternates without a bound
  (`(a+)+`, `(.*)*`, `(a|aa)*`): the shapes behind exponential backtracking.
- A user pattern is only run on text up to **32,768 characters**; longer text makes the grader **fail** (closed),
  because a quadratic pattern on a megabyte of adversarial text would still be too slow.
- The built-in secret and personal-data shapes are anchored at token boundaries with bounded possessive
  repetition: scanning a hostile megabyte (twelve adversarial shapes are tested) takes linear time.

The screen is conservative, not a proof. A pattern that passes can still be slow on adversarial input within the
32 KiB limit (seconds, not hours); keep patterns simple.

## Not protected (read this before relying on it)

- **The program under test is not sandboxed.** It runs with your privileges, with the working directory you chose.
  A malicious or compromised pipeline can read your files and use the network. The environment allow-list stops
  *accidental* secret exposure, not a hostile program. Run unknown pipelines in a container with no secrets and
  no network, as the development scripts do (`scripts/py.sh`).
- The harness does not limit CPU or memory of the program under test; only wall-clock time and output size.
- POSIX only (Linux, macOS). Process-group handling is not implemented for Windows.
- A time-of-check/time-of-use race exists in the file guard (a symlink swapped between resolution and open); it
  requires an attacker who can write inside your project at the same moment. Process-group kill relies on the
  group id not being reused within the instant between the program exiting and the kill.
- The graders are heuristics. A `secret_leak` pass does not mean no secret was leaked ([graders.md](docs/graders.md)
  lists each grader's failure modes). A judge is non-deterministic and can be fooled by the text it grades.
- `http` over `https` uses the platform's default certificate store and verification.
- Redaction is by shape: a secret with no recognisable shape is stored as is. The baseline holds no text at all.

## Verification status

What the tests establish and what they do not is stated in the README ("What is verified, and what is not").
