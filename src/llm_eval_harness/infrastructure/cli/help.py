"""Help texts."""

from __future__ import annotations

HELP = """\
llm-eval: treat LLM behaviour like tested software.

usage: llm-eval [--root DIR] [--config FILE] COMMAND [options]

commands:
  run          run the dataset through the provider, grade it, write out/report.json and out/report.md
  compare      compare a report with a baseline (informational; always exits 0)
  gate         decide for CI: run (or read a report), compare, exit 0/1/2/3/4
  baseline     turn a report into a baseline file to commit
  validate     check a dataset file and print its hash
  init-safety  write a starter safety dataset to adapt

exit codes (run and gate):
  0 pass   1 quality regression   2 usage error   3 provider or infrastructure error   4 safety failure

options:
  --root DIR     the project directory; files are read and written only inside it (default: current directory)
  --config FILE  configuration file (default: llm-eval.json if it exists)
  --version      print the version
  -h, --help     this text, or the help of a command: llm-eval COMMAND --help

documentation: README.md and docs/ in the repository.
"""

RUN_OPTIONS = """\
  --dataset FILE         dataset (.jsonl or .json); default: "dataset" of the configuration
  --repeat N             run every case N times (flakiness statistics need N >= 3)
  --concurrency N        parallel provider calls (default 4)
  --timeout SECONDS      per call (default 30)
  --retries N            retries of infrastructure errors only (default 1)
  --seed N               seed passed to the provider and used for --sample
  --sample K             run a seeded sample of K cases
  --tag T                only cases with tag T (repeatable, or comma separated)
  --category C           only quality or safety cases (repeatable)
  --case ID              only this case (repeatable)
  --out DIR              output directory inside the project (default "out")
  --no-redact            store outputs without redacting secret-shaped strings
  --price-table FILE     price table (JSON) for cost accounting; default: "price_table" of the configuration
"""

COMPARE_OPTIONS = """\
  --baseline FILE                  baseline file; default: "baseline" of the configuration
  --tolerance X                    pass-rate drop per case that is not worth flagging (default 0)
  --alpha X                        significance level for calling a change real (default 0.05)
  --deterministic                  the provider is deterministic: one run per case is a fact, not a sample
  --max-cost-increase-pct X        regression when the mean cost per run rises by more than X percent
  --max-p95-latency-increase-pct X regression when p95 latency rises by more than X percent
  --max-regressed-cases N          regressed cases tolerated (default 0)
  --markdown FILE                  write the pull request Markdown to FILE
"""

COMMAND_HELP = {
    "run": "usage: llm-eval run [options]\n\nRun every selected case through the provider and write the reports.\nExit: 0 normally, 3 if runs failed for infrastructure reasons, 4 if a safety case failed.\n\noptions:\n"
    + RUN_OPTIONS
    + "  --write-baseline FILE  also write a baseline from this run\n  --force                replace an existing baseline\n",
    "compare": 'usage: llm-eval compare [--report FILE] [--baseline FILE] [options]\n\nCompare a report with a baseline. Never fails on a regression; use gate for that.\n\noptions:\n  --report FILE          report to compare (default: OUT/report.json)\n  --out DIR              output directory of the report (default "out")\n'
    + COMPARE_OPTIONS,
    "gate": "usage: llm-eval gate [--report FILE] [--baseline FILE] [options]\n\nThe CI gate. Without --report it runs first. Exit codes: 0 pass, 1 quality regression,\n2 usage error, 3 provider or infrastructure error, 4 safety failure (always fatal).\nWithout a baseline every quality case must pass (--min-pass-rate 1).\n\noptions:\n  --report FILE          gate an existing report instead of running\n  --min-pass-rate X      minimum mean pass rate of the quality cases\n  --on-inconclusive M    fail (default) or pass when the data cannot support a verdict\n  --fail-on-removed      fail when baseline cases were not run\n"
    + COMPARE_OPTIONS
    + RUN_OPTIONS,
    "baseline": 'usage: llm-eval baseline [--report FILE] [--out FILE] [--force]\n\nWrite a stable baseline file from a report.\n\noptions:\n  --report FILE  report (default: out/report.json)\n  --out FILE     baseline file (default: "baseline" of the configuration, else baseline.json)\n  --force        replace an existing file\n',
    "validate": "usage: llm-eval validate [--dataset FILE]\n\nValidate a dataset file against the dataset format and print its hash.\n",
    "init-safety": "usage: llm-eval init-safety [--out FILE] [--force]\n\nWrite a starter dataset of prompt injection, data exfiltration and unauthorised tool call cases.\n\noptions:\n  --out FILE  default: datasets/safety.jsonl\n  --force     replace an existing file\n",
}


def command_help(command: str) -> str:
    return COMMAND_HELP.get(command, HELP)
