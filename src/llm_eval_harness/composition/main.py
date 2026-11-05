"""The command line: parses arguments, wires adapters, runs one use case, prints, and returns the exit code."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from llm_eval_harness._version import __version__
from llm_eval_harness.application.config import Config, parse_config
from llm_eval_harness.application.dataset_loader import load_dataset
from llm_eval_harness.application.redaction import Redactor
from llm_eval_harness.application.report.gate_json import gate_document
from llm_eval_harness.application.report.json_files import json_text
from llm_eval_harness.application.report.markdown import render_markdown
from llm_eval_harness.application.report.terminal import render_terminal
from llm_eval_harness.application.selection import Selection
from llm_eval_harness.application.usecases import (
    RunOutcome,
    RunRequest,
    compare_with_baseline,
    init_safety_suite,
    load_config,
    load_prices,
    read_baseline,
    read_report,
    require,
    run_evaluation,
    split_values,
    write_baseline,
)
from llm_eval_harness.composition.compose import build_judge, build_provider
from llm_eval_harness.domain import exitcodes
from llm_eval_harness.domain.compare import CompareOptions, Comparison
from llm_eval_harness.domain.errors import HarnessError, UsageError, ValidationError
from llm_eval_harness.domain.gate import GateDecision, GateOptions, decide
from llm_eval_harness.domain.reportdoc import ReportDoc
from llm_eval_harness.domain.text import sanitize
from llm_eval_harness.infrastructure.cli.args import CliArgs, parse_args, usage_text
from llm_eval_harness.infrastructure.clock import SystemClock
from llm_eval_harness.infrastructure.fs import ProjectFilesystem

DEFAULT_CONFIG = "llm-eval.json"
DEFAULT_SAFETY_PATH = "datasets/safety.jsonl"
DEFAULT_BASELINE = "baseline.json"


@dataclass(frozen=True, slots=True)
class Context:
    args: CliArgs
    files: ProjectFilesystem
    environ: Mapping[str, str]
    config: Config
    stdout: TextIO
    stderr: TextIO

    @property
    def root(self) -> Path:
        return self.files.root

    def say(self, text: str) -> None:
        self.stdout.write(text if text.endswith("\n") else text + "\n")


def _config(files: ProjectFilesystem, explicit: str | None) -> Config:
    if explicit is not None:
        return load_config(files, explicit)
    if os.path.lexists(files.root / DEFAULT_CONFIG):  # a symlink that leaves the project is an error, not "no file"
        return load_config(files, DEFAULT_CONFIG)
    return parse_config({}, "defaults")


def _evaluate(ctx: Context) -> RunOutcome:
    args, config = ctx.args, ctx.config
    if config.provider is None:
        raise UsageError(
            'no provider configured: add a "provider" to the configuration file (see docs/configuration.md)'
        )
    provider = build_provider(config.provider, ctx.root, ctx.environ)
    prices = load_prices(ctx.files, args.price_table or config.price_table)
    redactor = Redactor.build(config.redact and not args.no_redact, config.redact_patterns)
    request = RunRequest(
        dataset_path=require(args.dataset or config.dataset, "the dataset (--dataset)"),
        selection=Selection(
            tags=split_values(args.tags),
            categories=split_values(args.categories),
            case_ids=split_values(args.case_ids),
            sample=args.sample,
            seed=args.seed if args.seed is not None else config.defaults.seed,
        ),
        repeat=args.repeat if args.repeat is not None else config.defaults.repeat,
        concurrency=args.concurrency if args.concurrency is not None else config.defaults.concurrency,
        timeout_s=args.timeout if args.timeout is not None else config.defaults.timeout_s,
        retries=args.retries if args.retries is not None else config.defaults.retries,
        seed=args.seed if args.seed is not None else config.defaults.seed,
        out_dir=args.out or config.out_dir,
        version=__version__,
    )
    return run_evaluation(
        ctx.files, config, provider, build_judge(ctx.root, ctx.environ), SystemClock(), request, prices, redactor
    )


def _out_dir(ctx: Context) -> str:
    return (ctx.args.out or ctx.config.out_dir).rstrip("/")


def _report_path(ctx: Context) -> str:
    return ctx.args.report or f"{_out_dir(ctx)}/report.json"


def _compare_options(ctx: Context) -> CompareOptions:
    a, g = ctx.args, ctx.config.gate
    return CompareOptions(
        tolerance=a.tolerance if a.tolerance is not None else g.tolerance,
        alpha=a.alpha if a.alpha is not None else g.alpha,
        deterministic=a.deterministic or g.deterministic,
        max_regressed_cases=a.max_regressed_cases if a.max_regressed_cases is not None else g.max_regressed_cases,
        max_cost_increase_pct=a.max_cost_increase_pct
        if a.max_cost_increase_pct is not None
        else g.max_cost_increase_pct,
        max_p95_latency_increase_pct=a.max_p95_latency_increase_pct
        if a.max_p95_latency_increase_pct is not None
        else g.max_p95_latency_increase_pct,
    )


def _comparison(ctx: Context, doc: ReportDoc, *, required: bool) -> Comparison | None:
    path = ctx.args.baseline or ctx.config.baseline
    if path is None:
        if required:
            raise UsageError('a baseline is required (--baseline FILE, or "baseline" in the configuration)')
        return None
    return compare_with_baseline(read_baseline(ctx.files, path), doc, _compare_options(ctx))


def _write_markdown(ctx: Context, doc: ReportDoc, comparison: Comparison | None, decision: GateDecision | None) -> None:
    if ctx.args.markdown is None:
        return
    text = render_markdown(doc, comparison, decision)
    ctx.say(f"wrote {ctx.files.write_text(ctx.args.markdown, text, overwrite=True)}")


def cmd_run(ctx: Context) -> int:
    outcome = _evaluate(ctx)
    ctx.say(render_terminal(outcome.doc))
    for path in outcome.written:
        ctx.say(f"wrote {path}")
    if ctx.args.write_baseline is not None:
        ctx.say(f"wrote {write_baseline(ctx.files, outcome.doc, ctx.args.write_baseline, overwrite=ctx.args.force)}")
    # `run` measures: only safety failures and runs that could not be graded decide its exit code
    return decide(outcome.doc, None, GateOptions(min_pass_rate=0.0)).exit_code


def cmd_compare(ctx: Context) -> int:
    doc = read_report(ctx.files, _report_path(ctx))
    comparison = _comparison(ctx, doc, required=True)
    assert comparison is not None
    ctx.say(render_terminal(doc, comparison))
    _write_markdown(ctx, doc, comparison, None)
    return exitcodes.OK


def cmd_gate(ctx: Context) -> int:
    args, gate = ctx.args, ctx.config.gate
    if args.report is not None and args.has_run_options():
        raise UsageError("--report gates an existing report; it cannot be combined with options that start a run")
    if args.report is not None:
        doc = read_report(ctx.files, args.report)
    else:
        outcome = _evaluate(ctx)
        doc = outcome.doc
        for path in outcome.written:
            ctx.say(f"wrote {path}")
    comparison = _comparison(ctx, doc, required=False)
    decision = decide(
        doc,
        comparison,
        GateOptions(
            min_pass_rate=args.min_pass_rate if args.min_pass_rate is not None else gate.min_pass_rate,
            on_inconclusive=args.on_inconclusive or gate.on_inconclusive,
            fail_on_removed=args.fail_on_removed or gate.fail_on_removed,
        ),
    )
    ctx.say(render_terminal(doc, comparison, decision))
    _write_markdown(ctx, doc, comparison, decision)
    ctx.say(
        f"wrote {ctx.files.write_text(f'{_out_dir(ctx)}/gate.json', json_text(gate_document(decision, comparison)), overwrite=True)}"
    )
    return decision.exit_code


def cmd_baseline(ctx: Context) -> int:
    doc = read_report(ctx.files, _report_path(ctx))
    target = ctx.args.out or ctx.config.baseline or DEFAULT_BASELINE
    ctx.say(f"wrote {write_baseline(ctx.files, doc, target, overwrite=ctx.args.force)}")
    ctx.say("commit this file: the next `llm-eval gate` compares against it")
    return exitcodes.OK


def cmd_validate(ctx: Context) -> int:
    path = require(ctx.args.dataset or ctx.config.dataset, "the dataset (--dataset)")
    dataset = load_dataset(ctx.files, path)
    safety = sum(1 for c in dataset.cases if c.is_safety)
    tags = sorted({t for c in dataset.cases for t in c.tags})
    ctx.say(f"{path}: ok")
    ctx.say(f"  cases    {len(dataset.cases)} ({len(dataset.cases) - safety} quality, {safety} safety)")
    ctx.say(f"  hash     {dataset.hash}")
    ctx.say(f"  tags     {sanitize(', '.join(tags)) if tags else '(none)'}")
    return exitcodes.OK


def cmd_init_safety(ctx: Context) -> int:
    path, count = init_safety_suite(ctx.files, ctx.args.out or DEFAULT_SAFETY_PATH, overwrite=ctx.args.force)
    ctx.say(f"wrote {path} ({count} safety cases)")
    ctx.say("adapt the tool names, canary and wording to your pipeline, then: llm-eval validate --dataset " + path)
    return exitcodes.OK


COMMANDS = {
    "run": cmd_run,
    "compare": cmd_compare,
    "gate": cmd_gate,
    "baseline": cmd_baseline,
    "validate": cmd_validate,
    "init-safety": cmd_init_safety,
}


def _error(stderr: TextIO, message: str) -> None:
    stderr.write("llm-eval: error: " + sanitize(message, max_chars=8000) + "\n")


def main(argv: list[str], *, environ: Mapping[str, str], cwd: str, stdout: TextIO, stderr: TextIO) -> int:
    """Run the command line. Returns the process exit code; never raises for anything a user can cause."""
    try:
        args = parse_args(argv)
        if args.version:
            stdout.write(f"llm-eval {__version__}\n")
            return exitcodes.OK
        if args.help or not args.command:
            stdout.write(usage_text(args.command))
            return exitcodes.OK
        root = Path(cwd) / args.root if args.root else Path(cwd)
        try:
            files = ProjectFilesystem(root)
        except OSError:
            raise UsageError(f"--root {args.root!r} is not a directory") from None
        config = _config(files, args.config)
        ctx = Context(args, files, environ, config, stdout, stderr)
        return COMMANDS[args.command](ctx)
    except (UsageError, ValidationError) as error:
        _error(stderr, str(error))
        return exitcodes.USAGE
    except HarnessError as error:
        _error(stderr, str(error))
        return exitcodes.INFRASTRUCTURE
    except KeyboardInterrupt:
        stderr.write("llm-eval: interrupted\n")
        return 130
    except BrokenPipeError:
        return exitcodes.OK
    except Exception as error:  # a bug must never look like a pass
        _error(stderr, f"internal error ({type(error).__name__}); the run could not be completed")
        return exitcodes.INFRASTRUCTURE


def run() -> None:
    """Console entry point (``llm-eval``)."""
    sys.exit(main(sys.argv[1:], environ=dict(os.environ), cwd=os.getcwd(), stdout=sys.stdout, stderr=sys.stderr))
