"""Command line parsing (standard library ``argparse``). Every problem is a ``UsageError`` (exit code 2)."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import NoReturn

from llm_eval_harness.domain.errors import UsageError
from llm_eval_harness.infrastructure.cli.help import HELP, command_help

COMMANDS = ("run", "compare", "gate", "baseline", "validate", "init-safety")


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise UsageError(message)

    def exit(self, status: int = 0, message: str | None = None) -> NoReturn:  # noqa: ARG002
        raise UsageError(message or "invalid command line")


@dataclass(frozen=True, slots=True)
class CliArgs:
    command: str
    help: bool = False
    version: bool = False
    root: str | None = None
    config: str | None = None
    dataset: str | None = None
    repeat: int | None = None
    concurrency: int | None = None
    timeout: float | None = None
    retries: int | None = None
    seed: int | None = None
    sample: int | None = None
    tags: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    case_ids: tuple[str, ...] = ()
    out: str | None = None
    no_redact: bool = False
    price_table: str | None = None
    write_baseline: str | None = None
    force: bool = False
    report: str | None = None
    baseline: str | None = None
    tolerance: float | None = None
    alpha: float | None = None
    deterministic: bool = False
    max_cost_increase_pct: float | None = None
    max_p95_latency_increase_pct: float | None = None
    max_regressed_cases: int | None = None
    min_pass_rate: float | None = None
    on_inconclusive: str | None = None
    fail_on_removed: bool = False
    markdown: str | None = None

    def has_run_options(self) -> bool:
        return any(
            value is not None
            for value in (
                self.dataset,
                self.repeat,
                self.concurrency,
                self.timeout,
                self.retries,
                self.seed,
                self.sample,
                self.price_table,
            )
        ) or bool(self.tags or self.categories or self.case_ids or self.no_redact)


def _whole(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from None
    return value


def _number(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError("must be finite")
    return value


def _build() -> argparse.ArgumentParser:
    parser = _Parser(prog="llm-eval", add_help=False)
    parser.add_argument("-h", "--help", action="store_true")
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--root", metavar="DIR")
    parser.add_argument("--config", metavar="FILE")
    sub = parser.add_subparsers(dest="command", parser_class=_Parser)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("-h", "--help", action="store_true")
        p.add_argument("--root", metavar="DIR", default=argparse.SUPPRESS)
        p.add_argument("--config", metavar="FILE", default=argparse.SUPPRESS)

    def run_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--dataset")
        p.add_argument("--repeat", type=_whole)
        p.add_argument("--concurrency", type=_whole)
        p.add_argument("--timeout", type=_number)
        p.add_argument("--retries", type=_whole)
        p.add_argument("--seed", type=_whole)
        p.add_argument("--sample", type=_whole)
        p.add_argument("--tag", dest="tags", action="append", default=[])
        p.add_argument("--category", dest="categories", action="append", default=[])
        p.add_argument("--case", dest="case_ids", action="append", default=[])
        p.add_argument("--out")
        p.add_argument("--no-redact", action="store_true")
        p.add_argument("--price-table")

    def compare_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--baseline")
        p.add_argument("--tolerance", type=_number)
        p.add_argument("--alpha", type=_number)
        p.add_argument("--deterministic", action="store_true")
        p.add_argument("--max-cost-increase-pct", type=_number)
        p.add_argument("--max-p95-latency-increase-pct", type=_number)
        p.add_argument("--max-regressed-cases", type=_whole)
        p.add_argument("--markdown")

    for name in COMMANDS:
        p = sub.add_parser(name, add_help=False)
        common(p)
        if name == "run":
            run_options(p)
            p.add_argument("--write-baseline")
            p.add_argument("--force", action="store_true")
        elif name == "compare":
            compare_options(p)
            p.add_argument("--report")
            p.add_argument("--out")
        elif name == "gate":
            run_options(p)
            compare_options(p)
            p.add_argument("--report")
            p.add_argument("--min-pass-rate", type=_number)
            p.add_argument("--on-inconclusive", choices=("fail", "pass"))
            p.add_argument("--fail-on-removed", action="store_true")
        elif name == "baseline":
            p.add_argument("--report")
            p.add_argument("--out")
            p.add_argument("--force", action="store_true")
        elif name == "validate":
            p.add_argument("--dataset")
        elif name == "init-safety":
            p.add_argument("--out")
            p.add_argument("--force", action="store_true")
    return parser


def parse_args(argv: list[str]) -> CliArgs:
    """Parse the command line. ``llm-eval --help`` and ``llm-eval COMMAND --help`` are recognised."""
    namespace = _build().parse_args(argv)
    values = vars(namespace)
    if values.get("command") is None:
        if values.get("help") or values.get("version"):
            return CliArgs(command="", help=bool(values.get("help")), version=bool(values.get("version")))
        raise UsageError("no command given (try: llm-eval --help)")
    kwargs: dict[str, object] = {k: v for k, v in values.items() if k in CliArgs.__dataclass_fields__}
    for key in ("tags", "categories", "case_ids"):
        kwargs[key] = tuple(values.get(key) or ())
    args = CliArgs(**kwargs)  # type: ignore[arg-type]
    if args.sample is not None and args.sample < 1:
        raise UsageError("--sample must be at least 1")
    for name, value in (("--repeat", args.repeat), ("--concurrency", args.concurrency), ("--timeout", args.timeout)):
        if value is not None and value <= 0:
            raise UsageError(f"{name} must be positive")
    if args.retries is not None and args.retries < 0:
        raise UsageError("--retries must not be negative")
    return args


def usage_text(command: str) -> str:
    return command_help(command) if command else HELP
