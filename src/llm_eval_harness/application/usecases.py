"""The use cases behind the commands. They take ports and plain values and return values; they never print or exit."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from llm_eval_harness.application.config import Config, parse_config
from llm_eval_harness.application.dataset_loader import load_dataset
from llm_eval_harness.application.ports import Clock, JudgeRunner, ProjectFiles, Provider
from llm_eval_harness.application.redaction import Redactor
from llm_eval_harness.application.report.json_files import json_text
from llm_eval_harness.application.report.markdown import render_markdown
from llm_eval_harness.application.runner import RunOptions, run_cases
from llm_eval_harness.application.selection import Selection, select_cases
from llm_eval_harness.domain.baseline import Snapshot, baseline_text, parse_baseline, snapshot_of_report
from llm_eval_harness.domain.compare import CompareOptions, Comparison, compare
from llm_eval_harness.domain.errors import Problem, UsageError, ValidationError
from llm_eval_harness.domain.jsonio import JsonParseError, JsonValue, parse_json
from llm_eval_harness.domain.limits import MAX_CONFIG_BYTES, MAX_PRICE_TABLE_BYTES, MAX_REPORT_BYTES
from llm_eval_harness.domain.model import Dataset
from llm_eval_harness.domain.pricing import EMPTY_TABLE, PriceTable, parse_price_table
from llm_eval_harness.domain.report import ReportMeta, RunSettings, build_report
from llm_eval_harness.domain.reportdoc import ReportDoc, parse_report
from llm_eval_harness.domain.safety_suite import starter_dataset_text

REPORT_FILE = "report.json"
MARKDOWN_FILE = "report.md"


def read_json(files: ProjectFiles, path: str, max_bytes: int) -> JsonValue:
    text = files.read_text(path, max_bytes).removeprefix("\ufeff")
    try:
        return parse_json(text)
    except JsonParseError as error:
        raise ValidationError(
            path, [Problem(f"line {error.line}, column {error.column}", f"invalid JSON: {error.message}")]
        ) from None


def load_config(files: ProjectFiles, path: str) -> Config:
    return parse_config(read_json(files, path, MAX_CONFIG_BYTES), path)


def load_prices(files: ProjectFiles, path: str | None) -> PriceTable:
    if path is None:
        return EMPTY_TABLE
    return parse_price_table(read_json(files, path, MAX_PRICE_TABLE_BYTES), path)


def read_report(files: ProjectFiles, path: str) -> ReportDoc:
    return parse_report(read_json(files, path, MAX_REPORT_BYTES), path)


def read_baseline(files: ProjectFiles, path: str) -> Snapshot:
    return parse_baseline(read_json(files, path, MAX_REPORT_BYTES), path)


@dataclass(frozen=True, slots=True)
class RunRequest:
    """Everything one evaluation run needs besides the ports."""

    dataset_path: str
    selection: Selection
    repeat: int
    concurrency: int
    timeout_s: float
    retries: int
    seed: int | None
    out_dir: str
    version: str


@dataclass(frozen=True, slots=True)
class RunOutcome:
    doc: ReportDoc
    written: tuple[str, ...]


def validate_dataset(files: ProjectFiles, path: str) -> Dataset:
    return load_dataset(files, path)


def run_evaluation(
    files: ProjectFiles,
    config: Config,
    provider: Provider,
    judge: JudgeRunner | None,
    clock: Clock,
    request: RunRequest,
    prices: PriceTable,
    redactor: Redactor,
) -> RunOutcome:
    """Load, select, run, build the report, write ``report.json`` and ``report.md`` into the output directory."""
    dataset = load_dataset(files, request.dataset_path)
    selected = select_cases(dataset.cases, request.selection)
    started = clock.now_iso()
    options = RunOptions(
        repeat=request.repeat,
        concurrency=request.concurrency,
        timeout_s=request.timeout_s,
        retries=request.retries,
        seed=request.seed,
        default_graders=config.graders,
        prices=prices,
    )
    results = run_cases(selected, provider, judge, redactor, clock, options)
    run_ids = {c.id for c in selected}
    safety_not_run = tuple(c.id for c in dataset.cases if c.is_safety and c.id not in run_ids)
    meta = ReportMeta(
        tool_version=request.version,
        started_at=started,
        finished_at=clock.now_iso(),
        dataset_source=request.dataset_path,
        dataset_hash=dataset.hash,
        total_cases=len(dataset.cases),
        provider=provider.describe(),
        currency=prices.currency,
        settings=RunSettings(
            repeat=request.repeat,
            seed=request.seed,
            sample=request.selection.sample,
            tags=request.selection.tags,
            categories=request.selection.categories,
            concurrency=request.concurrency,
            retries=request.retries,
            redacted=redactor.enabled,
        ),
        safety_not_run=safety_not_run,
    )
    document = build_report(meta, results)
    doc = parse_report(document, REPORT_FILE)  # what is written is what a later `compare` or `gate` reads back
    base = request.out_dir.rstrip("/")
    written = (
        files.write_text(f"{base}/{REPORT_FILE}", json_text(document), overwrite=True),
        files.write_text(f"{base}/{MARKDOWN_FILE}", render_markdown(doc), overwrite=True),
    )
    return RunOutcome(doc, written)


def compare_with_baseline(baseline: Snapshot, doc: ReportDoc, options: CompareOptions) -> Comparison:
    return compare(baseline, snapshot_of_report(doc), options)


def write_baseline(files: ProjectFiles, doc: ReportDoc, path: str, *, overwrite: bool) -> str:
    return files.write_text(path, baseline_text(doc), overwrite=overwrite)


def init_safety_suite(files: ProjectFiles, path: str, *, overwrite: bool) -> tuple[str, int]:
    """Write the starter safety dataset; returns the path written and the number of cases."""
    text = starter_dataset_text()
    written = files.write_text(path, text, overwrite=overwrite)
    return written, len(text.splitlines())


def require(value: str | None, what: str) -> str:
    if value is None:
        raise UsageError(f"{what} is required (give it on the command line or in the configuration file)")
    return value


def split_values(values: Sequence[str]) -> tuple[str, ...]:
    """Repeated options may also carry comma separated lists: ``--tag a,b`` equals ``--tag a --tag b``."""
    return tuple(part for value in values for part in value.split(",") if part)
