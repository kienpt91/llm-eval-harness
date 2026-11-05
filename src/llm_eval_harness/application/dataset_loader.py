"""Reading a dataset file (JSON Lines, or one JSON document) into a validated ``Dataset``."""

from __future__ import annotations

from llm_eval_harness.application.ports import ProjectFiles
from llm_eval_harness.domain.dataset import MAX_PROBLEMS, parse_dataset
from llm_eval_harness.domain.errors import DatasetError, Problem
from llm_eval_harness.domain.jsonio import JsonParseError, JsonValue, kind_of, parse_json
from llm_eval_harness.domain.limits import MAX_CASE_BYTES, MAX_DATASET_BYTES
from llm_eval_harness.domain.model import Dataset


def load_dataset(files: ProjectFiles, path: str) -> Dataset:
    """``.jsonl``: one case per line (blank lines allowed). ``.json``: an array of cases or ``{"cases": [...]}``."""
    text = files.read_text(path, MAX_DATASET_BYTES).removeprefix("\ufeff")
    if path.endswith(".jsonl"):
        return parse_dataset(_jsonl_records(text, path), path)
    if path.endswith(".json"):
        return parse_dataset(_json_records(text, path), path)
    raise DatasetError(path, [Problem("", "unsupported file type: datasets are .jsonl or .json files")])


def _jsonl_records(text: str, source: str) -> list[tuple[str, JsonValue]]:
    records: list[tuple[str, JsonValue]] = []
    problems: list[Problem] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        label = f"line {number}"
        if len(line) > MAX_CASE_BYTES:
            problems.append(Problem(label, f"line is {len(line)} characters, the limit is {MAX_CASE_BYTES}"))
        else:
            try:
                records.append((label, parse_json(line)))
            except JsonParseError as error:
                problems.append(Problem(f"line {number}, column {error.column}", f"invalid JSON: {error.message}"))
        if len(problems) >= MAX_PROBLEMS:
            problems.append(Problem("", f"stopped after {MAX_PROBLEMS} problems"))
            break
    if problems:
        raise DatasetError(source, problems)
    return records


def _json_records(text: str, source: str) -> list[tuple[str, JsonValue]]:
    try:
        document = parse_json(text)
    except JsonParseError as error:
        raise DatasetError(
            source, [Problem(f"line {error.line}, column {error.column}", f"invalid JSON: {error.message}")]
        ) from None
    items = document.get("cases") if isinstance(document, dict) else document
    if not isinstance(items, list):
        found = kind_of(items) if isinstance(document, dict) else kind_of(document)
        raise DatasetError(
            source, [Problem("", f'must be an array of cases or an object with a "cases" array (got {found})')]
        )
    return [(f"cases[{i}]", item) for i, item in enumerate(items)]
