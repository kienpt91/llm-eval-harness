"""Dataset validation: from parsed JSON records to ``Case`` objects, with precise errors, and dataset hashing.

The format is documented in docs/dataset-format.md. Every problem found is reported (up to a cap) with the
record it belongs to and the path inside it, so a dataset can be fixed in one pass.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from llm_eval_harness.domain.errors import DatasetError, Problem
from llm_eval_harness.domain.grading.params import Params
from llm_eval_harness.domain.grading.registry import GRADERS, validate_params
from llm_eval_harness.domain.grading.toolcalls import parse_expectations
from llm_eval_harness.domain.identifiers import ID_PATTERN, TAG_PATTERN
from llm_eval_harness.domain.jsonio import JsonValue, canonical, kind_of
from llm_eval_harness.domain.limits import (
    MAX_CASE_BYTES,
    MAX_CASES,
    MAX_CONTEXT_DOCUMENTS,
    MAX_GRADERS,
    MAX_ID_LENGTH,
    MAX_TAG_LENGTH,
    MAX_TAGS,
)
from llm_eval_harness.domain.model import CATEGORIES, Case, ContextDocument, Dataset, Expected, GraderSpec

MAX_PROBLEMS = 50
CASE_KEYS = ("id", "category", "tags", "input", "context", "expected", "graders", "threshold", "metadata")
EXPECTED_KEYS = (
    "answers",
    "facts",
    "forbidden",
    "forbidden_patterns",
    "tool_calls",
    "forbidden_tools",
    "tool_order",
    "allow_extra_tools",
)
GRADER_RESERVED = ("type", "weight", "must", "name")


def _digest(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def case_hash(record: JsonValue) -> str:
    """Hash of one case as written (canonical JSON), so any change to it changes the hash."""
    return _digest(canonical(record))


def dataset_hash(cases: Sequence[Case]) -> str:
    """Hash of a dataset: independent of record order, sensitive to every case."""
    lines = sorted(f"{c.id}\0{c.hash}" for c in cases)
    return _digest("\n".join(lines))


def parse_dataset(records: Sequence[tuple[str, JsonValue]], source: str) -> Dataset:
    """Validate records (each with a label such as ``line 3``) and build the dataset or raise ``DatasetError``."""
    problems: list[Problem] = []
    cases: list[Case] = []
    seen: dict[str, str] = {}
    if len(records) > MAX_CASES:
        raise DatasetError(source, [Problem("", f"has {len(records)} cases, the limit is {MAX_CASES}")])
    if not records:
        raise DatasetError(source, [Problem("", "contains no cases")])
    for label, record in records:
        found: list[Problem] = []
        case = _parse_case(record, label, found)
        problems.extend(found)
        if case is not None and not found:
            if case.id in seen:
                problems.append(Problem(label, f"duplicate case id {case.id!r} (first used at {seen[case.id]})"))
            else:
                seen[case.id] = label
                cases.append(case)
        if len(problems) >= MAX_PROBLEMS:
            problems.append(Problem("", f"stopped after {MAX_PROBLEMS} problems"))
            break
    if problems:
        raise DatasetError(source, problems[: MAX_PROBLEMS + 1])
    ordered = tuple(sorted(cases, key=lambda c: c.id))
    return Dataset(ordered, dataset_hash(ordered))


def _parse_case(record: JsonValue, label: str, problems: list[Problem]) -> Case | None:
    start = len(problems)
    if not isinstance(record, dict):
        problems.append(Problem(label, f"each case must be a JSON object (got {kind_of(record)})"))
        return None
    size = len(canonical(record))
    if size > MAX_CASE_BYTES:
        problems.append(Problem(label, f"case is {size} bytes, the limit is {MAX_CASE_BYTES}"))
        return None
    local: list[Problem] = []
    p = Params(record, label, local, CASE_KEYS)
    case_id = p.string("id", required=True, max_len=MAX_ID_LENGTH)
    if case_id is not None and not ID_PATTERN.fullmatch(case_id):
        p.fail("id", "may use letters, digits and . _ : / - only, and must start with a letter or digit")
        case_id = None
    category = p.choice("category", CATEGORIES, default="quality")
    tags = _tags(p)
    if "input" not in record:
        p.fail("input", "is required")
    context = _context(p)
    expected = _expected(p, label, local)
    graders = _graders(p, label, local)
    threshold = p.number("threshold", default=1.0, minimum=0.0, maximum=1.0)
    metadata = p.obj("metadata") or {}
    if category == "safety":
        _safety_rules(p, record, graders)
    if expected is not None and not _has_any_expectation(expected) and not graders and not local:
        p.fail("", "has nothing to grade: add expected fields or graders")
    problems.extend(local)
    if len(problems) > start or case_id is None or expected is None:
        return None
    return Case(
        id=case_id,
        category=category,
        tags=tags,
        input=record["input"],
        context=context,
        expected=expected,
        graders=graders,
        threshold=1.0 if threshold is None else threshold,
        metadata=metadata,
        hash=case_hash(record),
    )


def _safety_rules(p: Params, record: dict[str, JsonValue], graders: tuple[GraderSpec, ...]) -> None:
    """A safety case is all or nothing: no lowered threshold, no grader that could never make it fail."""
    if "threshold" in record and record["threshold"] != 1:
        p.fail("threshold", "a safety case cannot lower its threshold: it passes only when everything it checks passes")
    for index, spec in enumerate(graders):
        if spec.weight == 0 and not spec.must:
            p.fail(
                f"graders[{index}]",
                "has weight 0 and is not a must, so it could never make a safety case fail: set must: true or a weight",
            )


def _tags(p: Params) -> tuple[str, ...]:
    tags = p.strings("tags", max_len=MAX_TAG_LENGTH)
    if len(tags) > MAX_TAGS:
        p.fail("tags", f"has too many tags ({len(tags)}, limit {MAX_TAGS})")
    for index, tag in enumerate(tags):
        if not TAG_PATTERN.fullmatch(tag):
            p.fail(f"tags[{index}]", "may use letters, digits and . _ : - only, and must start with a letter or digit")
    if len(set(tags)) != len(tags):
        p.fail("tags", "has duplicates")
    return tags


def _context(p: Params) -> tuple[ContextDocument, ...]:
    items = p.array("context")
    if items is None:
        return ()
    if len(items) > MAX_CONTEXT_DOCUMENTS:
        p.fail("context", f"has too many documents ({len(items)}, limit {MAX_CONTEXT_DOCUMENTS})")
        return ()
    docs: list[ContextDocument] = []
    ids: set[str] = set()
    for index, item in enumerate(items):
        where = f"context[{index}]"
        if not isinstance(item, dict):
            p.fail(where, f"must be an object with id and text (got {kind_of(item)})")
            continue
        dp = Params(item, f"{p.path}.{where}", p.problems, ("id", "text", "metadata"))
        doc_id = dp.string("id", required=True, max_len=MAX_ID_LENGTH)
        text = dp.string("text", required=True, non_empty=False)
        meta = dp.obj("metadata") or {}
        if doc_id is not None and not ID_PATTERN.fullmatch(doc_id):
            dp.fail("id", "may use letters, digits and . _ : / - only")
        elif doc_id is not None and doc_id in ids:
            dp.fail("id", f"duplicate document id {doc_id!r}")
        elif doc_id is not None and text is not None:
            ids.add(doc_id)
            docs.append(ContextDocument(doc_id, text, meta))
    return tuple(docs)


def _expected(p: Params, label: str, problems: list[Problem]) -> Expected | None:
    raw = p.obj("expected")
    if raw is None:
        return Expected()
    ep = Params(raw, f"{label}.expected", problems, EXPECTED_KEYS)
    patterns = ep.strings("forbidden_patterns", max_len=512)
    # reuse the pattern screen of the not_contains grader so the messages are the same everywhere
    if patterns:
        for sub in validate_params(
            "not_contains", {"patterns": list(patterns)}, f"{label}.expected.forbidden_patterns"
        ):
            problems.append(
                Problem(sub.location.replace(".forbidden_patterns.patterns", ".forbidden_patterns"), sub.message)
            )
    tool_calls = None
    items = ep.array("tool_calls")
    if items is not None:
        tool_calls = parse_expectations(items, f"{label}.expected.tool_calls", problems)
    return Expected(
        answers=ep.strings("answers"),
        facts=ep.strings("facts"),
        forbidden=ep.strings("forbidden"),
        forbidden_patterns=patterns,
        tool_calls=tool_calls,
        forbidden_tools=ep.strings("forbidden_tools", max_len=200),
        tool_order=ep.choice("tool_order", ("any", "ordered"), default="any"),
        allow_extra_tools=ep.boolean("allow_extra_tools", default=True),
    )


def _has_any_expectation(e: Expected) -> bool:
    return bool(
        e.answers or e.facts or e.forbidden or e.forbidden_patterns or e.tool_calls is not None or e.forbidden_tools
    )


def _graders(p: Params, label: str, problems: list[Problem]) -> tuple[GraderSpec, ...]:
    items = p.array("graders")
    if items is None:
        return ()
    if len(items) > MAX_GRADERS:
        p.fail("graders", f"has too many graders ({len(items)}, limit {MAX_GRADERS})")
        return ()
    specs: list[GraderSpec] = []
    for index, item in enumerate(items):
        where = f"{label}.graders[{index}]"
        spec = parse_grader_spec(item, where, problems)
        if spec is not None:
            specs.append(spec)
    return tuple(specs)


def parse_grader_spec(item: JsonValue, where: str, problems: list[Problem]) -> GraderSpec | None:
    """One entry of a ``graders`` list: ``{"type": ..., "weight": ..., "must": ..., ...parameters}``."""
    if not isinstance(item, dict):
        problems.append(Problem(where, f"must be an object with a type (got {kind_of(item)})"))
        return None
    grader_type = item.get("type")
    if not isinstance(grader_type, str):
        problems.append(Problem(f"{where}.type", "is required and must be a string"))
        return None
    if grader_type not in GRADERS:
        problems.append(
            Problem(f"{where}.type", f"unknown grader {grader_type!r} (known: {', '.join(sorted(GRADERS))})")
        )
        return None
    meta = Params({k: v for k, v in item.items() if k in GRADER_RESERVED}, where, problems, GRADER_RESERVED)
    weight = meta.number("weight", default=1.0, minimum=0.0, maximum=1000.0)
    must = meta.boolean("must", default=False)
    name = meta.string("name", default="", max_len=64) or ""
    params = {k: v for k, v in item.items() if k not in GRADER_RESERVED}
    before = len(problems)
    problems.extend(validate_params(grader_type, params, where))
    if len(problems) > before:
        return None
    return GraderSpec(grader_type, params, 1.0 if weight is None else weight, must, name)
