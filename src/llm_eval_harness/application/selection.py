"""Choosing which cases of a dataset to run: filters and seeded sampling."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from llm_eval_harness.domain.errors import UsageError
from llm_eval_harness.domain.model import CATEGORIES, Case
from llm_eval_harness.domain.prng import sample


@dataclass(frozen=True, slots=True)
class Selection:
    tags: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    case_ids: tuple[str, ...] = ()
    sample: int | None = None
    seed: int | None = None


def select_cases(cases: Sequence[Case], selection: Selection) -> list[Case]:
    """Apply the filters (a case must match every non-empty filter; tags match if any given tag is present),
    then take a seeded sample of at most ``sample`` of what is left. Order is by case id, always."""
    for category in selection.categories:
        if category not in CATEGORIES:
            raise UsageError(f"unknown category {category!r} (known: {', '.join(CATEGORIES)})")
    known = {c.id for c in cases}
    missing = [i for i in selection.case_ids if i not in known]
    if missing:
        raise UsageError(f"unknown case id(s): {', '.join(missing[:5])}")
    chosen = [
        c
        for c in sorted(cases, key=lambda c: c.id)
        if (not selection.tags or any(t in c.tags for t in selection.tags))
        and (not selection.categories or c.category in selection.categories)
        and (not selection.case_ids or c.id in selection.case_ids)
    ]
    if selection.sample is not None:
        if selection.sample < 1:
            raise UsageError("--sample must be at least 1")
        chosen = sample(chosen, selection.sample, selection.seed if selection.seed is not None else 0)
    if not chosen:
        raise UsageError("no case matches the filters")
    return chosen
