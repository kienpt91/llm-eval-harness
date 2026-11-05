"""set_overlap and token_f1: overlap graders that need no embeddings."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

from llm_eval_harness.domain.errors import Problem
from llm_eval_harness.domain.grading.params import Params
from llm_eval_harness.domain.jsonio import JsonValue
from llm_eval_harness.domain.model import GradeContext, Outcome
from llm_eval_harness.domain.text import fold

_TOKEN = re.compile(r"\w+")
MAX_TOKENS = 200_000


def tokens(text: str) -> list[str]:
    """Lower-cased word tokens (letters, digits, underscore), at most ``MAX_TOKENS``."""
    out: list[str] = []
    for match in _TOKEN.finditer(fold(text)):
        out.append(match.group())
        if len(out) >= MAX_TOKENS:
            break
    return out


# --- set_overlap ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SetOverlapConfig:
    expected: tuple[str, ...]
    min_recall: float
    min_precision: float
    extract: str
    case_sensitive: bool


def parse_set_overlap(raw: Mapping[str, JsonValue], path: str, problems: list[Problem]) -> SetOverlapConfig:
    p = Params(raw, path, problems, {"expected", "min_recall", "min_precision", "extract", "case_sensitive"})
    return SetOverlapConfig(
        p.strings("expected", required=True, min_items=1),
        p.number("min_recall", default=1.0, minimum=0.0, maximum=1.0) or 0.0,
        p.number("min_precision", default=0.0, minimum=0.0, maximum=1.0) or 0.0,
        p.choice("extract", ("auto", "lines", "commas", "structured"), default="auto"),
        p.boolean("case_sensitive", default=False),
    )


def _items(config: SetOverlapConfig, ctx: GradeContext) -> set[str] | None:
    out = ctx.output
    mode = config.extract
    if mode == "structured" or (mode == "auto" and out.has_structured and isinstance(out.structured, list)):
        if not isinstance(out.structured, list):
            return None
        raw = [x if isinstance(x, str) else str(x) for x in out.structured]
    elif mode == "commas":
        raw = out.text.split(",")
    else:
        raw = out.text.splitlines()
    cleaned = (item.strip().lstrip("-*•0123456789.) ").strip() for item in raw)
    normalise = (lambda s: s) if config.case_sensitive else fold
    return {normalise(c) for c in cleaned if c}


def grade_set_overlap(config: SetOverlapConfig, ctx: GradeContext) -> Outcome:
    """Compares the set of items in the answer with the expected set; the score is the Jaccard index.

    Passes when recall (expected items found) is at least ``min_recall`` and precision (answer items that
    were expected) is at least ``min_precision``. Items come from structured output, lines or commas; list
    markers and numbering are stripped. Failure modes: items must match as whole normalised strings, so
    "Paris, France" does not match "Paris"; duplicates collapse; order is ignored.
    """
    got = _items(config, ctx)
    if got is None:
        return Outcome(False, 0.0, "provider returned no structured list")
    normalise = (lambda s: s) if config.case_sensitive else fold
    want = {normalise(e.strip()) for e in config.expected}
    hit = got & want
    recall = len(hit) / len(want)
    precision = len(hit) / len(got) if got else 0.0
    union = len(got | want)
    jaccard = len(hit) / union if union else 1.0
    ok = recall >= config.min_recall and precision >= config.min_precision
    return Outcome(
        ok,
        jaccard,
        f"recall {recall:.2f} (need {config.min_recall:.2f}), precision {precision:.2f} (need {config.min_precision:.2f})",
    )


# --- token_f1 ------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TokenF1Config:
    references: tuple[str, ...]
    min_f1: float


def parse_token_f1(raw: Mapping[str, JsonValue], path: str, problems: list[Problem]) -> TokenF1Config:
    p = Params(raw, path, problems, {"references", "min_f1"})
    return TokenF1Config(
        p.strings("references", required=True, min_items=1),
        p.number("min_f1", default=0.6, minimum=0.0, maximum=1.0) or 0.0,
    )


def f1(answer: list[str], reference: list[str]) -> float:
    """Token-level F1 (harmonic mean of token precision and recall, counting repeats)."""
    if not answer or not reference:
        return 1.0 if not answer and not reference else 0.0
    common = sum((Counter(answer) & Counter(reference)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(answer), common / len(reference)
    return 2 * precision * recall / (precision + recall)


def grade_token_f1(config: TokenF1Config, ctx: GradeContext) -> Outcome:
    """Scores the answer by token F1 against the best of the reference answers; passes at ``min_f1``.

    Embedding-free and deterministic. Failure modes: it rewards shared words, not meaning. A short correct
    answer scores low against a long reference, a long rambling one scores low on precision, a paraphrase
    with different words scores near zero, and a wrong answer that reuses the question's words can score
    well. Use it as a cheap similarity signal, with ``contains`` for the facts that must be there.
    """
    got = tokens(ctx.output.text)
    best = max(f1(got, tokens(ref)) for ref in config.references)
    ok = best >= config.min_f1
    return Outcome(ok, best, f"token F1 {best:.2f} (need {config.min_f1:.2f})")
