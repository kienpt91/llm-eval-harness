"""grounding: every citation must point at a document that was actually supplied."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from llm_eval_harness.domain.errors import Problem
from llm_eval_harness.domain.grading.params import Params
from llm_eval_harness.domain.jsonio import JsonValue
from llm_eval_harness.domain.model import GradeContext, Outcome
from llm_eval_harness.domain.text import one_line

_MARKER = re.compile(r"\[(?:doc:)?([A-Za-z0-9._:-]{1,64})\]")


@dataclass(frozen=True, slots=True)
class GroundingConfig:
    min_citations: int
    must_cite: tuple[str, ...]
    allow_retrieved: bool
    text_markers: bool


def parse_grounding(raw: Mapping[str, JsonValue], path: str, problems: list[Problem]) -> GroundingConfig:
    p = Params(raw, path, problems, {"min_citations", "must_cite", "allow_retrieved", "text_markers"})
    return GroundingConfig(
        p.integer("min_citations", default=1, minimum=0),
        p.strings("must_cite", max_len=128),
        p.boolean("allow_retrieved", default=True),
        p.boolean("text_markers", default=False),
    )


def grade_grounding(config: GroundingConfig, ctx: GradeContext) -> Outcome:
    """Passes when each citation names a supplied document and the required citations are present.

    A citation is an id in the provider's ``citations`` list (and, with ``text_markers``, an inline marker
    like ``[doc-12]`` in the answer). It is grounded when it is the id of a document in the case's context
    or, with ``allow_retrieved``, of a document the pipeline reports it retrieved. ``min_citations`` sets how
    many distinct citations an answer needs and ``must_cite`` lists ids that have to be among them.
    Failure modes: this checks that citations exist, not that the cited text supports the claim; a pipeline
    that cites a real but irrelevant document passes. Inline markers can collide with ordinary bracketed text.
    """
    allowed = {d.id for d in ctx.case.context}
    if config.allow_retrieved:
        allowed |= set(ctx.output.retrieved)
    cited = list(dict.fromkeys(ctx.output.citations))
    if config.text_markers:
        cited += [m for m in dict.fromkeys(_MARKER.findall(ctx.output.text)) if m not in cited]
    ungrounded = [c for c in cited if c not in allowed]
    problems: list[str] = []
    if ungrounded:
        shown = ", ".join(repr(one_line(c, max_chars=40)) for c in ungrounded[:5])
        problems.append(f"{len(ungrounded)} citation(s) reference no supplied document: {shown}")
    if len(cited) < config.min_citations:
        problems.append(f"{len(cited)} citation(s), need at least {config.min_citations}")
    absent = [m for m in config.must_cite if m not in cited]
    if absent:
        problems.append(
            "required citation(s) missing: " + ", ".join(repr(one_line(m, max_chars=40)) for m in absent[:5])
        )
    if problems:
        grounded = len(cited) - len(ungrounded)
        return Outcome(False, grounded / len(cited) if cited else 0.0, "; ".join(problems))
    return Outcome(True, 1.0, f"{len(cited)} citation(s), all grounded")
