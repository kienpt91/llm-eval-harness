"""The catalogue of grader types: how each one is validated and evaluated."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from llm_eval_harness.domain.errors import Problem
from llm_eval_harness.domain.grading import (
    budgets,
    grounding,
    judge,
    leaks,
    numeric,
    sets,
    structured,
    text_graders,
    toolcalls,
)
from llm_eval_harness.domain.jsonio import JsonValue
from llm_eval_harness.domain.model import GradeContext, GraderSpec, Outcome

Parser = Callable[[Mapping[str, JsonValue], str, list[Problem]], Any]
Grader = Callable[[Any, GradeContext], Outcome]


@dataclass(frozen=True, slots=True)
class GraderDef:
    parse: Parser
    grade: Grader | None  # None: evaluated through a port (judge)
    deterministic: bool = True


GRADERS: dict[str, GraderDef] = {
    "exact": GraderDef(text_graders.parse_exact, text_graders.grade_exact),
    "contains": GraderDef(text_graders.parse_contains, text_graders.grade_contains),
    "not_contains": GraderDef(text_graders.parse_not_contains, text_graders.grade_not_contains),
    "regex": GraderDef(text_graders.parse_regex, text_graders.grade_regex),
    "refusal": GraderDef(text_graders.parse_refusal, text_graders.grade_refusal),
    "json_schema": GraderDef(structured.parse_json_schema, structured.grade_json_schema),
    "numeric": GraderDef(numeric.parse_numeric, numeric.grade_numeric),
    "set_overlap": GraderDef(sets.parse_set_overlap, sets.grade_set_overlap),
    "token_f1": GraderDef(sets.parse_token_f1, sets.grade_token_f1),
    "tool_calls": GraderDef(toolcalls.parse_tool_calls, toolcalls.grade_tool_calls),
    "forbidden_tools": GraderDef(toolcalls.parse_forbidden_tools, toolcalls.grade_forbidden_tools),
    "grounding": GraderDef(grounding.parse_grounding, grounding.grade_grounding),
    "secret_leak": GraderDef(leaks.parse_secret_leak, leaks.grade_secret_leak),
    "pii_leak": GraderDef(leaks.parse_pii_leak, leaks.grade_pii_leak),
    "latency_budget": GraderDef(budgets.parse_latency_budget, budgets.grade_latency_budget),
    "cost_budget": GraderDef(budgets.parse_cost_budget, budgets.grade_cost_budget),
    "token_budget": GraderDef(budgets.parse_token_budget, budgets.grade_token_budget),
    "judge": GraderDef(judge.parse_judge, None, deterministic=False),
}


def validate_params(grader_type: str, params: Mapping[str, JsonValue], path: str) -> list[Problem]:
    """Every problem with the parameters of one grader of a known type."""
    problems: list[Problem] = []
    GRADERS[grader_type].parse(params, path, problems)
    return problems


def parsed_config(spec: GraderSpec) -> Any:
    """The typed configuration of a grader that already passed validation."""
    problems: list[Problem] = []
    config = GRADERS[spec.type].parse(spec.params, spec.type, problems)
    if problems:  # only reachable when a spec bypassed the dataset validator
        raise ValueError("; ".join(p.render() for p in problems))
    return config


def evaluate(spec: GraderSpec, ctx: GradeContext) -> Outcome:
    """Run one deterministic grader. Judge graders are run by the application through its port."""
    definition = GRADERS[spec.type]
    if definition.grade is None:
        raise ValueError(f"grader type {spec.type!r} is evaluated through a port")
    return definition.grade(parsed_config(spec), ctx)
