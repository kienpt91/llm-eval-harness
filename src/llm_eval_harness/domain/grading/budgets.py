"""latency_budget, cost_budget and token_budget: performance as a graded property."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal

from llm_eval_harness.domain.errors import Problem
from llm_eval_harness.domain.grading.params import Params
from llm_eval_harness.domain.jsonio import JsonValue
from llm_eval_harness.domain.model import GradeContext, Outcome


@dataclass(frozen=True, slots=True)
class LatencyConfig:
    max_ms: float


def parse_latency_budget(raw: Mapping[str, JsonValue], path: str, problems: list[Problem]) -> LatencyConfig:
    p = Params(raw, path, problems, {"max_ms"})
    return LatencyConfig(p.number("max_ms", required=True, minimum=0.0) or 0.0)


def grade_latency_budget(config: LatencyConfig, ctx: GradeContext) -> Outcome:
    """Passes when the provider call took at most ``max_ms`` milliseconds.

    Failure modes: latency is measured around the provider call on the machine running the harness, so it is
    noisy (a busy CI runner, a cold start) and is the only grader whose result depends on the environment. Set
    budgets with headroom, use ``--repeat`` to see the spread, and do not put it in a ``must``.
    """
    took = ctx.output.latency_ms
    if took <= config.max_ms:
        return Outcome(True, 1.0, f"{took:.0f} ms within {config.max_ms:g} ms")
    return Outcome(False, 0.0, f"{took:.0f} ms exceeds the budget of {config.max_ms:g} ms")


@dataclass(frozen=True, slots=True)
class CostConfig:
    max_usd: Decimal


def parse_cost_budget(raw: Mapping[str, JsonValue], path: str, problems: list[Problem]) -> CostConfig:
    p = Params(raw, path, problems, {"max_usd"})
    return CostConfig(Decimal(str(p.number("max_usd", required=True, minimum=0.0) or 0.0)))


def grade_cost_budget(config: CostConfig, ctx: GradeContext) -> Outcome:
    """Passes when the run cost at most ``max_usd`` according to the price table.

    Failure modes: the cost is computed from the token counts the provider reports and the user's price
    table. When either is missing (no usage, a model that is not in the table) the cost is unknown and the
    grader FAILS rather than guess: an unverifiable budget is not a met budget.
    """
    if ctx.cost_usd is None:
        return Outcome(False, 0.0, "cost is unknown (no usage reported, or the model is not in the price table)")
    if ctx.cost_usd <= config.max_usd:
        return Outcome(True, 1.0, f"cost {ctx.cost_usd:f} within {config.max_usd:f}")
    return Outcome(False, 0.0, f"cost {ctx.cost_usd:f} exceeds the budget of {config.max_usd:f}")


@dataclass(frozen=True, slots=True)
class TokenConfig:
    max_total: int


def parse_token_budget(raw: Mapping[str, JsonValue], path: str, problems: list[Problem]) -> TokenConfig:
    p = Params(raw, path, problems, {"max_total"})
    if "max_total" not in raw:
        p.fail("max_total", "is required")
    return TokenConfig(p.integer("max_total", default=0, minimum=0))


def grade_token_budget(config: TokenConfig, ctx: GradeContext) -> Outcome:
    """Passes when input plus output tokens, as reported by the provider, are at most ``max_total``.

    Failure modes: the counts are whatever the provider reports; a provider that reports none fails the
    budget (unknown is not within budget) and one that under-reports passes it.
    """
    usage = ctx.output.usage
    if usage is None:
        return Outcome(False, 0.0, "no usage reported by the provider")
    if usage.total_tokens <= config.max_total:
        return Outcome(True, 1.0, f"{usage.total_tokens} tokens within {config.max_total}")
    return Outcome(False, 0.0, f"{usage.total_tokens} tokens exceeds the budget of {config.max_total}")
