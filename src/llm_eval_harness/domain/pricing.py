"""Token prices from a user-supplied table. There are no built-in prices: the table is the source of truth."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from llm_eval_harness.domain.errors import Problem, ValidationError
from llm_eval_harness.domain.grading.params import Params
from llm_eval_harness.domain.jsonio import JsonValue, kind_of
from llm_eval_harness.domain.model import Usage

MILLION = Decimal(1_000_000)
MAX_MODELS = 1000


@dataclass(frozen=True, slots=True)
class ModelPrice:
    input_per_mtok: Decimal
    output_per_mtok: Decimal


@dataclass(frozen=True, slots=True)
class PriceTable:
    currency: str
    models: dict[str, ModelPrice]


EMPTY_TABLE = PriceTable("USD", {})


def parse_price_table(obj: JsonValue, source: str) -> PriceTable:
    """Read ``{"currency": "USD", "models": {"name": {"input_per_mtok": 1, "output_per_mtok": 2}}}``."""
    problems: list[Problem] = []
    if not isinstance(obj, dict):
        raise ValidationError(source, [Problem("", f"must be a JSON object (got {kind_of(obj)})")])
    p = Params(obj, "", problems, ("currency", "models"))
    currency = p.string("currency", default="USD", max_len=8) or "USD"
    raw_models = p.obj("models", required=True) or {}
    if len(raw_models) > MAX_MODELS:
        problems.append(Problem("models", f"has too many models (limit {MAX_MODELS})"))
    models: dict[str, ModelPrice] = {}
    for name, entry in raw_models.items():
        where = f"models.{name}"
        if not isinstance(entry, dict):
            problems.append(Problem(where, f"must be an object (got {kind_of(entry)})"))
            continue
        mp = Params(entry, where, problems, ("input_per_mtok", "output_per_mtok"))
        price_in = mp.number("input_per_mtok", required=True, minimum=0.0)
        price_out = mp.number("output_per_mtok", required=True, minimum=0.0)
        if price_in is not None and price_out is not None:
            models[name] = ModelPrice(Decimal(repr(price_in)), Decimal(repr(price_out)))
    if problems:
        raise ValidationError(source, problems)
    return PriceTable(currency, models)


def cost_of(usage: Usage | None, table: PriceTable) -> Decimal | None:
    """Cost of a call in the table's currency, or ``None`` when usage or the model's price is missing."""
    if usage is None or usage.model is None:
        return None
    price = table.models.get(usage.model)
    if price is None:
        return None
    return (usage.input_tokens * price.input_per_mtok + usage.output_tokens * price.output_per_mtok) / MILLION
