from __future__ import annotations

import math
from decimal import Decimal

import pytest

from llm_eval_harness.domain.errors import ValidationError
from llm_eval_harness.domain.model import Expected, GradeResult, ToolCallExpectation, Usage
from llm_eval_harness.domain.plan import combine, plan_graders
from llm_eval_harness.domain.pricing import EMPTY_TABLE, cost_of, parse_price_table
from llm_eval_harness.domain.prng import SplitMix64, sample
from llm_eval_harness.domain.stats import fisher_exact, mean, percentile, variance, wilson_interval
from support.builders import case, js, spec
from support.prng import Prng


def result(
    name: str, passed: bool, score: float | None = None, *, weight: float = 1.0, must: bool = False
) -> GradeResult:
    return GradeResult(name, "x", passed, (1.0 if passed else 0.0) if score is None else score, weight, must, "d")


def test_expected_fields_become_graders_with_the_right_strictness() -> None:
    expected = Expected(
        answers=("a",),
        facts=("f",),
        forbidden=("x",),
        forbidden_patterns=("y+",),
        tool_calls=(ToolCallExpectation("t", {"k": 1}),),
        forbidden_tools=("bad",),
        tool_order="ordered",
        allow_extra_tools=False,
    )
    specs = {s.name: s for s in plan_graders(case(expected=expected))}
    assert set(specs) == {"answer", "facts", "forbidden", "tool_calls", "forbidden_tools"}
    assert specs["forbidden"].must and specs["forbidden_tools"].must
    assert not specs["answer"].must and not specs["facts"].must and not specs["tool_calls"].must
    assert specs["forbidden"].params == {"values": ["x"], "patterns": ["y+"]}
    assert specs["tool_calls"].params == {
        "calls": [{"name": "t", "args": {"k": 1}}],
        "ordered": True,
        "allow_extra": False,
    }


def test_empty_tool_call_expectation_means_no_tools() -> None:
    [only] = plan_graders(case(expected=Expected(tool_calls=(), allow_extra_tools=False)))
    assert only.type == "tool_calls" and only.params["calls"] == [] and only.params["allow_extra"] is False


def test_safety_cases_always_get_a_secret_leak_must() -> None:
    specs = plan_graders(case(category="safety", expected=Expected(facts=("a",))))
    leak = [s for s in specs if s.type == "secret_leak"]
    assert len(leak) == 1 and leak[0].must
    explicit = plan_graders(case(category="safety", graders=(spec("secret_leak", {"kinds": ["jwt"]}),)))
    assert [s.type for s in explicit].count("secret_leak") == 1 and not explicit[0].must
    assert not [s for s in plan_graders(case(expected=Expected(facts=("a",)))) if s.type == "secret_leak"]


def test_names_are_unique_and_defaults_come_last() -> None:
    c = case(
        expected=Expected(facts=("a",)), graders=(spec("regex", {"pattern": "a"}), spec("regex", {"pattern": "b"}))
    )
    names = [s.name for s in plan_graders(c, [spec("latency_budget", {"max_ms": 1})])]
    assert names == ["facts", "regex", "regex#2", "latency_budget"]


def test_combine_weighted_mean_threshold_and_must() -> None:
    v = combine([result("a", True), result("b", False, 0.5, weight=3.0)], 0.5)
    assert v.score == pytest.approx((1 + 1.5) / 4) and v.passed and v.failed_musts == ()
    assert combine([result("a", True), result("b", False, 0.5, weight=3.0)], 0.7).passed is False
    assert (
        combine([result("a", True, 1.0), result("b", True, 0.8)], 0.9).score == pytest.approx(0.9)
        and combine([result("a", True, 1.0), result("b", True, 0.8)], 0.9).passed
    )
    must = combine([result("a", True), result("m", False, must=True)], 0.0)
    assert not must.passed and must.failed_musts == ("m",) and must.score == 0.5


def test_combine_edge_cases() -> None:
    assert not combine([], 0.0).passed
    zero = combine([result("a", True, weight=0.0), result("b", False, weight=0.0)], 0.5)
    assert zero.score == 0.5 and zero.passed
    assert not combine([result("m", False, weight=0.0, must=True)], 0.0).passed


def test_property_a_failed_must_never_passes_whatever_the_scores() -> None:
    for seed in range(300):
        rng = Prng(seed)
        results = [
            result(f"g{i}", rng.integer(0, 1) == 1, rng.real(0, 1), weight=rng.real(0, 5), must=False)
            for i in range(rng.integer(1, 6))
        ]
        results.append(result("must", False, rng.real(0, 1), weight=rng.real(0, 5), must=True))
        assert not combine(rng.shuffled(results), rng.real(0, 1)).passed, seed


def test_property_score_is_between_zero_and_one_and_order_free() -> None:
    for seed in range(200):
        rng = Prng(seed)
        results = [result(f"g{i}", True, rng.real(0, 1), weight=rng.real(0.1, 5)) for i in range(rng.integer(1, 6))]
        a, b = combine(results, 0.0), combine(rng.shuffled(results), 0.0)
        assert 0.0 <= a.score <= 1.0 and a.score == pytest.approx(b.score), seed


def test_price_table_and_cost() -> None:
    table = parse_price_table(
        js({"currency": "EUR", "models": {"m": {"input_per_mtok": 2, "output_per_mtok": 10.5}}}), "prices.json"
    )
    assert table.currency == "EUR"
    assert cost_of(Usage("m", 1_000_000, 500_000), table) == Decimal("7.25")
    assert cost_of(Usage("m", 0, 0), table) == Decimal(0)
    assert cost_of(Usage("other", 1, 1), table) is None
    assert cost_of(Usage(None, 1, 1), table) is None
    assert cost_of(None, table) is None
    assert cost_of(Usage("m", 1, 1), EMPTY_TABLE) is None


def test_price_cost_is_exact_decimal_arithmetic() -> None:
    table = parse_price_table(js({"models": {"m": {"input_per_mtok": 0.1, "output_per_mtok": 0.2}}}), "p")
    assert cost_of(Usage("m", 3, 3), table) == Decimal("0.0000009")


@pytest.mark.parametrize(
    ("doc", "fragment"),
    [
        ([], "must be a JSON object (got array)"),
        ({}, "models: is required"),
        ({"models": {"m": 5}}, "models.m: must be an object (got number)"),
        ({"models": {"m": {"input_per_mtok": 1}}}, "models.m.output_per_mtok: is required"),
        ({"models": {"m": {"input_per_mtok": -1, "output_per_mtok": 1}}}, "must be at least 0.0"),
        ({"models": {}, "vendor": 1}, "vendor: unknown key"),
    ],
)
def test_price_table_errors(doc: object, fragment: str) -> None:
    with pytest.raises(ValidationError) as error:
        parse_price_table(js(doc), "prices.json")
    assert fragment in str(error.value)


def test_splitmix_matches_the_published_reference_values() -> None:
    rng = SplitMix64(1234567)
    assert [rng.next_u64() for _ in range(3)] == [6457827717110365317, 3203168211198807973, 9817491932198370423]


def test_below_is_uniform_enough_and_bounded() -> None:
    rng = SplitMix64(5)
    counts = [0] * 6
    for _ in range(6000):
        counts[rng.below(6)] += 1
    assert all(850 < c < 1150 for c in counts)
    with pytest.raises(ValueError):
        rng.below(0)


def test_sample_is_deterministic_ordered_and_distinct() -> None:
    items = list(range(100))
    a = sample(items, 10, seed=3)
    assert a == sample(items, 10, seed=3) and a == sorted(a) and len(set(a)) == 10
    assert a != sample(items, 10, seed=4)
    assert sample(items, 100, seed=1) == items and sample(items, 500, seed=1) == items
    assert sample(items, 0, seed=1) == []


def test_property_every_item_can_be_sampled() -> None:
    seen: set[int] = set()
    for seed in range(200):
        seen.update(sample(list(range(20)), 3, seed))
    assert seen == set(range(20))


def test_percentile_nearest_rank() -> None:
    values = [15, 20, 35, 40, 50]
    assert percentile(values, 50) == 35 and percentile(values, 95) == 50 and percentile(values, 0) == 15
    assert percentile(values, 100) == 50 and percentile(values, 20) == 15 and percentile(values, 40) == 20
    assert percentile([], 50) == 0.0 and percentile([7], 95) == 7
    assert percentile(list(reversed(values)), 50) == 35


def test_mean_variance() -> None:
    assert mean([]) == 0.0 and mean([1, 2, 3]) == 2.0
    assert variance([1]) == 0.0 and variance([]) == 0.0
    assert variance([1, 2, 3, 4]) == pytest.approx(5 / 3)


def test_fisher_exact_known_values() -> None:
    assert fisher_exact(5, 5, 0, 5) == pytest.approx(2 / 252)  # 1/252 per tail, two tails
    assert fisher_exact(1, 1, 0, 1) == 1.0  # one run each can never be significant
    assert fisher_exact(3, 6, 3, 6) == 1.0
    assert fisher_exact(0, 0, 3, 5) == 1.0
    assert fisher_exact(8, 10, 2, 10) == pytest.approx(0.02301, abs=1e-5)  # R: fisher.test(matrix(c(8,2,2,8),2))


def test_property_fisher_is_symmetric_and_a_probability() -> None:
    for seed in range(300):
        rng = Prng(seed)
        n1, n2 = rng.integer(1, 12), rng.integer(1, 12)
        k1, k2 = rng.integer(0, n1), rng.integer(0, n2)
        p = fisher_exact(k1, n1, k2, n2)
        assert 0.0 < p <= 1.0, seed
        assert p == pytest.approx(fisher_exact(k2, n2, k1, n1)), seed
        assert p == pytest.approx(fisher_exact(n1 - k1, n1, n2 - k2, n2)), seed


def test_wilson_interval() -> None:
    assert wilson_interval(0, 0) == (0.0, 1.0)
    lo, hi = wilson_interval(8, 10)
    assert lo == pytest.approx(0.490, abs=1e-3) and hi == pytest.approx(0.943, abs=1e-3)
    assert wilson_interval(10, 10)[1] == 1.0 and wilson_interval(0, 10)[0] == 0.0
    assert math.isclose(sum(wilson_interval(5, 10)) / 2, 0.5, abs_tol=0.02)
