from __future__ import annotations

import pytest

from llm_eval_harness.domain.grading.numeric import extract_numbers
from llm_eval_harness.domain.grading.sets import f1, tokens
from support.builders import grade, output
from support.prng import Prng


def test_numbers_are_extracted_with_signs_decimals_and_thousands() -> None:
    assert extract_numbers("pay 1,234.50 or -3, then 2e3 and 7") == [1234.5, -3.0, 2000.0, 7.0]
    assert extract_numbers("none here") == []
    assert extract_numbers("v1.2.3") == []  # a version is not a number
    assert extract_numbers("a1 b2") == []


def test_numeric_tolerances() -> None:
    p = {"expected": 100, "abs_tol": 1}
    assert grade("numeric", p, output("about 100.9")).passed
    assert not grade("numeric", p, output("about 101.5")).passed
    assert grade("numeric", {"expected": 100, "rel_tol": 0.05}, output("104")).passed
    assert not grade("numeric", {"expected": 100, "rel_tol": 0.05}, output("106")).passed
    assert grade("numeric", {"expected": 5}, output("5.0")).passed
    assert not grade("numeric", {"expected": 5}, output("no numbers")).passed


def test_numeric_which_number_counts() -> None:
    text = "3 items cost 12.50"
    assert not grade("numeric", {"expected": 12.5}, output(text)).passed
    assert grade("numeric", {"expected": 12.5, "which": "last"}, output(text)).passed
    assert grade("numeric", {"expected": 12.5, "which": "any"}, output(text)).passed
    assert "differs" in grade("numeric", {"expected": 12.5}, output(text)).detail


def test_set_overlap_recall_precision_and_jaccard() -> None:
    p = {"expected": ["red", "green", "blue"]}
    assert grade("set_overlap", p, output("Blue\n- Red\n3. green")).passed
    miss = grade("set_overlap", p, output("red\ngreen"))
    assert not miss.passed and abs(miss.score - 2 / 3) < 1e-9
    assert grade("set_overlap", {**p, "min_recall": 0.5}, output("red\ngreen")).passed
    assert not grade("set_overlap", {**p, "min_precision": 0.9}, output("red\ngreen\nblue\nextra")).passed
    assert grade("set_overlap", {**p, "extract": "commas"}, output("red, green , blue")).passed
    assert not grade(
        "set_overlap", {**p, "extract": "lines", "case_sensitive": True}, output("Red\nGreen\nBlue")
    ).passed


def test_set_overlap_structured_lists() -> None:
    p = {"expected": ["a", "b"]}
    assert grade("set_overlap", p, output("", structured=["a", "b"], has_structured=True)).passed
    assert not grade("set_overlap", {**p, "extract": "structured"}, output("a\nb")).passed
    assert grade(
        "set_overlap", {**p, "extract": "structured"}, output("", structured=["b", 1, "a"], has_structured=True)
    ).passed
    assert not grade("set_overlap", p, output("")).passed  # empty answer: nothing found


def test_token_f1() -> None:
    p = {"references": ["the refund window is 30 days"], "min_f1": 0.6}
    assert grade("token_f1", p, output("The refund window is 30 days.")).score == pytest.approx(1.0)
    assert grade("token_f1", p, output("refunds take 30 days")).score < 0.6
    assert not grade("token_f1", p, output("completely unrelated words")).passed
    two = {"references": ["alpha beta", "gamma delta"], "min_f1": 0.9}
    assert grade("token_f1", two, output("gamma delta")).passed


def test_f1_properties() -> None:
    assert f1([], []) == 1.0
    assert f1(["a"], []) == 0.0
    assert f1(["a"], ["b"]) == 0.0
    assert f1(["a", "a"], ["a"]) == pytest.approx(2 * 0.5 * 1 / 1.5)
    for seed in range(200):
        rng = Prng(seed)
        a = [rng.word() for _ in range(rng.integer(0, 8))]
        b = [rng.word() for _ in range(rng.integer(0, 8))]
        assert f1(a, b) == pytest.approx(f1(b, a)), seed
        assert 0.0 <= f1(a, b) <= 1.0, seed
        assert f1(a, a) == 1.0, seed
        assert f1(a, rng.shuffled(a)) == 1.0, seed


def test_tokens_are_bounded_and_folded() -> None:
    assert tokens("Hello, WORLD_1! ｗｏｒｌｄ") == ["hello", "world_1", "world"]
    assert len(tokens("a " * 500_000)) == 200_000
