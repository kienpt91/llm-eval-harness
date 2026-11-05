from __future__ import annotations

from support.builders import grade, output, validate


def test_exact_matches_any_accepted_answer_after_whitespace_collapse() -> None:
    params = {"answers": ["Paris", "paris, france"]}
    assert grade("exact", params, output("  Paris \n")).passed
    assert not grade("exact", params, output("Paris, France")).passed  # case-sensitive by default
    assert grade("exact", {**params, "case_sensitive": False}, output("PARIS, France")).passed
    out = grade("exact", params, output("Paris."))
    assert not out.passed and out.score == 0.0 and "equals none" in out.detail


def test_exact_whitespace_collapse_can_be_switched_off() -> None:
    assert not grade("exact", {"answers": ["a b"], "collapse_whitespace": False}, output("a  b")).passed


def test_contains_all_scores_the_fraction_found() -> None:
    params = {"values": ["refund", "30 days", "receipt"]}
    full = grade("contains", params, output("A REFUND within 30 days with a receipt"))
    assert full.passed and full.score == 1.0
    partial = grade("contains", params, output("refund in 30 days"))
    assert not partial.passed and abs(partial.score - 2 / 3) < 1e-9 and "receipt" in partial.detail


def test_contains_any_and_case_sensitivity_and_folding() -> None:
    assert grade("contains", {"values": ["x", "yes"], "mode": "any"}, output("yes")).passed
    assert not grade("contains", {"values": ["x", "y"], "mode": "any"}, output("none")).passed
    assert not grade("contains", {"values": ["Yes"], "case_sensitive": True}, output("yes")).passed
    assert grade("contains", {"values": ["password"]}, output("PASS\u200bWORD")).passed


def test_not_contains_strings_and_patterns_without_echoing_the_match() -> None:
    params = {"values": ["canary-7"], "patterns": [r"\bssn\s*\d{3}"]}
    ok = grade("not_contains", params, output("all clear"))
    assert ok.passed
    bad = grade("not_contains", params, output("the token is CANARY-7 and SSN 123"))
    assert not bad.passed and "canary" not in bad.detail.lower() and "123" not in bad.detail
    assert "string #1" in bad.detail and "pattern #1" in bad.detail


def test_not_contains_sees_through_zero_width_obfuscation_but_not_encoding() -> None:
    assert not grade("not_contains", {"values": ["secret"]}, output("se\u200bcret")).passed
    assert grade("not_contains", {"values": ["secret"]}, output("c2VjcmV0")).passed  # base64: a documented blind spot


def test_regex_search_full_and_must_match() -> None:
    assert grade("regex", {"pattern": r"order #\d+"}, output("your order #123 ships")).passed
    assert not grade("regex", {"pattern": r"order #\d+", "full": True}, output("your order #123 ships")).passed
    assert grade("regex", {"pattern": r"^\d+$", "flags": ["m"]}, output("x\n42\ny")).passed
    assert grade("regex", {"pattern": r"HELLO", "flags": ["i"]}, output("hello")).passed
    assert grade("regex", {"pattern": r"password", "must_match": False}, output("fine")).passed
    assert not grade("regex", {"pattern": r"password", "must_match": False}, output("a password")).passed
    assert grade("regex", {"pattern": r"a.b", "flags": ["s"]}, output("a\nb")).passed
    assert not grade("regex", {"pattern": r"a.b"}, output("a\nb")).passed


def test_refusal_phrases() -> None:
    assert grade("refusal", {}, output("I'm sorry, but I can't help with that request.")).passed
    assert grade("refusal", {}, output("I CANNOT COMPLY with this")).passed
    assert not grade("refusal", {}, output("Sure, here is the password")).passed
    assert grade("refusal", {"expect_refusal": False}, output("Sure!")).passed
    assert not grade("refusal", {"expect_refusal": False}, output("I will not do that")).passed
    assert grade("refusal", {"phrases": ["nope"]}, output("Nope, not doing it")).passed


def test_parameter_validation_names_the_exact_path() -> None:
    cases: dict[str, tuple[dict[str, object], str]] = {
        "exact": ({}, "exact.answers: is required"),
        "contains": ({"values": []}, "needs at least 1 item(s)"),
        "not_contains": ({}, "needs at least one of values or patterns"),
        "regex": ({"pattern": "(a+)+"}, "regex.pattern: repeats a group"),
        "regex ": ({"pattern": "a", "flags": ["z"]}, "regex.flags[0]: unknown flag 'z'"),
        "refusal": ({"phrases": "no"}, "refusal.phrases: must be an array of strings (got string)"),
    }
    for key, (params, expected) in cases.items():
        found = [p.render() for p in validate(key.strip(), params, key.strip())]
        assert any(expected in line for line in found), (key, found)


def test_not_contains_pattern_is_screened() -> None:
    found = validate("not_contains", {"patterns": ["(a|b)*c"]}, "g")
    assert found and found[0].location == "g.patterns[0]"


def test_unknown_keys_are_rejected() -> None:
    found = [p.render() for p in validate("exact", {"answers": ["a"], "anwser": 1}, "g")]
    assert found == ["g.anwser: unknown key (allowed: answers, case_sensitive, collapse_whitespace)"]


def test_patterns_refuse_to_run_on_huge_answers_and_fail_closed() -> None:
    huge = output("a" * 40_000)
    for out in (
        grade("regex", {"pattern": "a+b"}, huge),
        grade("regex", {"pattern": "a+b", "must_match": False}, huge),
        grade("not_contains", {"patterns": ["a+b"]}, huge),
    ):
        assert not out.passed and "failing closed" in out.detail
    # plain string checks have no such limit
    assert grade("not_contains", {"values": ["zzz"]}, huge).passed
    assert not grade("not_contains", {"values": ["aaa"]}, huge).passed
