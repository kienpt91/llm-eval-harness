from __future__ import annotations

import pytest

from llm_eval_harness.domain.grading.regexsafe import compiled
from llm_eval_harness.domain.shapes import (
    PII_KINDS,
    SECRET_KINDS,
    find_pii,
    find_secrets,
    luhn_valid,
    redact,
)
from support import keys
from support.scaling import assert_scales_linearly


@pytest.mark.parametrize(("kind", "secret"), sorted(keys.all_secrets().items()))
def test_every_secret_shape_is_found_with_its_kind(kind: str, secret: str) -> None:
    text = f"before {secret} after"
    found = find_secrets(text)
    assert kind in {f.kind for f in found}, (kind, found)
    matched = [text[f.start : f.end] for f in found if f.kind == kind]
    assert any(secret.startswith(m) or m.startswith(secret) for m in matched), (kind, matched)


def test_all_documented_kinds_have_a_fixture() -> None:
    assert set(keys.all_secrets()) <= set(SECRET_KINDS)
    missing = set(SECRET_KINDS) - set(keys.all_secrets())
    assert missing == {"github-fine-grained-token", "webhook-secret", "google-api-key", "package-registry-token"}


def test_remaining_prefix_shapes() -> None:
    texts = {
        "github-fine-grained-token": "github" + "_pat_" + keys.filler(40),
        "webhook-secret": "wh" + "sec_" + keys.filler(24),
        "google-api-key": "A" + "Iza" + keys.filler(35),
        "package-registry-token": "np" + "m_" + keys.filler(36),
    }
    for kind, text in texts.items():
        assert kind in {f.kind for f in find_secrets(f"x {text} y")}


@pytest.mark.parametrize(
    "benign",
    [
        "the weather is sunny",
        "AKIA is a prefix",
        "sk-short",
        "ghp_tooshort",
        "eyJ.eyJ.x",
        "password: hunter2",
        "Bearer token auth is documented",
        "0x1234",
        "see https://example.com/path",
        "user@example.com is fine for secrets",
    ],
)
def test_benign_text_has_no_secret_shapes(benign: str) -> None:
    assert find_secrets(benign) == []


def test_findings_hold_positions_not_text() -> None:
    secret = keys.github_token()
    [finding] = [f for f in find_secrets(f"a {secret}") if f.kind == "github-token"]
    assert (finding.start, finding.end) == (2, 2 + len(secret))
    assert not hasattr(finding, "text")


def test_kinds_filter() -> None:
    text = keys.github_token() + " " + keys.aws_key_id()
    assert {f.kind for f in find_secrets(text, ["aws-access-key-id"])} == {"aws-access-key-id"}


def test_redact_replaces_with_kind_marker_and_never_keeps_the_secret() -> None:
    secret = keys.github_token()
    out = redact(f"token {secret} end")
    assert secret not in out
    assert out == "token [REDACTED:github-token] end"
    assert redact("nothing here") == "nothing here"


def test_redact_handles_overlapping_matches_and_custom_patterns() -> None:
    text = f"{keys.quoted_assignment()} then {keys.sk_style_key()}"
    out = redact(text)
    assert "REDACTED" in out and keys.sk_style_key() not in out
    out2 = redact("order ORD-12345 here", [compiled(r"ORD-\d+").spans])
    assert out2 == "order [REDACTED:custom] here"
    nested = keys.url_with_password()
    assert nested not in redact(f"db={nested}&k={keys.bearer()}")


@pytest.mark.parametrize(
    "unit",
    ["a", "eyJ", "sk-", "ghp_", "a://", "Bearer ", "x@", "1 ", "api_key" + " " * 40, "-----" + "BEGIN ", "0x", "a."],
    ids=["a", "jwt", "sk", "ghp", "url", "bearer", "email", "digits", "assign", "pem", "hex", "dots"],
)
def test_scanning_hostile_text_scales_linearly(unit: str) -> None:
    """Twelve adversarial shapes: four times the text may cost about four times the CPU, never sixteen times."""

    def scan(repeats: int) -> None:
        text = unit * repeats
        find_secrets(text)
        find_pii(text)
        redact(text)

    assert_scales_linearly(scan, 4_000)


def test_pii_shapes() -> None:
    text = "mail jane.doe+x@example.co.uk, call +44 20 7946 0958 or 415-555-0132, ssn 078-05-1120, card 4111 1111 1111 1111"
    kinds = {f.kind for f in find_pii(text)}
    assert kinds == {"email", "phone", "us-ssn", "credit-card"}
    assert set(PII_KINDS) == {"email", "phone", "us-ssn", "credit-card"}


def test_credit_card_needs_a_valid_luhn_sum() -> None:
    assert luhn_valid("4111111111111111")
    assert not luhn_valid("4111111111111112")
    assert find_pii("order 4111111111111112") == []
    assert [f.kind for f in find_pii("pay 4111-1111-1111-1111 now")] == ["credit-card"]


@pytest.mark.parametrize(
    "benign", ["version 1.2.3", "year 2026-11-05", "id 12345678", "ratio 3:4", "call me maybe", "a@b"]
)
def test_benign_text_has_no_pii_shapes(benign: str) -> None:
    assert find_pii(benign) == []


def test_pii_kind_filter() -> None:
    assert [f.kind for f in find_pii("a@example.com 078-05-1120", ["us-ssn"])] == ["us-ssn"]
