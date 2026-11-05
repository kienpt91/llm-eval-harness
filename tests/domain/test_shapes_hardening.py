"""Evasion and coverage of the secret and personal-data shapes: what is recognised, what is redacted, what is not."""

from __future__ import annotations

import time

import pytest

from llm_eval_harness.domain.shapes import find_pii, find_secrets, redact
from support import keys


def pem_with_body() -> str:
    return keys.private_key_block()


def test_redaction_removes_the_whole_private_key_block_not_just_its_header() -> None:
    body = keys.filler(40)
    out = redact(f"before\n{pem_with_body()}\nafter")
    assert body not in out and "END" not in out and "PRIVATE KEY" not in out
    assert out.startswith("before\n[REDACTED:private-key-block]") and out.endswith("\nafter")


def test_a_private_key_block_without_an_end_marker_is_redacted_to_a_bounded_window() -> None:
    header = pem_with_body().split("\n")[0]
    body = keys.filler(40)
    out = redact(f"x {header}\n{body}\nmore text")
    assert body not in out and "more text" not in out  # no END: everything after the header, up to the window
    long_tail = "t" * 40_000
    out = redact(f"{header}\n{body}\n{long_tail}")
    assert body not in out and out.count("t") > 20_000  # only a bounded window is consumed


def test_several_private_key_blocks_are_each_redacted() -> None:
    text = f"{pem_with_body()}\nmiddle\n{pem_with_body()}"
    out = redact(text)
    assert out.count("[REDACTED:private-key-block]") == 2 and "middle" in out and keys.filler(40) not in out


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s[:3] + "\u200b" + s[3:],  # zero width space inside the token
        lambda s: s[:3] + "\u2060" + s[3:],  # word joiner
        lambda s: s[:2] + "\u00ad" + s[2:],  # soft hyphen
        lambda s: "".join(chr(ord(c) + 0xFEE0) if "!" <= c <= "~" else c for c in s[:5]) + s[5:],  # fullwidth prefix
    ],
    ids=["zwsp", "word-joiner", "soft-hyphen", "fullwidth"],
)
def test_invisible_and_fullwidth_obfuscation_does_not_hide_a_secret(mutate: object) -> None:
    token = keys.github_token()
    hidden = mutate(token)  # type: ignore[operator]
    assert hidden != token
    text = f"the key is {hidden} ok"
    assert [f.kind for f in find_secrets(text)] == ["github-token"]
    out = redact(text)
    assert keys.filler(36)[:20] not in out and out == "the key is [REDACTED:github-token] ok"


def test_obfuscated_pii_is_found_too() -> None:
    assert [f.kind for f in find_pii("mail jane\u200b@example.com now")] == ["email"]
    assert [f.kind for f in find_pii("ssn 078\u200b-05-1120")] == ["us-ssn"]


@pytest.mark.parametrize("glue", ["my_", "x-", "key=", "KEY:", "(", '"', "_", "-", "a-", "9_"])
def test_a_secret_glued_to_other_text_is_still_found_unless_it_is_part_of_a_longer_word(glue: str) -> None:
    token = keys.github_token()
    found = find_secrets(glue + token)
    if glue[-1].isalnum():
        assert found == []  # inside a longer alphanumeric word: not a token boundary
    else:
        assert [f.kind for f in found] == ["github-token"], glue


def test_tokens_longer_than_the_pattern_cap_are_redacted_whole() -> None:
    token = keys.github_token() + keys.filler(2000, seed=3)
    out = redact(f"k={token} end")
    assert out == "k=[REDACTED:github-token] end"
    sk = keys.sk_style_key() + keys.filler(3000, seed=5)
    assert redact(f"[{sk}]") == "[[REDACTED:sk-style-key]]"


def test_a_jwt_with_a_long_payload_is_found() -> None:
    long_jwt = "ey" + "J" + keys.filler(30) + ".ey" + "J" + keys.filler(3000, seed=2) + "." + keys.filler(20)
    assert [f.kind for f in find_secrets(long_jwt)] == ["jwt"]
    assert redact(f"t {long_jwt} t") == "t [REDACTED:jwt] t"


@pytest.mark.parametrize(
    "text",
    [
        ("eyJ-" * 250_000),
        ("sk-" * 300_000),
        ("ghp_" + "a" * 40 + "-") * 20_000,
        ("-" * 5 + "BEGIN " + "PRIVATE " + "KEY" + "-" * 5) * 40_000,
        ("x-ghp_" * 160_000),
    ],
    ids=["jwt-dashes", "sk", "ghp-dashes", "pem-headers", "ghp-glue"],
)
def test_relaxed_boundaries_and_tails_stay_linear_on_hostile_megabytes(text: str) -> None:
    started = time.monotonic()
    find_secrets(text)
    redact(text)
    assert time.monotonic() - started < 20


def test_non_ascii_megabytes_are_normalised_in_linear_time() -> None:
    started = time.monotonic()
    find_secrets("é" * 400_000 + keys.github_token() + "\u200b" * 400_000)
    assert time.monotonic() - started < 15


def test_spans_map_back_to_the_original_text() -> None:
    token = keys.github_token()
    hidden = token[:3] + "\u200b" + token[3:]
    text = f"aé {hidden} z"
    [finding] = find_secrets(text)
    assert text[finding.start : finding.end] == hidden
