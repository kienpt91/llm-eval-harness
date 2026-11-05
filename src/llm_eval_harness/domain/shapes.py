"""Secret-shaped and PII-shaped strings.

The shapes are the ones the repository's secret scanner (``scripts/secret-scan.sh``) uses. They are assembled
at import time from fragments so that no source file contains a complete key-shaped literal, which keeps the
scanner quiet about its own rules. Every pattern is anchored at a boundary and uses bounded, possessive
repetition, so scanning a hostile megabyte of text takes linear time.

Findings never carry the matched text, only its kind and position: a report must not leak what it detects.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Finding:
    kind: str
    start: int
    end: int


def _join(*parts: str) -> str:
    return "".join(parts)


_B = r"(?<![A-Za-z0-9_-])"  # token boundary on the left
_DASHES = "-" * 5

SECRET_SHAPES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private-key-block", re.compile(_join(_DASHES, "BEGIN ", r"[A-Z ]{0,40}?", "PRIVATE ", "KEY", _DASHES))),
    ("aws-access-key-id", re.compile(_join(_B, "(?:", "AK", "IA|AS", "IA)", r"[0-9A-Z]{16}+(?![A-Za-z0-9])"))),
    ("github-token", re.compile(_join(_B, "(?:gh", "p|gh", "o|gh", "u|gh", "s|gh", "r)", r"_[A-Za-z0-9]{30,255}+"))),
    ("github-fine-grained-token", re.compile(_join(_B, "github", "_pat_", r"[A-Za-z0-9_]{30,255}+"))),
    ("payment-key", re.compile(_join(_B, "[sr]", "k_", "(?:live|test)_", r"[A-Za-z0-9]{16,255}+"))),
    ("webhook-secret", re.compile(_join(_B, "wh", "sec_", r"[A-Za-z0-9]{16,255}+"))),
    ("slack-token", re.compile(_join(_B, "xo", "x[abprs]-", r"[A-Za-z0-9-]{10,255}+"))),
    ("google-api-key", re.compile(_join(_B, "AI", "za", r"[0-9A-Za-z_-]{35}+"))),
    ("package-registry-token", re.compile(_join(_B, "np", "m_", r"[A-Za-z0-9]{36}+"))),
    ("sk-style-key", re.compile(_join(_B, "s", "k-", r"[A-Za-z0-9_-]{32,512}+"))),
    (
        "jwt",
        re.compile(
            _join(
                _B, "ey", "J", r"[A-Za-z0-9_-]{15,512}+\.ey", "J", r"[A-Za-z0-9_-]{15,2048}+\.[A-Za-z0-9_-]{10,1024}+"
            )
        ),
    ),
    (
        "url-with-password",
        re.compile(r"(?<![A-Za-z0-9+.-])[a-z][a-z0-9+.-]{0,31}+://[^/\s:@]{1,256}+:[^/\s@]{6,256}+@[^/\s]{1,255}+"),
    ),
    (
        "quoted-secret-assignment",
        re.compile(
            r"(?<![A-Za-z0-9_])(?:api[_-]?key|secret|token|passwd|password|private[_-]?key)[\"']?\s{0,32}+[:=]\s{0,32}+[\"'][A-Za-z0-9+/_=-]{24,512}+[\"']",
            re.IGNORECASE,
        ),
    ),
    ("raw-private-hex", re.compile(r"(?<![0-9A-Fa-f])0x[0-9a-fA-F]{64}+(?![0-9A-Fa-f])")),
    ("bearer-token", re.compile(r"(?<![A-Za-z0-9])Bearer\s{1,8}+[A-Za-z0-9._~+/=-]{16,1024}+", re.IGNORECASE)),
)

_LUHN_CANDIDATE = re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?!\d)")

PII_SHAPES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "email",
        re.compile(r"(?<![\w.+-])[\w.+-]{1,64}+@[A-Za-z0-9-]{1,63}+(?:\.[A-Za-z0-9-]{1,63}+){1,8}+(?![\w-])"),
    ),
    ("us-ssn", re.compile(r"(?<![\d-])\d{3}-\d{2}-\d{4}(?![\d-])")),
    (
        "phone",
        re.compile(r"(?<![\w.+-])(?:\+\d{1,3}[ .-]?)?(?:\(\d{2,4}\)|\d{2,4})[ .-]\d{3,4}[ .-]\d{3,4}(?![\d-])"),
    ),
)

SECRET_KINDS = tuple(name for name, _ in SECRET_SHAPES)
PII_KINDS = (*[name for name, _ in PII_SHAPES], "credit-card")


def luhn_valid(digits: str) -> bool:
    """The Luhn checksum of a string of digits."""
    total = 0
    for index, ch in enumerate(reversed(digits)):
        n = ord(ch) - 48
        if index % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def find_secrets(text: str, kinds: Iterable[str] | None = None) -> list[Finding]:
    wanted = None if kinds is None else frozenset(kinds)
    found = [
        Finding(name, m.start(), m.end())
        for name, pattern in SECRET_SHAPES
        if wanted is None or name in wanted
        for m in pattern.finditer(text)
    ]
    return sorted(found, key=lambda f: (f.start, f.kind))


def find_pii(text: str, kinds: Iterable[str] | None = None) -> list[Finding]:
    wanted = None if kinds is None else frozenset(kinds)
    found = [
        Finding(name, m.start(), m.end())
        for name, pattern in PII_SHAPES
        if wanted is None or name in wanted
        for m in pattern.finditer(text)
    ]
    if wanted is None or "credit-card" in wanted:
        for m in _LUHN_CANDIDATE.finditer(text):
            digits = re.sub(r"\D", "", m.group())
            if 13 <= len(digits) <= 19 and luhn_valid(digits):
                # the card number is one finding: drop the phone-shaped pieces inside it
                found = [f for f in found if not (f.kind == "phone" and f.start < m.end() and m.start() < f.end)]
                found.append(Finding("credit-card", m.start(), m.end()))
    return sorted(found, key=lambda f: (f.start, f.kind))


def redact(text: str, extra_patterns: Iterable[re.Pattern[str]] = ()) -> str:
    """Replace every secret-shaped string (and every match of the extra patterns) with a marker naming its kind."""
    spans = [(f.start, f.end, f.kind) for f in find_secrets(text)]
    spans.extend((m.start(), m.end(), "custom") for pattern in extra_patterns for m in pattern.finditer(text))
    if not spans:
        return text
    spans.sort()
    out: list[str] = []
    position = 0
    for start, end, kind in spans:
        if start < position:  # overlaps an earlier match: already replaced
            position = max(position, end)
            continue
        out.append(text[position:start])
        out.append(f"[REDACTED:{kind}]")
        position = end
    out.append(text[position:])
    return "".join(out)
