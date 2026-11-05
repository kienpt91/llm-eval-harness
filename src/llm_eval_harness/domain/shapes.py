"""Secret-shaped and PII-shaped strings.

The shapes are the ones the repository's secret scanner (``scripts/secret-scan.sh``) uses. They are assembled
at import time from fragments so that no source file contains a complete key-shaped literal, which keeps the
scanner quiet about its own rules.

What is recognised, precisely (see also docs/graders.md):

* **Shapes, not secrets.** A string is found when it *looks like* a credential of a known kind. A password or a
  token of no particular shape is not found, a revoked key is. Nothing here knows your secrets: for a secret you
  know, plant a canary and forbid it with ``not_contains``.
* **Unicode is normalised before matching.** Text is NFKC-normalised and invisible format characters (zero width
  characters, bidirectional marks, soft hyphens, word joiners) are removed, so full-width letters and an invisible
  character in the middle of a key do not hide it. Spans are mapped back to the original text, so redaction removes
  the original characters. Homoglyphs from other scripts (a Cyrillic letter that looks like a Latin one) are *not* folded.
* **Token boundaries.** A prefixed token is found when it is not glued to a preceding letter or digit
  (``my_<token>``, ``x-<token>``, ``key=<token>`` are found, ``abc<token>`` is not).
* **Long tokens are taken whole.** After a match, the span is extended over the token characters that follow, so
  a token longer than the pattern's cap is redacted entirely. A JWT segment longer than 4,096 characters is not
  recognised as a JWT (the pattern is bounded to stay linear).
* **PEM private keys are taken whole**: from the ``BEGIN`` line through the matching ``END`` line, or, if there is
  no ``END`` line within 16 KiB, through the next 16 KiB.
* **Split or encoded secrets are not found**: a key broken over two lines or strings, base64 or hex of a key,
  a key spelled out. Only text the harness sees is scanned.

Every pattern is anchored at a boundary and uses bounded, possessive repetition, so scanning a hostile megabyte
of text takes linear time (twelve adversarial shapes are tested).

Findings never carry the matched text, only its kind and position: a report must not leak what it detects.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Finding:
    kind: str
    start: int
    end: int


def _join(*parts: str) -> str:
    return "".join(parts)


_B = r"(?<![A-Za-z0-9])"  # token boundary on the left: not glued to a letter or digit
_DASHES = "-" * 5
PEM_WINDOW = 16 * 1024
_TOKEN_TAIL = re.compile(r"[A-Za-z0-9_-]*+")

SECRET_SHAPES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("aws-access-key-id", re.compile(_join(_B, "(?:", "AK", "IA|AS", "IA)", r"[0-9A-Z]{16}+(?![A-Za-z0-9])"))),
    ("github-token", re.compile(_join(_B, "(?:gh", "p|gh", "o|gh", "u|gh", "s|gh", "r)", r"_[A-Za-z0-9]{30,255}+"))),
    ("github-fine-grained-token", re.compile(_join(_B, "github", "_pat_", r"[A-Za-z0-9_]{30,255}+"))),
    ("payment-key", re.compile(_join(_B, "[sr]", "k_", "(?:live|test)_", r"[A-Za-z0-9]{16,255}+"))),
    ("webhook-secret", re.compile(_join(_B, "wh", "sec_", r"[A-Za-z0-9]{16,255}+"))),
    ("slack-token", re.compile(_join(_B, "xo", "x[abprs]-", r"[A-Za-z0-9-]{10,255}+"))),
    ("google-api-key", re.compile(_join(_B, "A", "Iza", r"[0-9A-Za-z_-]{35}+"))),
    ("package-registry-token", re.compile(_join(_B, "np", "m_", r"[A-Za-z0-9]{36}+"))),
    ("sk-style-key", re.compile(_join(_B, "s", "k-", r"[A-Za-z0-9_-]{32,512}+"))),
    (
        "jwt",
        re.compile(
            _join(
                _B, "ey", "J", r"[A-Za-z0-9_-]{15,512}+\.ey", "J", r"[A-Za-z0-9_-]{15,4096}+\.[A-Za-z0-9_-]{10,1024}+"
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
# kinds whose match is extended over the token characters that follow it (a token longer than the pattern's cap)
_EXTENDED = frozenset(
    {
        "github-token",
        "github-fine-grained-token",
        "payment-key",
        "webhook-secret",
        "slack-token",
        "google-api-key",
        "package-registry-token",
        "sk-style-key",
        "jwt",
        "bearer-token",
    }
)
_PEM_BEGIN = re.compile(_join(_DASHES, "BEGIN ", r"[A-Z ]{0,40}?", "PRIVATE ", "KEY", _DASHES))
_PEM_END = re.compile(_join(_DASHES, "END ", r"[A-Z ]{0,40}?", "PRIVATE ", "KEY", _DASHES))
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

SECRET_KINDS = ("private-key-block", *[name for name, _ in SECRET_SHAPES])
PII_KINDS = (*[name for name, _ in PII_SHAPES], "credit-card")

_NON_ASCII_RUN = re.compile(r"[^\x00-\x7f]+")


def normalised(text: str) -> tuple[str, list[int] | None]:
    """The text as the shapes see it, and for each of its characters the index in ``text`` it came from.

    NFKC-normalises and drops invisible format characters (category Cf). Pure ASCII text is returned as is, with
    ``None`` instead of a map (the identity), so the common case costs nothing.
    """
    if text.isascii():
        return text, None
    pieces: list[str] = []
    origin: list[int] = []
    last = 0
    for run in _NON_ASCII_RUN.finditer(text):
        pieces.append(text[last : run.start()])
        origin.extend(range(last, run.start()))
        for index, char in enumerate(run.group(), start=run.start()):
            if unicodedata.category(char) == "Cf":
                continue
            for folded in unicodedata.normalize("NFKC", char):
                if unicodedata.category(folded) == "Cf":
                    continue
                pieces.append(folded)
                origin.append(index)
        last = run.end()
    pieces.append(text[last:])
    origin.extend(range(last, len(text)))
    return "".join(pieces), origin


def _map(origin: list[int] | None, start: int, end: int) -> tuple[int, int]:
    if origin is None:
        return start, end
    return origin[start], origin[end - 1] + 1


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


def _pem_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    position = 0
    while True:
        begin = _PEM_BEGIN.search(text, position)
        if begin is None:
            return spans
        window_end = min(len(text), begin.end() + PEM_WINDOW)
        end = _PEM_END.search(text, begin.end(), window_end)
        stop = end.end() if end is not None else window_end
        spans.append((begin.start(), stop))
        position = stop


def _secret_spans(text: str, wanted: frozenset[str] | None) -> list[tuple[str, int, int]]:
    found: list[tuple[str, int, int]] = []
    if wanted is None or "private-key-block" in wanted:
        found.extend(("private-key-block", s, e) for s, e in _pem_spans(text))
    for name, pattern in SECRET_SHAPES:
        if wanted is not None and name not in wanted:
            continue
        position = 0
        while True:
            match = pattern.search(text, position)
            if match is None:
                break
            end = _TOKEN_TAIL.match(text, match.end()).end() if name in _EXTENDED else match.end()  # type: ignore[union-attr]
            found.append((name, match.start(), end))
            position = max(end, match.start() + 1)
    return found


def find_secrets(text: str, kinds: Iterable[str] | None = None) -> list[Finding]:
    wanted = None if kinds is None else frozenset(kinds)
    seen, origin = normalised(text)
    found = [Finding(kind, *_map(origin, s, e)) for kind, s, e in _secret_spans(seen, wanted)]
    return sorted(found, key=lambda f: (f.start, f.kind))


def find_pii(text: str, kinds: Iterable[str] | None = None) -> list[Finding]:
    wanted = None if kinds is None else frozenset(kinds)
    seen, origin = normalised(text)
    found = [
        Finding(name, *_map(origin, m.start(), m.end()))
        for name, pattern in PII_SHAPES
        if wanted is None or name in wanted
        for m in pattern.finditer(seen)
    ]
    if wanted is None or "credit-card" in wanted:
        for m in _LUHN_CANDIDATE.finditer(seen):
            digits = re.sub(r"\D", "", m.group())
            if 13 <= len(digits) <= 19 and luhn_valid(digits):
                # the card number is one finding: drop the phone-shaped pieces inside it
                start, end = _map(origin, m.start(), m.end())
                found = [f for f in found if not (f.kind == "phone" and f.start < end and start < f.end)]
                found.append(Finding("credit-card", start, end))
    return sorted(found, key=lambda f: (f.start, f.kind))


def redact(text: str, extra_spans: Iterable[Callable[[str], list[tuple[int, int]]]] = ()) -> str:
    """Replace every secret-shaped string, and every span the extra finders return, with a marker naming its kind.

    An extra finder takes the text and returns ``(start, end)`` spans; it may raise (the caller decides what failing
    closed means).
    """
    spans = [(f.start, f.end, f.kind) for f in find_secrets(text)]
    spans.extend((start, end, "custom") for finder in extra_spans for start, end in finder(text))
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
