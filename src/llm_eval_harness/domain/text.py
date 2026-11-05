"""Text hygiene: terminal sanitising, matching normalisation, truncation. Pure string functions.

Model output and datasets are untrusted. Before any of it reaches a terminal or a Markdown report it passes
through ``sanitize``: escape sequences are removed and every remaining control, bidirectional-control or
invisible character is shown as a visible escape instead of being interpreted.
"""

from __future__ import annotations

import re
import unicodedata

# Complete escape sequences. Each pattern is linear: no nested quantifiers, bounded alternatives.
_CSI = r"\x1b\[[0-?]{0,256}[ -/]{0,64}[@-~]"
_OSC = r"\x1b\][^\x07\x1b]{0,8192}(?:\x07|\x1b\\)"
_STRING_SEQ = r"\x1b[PX^_][^\x1b]{0,8192}\x1b\\"
_SHORT = r"\x1b[0-?@-OQ-WYZ\\`a-~]"
_C1_CSI = r"\x9b[0-?]{0,256}[ -/]{0,64}[@-~]"
_ESCAPES = re.compile("|".join((_CSI, _OSC, _STRING_SEQ, _C1_CSI, _SHORT)))

_KEEP = frozenset("\n\t")


def _escape_char(ch: str) -> str:
    code = ord(ch)
    return f"\\x{code:02x}" if code < 0x100 else f"<U+{code:04X}>"


def _needs_escape(ch: str) -> bool:
    if ch in _KEEP:
        return False
    category = unicodedata.category(ch)
    # Cc controls, Cf format (bidi, zero width, joiners), Cs surrogates, Co private use, Cn unassigned,
    # Zl/Zp line and paragraph separators
    return category in ("Cc", "Cf", "Cs", "Co", "Cn", "Zl", "Zp")


def sanitize(text: str, *, max_chars: int | None = None) -> str:
    """Make untrusted text safe to print: strip escape sequences, escape control and invisible characters."""
    if max_chars is not None and len(text) > max_chars:
        cut = text[:max_chars]
        suffix = f"…[{len(text) - max_chars} more characters]"
    else:
        cut, suffix = text, ""
    stripped = _ESCAPES.sub("", cut)
    # a lone ESC left over (an unterminated sequence) is escaped below like any control character
    return "".join(_escape_char(c) if _needs_escape(c) else c for c in stripped) + suffix


def one_line(text: str, *, max_chars: int = 200) -> str:
    """Sanitised, single-line rendering for table cells and headlines."""
    clean = sanitize(text, max_chars=max_chars)
    return clean.replace("\n", "\\n").replace("\t", "\\t")


def truncate(text: str, max_chars: int) -> str:
    """Cut text to at most ``max_chars`` characters and say how much was dropped."""
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}…[{len(text) - max_chars} more characters]"


def fold(text: str) -> str:
    """Normalisation for matching forbidden or expected strings.

    NFKC folds width and compatibility variants; invisible format characters (zero width spaces, bidi marks,
    soft hyphens) are removed; case is folded. It defeats cheap obfuscation, not paraphrase or encoding.
    """
    normalized = unicodedata.normalize("NFKC", text)
    visible = "".join(c for c in normalized if unicodedata.category(c) != "Cf")
    return visible.casefold()


def collapse_whitespace(text: str) -> str:
    return " ".join(text.split())


def fence_for(text: str) -> str:
    """A Markdown code fence longer than any run of backticks in ``text``."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)
