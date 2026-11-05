"""Serialising the report document to the bytes that are written to disk."""

from __future__ import annotations

import json

from llm_eval_harness.domain.jsonio import JsonValue


def json_text(document: dict[str, JsonValue]) -> str:
    """Report JSON: ASCII only (no raw control or bidi characters can reach the file), two-space indent, insertion order (the report builder already orders everything), trailing newline."""
    return json.dumps(document, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
