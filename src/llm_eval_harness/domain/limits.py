"""Hard limits that bound memory and work. Every input that crosses a boundary is checked against these."""

from __future__ import annotations

MAX_DATASET_BYTES = 8 * 1024 * 1024
MAX_CASES = 10_000
MAX_CASE_BYTES = 256 * 1024
MAX_JSON_DEPTH = 32
MAX_ID_LENGTH = 128
MAX_TAGS = 32
MAX_TAG_LENGTH = 64
MAX_CONTEXT_DOCUMENTS = 64
MAX_LIST_ITEMS = 256
MAX_GRADERS = 32

DEFAULT_MAX_OUTPUT_BYTES = 1024 * 1024
HARD_MAX_OUTPUT_BYTES = 16 * 1024 * 1024
MAX_STDERR_BYTES = 64 * 1024
MAX_CONFIG_BYTES = 1024 * 1024
MAX_REPORT_BYTES = 256 * 1024 * 1024
MAX_PRICE_TABLE_BYTES = 256 * 1024

MAX_REGEX_LENGTH = 512
MAX_MESSAGE_CHARS = 2000
"""Longest message (error text, grader detail) kept in a report."""
STORED_OUTPUT_CHARS = 20_000
"""Longest model output stored in a report; the grading itself always sees the whole output."""
