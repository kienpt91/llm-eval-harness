"""Typed access to the parameter object of a grader or other JSON object, with precise error messages."""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping

from llm_eval_harness.domain.errors import Problem
from llm_eval_harness.domain.jsonio import JsonValue, kind_of
from llm_eval_harness.domain.limits import MAX_LIST_ITEMS


class Params:
    """Reads keys of a JSON object; every violation is appended to ``problems`` with its path."""

    def __init__(
        self, raw: Mapping[str, JsonValue], path: str, problems: list[Problem], allowed: Collection[str]
    ) -> None:
        self.raw = raw
        self.path = path
        self.problems = problems
        for key in sorted(raw):
            if key not in allowed:
                self.fail(key, f"unknown key (allowed: {', '.join(sorted(allowed))})")

    def fail(self, key: str, message: str) -> None:
        self.problems.append(Problem(f"{self.path}.{key}" if key else self.path, message))

    def _get(self, key: str, required: bool) -> JsonValue | None:
        if key not in self.raw:
            if required:
                self.fail(key, "is required")
            return None
        return self.raw[key]

    def string(
        self,
        key: str,
        *,
        required: bool = False,
        default: str | None = None,
        max_len: int = 100_000,
        non_empty: bool = True,
    ) -> str | None:
        value = self._get(key, required)
        if key not in self.raw:
            return default
        if not isinstance(value, str):
            self.fail(key, f"must be a string (got {kind_of(value)})")
            return default
        if non_empty and not value.strip():
            self.fail(key, "must not be empty")
            return default
        if len(value) > max_len:
            self.fail(key, f"is too long ({len(value)} characters, limit {max_len})")
            return default
        return value

    def strings(
        self, key: str, *, required: bool = False, min_items: int = 0, max_len: int = 10_000
    ) -> tuple[str, ...]:
        value = self._get(key, required)
        if key not in self.raw:
            return ()
        if not isinstance(value, list):
            self.fail(key, f"must be an array of strings (got {kind_of(value)})")
            return ()
        if len(value) > MAX_LIST_ITEMS:
            self.fail(key, f"has too many items ({len(value)}, limit {MAX_LIST_ITEMS})")
            return ()
        if len(value) < min_items:
            self.fail(key, f"needs at least {min_items} item(s)")
        out: list[str] = []
        for index, item in enumerate(value):
            if not isinstance(item, str):
                self.fail(f"{key}[{index}]", f"must be a string (got {kind_of(item)})")
            elif not item.strip():
                self.fail(f"{key}[{index}]", "must not be empty")
            elif len(item) > max_len:
                self.fail(f"{key}[{index}]", f"is too long ({len(item)} characters, limit {max_len})")
            else:
                out.append(item)
        return tuple(out)

    def number(
        self,
        key: str,
        *,
        required: bool = False,
        default: float | None = None,
        minimum: float | None = None,
        maximum: float | None = None,
        integer: bool = False,
    ) -> float | None:
        value = self._get(key, required)
        if key not in self.raw:
            return default
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            self.fail(key, f"must be a number (got {kind_of(value)})")
            return default
        if not math.isfinite(value):
            self.fail(key, "must be finite")
            return default
        if integer and float(value) != int(value):
            self.fail(key, f"must be a whole number (got {value})")
            return default
        if minimum is not None and value < minimum:
            self.fail(key, f"must be at least {minimum} (got {value})")
            return default
        if maximum is not None and value > maximum:
            self.fail(key, f"must be at most {maximum} (got {value})")
            return default
        return float(value)

    def integer(self, key: str, *, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
        """A whole number in range; ``default`` when absent or invalid (the problem is recorded)."""
        value = self.number(key, default=float(default), minimum=minimum, maximum=maximum, integer=True)
        return default if value is None else int(value)

    def boolean(self, key: str, *, default: bool) -> bool:
        value = self._get(key, False)
        if key not in self.raw:
            return default
        if not isinstance(value, bool):
            self.fail(key, f"must be true or false (got {kind_of(value)})")
            return default
        return value

    def choice(self, key: str, options: tuple[str, ...], *, default: str) -> str:
        value = self._get(key, False)
        if key not in self.raw:
            return default
        if not isinstance(value, str) or value not in options:
            self.fail(key, f"must be one of {', '.join(options)} (got {value!r})")
            return default
        return value

    def obj(self, key: str, *, required: bool = False) -> dict[str, JsonValue] | None:
        value = self._get(key, required)
        if key not in self.raw:
            return None
        if not isinstance(value, dict):
            self.fail(key, f"must be an object (got {kind_of(value)})")
            return None
        return value

    def array(self, key: str, *, required: bool = False) -> list[JsonValue] | None:
        value = self._get(key, required)
        if key not in self.raw:
            return None
        if not isinstance(value, list):
            self.fail(key, f"must be an array (got {kind_of(value)})")
            return None
        if len(value) > MAX_LIST_ITEMS:
            self.fail(key, f"has too many items ({len(value)}, limit {MAX_LIST_ITEMS})")
            return None
        return value
