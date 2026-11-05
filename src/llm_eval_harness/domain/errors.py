"""Exceptions of the domain. Callers translate them into messages and exit codes; nothing here prints."""

from __future__ import annotations

from dataclasses import dataclass


class HarnessError(Exception):
    """Base class of every error the tool raises on purpose."""


class UsageError(HarnessError):
    """The command line or a configuration file is wrong. Exit code 2."""


@dataclass(frozen=True, slots=True)
class Problem:
    """One precise validation finding: where it is and what is wrong."""

    location: str
    message: str

    def render(self) -> str:
        return f"{self.location}: {self.message}" if self.location else self.message


class ValidationError(HarnessError):
    """Input data failed validation; carries every problem found (up to a cap), not just the first."""

    def __init__(self, source: str, problems: list[Problem]) -> None:
        self.source = source
        self.problems = problems
        super().__init__(self.render())

    def render(self) -> str:
        lines = [f"{self.source}: {len(self.problems)} problem(s)"]
        lines.extend(f"  {p.render()}" for p in self.problems)
        return "\n".join(lines)


class DatasetError(ValidationError):
    """The dataset does not follow the dataset format. Exit code 2."""


class ProviderError(HarnessError):
    """A provider failed to produce a usable answer.

    ``kind`` is one of: ``timeout``, ``spawn``, ``exit``, ``network``, ``protocol``, ``refused``.
    ``retryable`` is true only for failures of the infrastructure that a second attempt can fix.
    """

    def __init__(self, kind: str, message: str, *, retryable: bool) -> None:
        self.kind = kind
        self.retryable = retryable
        super().__init__(message)


RETRYABLE_KINDS = frozenset({"timeout", "spawn", "exit", "network"})
