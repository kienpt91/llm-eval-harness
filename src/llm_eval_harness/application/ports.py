"""The ports: everything the application needs from the outside world, as narrow protocols."""

from __future__ import annotations

from typing import Protocol

from llm_eval_harness.domain.grading.judge import JudgeConfig
from llm_eval_harness.domain.jsonio import JsonValue
from llm_eval_harness.domain.model import ProviderOutput


class Provider(Protocol):
    """Runs one case through the system under test.

    ``generate`` takes the wire-protocol request (see ``domain.protocol``) and returns the parsed output with
    the latency the adapter measured. It raises ``ProviderError`` for every failure; the adapter decides
    whether the failure is retryable. It must be safe to call from several threads at once.
    """

    def describe(self) -> dict[str, JsonValue]:
        """A short, secret-free description for the report (type, command name, host)."""
        ...

    def generate(self, request: dict[str, JsonValue], timeout_s: float) -> ProviderOutput: ...


class JudgeRunner(Protocol):
    """Runs an external judge program for a ``judge`` grader and returns its JSON answer."""

    def run(self, config: JudgeConfig, request: dict[str, JsonValue]) -> JsonValue:
        """Raises ``ProviderError`` when the program fails, times out or answers with something unreadable."""
        ...


class Clock(Protocol):
    def now_iso(self) -> str:
        """The current UTC time as an ISO 8601 string, for report timestamps."""
        ...

    def sleep(self, seconds: float) -> None: ...


class ProjectFiles(Protocol):
    """Reads and writes files inside the project, nowhere else."""

    def read_text(self, path: str, max_bytes: int) -> str:
        """UTF-8 text of a file. Raises ``UsageError`` if it is missing, too large, not text, or not inside the
        project (including through a symbolic link)."""
        ...

    def write_text(self, path: str, text: str, *, overwrite: bool) -> str:
        """Write a file inside the project (creating directories) and return the path as written.
        Raises ``UsageError`` if the path leaves the project, goes through a symbolic link that does, or
        exists and ``overwrite`` is false."""
        ...
