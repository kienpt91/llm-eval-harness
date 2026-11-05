"""In-memory doubles of the ports, for application tests."""

from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence

from llm_eval_harness.domain.errors import ProviderError, UsageError
from llm_eval_harness.domain.grading.judge import JudgeConfig
from llm_eval_harness.domain.jsonio import JsonValue
from llm_eval_harness.domain.model import ProviderOutput

Behaviour = ProviderOutput | ProviderError | Callable[[dict[str, JsonValue]], ProviderOutput]


class ScriptedProvider:
    """Answers each case id from a script: a list of behaviours consumed per call (the last one repeats)."""

    def __init__(self, script: Mapping[str, Sequence[Behaviour]], default: Behaviour | None = None) -> None:
        self.script = script
        self.default = default
        self.calls: dict[str, int] = defaultdict(int)
        self.requests: list[dict[str, JsonValue]] = []
        self.timeouts: list[float] = []
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.delay: threading.Event | None = None

    def describe(self) -> dict[str, JsonValue]:
        return {"type": "scripted"}

    def generate(self, request: dict[str, JsonValue], timeout_s: float) -> ProviderOutput:
        case = request["case"]
        assert isinstance(case, dict)
        case_id = str(case["id"])
        with self.lock:
            index = self.calls[case_id]
            self.calls[case_id] += 1
            self.requests.append(request)
            self.timeouts.append(timeout_s)
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            if self.delay is not None:
                self.delay.wait(0.05)
            behaviours = self.script.get(case_id)
            behaviour = behaviours[min(index, len(behaviours) - 1)] if behaviours else self.default
            if behaviour is None:
                raise AssertionError(f"no script for {case_id}")
            if isinstance(behaviour, ProviderError):
                raise behaviour
            if callable(behaviour):
                return behaviour(request)
            return behaviour
        finally:
            with self.lock:
                self.active -= 1


class FakeClock:
    def __init__(self) -> None:
        self.ticks = 0
        self.sleeps: list[float] = []

    def now_iso(self) -> str:
        self.ticks += 1
        return f"2026-01-01T00:00:{self.ticks:02d}Z"

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)


class MemoryFiles:
    def __init__(self, files: dict[str, str] | None = None) -> None:
        self.files: dict[str, str] = dict(files or {})
        self.writes: list[str] = []

    def read_text(self, path: str, max_bytes: int) -> str:
        if path not in self.files:
            raise UsageError(f"{path}: no such file")
        text = self.files[path]
        if len(text.encode()) > max_bytes:
            raise UsageError(f"{path}: larger than {max_bytes} bytes")
        return text

    def write_text(self, path: str, text: str, *, overwrite: bool) -> str:
        if path in self.files and not overwrite:
            raise UsageError(f"{path}: already exists")
        self.files[path] = text
        self.writes.append(path)
        return path


class ScriptedJudge:
    def __init__(self, answers: list[JsonValue | ProviderError]) -> None:
        self.answers = answers
        self.calls = 0
        self.requests: list[dict[str, JsonValue]] = []

    def run(self, config: JudgeConfig, request: dict[str, JsonValue]) -> JsonValue:
        answer = self.answers[min(self.calls, len(self.answers) - 1)]
        self.calls += 1
        self.requests.append(request)
        if isinstance(answer, ProviderError):
            raise answer
        return answer
