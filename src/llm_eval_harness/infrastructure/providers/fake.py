"""The built-in deterministic ``fake`` provider, used by the tests and by the demo.

Spec (the ``provider`` object of the configuration, without ``type``)::

    {"model": "fake-model", "latency_ms": 2,
     "default": {"text": "..."},                          # answer for cases without a script entry; echo if absent
     "script": {"case-id": ENTRY}}

``ENTRY`` is one of: an answer object (the keys of the wire protocol, plus optional ``latency_ms``); a list of
entries, one per repeat (``repeat % len`` picks one: this is how flaky behaviour is scripted); or an error entry
``{"error": {"kind": "timeout", "message": "...", "fail_attempts": 1, "then": ANSWER}}`` that raises the error on
the first ``fail_attempts`` attempts and answers with ``then`` afterwards. Token usage is estimated from the
text lengths unless the answer carries its own. Everything depends only on the request: no clock, no
randomness, no state.
"""

from __future__ import annotations

import dataclasses
import json
import math

from llm_eval_harness.application.config import FakeProviderConfig
from llm_eval_harness.domain.errors import RETRYABLE_KINDS, ProviderError, UsageError
from llm_eval_harness.domain.jsonio import JsonValue, canonical, kind_of
from llm_eval_harness.domain.model import ProviderOutput, Usage
from llm_eval_harness.domain.protocol import parse_response

KNOWN_KEYS = ("model", "latency_ms", "default", "script")
ERROR_KINDS = ("timeout", "spawn", "exit", "network", "protocol", "refused")


def _tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


class FakeProvider:
    def __init__(self, config: FakeProviderConfig) -> None:
        spec = config.spec
        extra = sorted(set(spec) - set(KNOWN_KEYS))
        if extra:
            raise UsageError(
                f"provider.{extra[0]}: unknown key for the fake provider (allowed: {', '.join(KNOWN_KEYS)})"
            )
        model = spec.get("model", "fake-model")
        latency = spec.get("latency_ms", 1.0)
        if (
            not isinstance(model, str)
            or isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or latency < 0
        ):
            raise UsageError("provider.model must be a string and provider.latency_ms a non-negative number")
        self._model, self._latency = model, float(latency)
        self._default = spec.get("default")
        script = spec.get("script", {})
        if not isinstance(script, dict):
            raise UsageError(f"provider.script: must be an object (got {kind_of(script)})")
        self._script = script
        for entry in [self._default, *script.values()]:
            if entry is not None:
                self._check(entry)

    def _check(self, entry: JsonValue, depth: int = 0) -> None:
        if isinstance(entry, list):
            if not entry or depth > 0:
                raise UsageError("provider.script: a list needs at least one entry and cannot nest")
            for item in entry:
                self._check(item, depth + 1)
        elif isinstance(entry, dict) and "error" in entry:
            error = entry["error"]
            if (
                not isinstance(error, dict)
                or error.get("kind") not in ERROR_KINDS
                or not isinstance(error.get("message", ""), str)
            ):
                raise UsageError(
                    f"provider.script: an error entry needs a kind (one of {', '.join(ERROR_KINDS)}) and a message"
                )
            if "then" in error:
                self._check(error["then"], depth + 1)
        elif isinstance(entry, dict):
            body = {k: v for k, v in entry.items() if k != "latency_ms"}
            try:
                parse_response(body)
            except ProviderError as error:
                raise UsageError(f"provider.script: {error}") from None
        else:
            raise UsageError(f"provider.script: entries are objects or lists (got {kind_of(entry)})")

    def describe(self) -> dict[str, JsonValue]:
        return {"type": "fake", "model": self._model}

    def generate(self, request: dict[str, JsonValue], timeout_s: float) -> ProviderOutput:  # noqa: ARG002 - the fake never waits
        case = request["case"]
        assert isinstance(case, dict)
        repeat, attempt = request["repeat"], request["attempt"]
        assert isinstance(repeat, int) and isinstance(attempt, int)
        entry = self._script.get(str(case["id"]), self._default)
        if entry is None:
            return self._answer(self._echo(case), case)
        if isinstance(entry, list):
            entry = entry[repeat % len(entry)]
        return self._resolve(entry, case, attempt)

    def _resolve(self, entry: JsonValue, case: dict[str, JsonValue], attempt: int) -> ProviderOutput:
        assert isinstance(entry, dict)
        if "error" not in entry:
            return self._answer(entry, case)
        error = entry["error"]
        assert isinstance(error, dict)
        kind = str(error["kind"])
        fail_attempts = error.get("fail_attempts", 10**9)
        if isinstance(fail_attempts, int) and attempt <= fail_attempts:
            raise ProviderError(kind, str(error.get("message", "scripted failure")), retryable=kind in RETRYABLE_KINDS)
        then = error.get("then")
        assert isinstance(then, dict)
        return self._answer(then, case)

    def _echo(self, case: dict[str, JsonValue]) -> dict[str, JsonValue]:
        value = case.get("input")
        if isinstance(value, dict) and isinstance(value.get("question"), str):
            return {"text": str(value["question"])}
        return {"text": value if isinstance(value, str) else canonical(value)}

    def _answer(self, body: dict[str, JsonValue], case: dict[str, JsonValue]) -> ProviderOutput:
        latency = body.get("latency_ms", self._latency)
        clean = {k: v for k, v in body.items() if k != "latency_ms"}
        output = parse_response(clean)
        if output.usage is None:
            usage = Usage(self._model, _tokens(json.dumps(case.get("input"), sort_keys=True)), _tokens(output.text))
            output = dataclasses.replace(output, usage=usage)
        if isinstance(latency, bool) or not isinstance(latency, (int, float)):
            latency = self._latency
        return dataclasses.replace(output, latency_ms=float(latency))
