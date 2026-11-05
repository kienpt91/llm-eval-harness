"""judge: an external program decides. Non-deterministic by nature; its verdict and confidence are recorded.

The domain only validates the parameters and turns a verdict into an outcome. Running the program happens
behind the ``Judge`` port; see ``docs/wiring.md`` for how to put a judge model behind it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from llm_eval_harness.domain.errors import Problem
from llm_eval_harness.domain.grading.params import Params
from llm_eval_harness.domain.jsonio import JsonValue, kind_of
from llm_eval_harness.domain.limits import MAX_MESSAGE_CHARS
from llm_eval_harness.domain.model import Outcome
from llm_eval_harness.domain.text import truncate

DEFAULT_ENV_ALLOW = ("PATH", "LANG", "LC_ALL")


@dataclass(frozen=True, slots=True)
class JudgeConfig:
    argv: tuple[str, ...]
    timeout_s: float
    env_allow: tuple[str, ...]
    rubric: str
    min_confidence: float
    low_confidence: str


@dataclass(frozen=True, slots=True)
class JudgeVerdict:
    verdict: str
    score: float
    confidence: float
    reason: str


def parse_judge(raw: Mapping[str, JsonValue], path: str, problems: list[Problem]) -> JudgeConfig:
    p = Params(raw, path, problems, {"argv", "timeout_s", "env_allow", "rubric", "min_confidence", "low_confidence"})
    argv = p.strings("argv", required=True, min_items=1, max_len=1000)
    if len(argv) > 32:
        p.fail("argv", "has too many items (limit 32)")
    return JudgeConfig(
        argv,
        p.number("timeout_s", default=60.0, minimum=0.1, maximum=600.0) or 60.0,
        p.strings("env_allow", max_len=128),
        p.string("rubric", default="", max_len=20_000, non_empty=False) or "",
        p.number("min_confidence", default=0.7, minimum=0.0, maximum=1.0) or 0.0,
        p.choice("low_confidence", ("fail", "pass"), default="fail"),
    )


def parse_verdict(obj: JsonValue) -> JudgeVerdict:
    """Read a judge program's answer. Raises ``ValueError`` with a precise message when it is malformed."""
    if not isinstance(obj, dict):
        raise ValueError(f"judge answer must be a JSON object (got {kind_of(obj)})")
    verdict = obj.get("verdict")
    if verdict not in ("pass", "fail"):
        raise ValueError('judge answer needs "verdict": "pass" or "fail"')
    confidence = obj.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0.0 <= confidence <= 1.0:
        raise ValueError('judge answer needs "confidence": a number between 0 and 1')
    score = obj.get("score", 1.0 if verdict == "pass" else 0.0)
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0.0 <= score <= 1.0:
        raise ValueError('"score" must be a number between 0 and 1')
    reason = obj.get("reason", "")
    if not isinstance(reason, str):
        raise ValueError('"reason" must be a string')
    return JudgeVerdict(str(verdict), float(score), float(confidence), truncate(reason, MAX_MESSAGE_CHARS))


def judge_outcome(config: JudgeConfig, verdict: JudgeVerdict) -> Outcome:
    """Pass when the judge says pass with enough confidence; a low-confidence verdict follows ``low_confidence``."""
    note = f"judge says {verdict.verdict} (confidence {verdict.confidence:.2f})"
    if verdict.reason:
        note += f": {verdict.reason}"
    if verdict.confidence < config.min_confidence:
        ok = config.low_confidence == "pass"
        return Outcome(
            ok,
            verdict.score,
            f"{note}; below min_confidence {config.min_confidence:.2f}, counted as {config.low_confidence}",
        )
    return Outcome(verdict.verdict == "pass", verdict.score, note)
