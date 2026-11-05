"""Records of a run and the per-case statistics derived from them."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from llm_eval_harness.domain.model import GradeResult, ToolCall, Usage
from llm_eval_harness.domain.stats import mean, variance

PASSED, FAILED, ERROR = "passed", "failed", "error"


@dataclass(frozen=True, slots=True)
class OutputRecord:
    """The stored form of an answer: redacted and truncated as configured."""

    text: str
    truncated: bool
    tool_calls: tuple[ToolCall, ...]
    citations: tuple[str, ...]
    retrieved: tuple[str, ...]
    usage: Usage | None
    latency_ms: float
    cost_usd: Decimal | None


@dataclass(frozen=True, slots=True)
class ErrorRecord:
    kind: str
    message: str


@dataclass(frozen=True, slots=True)
class RunRecord:
    repeat: int
    attempts: int
    status: str
    score: float
    failed_musts: tuple[str, ...]
    grades: tuple[GradeResult, ...]
    output: OutputRecord | None
    error: ErrorRecord | None


@dataclass(frozen=True, slots=True)
class CaseResult:
    id: str
    category: str
    tags: tuple[str, ...]
    case_hash: str
    input_preview: str
    runs: tuple[RunRecord, ...]


@dataclass(frozen=True, slots=True)
class CaseStat:
    """What is known about one case after all its runs: the unit that comparison and gating work on."""

    id: str
    category: str
    case_hash: str
    graded_runs: int
    passes: int
    errors: int
    mean_score: float
    score_variance: float = 0.0

    @property
    def failures(self) -> int:
        return self.graded_runs - self.passes

    @property
    def pass_rate(self) -> float | None:
        return self.passes / self.graded_runs if self.graded_runs else None

    @property
    def flaky(self) -> bool:
        return 0 < self.passes < self.graded_runs

    @property
    def status(self) -> str:
        if self.failures:
            return FAILED
        return ERROR if self.errors else PASSED


def stat_of(result: CaseResult) -> CaseStat:
    graded = [r for r in result.runs if r.status != ERROR]
    scores = [r.score for r in graded]
    return CaseStat(
        id=result.id,
        category=result.category,
        case_hash=result.case_hash,
        graded_runs=len(graded),
        passes=sum(1 for r in graded if r.status == PASSED),
        errors=sum(1 for r in result.runs if r.status == ERROR),
        mean_score=mean(scores),
        score_variance=variance(scores),
    )


def stats_of(results: Sequence[CaseResult]) -> list[CaseStat]:
    return [stat_of(r) for r in sorted(results, key=lambda r: r.id)]
