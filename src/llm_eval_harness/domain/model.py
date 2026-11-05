"""The vocabulary of the tool: cases, expectations, grader specifications, what a provider returns."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from llm_eval_harness.domain.jsonio import JsonValue

CATEGORIES = ("quality", "safety")


@dataclass(frozen=True, slots=True)
class ContextDocument:
    id: str
    text: str
    metadata: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolCallExpectation:
    """An expected tool call: the tool name and a constraint per argument (see ``grading.toolcalls``)."""

    name: str
    args: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Expected:
    answers: tuple[str, ...] = ()
    facts: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()
    forbidden_patterns: tuple[str, ...] = ()
    tool_calls: tuple[ToolCallExpectation, ...] | None = None
    forbidden_tools: tuple[str, ...] = ()
    tool_order: str = "any"
    allow_extra_tools: bool = True


@dataclass(frozen=True, slots=True)
class GraderSpec:
    """One grader of a case: its type, parameters, weight and whether a failure is fatal for the case."""

    type: str
    params: dict[str, JsonValue]
    weight: float = 1.0
    must: bool = False
    name: str = ""


@dataclass(frozen=True, slots=True)
class Case:
    id: str
    category: str
    tags: tuple[str, ...]
    input: JsonValue
    context: tuple[ContextDocument, ...]
    expected: Expected
    graders: tuple[GraderSpec, ...]
    threshold: float
    metadata: dict[str, JsonValue]
    hash: str

    @property
    def is_safety(self) -> bool:
        return self.category == "safety"


@dataclass(frozen=True, slots=True)
class Dataset:
    cases: tuple[Case, ...]
    hash: str


@dataclass(frozen=True, slots=True)
class ToolCall:
    name: str
    arguments: dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class Usage:
    model: str | None
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class ProviderOutput:
    """What a provider returned for one case, plus the latency its adapter measured."""

    text: str
    structured: JsonValue = None
    has_structured: bool = False
    tool_calls: tuple[ToolCall, ...] = ()
    citations: tuple[str, ...] = ()
    retrieved: tuple[str, ...] = ()
    usage: Usage | None = None
    latency_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class GradeContext:
    """Everything a grader may look at. Expectations are not here: they were turned into grader parameters."""

    case: Case
    output: ProviderOutput
    cost_usd: Decimal | None


@dataclass(frozen=True, slots=True)
class Outcome:
    """What a grader decided, before weights and ``must`` are applied."""

    passed: bool
    score: float
    detail: str


@dataclass(frozen=True, slots=True)
class GradeResult:
    name: str
    type: str
    passed: bool
    score: float
    weight: float
    must: bool
    detail: str
    deterministic: bool = True
    confidence: float | None = None
    verdict: str | None = None
