"""The machine-readable gate decision (schema version 1), written next to the report."""

from __future__ import annotations

from llm_eval_harness.domain.compare import Comparison, Delta
from llm_eval_harness.domain.gate import GateDecision
from llm_eval_harness.domain.jsonio import JsonValue
from llm_eval_harness.domain.report import SCHEMA_VERSION

GATE_KIND = "llm-eval-gate"


def _delta(d: Delta) -> JsonValue:
    return {"baseline": d.baseline, "current": d.current, "absolute": d.absolute, "percent": d.percent}


def _comparison(c: Comparison) -> JsonValue:
    cases: list[JsonValue] = [
        {
            "id": x.id,
            "category": x.category,
            "verdict": x.verdict,
            "delta": x.delta,
            "p_value": x.p_value,
            "reason": x.reason,
        }
        for x in c.cases
    ]
    return {
        "verdict": c.verdict,
        "dataset_changed": c.dataset_changed,
        "reasons": list(c.reasons),
        "notes": list(c.notes),
        "accuracy": _delta(c.accuracy),
        "cost_per_run": _delta(c.cost_per_run),
        "tokens_per_run": _delta(c.tokens_per_run),
        "latency_p50_ms": _delta(c.latency_p50),
        "latency_p95_ms": _delta(c.latency_p95),
        "cases": cases,
    }


def gate_document(decision: GateDecision, comparison: Comparison | None) -> dict[str, JsonValue]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": GATE_KIND,
        "exit_code": decision.exit_code,
        "passed": decision.passed,
        "verdict": decision.verdict,
        "reasons": list(decision.reasons),
        "notes": list(decision.notes),
        "comparison": None if comparison is None else _comparison(comparison),
    }
