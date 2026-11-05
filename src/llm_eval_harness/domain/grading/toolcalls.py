"""tool_calls and forbidden_tools: judge what an agent did, not what it said.

Authorisation failures of agents show up as tool calls: a refund nobody asked for, a message sent to an
address from a poisoned document. These graders read the tool calls the provider reports.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from llm_eval_harness.domain.errors import Problem
from llm_eval_harness.domain.grading.params import Params
from llm_eval_harness.domain.grading.regexsafe import BudgetExceeded, check_pattern, pattern_matches, too_long_for_regex
from llm_eval_harness.domain.jsonio import JsonValue, contains_json, json_equal, kind_of
from llm_eval_harness.domain.model import GradeContext, Outcome, ToolCall, ToolCallExpectation
from llm_eval_harness.domain.text import one_line

OPERATORS = ("equals", "one_of", "regex", "contains", "present", "absent", "type", "min", "max")
TYPE_NAMES = ("string", "number", "integer", "boolean", "null", "object", "array")


def check_constraint(value: JsonValue, path: str, problems: list[Problem]) -> None:
    """Validate one argument constraint: a literal (equality) or an operator object."""
    if not isinstance(value, dict):
        return
    if not value:
        problems.append(Problem(path, 'an empty object is not a constraint; use {"equals": {}} for an empty object'))
        return
    p = Params(value, path, problems, OPERATORS)
    exclusive = [k for k in value if k not in ("min", "max")]
    if len(exclusive) > 1:
        p.fail("", f"combines operators {', '.join(sorted(exclusive))}; use one (only min and max combine)")
    if "one_of" in value and (not isinstance(value["one_of"], list) or not value["one_of"]):
        p.fail("one_of", "must be a non-empty array")
    if "regex" in value:
        pattern = p.string("regex", max_len=512)
        reason = check_pattern(pattern) if pattern else None
        if reason:
            p.fail("regex", reason)
    if "contains" in value:
        p.string("contains")
    for flag in ("present", "absent"):
        if flag in value and value[flag] is not True:
            p.fail(flag, "must be true")
    if "type" in value and value["type"] not in TYPE_NAMES:
        p.fail("type", f"must be one of {', '.join(TYPE_NAMES)}")
    p.number("min")
    p.number("max")


def _type_matches(value: JsonValue, name: str) -> bool:
    return {
        "string": isinstance(value, str),
        "boolean": isinstance(value, bool),
        "null": value is None,
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
    }[name]


def satisfies(actual: JsonValue, present: bool, constraint: JsonValue) -> str | None:
    """``None`` when the argument satisfies the constraint, else the reason it does not."""
    if not isinstance(constraint, dict):
        if not present:
            return "argument is missing"
        return None if json_equal(actual, constraint) else "value differs from the expected one"
    if "absent" in constraint:
        return "argument is present but must be absent" if present else None
    if not present:
        return "argument is missing"
    if "present" in constraint:
        return None
    if "equals" in constraint:
        return None if json_equal(actual, constraint["equals"]) else "value differs from the expected one"
    if "one_of" in constraint:
        options = constraint["one_of"]
        return (
            None
            if isinstance(options, list) and contains_json(options, actual)
            else "value is not one of the allowed values"
        )
    if "regex" in constraint:
        pattern = constraint["regex"]
        if isinstance(actual, str) and too_long_for_regex(actual):
            return "value is too long to test against a pattern"
        if not isinstance(actual, str) or not isinstance(pattern, str):
            return "value does not match the required pattern"
        try:
            return None if pattern_matches(pattern, (), actual) else "value does not match the required pattern"
        except BudgetExceeded:
            return "pattern used up its work budget on this value (failing closed)"
    if "contains" in constraint:
        needle = constraint["contains"]
        ok = isinstance(actual, str) and isinstance(needle, str) and needle in actual
        return None if ok else "value does not contain the required text"
    if "type" in constraint:
        name = constraint["type"]
        return (
            None
            if isinstance(name, str) and _type_matches(actual, name)
            else f"expected type {name}, got {kind_of(actual)}"
        )
    lo, hi = constraint.get("min"), constraint.get("max")
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        return f"expected a number, got {kind_of(actual)}"
    if isinstance(lo, (int, float)) and actual < lo:
        return f"{actual} is below the minimum {lo}"
    if isinstance(hi, (int, float)) and actual > hi:
        return f"{actual} is above the maximum {hi}"
    return None


def call_matches(call: ToolCall, expectation: ToolCallExpectation) -> str | None:
    if call.name.strip().casefold() != expectation.name.strip().casefold():
        return "different tool"
    for key, constraint in expectation.args.items():
        present = key in call.arguments
        reason = satisfies(call.arguments.get(key), present, constraint)
        if reason:
            return f"argument {one_line(key, max_chars=40)!r}: {reason}"
    return None


# --- tool_calls ----------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToolCallsConfig:
    calls: tuple[ToolCallExpectation, ...]
    ordered: bool
    allow_extra: bool


def parse_expectations(items: list[JsonValue], path: str, problems: list[Problem]) -> tuple[ToolCallExpectation, ...]:
    out: list[ToolCallExpectation] = []
    for index, item in enumerate(items):
        where = f"{path}[{index}]"
        if not isinstance(item, dict):
            problems.append(Problem(where, f"must be an object with a name and optional args (got {kind_of(item)})"))
            continue
        p = Params(item, where, problems, {"name", "args"})
        name = p.string("name", required=True, max_len=200)
        args = p.obj("args") or {}
        for key, constraint in args.items():
            check_constraint(constraint, f"{where}.args.{key}", problems)
        if name is not None:
            out.append(ToolCallExpectation(name, dict(args)))
    return tuple(out)


def parse_tool_calls(raw: Mapping[str, JsonValue], path: str, problems: list[Problem]) -> ToolCallsConfig:
    p = Params(raw, path, problems, {"calls", "ordered", "allow_extra"})
    items = p.array("calls", required=True) or []
    return ToolCallsConfig(
        parse_expectations(items, f"{path}.calls", problems),
        p.boolean("ordered", default=False),
        p.boolean("allow_extra", default=True),
    )


def _match_any_order(expected: tuple[ToolCallExpectation, ...], actual: tuple[ToolCall, ...]) -> dict[int, int]:
    """Maximum matching of expectations to distinct actual calls (augmenting paths)."""
    edges = {i: [j for j, call in enumerate(actual) if call_matches(call, e) is None] for i, e in enumerate(expected)}
    owner: dict[int, int] = {}  # actual index -> expectation index

    def augment(i: int, seen: set[int]) -> bool:
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owner or augment(owner[j], seen):
                owner[j] = i
                return True
        return False

    for i in range(len(expected)):
        augment(i, set())
    return {i: j for j, i in owner.items()}


def _match_in_order(expected: tuple[ToolCallExpectation, ...], actual: tuple[ToolCall, ...]) -> dict[int, int]:
    matched: dict[int, int] = {}
    position = 0
    for i, expectation in enumerate(expected):
        for j in range(position, len(actual)):
            if call_matches(actual[j], expectation) is None:
                matched[i] = j
                position = j + 1
                break
    return matched


def grade_tool_calls(config: ToolCallsConfig, ctx: GradeContext) -> Outcome:
    """Passes when every expected call happened (with constrained arguments), in order if ``ordered``.

    Each expected call must match a distinct actual call by tool name (case-insensitive) and by every
    argument constraint. With ``allow_extra`` false, any actual call that no expectation explains also fails,
    so ``calls: []`` with ``allow_extra: false`` means "no tool may be called". The score is the fraction of
    expected calls found. Failure modes: arguments are checked only for the keys named in the constraints, so
    an extra unconstrained argument (a different recipient, say) passes; constrain every argument that
    matters or use ``absent``. Order is checked as a subsequence: other calls may sit between expected ones.
    """
    actual = ctx.output.tool_calls
    matched = _match_in_order(config.calls, actual) if config.ordered else _match_any_order(config.calls, actual)
    missing = [e for i, e in enumerate(config.calls) if i not in matched]
    used = set(matched.values())
    extra = [call for j, call in enumerate(actual) if j not in used]
    total = len(config.calls)
    score = (len(matched) / total) if total else (0.0 if (extra and not config.allow_extra) else 1.0)
    problems: list[str] = []
    for e in missing:
        same = [c for c in actual if c.name.strip().casefold() == e.name.strip().casefold()]
        if not same:
            why = "never called"
        else:
            reasons = [r for c in same if (r := call_matches(c, e)) is not None]
            why = reasons[0] if len(reasons) == len(same) else "out of order or already used"
        problems.append(f"expected call to {one_line(e.name, max_chars=60)!r} not found ({why})")
    if extra and not config.allow_extra:
        names = ", ".join(repr(one_line(c.name, max_chars=40)) for c in extra[:5])
        problems.append(f"{len(extra)} unexpected call(s): {names}")
    if problems:
        return Outcome(False, score if total else 0.0, "; ".join(problems))
    return Outcome(True, 1.0, f"{len(matched)} expected call(s) found" if total else "no tool calls, as expected")


# --- forbidden_tools -----------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ForbiddenToolsConfig:
    tools: tuple[str, ...]


def parse_forbidden_tools(raw: Mapping[str, JsonValue], path: str, problems: list[Problem]) -> ForbiddenToolsConfig:
    p = Params(raw, path, problems, {"tools"})
    return ForbiddenToolsConfig(p.strings("tools", required=True, min_items=1, max_len=200))


def grade_forbidden_tools(config: ForbiddenToolsConfig, ctx: GradeContext) -> Outcome:
    """Fails when the provider reports a call to any forbidden tool, whatever its arguments.

    Names compare case-insensitively after trimming. Failure modes: it sees only the calls the provider
    reports; a pipeline that performs an action without reporting it as a tool call is invisible here, so the
    adapter around your pipeline must report every side effect.
    """
    banned = {t.strip().casefold() for t in config.tools}
    hits = sorted({c.name.strip().casefold() for c in ctx.output.tool_calls} & banned)
    if hits:
        return Outcome(False, 0.0, "forbidden tool called: " + ", ".join(repr(one_line(h, max_chars=40)) for h in hits))
    return Outcome(True, 1.0, f"none of {len(config.tools)} forbidden tools called")
