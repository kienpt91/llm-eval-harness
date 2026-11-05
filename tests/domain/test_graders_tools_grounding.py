from __future__ import annotations

from llm_eval_harness.domain.model import ContextDocument, ToolCall
from support.builders import case, grade, output, validate


def call(name: str, **arguments: object) -> ToolCall:
    return ToolCall(name, arguments)  # type: ignore[arg-type]


def test_tool_call_name_and_argument_constraints() -> None:
    params = {
        "calls": [
            {
                "name": "lookup_order",
                "args": {
                    "order_id": {"regex": r"^ORD-\d+$"},
                    "region": "eu",
                    "limit": {"min": 1, "max": 5},
                    "mode": {"one_of": ["a", "b"]},
                    "note": {"contains": "refund"},
                    "force": {"absent": True},
                    "trace": {"present": True},
                    "kind": {"type": "string"},
                    "exact": {"equals": {"x": [1]}},
                },
            }
        ]
    }
    good = call(
        "Lookup_Order",
        order_id="ORD-12",
        region="eu",
        limit=3,
        mode="a",
        note="a refund",
        trace=1,
        kind="k",
        exact={"x": [1.0]},
    )
    assert grade("tool_calls", params, output(tool_calls=(good,))).passed
    bad_cases = {
        "ORD-12x": ("order_id", "ORD-12x"),
        "region": ("region", "us"),
        "limit": ("limit", 9),
        "mode": ("mode", "z"),
        "note": ("note", "nothing"),
        "kind": ("kind", 5),
        "exact": ("exact", {"x": [2]}),
    }
    for key, value in bad_cases.values():
        args: dict[str, object] = {
            "order_id": "ORD-12",
            "region": "eu",
            "limit": 3,
            "mode": "a",
            "note": "refund",
            "trace": 1,
            "kind": "k",
            "exact": {"x": [1]},
        }
        args[key] = value
        out = grade("tool_calls", params, output(tool_calls=(call("lookup_order", **args),)))
        assert not out.passed, key
    no_trace = call(
        "lookup_order", order_id="ORD-1", region="eu", limit=1, mode="a", note="refund", kind="k", exact={"x": [1]}
    )
    assert not grade("tool_calls", params, output(tool_calls=(no_trace,))).passed
    forced = call(
        "lookup_order",
        order_id="ORD-1",
        region="eu",
        limit=1,
        mode="a",
        note="refund",
        trace=1,
        kind="k",
        exact={"x": [1]},
        force=True,
    )
    assert "must be absent" in grade("tool_calls", params, output(tool_calls=(forced,))).detail
    missing = call("lookup_order", region="eu")
    assert (
        "argument 'order_id': argument is missing" in grade("tool_calls", params, output(tool_calls=(missing,))).detail
    )


def test_missing_tool_and_score_is_fraction_found() -> None:
    params = {"calls": [{"name": "a"}, {"name": "b"}]}
    out = grade("tool_calls", params, output(tool_calls=(call("a"),)))
    assert not out.passed and out.score == 0.5 and "'b' not found (never called)" in out.detail


def test_allow_extra_false_means_only_the_expected_calls() -> None:
    params = {"calls": [{"name": "a"}], "allow_extra": False}
    assert grade("tool_calls", params, output(tool_calls=(call("a"),))).passed
    out = grade("tool_calls", params, output(tool_calls=(call("a"), call("send_email"))))
    assert not out.passed and "unexpected call(s): 'send_email'" in out.detail
    assert grade("tool_calls", {"calls": [{"name": "a"}]}, output(tool_calls=(call("a"), call("x")))).passed


def test_no_tools_at_all() -> None:
    params = {"calls": [], "allow_extra": False}
    assert grade("tool_calls", params, output()).passed
    out = grade("tool_calls", params, output(tool_calls=(call("issue_refund"),)))
    assert not out.passed and out.score == 0.0


def test_order_is_a_subsequence_check() -> None:
    params = {"calls": [{"name": "a"}, {"name": "b"}], "ordered": True}
    assert grade("tool_calls", params, output(tool_calls=(call("a"), call("x"), call("b")))).passed
    out = grade("tool_calls", params, output(tool_calls=(call("b"), call("a"))))
    assert not out.passed and "out of order" in out.detail
    assert grade("tool_calls", {**params, "ordered": False}, output(tool_calls=(call("b"), call("a")))).passed


def test_two_expectations_need_two_distinct_calls() -> None:
    params = {"calls": [{"name": "a"}, {"name": "a"}]}
    assert not grade("tool_calls", params, output(tool_calls=(call("a"),))).passed
    assert grade("tool_calls", params, output(tool_calls=(call("a"), call("a")))).passed


def test_any_order_matching_finds_the_assignment_greedy_would_miss() -> None:
    params = {"calls": [{"name": "a"}, {"name": "a", "args": {"x": 1}}]}
    calls = (call("a", x=1), call("a", x=2))
    assert grade("tool_calls", params, output(tool_calls=calls)).passed


def test_forbidden_tools_ignore_case_and_arguments() -> None:
    params = {"tools": ["issue_refund", "delete_account"]}
    assert grade("forbidden_tools", params, output(tool_calls=(call("lookup"),))).passed
    out = grade("forbidden_tools", params, output(tool_calls=(call(" Issue_Refund "), call("lookup"))))
    assert not out.passed and "issue_refund" in out.detail


def test_tool_call_parameter_validation() -> None:
    def render(params: dict[str, object]) -> list[str]:
        return [p.render() for p in validate("tool_calls", params, "g")]

    assert render({}) == ["g.calls: is required"]
    assert render({"calls": [1]}) == ["g.calls[0]: must be an object with a name and optional args (got number)"]
    assert render({"calls": [{}]}) == ["g.calls[0].name: is required"]
    assert render({"calls": [{"name": "a", "args": {"x": {}}}]})[0].startswith("g.calls[0].args.x: an empty object")
    assert any(
        "combines operators" in line
        for line in render({"calls": [{"name": "a", "args": {"x": {"equals": 1, "regex": "a"}}}]})
    )
    assert any(
        "one_of: must be a non-empty array" in line
        for line in render({"calls": [{"name": "a", "args": {"x": {"one_of": []}}}]})
    )
    assert any(
        "back-references" in line for line in render({"calls": [{"name": "a", "args": {"x": {"regex": "(a)\\1"}}}]})
    )
    assert any(
        "present: must be true" in line
        for line in render({"calls": [{"name": "a", "args": {"x": {"present": False}}}]})
    )
    assert any(
        "type: must be one of" in line for line in render({"calls": [{"name": "a", "args": {"x": {"type": "str"}}}]})
    )
    assert any("unknown key" in line for line in render({"calls": [{"name": "a", "args": {"x": {"like": 1}}}]}))
    assert render({"calls": [{"name": "a", "args": {"x": {"min": 1, "max": 2}}}]}) == []


def test_huge_argument_is_not_run_through_a_pattern() -> None:
    params = {"calls": [{"name": "a", "args": {"x": {"regex": "a+b"}}}]}
    out = grade("tool_calls", params, output(tool_calls=(call("a", x="a" * 50_000),)))
    assert not out.passed and "too long" in out.detail


def test_grounding_checks_citations_against_supplied_documents() -> None:
    docs = (ContextDocument("d1", "text"), ContextDocument("d2", "text"))
    the_case = case(context=docs)
    assert grade("grounding", {}, output(citations=("d1", "d2")), the_case=the_case).passed
    out = grade("grounding", {}, output(citations=("d1", "ghost")), the_case=the_case)
    assert not out.passed and "'ghost'" in out.detail and out.score == 0.5
    assert not grade("grounding", {}, output(), the_case=the_case).passed  # min_citations 1
    assert grade("grounding", {"min_citations": 0}, output(), the_case=the_case).passed
    assert not grade("grounding", {"must_cite": ["d2"]}, output(citations=("d1",)), the_case=the_case).passed
    assert grade("grounding", {"must_cite": ["d2"]}, output(citations=("d1", "d2")), the_case=the_case).passed


def test_grounding_retrieved_documents_and_text_markers() -> None:
    the_case = case()
    assert grade("grounding", {}, output(citations=("r1",), retrieved=("r1", "r2")), the_case=the_case).passed
    assert not grade(
        "grounding", {"allow_retrieved": False}, output(citations=("r1",), retrieved=("r1",)), the_case=the_case
    ).passed
    marked = output("See [doc:r1] and [r2] but not [ghost]. Ignore [not a marker].", retrieved=("r1", "r2"))
    out = grade("grounding", {"text_markers": True}, marked, the_case=the_case)
    assert not out.passed and "'ghost'" in out.detail
    assert grade("grounding", {"text_markers": False, "min_citations": 0}, marked, the_case=the_case).passed
