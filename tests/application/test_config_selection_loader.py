from __future__ import annotations

import json

import pytest

from llm_eval_harness.application.config import (
    CommandProviderConfig,
    Config,
    FakeProviderConfig,
    HttpProviderConfig,
    parse_config,
)
from llm_eval_harness.application.dataset_loader import load_dataset
from llm_eval_harness.application.redaction import Redactor, bounded_json
from llm_eval_harness.application.selection import Selection, select_cases
from llm_eval_harness.application.usecases import load_config, load_prices, read_json, split_values
from llm_eval_harness.domain.errors import DatasetError, UsageError, ValidationError
from llm_eval_harness.domain.model import Case
from support import keys
from support.builders import case, js
from support.fakes import MemoryFiles

CASE = {"id": "a", "input": "x", "expected": {"facts": ["f"]}}


def cfg(obj: object) -> Config:
    return parse_config(js(obj), "llm-eval.json")


def errors(obj: object) -> list[str]:
    with pytest.raises(ValidationError) as error:
        cfg(obj)
    return [p.render() for p in error.value.problems]


def test_a_complete_configuration_is_read() -> None:
    config = cfg(
        {
            "dataset": "d.jsonl",
            "provider": {
                "type": "command",
                "argv": ["python3", "p.py"],
                "env_allow": ["TOKEN"],
                "env_set": {"MODE": "x"},
                "cwd": "sub",
            },
            "price_table": "prices.json",
            "defaults": {"repeat": 3, "concurrency": 8, "timeout_s": 5, "retries": 0, "seed": 4},
            "graders": [{"type": "latency_budget", "max_ms": 1000}],
            "redact": {"enabled": False, "patterns": ["ORD-\\d+"]},
            "out_dir": "build/eval",
            "baseline": "baseline.json",
            "gate": {
                "min_pass_rate": 0.9,
                "on_inconclusive": "pass",
                "tolerance": 0.05,
                "alpha": 0.01,
                "deterministic": True,
                "max_cost_increase_pct": 20,
                "max_p95_latency_increase_pct": 50,
                "max_regressed_cases": 2,
                "fail_on_removed": True,
            },
        }
    )
    assert isinstance(config.provider, CommandProviderConfig)
    assert (
        config.provider.argv == ("python3", "p.py")
        and config.provider.env_allow == ("TOKEN",)
        and config.provider.cwd == "sub"
    )
    assert config.provider.env_set == (("MODE", "x"),) and config.provider.max_output_bytes == 1024 * 1024
    assert (config.defaults.repeat, config.defaults.concurrency, config.defaults.retries, config.defaults.seed) == (
        3,
        8,
        0,
        4,
    )
    assert config.graders[0].type == "latency_budget" and not config.redact and config.out_dir == "build/eval"
    assert config.gate.min_pass_rate == 0.9 and config.gate.deterministic and config.gate.max_regressed_cases == 2


def test_an_empty_configuration_has_safe_defaults() -> None:
    config = cfg({})
    assert config.provider is None and config.redact and config.out_dir == "out" and config.defaults.retries == 1
    assert config.gate.on_inconclusive == "fail" and config.gate.min_pass_rate is None


def test_http_and_fake_providers() -> None:
    http = cfg(
        {
            "provider": {
                "type": "http",
                "url": "http://127.0.0.1:8080/run",
                "allow_hosts": ["api.example.org"],
                "headers": {"X-Team": "a"},
                "headers_env": {"Authorization": "MY_TOKEN"},
            }
        }
    ).provider
    assert (
        isinstance(http, HttpProviderConfig)
        and http.headers_env == (("Authorization", "MY_TOKEN"),)
        and http.allow_hosts == ("api.example.org",)
    )
    fake = cfg({"provider": {"type": "fake", "script": {"a": {"text": "x"}}}}).provider
    assert isinstance(fake, FakeProviderConfig) and fake.spec == {"script": {"a": {"text": "x"}}}


@pytest.mark.parametrize(
    ("obj", "fragment"),
    [
        ([], "must be a JSON object (got array)"),
        ({"dataset": 5}, "dataset: must be a string (got number)"),
        ({"datasets": "x"}, "datasets: unknown key"),
        ({"provider": {"type": "grpc"}}, "provider.type: must be one of command, http, fake (got 'grpc')"),
        ({"provider": {}}, "provider.type: must be one of"),
        ({"provider": {"type": "command"}}, "provider.argv: is required"),
        ({"provider": {"type": "command", "argv": []}}, "provider.argv: needs at least 1 item(s)"),
        ({"provider": {"type": "command", "argv": ["a"], "shell": True}}, "provider.shell: unknown key"),
        (
            {"provider": {"type": "command", "argv": ["a"], "env_allow": ["BAD NAME"]}},
            "env_allow[0]: is not a valid environment variable name",
        ),
        (
            {"provider": {"type": "command", "argv": ["a"], "env_set": {"1X": "v"}}},
            "env_set.1X: is not a valid environment variable name",
        ),
        ({"provider": {"type": "command", "argv": ["a"], "env_set": {"X": 1}}}, "env_set.X: must be a string"),
        (
            {"provider": {"type": "command", "argv": ["a"], "max_output_bytes": 10}},
            "max_output_bytes: must be at least 1024",
        ),
        ({"provider": {"type": "http"}}, "provider.url: is required"),
        ({"provider": {"type": "http", "url": "u", "headers": {"bad header": "x"}}}, "is not a valid header name"),
        (
            {"provider": {"type": "http", "url": "u", "headers_env": {"X-A": "not valid"}}},
            "must name an environment variable",
        ),
        ({"defaults": {"repeat": 0}}, "defaults.repeat: must be at least 1"),
        ({"defaults": {"concurrency": 1000}}, "defaults.concurrency: must be at most 64"),
        ({"defaults": {"timeout_s": 0}}, "defaults.timeout_s: must be at least 0.1"),
        ({"defaults": {"retries": 99}}, "defaults.retries: must be at most 10"),
        ({"graders": [{"type": "nope"}]}, "graders[0].type: unknown grader 'nope'"),
        ({"redact": {"patterns": ["(a+)+"]}}, "redact.patterns[0]: repeats a group"),
        ({"gate": {"on_inconclusive": "maybe"}}, "gate.on_inconclusive: must be one of fail, pass"),
        ({"gate": {"min_pass_rate": 2}}, "gate.min_pass_rate: must be at most 1.0"),
        ({"gate": {"alpha": 0}}, "gate.alpha: must be at least"),
    ],
)
def test_configuration_errors_name_the_key(obj: object, fragment: str) -> None:
    assert any(fragment in line for line in errors(obj)), errors(obj)


def test_load_config_and_prices_through_the_port() -> None:
    files = MemoryFiles(
        {
            "llm-eval.json": '{"dataset": "d.jsonl"}',
            "prices.json": '{"models": {"m": {"input_per_mtok": 1, "output_per_mtok": 2}}}',
            "bad.json": "{",
            "dup.json": '{"a": 1, "a": 2}',
        }
    )
    assert load_config(files, "llm-eval.json").dataset == "d.jsonl"
    assert load_prices(files, "prices.json").models["m"].output_per_mtok == 2
    assert load_prices(files, None).models == {}
    with pytest.raises(ValidationError, match="invalid JSON"):
        read_json(files, "bad.json", 1000)
    with pytest.raises(ValidationError, match="duplicate key"):
        read_json(files, "dup.json", 1000)
    with pytest.raises(UsageError):
        load_config(files, "missing.json")


def cases(*specs: tuple[str, str, tuple[str, ...]]) -> list[Case]:
    return [case(i, category=c, tags=t) for i, c, t in specs]


def test_selection_filters_and_orders_by_id() -> None:
    pool = cases(("c", "quality", ("a",)), ("a", "safety", ("a", "b")), ("b", "quality", ("b",)))
    assert [c.id for c in select_cases(pool, Selection())] == ["a", "b", "c"]
    assert [c.id for c in select_cases(pool, Selection(tags=("b",)))] == ["a", "b"]
    assert [c.id for c in select_cases(pool, Selection(categories=("safety",)))] == ["a"]
    assert [c.id for c in select_cases(pool, Selection(case_ids=("c", "a")))] == ["a", "c"]
    assert [c.id for c in select_cases(pool, Selection(tags=("a",), categories=("quality",)))] == ["c"]


def test_selection_errors() -> None:
    pool = cases(("a", "quality", ()))
    for selection, fragment in [
        (Selection(tags=("zzz",)), "no case matches the filters"),
        (Selection(categories=("security",)), "unknown category 'security'"),
        (Selection(case_ids=("nope",)), "unknown case id(s): nope"),
        (Selection(sample=0), "--sample must be at least 1"),
    ]:
        with pytest.raises(UsageError, match=fragment.replace("(", r"\(").replace(")", r"\)")):
            select_cases(pool, selection)


def test_seeded_sampling_is_reproducible_and_seed_dependent() -> None:
    pool = [case(f"c{i:02d}") for i in range(40)]
    a = [c.id for c in select_cases(pool, Selection(sample=8, seed=1))]
    assert a == [c.id for c in select_cases(pool, Selection(sample=8, seed=1))] and a == sorted(a)
    assert a != [c.id for c in select_cases(pool, Selection(sample=8, seed=2))]
    assert len(select_cases(pool, Selection(sample=100, seed=1))) == 40
    assert [c.id for c in select_cases(pool, Selection(sample=3))] == [
        c.id for c in select_cases(pool, Selection(sample=3, seed=0))
    ]


def test_jsonl_dataset_with_blank_lines_crlf_and_bom() -> None:
    line = json.dumps(CASE)
    files = MemoryFiles({"d.jsonl": f"\ufeff{line}\r\n\r\n{json.dumps({**CASE, 'id': 'b'})}\n"})
    assert [c.id for c in load_dataset(files, "d.jsonl").cases] == ["a", "b"]


def test_json_dataset_forms() -> None:
    files = MemoryFiles(
        {
            "a.json": json.dumps([CASE]),
            "b.json": json.dumps({"cases": [CASE]}),
            "c.json": '{"cases": 5}',
            "d.json": "[1,",
            "e.json": '"x"',
        }
    )
    assert len(load_dataset(files, "a.json").cases) == 1 and len(load_dataset(files, "b.json").cases) == 1
    for name, fragment in [("c.json", "got number"), ("d.json", "invalid JSON"), ("e.json", "got string")]:
        with pytest.raises(DatasetError, match=fragment):
            load_dataset(files, name)


def test_dataset_errors_carry_line_and_column() -> None:
    files = MemoryFiles({"d.jsonl": f'{json.dumps(CASE)}\n{{"id": "b", oops}}\n{{"id": "c", "id": "d"}}\n'})
    with pytest.raises(DatasetError) as error:
        load_dataset(files, "d.jsonl")
    rendered = [p.render() for p in error.value.problems]
    assert rendered[0].startswith("line 2, column ") and "duplicate key" in rendered[1]


def test_unsupported_extension_and_giant_line_and_deep_nesting() -> None:
    with pytest.raises(DatasetError, match="unsupported file type"):
        load_dataset(MemoryFiles({"d.yaml": "x"}), "d.yaml")
    giant = MemoryFiles({"d.jsonl": '{"id": "a", "input": "' + "x" * 300_000 + '"}\n'})
    with pytest.raises(DatasetError, match="limit is 262144"):
        load_dataset(giant, "d.jsonl")
    deep = MemoryFiles({"d.jsonl": '{"id": "a", "input": ' + "[" * 5000 + "]" * 5000 + "}\n"})
    with pytest.raises(DatasetError, match="deeper than 32"):
        load_dataset(deep, "d.jsonl")


def test_problem_list_from_a_hostile_file_is_capped() -> None:
    files = MemoryFiles({"d.jsonl": "{\n" * 500})
    with pytest.raises(DatasetError) as error:
        load_dataset(files, "d.jsonl")
    assert len(error.value.problems) <= 51


def test_redactor_json_and_bounds() -> None:
    token = keys.github_token()
    r = Redactor()
    assert r.json({token: [token, {"a": token}], "n": 1}) == {
        "[REDACTED:github-token]": ["[REDACTED:github-token]", {"a": "[REDACTED:github-token]"}],
        "n": 1,
    }
    deep: object = "x"
    for _ in range(100):
        deep = [deep]
    assert "[TRUNCATED]" in json.dumps(r.json(js(deep)))
    assert Redactor(False).json({"a": token}) == {"a": token} and Redactor(False).text(token) == token
    with pytest.raises(UsageError, match="catastrophic"):
        Redactor.build(True, ["(a+)+"])
    assert bounded_json({"a": "b"}, 100) == {"a": "b"}
    assert "_truncated" in bounded_json({"a": "b" * 500}, 100)


def test_split_values_accepts_commas_and_repeats() -> None:
    assert split_values(["a,b", "c", ""]) == ("a", "b", "c")
