"""The configuration file (JSON): which dataset, which provider, defaults and gate settings. See docs/configuration.md."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from llm_eval_harness.domain.dataset import parse_grader_spec
from llm_eval_harness.domain.errors import Problem, ValidationError
from llm_eval_harness.domain.grading.params import Params
from llm_eval_harness.domain.grading.regexsafe import check_pattern
from llm_eval_harness.domain.jsonio import JsonValue, kind_of
from llm_eval_harness.domain.limits import DEFAULT_MAX_OUTPUT_BYTES, HARD_MAX_OUTPUT_BYTES
from llm_eval_harness.domain.model import GraderSpec

ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")
HEADER_NAME = re.compile(r"[A-Za-z0-9-]{1,64}")
PROVIDER_TYPES = ("command", "http", "fake")
MAX_CONCURRENCY = 64
MAX_REPEAT = 1000
MAX_RETRIES = 10


@dataclass(frozen=True, slots=True)
class CommandProviderConfig:
    argv: tuple[str, ...]
    cwd: str | None
    env_allow: tuple[str, ...]
    env_set: tuple[tuple[str, str], ...]
    max_output_bytes: int


@dataclass(frozen=True, slots=True)
class HttpProviderConfig:
    url: str
    allow_hosts: tuple[str, ...]
    headers: tuple[tuple[str, str], ...]
    headers_env: tuple[tuple[str, str], ...]
    max_output_bytes: int


@dataclass(frozen=True, slots=True)
class FakeProviderConfig:
    spec: dict[str, JsonValue]


type ProviderConfig = CommandProviderConfig | HttpProviderConfig | FakeProviderConfig


@dataclass(frozen=True, slots=True)
class Defaults:
    repeat: int = 1
    concurrency: int = 4
    timeout_s: float = 30.0
    retries: int = 1
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class GateSettings:
    min_pass_rate: float | None = None
    on_inconclusive: str = "fail"
    fail_on_removed: bool = False
    tolerance: float = 0.0
    alpha: float = 0.05
    deterministic: bool = False
    max_regressed_cases: int = 0
    max_cost_increase_pct: float | None = None
    max_p95_latency_increase_pct: float | None = None


@dataclass(frozen=True, slots=True)
class Config:
    dataset: str | None
    provider: ProviderConfig | None
    price_table: str | None
    defaults: Defaults
    graders: tuple[GraderSpec, ...]
    redact: bool
    redact_patterns: tuple[str, ...]
    out_dir: str
    baseline: str | None
    gate: GateSettings = field(default_factory=GateSettings)


def parse_config(obj: JsonValue, source: str) -> Config:
    problems: list[Problem] = []
    if not isinstance(obj, dict):
        raise ValidationError(source, [Problem("", f"must be a JSON object (got {kind_of(obj)})")])
    p = Params(
        obj,
        "",
        problems,
        ("dataset", "provider", "price_table", "defaults", "graders", "redact", "out_dir", "baseline", "gate"),
    )
    provider_raw = p.obj("provider")
    provider = _provider(provider_raw, problems) if provider_raw is not None else None
    defaults = _defaults(p.obj("defaults"), problems)
    graders = _graders(p, problems)
    redact_raw = p.obj("redact") or {}
    rp = Params(redact_raw, "redact", problems, ("enabled", "patterns"))
    patterns = rp.strings("patterns", max_len=512)
    for index, pattern in enumerate(patterns):
        reason = check_pattern(pattern)
        if reason:
            rp.fail(f"patterns[{index}]", reason)
    config = Config(
        dataset=p.string("dataset", max_len=1000),
        provider=provider,
        price_table=p.string("price_table", max_len=1000),
        defaults=defaults,
        graders=graders,
        redact=rp.boolean("enabled", default=True),
        redact_patterns=patterns,
        out_dir=p.string("out_dir", default="out", max_len=1000) or "out",
        baseline=p.string("baseline", max_len=1000),
        gate=_gate(p.obj("gate"), problems),
    )
    if problems:
        raise ValidationError(source, problems)
    return config


def _graders(p: Params, problems: list[Problem]) -> tuple[GraderSpec, ...]:
    items = p.array("graders") or []
    specs: list[GraderSpec] = []
    for index, item in enumerate(items):
        spec = parse_grader_spec(item, f"graders[{index}]", problems)
        if spec is not None:
            specs.append(spec)
    return tuple(specs)


def _env_pairs(p: Params, key: str) -> tuple[tuple[str, str], ...]:
    raw = p.obj(key) or {}
    pairs: list[tuple[str, str]] = []
    for name, value in raw.items():
        if not isinstance(value, str) or len(value) > 4096:
            p.fail(f"{key}.{name}", "must be a string of at most 4096 characters")
        else:
            pairs.append((name, value))
    return tuple(pairs)


def _provider(raw: dict[str, JsonValue], problems: list[Problem]) -> ProviderConfig | None:
    kind = raw.get("type")
    if kind not in PROVIDER_TYPES:
        problems.append(Problem("provider.type", f"must be one of {', '.join(PROVIDER_TYPES)} (got {kind!r})"))
        return None
    allowed = {
        "command": ("type", "argv", "cwd", "env_allow", "env_set", "max_output_bytes"),
        "http": ("type", "url", "allow_hosts", "headers", "headers_env", "max_output_bytes"),
        "fake": ("type", "script", "default", "latency_ms", "model"),
    }[str(kind)]
    p = Params(raw, "provider", problems, allowed)
    if kind == "fake":
        return FakeProviderConfig({k: v for k, v in raw.items() if k != "type"})
    max_bytes = p.integer(
        "max_output_bytes", default=DEFAULT_MAX_OUTPUT_BYTES, minimum=1024, maximum=HARD_MAX_OUTPUT_BYTES
    )
    if kind == "command":
        argv = p.strings("argv", required=True, min_items=1, max_len=4096)
        if len(argv) > 64:
            p.fail("argv", "has too many items (limit 64)")
        env_allow = p.strings("env_allow", max_len=128)
        for index, name in enumerate(env_allow):
            if not ENV_NAME.fullmatch(name):
                p.fail(f"env_allow[{index}]", "is not a valid environment variable name")
        env_set = _env_pairs(p, "env_set")
        for name, _ in env_set:
            if not ENV_NAME.fullmatch(name):
                p.fail(f"env_set.{name}", "is not a valid environment variable name")
        return CommandProviderConfig(argv, p.string("cwd", max_len=1000), env_allow, env_set, max_bytes)
    url = p.string("url", required=True, max_len=2000) or ""
    headers = _env_pairs(p, "headers")
    headers_env = _env_pairs(p, "headers_env")
    for name, _ in (*headers, *headers_env):
        if not HEADER_NAME.fullmatch(name):
            p.fail(f"headers.{name}", "is not a valid header name")
    for name, env_name in headers_env:
        if not ENV_NAME.fullmatch(env_name):
            p.fail(f"headers_env.{name}", "must name an environment variable")
    return HttpProviderConfig(url, p.strings("allow_hosts", max_len=255), headers, headers_env, max_bytes)


def _defaults(raw: dict[str, JsonValue] | None, problems: list[Problem]) -> Defaults:
    p = Params(raw or {}, "defaults", problems, ("repeat", "concurrency", "timeout_s", "retries", "seed"))
    seed = p.number("seed", integer=True)
    return Defaults(
        repeat=p.integer("repeat", default=1, minimum=1, maximum=MAX_REPEAT),
        concurrency=p.integer("concurrency", default=4, minimum=1, maximum=MAX_CONCURRENCY),
        timeout_s=p.number("timeout_s", default=30.0, minimum=0.1, maximum=3600.0) or 30.0,
        retries=p.integer("retries", default=1, minimum=0, maximum=MAX_RETRIES),
        seed=None if seed is None else int(seed),
    )


def _gate(raw: dict[str, JsonValue] | None, problems: list[Problem]) -> GateSettings:
    p = Params(
        raw or {},
        "gate",
        problems,
        (
            "min_pass_rate",
            "on_inconclusive",
            "fail_on_removed",
            "tolerance",
            "alpha",
            "deterministic",
            "max_regressed_cases",
            "max_cost_increase_pct",
            "max_p95_latency_increase_pct",
        ),
    )
    return GateSettings(
        min_pass_rate=p.number("min_pass_rate", minimum=0.0, maximum=1.0),
        on_inconclusive=p.choice("on_inconclusive", ("fail", "pass"), default="fail"),
        fail_on_removed=p.boolean("fail_on_removed", default=False),
        tolerance=p.number("tolerance", default=0.0, minimum=0.0, maximum=1.0) or 0.0,
        alpha=p.number("alpha", default=0.05, minimum=1e-6, maximum=0.5) or 0.05,
        deterministic=p.boolean("deterministic", default=False),
        max_regressed_cases=p.integer("max_regressed_cases", default=0, minimum=0),
        max_cost_increase_pct=p.number("max_cost_increase_pct", minimum=0.0),
        max_p95_latency_increase_pct=p.number("max_p95_latency_increase_pct", minimum=0.0),
    )
