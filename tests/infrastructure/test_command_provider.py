from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from llm_eval_harness.application.config import CommandProviderConfig
from llm_eval_harness.domain.errors import ProviderError
from llm_eval_harness.domain.grading.judge import JudgeConfig
from llm_eval_harness.domain.jsonio import JsonValue
from llm_eval_harness.domain.model import Case
from llm_eval_harness.domain.protocol import build_request
from llm_eval_harness.infrastructure.providers.command import CommandProvider, build_env
from llm_eval_harness.infrastructure.providers.judge import CommandJudge
from support import programs
from support.builders import case
from support.programs import write_program

pytestmark = pytest.mark.slow
PATH = os.environ.get("PATH", "/usr/bin:/bin")
ENVIRON = {
    "PATH": PATH,
    "LANG": "C.UTF-8",
    "SENTINEL_SECRET_TOKEN": "do-not-leak-me",
    "HOME": "/home/x",
    "ALLOWED_VAR": "yes",
}


def provider(
    tmp_path: Path,
    source: str,
    *,
    extra: list[str] | None = None,
    allow: tuple[str, ...] = (),
    fixed: tuple[tuple[str, str], ...] = (),
    max_bytes: int = 1_000_000,
    cwd: str | None = None,
) -> CommandProvider:
    argv = [*write_program(tmp_path, "prog", source), *(extra or [])]
    return CommandProvider(CommandProviderConfig(tuple(argv), cwd, allow, fixed, max_bytes), tmp_path, ENVIRON)


def request(the_case: Case | None = None) -> dict[str, JsonValue]:
    return build_request(the_case or case("c1", input="hello"), repeat=0, attempt=1, seed=None)


def fails(p: CommandProvider, timeout_s: float = 20.0) -> ProviderError:
    with pytest.raises(ProviderError) as error:
        p.generate(request(), timeout_s)
    return error.value


def test_a_well_behaved_program_is_called_with_the_protocol_request(tmp_path: Path) -> None:
    out = provider(tmp_path, programs.ECHO).generate(request(), 20.0)
    assert out.text == "echo: hello" and out.usage is not None and out.usage.total_tokens == 7
    assert out.latency_ms > 0 and not out.tool_calls


def test_describe_names_the_program_but_not_its_arguments(tmp_path: Path) -> None:
    p = provider(tmp_path, programs.ECHO, extra=["--api-key=hunter2"])
    assert p.describe() == {"type": "command", "program": sys.executable.rsplit("/", 1)[-1], "arguments": 2}
    assert "hunter2" not in repr(p.describe())


def test_the_child_environment_is_an_explicit_allow_list(tmp_path: Path) -> None:
    seen = json.loads(provider(tmp_path, programs.ENV_DUMP).generate(request(), 20.0).text)
    assert "SENTINEL_SECRET_TOKEN" not in seen and "HOME" not in seen and "ALLOWED_VAR" not in seen
    assert seen["PATH"] == PATH and seen["LLM_EVAL"] == "1" and seen["LANG"] == "C.UTF-8"
    seen = json.loads(
        provider(tmp_path, programs.ENV_DUMP, allow=("ALLOWED_VAR", "NOT_SET_ANYWHERE"), fixed=(("MODE", "strict"),))
        .generate(request(), 20.0)
        .text
    )
    assert seen["ALLOWED_VAR"] == "yes" and seen["MODE"] == "strict" and "SENTINEL_SECRET_TOKEN" not in seen
    assert "NOT_SET_ANYWHERE" not in seen


def test_build_env_never_passes_more_than_asked() -> None:
    env = build_env({"A": "1", "B": "2", "PATH": "/bin"}, ("A",), (("C", "3"),))
    assert env == {"PATH": "/bin", "A": "1", "C": "3", "LLM_EVAL": "1"}
    assert build_env({}, (), ()) == {"LLM_EVAL": "1"}


def test_working_directory_inside_the_project(tmp_path: Path) -> None:
    (tmp_path / "work").mkdir()
    out = provider(tmp_path, programs.CWD_DUMP, cwd="work").generate(request(), 20.0)
    assert os.path.realpath(out.text) == os.path.realpath(tmp_path / "work")
    from llm_eval_harness.domain.errors import UsageError

    with pytest.raises(UsageError, match="outside the project"):
        provider(tmp_path, programs.ECHO, cwd="../..")
    with pytest.raises(UsageError, match="no such file"):
        provider(tmp_path, programs.ECHO, cwd="missing")


@pytest.mark.parametrize(
    ("source", "kind", "fragment", "retryable"),
    [
        (programs.INVALID_JSON, "protocol", "not valid JSON", False),
        (programs.NOT_AN_OBJECT, "protocol", "must be a JSON object (got array)", False),
        (programs.EMPTY, "protocol", "produced no output", False),
        (programs.BINARY, "protocol", "not UTF-8 text", False),
        (programs.EXIT_NONZERO, "exit", "exited with status 7: boom: something went wrong", True),
        (programs.SELF_KILL, "exit", "killed by signal SIGKILL", True),
        (programs.DEEP_JSON, "protocol", "deeper than 32", False),
        (programs.DUPLICATE_KEYS, "protocol", "duplicate key", False),
        (programs.NAN, "protocol", "NaN is not valid JSON", False),
        (programs.UNKNOWN_KEY, "protocol", "unknown key 'tool_call'", False),
    ],
)
def test_hostile_programs_become_precise_provider_errors(
    tmp_path: Path, source: str, kind: str, fragment: str, retryable: bool
) -> None:
    error = fails(provider(tmp_path, source))
    assert error.kind == kind and error.retryable is retryable and fragment in str(error), str(error)
    assert "\x1b" not in str(error)


def test_a_program_that_prints_endlessly_is_stopped_at_the_cap(tmp_path: Path) -> None:
    error = fails(provider(tmp_path, programs.HUGE_STDOUT, max_bytes=2048))
    assert error.kind == "protocol" and "more than 2048 bytes" in str(error) and not error.retryable


def test_a_hung_program_times_out_and_is_retryable(tmp_path: Path) -> None:
    error = fails(provider(tmp_path, programs.HANG), timeout_s=0.5)
    assert error.kind == "timeout" and error.retryable and "within 0.5s" in str(error)


def test_a_missing_program_is_a_non_retryable_spawn_error(tmp_path: Path) -> None:
    p = CommandProvider(CommandProviderConfig(("/definitely/not/here", "x"), None, (), (), 10_000), tmp_path, ENVIRON)
    error = fails(p)
    assert error.kind == "spawn" and not error.retryable and "'here'" in str(error)


def test_control_characters_in_the_answer_are_data_not_instructions(tmp_path: Path) -> None:
    out = provider(tmp_path, programs.CONTROL_TEXT).generate(request(), 20.0)
    assert "\x1b[31m" in out.text and "\u202e" in out.text  # kept as data; sanitising happens at output time


def test_prototype_pollution_keys_in_structured_output_are_plain_data(tmp_path: Path) -> None:
    out = provider(tmp_path, programs.PROTO_KEYS).generate(request(), 20.0)
    assert out.structured == {"__proto__": {"admin": True}, "constructor": {"prototype": 1}}


def test_the_answer_key_is_never_sent_to_the_program(tmp_path: Path) -> None:
    from llm_eval_harness.domain.model import Expected

    script = "import json,sys\nprint(json.dumps({'text': sys.stdin.read()}))\n"
    c = case("c1", input="q", expected=Expected(facts=("TOP-SECRET-EXPECTATION",)))
    out = provider(tmp_path, script).generate(request(c), 20.0)
    assert "TOP-SECRET-EXPECTATION" not in out.text and '"id": "c1"' in out.text


def test_model_output_is_never_executed(tmp_path: Path) -> None:
    marker = tmp_path / "executed"
    script = f"import json\nprint(json.dumps({{'text': '$(touch {marker}); `touch {marker}`; __import__(\"os\").system(\"touch {marker}\")'}}))\n"
    out = provider(tmp_path, script).generate(request(), 20.0)
    assert "touch" in out.text and not marker.exists()


def test_slow_but_in_time_is_fine(tmp_path: Path) -> None:
    assert provider(tmp_path, programs.SLOW_THEN_OK).generate(request(), 20.0).text == "slow but fine"


def test_judge_program(tmp_path: Path) -> None:
    argv = tuple(write_program(tmp_path, "judge", programs.JUDGE_PASS))
    judge = CommandJudge(tmp_path, ENVIRON)
    config = JudgeConfig(argv, 20.0, (), "be strict", 0.7, "fail")
    answer = judge.run(config, {"rubric": "be strict"})
    assert answer == {"verdict": "pass", "confidence": 0.8, "reason": "rubric: be strict"}


def test_judge_failures_are_provider_errors(tmp_path: Path) -> None:
    judge = CommandJudge(tmp_path, ENVIRON)
    for source, kind in [
        (programs.HANG, "timeout"),
        (programs.INVALID_JSON, "protocol"),
        (programs.EXIT_NONZERO, "exit"),
    ]:
        config = JudgeConfig(tuple(write_program(tmp_path, "j", source)), 0.5, (), "", 0.7, "fail")
        with pytest.raises(ProviderError) as error:
            judge.run(config, {})
        assert error.value.kind == kind


def test_judge_environment_is_an_allow_list_too(tmp_path: Path) -> None:
    config = JudgeConfig(
        tuple(write_program(tmp_path, "env", programs.ENV_DUMP)), 20.0, ("ALLOWED_VAR",), "", 0.7, "fail"
    )
    answer = CommandJudge(tmp_path, ENVIRON).run(config, {})
    assert isinstance(answer, dict)
    seen = json.loads(str(answer["text"]))
    assert seen["ALLOWED_VAR"] == "yes" and "SENTINEL_SECRET_TOKEN" not in seen and "HOME" not in seen
