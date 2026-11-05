from __future__ import annotations

import time

import pytest

from llm_eval_harness.application.config import HttpProviderConfig
from llm_eval_harness.domain.errors import ProviderError, UsageError
from llm_eval_harness.domain.jsonio import JsonValue
from llm_eval_harness.domain.model import Expected
from llm_eval_harness.domain.protocol import build_request
from llm_eval_harness.infrastructure.providers.http import HttpProvider, parse_endpoint
from support.builders import case
from support.fake_server import FakeServer, serve

pytestmark = pytest.mark.slow


def provider(
    server: FakeServer,
    path: str,
    *,
    max_bytes: int = 1_000_000,
    headers: tuple[tuple[str, str], ...] = (),
    headers_env: tuple[tuple[str, str], ...] = (),
    environ: dict[str, str] | None = None,
) -> HttpProvider:
    return HttpProvider(HttpProviderConfig(server.base + path, (), headers, headers_env, max_bytes), environ or {})


def request(**kw: object) -> dict[str, JsonValue]:
    return build_request(case("c1", input="hello", **kw), repeat=0, attempt=1, seed=None)  # type: ignore[arg-type]


def fails(p: HttpProvider, timeout_s: float = 5.0) -> ProviderError:
    with pytest.raises(ProviderError) as error:
        p.generate(request(), timeout_s)
    return error.value


def test_a_normal_exchange_posts_json_and_reads_the_answer() -> None:
    with serve() as server:
        out = provider(server, "/echo").generate(request(), 5.0)
        assert out.text == "echo hello" and out.latency_ms > 0
        [seen] = server.requests
        assert (
            seen["headers"]["Content-Type"] == "application/json"
            and seen["headers"]["User-Agent"] == "llm-eval-harness"
        )
        assert b'"protocol": 1' in seen["body"] and b'"id": "c1"' in seen["body"]


def test_expectations_are_not_sent() -> None:
    with serve() as server:
        c = case("c1", input="q", expected=Expected(facts=("TOP-SECRET-EXPECTATION",)))
        provider(server, "/ok").generate(build_request(c, repeat=0, attempt=1, seed=None), 5.0)
        assert b"TOP-SECRET-EXPECTATION" not in server.requests[0]["body"]


def test_headers_static_and_from_the_environment_and_never_described() -> None:
    with serve() as server:
        p = provider(
            server,
            "/ok",
            headers=(("X-Team", "evals"),),
            headers_env=(("Authorization", "MY_TOKEN"),),
            environ={"MY_TOKEN": "Bearer s3cr3t-value"},
        )
        p.generate(request(), 5.0)
        sent = server.requests[0]["headers"]
        assert sent["X-Team"] == "evals" and sent["Authorization"] == "Bearer s3cr3t-value"
        described = repr(p.describe())
        assert "s3cr3t" not in described and "Authorization" in described
        assert str(p.describe()["host"]).startswith("127.0.0.1:")


def test_a_missing_environment_variable_for_a_header_is_a_usage_error() -> None:
    with serve() as server, pytest.raises(UsageError, match="MY_TOKEN is not set"):
        provider(server, "/ok", headers_env=(("Authorization", "MY_TOKEN"),), environ={})


def test_header_values_with_line_breaks_are_refused() -> None:
    with serve() as server, pytest.raises(UsageError, match="line breaks"):
        provider(server, "/ok", headers=(("X-A", "a\r\nX-Injected: 1"),))


@pytest.mark.parametrize(
    ("code", "kind", "retryable"),
    [
        (500, "network", True),
        (503, "network", True),
        (429, "network", True),
        (404, "protocol", False),
        (400, "protocol", False),
        (401, "protocol", False),
    ],
)
def test_http_status_classification(code: int, kind: str, retryable: bool) -> None:
    with serve() as server:
        error = fails(provider(server, f"/status?code={code}"))
        assert error.kind == kind and error.retryable is retryable and f"HTTP {code}" in str(error)


def test_redirects_are_never_followed() -> None:
    with serve() as server:
        error = fails(provider(server, "/redirect"))
        assert error.kind == "protocol" and "redirects are not followed" in str(error)
        assert [r["path"] for r in server.requests] == ["/redirect"]


def test_oversized_and_endless_bodies_are_cut_at_the_cap() -> None:
    with serve() as server:
        for path in ("/huge", "/endless"):
            started = time.monotonic()
            error = fails(provider(server, path, max_bytes=4096))
            assert error.kind == "protocol" and "more than 4096 bytes" in str(error) and not error.retryable
            assert time.monotonic() - started < 5


def test_a_slow_endpoint_times_out_and_is_retryable() -> None:
    with serve() as server:
        server.hang_seconds = 3.0
        started = time.monotonic()
        error = fails(provider(server, "/hang"), timeout_s=0.4)
        assert error.kind == "timeout" and error.retryable and time.monotonic() - started < 2.5


def test_a_dripping_body_hits_the_overall_deadline_not_a_per_read_one() -> None:
    with serve() as server:
        started = time.monotonic()
        error = fails(provider(server, "/drip"), timeout_s=0.6)
        assert error.kind == "timeout" and time.monotonic() - started < 3


def test_garbage_bodies_are_protocol_errors() -> None:
    with serve() as server:
        assert "not valid JSON" in str(fails(provider(server, "/invalid")))
        assert "not UTF-8" in str(fails(provider(server, "/binary")))


def test_connection_problems_are_retryable_network_errors() -> None:
    with serve() as server:
        reset = fails(provider(server, "/reset"))
        assert reset.kind == "network" and reset.retryable
        dead = HttpProvider(HttpProviderConfig("http://127.0.0.1:9/x", (), (), (), 1000), {})
        refused = fails(dead)
        assert refused.kind == "network" and refused.retryable and "ConnectionRefusedError" in str(refused)


# --- the host policy ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8080/run",
        "http://localhost:9000/x",
        "http://127.9.9.9/x",
        "http://[::1]:8000/x",
        "https://localhost/x",
    ],
)
def test_loopback_hosts_are_allowed(url: str) -> None:
    endpoint = parse_endpoint(url, ())
    assert endpoint.connect_host in ("127.0.0.1", "127.9.9.9", "::1")


@pytest.mark.parametrize(
    ("url", "fragment"),
    [
        ("http://example.com/x", "neither on loopback nor listed in allow_hosts"),
        ("http://169.254.169.254/latest/meta-data", "neither on loopback"),
        ("http://10.0.0.5/x", "neither on loopback"),
        ("http://localhost.evil.example/x", "neither on loopback"),
        ("http://127.0.0.1.evil.example/x", "neither on loopback"),
        ("http://127.0.0.1@evil.example/x", "credentials in the URL are refused"),
        ("http://user:pw@127.0.0.1/x", "credentials in the URL are refused"),
        ("ftp://127.0.0.1/x", "scheme must be http or https"),
        ("file:///etc/passwd", "scheme must be http or https"),
        ("http://127.0.0.1/x#frag", "fragment"),
        ("http:///x", "has no host"),
        ("http://127.0.0.1:notaport/x", "not a valid URL"),
        ("//127.0.0.1/x", "scheme must be http or https"),
    ],
)
def test_everything_else_is_refused(url: str, fragment: str) -> None:
    with pytest.raises(UsageError) as error:
        parse_endpoint(url, ())
    assert fragment in str(error.value)


def test_an_allow_listed_host_must_use_https() -> None:
    assert parse_endpoint("https://api.example.org/run?x=1", ("api.example.org",)).target == "/run?x=1"
    assert parse_endpoint("https://API.example.org:8443/run", ("api.example.org:8443",)).port == 8443
    with pytest.raises(UsageError, match="must use https"):
        parse_endpoint("http://api.example.org/run", ("api.example.org",))
    with pytest.raises(UsageError, match="neither on loopback"):
        parse_endpoint("https://other.example.org/run", ("api.example.org",))
    with pytest.raises(UsageError, match="neither on loopback"):
        parse_endpoint("https://api.example.org:8443/run", ("api.example.org:9999",))


def test_the_allow_list_is_exact_not_a_suffix_match() -> None:
    with pytest.raises(UsageError):
        parse_endpoint("https://evil-api.example.org/x", ("api.example.org",))
    with pytest.raises(UsageError):
        parse_endpoint("https://api.example.org.evil.net/x", ("api.example.org",))
