"""The ``http`` provider: POST the request to the user's endpoint.

The only module with network access. The endpoint must be on the loopback interface or be named explicitly in
``allow_hosts`` (and then must use https). Redirects are never followed, credentials in the URL are refused,
the whole exchange has one deadline, and the response is read up to a fixed number of bytes.
"""

from __future__ import annotations

import http.client
import ipaddress
import json
import ssl
import time
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit

from llm_eval_harness.application.config import HttpProviderConfig
from llm_eval_harness.domain.errors import ProviderError, UsageError
from llm_eval_harness.domain.jsonio import JsonValue
from llm_eval_harness.domain.model import ProviderOutput
from llm_eval_harness.infrastructure.providers.common import decode_answer

USER_AGENT = "llm-eval-harness"
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


@dataclass(frozen=True, slots=True)
class Endpoint:
    scheme: str
    host: str
    port: int
    target: str
    connect_host: str


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def parse_endpoint(url: str, allow_hosts: tuple[str, ...]) -> Endpoint:
    """Check the URL against the host policy and split it. Raises ``UsageError`` for anything not allowed."""
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        raise UsageError("provider.url: is not a valid URL") from None
    if parts.scheme not in ("http", "https"):
        raise UsageError("provider.url: the scheme must be http or https")
    if parts.username is not None or parts.password is not None:
        raise UsageError("provider.url: credentials in the URL are refused; use headers_env")
    if parts.fragment:
        raise UsageError("provider.url: a fragment makes no sense in a request URL")
    host = (parts.hostname or "").lower()
    if not host:
        raise UsageError("provider.url: has no host")
    effective = port if port is not None else (443 if parts.scheme == "https" else 80)
    allowed = {a.lower() for a in allow_hosts}
    if _is_loopback(host):
        connect_host = "127.0.0.1" if host == "localhost" else host
    elif host in allowed or f"{host}:{effective}" in allowed:
        if parts.scheme != "https":
            raise UsageError(f"provider.url: {host} is allow-listed but not on loopback, so it must use https")
        connect_host = host
    else:
        raise UsageError(
            f"provider.url: host {host!r} is neither on loopback nor listed in allow_hosts; "
            "this tool only talks to hosts you name explicitly"
        )
    target = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
    return Endpoint(parts.scheme, host, effective, target, connect_host)


class HttpProvider:
    def __init__(self, config: HttpProviderConfig, environ: Mapping[str, str]) -> None:
        self._endpoint = parse_endpoint(config.url, config.allow_hosts)
        self._max = config.max_output_bytes
        headers = dict(config.headers)
        for name, variable in config.headers_env:
            if variable not in environ:
                raise UsageError(f"provider.headers_env.{name}: environment variable {variable} is not set")
            headers[name] = environ[variable]
        for name, value in headers.items():
            if "\r" in value or "\n" in value or "\r" in name or "\n" in name:
                raise UsageError(f"provider.headers.{name}: header values must not contain line breaks")
        self._header_names = tuple(sorted(headers))
        self._headers = headers

    def describe(self) -> dict[str, JsonValue]:
        e = self._endpoint
        return {"type": "http", "host": f"{e.host}:{e.port}", "scheme": e.scheme, "headers": list(self._header_names)}

    def generate(self, request: dict[str, JsonValue], timeout_s: float) -> ProviderOutput:
        e = self._endpoint
        deadline = time.monotonic() + timeout_s
        body = json.dumps(request, ensure_ascii=True, allow_nan=False).encode("ascii")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
            "Connection": "close",
            **self._headers,
        }
        started = time.monotonic()
        connection: http.client.HTTPConnection
        if e.scheme == "https":
            connection = http.client.HTTPSConnection(
                e.connect_host, e.port, timeout=timeout_s, context=ssl.create_default_context()
            )
        else:
            connection = http.client.HTTPConnection(e.connect_host, e.port, timeout=timeout_s)
        response: http.client.HTTPResponse | None = None
        try:
            connection.request("POST", e.target, body=body, headers=headers)
            response = connection.getresponse()
            status = response.status
            raw = self._read(connection, response, deadline)
        except TimeoutError:
            raise ProviderError(
                "timeout", f"no answer from the endpoint within {timeout_s:g}s", retryable=True
            ) from None
        except (OSError, http.client.HTTPException) as error:
            raise ProviderError("network", f"request failed: {type(error).__name__}", retryable=True) from None
        finally:
            if response is not None:
                response.close()  # a `Connection: close` answer keeps its own handle on the socket
            connection.close()
        if raw is None:
            raise ProviderError("protocol", f"the endpoint answered with more than {self._max} bytes", retryable=False)
        if status in RETRYABLE_STATUS:
            raise ProviderError("network", f"the endpoint answered HTTP {status}", retryable=True)
        if 300 <= status < 400:
            raise ProviderError(
                "protocol", f"the endpoint answered HTTP {status}; redirects are not followed", retryable=False
            )
        if status != 200:
            raise ProviderError("protocol", f"the endpoint answered HTTP {status}", retryable=False)
        return decode_answer(raw, source="the endpoint", latency_ms=(time.monotonic() - started) * 1000.0)

    def _read(
        self, connection: http.client.HTTPConnection, response: http.client.HTTPResponse, deadline: float
    ) -> bytes | None:
        """The body, or ``None`` if it is longer than the cap. One deadline for the whole body."""
        data = bytearray()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            if connection.sock is not None:
                connection.sock.settimeout(remaining)
            chunk = response.read1(min(65536, self._max + 1 - len(data)))
            if not chunk:
                return bytes(data)
            data += chunk
            if len(data) > self._max:
                return None
