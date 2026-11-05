"""A fake pipeline endpoint on loopback, for the http adapter tests. Behaviour is chosen by the request path."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class Handler(BaseHTTPRequestHandler):
    server: FakeServer  # type: ignore[assignment]

    def log_message(self, format: str, *args: Any) -> None:  # silence
        return

    def _read_body(self) -> bytes:
        return self.rfile.read(int(self.headers.get("Content-Length", "0")))

    def do_POST(self) -> None:
        body = self._read_body()
        self.server.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
        route = self.path.split("?")[0]
        answer = {"text": "ok from fake endpoint", "usage": {"model": "m", "input_tokens": 1, "output_tokens": 2}}
        if route == "/ok":
            self._send(200, json.dumps(answer).encode())
        elif route == "/echo":
            request = json.loads(body)
            self._send(200, json.dumps({"text": "echo " + str(request["case"]["input"])}).encode())
        elif route == "/status":
            code = int(self.path.split("code=")[1])
            self._send(code, b'{"error": "no"}')
        elif route == "/redirect":
            self.send_response(302)
            self.send_header("Location", "http://169.254.169.254/latest")
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif route == "/huge":
            payload = b'{"text": "' + b"x" * 5_000_000 + b'"}'
            self._send(200, payload)
        elif route == "/endless":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            try:
                while True:
                    self.wfile.write(b"x" * 65536)
            except OSError:
                return
        elif route == "/hang":
            time.sleep(self.server.hang_seconds)
            self._send(200, json.dumps(answer).encode())
        elif route == "/drip":
            self.send_response(200)
            self.send_header("Content-Length", "1000")
            self.end_headers()
            try:
                for _ in range(1000):
                    self.wfile.write(b" ")
                    self.wfile.flush()
                    time.sleep(0.05)
            except OSError:
                return
        elif route == "/invalid":
            self._send(200, b"{nope")
        elif route == "/binary":
            self._send(200, bytes(range(256)))
        elif route == "/reset":
            self.connection.close()
        else:
            self._send(404, b"{}")

    def _send(self, code: int, payload: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class FakeServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request: Any, client_address: Any) -> None:  # clients that give up are part of the tests
        return

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), Handler)
        self.requests: list[dict[str, Any]] = []
        self.hang_seconds = 5.0

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


@contextmanager
def serve() -> Iterator[FakeServer]:
    server = FakeServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
