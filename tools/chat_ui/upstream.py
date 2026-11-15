"""The compatible llmx transport, with bounded reads and explicit cancellation."""

import http.client
import io
import json
import socket
import threading
import time
from urllib.parse import urlsplit


class Cancelled(Exception):
    pass


class _Reader(io.RawIOBase):
    def __init__(self, sock, cancelled, timeout):
        self.sock, self.cancelled, self.timeout = sock, cancelled, timeout
        sock.settimeout(0.25)

    def readable(self):
        return True

    def readinto(self, buffer):
        deadline = time.monotonic() + self.timeout
        while not self.cancelled.is_set():
            try:
                return self.sock.recv_into(buffer)
            except socket.timeout:
                if time.monotonic() >= deadline:
                    raise TimeoutError("The upstream stopped responding") from None
        raise Cancelled()


class _Response(http.client.HTTPResponse):
    def __init__(self, sock, cancelled, timeout, **kwargs):
        super().__init__(sock, **kwargs)
        self.socket_file = self.fp
        self.fp = io.BufferedReader(_Reader(sock, cancelled, timeout))

    def close(self):
        try:
            super().close()
        finally:
            self.socket_file.close()


class Connection:
    def __init__(self, base, timeout=30, key=None):
        parsed = urlsplit(base)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Upstream must be an HTTP(S) URL without credentials, query or fragment")
        self.base = parsed.path.rstrip("/")
        if not self.base.endswith("/v1"):
            self.base += "/v1"
        cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        self.connection = cls(parsed.hostname, parsed.port, timeout=timeout)
        self.key = key
        self.response = None
        self.socket = None
        self.cancelled = threading.Event()
        self.connection.response_class = lambda sock, **kwargs: _Response(sock, self.cancelled, timeout, **kwargs)

    def request(self, path, body=None):
        if self.cancelled.is_set():
            raise Cancelled()
        headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        payload = None if body is None else json.dumps(body, ensure_ascii=True, allow_nan=False).encode()
        self.connection.request("GET" if body is None else "POST", self.base + path, payload, headers)
        self.socket = self.connection.sock
        if self.cancelled.is_set():
            self.cancel()
            raise Cancelled()
        self.response = self.connection.getresponse()
        if self.cancelled.is_set():
            self.close()
            raise Cancelled()
        if not 200 <= self.response.status < 300:
            detail = self.response.read(4096).decode("utf-8", "replace")
            try:
                error = json.loads(detail).get("error", detail)
                detail = error.get("message", "Upstream request failed") if isinstance(error, dict) else str(error)
            except (ValueError, AttributeError):
                detail = "Upstream request failed"
            raise ValueError(f"llmx returned HTTP {self.response.status}: {detail[:1000]}")
        return self.response

    def json(self, path, body=None):
        try:
            response = self.request(path, body)
            data = response.read(8 * 1024 * 1024 + 1)
            if len(data) > 8 * 1024 * 1024:
                raise ValueError("Upstream response is too large")
            return json.loads(data)
        finally:
            self.close()

    def events(self, body):
        try:
            response = self.request("/chat/completions", body)
            if "text/event-stream" not in response.getheader("Content-Type", ""):
                raise ValueError("The upstream did not return a chat event stream")
            data, size = [], 0
            while True:
                if self.cancelled.is_set():
                    raise Cancelled()
                line = response.readline(2 * 1024 * 1024 + 1)
                if len(line) > 2 * 1024 * 1024:
                    raise ValueError("Upstream event is too large")
                if not line:
                    raise ValueError("The upstream disconnected before completing the reply")
                line = line.decode("utf-8").rstrip("\r\n")
                if not line:
                    if not data:
                        continue
                    text = "\n".join(data)
                    data, size = [], 0
                    if text == "[DONE]":
                        return
                    item = json.loads(text)
                    if not isinstance(item, dict):
                        raise ValueError("Invalid upstream event")
                    if "error" in item:
                        raise ValueError("The upstream reported a generation error")
                    yield item
                elif line.startswith("data:"):
                    text = line[5:].lstrip(" ")
                    size += len(text)
                    if size > 2 * 1024 * 1024:
                        raise ValueError("Upstream event is too large")
                    data.append(text)
        finally:
            self.close()

    def cancel(self):
        self.cancelled.set()
        sockets = [self.socket, self.connection.sock]
        if self.response and self.response.fp:
            sockets.append(getattr(getattr(self.response.fp, "raw", None), "_sock", None))
        for sock in sockets:
            if sock:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def close(self):
        if self.response:
            self.response.close()
        self.connection.close()
