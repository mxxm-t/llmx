#!/usr/bin/env python3
"""Small read-only dashboard collector for llmx /v1/health."""

import argparse
import collections
import gzip
import http.client
import json
import math
import os
import signal
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_BODY = 1024 * 1024
MAX_SAMPLES = 3601
MAX_EVENTS = 128


def _reject_constant(value):
    raise ValueError("non-finite JSON constant")


def _finite_tree(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite_tree(v) for v in value.values())
    if isinstance(value, list):
        return all(_finite_tree(v) for v in value)
    return True


def _decode_health(data):
    if len(data) > MAX_BODY:
        raise ValueError("response exceeds 1 MiB")
    try:
        obj = json.loads(data.decode("utf-8"), parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid JSON response") from exc
    if not isinstance(obj, dict):
        raise ValueError("health response must be an object")
    if not _finite_tree(obj):
        raise ValueError("health response contains a non-finite number")
    return obj


def _nonnegative_integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _nonnegative_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0


def parse_health_response(data, http_status=200):
    obj = _decode_health(data)
    if not isinstance(obj.get("server"), dict) or not isinstance(obj.get("requests"), dict):
        raise ValueError("health response needs server and requests objects")
    if http_status == 200 and obj.get("status") == "ok":
        return obj
    if http_status != 503 or obj.get("status") not in ("starting", "failed"):
        raise ValueError("health response status and body do not match")
    startup = obj.get("startup")
    if not isinstance(startup, dict):
        raise ValueError("startup health response needs a startup object")
    if (not _nonnegative_integer(startup.get("started_unix")) or
            not _nonnegative_number(startup.get("elapsed_s")) or
            not isinstance(startup.get("phase"), str) or
            not isinstance(startup.get("phases"), list)):
        raise ValueError("startup health response has invalid startup fields")
    if obj["status"] == "failed" and not isinstance(startup.get("error"), str):
        raise ValueError("failed startup response needs an error string")
    for phase in startup["phases"]:
        if (not isinstance(phase, dict) or not isinstance(phase.get("name"), str) or
                phase.get("state") not in ("running", "done", "failed") or
                not _nonnegative_number(phase.get("seconds"))):
            raise ValueError("startup health response has an invalid phase")
        byte_fields = ("bytes_done", "bytes_total", "bytes_per_s")
        if any(key in phase for key in byte_fields):
            if not all(_nonnegative_integer(phase.get(key)) for key in byte_fields):
                raise ValueError("startup phase has invalid byte progress")
            files = phase.get("files")
            if not isinstance(files, list):
                raise ValueError("startup phase byte progress needs a files array")
            for item in files:
                if (not isinstance(item, dict) or not isinstance(item.get("name"), str) or
                        not _nonnegative_integer(item.get("bytes_done")) or
                        not _nonnegative_integer(item.get("bytes_total"))):
                    raise ValueError("startup phase has an invalid file progress entry")
    return obj


def parse_health(data):
    """Validate a ready HTTP 200 health response (kept for existing callers)."""
    return parse_health_response(data, 200)


def flatten_metrics(obj):
    result = {}

    def visit(value, path):
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, path + (str(key),))
        elif isinstance(value, bool) or isinstance(value, (int, float)):
            result[".".join(path)] = value

    visit(obj, ())
    return result


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Collector:
    def __init__(self, url, interval_s=2.0, retention_s=3600.0, fetch=None, clock=time.time):
        self.url = url
        self.interval_s = interval_s
        self.retention_s = retention_s
        self.fetch = fetch or self._fetch
        self.clock = clock
        self.collector_id = str(uuid.uuid4())
        self.lock = threading.Lock()
        self.samples = collections.deque()
        self.events = collections.deque(maxlen=MAX_EVENTS)
        self.seq = 0
        self.latest = None
        self.startup = None
        self.startup_observed_at = None
        self.startup_seen_since_ready = False
        self.last_success = None
        self.error = None
        self.identity = None
        self.was_down = False
        self.stopped = threading.Event()
        self.thread = None

    def _fetch(self):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect)
        request = urllib.request.Request(self.url, headers={"Accept": "application/json"}, method="GET")
        try:
            with opener.open(request, timeout=3.0) as response:
                raw = response.read(MAX_BODY + 1)
                if response.status != 200:
                    raise ValueError("upstream returned HTTP " + str(response.status))
                return 200, raw
        except urllib.error.HTTPError as exc:
            if exc.code != 503:
                raise ValueError("upstream returned HTTP " + str(exc.code)) from exc
            with exc:
                return 503, exc.read(MAX_BODY + 1)
        except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
            raise ValueError("upstream unavailable") from exc

    def poll_once(self):
        try:
            response = self.fetch()
            if isinstance(response, tuple) and len(response) == 2:
                http_status, body = response
                if isinstance(body, dict):
                    body = json.dumps(body, allow_nan=False).encode("utf-8")
                health = parse_health_response(body, http_status)
            else:
                if isinstance(response, dict):
                    response = json.dumps(response, allow_nan=False).encode("utf-8")
                health = parse_health(response)
                http_status = 200
        except Exception as exc:
            now = self.clock()
            with self.lock:
                self.startup = None
                self.startup_observed_at = None
                self.error = str(exc)[:240] or "upstream unavailable"
                if not self.was_down:
                    self.events.append({"t": now, "kind": "disconnect", "message": "Health source disconnected"})
                self.was_down = True
            return False

        now = self.clock()
        if http_status == 503:
            with self.lock:
                if self.was_down:
                    self.events.append({"t": now, "kind": "recovery", "message": "Health source recovered"})
                self.startup = health
                self.startup_observed_at = now
                self.startup_seen_since_ready = True
                self.error = None
                self.was_down = False
            return True

        server = health["server"]
        current_identity = tuple(server.get(k) for k in ("model", "version", "numerics"))
        uptime = server.get("uptime_s")
        with self.lock:
            reset = self.startup_seen_since_ready
            if self.identity is not None:
                old_uptime = self.latest.get("server", {}).get("uptime_s")
                reset = reset or current_identity != self.identity or (
                    isinstance(uptime, (int, float)) and isinstance(old_uptime, (int, float)) and uptime < old_uptime
                )
            if self.was_down:
                self.events.append({"t": now, "kind": "recovery", "message": "Health source recovered"})
            if reset and self.identity is not None:
                self.events.append({"t": now, "kind": "restart", "message": "Server identity or uptime changed"})
            self.seq += 1
            self.samples.append({"seq": self.seq, "t": now, "reset": bool(reset), "metrics": flatten_metrics(health)})
            self.latest = health
            self.startup = None
            self.startup_observed_at = None
            self.startup_seen_since_ready = False
            self.last_success = now
            self.identity = current_identity
            self.error = None
            self.was_down = False
            self._prune_locked(now)
        return True

    def _prune_locked(self, now):
        cutoff = now - self.retention_s
        while self.samples and (self.samples[0]["t"] < cutoff or len(self.samples) > min(MAX_SAMPLES, int(self.retention_s) + 1)):
            self.samples.popleft()
        while self.events and self.events[0]["t"] < cutoff:
            self.events.popleft()

    def start(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self._run, name="health-sampler", daemon=True)
            self.thread.start()

    def _run(self):
        while not self.stopped.is_set():
            self.poll_once()
            self.stopped.wait(self.interval_s)

    def stop(self):
        self.stopped.set()
        if self.thread is not None:
            self.thread.join(timeout=4.0)

    def snapshot(self, after=0):
        now = self.clock()
        with self.lock:
            self._prune_locked(now)
            return {
                "collector_id": self.collector_id,
                "ok": self.error is None and self.latest is not None and self.startup is None,
                "error": self.error,
                "startup": self.startup,
                "startup_observed_at": self.startup_observed_at,
                "interval_s": self.interval_s,
                "retention_s": self.retention_s,
                "last_success": self.last_success,
                "latest": self.latest,
                "samples": [s for s in self.samples if s["seq"] > after],
                "events": list(self.events),
            }


def _positive_float(text, label, low, high):
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(label + " must be a number") from exc
    if not math.isfinite(value) or not low <= value <= high:
        raise argparse.ArgumentTypeError(label + " must be between " + str(low) + " and " + str(high))
    return value


def make_handler(collector, html_path):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self):
            self.request.settimeout(5.0)
            super().setup()

        def log_message(self, fmt, *args):
            pass

        def _send(self, status, body, content_type):
            accept = self.headers.get("Accept-Encoding", "")
            gzip_ok = False
            for part in accept.split(","):
                fields = [field.strip() for field in part.split(";")]
                if fields[0].lower() != "gzip":
                    continue
                try:
                    quality = next((float(field[2:]) for field in fields[1:] if field.lower().startswith("q=")), 1.0)
                except ValueError:
                    quality = 0.0
                gzip_ok = math.isfinite(quality) and quality > 0.0
                break
            if gzip_ok and body:
                body = gzip.compress(body, compresslevel=1)
                self.send_response(status)
                self.send_header("Content-Encoding", "gzip")
                self.send_header("Vary", "Accept-Encoding")
            else:
                self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def do_HEAD(self):
            self._handle()

        def do_GET(self):
            self._handle()

        def _handle(self):
            try:
                parsed = urllib.parse.urlsplit(self.path)
            except ValueError:
                self._send(400, b"Bad request", "text/plain; charset=utf-8")
                return
            if parsed.scheme or parsed.netloc or parsed.fragment:
                self._send(400, b"Bad request", "text/plain; charset=utf-8")
                return
            if parsed.path == "/" and not parsed.query:
                try:
                    with open(html_path, "rb") as stream:
                        body = stream.read()
                except OSError:
                    self._send(404, b"Not found", "text/plain; charset=utf-8")
                    return
                self._send(200, body, "text/html; charset=utf-8")
                return
            if parsed.path == "/api/history" and not parsed.fragment:
                try:
                    query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True) if parsed.query else {}
                except ValueError:
                    self._send(400, b'{"error":"invalid cursor"}', "application/json")
                    return
                if set(query) - {"after"} or len(query.get("after", [])) > 1:
                    self._send(400, b'{"error":"invalid cursor"}', "application/json")
                    return
                after = 0
                if "after" in query:
                    raw = query["after"][0]
                    if not raw or len(raw) > 20 or any(char < "0" or char > "9" for char in raw):
                        self._send(400, b'{"error":"invalid cursor"}', "application/json")
                        return
                    after = int(raw)
                payload = json.dumps(collector.snapshot(after), ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("utf-8")
                self._send(200, payload, "application/json; charset=utf-8")
                return
            self._send(404, b"Not found", "text/plain; charset=utf-8")

        def do_POST(self):
            self._send(405, b"Method not allowed", "text/plain; charset=utf-8")

        def do_PUT(self):
            self._send(405, b"Method not allowed", "text/plain; charset=utf-8")

        def do_DELETE(self):
            self._send(405, b"Method not allowed", "text/plain; charset=utf-8")

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description="Serve a read-only dashboard for an llmx health endpoint")
    parser.add_argument("--upstream", default="http://127.0.0.1:8090/v1/health")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8182)
    parser.add_argument("--interval", type=lambda s: _positive_float(s, "interval", 1.0, 60.0), default=2.0)
    parser.add_argument("--history", type=lambda s: _positive_float(s, "history", 1.0, 3600.0), default=3600.0)
    args = parser.parse_args(argv)
    try:
        parsed = urllib.parse.urlsplit(args.upstream)
    except ValueError:
        parser.error("--upstream must be an absolute HTTP(S) URL without credentials")
    if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.username or parsed.password or parsed.path == "":
        parser.error("--upstream must be an absolute HTTP(S) URL without credentials")
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    collector = Collector(args.upstream, args.interval, args.history)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(collector, os.path.join(os.path.dirname(__file__), "index.html")))
    server.daemon_threads = True

    def stop_signal(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_signal)
    try:
        collector.start()
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        collector.stop()


if __name__ == "__main__":
    main()
