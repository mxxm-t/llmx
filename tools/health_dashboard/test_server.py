import http.client
import gzip
import io
import json
import os
import threading
import tempfile
import time
import unittest
from unittest import mock
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import server


def health(uptime=10, model="demo", version="v1", requests=3):
    return {
        "status": "ok",
        "server": {"uptime_s": uptime, "model": model, "version": version, "numerics": "abc", "devices": ["cpu"]},
        "requests": {"now": {"active": requests, "queued": 0}, "since_start": {"finished": requests}},
        "precision": {"effective": "f32"},
    }


def startup_health(status="starting", uptime=1):
    return {
        "status": status,
        "server": {"uptime_s": uptime, "model": "demo", "version": "v1", "numerics": "abc"},
        "requests": {"now": {"active": 0, "queued": 0, "paused": 0, "kv_tokens": 0}},
        "startup": {
            "started_unix": 1000, "elapsed_s": 2.5, "phase": "weights",
            **({"error": "bad model"} if status == "failed" else {}),
            "phases": [
                {"name": "metadata", "state": "done", "seconds": 0.2},
                {"name": "weights", "state": "running", "seconds": 2.3,
                 "bytes_done": 50, "bytes_total": 100, "bytes_per_s": 21,
                 "files": [{"name": "part-1.gguf", "bytes_done": 50, "bytes_total": 100}]},
            ],
        },
    }


class HealthTests(unittest.TestCase):
    def test_schema_and_nonfinite_rejection(self):
        for raw in (
            b"[]",
            b'{"status":"bad","server":{},"requests":{}}',
            b'{"status":"ok","server":{},"requests":[],"x":NaN}',
            b'{"status":"ok","server":{},"requests":{},"x":1e999}',
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                server.parse_health(raw)
        self.assertEqual(server.parse_health(json.dumps(health()).encode())["status"], "ok")
        self.assertEqual(server.parse_health_response(json.dumps(startup_health()).encode(), 503)["status"], "starting")
        for status, raw in (
            (200, json.dumps(startup_health()).encode()),
            (503, b'{"status":"bad","server":{},"requests":{},"startup":{}}'),
            (503, json.dumps({**startup_health("failed"), "startup": {"error": "no phases"}}).encode()),
            (503, json.dumps({**startup_health(), "startup": {**startup_health()["startup"], "phases": [{"name": "x", "state": "running", "seconds": 1, "bytes_done": 1}]}}).encode()),
        ):
            with self.subTest(status=status, raw=raw), self.assertRaises(ValueError):
                server.parse_health_response(raw, status)
        with self.assertRaisesRegex(ValueError, "exceeds 1 MiB"):
            server.parse_health(b" " * (server.MAX_BODY + 1))

    def test_failures_preserve_last_success_and_recovery_event(self):
        answers = [health(), ValueError("offline"), health(requests=4)]

        def fetch():
            item = answers.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        collector = server.Collector("http://unused", fetch=fetch)
        self.assertTrue(collector.poll_once())
        initial = collector.snapshot()
        self.assertFalse(collector.poll_once())
        failed = collector.snapshot()
        self.assertFalse(failed["ok"])
        self.assertEqual(failed["latest"], initial["latest"])
        self.assertEqual(failed["last_success"], initial["last_success"])
        self.assertEqual(failed["error"], "offline")
        self.assertIsNone(failed["startup"])
        self.assertTrue(collector.poll_once())
        recovered = collector.snapshot()
        self.assertTrue(recovered["ok"])
        self.assertIsNone(recovered["error"])
        self.assertEqual([event["kind"] for event in recovered["events"]], ["disconnect", "recovery"])

    def test_reset_and_bounded_history(self):
        values = [health(10), health(11), health(2), health(3, model="other")]
        now = [100.0]

        def fetch():
            return values.pop(0)

        def clock():
            now[0] += 0.4
            return now[0]

        collector = server.Collector("unused", retention_s=1.0, fetch=fetch, clock=clock)
        collector.poll_once()
        collector.poll_once()
        collector.poll_once()
        collector.poll_once()
        snapshot = collector.snapshot()
        self.assertLessEqual(len(snapshot["samples"]), 2)
        self.assertTrue(any(sample["reset"] for sample in snapshot["samples"]))
        self.assertEqual([event["kind"] for event in snapshot["events"]], ["restart", "restart"])

    def test_bad_body_and_redirect_refused(self):
        class Upstream(BaseHTTPRequestHandler):
            mode = "bad"

            def log_message(self, fmt, *args):
                pass

            def do_GET(self):
                if self.mode == "redirect":
                    self.send_response(302)
                    self.send_header("Location", "/valid")
                    self.end_headers()
                elif self.mode in ("starting", "failed"):
                    body = json.dumps(startup_health(self.mode)).encode()
                    self.send_response(503)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.mode == "ready":
                    body = json.dumps(health(uptime=10)).encode()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.mode in ("bad503", "oversize503"):
                    body = b"{" if self.mode == "bad503" else b" " * (server.MAX_BODY + 1)
                    self.send_response(503)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    body = b"{broken"
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

        mock = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        thread = threading.Thread(target=mock.serve_forever, daemon=True)
        thread.start()
        try:
            c = server.Collector("http://127.0.0.1:%d/health" % mock.server_port)
            self.assertFalse(c.poll_once())
            self.assertEqual(c.snapshot()["error"], "invalid JSON response")
            Upstream.mode = "ready"
            self.assertTrue(c.poll_once())
            initial_ready = c.snapshot()
            self.assertFalse(initial_ready["samples"][-1]["reset"])
            Upstream.mode = "starting"
            self.assertTrue(c.poll_once())
            starting = c.snapshot()
            self.assertFalse(starting["ok"])
            self.assertEqual(starting["startup"]["status"], "starting")
            self.assertEqual(len(starting["samples"]), len(initial_ready["samples"]))
            Upstream.mode = "ready"
            self.assertTrue(c.poll_once())
            ready = c.snapshot()
            self.assertIsNone(ready["startup"])
            self.assertTrue(ready["samples"][-1]["reset"])
            self.assertTrue(any(event["kind"] == "restart" for event in ready["events"]))
            Upstream.mode = "failed"
            self.assertTrue(c.poll_once())
            failed = c.snapshot()
            self.assertEqual(failed["startup"]["status"], "failed")
            self.assertEqual(failed["latest"]["status"], "ok")
            Upstream.mode = "bad"
            self.assertFalse(c.poll_once())
            offline = c.snapshot()
            self.assertIsNone(offline["startup"])
            self.assertEqual(offline["latest"], ready["latest"])
            for mode, message in (("bad503", "invalid JSON response"), ("oversize503", "exceeds 1 MiB")):
                Upstream.mode = mode
                self.assertFalse(c.poll_once())
                self.assertIn(message, c.snapshot()["error"])
            Upstream.mode = "redirect"
            self.assertFalse(c.poll_once())
            self.assertIn("HTTP 302", c.snapshot()["error"])
        finally:
            mock.shutdown()
            mock.server_close()
            thread.join()

    def test_http_cursor_path_restriction_and_shared_collector(self):
        calls = []

        def fetch():
            calls.append(1)
            return health(requests=len(calls))

        collector = server.Collector("unused", fetch=fetch)
        collector.poll_once()
        handler = server.make_handler(collector, os.path.join(os.path.dirname(__file__), "index.html"))
        web = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=web.serve_forever, daemon=True)
        thread.start()
        try:
            def get(path, headers=None):
                conn = http.client.HTTPConnection("127.0.0.1", web.server_port, timeout=2)
                conn.request("GET", path, headers=headers or {})
                response = conn.getresponse()
                data = response.read()
                conn.close()
                return response, data

            response, body = get("/api/history?after=0")
            self.assertEqual(response.status, 200)
            first = json.loads(body)
            self.assertEqual(len(first["samples"]), 1)
            self.assertTrue(first["ok"])
            response, body = get("/api/history?after=1")
            self.assertEqual(json.loads(body)["samples"], [])
            response, _ = get("/api/history?after=-1")
            self.assertEqual(response.status, 400)
            for cursor in ("%C2%B2", "1" * 21):
                response, _ = get("/api/history?after=" + cursor)
                self.assertEqual(response.status, 400, cursor)
            response, _ = get("/api/history?url=http://example.com")
            self.assertEqual(response.status, 400)
            for path in ("/secret", "/tools/health_dashboard/server.py", "/?x=y", "/api/history#fragment", "http://example.org/api/history"):
                response, _ = get(path)
                self.assertEqual(response.status, 400 if "#" in path or path.startswith("http:") else 404, path)
            head = http.client.HTTPConnection("127.0.0.1", web.server_port, timeout=2)
            head.request("HEAD", "/api/stream")
            response = head.getresponse()
            self.assertEqual(response.status, 405)
            self.assertEqual(response.read(), b"")
            head.close()
            response, _ = get("/api/history", {"Accept-Encoding": "gzip;q=0"})
            self.assertIsNone(response.getheader("Content-Encoding"))
            response, body = get("/api/history", {"Accept-Encoding": "gzip"})
            self.assertEqual(response.getheader("Content-Encoding"), "gzip")
            json.loads(gzip.decompress(body))
            for _ in range(4):
                response, body = get("/api/history")
                self.assertEqual(json.loads(body)["collector_id"], first["collector_id"])
            self.assertEqual(len(calls), 1)
        finally:
            web.shutdown()
            web.server_close()
            thread.join()

    def test_sse_replays_from_cursor_and_does_not_poll_per_client(self):
        calls = []

        def fetch():
            calls.append(len(calls) + 1)
            return health(requests=len(calls))

        collector = server.Collector("unused", fetch=fetch)
        collector.poll_once()
        handler = server.make_handler(collector, os.path.join(os.path.dirname(__file__), "index.html"))
        web = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        web.daemon_threads = True
        thread = threading.Thread(target=web.serve_forever, daemon=True)
        thread.start()

        def frame_reader(response, compressed):
            stream = zlib.decompressobj(16 + zlib.MAX_WBITS) if compressed else None
            pending = bytearray()

            def read_frame():
                while b"\n\n" not in pending:
                    chunk = response.read1(65536)
                    if not chunk:
                        raise AssertionError("stream ended before an SSE frame")
                    pending.extend(stream.decompress(chunk) if stream else chunk)
                frame, rest = pending.split(b"\n\n", 1)
                pending[:] = rest
                return frame.decode("utf-8").splitlines()

            read_frame.inflater = stream
            return read_frame

        def payload(lines):
            return json.loads(next(line[6:] for line in lines if line.startswith("data: ")))

        def connect(last_event_id=None, gzip_ok=True):
            conn = http.client.HTTPConnection("127.0.0.1", web.server_port, timeout=3)
            headers = {"Last-Event-ID": last_event_id} if last_event_id else {}
            headers["Accept-Encoding"] = "gzip" if gzip_ok else "gzip;q=0"
            conn.request("GET", "/api/stream", headers=headers)
            response = conn.getresponse()
            self.assertEqual(response.status, 200)
            self.assertEqual(response.getheader("Content-Type"), "text/event-stream; charset=utf-8")
            self.assertEqual(response.getheader("Content-Encoding"), "gzip" if gzip_ok else None)
            read_frame = frame_reader(response, gzip_ok)
            lines = read_frame()
            event_id = next(line[4:] for line in lines if line.startswith("id: "))
            data = payload(lines)
            return conn, response, read_frame, event_id, data

        try:
            first, _, _, event_id, initial = connect()
            self.assertEqual([sample["seq"] for sample in initial["samples"]], [1])
            second, _, _, _, same = connect()
            self.assertEqual([sample["seq"] for sample in same["samples"]], [1])
            plain, _, _, _, plain_initial = connect(gzip_ok=False)
            self.assertEqual([sample["seq"] for sample in plain_initial["samples"]], [1])
            self.assertEqual(calls, [1])
            first.close()
            second.close()
            plain.close()

            collector.poll_once()
            resumed, replay_response, read_replay, _, replay = connect(event_id)
            self.assertEqual([sample["seq"] for sample in replay["samples"]], [2])
            self.assertEqual(calls, [1, 2])
            collector.fetch = lambda: (503, startup_health())
            collector.poll_once()
            starting = payload(read_replay())
            self.assertEqual(starting["startup"]["status"], "starting")
            self.assertEqual(starting["samples"], [])
            collector.fetch = lambda: (_ for _ in ()).throw(ValueError("upstream unavailable"))
            collector.poll_once()
            disconnected = payload(read_replay())
            self.assertFalse(disconnected["ok"])
            self.assertEqual(disconnected["events"][-1]["kind"], "disconnect")
            collector.stop()
            while True:
                chunk = replay_response.read1(4096)
                if not chunk:
                    break
                read_replay.inflater.decompress(chunk)
            self.assertTrue(read_replay.inflater.eof)
            resumed.close()
        finally:
            web.shutdown()
            web.server_close()
            thread.join()

    def test_slow_stream_writer_does_not_block_collection(self):
        collector = server.Collector("unused", fetch=lambda: health())
        collector.poll_once()
        base_handler = server.make_handler(collector, os.path.join(os.path.dirname(__file__), "index.html"))
        writing = threading.Event()
        release = threading.Event()
        write_timeouts = []

        class SlowWriter:
            def __init__(self, stream):
                self.stream = stream
                self.first = True

            def write(self, data):
                if self.first:
                    self.first = False
                    writing.set()
                    release.wait(2.0)
                return self.stream.write(data)

            def flush(self):
                return self.stream.flush()

            def __getattr__(self, name):
                return getattr(self.stream, name)

        class SlowHandler(base_handler):
            def _stream(self):
                write_timeouts.append(self.connection.gettimeout())
                self.wfile = SlowWriter(self.wfile)
                super()._stream()

        web = ThreadingHTTPServer(("127.0.0.1", 0), SlowHandler)
        web.daemon_threads = True
        thread = threading.Thread(target=web.serve_forever, daemon=True)
        thread.start()
        conn = http.client.HTTPConnection("127.0.0.1", web.server_port, timeout=3)
        try:
            conn.request("GET", "/api/stream")
            response = conn.getresponse()
            self.assertEqual(response.status, 200)
            self.assertTrue(writing.wait(2.0))
            self.assertEqual(write_timeouts, [5.0])
            start = time.perf_counter()
            self.assertTrue(collector.poll_once())
            self.assertLess(time.perf_counter() - start, 1.0)
        finally:
            release.set()
            collector.stop()
            conn.close()
            web.shutdown()
            web.server_close()
            thread.join()


class ApiKeyFileTests(unittest.TestCase):
    def _write_key(self, directory, data):
        path = os.path.join(directory, "api-key")
        with open(path, "wb") as stream:
            stream.write(data)
        return path

    def test_cli_supplies_optional_key_before_starting(self):
        with tempfile.TemporaryDirectory() as directory:
            for contents, expected in ((None, {}),
                                       (b" \tsecret-key\r\n", {"Authorization": "Bearer secret-key"}),
                                       (b"x" * 4096, {"Authorization": "Bearer " + "x" * 4096})):
                with self.subTest(size=None if contents is None else len(contents)):
                    args = ["--port", "0"]
                    if contents is not None:
                        args += ["--api-key-file", self._write_key(directory, contents)]
                    with mock.patch.object(server, "Collector") as collector, \
                            mock.patch.object(server, "ThreadingHTTPServer") as bind, \
                            mock.patch.object(server.signal, "signal"):
                        bind.return_value.serve_forever.side_effect = KeyboardInterrupt
                        server.main(args)
                        collector.assert_called_once_with("http://127.0.0.1:8090/v1/health",
                                                          2.0, 3600.0, headers=expected)
                        collector.return_value.start.assert_called_once_with()
                        collector.return_value.stop.assert_called_once_with()
                        bind.return_value.server_close.assert_called_once_with()

    def test_invalid_key_files_are_argparse_errors_without_echoing_contents(self):
        invalid = (
            ("missing", None),
            ("nonascii", b"secr\xc3\xa9t"),
            ("empty", b" \t\r\n"),
            ("internal-space", b"secret key"),
            ("control", b"secret\x00key"),
            ("tab", b"secret\tkey"),
            ("delete", b"secret\x7fkey"),
            ("multiline", b"first\nsecond"),
            ("oversize", b"x" * 4097),
        )
        with tempfile.TemporaryDirectory() as directory:
            for name, contents in invalid:
                with self.subTest(name=name):
                    path = os.path.join(directory, name)
                    if contents is not None:
                        self._write_key(directory, contents)
                        os.replace(os.path.join(directory, "api-key"), path)
                    stderr = io.StringIO()
                    with mock.patch.object(server, "ThreadingHTTPServer") as bind, \
                            mock.patch("sys.stderr", stderr), self.assertRaises(SystemExit) as error:
                        server.main(["--api-key-file", path])
                    self.assertEqual(error.exception.code, 2)
                    self.assertFalse(bind.called)
                    self.assertNotIn("secret", stderr.getvalue())
                    self.assertNotIn("first", stderr.getvalue())

    def test_key_is_sent_to_health_upstream_and_not_redirected_or_exposed(self):
        key = "dashboard-secret"
        seen = []

        class Target(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def do_GET(self):
                seen.append(("target", self.headers.get("Authorization")))
                self.send_response(200)
                self.end_headers()

        target = ThreadingHTTPServer(("127.0.0.1", 0), Target)
        target_thread = threading.Thread(target=target.serve_forever, daemon=True)
        target_thread.start()

        class Upstream(BaseHTTPRequestHandler):
            mode = "ready"

            def log_message(self, fmt, *args):
                pass

            def do_GET(self):
                seen.append((self.path, self.headers.get("Authorization")))
                if self.mode == "redirect":
                    self.send_response(302)
                    self.send_header("Location", "http://127.0.0.1:%d/redirect-target" % target.server_port)
                    self.end_headers()
                    return
                if self.mode == "unauthorized":
                    self.send_response(401)
                    self.end_headers()
                    return
                status, value = ((503, startup_health()) if self.mode == "starting" else (200, health()))
                body = json.dumps(value).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        upstream_thread.start()
        collector = server.Collector("http://127.0.0.1:%d/health" % upstream.server_port,
                                     headers={"Authorization": "Bearer " + key})
        handler = server.make_handler(collector, os.path.join(os.path.dirname(__file__), "index.html"))
        web = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        web_thread = threading.Thread(target=web.serve_forever, daemon=True)
        web_thread.start()
        try:
            for mode, expected_status in (("ready", 200), ("starting", 503)):
                Upstream.mode = mode
                self.assertTrue(collector.poll_once())
                self.assertEqual(seen[-1], ("/health", "Bearer " + key))
                snapshot = collector.snapshot()
                if expected_status == 503:
                    self.assertEqual(snapshot["startup"]["status"], "starting")
                else:
                    self.assertEqual(snapshot["latest"]["status"], "ok")

            Upstream.mode = "unauthorized"
            self.assertFalse(collector.poll_once())
            snapshot = collector.snapshot()
            self.assertIn("HTTP 401", snapshot["error"])
            self.assertNotIn(key, json.dumps(snapshot))

            seen.clear()
            Upstream.mode = "redirect"
            self.assertFalse(collector.poll_once())
            self.assertIn("HTTP 302", collector.snapshot()["error"])
            self.assertEqual(seen, [("/health", "Bearer " + key)])

            conn = http.client.HTTPConnection("127.0.0.1", web.server_port, timeout=2)
            conn.request("GET", "/api/history")
            response = conn.getresponse()
            body = response.read().decode("utf-8")
            conn.close()
            self.assertEqual(response.status, 200)
            self.assertNotIn(key, body)
        finally:
            web.shutdown()
            web.server_close()
            web_thread.join()
            upstream.shutdown()
            upstream.server_close()
            upstream_thread.join()
            target.shutdown()
            target.server_close()
            target_thread.join()


if __name__ == "__main__":
    unittest.main()
