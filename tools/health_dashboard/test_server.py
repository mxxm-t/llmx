import http.client
import gzip
import json
import os
import threading
import time
import unittest
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


if __name__ == "__main__":
    unittest.main()
