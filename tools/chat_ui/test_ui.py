"""Real-browser behavior checks against a deterministic companion upstream."""
import base64
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer

from server import Application, Server
from monitor import DASHBOARD, health
from test_engine import FakeTools, FakeUpstream, text_event, tool_event


class StepGate(threading.Event):
    def wait(self, timeout=None):
        released = super().wait(30)
        if not released:
            raise TimeoutError("Browser did not release the streamed fixture")
        if released:
            self.clear()
        return released


class BrowserUpstream(FakeUpstream):
    def do_POST(self):
        if self.path == "/v1/tokenize":
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            refused = any("Reject image test" in str(m.get("content")) for m in payload.get("messages", []))
            data = json.dumps({"error": {"message": "can only concatenate str (not list) to str"}} if refused else {"tokens": [1], "count": 1}).encode()
            self.send_response(400 if refused else 200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        with type(self).lock:
            cancellation = bool(type(self).replies and type(self).replies[0][0] == text_event(reasoning="Cancellation test"))
        if cancellation:
            type(self).gate.clear()
        super().do_POST()

    def do_GET(self):
        if self.path == "/browser-step":
            # The browser observed the preceding chunk; exceed the engine's 50 ms coalescing window before releasing the next.
            time.sleep(0.1)
            type(self).gate.set()
            self.send_response(204)
            self.end_headers()
            return
        health = {"status": "ok", "server": {"uptime_s": 30, "model": "qwen3-local", "version": "browser-fixture", "devices": ["cpu"], "context_tokens": 40960, "request_tokens": 8192},
                  "requests": {"now": {"active": 1, "queued": 0}, "since_start": {"finished": 4}}}
        data = json.dumps(health if self.path.endswith('/health') else {"data": [{"id": "qwen3-local", "context_length": 40960}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class BrowserRunner:
    def run(self, files, command, emit, cancel):
        emit("Running browser fixture command\n")
        if command == "sleep-test":
            cancel.wait(8)
            return None, 130
        cancel.wait(0.7)
        emit("browser-job complete\n")
        return dict(files, **{"generated.txt": "Created by the deterministic browser fixture.\n"}), 0


class BrowserTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "posix", "Chrome launch diagnostics are Linux-specific")
    def test_browser_launch_timeout_reports_live_process_state(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node is unavailable")
        with tempfile.TemporaryDirectory(prefix="llmx-browser-launch-") as temporary:
            root = Path(temporary)
            fake_chrome = root / "fake-chrome"
            fake_chrome.write_text(
                "#!/usr/bin/env python3\n"
                "import sys, time\n"
                "print('fake browser started', file=sys.stderr, flush=True)\n"
                "time.sleep(2.5)\n",
                encoding="utf-8",
            )
            fake_chrome.chmod(0o755)
            result = subprocess.run(
                [node, str(Path(__file__).with_name("browser_check.mjs")),
                 "http://127.0.0.1:1", str(fake_chrome), str(root / "profile"),
                 str(root / "artifacts"), "--startup-timeout-ms=150"],
                capture_output=True, text=True, timeout=8,
            )
        self.assertNotEqual(result.returncode, 0)
        timeout = re.search(
            r"Chrome did not open DevTools within (\d+) ms \(limit 150 ms\)",
            result.stderr,
        )
        self.assertIsNotNone(timeout, result.stderr)
        self.assertGreaterEqual(int(timeout.group(1)), 150)
        self.assertIn("Chrome process:", result.stderr)
        self.assertIn("still running", result.stderr)
        self.assertIn("fake browser started", result.stderr)

    def test_workspace_in_chrome(self):
        node = shutil.which("node")
        chrome = next((str(p) for p in [
            Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "Google/Chrome/Application/chrome.exe",
            Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "Microsoft/Edge/Application/msedge.exe",
            Path("/usr/bin/google-chrome"), Path("/usr/bin/google-chrome-stable"),
            Path("/usr/bin/chromium"), Path("/usr/bin/chromium-browser"),
        ] if p.is_file()), None)
        if not node or not chrome:
            if os.environ.get("LLMX_CHAT_REQUIRE_BROWSER") == "1":
                self.fail("Hosted workspace checks require Node and Chrome/Chromium/Edge")
            self.skipTest("Real browser check requires Node 22+ and Chrome/Chromium/Edge")
        metrics = {"choices": [], "usage": {"prompt_tokens": 120, "completion_tokens": 40,
                   "total_tokens": 160, "prompt_tokens_details": {"cached_tokens": 32}},
                   "timings": {"prompt_per_second": 240, "predicted_per_second": 25,
                               "prompt_ms": 500, "predicted_ms": 1600, "queued_ms": 0, "prepared_ms": 10},
                   "context": {"tokens": 160, "limit": 4096}}
        first = [dict(text_event(reasoning="Let me think carefully."), id="r-browser-1"), "WAIT",
                 text_event(reasoning=" Compare the options."), "WAIT",
                 text_event(reasoning=" Choose a clear plan."), "WAIT",
                 text_event("A practical plan.\n\n```python\nprint('hello')\n```\n\n| Step | State |\n| --- | --- |\n| One | Ready |\n\n<script>window.injected=true</script>"), metrics]
        tool_first = tool_event(name="ec", arguments='{"text":"hel')
        tool_first["choices"][0]["finish_reason"] = None
        tool_second = tool_event(name="ho", arguments="lo")
        tool_second["choices"][0]["finish_reason"] = None
        with BrowserUpstream.lock:
            BrowserUpstream.replies = [first, [text_event(reasoning="Consider another approach."), "WAIT", text_event("An alternative approach."), metrics],
                [text_event("An edited branch."), metrics],
                [tool_first, "WAIT", tool_second, "WAIT", tool_event(name="", arguments='"}'), metrics],
                [text_event("The tool returned hello."), metrics], [text_event("The attached file is readable."), metrics],
                [text_event("Incomplete response"), "TRUNCATE"],
                [text_event(reasoning="Cancellation test"), "WAIT", text_event("Should be cancelled")],
                [text_event("Temporary response."), "WAIT", metrics],
                [text_event("Normal response saved."), metrics],
                [text_event("The other chat is active."), "WAIT", text_event("The other chat completed."), metrics],
                [text_event("Editable seed response."), metrics],
                [text_event("The submitted edit completed."), metrics],
                [text_event("The second submission completed."), metrics],
                [text_event("The delayed retry completed."), metrics],
                [text_event("The snapshot retry completed."), metrics],
                [text_event("The returning retry completed."), metrics],
                [text_event("The round trip submission completed."), metrics],
                [text_event("The stale snapshot submission completed."), metrics],
                [text_event("Preparing a background tool."), "WAIT", tool_event()],
                [text_event("A background reply failed."), "TRUNCATE"]]
            BrowserUpstream.requests = []
        BrowserUpstream.gate = StepGate()
        BrowserUpstream.stream_opened = threading.Event()
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), BrowserUpstream)
        threading.Thread(target=upstream.serve_forever, daemon=True).start()
        try:
            with tempfile.TemporaryDirectory(prefix="llmx-browser-") as temporary:
                app = Application(str(Path(temporary) / "chat.sqlite"),
                                  "http://127.0.0.1:%d" % upstream.server_address[1], password="browser-test")
                tools = FakeTools()
                app.tools = tools
                app.engine.tools = tools
                app.workspace.runner = BrowserRunner()
                server = Server(("127.0.0.1", 0), app)
                threading.Thread(target=server.serve_forever, daemon=True).start()
                dashboard = ThreadingHTTPServer(("127.0.0.1", 0), health.make_handler(app.monitor.collector, str(DASHBOARD / "index.html")))
                threading.Thread(target=dashboard.serve_forever, daemon=True).start()
                try:
                    artifacts = Path(__file__).resolve().parents[2] / "scratch/chat-ui-browser"
                    artifacts.mkdir(parents=True, exist_ok=True)
                    fixture = Path(temporary) / "notes.txt"
                    fixture.write_text("A useful attached note.", encoding="utf-8")
                    image = Path(temporary) / "pixel.png"
                    image.write_bytes(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aBy0AAAAASUVORK5CYII="))
                    text_max = Path(temporary) / "text-max.txt"
                    text_max.write_bytes(b"x" * (2 * 1024 * 1024))
                    text_over = Path(temporary) / "text-over.txt"
                    text_over.write_bytes(b"x" * (2 * 1024 * 1024 + 1))
                    text_raw_over = Path(temporary) / "text-raw-over.txt"
                    text_raw_over.write_bytes(b"x" * (8 * 1024 * 1024))
                    text_bom_max = Path(temporary) / "text-bom-max.txt"
                    text_bom_max.write_bytes(b"\xef\xbb\xbf" + b"x" * (2 * 1024 * 1024))
                    image_max = Path(temporary) / "image-max.png"
                    image_max.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * (8 * 1024 * 1024 - 8))
                    image_over = Path(temporary) / "image-over.png"
                    image_over.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * (8 * 1024 * 1024 - 7))
                    text_tiny = Path(temporary) / "tiny.txt"
                    text_tiny.write_bytes(b"x")
                    text_race_a = Path(temporary) / "race-a.txt"
                    text_race_a.write_bytes(b"a" * (3 * 1024 * 1024 // 2))
                    text_race_b = Path(temporary) / "race-b.txt"
                    text_race_b.write_bytes(b"b" * (3 * 1024 * 1024 // 2))
                    result = subprocess.run([node, str(Path(__file__).with_name("browser_check.mjs")),
                        "http://127.0.0.1:%d" % server.server_address[1], chrome,
                        str(Path(temporary) / "profile"), str(artifacts), str(fixture), str(image),
                        "http://127.0.0.1:%d" % dashboard.server_address[1],
                        "http://127.0.0.1:%d" % upstream.server_address[1],
                        str(text_max), str(text_over), str(image_max), str(image_over), str(text_tiny),
                        str(text_raw_over), str(text_bom_max), str(text_race_a), str(text_race_b)],
                        capture_output=True, text=True, timeout=100)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertGreaterEqual(len(tools.calls), 1)
                    print(result.stdout.strip())
                finally:
                    BrowserUpstream.gate.set()
                    app.engine.shutdown()
                    dashboard.shutdown()
                    dashboard.server_close()
                    server.shutdown()
                    server.server_close()
                    app.store.close()
        finally:
            upstream.shutdown()
            upstream.server_close()


if __name__ == "__main__":
    unittest.main()
