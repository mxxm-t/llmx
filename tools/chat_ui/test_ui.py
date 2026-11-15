"""Real-browser behavior checks against a deterministic companion upstream."""
import base64
import json
import os
from pathlib import Path
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


class BrowserUpstream(FakeUpstream):
    def do_POST(self):
        with type(self).lock:
            cancellation = bool(type(self).replies and type(self).replies[0][0] == text_event(reasoning="Cancellation test"))
        if cancellation:
            type(self).gate.clear()
        super().do_POST()

    def do_GET(self):
        health = {"status": "ok", "server": {"uptime_s": 30, "model": "qwen3-local", "version": "browser-fixture", "devices": ["cpu"]},
                  "requests": {"now": {"active": 1, "queued": 0}, "since_start": {"finished": 4}}}
        data = json.dumps(health if self.path.endswith('/health') else {"data": [{"id": "qwen3-local"}]}).encode()
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
        first = [text_event(reasoning="Let me think carefully."), "WAIT",
                 text_event("A practical plan.\n\n```python\nprint('hello')\n```\n\n| Step | State |\n| --- | --- |\n| One | Ready |\n\n<script>window.injected=true</script>"), metrics]
        with BrowserUpstream.lock:
            BrowserUpstream.replies = [first, [text_event("An alternative approach."), metrics],
                [text_event("An edited branch."), metrics], [tool_event(name="echo"), metrics],
                [text_event("The tool returned hello."), metrics], [text_event("The attached file is readable."), metrics],
                [text_event("Incomplete response"), "TRUNCATE"],
                [text_event(reasoning="Cancellation test"), "WAIT", text_event("Should be cancelled")]]
            BrowserUpstream.requests = []
        BrowserUpstream.gate = threading.Event()
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
                def release_stream():
                    if BrowserUpstream.stream_opened.wait(10):
                        time.sleep(1.5)
                        BrowserUpstream.gate.set()
                release = threading.Thread(target=release_stream, daemon=True)
                release.start()
                try:
                    artifacts = Path(__file__).resolve().parents[2] / "scratch/chat-ui-browser"
                    artifacts.mkdir(parents=True, exist_ok=True)
                    fixture = Path(temporary) / "notes.txt"
                    fixture.write_text("A useful attached note.", encoding="utf-8")
                    image = Path(temporary) / "pixel.png"
                    image.write_bytes(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aBy0AAAAASUVORK5CYII="))
                    result = subprocess.run([node, str(Path(__file__).with_name("browser_check.mjs")),
                        "http://127.0.0.1:%d" % server.server_address[1], chrome,
                        str(Path(temporary) / "profile"), str(artifacts), str(fixture), str(image),
                        "http://127.0.0.1:%d" % dashboard.server_address[1]],
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
