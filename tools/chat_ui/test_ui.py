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
from http.server import ThreadingHTTPServer
import unittest
from unittest.mock import patch

import install_pdf_reader
import pdf_reader
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


def write_pdf_fixture(path):
    pages = (
        b"BT /F1 14 Tf 20 150 Td (First page private marker.) Tj ET",
        b"BT /F1 14 Tf 20 150 Td (Only the second page belongs in chat.) Tj ET",
        b"0.6 g 10 10 80 90 re f",
    )
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [4 0 R 6 0 R 8 0 R] /Count 3 >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for content in pages:
        stream = len(objects) + 2
        objects.append((f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 320 200] "
                        f"/Resources << /Font << /F1 3 0 R >> >> /Contents {stream} 0 R >>").encode())
        objects.append(f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream")
    data = bytearray(b"%PDF-1.4\n% raw-pdf-fixture-marker\n")
    offsets = []
    for index, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data.extend(f"{index} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = len(data)
    size = len(objects) + 1
    data.extend(f"xref\n0 {size}\n0000000000 65535 f \n".encode())
    for offset in offsets:
        data.extend(f"{offset:010d} 00000 n \n".encode())
    data.extend(f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    path.write_bytes(data)


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

    def browser_executables(self):
        node = shutil.which("node")
        chrome = next((str(p) for p in [
            Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "Google/Chrome/Application/chrome.exe",
            Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "Microsoft/Edge/Application/msedge.exe",
            Path("/usr/bin/google-chrome"), Path("/usr/bin/google-chrome-stable"),
            Path("/usr/bin/chromium"), Path("/usr/bin/chromium-browser"),
        ] if p.is_file()), None)
        if not node or not chrome:
            if any(os.environ.get(name) == "1" for name in ("LLMX_CHAT_REQUIRE_BROWSER", "LLMX_CHAT_REQUIRE_PDF")):
                self.fail("Hosted workspace checks require Node and Chrome/Chromium/Edge")
            self.skipTest("Real browser check requires Node 22+ and Chrome/Chromium/Edge")
        return node, chrome

    def test_pdf_attachment_in_chrome(self):
        node, chrome = self.browser_executables()
        require_pdf = os.environ.get("LLMX_CHAT_REQUIRE_PDF") == "1"
        installed_directory = pdf_reader.READER_DIRECTORY
        if require_pdf:
            self.assertTrue(pdf_reader.installed(installed_directory), "Install the pinned PDF reader before this check")
        with tempfile.TemporaryDirectory(prefix="llmx-browser-pdf-") as temporary:
            root = Path(temporary)
            fixture = root / "quarterly.pdf"
            write_pdf_fixture(fixture)
            html_fixture = root / "uploaded.html"
            html_source = "<script>window.sentHtmlRan = true;</script>\n<p>Uploaded HTML as text.</p>"
            html_fixture.write_text(html_source, encoding="utf-8", newline="\n")
            mode = "installed" if require_pdf else "absent"
            reader = installed_directory if require_pdf else root / "missing-reader"
            with patch.object(pdf_reader, "READER_DIRECTORY", reader):
                self.run_focused_browser(node, chrome, root, f"pdf-{mode}",
                                         (str(fixture), "--pdf-only=" + mode, "--html-fixture=" + str(html_fixture)),
                                         [[text_event("Plain browser reply.")], [text_event("PDF browser reply.")]])
            expected = [{"role": "user", "content": "Browser plain-text prompt"}]
            if mode == "installed":
                expected.append({"role": "user", "content": [
                    {"type": "text", "text": "Keep this PDF draft."},
                    {"type": "text", "text": "\nFile: quarterly.pdf\nPage 2\nOnly the second page belongs in chat."},
                    {"type": "text", "text": "\nFile: uploaded.html\n" + html_source},
                ]})
            with BrowserUpstream.lock:
                self.assertEqual([request["messages"][-1] for request in BrowserUpstream.requests], expected)
                self.assertNotIn("raw-pdf-fixture-marker", json.dumps(BrowserUpstream.requests))
                self.assertNotIn("First page private marker.", json.dumps(BrowserUpstream.requests))

    def test_optional_features_in_chrome(self):
        node, chrome = self.browser_executables()
        with tempfile.TemporaryDirectory(prefix="llmx-browser-features-") as temporary:
            root = Path(temporary)
            reader = root / "pdf-reader"
            note = root / "notes.txt"
            note.write_text("Keep this local text attachment.", encoding="utf-8")
            attempts = []

            def install_fixture(directory):
                attempts.append(directory)
                attempt = len(attempts)
                BrowserUpstream.gate.wait()
                if attempt == 1:
                    raise OSError("fixture download failure")
                for resource in pdf_reader.RESOURCE_PATHS:
                    (directory / resource).mkdir(parents=True)
                for module in pdf_reader.MODULES:
                    (directory / module).write_text("export {};", encoding="utf-8")
                (directory / pdf_reader.INSTALL_MARKER).write_bytes(pdf_reader.INSTALL_RECORD)

            with patch.object(pdf_reader, "READER_DIRECTORY", reader), patch.object(install_pdf_reader, "install", side_effect=install_fixture):
                self.run_focused_browser(node, chrome, root, "features", (str(note), "--features-only"),
                                         [[text_event("Plain browser reply.")]])
            self.assertEqual(attempts, [reader, reader], "visiting Settings or repeated requests started an extra installer")
            self.assertTrue(pdf_reader.installed(reader), "successful feature install did not persist its marker")
            with BrowserUpstream.lock:
                self.assertEqual([request["messages"][-1] for request in BrowserUpstream.requests],
                                 [{"role": "user", "content": "Browser plain-text prompt"}])

    def run_focused_browser(self, node, chrome, root, name, arguments, replies):
        with BrowserUpstream.lock:
            BrowserUpstream.requests = []
            BrowserUpstream.replies = replies
        BrowserUpstream.gate = StepGate()
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), BrowserUpstream)
        threading.Thread(target=upstream.serve_forever, daemon=True).start()
        try:
            upstream_url = "http://127.0.0.1:%d" % upstream.server_address[1]
            app = Application(str(root / f"{name}.sqlite"),
                              upstream_url, password="browser-test")
            server = Server(("127.0.0.1", 0), app)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            try:
                artifacts = Path(__file__).resolve().parents[2] / "scratch" / "chat-ui-browser" / name
                artifacts.mkdir(parents=True, exist_ok=True)
                result = subprocess.run([node, str(Path(__file__).with_name("browser_check.mjs")),
                    "http://127.0.0.1:%d" % server.server_address[1], chrome,
                    str(root / f"profile-{name}"), str(artifacts), *arguments, "--fixture-upstream=" + upstream_url],
                    capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                print(result.stdout.strip())
            finally:
                BrowserUpstream.gate.set()
                app.engine.shutdown()
                server.shutdown()
                server.server_close()
                app.store.close()
        finally:
            upstream.shutdown()
            upstream.server_close()

    def test_workspace_in_chrome(self):
        node, chrome = self.browser_executables()
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
                [text_event("The tool returned hello."), metrics], [text_event("The attached file is readable."), "WAIT", metrics],
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
                [tool_event()],
                [text_event(reasoning="The tool returned a result, but no answer followed."), "WAIT",
                 {"choices": [{"delta": {}, "finish_reason": "stop"}]}, metrics],
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
                        capture_output=True, text=True, timeout=300)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    for prompt in ("Temporary browser privacy check", "Normal browser persistence check"):
                        with self.subTest(project_prompt=prompt):
                            expected = [{"type": "text", "text": text} for text in
                                        (prompt, "\nFile: src:index.js\nSaved src/index.js", "\nFile: tests:index.js\nSaved tests/index.js")]
                            matches = [request for request in BrowserUpstream.requests
                                       if request["messages"][-1].get("content") == expected]
                            self.assertGreaterEqual(len(matches), 1)
                            self.assertEqual(matches[0]["messages"][0], {"role": "system", "content": "Use the selected browser fixtures."})
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
