import http.client
import json
import os
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from server import Application, Server


def health():
    return {"status": "ok", "server": {"uptime_s": 10, "model": "fixture", "version": "v1",
                                           "numerics": "abc", "devices": ["cpu"]},
            "requests": {"now": {"active": 0, "queued": 0}, "since_start": {"finished": 0}},
            "precision": {"effective": "f32"}}


class FakeUpstream(BaseHTTPRequestHandler):
    lock = threading.Lock()
    health_calls = 0

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path.endswith("/health"):
            with type(self).lock:
                type(self).health_calls += 1
            body = json.dumps(health()).encode("utf-8")
        else:
            body = json.dumps({"data": [{"id": "fixture-model"}]}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class FakeMCP(BaseHTTPRequestHandler):
    tokens = []

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length))
        authorization = self.headers.get("Authorization")
        if authorization:
            type(self).tokens.append(authorization)
        method = request.get("method")
        if method == "initialize":
            result = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                      "serverInfo": {"name": "http fake", "version": "1"}}
        elif method == "tools/list":
            result = {"tools": [{"name": "safe", "description": "safe tool",
                                  "inputSchema": {"type": "object"}}]}
        else:
            result = None
        if result is None:
            self.send_response(202)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body = json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class FakeRunner:
    def __init__(self, block=False):
        self.started = threading.Event()
        self.release = threading.Event()
        self.block = block

    def run(self, files, command, emit, cancel):
        self.started.set()
        emit("runner output\n")
        if self.block:
            self.release.wait(3)
        if cancel.is_set():
            return None, -1
        return {**files, "generated.txt": command}, 0


class WorkspaceHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = ThreadingHTTPServer(("127.0.0.1", 0), FakeUpstream)
        cls.upstream_thread = threading.Thread(target=cls.upstream.serve_forever, daemon=True)
        cls.upstream_thread.start()
        cls.mcp = ThreadingHTTPServer(("127.0.0.1", 0), FakeMCP)
        cls.mcp_thread = threading.Thread(target=cls.mcp.serve_forever, daemon=True)
        cls.mcp_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.upstream.shutdown()
        cls.mcp.shutdown()
        cls.upstream_thread.join()
        cls.mcp_thread.join()
        cls.upstream.server_close()
        cls.mcp.server_close()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        database = os.path.join(self.temp.name, "chat.sqlite")
        upstream = "http://127.0.0.1:%d/v1" % self.upstream.server_address[1]
        self.tools_config = os.path.join(self.temp.name, "tools.json")
        with open(self.tools_config, "w", encoding="utf-8") as output:
            json.dump({"servers": [{"name": "remote", "url": "http://127.0.0.1:%d/mcp" % self.mcp.server_address[1],
                                     "headers": {"Authorization": "Bearer workspace-secret"}}]}, output)
        FakeMCP.tokens = []
        with FakeUpstream.lock:
            FakeUpstream.health_calls = 0
        self.app = Application(database, upstream, self.tools_config, password="correct horse")
        self.httpd = Server(("127.0.0.1", 0), self.app)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.httpd.server_address[1]

    def tearDown(self):
        self.httpd.shutdown()
        self.thread.join()
        self.httpd.server_close()
        self.app.engine.shutdown()
        self.app.store.close()
        self.temp.cleanup()

    def request(self, method, path, body=None, cookie=None, headers=None, timeout=3):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        hdr = dict(headers or {})
        if body is not None:
            hdr.setdefault("X-Chat-Request", "1")
            if isinstance(body, (bytes, bytearray)):
                hdr.setdefault("Content-Type", "application/vnd.sqlite3")
                raw = bytes(body)
            else:
                hdr.setdefault("Content-Type", "application/json")
                raw = json.dumps(body, separators=(",", ":")).encode("utf-8")
        else:
            raw = None
        if cookie:
            hdr["Cookie"] = cookie
        conn.request(method, path, body=raw, headers=hdr)
        response = conn.getresponse()
        data = response.read()
        result = (response.status, dict((k.lower(), v) for k, v in response.getheaders()), data)
        conn.close()
        return result

    def stream(self, path, cookie=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        headers = {"Cookie": cookie} if cookie else {}
        conn.request("GET", path, headers=headers)
        response = conn.getresponse()
        return conn, response

    def json_body(self, result):
        return json.loads(result[2].decode("utf-8"))

    def login(self):
        response = self.request("POST", "/api/login", {"password": "correct horse"})
        self.assertEqual(response[0], 200)
        return response[1]["set-cookie"].split(";", 1)[0]

    def wait_job(self, job_id, timeout=2):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = self.request("GET", "/api/workspace/jobs/" + job_id, cookie=self.cookie)
            job = self.json_body(result)["job"]
            if job["status"] != "running":
                return job
            threading.Event().wait(0.005)
        self.fail("HTTP workspace job did not finish")

    def test_workspace_routes_auth_origin_revision_and_checkpoint(self):
        self.assertEqual(self.request("GET", "/api/workspace")[0], 401)
        self.cookie = self.login()
        self.assertEqual(self.request("GET", "/api/workspace", cookie=self.cookie,
                                      headers={"Origin": "https://attacker.invalid"})[0], 403)
        first = self.json_body(self.request("GET", "/api/workspace", cookie=self.cookie))
        self.assertEqual(first["revision"], 0)
        written = self.request("POST", "/api/workspace/file",
                               {"path": "src/main.py", "content": "old\n", "revision": 0}, self.cookie)
        self.assertEqual(written[0], 200)
        checkpoint = self.json_body(self.request("POST", "/api/workspace/checkpoints",
                                                 {"label": "before edit"}, self.cookie))
        checkpoint_id = checkpoint["checkpoints"][-1]["id"]
        stale = self.request("POST", "/api/workspace/file",
                             {"path": "stale", "content": "x", "revision": 0}, self.cookie)
        self.assertEqual(stale[0], 400)
        escaped = self.request("POST", "/api/workspace/file",
                               {"path": "../escape", "content": "x", "revision": 1}, self.cookie)
        self.assertEqual(escaped[0], 400)
        current = self.json_body(self.request("GET", "/api/workspace", cookie=self.cookie))["revision"]
        self.request("POST", "/api/workspace/file",
                     {"path": "src/main.py", "content": "new\n", "revision": current}, self.cookie)
        diff = self.request("GET", "/api/workspace/diff?checkpoint=" + checkpoint_id, cookie=self.cookie)
        self.assertIn("-old", self.json_body(diff)["diff"])
        restored = self.request("POST", "/api/workspace/checkpoints/" + checkpoint_id + "/restore",
                                {"revision": current + 1}, self.cookie)
        self.assertEqual(restored[0], 200)
        read = self.request("GET", "/api/workspace/file?path=src%2Fmain.py", cookie=self.cookie)
        self.assertEqual(self.json_body(read)["content"], "old\n")
        removed = self.request("POST", "/api/workspace/delete",
                               {"path": "src/main.py", "revision": current + 2}, self.cookie)
        self.assertEqual(removed[0], 200)

    def test_job_routes_use_fake_runner_and_cancel_without_applying_files(self):
        self.cookie = self.login()
        runner = FakeRunner()
        self.app.workspace.runner = runner
        started = self.request("POST", "/api/workspace/jobs", {"command": "make check"}, self.cookie)
        self.assertEqual(started[0], 200)
        job = self.json_body(started)["job"]
        self.assertTrue(runner.started.wait(2))
        final = self.wait_job(job["id"])
        self.assertEqual(final["status"], "completed")
        self.assertEqual(self.json_body(self.request("GET", "/api/workspace", cookie=self.cookie))["files"][0]["path"], "generated.txt")

        blocker = FakeRunner(block=True)
        self.app.workspace.runner = blocker
        second = self.json_body(self.request("POST", "/api/workspace/jobs", {"command": "wait"}, self.cookie))["job"]
        self.assertTrue(blocker.started.wait(2))
        cancel = self.request("POST", "/api/workspace/jobs/" + second["id"] + "/cancel", {}, self.cookie)
        self.assertEqual(cancel[0], 200)
        blocker.release.set()
        final_cancelled = self.wait_job(second["id"])
        self.assertEqual(final_cancelled["status"], "cancelled")
        self.assertEqual(self.json_body(self.request("GET", "/api/workspace", cookie=self.cookie))["files"][0]["path"], "generated.txt")

    def test_backup_restore_invalid_restore_and_active_run_refusal(self):
        self.assertEqual(self.request("GET", "/api/backup")[0], 401)
        self.assertEqual(self.request("POST", "/api/restore", b"")[0], 401)
        self.cookie = self.login()
        chat = self.json_body(self.request("POST", "/api/chats", {"title": "saved"}, self.cookie))
        backup = self.request("GET", "/api/backup", cookie=self.cookie)
        self.assertEqual(backup[0], 200)
        self.assertEqual(backup[2][:16], b"SQLite format 3\x00")
        self.assertIn("attachment", backup[1]["content-disposition"])
        self.json_body(self.request("POST", "/api/chats", {"title": "later"}, self.cookie))
        bad = self.request("POST", "/api/restore", b"not sqlite", self.cookie)
        self.assertEqual(bad[0], 400)
        wrong_mime = self.request("POST", "/api/restore", b"not sqlite", self.cookie,
                                  {"Content-Type": "application/json"})
        self.assertEqual(wrong_mime[0], 400)
        self.assertEqual(len(self.app.store.chats()), 2)
        restored = self.request("POST", "/api/restore", backup[2], self.cookie)
        self.assertEqual(restored[0], 200)
        self.assertEqual([item["title"] for item in self.app.store.chats()], ["saved"])

        run = self.app.store.start(chat["id"], {"content": "active"}, {"model": "", "tools": []})
        refusal = self.request("POST", "/api/restore", backup[2], self.cookie)
        self.assertEqual(refusal[0], 400)
        self.assertEqual(self.app.store.run(run["id"])["status"], "running")

    def test_large_sqlite_backup_restores_without_base64_expansion(self):
        self.cookie = self.login()
        chat = self.json_body(self.request("POST", "/api/chats", {"title": "large"}, self.cookie))
        run = self.app.store.start(chat["id"], {"content": "x" * (14 * 1024 * 1024)},
                                   {"model": "", "tools": []})
        answer = self.app.store.message(run["message_id"])
        answer["status"] = "completed"
        self.app.store.publish(run["id"], "message", answer)
        self.app.store.publish(run["id"], "state", dict(self.app.store.run(run["id"]), status="completed"))
        backup = self.request("GET", "/api/backup", cookie=self.cookie, timeout=20)
        self.assertEqual(backup[0], 200)
        self.assertGreater(len(backup[2]), 12 * 1024 * 1024)
        self.json_body(self.request("POST", "/api/chats", {"title": "later"}, self.cookie))
        restored = self.request("POST", "/api/restore", backup[2], self.cookie, timeout=20)
        self.assertEqual(restored[0], 200, restored[2][:500])
        restored_chat = self.app.store.chat(chat["id"])
        snapshot = self.app.store.snapshot(restored_chat["id"])
        user_message = next(message for message in snapshot["messages"] if message["role"] == "user")
        self.assertEqual(len(user_message["content"]), 14 * 1024 * 1024)
        self.assertEqual([item["title"] for item in self.app.store.chats()], ["large"])

    def test_previews_are_authenticated_bounded_and_sandboxed(self):
        self.assertEqual(self.request("POST", "/api/previews", {"html": "<b>hi</b>"})[0], 401)
        self.cookie = self.login()
        self.assertEqual(self.request("POST", "/api/previews", {"html": "<b>hi</b>"}, self.cookie,
                                      {"Origin": "https://attacker.invalid"})[0], 403)
        self.assertEqual(self.request("POST", "/api/previews", {"html": "x" * (1024 * 1024 + 1)}, self.cookie)[0], 400)
        url = self.json_body(self.request("POST", "/api/previews", {"html": "<h1>preview</h1>"}, self.cookie))["url"]
        self.assertEqual(self.request("GET", url)[0], 401)
        preview = self.request("GET", url, cookie=self.cookie)
        self.assertEqual(preview[0], 200)
        csp = preview[1]["content-security-policy"]
        self.assertIn("sandbox allow-scripts", csp)
        self.assertNotIn("allow-same-origin", csp)
        self.assertIn("connect-src 'none'", csp)
        self.assertIn(b"preview", preview[2])

    def test_monitor_is_authenticated_and_shared_across_clients_without_tool_secret_leak(self):
        self.assertEqual(self.request("GET", "/api/monitor/history")[0], 401)
        self.assertEqual(self.request("GET", "/monitor/view")[0], 401)
        unauthorized, unauthorized_response = self.stream("/api/monitor/stream")
        try:
            self.assertEqual(unauthorized_response.status, 401)
            unauthorized_response.read()
        finally:
            unauthorized.close()
        self.cookie = self.login()
        config = self.request("GET", "/api/config", cookie=self.cookie)
        self.assertEqual(config[0], 200)
        self.assertNotIn("workspace-secret", config[2].decode("utf-8"))
        self.assertIn("remote__safe", [tool["name"] for tool in self.json_body(config)["tools"]])
        self.assertEqual(FakeMCP.tokens, ["Bearer workspace-secret"] * 3)
        count = FakeUpstream.health_calls
        first = self.json_body(self.request("GET", "/api/monitor/history?after=0", cookie=self.cookie))
        second = self.json_body(self.request("GET", "/api/monitor/history?after=0", cookie=self.cookie))
        self.assertEqual(first["collector_id"], second["collector_id"])
        self.assertEqual(FakeUpstream.health_calls, count)
        monitor = self.request("GET", "/monitor/view", cookie=self.cookie)
        self.assertEqual(monitor[0], 200)
        self.assertIn(b"data-embedded=\"true\"", monitor[2])
        conn, response = self.stream("/api/monitor/stream", self.cookie)
        try:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.getheader("Content-Type"), "text/event-stream; charset=utf-8")
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
