import http.client
import json
import os
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from server import Application, Server


class FakeUpstream(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        payload = json.dumps({"data": [{"id": "test-model"}]}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        payload = (b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
                   b"data: [DONE]\n\n")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class RouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = ThreadingHTTPServer(("127.0.0.1", 0), FakeUpstream)
        cls.upstream_thread = threading.Thread(target=cls.upstream.serve_forever, daemon=True)
        cls.upstream_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.upstream.shutdown()
        cls.upstream_thread.join()
        cls.upstream.server_close()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        upstream_url = "http://127.0.0.1:%d/v1" % self.upstream.server_address[1]
        self.app = Application(os.path.join(self.temp.name, "routes.sqlite"),
                               upstream_url, password="correct horse")
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

    def request(self, method, path, body=None, cookie=None, headers=None, host=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        request_headers = dict(headers or {})
        if body is not None:
            request_headers["Content-Type"] = "application/json"
            request_headers.setdefault("X-Chat-Request", "1")
        if cookie:
            request_headers["Cookie"] = cookie
        if host:
            request_headers["Host"] = host
        raw = None if body is None else json.dumps(body, separators=(",", ":")).encode("utf-8")
        connection.request(method, path, body=raw, headers=request_headers)
        response = connection.getresponse()
        data = response.read()
        result = (response.status, response.getheaders(), data)
        connection.close()
        return result

    @staticmethod
    def json_body(result):
        return json.loads(result[2].decode("utf-8"))

    def login(self):
        result = self.request("POST", "/api/login", {"password": "correct horse"})
        self.assertEqual(result[0], 200)
        set_cookie = dict((key.lower(), value) for key, value in result[1])["set-cookie"]
        self.assertIn("HttpOnly", set_cookie)
        self.assertIn("SameSite=Strict", set_cookie)
        return set_cookie.split(";", 1)[0]

    def test_login_cookie_authentication_and_logout(self):
        session = self.request("GET", "/api/session")
        self.assertEqual(self.json_body(session), {"authenticated": False, "password_required": True})
        denied = self.request("GET", "/api/chats")
        self.assertEqual(denied[0], 401)
        cookie = self.login()
        self.assertEqual(self.json_body(self.request("GET", "/api/chats", cookie=cookie)), {"chats": []})
        cleared = self.request("POST", "/api/logout", {}, cookie=cookie)
        self.assertEqual(cleared[0], 200)
        self.assertIn("Max-Age=0", dict((key.lower(), value) for key, value in cleared[1])["set-cookie"])
        self.assertEqual(self.request("GET", "/api/chats", cookie=cookie)[0], 401)

    def test_cross_origin_host_and_request_header_are_rejected(self):
        cookie = self.login()
        foreign_origin = self.request("POST", "/api/chats", {}, cookie,
                                      {"Origin": "https://evil.example"})
        self.assertEqual(foreign_origin[0], 403)
        foreign_host = self.request("GET", "/api/session", host="evil.example")
        self.assertEqual(foreign_host[0], 403)
        missing_header = self.request("POST", "/api/chats", {}, cookie,
                                      {"X-Chat-Request": "0"})
        self.assertEqual(missing_header[0], 403)
        cross_site = self.request("PATCH", "/api/chats/missing", {}, cookie,
                                  {"Sec-Fetch-Site": "cross-site"})
        self.assertEqual(cross_site[0], 403)

    def test_chat_create_patch_delete_and_invalid_inputs(self):
        cookie = self.login()
        created = self.request("POST", "/api/chats", {"title": "Review", "folder": "work"}, cookie)
        self.assertEqual(created[0], 201)
        chat = self.json_body(created)
        cid = chat["id"]
        patched = self.request("PATCH", "/api/chats/" + cid,
                               {"title": "Updated", "pinned": True}, cookie)
        self.assertEqual(patched[0], 200)
        self.assertEqual(self.json_body(patched)["title"], "Updated")
        self.assertTrue(self.json_body(patched)["pinned"])
        deleted = self.request("DELETE", "/api/chats/" + cid, {}, cookie)
        self.assertEqual(deleted[0], 200)
        self.assertEqual(self.request("GET", "/api/chats/" + cid, cookie=cookie)[0], 404)
        for body in ({"title": 12}, {"unknown": True}):
            with self.subTest(body=body):
                self.assertEqual(self.request("POST", "/api/chats", body, cookie)[0], 400)
        fresh = self.json_body(self.request("POST", "/api/chats", {}, cookie))["id"]
        self.assertEqual(self.request("POST", f"/api/chats/{fresh}/runs", {"content": "  "}, cookie)[0], 400)
        self.assertEqual(self.request("POST", f"/api/chats/{fresh}/runs",
                                      {"content": "x", "settings": {"top_p": 2}}, cookie)[0], 400)

    def test_attachments_are_validated_and_names_are_sanitized(self):
        cookie = self.login()
        cid = self.json_body(self.request("POST", "/api/chats", {}, cookie))["id"]
        invalid = [{"name": "x.svg", "type": "image/svg+xml", "data": "data:image/svg+xml;base64,PHN2Zz4="}]
        self.assertEqual(self.request("POST", f"/api/chats/{cid}/runs",
                                      {"content": "", "attachments": invalid}, cookie)[0], 400)
        text = [{"name": "../notes.txt", "type": "text/plain", "text": "hello"}]
        result = self.request("POST", f"/api/chats/{cid}/runs",
                              {"content": "read", "attachments": text}, cookie)
        self.assertEqual(result[0], 201)
        run = self.json_body(result)["run"]
        message = next(message for message in self.app.store.snapshot(cid)["messages"]
                       if message["role"] == "user")
        self.assertEqual(message["attachments"][0]["name"], "notes.txt")
        self.assertEqual(message["attachments"][0]["text"], "hello")
        end = time.monotonic() + 3
        with self.app.store.changed:
            while self.app.store.run(run["id"])["status"] not in ("completed", "failed"):
                remaining = end - time.monotonic()
                self.assertGreater(remaining, 0, "fake upstream did not finish the reply")
                self.app.store.changed.wait(remaining)
        self.assertEqual(self.app.store.run(run["id"])["status"], "completed")

    def _finished_branch_export(self, cookie):
        created = self.json_body(self.request("POST", "/api/chats", {"title": "Branches"}, cookie))
        chat_id = created["id"]
        settings = {"model": "", "system": "", "temperature": 0.7, "top_p": 0.95,
                    "top_k": 40, "max_tokens": 4096, "tools": []}
        run = self.app.store.start(chat_id, {"content": "question"}, settings)
        assistant = self.app.store.message(run["message_id"])
        assistant.update(content="first answer", status="completed")
        self.app.store.publish(run["id"], "message", assistant)
        self.app.store.publish(run["id"], "state", dict(self.app.store.run(run["id"]), status="completed"))
        user = next(message for message in self.app.store.snapshot(chat_id)["messages"] if message["role"] == "user")
        alternate = self.app.store.start(chat_id, {"regenerate": True, "parent_id": user["id"]}, settings)
        other = self.app.store.message(alternate["message_id"])
        other.update(content="second answer", status="completed")
        self.app.store.publish(alternate["id"], "message", other)
        self.app.store.publish(alternate["id"], "state", dict(self.app.store.run(alternate["id"]), status="completed"))
        return chat_id

    def test_export_import_preserves_branches_without_runs(self):
        cookie = self.login()
        source_id = self._finished_branch_export(cookie)
        exported = self.request("GET", f"/api/chats/{source_id}/export", cookie=cookie)
        self.assertEqual(exported[0], 200)
        payload = self.json_body(exported)
        imported = self.request("POST", "/api/import", payload, cookie)
        self.assertEqual(imported[0], 201)
        imported_id = self.json_body(imported)["id"]
        snapshot = self.json_body(self.request("GET", f"/api/chats/{imported_id}", cookie=cookie))
        self.assertEqual({m["content"] for m in snapshot["messages"] if m["role"] == "assistant"},
                         {"first answer", "second answer"})
        answers = [m for m in snapshot["messages"] if m["role"] == "assistant"]
        user = next(m for m in snapshot["messages"] if m["role"] == "user")
        self.assertEqual({m["parent_id"] for m in answers}, {user["id"]})
        self.assertEqual(snapshot["runs"], [])
        self.assertEqual(snapshot["chat"]["active_leaf"], self.json_body(imported)["active_leaf"])

    def test_import_converts_pending_approval_to_interrupted(self):
        cookie = self.login()
        payload = {"version": 1, "chat": {"title": "Imported active", "active_leaf": "assistant-1"},
                   "messages": [
                       {"id": "user-1", "parent_id": None, "role": "user", "content": "question"},
                       {"id": "assistant-1", "parent_id": "user-1", "role": "assistant",
                        "content": "", "status": "approval", "tool_calls": [
                            {"id": "call-1", "name": "dangerous", "arguments": "{}", "status": "approval"}]}]}
        imported = self.request("POST", "/api/import", payload, cookie)
        self.assertEqual(imported[0], 201)
        snapshot = self.json_body(self.request("GET", f"/api/chats/{self.json_body(imported)['id']}", cookie=cookie))
        answer = next(message for message in snapshot["messages"] if message["role"] == "assistant")
        self.assertEqual(answer["status"], "interrupted")
        self.assertEqual(answer["tool_calls"][0]["status"], "interrupted")
        self.assertEqual(snapshot["runs"], [])

    def test_invalid_import_metrics_and_tool_calls_leave_no_partial_chat(self):
        cookie = self.login()
        before = len(self.app.store.chats())
        base = {"version": 1, "chat": {"title": "Must not persist", "active_leaf": "message-2"},
                "messages": [
                    {"id": "message-1", "parent_id": None, "role": "user", "content": "q"},
                    {"id": "message-2", "parent_id": "message-1", "role": "assistant", "content": "a"}]}
        invalid_cases = []
        bad_metrics = json.loads(json.dumps(base))
        bad_metrics["messages"][1]["usage"] = []
        invalid_cases.append(bad_metrics)
        bad_call = json.loads(json.dumps(base))
        bad_call["messages"][1]["tool_calls"] = [{"id": "call", "name": 7, "arguments": "{}"}]
        invalid_cases.append(bad_call)
        for payload in invalid_cases:
            with self.subTest(payload=payload):
                response = self.request("POST", "/api/import", payload, cookie)
                self.assertEqual(response[0], 400)
                self.assertEqual(len(self.app.store.chats()), before)

    def test_events_replay_over_http_and_unknown_routes(self):
        cookie = self.login()
        chat = self.json_body(self.request("POST", "/api/chats", {}, cookie))
        settings = {"model": "", "system": "", "temperature": 0.7, "top_p": 0.95,
                    "top_k": 40, "max_tokens": 4096, "tools": []}
        run = self.app.store.start(chat["id"], {"content": "events"}, settings)
        self.app.store.publish(run["id"], "delta", {"message_id": run["message_id"], "content": "hi", "reasoning": ""})
        self.app.store.publish(run["id"], "state", dict(self.app.store.run(run["id"]), status="completed"))
        first = self.request("GET", f"/api/runs/{run['id']}/events?after=0", cookie=cookie)
        self.assertEqual(first[0], 200)
        self.assertIn(b"event: update", first[2])
        self.assertIn(b'"content":"hi"', first[2])
        last = self.app.store.run(run["id"])["last_seq"]
        replay = self.request("GET", f"/api/runs/{run['id']}/events?after=0", cookie=cookie,
                              headers={"Last-Event-ID": str(last - 1)})
        self.assertIn(("id: %d" % last).encode(), replay[2])
        self.assertEqual(self.request("GET", "/api/no-such-route", cookie=cookie)[0], 404)


if __name__ == "__main__":
    unittest.main()
