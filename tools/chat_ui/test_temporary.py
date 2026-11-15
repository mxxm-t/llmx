import json
from pathlib import Path
import sqlite3
import threading
import time
import unittest
from unittest.mock import patch

import test_server as routes
from engine import TERMINAL
from store import Store
from temporary import TemporaryChats


class TemporaryUpstream(routes.FakeUpstream):
    requests = []
    waiting = threading.Event()
    release = threading.Event()

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append(payload)
        text = payload["messages"][-1].get("content")
        delta = {"content": "ok"}
        if text == "force-tool":
            delta = {"tool_calls": [{"index": 0, "function": {"name": "workspace_write", "arguments": "{}"}}]}
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        try:
            self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": delta}]}).encode() + b"\n\n")
            self.wfile.flush()
            if text == "wait":
                type(self).waiting.set()
                type(self).release.wait(5)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except OSError:
            pass


class TemporaryRouteTests(unittest.TestCase):
    request = routes.RouteTests.request
    json_body = staticmethod(routes.RouteTests.json_body)
    login = routes.RouteTests.login
    setUp = routes.RouteTests.setUp
    tearDown = routes.RouteTests.tearDown

    @classmethod
    def setUpClass(cls):
        cls.upstream = routes.ThreadingHTTPServer(("127.0.0.1", 0), TemporaryUpstream)
        cls.upstream_thread = threading.Thread(target=cls.upstream.serve_forever, daemon=True)
        cls.upstream_thread.start()

    tearDownClass = classmethod(routes.RouteTests.tearDownClass.__func__)

    def create(self, cookie, **body):
        response = self.request("POST", "/api/chats", dict(temporary=True, **body), cookie)
        self.assertEqual(response[0], 201, response[2])
        chat = self.json_body(response)
        self.assertTrue(chat["temporary"])
        self.assertTrue(chat["id"].startswith("tmp_"))
        return chat["id"]

    def start_reply(self, cookie, cid, **body):
        response = self.request("POST", f"/api/chats/{cid}/runs", body, cookie)
        self.assertEqual(response[0], 201, response[2])
        rid = self.json_body(response)["run"]["id"]
        self.assertTrue(rid.startswith("tmp_"))
        return rid

    def finish(self, cookie, rid):
        response = self.request("GET", f"/api/runs/{rid}/events", cookie=cookie)
        self.assertEqual(response[0], 200)
        events = [json.loads(line[6:]) for line in response[2].decode().splitlines() if line.startswith("data: ")]
        self.assertTrue(events)
        self.assertIn(events[-1]["data"]["status"], TERMINAL)
        return events

    def test_temporary_run_sse_export_never_reach_durable_storage(self):
        cookie = self.login()
        sentinel = "private-temporary-unique-7bce420af0"
        config = self.json_body(self.request("GET", "/api/config", cookie=cookie))
        self.assertEqual(config["temporary"], {"enabled": True, "ttl_seconds": 3600, "tools": False, "max_chats": 16})
        regular = self.json_body(self.request("POST", "/api/chats", {"title": "saved conversation"}, cookie))
        cid = self.create(cookie, settings={"system": sentinel + "-settings"})
        self.assertEqual(self.request("PATCH", f"/api/chats/{cid}", {"title": sentinel + "-title"}, cookie)[0], 200)
        rid = self.start_reply(cookie, cid, content=sentinel, attachments=[{"name": "private.txt", "text": sentinel + "-attachment"}])
        events = self.finish(cookie, rid)
        self.assertTrue(any(event["kind"] == "delta" and event["data"]["content"] == "ok" for event in events))
        snapshot = self.json_body(self.request("GET", f"/api/chats/{cid}", cookie=cookie))
        self.assertEqual(snapshot["runs"][0]["status"], "completed")
        self.assertEqual(snapshot["messages"][0]["attachments"][0]["text"], sentinel + "-attachment")
        exported = self.json_body(self.request("GET", f"/api/chats/{cid}/export", cookie=cookie))
        self.assertEqual(exported["messages"][0]["content"], sentinel)
        saved = self.json_body(self.request("GET", "/api/chats", cookie=cookie))["chats"]
        self.assertEqual([chat["id"] for chat in saved], [regular["id"]])
        with self.assertRaises(KeyError):
            self.app.store.chat(cid)
        backup = self.request("GET", "/api/backup", cookie=cookie)
        self.assertEqual(backup[0], 200)
        self.assertNotIn(sentinel.encode(), backup[2])
        for file in Path(self.temp.name).glob("routes.sqlite*"):
            self.assertNotIn(sentinel.encode(), file.read_bytes(), file.name)
        self.assertEqual(self.app.temporary.store.db.execute("PRAGMA database_list").fetchone()[2], "")
        self.assertEqual(self.app.temporary.store.db.execute("PRAGMA journal_mode").fetchone()[0], "memory")
        self.assertEqual(self.app.temporary.store.db.execute("PRAGMA temp_store").fetchone()[0], 2)
        self.assertEqual(self.request("POST", f"/api/chats/{regular['id']}/runs", {"content": "saved prompt"}, cookie)[0], 201)

    def test_delete_and_expiry_remove_messages_runs_and_events(self):
        cookie = self.login()
        for expire in (False, True):
            cid = self.create(cookie)
            rid = self.start_reply(cookie, cid, content="expire me")
            self.finish(cookie, rid)
            if expire:
                self.app.temporary.accessed[cid] = time.monotonic() - 3601
                self.app.temporary.expire()
                self.assertEqual(self.app.temporary.store.run(rid)["status"], "completed")
                with patch("temporary.time.time", return_value=time.time() + 3601):
                    self.app.temporary.expire()
            else:
                self.assertEqual(self.request("DELETE", f"/api/chats/{cid}", {}, cookie)[0], 200)
            self.assertEqual(self.request("GET", f"/api/chats/{cid}", cookie=cookie)[0], 404)
            self.assertEqual(self.request("GET", f"/api/runs/{rid}/events", cookie=cookie)[0], 404)
            self.assertNotIn(cid, self.app.temporary.accessed)
        for table in ("chats", "messages", "runs", "events"):
            self.assertEqual(self.app.temporary.store.db.execute(f"SELECT count(*) FROM {table}").fetchone()[0], 0)

    def test_limit_expiry_and_access_refresh(self):
        cookie = self.login()
        ids = [self.create(cookie) for _ in range(16)]
        self.assertEqual(self.request("POST", "/api/chats", {"temporary": True}, cookie)[0], 400)
        self.app.temporary.accessed[ids[0]] = time.monotonic() - 3500
        self.assertEqual(self.request("GET", f"/api/chats/{ids[0]}", cookie=cookie)[0], 200)
        self.assertLess(time.monotonic() - self.app.temporary.accessed[ids[0]], 2)
        self.app.temporary.accessed[ids[1]] = time.monotonic() - 3601
        self.create(cookie)
        self.assertEqual(len(self.app.temporary.accessed), 16)
        self.assertEqual(self.request("GET", f"/api/chats/{ids[1]}", cookie=cookie)[0], 404)
        for flag in ("true", 1, None):
            self.assertEqual(self.request("POST", "/api/chats", {"temporary": flag}, cookie)[0], 400)

    def test_tools_rejected_on_create_patch_run_and_unsolicited_calls_cannot_execute(self):
        cookie = self.login()
        self.assertEqual(self.request("POST", "/api/chats", {"temporary": True, "settings": {"tools": ["workspace_write"]}}, cookie)[0], 400)
        cid = self.create(cookie)
        self.assertEqual(self.request("PATCH", f"/api/chats/{cid}", {"settings": {"tools": ["workspace_write"]}}, cookie)[0], 400)
        self.assertEqual(self.request("POST", f"/api/chats/{cid}/runs", {"content": "bad", "settings": {"tools": ["workspace_write"]}}, cookie)[0], 400)
        self.assertEqual(self.app.temporary.store.snapshot(cid)["messages"], [])
        with patch.object(self.app.tools, "list_tools", side_effect=AssertionError("MCP discovered")), \
                patch.object(self.app.tools, "call_tool", side_effect=AssertionError("Workspace changed")):
            rid = self.start_reply(cookie, cid, content="force-tool")
            events = self.finish(cookie, rid)
        self.assertTrue(any(event["kind"] == "tool" and event["data"]["call"]["status"] == "failed" for event in events))
        self.assertNotIn("tools", TemporaryUpstream.requests[-1])
        self.assertEqual(self.request("POST", f"/api/runs/{rid}/approve", {"call_id": "missing", "allow": True}, cookie)[0], 400)

    def test_active_runs_survive_expiry_cancel_and_shutdown_removes_temporary_data(self):
        cookie = self.login()
        TemporaryUpstream.waiting.clear()
        TemporaryUpstream.release.clear()
        self.addCleanup(TemporaryUpstream.release.set)
        cid = self.create(cookie)
        rid = self.start_reply(cookie, cid, content="wait")
        self.assertTrue(TemporaryUpstream.waiting.wait(2))
        self.app.temporary.accessed[cid] = time.monotonic() - 3601
        self.app.temporary.expire()
        self.assertEqual(self.app.temporary.store.run(rid)["status"], "running")
        self.assertEqual(self.request("DELETE", f"/api/chats/{cid}", {}, cookie)[0], 400)
        self.assertEqual(self.request("POST", f"/api/runs/{rid}/cancel", {}, cookie)[0], 200)
        self.assertEqual(self.finish(cookie, rid)[-1]["data"]["status"], "cancelled")
        TemporaryUpstream.release.set()
        self.app.temporary.close()
        self.assertFalse(self.app.temporary.thread.is_alive())
        self.assertEqual(self.app.temporary.accessed, {})
        with self.assertRaises(sqlite3.ProgrammingError):
            self.app.temporary.store.chats()
        replacement = TemporaryChats(self.app.upstream)
        try:
            with self.assertRaises(KeyError):
                replacement.access(cid)
            durable = Store(self.app.store.database_path)
            try:
                self.assertEqual(durable.chats(), [])
            finally:
                durable.close()
        finally:
            replacement.close()

    def test_server_restart_loses_only_temporary_conversations(self):
        cookie = self.login()
        cid = self.create(cookie)
        rid = self.start_reply(cookie, cid, content="vanishes after restart")
        self.finish(cookie, rid)
        saved = self.json_body(self.request("POST", "/api/chats", {"title": "survives restart"}, cookie))
        database, upstream = self.app.store.database_path, self.app.upstream
        self.httpd.shutdown()
        self.thread.join()
        self.httpd.server_close()
        self.app.engine.shutdown()
        self.app.store.close()
        self.app = routes.Application(database, upstream, password="correct horse")
        self.httpd = routes.Server(("127.0.0.1", 0), self.app)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.httpd.server_address[1]
        cookie = self.login()
        self.assertEqual(self.request("GET", f"/api/chats/{cid}", cookie=cookie)[0], 404)
        self.assertEqual(self.request("GET", f"/api/runs/{rid}/events", cookie=cookie)[0], 404)
        chats = self.json_body(self.request("GET", "/api/chats", cookie=cookie))["chats"]
        self.assertEqual([chat["id"] for chat in chats], [saved["id"]])

    def test_shutdown_cancels_an_active_temporary_run(self):
        cookie = self.login()
        TemporaryUpstream.waiting.clear()
        TemporaryUpstream.release.clear()
        self.addCleanup(TemporaryUpstream.release.set)
        cid = self.create(cookie)
        self.start_reply(cookie, cid, content="wait")
        self.assertTrue(TemporaryUpstream.waiting.wait(2))
        started = time.monotonic()
        self.app.temporary.close()
        self.assertLess(time.monotonic() - started, 2)
        self.assertEqual(self.app.temporary.engine.controls, {})
        self.assertEqual(self.app.temporary.engine.threads, set())
        with self.assertRaises(sqlite3.ProgrammingError):
            self.app.temporary.store.chats()


if __name__ == "__main__":
    unittest.main()
