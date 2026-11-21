import http.client
import io
import json
import os
import tempfile
import threading
import time
import unittest
import zipfile
from base64 import b64encode
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

    def test_draft_diff_reads_current_saved_file_without_writing(self):
        cookie = self.login()
        path = "src/main.py"
        body = {"path": path, "content": "proposed"}
        route = "/api/workspace/diff"
        self.assertEqual(self.request("POST", route, body)[0], 401)
        self.assertEqual(self.request("POST", route, body, cookie)[0], 404)
        for revision, text in enumerate(("first", "saved\n")):
            self.assertEqual(self.request("POST", "/api/workspace/file",
                                          {"path": path, "content": text, "revision": revision}, cookie)[0], 200)
        snapshot = self.json_body(self.request("GET", "/api/workspace", cookie=cookie))
        result = self.request("POST", route, body, cookie)
        self.assertEqual(result[0], 200)
        self.assertEqual(self.json_body(result)["diff"], "--- saved/src/main.py\n+++ draft/src/main.py\n"
                         "@@ -1 +1 @@\n-saved\n+proposed\n\\ No newline at end of file\n")
        self.assertEqual(self.json_body(self.request("POST", route, dict(body, content="saved\n"), cookie))["diff"], "")
        for invalid in (dict(body, path="../src/main.py"), dict(body, content=None)):
            self.assertEqual(self.request("POST", route, invalid, cookie)[0], 400)
        self.assertEqual(self.json_body(self.request("GET", "/api/workspace", cookie=cookie)), snapshot)
        self.assertEqual(self.json_body(self.request("GET", "/api/workspace/file?path=src/main.py", cookie=cookie))["content"], "saved\n")

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
                     {"path": "src/main.py", "content": "new", "revision": current}, self.cookie)
        diff = self.request("GET", "/api/workspace/diff?checkpoint=" + checkpoint_id, cookie=self.cookie)
        self.assertIn("-old\n+new\n\\ No newline at end of file\n", self.json_body(diff)["diff"])
        restored = self.request("POST", "/api/workspace/checkpoints/" + checkpoint_id + "/restore",
                                {"revision": current + 1}, self.cookie)
        self.assertEqual(restored[0], 200)
        read = self.request("GET", "/api/workspace/file?path=src%2Fmain.py", cookie=self.cookie)
        self.assertEqual(self.json_body(read)["content"], "old\n")
        removed = self.request("POST", "/api/workspace/delete",
                               {"path": "src/main.py", "revision": current + 2}, self.cookie)
        self.assertEqual(removed[0], 200)

    def test_import_local_files_preserves_paths_text_and_revision(self):
        route = "/api/workspace/files"
        files = [{"path": "site/index.html", "content": '<link rel="stylesheet" href="css/site.css"><script src="js/app.js"></script>'},
                 {"path": "site/js/app.js", "content": "const loaded = true;"},
                 {"path": "site/css/site.css", "content": "\ufeffbody {}\r\n"},
                 {"path": "site/empty.txt", "content": ""}]
        body = {"revision": 0, "files": files}
        self.assertEqual(self.request("POST", route, body)[0], 401)
        cookie = self.login()
        self.assertEqual(self.request("POST", route, body, cookie, {"Origin": "https://other.invalid"})[0], 403)
        result = self.request("POST", route, body, cookie)
        self.assertEqual(result[0], 200, result[2])
        self.assertEqual(self.json_body(result)["revision"], 1)
        paths = "&".join("path=" + file["path"] for file in files)
        self.assertEqual(self.json_body(self.request("GET", route + "?" + paths, cookie=cookie))["files"],
                         [{"path": file["path"], "text": file["content"]} for file in files])
        preview = self.request("POST", "/api/previews", {"html": files[0]["content"], "workspace_path": files[0]["path"]}, cookie)
        self.assertEqual(preview[0], 200, preview[2])
        assembled = self.json_body(preview)["workspace"]
        self.assertEqual(assembled["revision"], 1)
        self.assertEqual(assembled["assets"], ["site/css/site.css", "site/js/app.js"])
        self.assertEqual(self.request("POST", route, body, cookie)[0], 409)
        self.assertEqual(self.json_body(self.request("GET", route, cookie=cookie))["revision"], 1)

    def test_saved_workspace_file_picker_lists_metadata_and_reads_ordered_batches(self):
        route = "/api/workspace/files"
        self.assertEqual(self.request("GET", route)[0], 401)
        self.cookie = self.login()
        decoded = self.json_body(self.request("GET", route, cookie=self.cookie))
        self.assertEqual(decoded, {"revision": 0, "files": []})
        self.app.workspace.write("src/index.js", "source", 0)
        self.app.workspace.write("docs/notes+facts.md", "facts \u00e9", 1)
        self.app.workspace.checkpoint("Reference")
        self.assertEqual(self.request("GET", route, cookie=self.cookie, headers={"Origin": "https://other.invalid"})[0], 403)
        listing = self.request("GET", route, cookie=self.cookie)
        self.assertEqual(listing[0], 200)
        self.assertEqual(self.json_body(listing), {"revision": 2,
                         "files": [{"path": "docs/notes+facts.md", "size": 8}, {"path": "src/index.js", "size": 6}]})
        batch = route + "?path=src%2Findex.js&path=docs%2Fnotes%2Bfacts.md"
        selected = self.request("GET", batch, cookie=self.cookie)
        self.assertEqual(selected[0], 200)
        self.assertEqual(self.json_body(selected), {"revision": 2,
                         "files": [{"path": "src/index.js", "text": "source"}, {"path": "docs/notes+facts.md", "text": "facts \u00e9"}]})
        self.app.workspace.write("src/index.js", "revised", 2)
        updated = self.json_body(self.request("GET", batch, cookie=self.cookie))
        self.assertEqual(updated["revision"], 3)
        self.assertEqual(updated["files"][0], {"path": "src/index.js", "text": "revised"})
        for query in ("path", "path=", "path=%GG", "path=%FF", "other=one", "path=%2E%2E%2Fescape",
                      "path=src%2Findex.js&path=src/index.js", "&".join(["path=src/index.js"] * 9)):
            with self.subTest(query=query):
                self.assertEqual(self.request("GET", route + "?" + query, cookie=self.cookie)[0], 400)
        missing = self.request("GET", route + "?path=src/index.js&path=missing.txt", cookie=self.cookie)
        self.assertEqual(missing[0], 404)
        self.assertEqual(set(self.json_body(missing)), {"error"})

    def test_project_archive_contains_selected_saved_files(self):
        workspace = self.app.workspace
        entries = {"src/brief.txt": "\ufeffFirst line\r\nAnother line\n", "notes/brief.txt": "",
                   "pages/caf\u00e9.html": "<p>\u00e9</p>", "other.txt": "Unselected"}
        workspace.import_files([dict(path=path, content=text) for path, text in entries.items()], 0)
        paths = ["pages/caf\u00e9.html", "notes/brief.txt", "src/brief.txt"]
        project = workspace.save_project(dict(name='Release / "review"', paths=paths, instructions="Private instructions"))
        route = "/api/workspace/projects/" + project["id"] + "/archive"
        cookie = self.login()
        before = workspace.snapshot()
        self.assertEqual(self.request("GET", route)[0], 401)
        status, headers, body = self.request("GET", route, cookie=cookie)
        self.assertEqual(status, 200, body[:100])
        self.assertEqual(headers["content-type"], "application/zip")
        self.assertEqual(headers["content-disposition"], 'attachment; filename="llmx-Release-review.zip"')
        self.assertEqual(int(headers["content-length"]), len(body))
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            self.assertEqual(archive.namelist(), paths)
            self.assertEqual([archive.read(path) for path in paths], [entries[path].encode() for path in paths])
        self.assertEqual(before, workspace.snapshot())
        workspace.write(paths[0], "", before["revision"], delete=True)
        missing = self.request("GET", route, cookie=cookie)
        self.assertEqual(missing[0], 409)
        self.assertIn(paths[0], self.json_body(missing)["error"])
        empty = workspace.save_project(dict(name="Empty", paths=[]))
        self.assertEqual(self.request("GET", route.replace(project["id"], empty["id"]), cookie=cookie)[0], 400)
        self.assertEqual(self.request("GET", route.replace(project["id"], "absent"), cookie=cookie)[0], 404)

    def test_projects_routes_auth_round_trip_and_version_conflicts(self):
        route = "/api/workspace/projects"
        self.assertEqual(self.request("GET", route)[0], 401)
        self.cookie = self.login()
        self.assertEqual(self.json_body(self.request("GET", route, cookie=self.cookie)), {"projects": []})
        self.app.workspace.write("src/index.js", "source", 0)
        self.app.workspace.write("tests/index.js", "test", 1)
        values = dict(name="Release", instructions="Use short examples.", paths=["tests/index.js", "src/index.js"])
        self.assertEqual(self.request("POST", route, values, self.cookie, {"X-Chat-Request": "0"})[0], 403)
        self.assertEqual(self.request("POST", route, values, self.cookie, {"Origin": "https://other.invalid"})[0], 403)
        created = self.request("POST", route, values, self.cookie)
        self.assertEqual(created[0], 201, created[2])
        project = self.json_body(created)["project"]
        project_route = route + "/" + project["id"]
        self.assertEqual(project, dict(id=project["id"], version=1, **values))
        read = self.request("GET", project_route + "/context", cookie=self.cookie)
        self.assertEqual(read[0], 200, read[2])
        self.assertEqual(self.json_body(read), {
            "project": project, "revision": 2,
            "files": [{"path": "tests/index.js", "text": "test"}, {"path": "src/index.js", "text": "source"}],
        })
        revised = dict(name="Release checks", paths=["src/index.js"], version=1)
        edited = self.request("PATCH", project_route, revised, self.cookie)
        self.assertEqual(edited[0], 200, edited[2])
        updated = self.json_body(edited)["project"]
        self.assertEqual(updated, dict(id=project["id"], version=2, name="Release checks", instructions="", paths=["src/index.js"]))
        self.assertEqual(self.request("PATCH", project_route, revised, self.cookie)[0], 409)
        self.assertEqual(self.request("DELETE", project_route, {"version": 1}, self.cookie)[0], 409)
        self.assertEqual(self.request("DELETE", project_route, {"version": True}, self.cookie)[0], 400)
        self.assertEqual(self.json_body(self.request("GET", route, cookie=self.cookie)), {"projects": [updated]})
        deleted = self.request("DELETE", project_route, {"version": 2}, self.cookie)
        self.assertEqual(deleted[0], 200, deleted[2])
        self.assertEqual(self.json_body(deleted), {"ok": True})
        self.assertEqual(self.request("GET", project_route + "/context", cookie=self.cookie)[0], 404)
        self.assertEqual(self.app.workspace.snapshot()["revision"], 2)

    def test_projects_context_refuses_missing_paths_atomically(self):
        self.cookie = self.login()
        base = "/api/workspace/projects"
        self.app.workspace.write("kept.md", "keep", 0)
        self.app.workspace.write("deleted.md", "remove", 1)
        body = dict(name="Docs", paths=["kept.md", "deleted.md"])
        project = self.json_body(self.request("POST", base, body, self.cookie))["project"]
        path = base + "/" + project["id"]
        self.app.workspace.write("deleted.md", None, 2, delete=True)
        absent = self.request("GET", path + "/context", cookie=self.cookie)
        self.assertEqual(absent[0], 409, absent[2])
        self.assertEqual(set(self.json_body(absent)), {"error"})
        self.assertIn("deleted.md", self.json_body(absent)["error"])
        self.assertEqual(self.request("POST", base, body, self.cookie)[0], 409)
        self.assertEqual(self.request("PATCH", path, dict(body, version=1), self.cookie)[0], 409)
        self.assertEqual(self.json_body(self.request("GET", base, cookie=self.cookie)), {"projects": [project]})
        repaired = self.request("PATCH", path, dict(name="Docs", paths=["kept.md"], version=1), self.cookie)
        self.assertEqual(repaired[0], 200, repaired[2])
        current = self.json_body(self.request("GET", path + "/context", cookie=self.cookie))
        self.assertEqual(current["revision"], 3)
        self.assertEqual(current["files"], [{"path": "kept.md", "text": "keep"}])

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
        self.assertIn("script-src 'unsafe-inline';", csp)
        self.assertIn(b"preview", preview[2])

    def test_project_preview_embeds_saved_assets_from_one_revision(self):
        self.cookie = self.login()
        css, js = "h1 { color: #123456; }", "console.log('</script>');"
        for path, content in (("site/theme.css", css), ("site/pages/boot script.js", js), ("private.js", "unrelated")):
            revision = self.app.workspace.snapshot()["revision"]
            self.app.workspace.write(path, content, revision)
        html = ('<!doctype html>\n<html class="project"><head>\n'
                '<link rel="stylesheet" href="../theme.css?v=3" media="screen">\n'
                '<script defer src="boot%20script.js" src="missing.js"></script>\n'
                '<script type="module" src="module.js"></script>\n'
                '<link rel="stylesheet" href="https://external.invalid/style.css"></head>\n'
                '<body>Unsaved editor body<textarea/><script src="example.js"></script></textarea></body></html>')
        request = self.request("POST", "/api/previews", {"html": html, "workspace_path": "site/pages/index.html"}, self.cookie)
        self.assertEqual(request[0], 200, request[2])
        preview = self.json_body(request)
        self.assertEqual(preview["workspace"], {"path": "site/pages/index.html", "revision": 3,
                                                "assets": ["site/theme.css", "site/pages/boot script.js"]})
        self.app.workspace.write("site/theme.css", "changed", 3)
        self.assertEqual(self.request("GET", preview["url"])[0], 401)
        result = self.request("GET", preview["url"], cookie=self.cookie)
        self.assertEqual(result[0], 200, result[2])
        csp = result[1]["content-security-policy"]
        self.assertIn("script-src 'unsafe-inline' data:;", csp)
        self.assertIn("style-src 'unsafe-inline' data:;", csp)
        self.assertIn("connect-src 'none'", csp)
        self.assertIn("sandbox allow-scripts", csp)
        self.assertNotIn("allow-same-origin", csp)
        css_url = b"data:text/css;charset=utf-8;base64," + b64encode(css.encode())
        js_url = b"data:text/javascript;charset=utf-8;base64," + b64encode(js.encode())
        self.assertTrue(result[2].startswith(b'<!doctype html>\n<html class="project">'))
        self.assertIn(b'<link rel="stylesheet" href="' + css_url + b'" media="screen">', result[2])
        self.assertIn(b'<script defer src="' + js_url + b'"></script>', result[2])
        self.assertIn(b'<script type="module" src="module.js"></script>', result[2])
        self.assertIn(b"Unsaved editor body", result[2])
        self.assertNotIn(b64encode(b"unrelated"), result[2])
        self.assertNotIn(b64encode(b"changed"), result[2])

    def test_project_preview_rejects_missing_escaping_and_oversized_assets(self):
        self.cookie = self.login()
        self.app.workspace.write("page/large.js", "x" * (800 * 1024), 0)
        for source, message in (
            ('<link rel="stylesheet" href="missing.css">', "page/missing.css"),
            ('<script src="../../escape.js"></script>', "stay inside the workspace"),
            ('<script src="large.js"></script>', "assembled Workspace preview"),
        ):
            with self.subTest(message=message):
                result = self.request("POST", "/api/previews", {"html": source, "workspace_path": "page/index.html"}, self.cookie)
                self.assertEqual(result[0], 400, result[2])
                self.assertIn(message, self.json_body(result)["error"])
        self.assertEqual(self.app.previews, {})

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
