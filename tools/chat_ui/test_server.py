import http.client
import json
import os
from pathlib import Path
import queue
import socket
import tempfile
import threading
import time
import unittest
from unittest import mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import install_pdf_reader
import server as chat_server
import monitor as chat_monitor
from server import Application, Server


class FakeUpstream(BaseHTTPRequestHandler):
    models = ["test-model"]
    model_error = False
    posts = []

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path.endswith("/models") and type(self).model_error:
            payload = json.dumps({"error": "model server unavailable"}).encode("utf-8")
            self.send_response(503)
        else:
            payload = json.dumps({"data": [{"id": model} for model in type(self).models]}).encode("utf-8")
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        type(self).posts.append((self.path, json.loads(raw) if raw else {}))
        payload = (b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
                   b"data: [DONE]\n\n")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class ServerAdmissionTests(unittest.TestCase):
    def test_cli_reports_invalid_https_origin_and_overlong_password_without_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            password_file = os.path.join(directory, "password.txt")
            for password, origin, error in (("x" * 1025, "https://chat.example", "12 to 1024"),
                                             ("correct horse", "http://chat.example", "HTTPS")):
                with self.subTest(error=error):
                    with open(password_file, "w", encoding="utf-8") as file:
                        file.write(password)
                    argv = ["server.py", "--host", "0.0.0.0", "--password-file", password_file,
                            "--public-origin", origin, "--data-dir", directory]
                    app = mock.MagicMock(password="hash")
                    with mock.patch.object(chat_server.sys, "argv", argv), \
                            mock.patch.object(chat_server, "Application", return_value=app), \
                            mock.patch.object(chat_server, "Connection"), \
                            mock.patch.object(chat_server.sys, "stderr") as stderr:
                        with self.assertRaises(SystemExit) as exc:
                            chat_server.main()
                        self.assertEqual(exc.exception.code, 2)
                        self.assertIn(error, "".join(call.args[0] for call in stderr.write.call_args_list))
                    app.monitor.collector.start.assert_not_called()

    def test_invalid_deployment_settings_do_not_bind_or_start_collector(self):
        for host, password, origin in (("0.0.0.0", None, "https://chat.example"),
                                       ("0.0.0.0", "hash", None),
                                       ("127.0.0.1", "hash", "https://chat.example:invalid"),
                                       ("127.0.0.1", "hash", "https://chat.example/path")):
            with self.subTest(host=host, origin=origin):
                app = mock.MagicMock(password=password)
                with mock.patch.object(ThreadingHTTPServer, "__init__", autospec=True) as init:
                    with self.assertRaises(ValueError):
                        Server((host, 0), app, origin)
                    init.assert_not_called()
                app.monitor.collector.start.assert_not_called()

    def test_failed_bind_or_collector_start_closes_socket_without_closing_borrowed_app(self):
        bind = ThreadingHTTPServer.server_bind
        for fail_bind in (True, False):
            with self.subTest(fail_bind=fail_bind):
                app = mock.MagicMock()
                sockets = []

                def server_bind(server):
                    sockets.append(server.socket)
                    if fail_bind:
                        raise OSError("bind failed")
                    bind(server)

                app.monitor.collector.start.side_effect = RuntimeError("collector start failed")
                with mock.patch.object(ThreadingHTTPServer, "server_bind", autospec=True, side_effect=server_bind):
                    with self.assertRaises((OSError, RuntimeError)):
                        Server(("127.0.0.1", 0), app)
                self.assertEqual(sockets[0].fileno(), -1)
                app.temporary.close.assert_not_called()
                app.workspace.close.assert_not_called()
                app.monitor.stop.assert_not_called()

    def test_thread_start_and_handler_errors_release_connection_slots(self):
        workers = queue.Queue()
        with mock.patch.object(chat_server, "MAX_CONNECTIONS", 1):
            server = Server(("127.0.0.1", 0), mock.MagicMock())
        try:
            request = mock.Mock()
            address = ("127.0.0.1", 1234)
            with mock.patch.object(server, "get_request", return_value=(request, address)), \
                    mock.patch.object(server, "handle_error"), \
                    mock.patch.object(server, "shutdown_request") as shutdown, \
                    mock.patch.object(threading.Thread, "start", side_effect=RuntimeError("thread start failed")):
                server._handle_request_noblock()
                shutdown.assert_called_once_with(request)

            def finish_request(_request, _address):
                workers.put(threading.current_thread())
                raise RuntimeError("handler failed")

            with mock.patch.object(server, "finish_request", side_effect=finish_request), \
                    mock.patch.object(server, "handle_error") as error, \
                    mock.patch.object(server, "shutdown_request") as shutdown:
                for _ in range(2):
                    server.process_request(request, address)
                    worker = workers.get(timeout=1)
                    worker.join(timeout=1)
                    self.assertFalse(worker.is_alive())
                self.assertEqual(error.call_count, 2)
                self.assertEqual(shutdown.call_count, 2)
        finally:
            server.server_close()

    def test_nonloopback_http_public_origin_is_rejected_before_bind_and_https_is_accepted(self):
        secure_app = mock.MagicMock()
        secure_app.password = "password-hash"
        https_server = Server(("0.0.0.0", 0), secure_app, "https://chat.example")
        try:
            self.assertTrue(https_server.secure)
            secure_app.monitor.collector.start.assert_called_once_with()
        finally:
            https_server.server_close()

        app = mock.MagicMock()
        app.password = "password-hash"
        with mock.patch.object(ThreadingHTTPServer, "__init__", autospec=True,
                               side_effect=AssertionError("binding attempted")) as init:
            with self.assertRaisesRegex(ValueError, "HTTPS"):
                Server(("0.0.0.0", 0), app, "http://chat.example")
            init.assert_not_called()
        app.monitor.collector.start.assert_not_called()

    def test_connection_cap_closes_excess_socket_and_recovers_after_slots_free(self):
        entered = queue.Queue()
        release = threading.Event()

        def hold_handler(_handler):
            entered.put(threading.current_thread())
            release.wait(3)

        app = mock.MagicMock()
        server = None
        thread = None
        clients = []
        workers = []
        third_closed = False
        try:
            with mock.patch.object(chat_server, "MAX_CONNECTIONS", 2, create=True), \
                    mock.patch.object(chat_server.Handler, "handle", hold_handler):
                server = Server(("127.0.0.1", 0), app)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                for _ in range(2):
                    client = socket.create_connection(server.server_address, timeout=1)
                    clients.append(client)
                    client.settimeout(1)
                    workers.append(entered.get(timeout=1))

                excess = socket.create_connection(server.server_address, timeout=1)
                clients.append(excess)
                excess.settimeout(0.75)
                try:
                    third_closed = excess.recv(1) == b""
                except ConnectionResetError:
                    third_closed = True
                except TimeoutError:
                    third_closed = False

                release.set()
                for worker in workers:
                    worker.join(timeout=1)
                    self.assertFalse(worker.is_alive(), "handler did not release its connection slot")

            connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=1)
            connection.request("GET", "/unknown")
            recovery = connection.getresponse()
            self.assertEqual(recovery.status, 404)
            recovery.read()
            connection.close()
        finally:
            release.set()
            for client in clients:
                client.close()
            if thread is not None:
                server.shutdown()
                thread.join(timeout=2)
            if server is not None:
                server.server_close()
        self.assertTrue(third_closed, "excess connection remained open instead of being refused")


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
        FakeUpstream.models = ["test-model"]
        FakeUpstream.model_error = False
        FakeUpstream.posts = []
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

    def request(self, method, path, body=None, cookie=None, headers=None, host=None, source_address=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3,
                                                source_address=source_address)
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

    def test_settings_defaults_are_published_and_missing_reply_limit_is_server_default(self):
        cookie = self.login()
        response = self.json_body(self.request("GET", "/api/config", cookie=cookie))
        self.assertEqual(response["settings_defaults"], chat_server.DEFAULTS)
        self.assertIsNone(response["settings_defaults"]["max_tokens"])
        missing = chat_server.settings_of({"model": "test-model"})
        self.assertIsNone(missing["max_tokens"])
        explicit = chat_server.settings_of({"model": "test-model", "max_tokens": 2048})
        self.assertEqual(explicit["max_tokens"], 2048)

    def test_optional_feature_install_requires_owner_and_serializes_visible_retries(self):
        url = "/api/features"
        install_url = url + "/pdf-reader/install"
        cookie = self.login()
        reader = chat_server.pdf_reader
        package = Path(self.temp.name) / "optional-reader"
        entered, release = threading.Event(), threading.Event()
        attempts = []

        def install(directory):
            attempts.append(directory)
            if len(attempts) == 1:
                entered.set()
                if not release.wait(3):
                    raise RuntimeError("fixture timed out")
                raise OSError("private path and URL: fixture was unable to download")
            directory.mkdir()
            for name in reader.MODULES:
                (directory / name).write_text("fixture", encoding="utf-8")
            for path in reader.RESOURCE_PATHS:
                (directory / path).mkdir()
            (directory / reader.INSTALL_MARKER).write_bytes(reader.INSTALL_RECORD)

        def status(method="GET", path=url, body=None):
            result = self.request(method, path, body, cookie)
            self.assertEqual(result[0], 200, result[2])
            self.assertEqual(set(self.json_body(result)), {"features"})
            feature, = self.json_body(result)["features"]
            self.assertEqual(feature["id"], "pdf-reader")
            self.assertEqual(feature["name"], "PDF text import")
            self.assertEqual(feature["version"], reader.VERSION)
            self.assertTrue(feature["description"])
            return feature

        def wait_status(state):
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                value = status()
                if value["state"] == state:
                    return value
                threading.Event().wait(0.01)
            self.fail(f"feature did not reach {state}")

        with mock.patch.object(reader, "READER_DIRECTORY", package), mock.patch.object(install_pdf_reader, "install", side_effect=install):
            self.assertEqual(self.request("GET", url)[0], 401)
            self.assertEqual(self.request("POST", install_url, {})[0], 401)
            self.assertEqual(self.request("POST", install_url, {}, cookie, {"X-Chat-Request": "0"})[0], 403)
            self.assertEqual(self.request("POST", install_url, {}, cookie, {"Origin": "https://other.invalid"})[0], 403)
            self.assertEqual(self.request("POST", install_url, {"url": "https://other.invalid"}, cookie)[0], 400)
            self.assertEqual(self.request("POST", url + "/unknown/install", {}, cookie)[0], 404)
            self.assertEqual(status()["state"], "available")
            self.assertFalse(self.json_body(self.request("GET", "/api/config", cookie=cookie))["pdf_reader"])
            self.assertEqual(attempts, [])
            try:
                self.assertEqual(status("POST", install_url, {})["state"], "installing")
                self.assertTrue(entered.wait(1))
                self.assertEqual(status("POST", install_url, {})["state"], "installing")
                self.assertEqual(status()["state"], "installing")
                self.assertEqual(attempts, [package])
            finally:
                release.set()
            failed = wait_status("failed")
            self.assertNotIn("private path", failed["error"])
            self.assertIn("Retry", failed["error"])
            self.assertNotIn("error", status("POST", install_url, {}))
            self.assertNotIn("error", wait_status("installed"))
            self.assertTrue(self.json_body(self.request("GET", "/api/config", cookie=cookie))["pdf_reader"])
            self.assertEqual(status("POST", install_url, {})["state"], "installed")
            self.assertEqual(attempts, [package, package])

    def test_pdf_reader_assets_are_local_authenticated_and_allowlisted(self):
        cookie = self.login()
        reader = chat_server.pdf_reader
        package = Path(self.temp.name) / "optional-reader"
        config = "/api/config"
        prefix = "/pdf-reader/"
        with mock.patch.object(reader, "READER_DIRECTORY", package):
            self.assertFalse(self.json_body(self.request("GET", config, cookie=cookie))["pdf_reader"])
            self.assertEqual(self.request("GET", prefix + "pdf.mjs", cookie=cookie)[0], 404)
            package.mkdir()
            (package / "cmaps").mkdir()
            (package / "standard_fonts").mkdir()
            (package / "cmaps" / "Foo-H.bcmap").write_bytes(b"cmap")
            (package / "standard_fonts" / "Lib.ttf").write_bytes(b"font")
            (package / "private.txt").write_text("private", encoding="utf-8")
            for module in reader.MODULES:
                (package / module).write_bytes(b"script")
            (package / reader.INSTALL_MARKER).write_bytes(reader.INSTALL_RECORD)
            self.assertTrue(self.json_body(self.request("GET", config, cookie=cookie))["pdf_reader"])
            self.assertEqual(self.request("GET", prefix + "pdf.mjs")[0], 401)
            self.assertEqual(self.request("GET", prefix + "pdf.worker.mjs", cookie=cookie,
                                          headers={"Sec-Fetch-Site": "cross-site"})[0], 403)
            for path, mime, data in (("pdf.mjs", "text/javascript; charset=utf-8", b"script"),
                                     ("pdf.worker.mjs", "text/javascript; charset=utf-8", b"script"),
                                     ("cmaps/Foo-H.bcmap", "application/octet-stream", b"cmap"),
                                     ("standard_fonts/Lib.ttf", "font/ttf", b"font")):
                status, headers, body = self.request("GET", prefix + path, cookie=cookie)
                self.assertEqual((status, body), (200, data))
                fields = dict(headers)
                self.assertEqual(fields["Content-Type"], mime)
                self.assertEqual(fields["Cross-Origin-Resource-Policy"], "same-origin")
                if path == "pdf.worker.mjs":
                    csp = fields["Content-Security-Policy"]
                    self.assertIn(f"http://127.0.0.1:{self.port}/pdf-reader/cmaps/", csp)
                    self.assertIn("worker-src 'none'", csp)
                    self.assertNotIn("connect-src 'self'", csp)
            for path in ("private.txt", "cmaps", "cmaps/../pdf.mjs", "cmaps/%2e%2e/pdf.mjs", "pdf.mjs?x=1"):
                self.assertEqual(self.request("GET", prefix + path, cookie=cookie)[0], 404)

    def test_settings_allow_only_null_or_valid_integer_reply_limit(self):
        for value in (True, False, 0, -1, 1.5, float("nan"), float("inf"), "1024"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                chat_server.settings_of({"max_tokens": value})
        self.assertIsNone(chat_server.settings_of({"max_tokens": None})["max_tokens"])
        self.assertEqual(chat_server.settings_of({"max_tokens": 1})["max_tokens"], 1)
        self.assertEqual(chat_server.settings_of({"max_tokens": 1_000_000})["max_tokens"], 1_000_000)

    def test_model_state_preserves_only_positive_integer_context_lengths(self):
        data = {"data": [
            {"id": "valid", "context_length": 32768},
            {"id": "missing"},
            {"id": "zero", "context_length": 0},
            {"id": "negative", "context_length": -1},
            {"id": "bool", "context_length": True},
            {"id": "float", "context_length": 4096.0},
            {"id": "string", "context_length": "8192"},
        ]}
        with mock.patch.object(chat_server.Connection, "json", return_value=data):
            result = self.app.model_state()
        self.assertEqual(result["models"], [
            {"id": "valid", "context_length": 32768}, {"id": "missing"},
            {"id": "zero"}, {"id": "negative"}, {"id": "bool"},
            {"id": "float"}, {"id": "string"},
        ])

    def login(self):
        result = self.request("POST", "/api/login", {"password": "correct horse"})
        self.assertEqual(result[0], 200)
        set_cookie = dict((key.lower(), value) for key, value in result[1])["set-cookie"]
        self.assertIn("HttpOnly", set_cookie)
        self.assertIn("SameSite=Strict", set_cookie)
        return set_cookie.split(";", 1)[0]

    def test_sidebar_run_status_route_only_returns_saved_identifiers_and_states(self):
        path = "/api/chats/run-status"
        self.assertEqual(self.request("GET", path)[0], 401)
        cookie = self.login()
        saved = self.app.store.create_chat({"title": "Private title"})
        run = self.app.store.start(saved["id"], {"content": "Private prompt"}, {"system": "Private settings"})
        self.app.store.publish(run["id"], "state", {"status": "failed", "error": "Private error"})
        self.app.store.update_chat(saved["id"], {"archived": True})
        temp = self.app.temporary.create({"title": "Private temporary"})
        store, _ = self.app.temporary.access(temp["id"])
        store.start(temp["id"], {"content": "Temporary prompt"}, {})
        with self.app.store.search_lock:
            response = self.request("GET", path, cookie=cookie)
        self.assertEqual(response[0], 200)
        self.assertIn(("Cache-Control", "no-store"), response[1])
        self.assertEqual(self.json_body(response), {"chats": [{"id": saved["id"], "status": "failed"}]})

    def test_search_route_is_authenticated_bounded_and_saved_only(self):
        path = "/api/chats/search"
        query = {"query": " search witness ", "archived": False}
        self.assertEqual(self.request("POST", path, query)[0], 401)
        cookie = self.login()
        created = self.app.store.create_chat({"title": "Search witness", "settings": {"system": "hidden"}})
        temporary = self.app.temporary.create({"title": "Search witness temporary"})
        temporary_store, _ = self.app.temporary.access(temporary["id"])
        temporary_store.start(temporary["id"], {"content": "Search witness temporary text"}, {})
        result = self.request("POST", path, query, cookie)
        self.assertEqual(result[0], 200)
        self.assertEqual(self.json_body(result), {"chats": [{key: created[key] for key in
                ("id", "title", "folder", "pinned", "archived", "updated")}]})
        self.assertIn(("Cache-Control", "no-store"), result[1])
        self.assertEqual(self.request("POST", path, query, cookie, {"X-Chat-Request": "0"})[0], 403)
        for body in ({}, {"query": "", "archived": False}, {"query": "x" * 257, "archived": False},
                     {"query": "x", "archived": 0}, {"query": "x", "archived": False, "extra": True}):
            with self.subTest(body=body):
                self.assertEqual(self.request("POST", path, body, cookie)[0], 400)
        with self.app.store.search_lock:
            self.assertEqual(self.request("POST", path, query, cookie)[0], 429)

    def test_tool_connection_routes_require_auth_and_only_probe_the_named_server(self):
        from test_tools import FakeMCP
        from tools import ToolRegistry
        from workspace import WorkspaceTools

        FakeMCP.mode, FakeMCP.requests = "normal", []
        remote = ThreadingHTTPServer(("127.0.0.1", 0), FakeMCP)
        worker = threading.Thread(target=remote.serve_forever, daemon=True)
        worker.start()
        try:
            path = os.path.join(self.temp.name, "tools.json")
            with open(path, "w", encoding="utf-8") as output:
                json.dump({"servers": [{"name": "search", "url": "http://127.0.0.1:%d/mcp?key=hidden-query" % remote.server_port,
                                        "headers": {"Authorization": "Bearer hidden-header"}}]}, output)
            registry = ToolRegistry(path)
            self.app.tools = WorkspaceTools(registry, self.app.workspace)
            self.assertEqual(self.request("GET", "/api/tools")[0], 401)
            self.assertEqual(self.request("POST", "/api/tools/test", {"server": "mcp:search"})[0], 401)
            cookie = self.login()
            snapshot = self.json_body(self.request("GET", "/api/tools", cookie=cookie))
            self.assertEqual(snapshot["tool_servers"][1]["status"], "unchecked")
            self.assertEqual(len(snapshot["tools"]), 3)
            self.assertEqual(FakeMCP.requests, [])
            self.assertEqual(self.request("POST", "/api/tools/test", {"server": "mcp:search"}, cookie,
                                          headers={"X-Chat-Request": ""})[0], 403)
            for body, code in (({"server": "mcp:missing"}, 404), ({"server": "builtin:workspace"}, 404),
                               ({"server": "mcp:search", "url": "http://other"}, 400), ({"server": False}, 400)):
                self.assertEqual(self.request("POST", "/api/tools/test", body, cookie)[0], code)
            self.assertEqual(FakeMCP.requests, [])
            response = self.request("POST", "/api/tools/test", {"server": "mcp:search"}, cookie)
            self.assertEqual(response[0], 200)
            snapshot = self.json_body(response)
            self.assertEqual(snapshot["tool_servers"][1]["status"], "ready")
            self.assertEqual(len(snapshot["tools"]), 4)
            self.assertNotIn(b"hidden-query", response[2])
            self.assertNotIn(b"hidden-header", response[2])
            self.assertNotIn("tools/call", [request["method"] for request, _, _ in FakeMCP.requests])
            self.assertEqual(self.request("GET", "/api/config", cookie=cookie)[0], 200)
            FakeMCP.mode = "redirect"
            response = self.request("POST", "/api/tools/test", {"server": "mcp:search"}, cookie)
            snapshot = self.json_body(response)
            self.assertEqual(snapshot["tool_servers"][1]["status"], "error")
            self.assertEqual(len(snapshot["tools"]), 3)
            cached = self.json_body(self.request("GET", "/api/config", cookie=cookie))
            self.assertEqual(cached["tools_error"], snapshot["tools_error"])
            before = len(FakeMCP.requests)
            with registry._discovery_lock:
                self.assertEqual(self.request("POST", "/api/tools/test", {"server": "mcp:search"}, cookie)[0], 409)
            self.assertEqual(len(FakeMCP.requests), before)
            FakeMCP.mode = "normal"
            response = self.request("POST", "/api/tools/test", {"server": "mcp:search"}, cookie)
            self.assertEqual(self.json_body(response)["tool_servers"][1]["status"], "ready")
        finally:
            FakeMCP.mode = "normal"
            remote.shutdown()
            remote.server_close()
            worker.join()

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

    def test_shared_dashboard_assets_are_fixed_and_have_binary_mime_types(self):
        expected = {
            "favicon.svg": (b"<svg>favicon</svg>", "image/svg+xml"),
            "logo.svg": (b"<svg>logo</svg>", "image/svg+xml"),
            "icon-192.png": (b"\x89PNG\r\n\x1a\n192", "image/png"),
            "icon-512.png": (b"\x89PNG\r\n\x1a\n512", "image/png"),
            "theme.css": (b"/* shared theme */", "text/css; charset=utf-8"),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, (data, _) in expected.items():
                (root / name).write_bytes(data)
            with mock.patch.object(chat_monitor, "DASHBOARD", root):
                for name, (data, mime) in expected.items():
                    with self.subTest(asset=name):
                        status, headers, body = self.request("GET", "/" + name)
                        self.assertEqual(status, 200)
                        self.assertEqual(body, data)
                        self.assertEqual(dict((k.lower(), v) for k, v in headers)["content-type"], mime)
                for path in ("/server.py", "/../server.py", "/logo.svg?x=1"):
                    with self.subTest(path=path):
                        self.assertEqual(self.request("GET", path)[0], 404)

    def test_oversized_login_is_rejected_before_body_read_or_password_hash(self):
        with mock.patch.object(self.app, "digest", wraps=self.app.digest) as digest:
            with socket.create_connection(("127.0.0.1", self.port), timeout=1.5) as client:
                client.settimeout(1.5)
                request = (
                    f"POST /api/login HTTP/1.1\r\n"
                    f"Host: 127.0.0.1:{self.port}\r\n"
                    "Content-Type: application/json\r\n"
                    "X-Chat-Request: 1\r\n"
                    "Content-Length: 16385\r\n"
                    "Connection: close\r\n\r\n"
                ).encode("ascii")
                client.sendall(request)
                try:
                    response = client.recv(4096)
                except TimeoutError:
                    response = b""
        digest.assert_not_called()
        self.assertTrue(response.startswith(b"HTTP/1.1 400"), response[:200])

    def test_failed_logins_from_another_loopback_address_do_not_lock_out_owner(self):
        for _ in range(10):
            response = self.request("POST", "/api/login", {"password": "wrong"},
                                    source_address=("127.0.0.2", 0))
            self.assertIn(response[0], (400, 403))
        owner_login = self.request("POST", "/api/login", {"password": "correct horse"},
                                   source_address=("127.0.0.1", 0))
        self.assertEqual(owner_login[0], 200, self.json_body(owner_login))

    def test_forwarded_headers_do_not_bypass_same_source_limit_and_expiry(self):
        with mock.patch.object(chat_server.time, "monotonic", return_value=100) as clock:
            for attempt in range(10):
                self.assertEqual(self.request("POST", "/api/login", {"password": "wrong"},
                                             headers={"X-Forwarded-For": f"192.0.2.{attempt}"})[0], 403)
            with mock.patch.object(self.app, "digest", wraps=self.app.digest) as digest:
                blocked = self.request("POST", "/api/login", {"password": "correct horse"},
                                       headers={"Forwarded": "for=192.0.2.99", "X-Forwarded-For": "192.0.2.99"})
                self.assertEqual(blocked[0], 400)
                digest.assert_not_called()
                clock.return_value = 160
                self.assertEqual(self.request("POST", "/api/login", {"password": "correct horse"})[0], 200)
                digest.assert_called_once()

    def test_failure_client_records_are_bounded_and_expired_records_are_reclaimed(self):
        with mock.patch.object(chat_server.time, "monotonic", return_value=100) as clock, \
                mock.patch.object(self.app, "digest", return_value=b"incorrect") as digest:
            for client in range(1024):
                with self.assertRaises(PermissionError):
                    self.app.login("wrong", str(client))
            self.assertEqual(len(self.app.failures), 1024)
            with self.assertRaises(ValueError):
                self.app.login("wrong", "new-client")
            self.assertEqual(digest.call_count, 1024)
            self.assertEqual(len(self.app.failures), 1024)
            with self.assertRaises(PermissionError):
                self.app.login("wrong", "0")
            clock.return_value = 160
            with self.assertRaises(PermissionError):
                self.app.login("wrong", "new-client")
            self.assertEqual(set(self.app.failures), {"new-client"})

    def test_maximum_password_survives_escaped_json_and_overlong_password_skips_hash(self):
        password = "\U0001f512" * 1024
        self.app.password = self.app.digest(password)
        self.assertEqual(self.request("POST", "/api/login", {"password": password})[0], 200)
        with mock.patch.object(self.app, "digest", wraps=self.app.digest) as digest:
            self.assertEqual(self.request("POST", "/api/login", {"password": password + "x"})[0], 400)
            digest.assert_not_called()

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

    def test_run_rejects_model_removed_since_cached_config_before_mutation(self):
        cookie = self.login()
        chat = self.json_body(self.request("POST", "/api/chats", {}, cookie))
        self.assertEqual(self.json_body(self.request("GET", "/api/config", cookie=cookie))["models"],
                         [{"id": "test-model"}])
        FakeUpstream.models = ["replacement-model"]
        before = self.app.store.snapshot(chat["id"])
        settings = {"model": "test-model", "system": "", "temperature": 0.7,
                    "top_p": 0.95, "top_k": 40, "max_tokens": 4096, "tools": []}
        response = self.request("POST", f"/api/chats/{chat['id']}/runs",
                                 {"content": "question", "settings": settings}, cookie)
        self.assertEqual(response[0], 409, self.json_body(response))
        self.assertEqual(self.json_body(response), {
            "code": "model_unavailable",
            "error": "This model is not available on the connected server. Choose another model before sending.",
            "models": [{"id": "replacement-model"}],
            "upstream_available": True,
            "upstream_error": None,
        })
        after = self.app.store.snapshot(chat["id"])
        self.assertEqual(after["messages"], before["messages"])
        self.assertEqual(after["runs"], before["runs"])
        self.assertFalse(any(path.endswith("/chat/completions") for path, _ in FakeUpstream.posts))
        self.assertEqual(self.json_body(self.request("GET", "/api/config", cookie=cookie))["models"],
                         [{"id": "replacement-model"}])

    def test_run_admission_does_not_wait_for_configuration_discovery(self):
        cookie = self.login()
        chat = self.json_body(self.request("POST", "/api/chats", {}, cookie))
        responses = queue.Queue()
        worker = threading.Thread(target=lambda: responses.put(self.request(
            "POST", f"/api/chats/{chat['id']}/runs",
            {"content": "question", "settings": {"model": "test-model"}}, cookie)))
        response = None
        with self.app.config_lock:
            worker.start()
            try:
                response = responses.get(timeout=2)
            except queue.Empty:
                pass
        worker.join(timeout=3)
        self.assertFalse(worker.is_alive())
        self.assertIsNotNone(response, "run admission waited for unrelated configuration discovery")
        self.assertEqual(response[0], 201, self.json_body(response))

    def test_run_rejects_explicit_model_when_fresh_catalog_is_unavailable(self):
        cookie = self.login()
        chat = self.json_body(self.request("POST", "/api/chats", {}, cookie))
        self.json_body(self.request("GET", "/api/config", cookie=cookie))
        FakeUpstream.model_error = True
        settings = {"model": "test-model", "system": "", "temperature": 0.7,
                    "top_p": 0.95, "top_k": 40, "max_tokens": 4096, "tools": []}
        response = self.request("POST", f"/api/chats/{chat['id']}/runs",
                                 {"content": "question", "settings": settings}, cookie)
        self.assertEqual(response[0], 503, self.json_body(response))
        body = self.json_body(response)
        self.assertEqual(body["code"], "model_unavailable")
        self.assertEqual(body["models"], [])
        self.assertFalse(body["upstream_available"])
        self.assertTrue(body["upstream_error"])
        snapshot = self.app.store.snapshot(chat["id"])
        self.assertEqual(snapshot["messages"], [])
        self.assertEqual(snapshot["runs"], [])
        self.assertFalse(any(path.endswith("/chat/completions") for path, _ in FakeUpstream.posts))

    def test_run_accepts_available_explicit_model_and_empty_default(self):
        cookie = self.login()
        explicit = {"model": "test-model", "system": "", "temperature": 0.7,
                    "top_p": 0.95, "top_k": 40, "max_tokens": 4096, "tools": []}
        default = dict(explicit, model="")
        for settings in (explicit, default):
            chat = self.json_body(self.request("POST", "/api/chats", {}, cookie))
            response = self.request("POST", f"/api/chats/{chat['id']}/runs",
                                    {"content": "question", "settings": settings}, cookie)
            self.assertEqual(response[0], 201, self.json_body(response))
            response_body = self.json_body(response)
            run = response_body["run"]
            if settings["model"]:
                self.assertEqual({key: response_body[key] for key in
                                  ("models", "upstream_available", "upstream_error")}, {
                    "models": [{"id": "test-model"}], "upstream_available": True,
                    "upstream_error": None,
                })
            else:
                self.assertEqual(set(response_body), {"run"})
            deadline = time.monotonic() + 3
            while self.app.store.run(run["id"])["status"] not in ("completed", "failed"):
                self.assertGreater(deadline - time.monotonic(), 0, "fake upstream did not finish the reply")
                time.sleep(0.01)
            self.assertEqual(self.app.store.run(run["id"])["status"], "completed")
            if settings["model"]:
                before = self.app.store.snapshot(chat["id"])
                parent = next(message["id"] for message in before["messages"] if message["role"] == "user")
                FakeUpstream.models = ["replacement-model"]
                retry = self.request("POST", f"/api/chats/{chat['id']}/runs",
                                     {"regenerate": True, "parent_id": parent, "settings": settings}, cookie)
                self.assertEqual(retry[0], 409, self.json_body(retry))
                after = self.app.store.snapshot(chat["id"])
                self.assertEqual(after["messages"], before["messages"])
                self.assertEqual(after["runs"], before["runs"])
                FakeUpstream.models = ["test-model"]
        self.assertEqual([payload["model"] for path, payload in FakeUpstream.posts
                          if path.endswith("/chat/completions")], ["test-model", ""])

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
