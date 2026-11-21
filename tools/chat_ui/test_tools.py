import json
import os
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tools import ToolRegistry


class FakeMCP(BaseHTTPRequestHandler):
    requests = []
    mode = "normal"
    gate = None

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        request = json.loads(body) if body else {}
        type(self).requests.append((request, self.headers.get("Mcp-Session-Id"),
                                    self.headers.get("Mcp-Protocol-Version")))
        if type(self).mode == "redirect":
            self.send_response(307)
            self.send_header("Location", "/other")
            self.end_headers()
            return
        if type(self).mode == "large":
            payload = b"x" * (8 * 1024 * 1024 + 1)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload)
            return
        method = request.get("method")
        if method == "initialize":
            version = "2025-06-18" if type(self).mode != "unknown_protocol" else "2027-01-01"
            result = {"protocolVersion": version, "capabilities": {"tools": {}},
                      "serverInfo": {"name": "fake", "version": "1"}}
        elif method == "tools/list":
            if type(self).mode == "invalid_metadata":
                result = {"tools": [
                    {"name": "bad_schema", "description": "bad schema", "inputSchema": []},
                    {"name": "bad_description", "description": {"unexpected": True}, "inputSchema": {"type": "object"}},
                    {"name": "valid", "description": "valid", "inputSchema": {"type": "object"}}]}
            elif type(self).mode == "paginate" and request.get("params", {}).get("cursor"):
                result = {"tools": [{"name": "second", "inputSchema": {"type": "object"}}]}
            elif type(self).mode == "paginate":
                result = {"tools": [{"name": "echo", "inputSchema": {"type": "object"}}],
                          "nextCursor": "page-2"}
            else:
                result = {"tools": [{"name": "echo", "description": "echo tool",
                                      "inputSchema": {"type": "object"}}]}
        elif method == "tools/call":
            result = {"content": [{"type": "text", "text": request["params"]["arguments"].get("text", "")}]}
        else:
            result = None
        payload = (json.dumps({"jsonrpc": "2.0", "id": request.get("id"), "result": result})
                   if result is not None else "")
        data = payload.encode("utf-8")
        self.send_response(200 if data else 202)
        is_sse = type(self).mode in ("sse", "sse_open", "wrong_id") and data
        if is_sse:
            if type(self).mode == "wrong_id" and request.get("id") is not None:
                wrong = json.dumps({"jsonrpc": "2.0", "id": request["id"] + 100,
                                    "result": {"tools": []}}).encode("utf-8")
                data = b"event: message\ndata: " + wrong + b"\n\n" + b"event: message\ndata: " + data + b"\n\n"
            else:
                data = b"event: message\ndata: " + data + b"\n\n"
        self.send_header("Content-Type", "text/event-stream" if is_sse else "application/json")
        self.send_header("Mcp-Session-Id", "session-1")
        if type(self).mode != "sse_open":
            self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if data:
            self.wfile.write(data)
            self.wfile.flush()
        if type(self).mode == "sse_open" and data and type(self).gate is not None:
            type(self).gate.wait(5)


class ToolRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeMCP)
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = "http://127.0.0.1:%d/mcp" % cls.httpd.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.thread.join()
        cls.httpd.server_close()

    def setUp(self):
        FakeMCP.requests = []
        FakeMCP.mode = "normal"
        FakeMCP.gate = None
        self.temp = tempfile.TemporaryDirectory()
        self.config = os.path.join(self.temp.name, "tools.json")

    def tearDown(self):
        self.temp.cleanup()

    def registry(self, **overrides):
        server = {"name": "local", "url": self.url, "headers": {}, "approval": True}
        server.update(overrides)
        with open(self.config, "w", encoding="utf-8") as output:
            json.dump({"servers": [server]}, output)
        return ToolRegistry(self.config)

    def test_connection_label_preserves_tool_identity(self):
        self.assertEqual(self.registry().connections()[0]["name"], "local")
        registry = self.registry(label="Search")
        connection = registry.connections()[0]
        self.assertEqual((connection["id"], connection["name"]), ("mcp:local", "Search"))
        tool = registry.list_tools()[0]
        self.assertEqual((tool["connection"], tool["name"]), ("mcp:local", "local__echo"))
        registry.test_connection("mcp:local")
        self.assertEqual(registry.connections()[0]["status"], "ready")
        self.assertTrue(tool["approval"])
        for label in (None, 123, "  "):
            with self.subTest(label=label), self.assertRaises(ValueError):
                self.registry(label=label)

    def test_discovery_invocation_session_and_public_shape(self):
        registry = self.registry()
        tools = registry.list_tools()
        self.assertEqual(len(tools), 1)
        self.assertEqual(set(tools[0]), {"name", "description", "inputSchema", "server", "connection", "approval"})
        self.assertEqual(tools[0]["name"], "local__echo")
        result = registry.call_tool(tools[0]["name"], {"text": "hello"})
        self.assertEqual(result["content"][0]["text"], "hello")
        self.assertTrue(any(method.get("method") == "notifications/initialized"
                            for method, _, _ in FakeMCP.requests))
        self.assertTrue(any(session == "session-1" for _, session, _ in FakeMCP.requests))
        methods = [request for request, _, _ in FakeMCP.requests]
        rpc_ids = [request["id"] for request in methods if "id" in request]
        self.assertEqual(len(rpc_ids), len(set(rpc_ids)))
        self.assertEqual(next(version for request, _, version in FakeMCP.requests
                              if request.get("method") == "tools/list"), "2025-06-18")

    def test_cancel_does_not_call_remote_tool(self):
        registry = self.registry()
        tools = registry.list_tools()
        event = threading.Event()
        event.set()
        before = len(FakeMCP.requests)
        result = registry.call_tool(tools[0]["name"], {}, event)
        self.assertTrue(result["isError"])
        self.assertEqual(len(FakeMCP.requests), before)

    def test_sse_responses_and_pagination(self):
        FakeMCP.mode = "sse"
        self.assertEqual(len(self.registry().list_tools()), 1)
        FakeMCP.mode = "paginate"
        registry = self.registry()
        self.assertEqual([tool["name"] for tool in registry.list_tools()],
                         ["local__echo", "local__second"])

    def test_sse_returns_matching_response_without_waiting_for_eof(self):
        FakeMCP.mode = "sse_open"
        gate = threading.Event()
        FakeMCP.gate = gate
        registry = self.registry()
        completed = threading.Event()
        result = []

        def discover():
            result.extend(registry.list_tools())
            completed.set()

        worker = threading.Thread(target=discover)
        worker.start()
        try:
            self.assertTrue(completed.wait(2), "SSE client waited for the server to close its stream")
            self.assertEqual(len(result), 1)
        finally:
            gate.set()
            worker.join(2)

    def test_sse_ignores_unrelated_json_rpc_ids(self):
        FakeMCP.mode = "wrong_id"
        self.assertEqual(len(self.registry().list_tools()), 1)

    def test_untrusted_tool_metadata_types_are_not_exposed(self):
        FakeMCP.mode = "invalid_metadata"
        tools = self.registry().list_tools()
        self.assertEqual([tool["name"] for tool in tools], ["local__valid"])

    def test_negotiated_protocol_header_cannot_be_overridden_by_config(self):
        registry = self.registry(headers={"mcp-protocol-version": "wrong"})
        registry.list_tools()
        self.assertTrue(all(version == "2025-06-18" for request, _, version in FakeMCP.requests
                            if request.get("method") != "initialize"))

    def test_unsupported_protocol_version_is_refused(self):
        FakeMCP.mode = "unknown_protocol"
        with self.assertRaisesRegex(RuntimeError, "initialization failed"):
            self.registry().list_tools()

    def test_credentials_are_not_exposed(self):
        registry = self.registry(headers={"Authorization": "Bearer hidden-token"})
        result = registry.call_tool(registry.list_tools()[0]["name"], {"text": "hidden-token"})
        self.assertNotIn("hidden-token", json.dumps(registry.list_tools()))
        self.assertNotIn("hidden-token", json.dumps(result))

    def test_redirect_is_refused(self):
        FakeMCP.mode = "redirect"
        registry = self.registry()
        with self.assertRaisesRegex(RuntimeError, "redirect"):
            registry.list_tools()
        self.assertEqual(len(FakeMCP.requests), 1)

    def test_oversized_response_is_refused(self):
        FakeMCP.mode = "large"
        with self.assertRaisesRegex(RuntimeError, "size limit"):
            self.registry().list_tools()

    def test_failed_discovery_is_reported_and_retried(self):
        FakeMCP.mode = "redirect"
        registry = self.registry()
        with self.assertRaises(RuntimeError):
            registry.list_tools()
        self.assertIn("redirect", registry.last_error)
        FakeMCP.mode = "normal"
        registry._tools_time = time.monotonic() - 31
        self.assertEqual(len(registry.list_tools()), 1)
        self.assertIsNone(registry.last_error)

    def test_connection_status_probe_and_credentials(self):
        registry = self.registry(headers={"Authorization": "Bearer hidden-token"})
        state = registry.connections()[0]
        self.assertEqual(state["status"], "unchecked")
        self.assertIsNone(state["checked_at"])
        self.assertEqual(registry.list_tools(discover=False), [])
        self.assertEqual(FakeMCP.requests, [])
        with self.assertRaises(KeyError):
            registry.test_connection("mcp:not-configured")
        self.assertEqual(FakeMCP.requests, [])
        registry.test_connection("mcp:local")
        state = registry.connections()[0]
        self.assertEqual((state["status"], state["tool_count"], state["approval"]), ("ready", 1, True))
        self.assertGreater(state["checked_at"], 0)
        self.assertGreaterEqual(state["duration_ms"], 0)
        self.assertNotIn("hidden-token", json.dumps(state))
        self.assertNotIn(self.url, json.dumps(state))
        self.assertNotIn("tools/call", [request["method"] for request, _, _ in FakeMCP.requests])
        count = len(FakeMCP.requests)
        registry.list_tools()
        registry.connections()
        self.assertEqual(len(FakeMCP.requests), count)

    def test_probe_is_scoped_and_recovers_without_hiding_a_healthy_server(self):
        with open(self.config, "w", encoding="utf-8") as output:
            json.dump({"servers": [{"name": name, "url": self.url} for name in ("first", "second")]}, output)
        registry = ToolRegistry(self.config)
        original, calls = registry._post, []
        failing = True

        def post(server, payload, expected_id=None):
            calls.append(server["name"])
            if server["name"] == "second" and failing:
                raise RuntimeError("connection unavailable")
            return original(server, payload, expected_id)

        with patch.object(registry, "_post", side_effect=post):
            self.assertEqual([tool["name"] for tool in registry.list_tools()], ["first__echo"])
            self.assertEqual([state["status"] for state in registry.connections()], ["ready", "error"])
            calls.clear()
            failing = False
            registry.test_connection("mcp:second")
            self.assertEqual(set(calls), {"second"})
            self.assertEqual([tool["name"] for tool in registry.list_tools()], ["first__echo", "second__echo"])
            self.assertIsNone(registry.last_error)
            self.assertTrue(all(state["status"] == "ready" for state in registry.connections()))

    def test_active_probe_keeps_cached_tools_readable_and_rejects_a_second_probe(self):
        registry = self.registry()
        before = registry.list_tools()
        entered, release, read_done = threading.Event(), threading.Event(), threading.Event()
        original, failures = registry._post, []

        def post(server, payload, expected_id=None):
            if payload["method"] == "tools/list":
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("test did not release discovery")
            return original(server, payload, expected_id)

        def probe():
            try:
                registry.test_connection("mcp:local")
            except Exception as exc:
                failures.append(exc)

        observed = []

        def read():
            observed.extend(registry.list_tools())
            read_done.set()

        with patch.object(registry, "_post", side_effect=post):
            worker = threading.Thread(target=probe)
            worker.start()
            reader = None
            try:
                self.assertTrue(entered.wait(2))
                self.assertEqual(registry.connections()[0]["status"], "checking")
                with self.assertRaises(BlockingIOError):
                    registry.test_connection("mcp:local")
                reader = threading.Thread(target=read)
                reader.start()
                self.assertTrue(read_done.wait(2), "cached tools waited behind the probe")
            finally:
                release.set()
                worker.join(5)
                if reader:
                    reader.join(5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(observed, before)
        self.assertEqual(registry.connections()[0]["status"], "ready")

    def test_bad_pagination_does_not_publish_a_partial_server_catalog(self):
        registry = self.registry()
        original = registry._post

        def post(server, payload, expected_id=None):
            if payload["method"] == "tools/list":
                return {"jsonrpc": "2.0", "id": expected_id, "result": {
                    "tools": [{"name": "partial"}], "nextCursor": "repeat"}}
            return original(server, payload, expected_id)

        with patch.object(registry, "_post", side_effect=post):
            registry.test_connection("mcp:local")
        self.assertEqual(registry.connections()[0]["status"], "error")
        self.assertEqual(registry.connections()[0]["tool_count"], 0)
        with self.assertRaisesRegex(RuntimeError, "pagination"):
            registry.list_tools()

    def test_refresh_preserves_alias_identity_and_redacts_echoed_credentials(self):
        registry = self.registry(headers={"Authorization": "Bearer hidden-token"})
        original = registry._post
        names = ["same name", "same_name", "hidden-token"]

        def post(server, payload, expected_id=None):
            if payload["method"] == "tools/list":
                return {"jsonrpc": "2.0", "id": expected_id, "result": {
                    "tools": [{"name": name, "description": name} for name in names]}}
            return original(server, payload, expected_id)

        with patch.object(registry, "_post", side_effect=post):
            before = {tool["description"]: tool["name"] for tool in registry.list_tools()}
            names.reverse()
            registry.test_connection("mcp:local")
            after = {tool["description"]: tool["name"] for tool in registry.list_tools()}
            self.assertEqual(after, before)
            self.assertNotIn("hidden-token", json.dumps(registry.list_tools()))
            registry.call_tool(before["same name"], {"text": "ok"})
            self.assertEqual(FakeMCP.requests[-1][0]["params"]["name"], "same name")
            with patch("tools._MAX_ALIASES", 3):
                names.append("fourth")
                registry.test_connection("mcp:local")
                self.assertEqual(registry.connections()[0]["status"], "error")
                self.assertIn("history limit", registry.last_error)

    def test_invalid_urls_and_headers_are_rejected(self):
        for url in ("file:///tmp/x", "http://user:pass@example.test/mcp", "http://example.test/x#fragment"):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    self.registry(url=url)
        with self.assertRaises(ValueError):
            self.registry(headers={"X-Test": "bad\nvalue"})


if __name__ == "__main__":
    unittest.main()
