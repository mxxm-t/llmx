import json
import os
import tempfile
import threading
import time
import unittest
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

    def test_discovery_invocation_session_and_public_shape(self):
        registry = self.registry()
        tools = registry.list_tools()
        self.assertEqual(len(tools), 1)
        self.assertEqual(set(tools[0]), {"name", "description", "inputSchema", "server", "approval"})
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

    def test_invalid_urls_and_headers_are_rejected(self):
        for url in ("file:///tmp/x", "http://user:pass@example.test/mcp", "http://example.test/x#fragment"):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    self.registry(url=url)
        with self.assertRaises(ValueError):
            self.registry(headers={"X-Test": "bad\nvalue"})


if __name__ == "__main__":
    unittest.main()
