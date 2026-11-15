"""Bounded client for operator-configured MCP servers over HTTP.

Only configured HTTP(S) endpoints are contacted; redirects, stdio, and dynamic
server installation are unsupported. Calls run independently, so servers must
support concurrent requests when the caller makes them.
"""

import json
import http.client
import re
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


_PROTOCOL_VERSION = "2025-06-18"
_MAX_RESPONSE = 8 * 1024 * 1024
_MAX_TOOLS = 128
_MAX_PAGES = 16
_TIMEOUT = 30
_RETRY_AFTER = 30
_NAME_PART = re.compile(r"[^A-Za-z0-9_-]+")


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class ToolRegistry:
    """Discovers and invokes tools from an optional JSON server config file."""

    def __init__(self, config_path=None):
        self._lock = threading.RLock()
        self._opener = build_opener(_NoRedirect())
        self._servers = self._read_config(config_path) if config_path else []
        self._tools = None
        self._tools_time = 0
        self.last_error = None
        self._next_request_id = 0

    def _request_id(self):
        with self._lock:
            self._next_request_id += 1
            return self._next_request_id

    @staticmethod
    def _read_config(path):
        try:
            with open(path, "r", encoding="utf-8") as source:
                data = json.load(source)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid tool server configuration") from exc
        if not isinstance(data, dict) or not isinstance(data.get("servers", []), list):
            raise ValueError("invalid tool server configuration")
        servers = []
        names = set()
        for item in data.get("servers", []):
            if not isinstance(item, dict):
                raise ValueError("invalid tool server configuration")
            name, url = item.get("name"), item.get("url")
            headers = item.get("headers", {})
            approval = item.get("approval", True)
            if (not isinstance(name, str) or not name or name in names or
                    not isinstance(url, str) or not isinstance(headers, dict) or
                    not isinstance(approval, bool)):
                raise ValueError("invalid tool server configuration")
            try:
                parsed = urlsplit(url)
                valid_url = (parsed.scheme in ("http", "https") and parsed.hostname and
                             parsed.username is None and parsed.password is None and
                             not parsed.fragment)
                if parsed.port is not None and not (1 <= parsed.port <= 65535):
                    valid_url = False
            except ValueError:
                valid_url = False
            if not valid_url:
                raise ValueError("tool server URL must be an explicit HTTP(S) endpoint without credentials")
            checked_headers = {}
            for key, value in headers.items():
                if (not isinstance(key, str) or not key or
                        not isinstance(value, str) or "\r" in key + value or "\n" in key + value):
                    raise ValueError("invalid tool server configuration")
                checked_headers[key] = value
            names.add(name)
            servers.append({"name": name, "url": url, "headers": checked_headers,
                            "approval": approval, "session_id": None, "protocol_version": None,
                            "initialized": False, "init_lock": threading.Lock()})
        return servers

    def _redact(self, server, value):
        if isinstance(value, str):
            for secret in server["headers"].values():
                if secret:
                    value = value.replace(secret, "[redacted]")
                    parts = secret.split(None, 1)
                    if len(parts) == 2 and parts[0].lower() in ("bearer", "basic"):
                        value = value.replace(parts[1], "[redacted]")
            return value
        if isinstance(value, list):
            return [self._redact(server, item) for item in value]
        if isinstance(value, dict):
            return {self._redact(server, key): self._redact(server, item)
                    for key, item in value.items()}
        return value

    @staticmethod
    def _failure(message):
        return {"content": [{"type": "text", "text": message}], "isError": True}

    def _post(self, server, payload, expected_id=None):
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers = dict(server["headers"])
        with self._lock:
            session_id = server["session_id"]
            protocol_version = server["protocol_version"]
        required = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if session_id:
            required["Mcp-Session-Id"] = session_id
        if protocol_version:
            required["Mcp-Protocol-Version"] = protocol_version
        for key, value in required.items():
            for existing in list(headers):
                if existing.lower() == key.lower():
                    del headers[existing]
            headers[key] = value
        request = Request(server["url"], data=body, headers=headers, method="POST")
        try:
            response = self._opener.open(request, timeout=_TIMEOUT)
        except HTTPError as exc:
            exc.close()
            if exc.code in (301, 302, 303, 307, 308):
                raise RuntimeError("tool server redirects are not supported") from None
            raise RuntimeError("tool server returned an HTTP error") from None
        except (URLError, OSError, TimeoutError, ValueError):
            raise RuntimeError("tool server request failed") from None
        try:
            with response:
                status = response.status
                if status < 200 or status >= 300:
                    raise RuntimeError("tool server returned an HTTP error")
                raw_session = response.headers.get("Mcp-Session-Id")
                if raw_session:
                    with self._lock:
                        server["session_id"] = raw_session
                content_type = response.headers.get_content_type()
                if content_type == "text/event-stream":
                    return self._read_sse(response, expected_id)
                raw = response.read(_MAX_RESPONSE + 1)
                if len(raw) > _MAX_RESPONSE:
                    raise RuntimeError("tool server response exceeded the size limit")
        except (http.client.HTTPException, OSError, TimeoutError):
            raise RuntimeError("tool server response failed") from None
        if not raw:
            return None
        if content_type == "application/json" or content_type.endswith("+json"):
            try:
                return json.loads(raw.decode("utf-8"))
            except (UnicodeError, json.JSONDecodeError):
                raise RuntimeError("tool server returned an invalid response") from None
        raise RuntimeError("tool server returned an unsupported response type")

    @staticmethod
    def _read_sse(response, expected_id):
        total = 0
        data_lines = []
        while True:
            line = response.readline(65537)
            if not line:
                raise RuntimeError("tool server closed its event stream before replying")
            total += len(line)
            if len(line) > 65536 or total > _MAX_RESPONSE:
                raise RuntimeError("tool server response exceeded the size limit")
            try:
                line = line.decode("utf-8").rstrip("\r\n")
            except UnicodeError:
                raise RuntimeError("tool server returned an invalid response") from None
            if not line:
                if data_lines:
                    try:
                        message = json.loads("\n".join(data_lines))
                    except json.JSONDecodeError:
                        raise RuntimeError("tool server returned an invalid response") from None
                    data_lines = []
                    if (isinstance(message, dict) and message.get("jsonrpc") == "2.0" and
                            expected_id is not None and message.get("id") == expected_id):
                        return message
                continue
            if line.startswith(":"):
                continue
            field, separator, value = line.partition(":")
            if field == "data":
                data_lines.append(value[1:] if separator and value.startswith(" ") else value)

    def _initialize(self, server):
        with server["init_lock"]:
            with self._lock:
                if server["initialized"]:
                    return
            request_id = self._request_id()
            response = self._post(server, {"jsonrpc": "2.0", "id": request_id, "method": "initialize",
                                         "params": {"protocolVersion": _PROTOCOL_VERSION,
                                                     "capabilities": {},
                                                     "clientInfo": {"name": "llmx-chat", "version": "1"}}}, request_id)
            result = self._remote_result(response, request_id)
            if result.get("protocolVersion") != _PROTOCOL_VERSION:
                raise RuntimeError("tool server initialization failed")
            with self._lock:
                server["protocol_version"] = _PROTOCOL_VERSION
            self._post(server, {"jsonrpc": "2.0", "method": "notifications/initialized"})
            with self._lock:
                server["initialized"] = True

    @staticmethod
    def _remote_result(response, expected_id):
        if not isinstance(response, dict) or response.get("jsonrpc") != "2.0" or response.get("id") != expected_id:
            raise RuntimeError("tool server returned an invalid response")
        if "error" in response:
            raise RuntimeError("tool server reported a protocol error")
        response = response.get("result")
        if not isinstance(response, dict):
            raise RuntimeError("tool server returned an invalid result")
        return response

    def list_tools(self):
        with self._lock:
            if self._tools is not None and (not self.last_error or time.monotonic() - self._tools_time < _RETRY_AFTER):
                public = [{key: value for key, value in tool.items() if not key.startswith("_")}
                          for tool in self._tools]
                if self.last_error and not public:
                    raise RuntimeError(self.last_error)
                return public
        result_tools = []
        errors = []
        for server in self._servers:
            try:
                self._initialize(server)
                cursor = None
                seen_cursors = set()
                for _ in range(_MAX_PAGES):
                    params = {"cursor": cursor} if cursor else {}
                    request_id = self._request_id()
                    response = self._post(server, {"jsonrpc": "2.0", "id": request_id,
                                                   "method": "tools/list", "params": params}, request_id)
                    result = self._remote_result(response, request_id)
                    batch = result.get("tools", [])
                    if not isinstance(batch, list):
                        raise RuntimeError("tool server returned an invalid tool list")
                    for item in batch:
                        if len(result_tools) >= _MAX_TOOLS:
                            break
                        if (not isinstance(item, dict) or not isinstance(item.get("name"), str) or
                                not 1 <= len(item["name"]) <= 128 or
                                not isinstance(item.get("description", ""), str) or
                                len(item.get("description", "")) > 16000 or
                                not isinstance(item.get("inputSchema", {}), dict)):
                            continue
                        safe_server = _NAME_PART.sub("_", server["name"]).strip("_-") or "server"
                        safe_tool = _NAME_PART.sub("_", item["name"]).strip("_-") or "tool"
                        public_name = safe_server + "__" + safe_tool
                        used_names = {existing["name"] for existing in result_tools}
                        suffix = 2
                        while public_name in used_names:
                            public_name = safe_server + "__" + safe_tool + "_" + str(suffix)
                            suffix += 1
                        result_tools.append({
                            "name": public_name,
                            "description": self._redact(server, item.get("description", "")),
                            "inputSchema": self._redact(server, item.get("inputSchema", {"type": "object"})),
                            "server": server["name"], "approval": server["approval"],
                            "_remote_name": item["name"],
                        })
                    if len(result_tools) >= _MAX_TOOLS:
                        break
                    cursor = result.get("nextCursor")
                    if not cursor:
                        break
                    if not isinstance(cursor, str) or cursor in seen_cursors:
                        raise RuntimeError("tool server returned invalid pagination")
                    seen_cursors.add(cursor)
                else:
                    if cursor:
                        raise RuntimeError("tool server pagination limit exceeded")
            except RuntimeError as exc:
                errors.append(str(exc))
        with self._lock:
            self._tools = result_tools
            self._tools_time = time.monotonic()
            self.last_error = "; ".join(errors)[:500] or None
            public = [{key: value for key, value in tool.items() if not key.startswith("_")}
                      for tool in self._tools]
            if self.last_error and not public:
                raise RuntimeError(self.last_error)
            return public

    def call_tool(self, name, arguments, cancel_event=None):
        if cancel_event is not None and cancel_event.is_set():
            return self._failure("tool call cancelled")
        self.list_tools()
        with self._lock:
            entry = next((item for item in self._tools if item["name"] == name), None)
        if entry is None:
            return self._failure("tool discovery failed: " + self.last_error if self.last_error else "unknown tool")
        server = next(item for item in self._servers if item["name"] == entry["server"])
        if cancel_event is not None and cancel_event.is_set():
            return self._failure("tool call cancelled")
        try:
            self._initialize(server)
            if cancel_event is not None and cancel_event.is_set():
                return self._failure("tool call cancelled")
            request_id = self._request_id()
            response = self._post(server, {"jsonrpc": "2.0", "id": request_id, "method": "tools/call",
                                           "params": {"name": entry["_remote_name"],
                                                      "arguments": arguments if isinstance(arguments, dict) else {}}}, request_id)
            result = self._remote_result(response, request_id)
            if not isinstance(result.get("content", []), list):
                raise RuntimeError("tool server returned an invalid tool result")
            return self._redact(server, result)
        except RuntimeError as exc:
            return self._failure(str(exc))
