"""Serve the owner chat workspace without adding dependencies to llmx."""

import argparse
import base64
import hashlib
import hmac
import json
import math
import mimetypes
from pathlib import Path
import re
import secrets
import sys
import threading
import time
import tempfile
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from engine import Engine, TERMINAL
from store import Store, encoded
from temporary import TemporaryChats
from tools import ToolRegistry
from upstream import Connection
from monitor import ASSETS, Monitor
from optional_features import OptionalFeatures
import pdf_reader
from project_preview import MAX_PREVIEW_BYTES, assemble
from workspace import MAX_PROJECT, MAX_SELECTED_FILES, ProjectConflict, Workspace, WorkspaceTools


ROOT = Path(__file__).resolve().parent
MAX_BODY = 16 * 1024 * 1024
MAX_LOGIN_BODY = 16 * 1024
MAX_BACKUP = 256 * 1024 * 1024
MAX_CONNECTIONS = 64
MAX_LOGIN_CLIENTS = 1024
DEFAULTS = dict(model="", system="", temperature=0.7, top_p=0.95, top_k=40, max_tokens=None, tools=[])


def settings_of(value):
    if not isinstance(value, dict) or set(value) - set(DEFAULTS):
        raise ValueError("Unsupported chat settings")
    settings = dict(DEFAULTS, **value)
    for name, limit in (("model", 512), ("system", 100000)):
        if not isinstance(settings[name], str) or len(settings[name]) > limit:
            raise ValueError(f"Invalid {name}")
    for name, lo, hi in (("temperature", 0, 5), ("top_p", 0, 1), ("top_k", 0, 100000), ("max_tokens", 1, 1000000)):
        number = settings[name]
        if name == "max_tokens" and number is None:
            continue
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or not lo <= number <= hi:
            raise ValueError(f"Invalid {name}")
        if name in ("top_k", "max_tokens") and not isinstance(number, int):
            raise ValueError(f"{name} must be an integer")
    if not isinstance(settings["tools"], list) or len(settings["tools"]) > 64 or any(not isinstance(t, str) or len(t) > 256 for t in settings["tools"]):
        raise ValueError("Invalid tool selection")
    return settings


def attachments_of(value):
    if not isinstance(value, list) or len(value) > 8:
        raise ValueError("Attach at most eight files")
    clean, total = [], 0
    for a in value:
        if not isinstance(a, dict) or not isinstance(a.get("name"), str) or not 1 <= len(a["name"]) <= 255:
            raise ValueError("Invalid attachment")
        name = a["name"].replace("\\", "/").split("/")[-1]
        if "text" in a:
            if not isinstance(a["text"], str) or "\x00" in a["text"]:
                raise ValueError("Text attachments must be UTF-8 text")
            size = len(a["text"].encode("utf-8"))
            if size > 2 * 1024 * 1024:
                raise ValueError("A text attachment may be at most 2 MiB")
            clean.append(dict(name=name, type="text/plain", text=a["text"]))
        else:
            mime, data = a.get("type"), a.get("data")
            if mime not in ("image/png", "image/jpeg") or not isinstance(data, str) or not data.startswith(f"data:{mime};base64,"):
                raise ValueError("Images must be PNG or JPEG data URLs")
            try:
                decoded = base64.b64decode(data.split(",", 1)[1], validate=True)
            except ValueError as exc:
                raise ValueError("Invalid image encoding") from exc
            magic = b"\x89PNG\r\n\x1a\n" if mime == "image/png" else b"\xff\xd8\xff"
            if not decoded.startswith(magic):
                raise ValueError("Image content does not match its type")
            size = len(decoded)
            if size > 8 * 1024 * 1024:
                raise ValueError("An image may be at most 8 MiB")
            clean.append(dict(name=name, type=mime, data=data))
        total += size
    if total > 10 * 1024 * 1024:
        raise ValueError("Attachments exceed 10 MiB")
    return clean


def chat_values(body, patch=False):
    if not isinstance(body, dict):
        raise ValueError("Expected a JSON object")
    allowed = {"title", "folder", "settings"} | ({"pinned", "archived", "active_leaf"} if patch else set())
    if set(body) - allowed:
        raise ValueError("Unsupported conversation field")
    values = dict(body)
    for key in ("title", "folder"):
        if key in values and (not isinstance(values[key], str) or len(values[key]) > 200):
            raise ValueError(f"Invalid {key}")
    for key in ("pinned", "archived"):
        if key in values and not isinstance(values[key], bool):
            raise ValueError(f"Invalid {key}")
    if "active_leaf" in values and values["active_leaf"] is not None and not isinstance(values["active_leaf"], str):
        raise ValueError("Invalid branch")
    if "settings" in values:
        values["settings"] = settings_of(values["settings"])
    return values


class Application:
    def __init__(self, database, upstream, tool_config=None, password=None, api_key=None, sandbox_image=None, docker_host=None):
        self.store = Store(database)
        self.upstream, self.api_key = upstream, api_key
        self.workspace = Workspace(Path(database).parent / "workspace", sandbox_image, docker_host)
        self.tools = WorkspaceTools(ToolRegistry(tool_config), self.workspace)
        self.engine = Engine(self.store, upstream, self.tools, api_key)
        self.temporary = TemporaryChats(upstream, api_key)
        self.salt = secrets.token_bytes(32)
        self.password = self.digest(password) if password else None
        self.sessions, self.failures = {}, {}
        self.lock = threading.Lock()
        self.config_lock = threading.Lock()
        self.cached_config, self.config_time = None, 0
        self.monitor = Monitor(upstream, api_key)
        self.features = OptionalFeatures()
        self.previews = {}

    def digest(self, password):
        return hashlib.pbkdf2_hmac("sha256", password.encode(), self.salt, 200000)

    def authenticated(self, raw):
        if self.password is None:
            return True
        try:
            jar = cookies.SimpleCookie(raw)
            token = jar["llmx_chat"].value
        except (KeyError, cookies.CookieError):
            return False
        with self.lock:
            now = time.time()
            self.sessions = {k: v for k, v in self.sessions.items() if v > now}
            return self.sessions.get(token, 0) > now

    def login(self, password, client):
        if not isinstance(password, str) or len(password) > 1024:
            raise ValueError("Invalid password")
        with self.lock:
            now = time.monotonic()
            self.failures = {key: active for key, times in self.failures.items()
                             if (active := [t for t in times if t > now - 60])}
            failures = self.failures.get(client, [])
            if len(failures) >= 10 or (client not in self.failures and len(self.failures) >= MAX_LOGIN_CLIENTS):
                raise ValueError("Too many attempts. Try again in one minute.")
            if self.password and not hmac.compare_digest(self.digest(password), self.password):
                self.failures.setdefault(client, []).append(now)
                raise PermissionError("Incorrect password")
            token = secrets.token_urlsafe(32)
            if len(self.sessions) > 64:
                self.sessions.pop(next(iter(self.sessions)))
            self.sessions[token] = time.time() + 12 * 3600
            return token

    def logout(self, raw):
        try:
            token = cookies.SimpleCookie(raw)["llmx_chat"].value
        except (KeyError, cookies.CookieError):
            return
        with self.lock:
            self.sessions.pop(token, None)
            self.previews.clear()
        with self.monitor.collector.changed:
            self.monitor.collector.changed.notify_all()

    def tool_state(self, discover=True):
        result = dict(tools=[], tools_error=None)
        try:
            result["tools"] = self.tools.list_tools(discover=discover)
            result["tools_error"] = getattr(self.tools, "last_error", None)
        except RuntimeError as exc:
            result["tools_error"] = str(exc)[:500]
        result["tool_servers"] = self.tools.connections()
        return result

    def model_state(self):
        result = dict(models=[], upstream_available=False, upstream_error=None)
        try:
            models = Connection(self.upstream, timeout=8, key=self.api_key).json("/models")
            for model in models.get("data", []):
                if not isinstance(model, dict) or not isinstance(model.get("id"), str):
                    continue
                item = {"id": model["id"]}
                context_length = model.get("context_length")
                if type(context_length) is int and context_length > 0:
                    item["context_length"] = context_length
                result["models"].append(item)
            result["upstream_available"] = True
        except Exception as exc:
            result["upstream_error"] = f"Unable to reach llmx: {type(exc).__name__}"
        return result

    def config(self):
        with self.config_lock:
            if not self.cached_config or time.monotonic() - self.config_time >= 10:
                result = dict(tools=[], tools_error=None)
                result["monitor"] = dict(url="/monitor/view", interval_s=self.monitor.collector.interval_s)
                result["authentication"] = "owner-password" if self.password else "local-only"
                result["upstream"] = self.upstream
                result["backup_max_bytes"] = MAX_BACKUP
                result["temporary"] = dict(enabled=True, ttl_seconds=self.temporary.TTL_SECONDS,
                                           tools=False, max_chats=self.temporary.MAX_CHATS)
                result["workspace"] = self.workspace.snapshot()["execution"]
                result.update(self.tool_state())
                self.cached_config, self.config_time = result, time.monotonic()
            result = dict(self.cached_config)
        result.update(self.model_state())
        result["settings_defaults"] = dict(DEFAULTS)
        result["pdf_reader"] = pdf_reader.installed(pdf_reader.READER_DIRECTORY)
        return result


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, app, public_origin=None):
        parsed = None
        if public_origin:
            parsed = urlsplit(public_origin)
            if (parsed.scheme not in ("http", "https") or not parsed.hostname
                    or parsed.path not in ("", "/") or parsed.query or parsed.fragment
                    or parsed.username is not None or (parsed.port is not None and not 1 <= parsed.port <= 65535)):
                raise ValueError("Public origin must be an HTTP(S) origin without path or credentials")
        if address[0] not in ("127.0.0.1", "localhost") and (not app.password or not parsed or parsed.scheme != "https"):
            raise ValueError("Non-loopback listening needs an owner password and an HTTPS public origin")
        self.app = app
        self._started = False
        self._connections = threading.BoundedSemaphore(MAX_CONNECTIONS)
        super().__init__(address, Handler)
        port = self.server_address[1]
        self.origins = {f"http://localhost:{port}", f"http://127.0.0.1:{port}"}
        if public_origin:
            self.origins.add(public_origin.rstrip("/"))
        self.hosts = {urlsplit(origin).netloc for origin in self.origins}
        self.secure = bool(parsed and parsed.scheme == "https")
        try:
            app.monitor.collector.start()
        except BaseException:
            super().server_close()
            raise
        self._started = True

    def process_request(self, request, client_address):
        if not self._connections.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._connections.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._connections.release()

    def server_close(self):
        try:
            if self._started:
                self._started = False
                self.app.temporary.close()
                self.app.monitor.stop()
                self.app.workspace.close()
        finally:
            super().server_close()

    def handle_error(self, request, client_address):
        if not isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):
            super().handle_error(request, client_address)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    @property
    def app(self):
        return self.server.app

    def log_message(self, _format, *_args):
        pass

    def setup(self):
        super().setup()
        self.connection.settimeout(20)

    def _headers(self, status, mime, length=None, extra=None, policy=None):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", policy or "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'")
        if length is not None:
            self.send_header("Content-Length", str(length))
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()

    def _json(self, value, status=200, extra=None):
        data = encoded(value).encode()
        self._headers(status, "application/json; charset=utf-8", len(data), extra)
        self.wfile.write(data)

    def _length(self, mime, maximum):
        if self.headers.get_content_type() != mime:
            raise ValueError("Send " + mime)
        if self.headers.get("Transfer-Encoding"):
            raise ValueError("Chunked request bodies are not accepted")
        raw_length = self.headers.get("Content-Length", "0")
        if not raw_length.isdecimal():
            raise ValueError("Invalid content length")
        length = int(raw_length)
        if not 0 < length <= maximum:
            raise ValueError(f"Request body must be between 1 and {maximum} bytes")
        return length

    def _body(self, maximum=MAX_BODY):
        length = self._length("application/json", maximum)
        raw = self.rfile.read(length)
        if len(raw) != length:
            raise ValueError("Incomplete request body")
        body = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON number")))
        if not isinstance(body, dict):
            raise ValueError("Expected a JSON object")
        return body

    def _guard(self):
        if self.headers.get("Host", "") not in self.server.hosts:
            raise PermissionError("Unrecognized request host")
        origin = self.headers.get("Origin")
        if origin and origin not in self.server.origins:
            raise PermissionError("Cross-origin requests are not allowed")
        if self.headers.get("Sec-Fetch-Site") == "cross-site":
            raise PermissionError("Cross-site requests are not allowed")
        if self.command != "GET" and self.headers.get("X-Chat-Request") != "1":
            raise PermissionError("Missing request header")

    def do_GET(self):
        self._handle()

    def do_POST(self):
        self._handle()

    def do_PATCH(self):
        self._handle()

    def do_DELETE(self):
        self._handle()

    def _handle(self):
        try:
            self._guard()
            self._route()
        except PermissionError as exc:
            self.close_connection = True
            self._json({"error": str(exc)}, 403)
        except KeyError:
            self.close_connection = True
            self._json({"error": "Not found"}, 404)
        except ProjectConflict as exc:
            self.close_connection = True
            self._json({"error": str(exc)}, 409)
        except (ValueError, TypeError, UnicodeError) as exc:
            self.close_connection = True
            self._json({"error": str(exc)[:1000]}, 400)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError):
            self.close_connection = True
        except Exception:
            self.close_connection = True
            self._json({"error": "The chat service could not complete the request."}, 500)

    def _route(self):
        parsed = urlsplit(self.path)
        path, method = parsed.path, self.command
        raw_cookie = self.headers.get("Cookie", "")
        authenticated = self.app.authenticated(raw_cookie)
        if path == "/api/session" and method == "GET":
            return self._json(dict(authenticated=authenticated, password_required=self.app.password is not None))
        if path == "/api/login" and method == "POST":
            token = self.app.login(self._body(MAX_LOGIN_BODY).get("password"), self.client_address[0])
            cookie = f"llmx_chat={token}; Path=/; HttpOnly; SameSite=Strict; Max-Age=43200" + ("; Secure" if self.server.secure else "")
            return self._json({"ok": True}, extra={"Set-Cookie": cookie})
        if path == "/api/logout" and method == "POST":
            self._body()
            self.app.logout(raw_cookie)
            return self._json({"ok": True}, extra={"Set-Cookie": "llmx_chat=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0"})
        if path.startswith("/pdf-reader/"):
            if not authenticated:
                return self._json({"error": "Sign in to continue"}, 401)
            if method != "GET" or parsed.query:
                raise KeyError()
            name = path.removeprefix("/pdf-reader/")
            data, mime = pdf_reader.asset(pdf_reader.READER_DIRECTORY, name)
            policy = pdf_reader.worker_policy(self.server.origins) if name == "pdf.worker.mjs" else None
            self._headers(200, mime, len(data), {"Cross-Origin-Resource-Policy": "same-origin"}, policy=policy)
            return self.wfile.write(data)
        protected_document = path == "/monitor/view" or path.startswith("/preview/")
        if protected_document and not authenticated:
            return self._json({"error": "Sign in to continue"}, 401)
        if protected_document and method == "GET":
            if path == "/monitor/view":
                data = self.app.monitor.document()
                policy = "default-src 'none'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'self'; form-action 'none'"
            else:
                with self.app.lock:
                    created, data, project = self.app.previews[path[len('/preview/'):]]
                if time.monotonic() - created > 1800:
                    raise KeyError()
                assets = " data:" if project else ""
                policy = f"default-src 'none'; script-src 'unsafe-inline'{assets}; style-src 'unsafe-inline'{assets}; img-src data: blob:; connect-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'self'; sandbox allow-scripts"
            self._headers(200, "text/html; charset=utf-8", len(data), policy=policy)
            return self.wfile.write(data)
        if not path.startswith("/api/"):
            if path.startswith("/") and path[1:] in ASSETS and method == "GET" and not parsed.query:
                data, mime = self.app.monitor.asset(path[1:])
                self._headers(200, mime, len(data))
                return self.wfile.write(data)
            if method != "GET" or path not in ("/", "/index.html", "/app.js", "/render.js", "/app.css", "/shell.js", "/shell.css", "/artifacts.js", "/monitor.css", "/workspace.js", "/project_context.js", "/pdf_attachment.js"):
                raise KeyError()
            name = "index.html" if path == "/" else path[1:]
            file = ROOT / name
            if not file.is_file():
                raise KeyError()
            data = file.read_bytes()
            mime = {".js": "text/javascript", ".css": "text/css", ".html": "text/html"}.get(file.suffix, mimetypes.guess_type(name)[0] or "application/octet-stream")
            self._headers(200, mime + "; charset=utf-8", len(data))
            return self.wfile.write(data)
        if not authenticated:
            self.close_connection = True
            return self._json({"error": "Sign in to continue"}, 401)
        if path == "/api/tools" and method == "GET":
            return self._json(self.app.tool_state(discover=False))
        if path == "/api/tools/test" and method == "POST":
            body = self._body(4096)
            if set(body) != {"server"} or not isinstance(body["server"], str):
                raise ValueError("Choose a configured tool connection")
            try:
                self.app.tools.test_connection(body["server"])
            except BlockingIOError as exc:
                return self._json({"error": str(exc)}, 409)
            result = self.app.tool_state(discover=False)
            with self.app.config_lock:
                if self.app.cached_config is not None:
                    self.app.cached_config = dict(self.app.cached_config, **result)
            return self._json(result)
        if path == "/api/config" and method == "GET":
            return self._json(self.app.config())
        if path == "/api/features" and method == "GET" and not parsed.query:
            return self._json(self.app.features.snapshot())
        if path.startswith("/api/features/") and path.endswith("/install") and method == "POST" and not parsed.query:
            if self._body(128):
                raise ValueError("Expected an empty feature install request")
            key = path.removeprefix("/api/features/").removesuffix("/install")
            return self._json(self.app.features.start(key))
        if path == "/api/backup" and method == "GET":
            with tempfile.TemporaryDirectory(prefix="llmx-backup-") as directory:
                destination = Path(directory) / "chat.sqlite3"
                self.app.store.backup(destination)
                self._headers(200, "application/vnd.sqlite3", destination.stat().st_size,
                              {"Content-Disposition": 'attachment; filename="llmx-chat.sqlite3"'})
                with destination.open("rb") as file:
                    while chunk := file.read(65536):
                        self.wfile.write(chunk)
            return
        if path == "/api/restore" and method == "POST":
            remaining = self._length("application/vnd.sqlite3", MAX_BACKUP)
            with tempfile.TemporaryDirectory(prefix="llmx-restore-") as directory:
                source = Path(directory) / "chat.sqlite3"
                with source.open("wb") as file:
                    while remaining:
                        chunk = self.rfile.read(min(remaining, 65536))
                        if not chunk:
                            raise ValueError("Incomplete backup upload")
                        file.write(chunk)
                        remaining -= len(chunk)
                with self.app.engine.lock:
                    if self.app.engine.controls:
                        raise ValueError("Wait for active replies before restoring")
                    self.app.store.restore(source)
            return self._json({"ok": True})
        if path == "/api/workspace" or path.startswith("/api/workspace/"):
            return self._workspace(path, method, parsed.query)
        if path == "/api/monitor/history" and method == "GET":
            query = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True)
            raw = query.get("after", ["0"])
            if set(query) - {"after"} or len(raw) != 1 or not raw[0].isascii() or not raw[0].isdigit() or len(raw[0]) > 20:
                raise ValueError("Invalid monitor cursor")
            return self._json(self.app.monitor.collector.snapshot(int(raw[0])))
        if path == "/api/monitor/stream" and method == "GET" and not parsed.query:
            return self.app.monitor.stream(self, lambda: self.app.authenticated(raw_cookie))
        if path == "/api/previews" and method == "POST":
            body = self._body()
            html = body.get("html")
            if not isinstance(html, str) or len(html.encode()) > MAX_PREVIEW_BYTES:
                raise ValueError("Preview must be an HTML document of at most 1 MiB")
            project = "workspace_path" in body
            metadata = {}
            if project:
                revision, files = self.app.workspace.preview_snapshot()
                data, assets = assemble(body["workspace_path"], html, files)
                metadata = {"workspace": {"path": body["workspace_path"], "revision": revision, "assets": assets}}
            else:
                data = html.encode()
            token = secrets.token_urlsafe(24)
            with self.app.lock:
                while len(self.app.previews) >= 16:
                    self.app.previews.pop(next(iter(self.app.previews)))
                self.app.previews[token] = (time.monotonic(), data, project)
            return self._json({"url": "/preview/" + token, **metadata})
        if path == "/api/chats/run-status" and method == "GET":
            return self._json({"chats": self.app.store.sidebar_run_status()})
        if path == "/api/chats/search" and method == "POST":
            body = self._body(4096)
            query = body.get("query")
            archived = body.get("archived")
            if set(body) != {"query", "archived"} or not isinstance(query, str) or not isinstance(archived, bool):
                raise ValueError("Search requires a query and an archive selection.")
            query = query.strip()
            if not 1 <= len(query) <= 256:
                raise ValueError("Search requires between 1 and 256 characters.")
            try:
                matches = self.app.store.search_chats(query, archived)
            except BlockingIOError as exc:
                return self._json({"error": str(exc)}, 429)
            return self._json({"chats": matches})
        if path == "/api/chats":
            if method == "GET":
                return self._json({"chats": self.app.store.chats()})
            if method == "POST":
                body = self._body()
                temporary = body.pop("temporary", False)
                if not isinstance(temporary, bool):
                    raise ValueError("Invalid temporary flag")
                values = chat_values(body)
                create = self.app.temporary.create if temporary else self.app.store.create_chat
                return self._json(create(values), 201)
        if path == "/api/tokenize" and method == "POST":
            text = self._body().get("text")
            if not isinstance(text, str) or len(text) > 1000000:
                raise ValueError("Invalid text")
            return self._json(Connection(self.app.upstream, timeout=15, key=self.app.api_key).json("/tokenize", {"text": text}))
        if path == "/api/import" and method == "POST":
            payload = self._body()
            source = payload.get("chat", {})
            if not isinstance(source, dict):
                raise ValueError("Invalid imported conversation")
            payload["chat"] = dict(chat_values({k: source[k] for k in ("title", "folder", "settings") if k in source}), active_leaf=source.get("active_leaf"))
            if not isinstance(payload.get("messages"), list):
                raise ValueError("Missing imported messages")
            for message in payload["messages"]:
                if not isinstance(message, dict):
                    raise ValueError("Invalid imported message")
                message["attachments"] = attachments_of(message.get("attachments", []))
            return self._json(self.app.store.import_chat(payload), 201)
        bits = path.strip("/").split("/")
        if len(bits) >= 3 and bits[:2] == ["api", "chats"]:
            cid = bits[2]
            temporary = cid.startswith("tmp_")
            store, engine = self.app.temporary.access(cid) if temporary else (self.app.store, self.app.engine)
            if len(bits) == 3:
                if method == "GET":
                    return self._json(store.snapshot(cid))
                if method == "PATCH":
                    values = chat_values(self._body(), True)
                    if temporary:
                        self.app.temporary.check_settings(values.get("settings", {}))
                    return self._json(store.update_chat(cid, values))
                if method == "DELETE":
                    delete = self.app.temporary.delete if temporary else store.delete_chat
                    delete(cid)
                    return self._json({"ok": True})
            if len(bits) == 4 and bits[3] == "export" and method == "GET":
                snapshot = store.snapshot(cid)
                return self._json(dict(version=1, chat=snapshot["chat"], messages=snapshot["messages"]), extra={"Content-Disposition": 'attachment; filename="conversation.json"'})
            if len(bits) == 4 and bits[3] == "runs" and method == "POST":
                body = self._body()
                if set(body) - {"content", "attachments", "parent_id", "settings", "regenerate"}:
                    raise ValueError("Unsupported run field")
                content = body.get("content", "")
                if not isinstance(content, str) or len(content) > 1000000:
                    raise ValueError("Invalid message text")
                body["attachments"] = attachments_of(body.get("attachments", []))
                if not content.strip() and not body["attachments"] and not body.get("regenerate"):
                    raise ValueError("Write a message or attach a file")
                if "parent_id" in body and body["parent_id"] is not None and not isinstance(body["parent_id"], str):
                    raise ValueError("Invalid parent message")
                if "regenerate" in body and not isinstance(body["regenerate"], bool):
                    raise ValueError("Invalid regenerate flag")
                chat = store.chat(cid)
                settings = settings_of(body.get("settings", chat["settings"]))
                if temporary:
                    self.app.temporary.check_settings(settings)
                catalog = None
                if settings["model"]:
                    catalog = self.app.model_state()
                    if not catalog["upstream_available"]:
                        return self._json({"code": "model_unavailable",
                                           "error": "The model server is unavailable. Try again when it is available.",
                                           **catalog}, 503)
                    if not any(model["id"] == settings["model"] for model in catalog["models"]):
                        return self._json({"code": "model_unavailable",
                                           "error": "This model is not available on the connected server. Choose another model before sending.",
                                           **catalog}, 409)
                result = {"run": engine.start(cid, body, settings)}
                if catalog is not None:
                    result.update(catalog)
                return self._json(result, 201)
        if len(bits) == 4 and bits[:2] == ["api", "runs"]:
            rid, action = bits[2:]
            store, engine = self.app.temporary.access(rid, run=True) if rid.startswith("tmp_") else (self.app.store, self.app.engine)
            if action == "events" and method == "GET":
                cursor = self.headers.get("Last-Event-ID", parse_qs(parsed.query).get("after", ["0"])[0])
                if not cursor.isdecimal():
                    raise ValueError("Invalid event cursor")
                return self._events(rid, int(cursor), store)
            if action == "cancel" and method == "POST":
                self._body()
                engine.cancel(rid)
                return self._json({"ok": True})
            if action == "approve" and method == "POST":
                body = self._body()
                if not isinstance(body.get("call_id"), str) or not isinstance(body.get("allow"), bool) or not isinstance(body.get("remember", False), bool):
                    raise ValueError("Invalid approval")
                engine.approve(rid, body["call_id"], body["allow"], body.get("remember", False))
                return self._json({"ok": True})
        raise KeyError()

    def _workspace(self, path, method, query):
        workspace = self.app.workspace
        parts = path.split("/")[3:]
        if not parts and method == "GET":
            return self._json(workspace.snapshot())
        if parts == ["files"] and method == "POST" and not query:
            # Allow six bytes per escaped text byte plus the paths and JSON framing.
            body = self._body(7 * MAX_PROJECT)
            return self._json(workspace.import_files(body.get("files"), body.get("revision")))
        if parts == ["files"] and method == "GET":
            if not query:
                return self._json(workspace.files())
            if re.search(r"%(?![0-9a-fA-F]{2})", query):
                raise ValueError("Invalid Workspace file query")
            params = parse_qs(query, keep_blank_values=True, strict_parsing=True, errors="strict", max_num_fields=MAX_SELECTED_FILES)
            if set(params) != {"path"}:
                raise ValueError("Expected saved Workspace file paths")
            return self._json(workspace.files(params["path"]))
        if parts and parts[0] == "projects" and not query:
            if len(parts) == 1 and method == "GET":
                return self._json({"projects": workspace.projects()})
            if len(parts) == 1 and method == "POST":
                return self._json({"project": workspace.save_project(self._body(128 * 1024))}, 201)
            if len(parts) == 2 and method == "PATCH":
                return self._json({"project": workspace.save_project(self._body(128 * 1024), parts[1])})
            if len(parts) == 2 and method == "DELETE":
                body = self._body(128)
                if set(body) != {"version"}:
                    raise ValueError("Expected a Workspace project version")
                workspace.delete_project(parts[1], body["version"])
                return self._json({"ok": True})
            if len(parts) == 3 and parts[2] == "context" and method == "GET":
                return self._json(workspace.project_context(parts[1]))
            if len(parts) == 3 and parts[2] == "archive" and method == "GET":
                name, data = workspace.project_archive(parts[1])
                slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-_") or parts[1]
                self._headers(200, "application/zip", len(data),
                              {"Content-Disposition": f'attachment; filename="llmx-{slug}.zip"'})
                return self.wfile.write(data)
        if parts == ["file"] and method == "GET":
            params = parse_qs(query, keep_blank_values=True, strict_parsing=True)
            if set(params) != {"path"} or len(params["path"]) != 1:
                raise ValueError("Expected one file path")
            return self._json(workspace.read(params["path"][0]))
        if parts in (["file"], ["delete"]) and method == "POST":
            body = self._body()
            return self._json(workspace.write(body.get("path"), body.get("content"), body.get("revision"), parts == ["delete"]))
        if parts == ["checkpoints"] and method == "POST":
            return self._json(workspace.checkpoint(self._body().get("label")))
        if len(parts) == 3 and parts[0] == "checkpoints" and parts[2] == "restore" and method == "POST":
            return self._json(workspace.restore(parts[1], self._body().get("revision")))
        if parts == ["diff"] and method == "POST":
            body = self._body()
            return self._json(workspace.draft_diff(body.get("path"), body.get("content")))
        if parts == ["diff"] and method == "GET":
            params = parse_qs(query, keep_blank_values=True, strict_parsing=True)
            if set(params) != {"checkpoint"} or len(params["checkpoint"]) != 1:
                raise ValueError("Expected one checkpoint")
            return self._json(workspace.diff(params["checkpoint"][0]))
        if parts == ["jobs"] and method == "POST":
            return self._json({"job": workspace.start(self._body().get("command"))})
        if len(parts) == 2 and parts[0] == "jobs" and method == "GET":
            return self._json({"job": workspace.job(parts[1])})
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "cancel" and method == "POST":
            self._body()
            return self._json({"job": workspace.cancel(parts[1])})
        raise KeyError()

    def _events(self, rid, after, store):
        store.run(rid)
        self.close_connection = True
        self._headers(200, "text/event-stream", extra={"Connection": "close", "X-Accel-Buffering": "no"})
        try:
            while True:
                with store.changed:
                    events = store.events(rid, after)
                    run = store.run(rid)
                    if not events and run["status"] not in TERMINAL:
                        store.changed.wait(10)
                for event in events:
                    self.wfile.write(f"id: {event['seq']}\nevent: update\ndata: {encoded(event)}\n\n".encode())
                    after = event["seq"]
                self.wfile.write(b": heartbeat\n\n")
                self.wfile.flush()
                if run["status"] in TERMINAL and after >= run["last_seq"]:
                    break
        except (OSError, TimeoutError, KeyError):
            pass


def main():
    parser = argparse.ArgumentParser(description="A focused chat workspace for llmx.")
    parser.add_argument("--upstream", default="http://127.0.0.1:8090", help="llmx HTTP origin or /v1 URL")
    parser.add_argument("--host", default="127.0.0.1", help="Listen address (default: loopback)")
    parser.add_argument("--port", type=int, default=8183)
    parser.add_argument("--data-dir", type=Path, default=Path.home() / ".llmx-chat")
    parser.add_argument("--tools", type=Path, help="Operator-managed MCP HTTP server configuration")
    parser.add_argument("--password-file", type=Path, help="Owner password file, required for non-loopback listening")
    parser.add_argument("--api-key-file", type=Path, help="Optional upstream API key file")
    parser.add_argument("--public-origin", help="Browser-facing origin, for example https://chat.example.com")
    parser.add_argument("--sandbox-image", help="Existing Linux container image for isolated workspace commands")
    parser.add_argument("--sandbox-docker-host", help="Optional Docker endpoint, for example ssh://user@host")
    args = parser.parse_args()
    password = args.password_file.read_text(encoding="utf-8").strip() if args.password_file else None
    if password is not None and not 12 <= len(password) <= 1024:
        parser.error("Use an owner password of 12 to 1024 characters")
    if args.host not in ("127.0.0.1", "localhost") and (not password or not args.public_origin):
        parser.error("Non-loopback listening needs --password-file and --public-origin")
    Connection(args.upstream).close()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    key = args.api_key_file.read_text(encoding="utf-8").strip() if args.api_key_file else None
    app = Application(args.data_dir / "chat.sqlite3", args.upstream, args.tools, password, key, args.sandbox_image, args.sandbox_docker_host)
    try:
        server = Server((args.host, args.port), app, args.public_origin)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"llmx chat: http://{args.host}:{server.server_address[1]}", flush=True)
    print("Owner workspace. Conversation data stays in the configured data directory.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        app.engine.shutdown()


if __name__ == "__main__":
    main()
