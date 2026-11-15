"""Durable conversations and the event journal behind chat replay."""

import json
import os
import sqlite3
import tempfile
import threading
import time
from urllib.parse import quote
import uuid


SCHEMA_VERSION = 1


def identity():
    return uuid.uuid4().hex


def encoded(value):
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


class Store:
    def __init__(self, path, id_prefix=""):
        self.lock = threading.RLock()
        self.changed = threading.Condition(self.lock)
        self.database_path = os.fspath(path)
        self.id_prefix = id_prefix
        self.file_path = None if self.database_path == ":memory:" else os.path.realpath(self.database_path)
        self.db = sqlite3.connect(self.database_path, check_same_thread=False)
        version = self.db.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            self.db.close()
            raise ValueError("Database schema is newer than this chat service supports.")
        self.db.execute("PRAGMA journal_mode=WAL")
        if self.file_path is None:
            self.db.execute("PRAGMA temp_store=MEMORY")
        self.db.execute("PRAGMA foreign_keys=ON")
        self._migrate(version)
        self._recover_runs()

    def _recover_runs(self):
        for row in self.db.execute("SELECT data FROM runs").fetchall():
            run = json.loads(row[0])
            if run["status"] in ("running", "approval"):
                run.update(status="interrupted", error="The chat service restarted. The previous operation was not retried.")
                if run.get("message_id"):
                    message = self.message(run["message_id"])
                    message["status"] = "interrupted"
                    for call in message.get("tool_calls", []):
                        if call["status"] in ("pending", "approval", "running"):
                            call.update(status="interrupted", error="Outcome unknown after service restart.")
                    self.publish(run["id"], "message", message)
                self.publish(run["id"], "state", run)

    def _migrate(self, version):
        if version == SCHEMA_VERSION:
            return
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute("CREATE TABLE IF NOT EXISTS chats(id TEXT PRIMARY KEY, data TEXT NOT NULL)")
            self.db.execute("CREATE TABLE IF NOT EXISTS messages(id TEXT PRIMARY KEY, chat TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE, parent TEXT, data TEXT NOT NULL)")
            self.db.execute("CREATE INDEX IF NOT EXISTS messages_chat ON messages(chat)")
            self.db.execute("CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, chat TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE, data TEXT NOT NULL)")
            self.db.execute("CREATE INDEX IF NOT EXISTS runs_chat ON runs(chat)")
            self.db.execute("CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT, run TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE, data TEXT NOT NULL)")
            self.db.execute("CREATE INDEX IF NOT EXISTS events_run ON events(run, seq)")
            self.db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            self.db.commit()
        except Exception:
            self.db.rollback()
            self.db.close()
            raise

    @staticmethod
    def _validate_database(db):
        version = db.execute("PRAGMA user_version").fetchone()[0]
        if version != SCHEMA_VERSION:
            raise ValueError("Backup schema version is not supported.")
        required = {
            "chats": {"id", "data"},
            "messages": {"id", "chat", "parent", "data"},
            "runs": {"id", "chat", "data"},
            "events": {"seq", "run", "data"},
        }
        for table, columns in required.items():
            actual = {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
            if not columns.issubset(actual):
                raise ValueError("Backup schema is invalid.")
        extras = db.execute(
            "SELECT type FROM sqlite_master WHERE type IN ('trigger','view')"
        ).fetchall()
        if extras:
            raise ValueError("Backup schema contains unsupported objects.")
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Backup integrity check failed.")
        if db.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise ValueError("Backup contains invalid references.")
        checks = (
            ("chats", "SELECT id,data FROM chats", ("id", "title")),
            ("messages", "SELECT id,chat,data FROM messages", ("id", "chat_id", "role", "content", "status")),
            ("runs", "SELECT id,chat,data FROM runs", ("id", "chat_id", "status", "message_id")),
            ("events", "SELECT seq,run,data FROM events", ("kind", "data", "time")),
        )
        for table, query, fields in checks:
            for row in db.execute(query):
                try:
                    value = json.loads(row[-1])
                except (TypeError, json.JSONDecodeError, RecursionError) as exc:
                    raise ValueError("Backup contains invalid record data.") from exc
                if not isinstance(value, dict):
                    raise ValueError("Backup contains invalid record data.")
                if table == "events":
                    if (not isinstance(value.get("kind"), str) or
                            not isinstance(value.get("data"), dict) or
                            not isinstance(value.get("time"), (int, float))):
                        raise ValueError("Backup contains invalid record data.")
                    continue
                expected_id = row[0]
                if value.get("id") != expected_id:
                    raise ValueError("Backup record identity does not match its row.")
                if table in ("chats", "runs") and (not isinstance(expected_id, str) or expected_id.startswith("tmp_")):
                    raise ValueError("Backup contains a reserved temporary identifier.")
                if table in ("messages", "runs") and value.get("chat_id") != row[1]:
                    raise ValueError("Backup record identity does not match its row.")
                if any(field not in value for field in fields):
                    raise ValueError("Backup contains incomplete record data.")
                if table == "chats":
                    if "temporary" in value:
                        raise ValueError("Backup contains reserved temporary chat metadata.")
                    if not isinstance(value["title"], str):
                        raise ValueError("Backup contains invalid record data.")
                elif table == "messages":
                    if (value["role"] not in ("user", "assistant", "tool") or
                            not isinstance(value["content"], str) or
                            not isinstance(value["status"], str)):
                        raise ValueError("Backup contains invalid record data.")
                elif table == "runs":
                    if (not isinstance(value["status"], str) or
                            value["message_id"] is not None and not isinstance(value["message_id"], str)):
                        raise ValueError("Backup contains invalid record data.")

    @staticmethod
    def _temporary_path(directory):
        fd, path = tempfile.mkstemp(prefix=".llmx-chat-", suffix=".sqlite", dir=directory)
        os.close(fd)
        return path

    def _same_file(self, other):
        other = os.path.abspath(os.fspath(other))
        if self.file_path and os.path.normcase(other) == os.path.normcase(self.file_path):
            return True
        try:
            return self.file_path is not None and os.path.samefile(other, self.file_path)
        except OSError:
            return False

    def backup(self, destination):
        destination = os.path.abspath(os.fspath(destination))
        if self._same_file(destination):
            raise ValueError("Backup destination must differ from the active database.")
        directory = os.path.dirname(destination) or os.curdir
        temporary = self._temporary_path(directory)
        try:
            with self.lock:
                target = sqlite3.connect(temporary)
                try:
                    self.db.backup(target)
                    target.commit()
                    self._validate_database(target)
                finally:
                    target.close()
            os.replace(temporary, destination)
            temporary = None
        except sqlite3.DatabaseError as exc:
            raise ValueError("Could not create a valid database backup.") from exc
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)

    def restore(self, source):
        if self.file_path is None:
            raise ValueError("Restore requires a file-backed database.")
        source = os.path.abspath(os.fspath(source))
        if self._same_file(source):
            raise ValueError("Restore source must differ from the active database.")
        directory = os.path.dirname(self.file_path)
        prepared = self._temporary_path(directory)
        rollback = None
        try:
            uri_path = quote(source.replace("\\", "/"), safe="/:")
            original = sqlite3.connect("file:" + uri_path + "?mode=ro", uri=True)
            candidate = sqlite3.connect(prepared)
            try:
                original.backup(candidate)
                candidate.commit()
                self._validate_database(candidate)
            finally:
                candidate.close()
                original.close()
        except (sqlite3.DatabaseError, OSError, ValueError) as exc:
            if os.path.exists(prepared):
                os.unlink(prepared)
            raise ValueError("Restore source is not a valid chat database.") from exc

        try:
            with self.lock:
                if self._has_active_runs():
                    raise ValueError("Stop active replies before restoring a database.")
                rollback = self._temporary_path(directory)
                saved = sqlite3.connect(rollback)
                try:
                    self.db.backup(saved)
                    saved.commit()
                    self._validate_database(saved)
                finally:
                    saved.close()
                self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                self.db.execute("PRAGMA journal_mode=DELETE")
                self.db.close()
                self.db = None
                try:
                    os.replace(prepared, self.file_path)
                    prepared = None
                    self.db = sqlite3.connect(self.database_path, check_same_thread=False)
                    self.db.execute("PRAGMA foreign_keys=ON")
                    self.db.execute("PRAGMA journal_mode=WAL")
                    self._recover_runs()
                except Exception:
                    if self.db is not None:
                        self.db.close()
                        self.db = None
                    if rollback and os.path.exists(rollback):
                        os.replace(rollback, self.file_path)
                        rollback = None
                    self.db = sqlite3.connect(self.database_path, check_same_thread=False)
                    self.db.execute("PRAGMA foreign_keys=ON")
                    self.db.execute("PRAGMA journal_mode=WAL")
                    raise ValueError("Restore failed; the original database was restored.") from None
        finally:
            for path in (prepared, rollback):
                if path and os.path.exists(path):
                    os.unlink(path)

    def _has_active_runs(self):
        return any(json.loads(row[0])["status"] in ("running", "approval")
                   for row in self.db.execute("SELECT data FROM runs"))

    def close(self):
        with self.lock:
            self.db.close()

    def _get(self, table, key):
        row = self.db.execute(f"SELECT data FROM {table} WHERE id=?", (key,)).fetchone()
        if not row:
            raise KeyError("Not found")
        return json.loads(row[0])

    def chat(self, key):
        with self.lock:
            return self._get("chats", key)

    def message(self, key):
        with self.lock:
            return self._get("messages", key)

    def run(self, key):
        with self.lock:
            return self._get("runs", key)

    def chats(self):
        with self.lock:
            return sorted((json.loads(r[0]) for r in self.db.execute("SELECT data FROM chats")),
                          key=lambda c: (not c["pinned"], -c["updated"]))

    def create_chat(self, values):
        now = time.time()
        chat = dict(id=self.id_prefix + identity(), title=values.get("title", "New conversation"), folder=values.get("folder", ""),
                    pinned=False, archived=False, created=now, updated=now, active_leaf=None, settings=values.get("settings", {}))
        with self.lock, self.db:
            self.db.execute("INSERT INTO chats VALUES(?,?)", (chat["id"], encoded(chat)))
        return chat

    def snapshot(self, key):
        with self.lock:
            return dict(chat=self.chat(key),
                        messages=[json.loads(r[0]) for r in self.db.execute("SELECT data FROM messages WHERE chat=? ORDER BY rowid", (key,))],
                        runs=[json.loads(r[0]) for r in self.db.execute("SELECT data FROM runs WHERE chat=? ORDER BY rowid", (key,))])

    def _idle(self, key):
        updated = 0
        for row in self.db.execute("SELECT data FROM runs WHERE chat=?", (key,)):
            run = json.loads(row[0])
            if run["status"] in ("running", "approval"):
                raise ValueError("Stop the active reply before changing this conversation.")
            updated = max(updated, run["updated"])
        return updated

    def update_chat(self, key, values):
        with self.lock, self.db:
            chat = self.chat(key)
            if "active_leaf" in values:
                self._idle(key)
                leaf = values["active_leaf"]
                if leaf is not None and self.message(leaf)["chat_id"] != key:
                    raise ValueError("The selected branch belongs to another conversation.")
            chat.update(values, updated=time.time())
            self.db.execute("UPDATE chats SET data=? WHERE id=?", (encoded(chat), key))
            return chat

    def delete_chat(self, key):
        with self.lock, self.db:
            self.chat(key)
            self._idle(key)
            self.db.execute("DELETE FROM chats WHERE id=?", (key,))

    def path(self, leaf):
        with self.lock:
            chain, seen = [], set()
            while leaf:
                if leaf in seen:
                    raise ValueError("Invalid conversation branch")
                seen.add(leaf)
                m = self.message(leaf)
                chain.append(m)
                leaf = m["parent_id"]
            return list(reversed(chain))

    def start(self, chat_id, body, settings):
        with self.lock, self.db:
            chat = self.chat(chat_id)
            self._idle(chat_id)
            parent = body.get("parent_id", chat["active_leaf"])
            if parent and self.message(parent)["chat_id"] != chat_id:
                raise ValueError("The parent belongs to another conversation.")
            if body.get("regenerate") and (not parent or self.message(parent)["role"] != "user"):
                raise ValueError("Regeneration needs a user message.")
            rid, now = self.id_prefix + identity(), time.time()
            run = dict(id=rid, chat_id=chat_id, status="running", message_id=None, created=now,
                       updated=now, error=None, last_seq=0, settings=settings)
            self.db.execute("INSERT INTO runs VALUES(?,?,?)", (rid, chat_id, encoded(run)))
            if not body.get("regenerate"):
                user = self.new_message(chat_id, parent, "user", rid, content=body.get("content", ""),
                                        attachments=body.get("attachments", []), status="completed")
                self._publish(rid, "message", user)
                parent = user["id"]
                if chat["title"] == "New conversation":
                    chat["title"] = (body.get("content", "").strip() or "Attachment conversation")[:80]
            message = self.new_message(chat_id, parent, "assistant", rid)
            self._publish(rid, "message", message)
            run.update(message_id=message["id"])
            self._publish(rid, "state", run)
            chat.update(active_leaf=message["id"], updated=now, settings=settings)
            self.db.execute("UPDATE chats SET data=? WHERE id=?", (encoded(chat), chat_id))
            self.changed.notify_all()
            return self.run(rid)

    @staticmethod
    def new_message(chat_id, parent, role, rid, **values):
        return dict(id=identity(), chat_id=chat_id, parent_id=parent, role=role, run_id=rid,
                    created=time.time(), **dict(dict(content="", reasoning="", attachments=[], tool_calls=[], status="running"), **values))

    def _save_message(self, message):
        self.db.execute("INSERT INTO messages VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET data=excluded.data",
                        (message["id"], message["chat_id"], message["parent_id"], encoded(message)))

    def _publish(self, rid, kind, data):
        run = self.run(rid)
        if kind == "message":
            self._save_message(data)
            chat = self.chat(run["chat_id"])
            chat.update(active_leaf=data["id"], updated=time.time())
            self.db.execute("UPDATE chats SET data=? WHERE id=?", (encoded(chat), chat["id"]))
        elif kind in ("delta", "metrics", "tool"):
            m = self.message(data["message_id"])
            if kind == "delta":
                for key in ("content", "reasoning"):
                    m[key] += data.get(key, "")
            elif kind == "metrics":
                for key in ("usage", "timings", "context"):
                    if key in data:
                        m[key] = data[key]
            else:
                found = False
                for index, call in enumerate(m["tool_calls"]):
                    if call["id"] == data["call"]["id"]:
                        m["tool_calls"][index] = data["call"]
                        found = True
                        break
                if not found:
                    m["tool_calls"].append(data["call"])
            self._save_message(m)
        elif kind == "state":
            run.update(data)
        else:
            raise ValueError("Unknown event")
        now = time.time()
        event = dict(kind=kind, data=data, time=now)
        cursor = self.db.execute("INSERT INTO events(run,data) VALUES(?,?)", (rid, encoded(event)))
        run.update(last_seq=cursor.lastrowid, updated=now)
        if kind == "state":
            event["data"] = run
            self.db.execute("UPDATE events SET data=? WHERE seq=?", (encoded(event), cursor.lastrowid))
        self.db.execute("UPDATE runs SET data=? WHERE id=?", (encoded(run), rid))
        return dict(event, seq=cursor.lastrowid)

    def publish(self, rid, kind, data):
        with self.changed, self.db:
            event = self._publish(rid, kind, data)
            self.changed.notify_all()
            return event

    def events(self, rid, after):
        with self.lock:
            self.run(rid)
            return [dict(json.loads(data), seq=seq) for seq, data in self.db.execute(
                "SELECT seq,data FROM events WHERE run=? AND seq>? ORDER BY seq LIMIT 256", (rid, after))]

    def import_chat(self, payload):
        if payload.get("version") != 1 or not isinstance(payload.get("messages"), list):
            raise ValueError("Not a version 1 chat export")
        messages = payload["messages"]
        if len(messages) > 20000:
            raise ValueError("Too many imported messages")
        mapping, converted = {}, []
        for m in messages:
            if not isinstance(m, dict) or not isinstance(m.get("id"), str) or m["id"] in mapping:
                raise ValueError("Invalid message identifiers")
            parent = m.get("parent_id")
            if parent is not None and parent not in mapping:
                raise ValueError("Parents must precede their children")
            if m.get("role") not in ("user", "assistant", "tool") or not isinstance(m.get("content"), str):
                raise ValueError("Invalid imported message")
            mapped = identity()
            status = m.get("status", "completed")
            if status not in ("completed", "cancelled", "failed", "interrupted"):
                status = "interrupted"
            clean = dict(id=mapped, parent_id=mapping.get(parent), run_id=None, role=m["role"],
                         content=m["content"], reasoning=m.get("reasoning", ""), created=m.get("created", time.time()),
                         status=status, attachments=m.get("attachments", []), tool_calls=m.get("tool_calls", []))
            if not isinstance(clean["reasoning"], str) or not isinstance(clean["tool_calls"], list):
                raise ValueError("Invalid imported message content")
            for call in clean["tool_calls"]:
                if not isinstance(call, dict) or any(not isinstance(call.get(k), str) for k in ("id", "name", "arguments")):
                    raise ValueError("Invalid imported tool call")
                if call.get("status") not in ("completed", "denied", "failed", "cancelled", "interrupted"):
                    call["status"] = "interrupted"
            if m["role"] == "tool":
                if not isinstance(m.get("tool_call_id"), str):
                    raise ValueError("Invalid imported tool result")
                clean["tool_call_id"] = m["tool_call_id"]
            for key in ("usage", "timings", "context"):
                if key in m:
                    if not isinstance(m[key], dict):
                        raise ValueError("Invalid imported metrics")
                    clean[key] = m[key]
            converted.append(clean)
            mapping[m["id"]] = mapped
        source = payload.get("chat", {})
        leaf = source.get("active_leaf")
        if leaf is not None and leaf not in mapping:
            raise ValueError("Invalid active branch")
        with self.lock, self.db:
            chat = self.create_chat(source)
            for m in converted:
                m.update(chat_id=chat["id"])
                self._save_message(m)
            chat.update(active_leaf=mapping.get(leaf))
            self.db.execute("UPDATE chats SET data=? WHERE id=?", (encoded(chat), chat["id"]))
            return chat
