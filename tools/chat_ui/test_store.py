import json
import os
import sqlite3
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from unittest import mock

from store import SCHEMA_VERSION, Store


class StorePersistenceTests(unittest.TestCase):
    def test_preparing_tool_snapshot_recovers_as_interrupted(self):
        path = self.path("tool-stream")
        store = Store(path)
        chat = store.create_chat({})
        run = store.start(chat["id"], {"content": "question"}, self.settings())
        fragment = dict(id="call", index=0, name="echo", arguments='{"text":')
        store.publish(run["id"], "delta", dict(message_id=run["message_id"], tool_calls=[fragment]))
        before = store.snapshot(chat["id"])
        store.close()
        recovered = Store(path)
        try:
            after = recovered.snapshot(chat["id"])
            self.assertEqual(len(before["messages"][-1]["tool_calls"]), 1)
            call = after["messages"][-1]["tool_calls"][0]
            self.assertEqual(call["arguments"], fragment["arguments"])
            self.assertEqual(call["status"], "interrupted")
        finally:
            recovered.close()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = self.temp.name

    def tearDown(self):
        self.temp.cleanup()

    def path(self, name):
        return os.path.join(self.directory, name + ".sqlite")

    @staticmethod
    def settings():
        return {"model": "fixture", "tools": []}

    def populated(self, path, title):
        store = Store(path)
        chat = store.create_chat({"title": title})
        run = store.start(chat["id"], {"content": "question"}, self.settings())
        answer = store.message(run["message_id"])
        answer.update(content="answer: " + title, status="completed")
        store.publish(run["id"], "message", answer)
        store.publish(run["id"], "metrics", {"message_id": answer["id"], "usage": {"total_tokens": 3}})
        store.publish(run["id"], "state", dict(store.run(run["id"]), status="completed"))
        return store, chat["id"], run["id"]

    def test_search_saved_content_and_current_metadata(self):
        store = Store(self.path("search"))
        try:
            chat = store.import_chat({"version": 1, "chat": {"title": "Saved title", "folder": "Project folder",
                                      "active_leaf": "visible", "settings": {"system": "private-setting"}}, "messages": [
                {"id": "prompt", "role": "user", "content": "Stra\u00dfe %_ prompt",
                 "attachments": [{"name": "private-file", "text": "private-attachment"}]},
                {"id": "visible", "parent_id": "prompt", "role": "assistant", "content": "visible answer"},
                {"id": "branch", "parent_id": "prompt", "role": "assistant", "content": "alternate answer",
                 "reasoning": "private-reasoning", "tool_calls": [{"id": "invocation", "name": "private-name",
                 "arguments": "private-arguments", "result": "private-result", "status": "completed"}]},
                {"id": "tool", "parent_id": "branch", "role": "tool", "content": "private-tool-message", "tool_call_id": "invocation"},
            ]})
            cid = chat["id"]
            for query in ("saved title", "project folder", "STRASSE %_", "visible answer", "alternate answer"):
                with self.subTest(query=query):
                    [summary] = store.search_chats(query, False)
                    self.assertEqual(summary["id"], cid)
                    self.assertEqual(set(summary), {"id", "title", "folder", "pinned", "archived", "updated"})
            self.assertEqual(store.search_chats("private-", False), [])
            self.assertEqual(store.search_chats("%x", False), [])
            store.update_chat(cid, {"title": "Renamed", "archived": True})
            self.assertEqual(store.search_chats("Saved title", True), [])
            self.assertEqual(store.search_chats("visible answer", False), [])
            self.assertEqual(store.search_chats("renamed", True)[0]["id"], cid)
            run = store.start(cid, {"content": "new turn"}, self.settings())
            store.publish(run["id"], "delta", {"message_id": run["message_id"], "content": "freshly persisted answer"})
            self.assertEqual(store.search_chats("freshly persisted", True)[0]["id"], cid)
            self.assertEqual(store.search_chats("delta", True), [])
            store.publish(run["id"], "state", {"status": "completed"})
            store.delete_chat(cid)
            self.assertEqual(store.search_chats("alternate answer", True), [])
        finally:
            store.close()

    def test_search_reader_gates_restore_and_reopens_replaced_database(self):
        active_path, backup_path = self.path("active-search"), self.path("restore-search")
        active, original_id, _ = self.populated(active_path, "original searchable")
        backup, restored_id, _ = self.populated(backup_path, "restored searchable")
        backup.close()
        opened, release = threading.Event(), threading.Event()
        connect = sqlite3.connect

        def hold_reader(database, *args, **kwargs):
            db = connect(database, *args, **kwargs)
            if "active-search.sqlite?mode=ro" in str(database):
                opened.set()
                if not release.wait(5):
                    db.close()
                    raise TimeoutError("Reader was not released")
            return db

        try:
            with ThreadPoolExecutor(max_workers=2) as pool, mock.patch("store.sqlite3.connect", side_effect=hold_reader):
                scan = pool.submit(active.search_chats, "original", False)
                try:
                    self.assertTrue(opened.wait(2), "search did not open its read-only connection")
                    with self.assertRaises(BlockingIOError):
                        active.search_chats("another query", False)
                    restore = pool.submit(active.restore, backup_path)
                    with self.assertRaises(TimeoutError):
                        restore.result(timeout=0.1)
                finally:
                    release.set()
                self.assertEqual(scan.result(timeout=5)[0]["id"], original_id)
                restore.result(timeout=5)
            self.assertEqual(active.search_chats("original", False), [])
            self.assertEqual(active.search_chats("restored", False)[0]["id"], restored_id)
        finally:
            active.close()

    def test_unversioned_database_migrates_without_losing_data(self):
        path = self.path("legacy")
        store, chat_id, run_id = self.populated(path, "legacy chat")
        before = store.snapshot(chat_id)
        before_events = store.events(run_id, 0)
        store.close()
        connection = sqlite3.connect(path)
        connection.execute("PRAGMA user_version=0")
        connection.commit()
        connection.close()

        upgraded = Store(path)
        try:
            self.assertEqual(upgraded.db.execute("PRAGMA user_version").fetchone()[0], SCHEMA_VERSION)
            self.assertEqual(upgraded.snapshot(chat_id), before)
            self.assertEqual(upgraded.events(run_id, 0), before_events)
        finally:
            upgraded.close()

    def test_newer_schema_is_refused_without_mutation(self):
        path = self.path("newer")
        store, _, _ = self.populated(path, "newer database")
        store.close()
        connection = sqlite3.connect(path)
        version = SCHEMA_VERSION + 1
        connection.execute("PRAGMA user_version=%d" % version)
        connection.commit()
        connection.close()
        with self.assertRaisesRegex(ValueError, "newer"):
            Store(path)
        connection = sqlite3.connect(path)
        try:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], version)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM chats").fetchone()[0], 1)
        finally:
            connection.close()

    def test_online_backup_preserves_a_consistent_chat_and_event_log(self):
        source_path = self.path("live")
        backup_path = self.path("backup")
        source, chat_id, run_id = self.populated(source_path, "backup source")
        expected = source.snapshot(chat_id)
        events = source.events(run_id, 0)
        source.backup(backup_path)
        copied = Store(backup_path)
        try:
            self.assertEqual(copied.snapshot(chat_id), expected)
            self.assertEqual(copied.events(run_id, 0), events)
        finally:
            copied.close()
            source.close()

    def test_backup_refuses_the_active_database_path(self):
        path = self.path("same")
        store, _, _ = self.populated(path, "same file")
        try:
            with self.assertRaisesRegex(ValueError, "differ"):
                store.backup(path)
        finally:
            store.close()

    def test_invalid_restore_sources_leave_original_unchanged(self):
        destination_path = self.path("destination")
        destination, chat_id, run_id = self.populated(destination_path, "keep me")
        baseline = destination.snapshot(chat_id)
        baseline_events = destination.events(run_id, 0)
        invalid_paths = [self.path("garbage"), self.path("wrong-schema")]
        with open(invalid_paths[0], "wb") as output:
            output.write(b"not a sqlite database")
        malformed = sqlite3.connect(invalid_paths[1])
        malformed.execute("PRAGMA user_version=%d" % SCHEMA_VERSION)
        malformed.execute("CREATE TABLE unrelated(value TEXT)")
        malformed.commit()
        malformed.close()
        try:
            for source_path in invalid_paths:
                with self.subTest(source=source_path):
                    with self.assertRaises(ValueError):
                        destination.restore(source_path)
                    self.assertEqual(destination.snapshot(chat_id), baseline)
                    self.assertEqual(destination.events(run_id, 0), baseline_events)
        finally:
            destination.close()

    def test_malformed_backup_records_and_schema_objects_leave_original_unchanged(self):
        destination, chat_id, run_id = self.populated(self.path("destination-records"), "keep me")
        baseline = destination.snapshot(chat_id)
        baseline_events = destination.events(run_id, 0)
        source, source_chat, _ = self.populated(self.path("source-records"), "candidate")
        source.close()

        mutations = [
            ("chat-json", "UPDATE chats SET data='not json'"),
            ("chat-array", "UPDATE chats SET data='[]'"),
            ("temporary-flag", "UPDATE chats SET data=json_set(data,'$.temporary',json('true'))"),
            ("temporary-truthy", "UPDATE chats SET data=json_set(data,'$.temporary','true')"),
            ("temporary-object", "UPDATE chats SET data=json_set(data,'$.temporary',json('{}'))"),
            ("chat-id-mismatch", "UPDATE chats SET data=json_set(data,'$.id','wrong')"),
            ("message-id-mismatch", "UPDATE messages SET data=json_set(data,'$.id','wrong')"),
            ("run-chat-mismatch", "UPDATE runs SET data=json_set(data,'$.chat_id','wrong')"),
            ("event-json", "UPDATE events SET data='[]'"),
            ("trigger", "CREATE TRIGGER extra AFTER INSERT ON chats BEGIN SELECT 1; END"),
            ("view", "CREATE VIEW extra AS SELECT id FROM chats"),
        ]
        try:
            for name, statement in mutations:
                with self.subTest(name=name):
                    candidate_path = self.path("candidate-" + name)
                    candidate, _, _ = self.populated(candidate_path, "candidate")
                    candidate.close()
                    connection = sqlite3.connect(candidate_path)
                    try:
                        connection.execute(statement)
                        connection.commit()
                    finally:
                        connection.close()
                    with self.assertRaises(ValueError):
                        destination.restore(candidate_path)
                    self.assertEqual(destination.snapshot(chat_id), baseline)
                    self.assertEqual(destination.events(run_id, 0), baseline_events)
        finally:
            destination.close()

    def test_restore_rejects_reserved_temporary_chat_and_run_identifiers(self):
        destination, cid, _ = self.populated(self.path("reserved-destination"), "keep me")
        baseline = destination.snapshot(cid)
        try:
            for table in ("chats", "runs"):
                with self.subTest(table=table):
                    path = self.path("reserved-" + table)
                    candidate = Store(path)
                    chat = candidate.create_chat({})
                    if table == "runs":
                        run = candidate.start(chat["id"], {"content": "candidate"}, self.settings())
                        candidate.publish(run["id"], "state", dict(run, status="completed"))
                    candidate.close()
                    connection = sqlite3.connect(path)
                    try:
                        connection.execute(f"UPDATE {table} SET id='tmp_reserved',data=json_set(data,'$.id','tmp_reserved')")
                        if table == "runs":
                            connection.execute("UPDATE events SET run='tmp_reserved'")
                        connection.commit()
                    finally:
                        connection.close()
                    with self.assertRaises(ValueError):
                        destination.restore(path)
                    self.assertEqual(destination.snapshot(cid), baseline)
        finally:
            destination.close()

    def test_successful_restore_replaces_database_contents(self):
        source, source_chat, source_run = self.populated(self.path("source"), "restored")
        destination, destination_chat, _ = self.populated(self.path("destination"), "discarded")
        expected = source.snapshot(source_chat)
        expected_events = source.events(source_run, 0)
        try:
            destination.restore(self.path("source"))
            self.assertEqual(destination.snapshot(source_chat), expected)
            self.assertEqual(destination.events(source_run, 0), expected_events)
            with self.assertRaises(KeyError):
                destination.chat(destination_chat)
        finally:
            source.close()
            destination.close()

    def test_restore_rejects_invalid_message_graphs_without_changing_destination(self):
        source_path = self.path("graph-source")
        source, chat_id, _ = self.populated(source_path, "graph source")
        other_chat = source.create_chat({"title": "other chat"})["id"]
        other_message = source.new_message(other_chat, None, "user", None, content="other")
        source._save_message(other_message)
        source.db.commit()
        cases = ("self-cycle", "inactive-cycle", "two-message-cycle", "missing-parent", "cross-chat-parent",
                 "parent-mismatch", "missing-active-leaf", "cross-chat-active-leaf", "invalid-active-leaf")
        try:
            for name in cases:
                with self.subTest(name=name):
                    destination, keep_chat, keep_run = self.populated(self.path("graph-destination-" + name), "keep me")
                    try:
                        baseline = destination.snapshot(keep_chat)
                        baseline_events = destination.events(keep_run, 0)
                        candidate_path = self.path("graph-" + name)
                        source.backup(candidate_path)
                        connection = sqlite3.connect(candidate_path)
                        try:
                            rows = connection.execute("SELECT id,chat,parent,data FROM messages ORDER BY rowid").fetchall()
                            user = next(row for row in rows if json.loads(row[3])["role"] == "user")
                            answer = next(row for row in rows if json.loads(row[3])["role"] == "assistant")
                            data = json.loads(answer[3])
                            if name in ("self-cycle", "inactive-cycle"):
                                parent = answer[0]
                                data["parent_id"] = parent
                                connection.execute("UPDATE messages SET parent=?,data=? WHERE id=?", (parent, json.dumps(data), answer[0]))
                                if name == "inactive-cycle":
                                    connection.execute("UPDATE chats SET data=json_set(data,'$.active_leaf',NULL) WHERE id=?", (chat_id,))
                            elif name == "two-message-cycle":
                                data["parent_id"] = user[0]
                                connection.execute("UPDATE messages SET parent=?,data=? WHERE id=?", (user[0], json.dumps(data), answer[0]))
                                user_data = json.loads(user[3])
                                user_data["parent_id"] = answer[0]
                                connection.execute("UPDATE messages SET parent=?,data=? WHERE id=?", (answer[0], json.dumps(user_data), user[0]))
                            elif name == "missing-parent":
                                parent = "missing-parent-id"
                                data["parent_id"] = parent
                                connection.execute("UPDATE messages SET parent=?,data=? WHERE id=?", (parent, json.dumps(data), answer[0]))
                            elif name in ("cross-chat-parent", "parent-mismatch"):
                                parent = other_message["id"] if name == "cross-chat-parent" else user[0]
                                data["parent_id"] = parent
                                sql_parent = None if name == "parent-mismatch" else parent
                                connection.execute("UPDATE messages SET parent=?,data=? WHERE id=?", (sql_parent, json.dumps(data), answer[0]))
                            else:
                                chat = json.loads(connection.execute("SELECT data FROM chats WHERE id=?", (chat_id,)).fetchone()[0])
                                chat["active_leaf"] = {"missing-active-leaf": "missing-leaf",
                                                       "cross-chat-active-leaf": other_message["id"],
                                                       "invalid-active-leaf": []}[name]
                                connection.execute("UPDATE chats SET data=? WHERE id=?", (json.dumps(chat), chat_id))
                            connection.commit()
                        finally:
                            connection.close()
                        with self.assertRaises(ValueError):
                            destination.restore(candidate_path)
                        self.assertEqual(destination.snapshot(keep_chat), baseline)
                        self.assertEqual(destination.events(keep_run, 0), baseline_events)
                    finally:
                        destination.close()
        finally:
            source.close()

    def test_restore_accepts_alternative_branches_and_child_before_parent_rows(self):
        source_path = self.path("branch-source")
        source, chat_id, _ = self.populated(source_path, "branch source")
        parent = source.new_message(chat_id, None, "user", None, content="alternate")
        child = source.new_message(chat_id, parent["id"], "assistant", None, content="alternate answer", status="completed")
        source._save_message(child)
        source._save_message(parent)
        chat = source.chat(chat_id)
        chat["active_leaf"] = child["id"]
        source.db.execute("UPDATE chats SET data=? WHERE id=?", (json.dumps(chat), chat_id))
        source.db.commit()
        expected = source.snapshot(chat_id)
        source.close()
        destination, _, _ = self.populated(self.path("branch-destination"), "discarded")
        try:
            destination.restore(source_path)
            self.assertEqual(destination.snapshot(chat_id), expected)
            self.assertEqual([message["content"] for message in destination.path(child["id"])],
                             ["alternate", "alternate answer"])
        finally:
            destination.close()

    def test_restore_refuses_active_runs_and_preserves_database(self):
        source, _, _ = self.populated(self.path("source"), "source")
        destination = Store(self.path("destination"))
        chat = destination.create_chat({"title": "active destination"})
        run = destination.start(chat["id"], {"content": "still running"}, self.settings())
        before = destination.snapshot(chat["id"])
        before_events = destination.events(run["id"], 0)
        try:
            with self.assertRaisesRegex(ValueError, "active replies"):
                destination.restore(self.path("source"))
            self.assertEqual(destination.snapshot(chat["id"]), before)
            self.assertEqual(destination.events(run["id"], 0), before_events)
        finally:
            source.close()
            destination.close()

    def test_restored_active_run_is_recovered_as_interrupted(self):
        source_path = self.path("source-active")
        source = Store(source_path)
        chat = source.create_chat({"title": "active source"})
        run = source.start(chat["id"], {"content": "pending"}, self.settings())
        message = source.message(run["message_id"])
        message["tool_calls"] = [{"id": "call-1", "name": "tool", "arguments": "{}", "status": "approval"}]
        source.publish(run["id"], "message", message)
        source.publish(run["id"], "state", dict(source.run(run["id"]), status="approval"))
        destination, _, _ = self.populated(self.path("destination"), "old destination")
        try:
            destination.restore(source_path)
            self.assertEqual(destination.run(run["id"])["status"], "interrupted")
            recovered = destination.message(message["id"])
            self.assertEqual(recovered["status"], "interrupted")
            self.assertEqual(recovered["tool_calls"][0]["status"], "interrupted")
        finally:
            source.close()
            destination.close()

    def test_restore_requires_file_backed_store_and_distinct_source(self):
        memory = Store(":memory:")
        try:
            with self.assertRaisesRegex(ValueError, "file-backed"):
                memory.restore(self.path("backup"))
        finally:
            memory.close()
        path = self.path("same")
        store, _, _ = self.populated(path, "same")
        try:
            with self.assertRaisesRegex(ValueError, "differ"):
                store.restore(path)
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
