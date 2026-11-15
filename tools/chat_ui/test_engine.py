import json
import os
import tempfile
import threading
import time
import unittest
from unittest import mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from engine import Engine
from store import Store


class FakeUpstream(BaseHTTPRequestHandler):
    lock = threading.Lock()
    replies = []
    requests = []
    request_seen = threading.Event()
    stream_opened = threading.Event()
    gate = None

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        size = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(size))
        with type(self).lock:
            type(self).requests.append(payload)
            if not type(self).replies:
                self.send_error(500)
                return
            reply = type(self).replies.pop(0)
        type(self).request_seen.set()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        try:
            for event in reply:
                if event == "WAIT":
                    type(self).stream_opened.set()
                    gate = type(self).gate
                    if gate is not None:
                        gate.wait(5)
                    continue
                if event == "TRUNCATE":
                    continue
                self.wfile.write(b"data: " + json.dumps(event).encode("utf-8") + b"\n\n")
                self.wfile.flush()
            if reply[-1] != "TRUNCATE":
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass


def text_event(content=None, reasoning=None, usage=None):
    delta = {}
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    event = {"choices": [{"delta": delta, "finish_reason": None}]}
    if usage is not None:
        event["usage"] = usage
    return event


def tool_event(call_id="call-1", name="echo", arguments='{"text":"hello"}'):
    return {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": call_id,
                "type": "function", "function": {"name": name, "arguments": arguments}}]},
                "finish_reason": "tool_calls"}]}


class FakeTools:
    def __init__(self, approval=True):
        self.calls = []
        self.lock = threading.Lock()
        self.approval = approval

    def list_tools(self):
        return [{"name": "echo", "description": "Echo text", "inputSchema": {"type": "object"},
                 "server": "fake", "approval": self.approval}]

    def call_tool(self, name, arguments, cancel_event=None):
        with self.lock:
            self.calls.append((name, arguments))
        return {"content": [{"type": "text", "text": arguments.get("text", "")}], "isError": False}


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeUpstream)
        cls.server_thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.server_thread.start()
        cls.url = "http://127.0.0.1:%d" % cls.httpd.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.server_thread.join()
        cls.httpd.server_close()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp.name, "chat.sqlite")
        self.store = Store(self.db_path)
        with FakeUpstream.lock:
            FakeUpstream.replies = []
            FakeUpstream.requests = []
        FakeUpstream.request_seen = threading.Event()
        FakeUpstream.stream_opened = threading.Event()
        FakeUpstream.gate = None
        self.tools = FakeTools()
        self.engine = Engine(self.store, self.url, self.tools)
        self.chat = self.store.create_chat({})

    def tearDown(self):
        self.engine.shutdown()
        if self.store is not None:
            self.store.close()
        self.temp.cleanup()

    def test_rejected_image_turn_allows_text_followup(self):
        self.reply(text_event("Earlier answer"))
        earlier = self.start({"content": "Earlier question"})
        self.wait_run(earlier["id"], {"completed"})
        image = {"name": "picture.png", "data": "data:image/png;base64,aGVsbG8="}
        rejected = self.store.start(self.chat["id"], {"content": "Describe this", "attachments": [image]}, self.settings())
        assistant = self.store.message(rejected["message_id"])
        assistant["status"] = "failed"
        self.store.publish(rejected["id"], "message", assistant)
        self.store.publish(rejected["id"], "state", {"status": "failed", "error": "llmx returned HTTP 400: the chat template refused the messages: can only concatenate str (not list) to str"})
        image_user = self.store.message(assistant["parent_id"])
        self.reply(text_event("Text works"))
        with mock.patch("engine.Connection.json", side_effect=ValueError("Images are unsupported")) as probe:
            run = self.start({"content": "Continue with text"})
            self.assertEqual(self.wait_run(run["id"], {"completed", "failed"})["status"], "completed")
            probe.assert_not_called()
            with self.assertRaisesRegex(ValueError, "images"):
                self.engine.start(self.chat["id"], {"regenerate": True, "parent_id": image_user["id"]}, self.settings())
        self.assertEqual(FakeUpstream.requests[-1]["messages"], [
            {"role": "user", "content": "Earlier question"},
            {"role": "assistant", "content": "Earlier answer"},
            {"role": "user", "content": "Continue with text"}])
        self.assertEqual(self.store.message(image_user["id"]), image_user)
        self.assertEqual(self.store.message(assistant["id"]), assistant)

    def test_image_rejection_recovery_preserves_other_failures_and_output(self):
        image = {"name": "picture.png", "data": "data:image/png;base64,aGVsbG8="}
        refusal = "llmx returned HTTP 400: the chat template refused the messages"
        cases = [("partial answer", {"content": "A visible answer"}, refusal),
                 ("reasoning", {"reasoning": "A visible thought"}, refusal),
                 ("tool action", {"tool_calls": [{"id": "action", "name": "echo", "arguments": "{}"}]}, refusal),
                 ("other failure", {}, "The upstream stopped responding")]
        for label, output, error in cases:
            with self.subTest(label=label):
                chat = self.store.create_chat({})
                run = self.store.start(chat["id"], {"content": "Image", "attachments": [image]}, self.settings())
                assistant = self.store.message(run["message_id"])
                assistant.update(status="failed", **output)
                self.store.publish(run["id"], "message", assistant)
                self.store.publish(run["id"], "state", {"status": "failed", "error": error})
                before = self.store.snapshot(chat["id"])
                with mock.patch("engine.Connection.json", side_effect=ValueError("Images unsupported")) as probe:
                    with self.assertRaisesRegex(ValueError, "images"):
                        self.engine.start(chat["id"], {"content": "Follow up"}, self.settings())
                    probe.assert_called_once()
                self.assertEqual(self.store.snapshot(chat["id"]), before)

    def test_image_rejection_keeps_conversation_unchanged(self):
        body = {"content": "Describe this", "attachments": [{"name": "picture.png", "data": "data:image/png;base64,aGVsbG8="}]}
        before = self.store.snapshot(self.chat["id"])
        with mock.patch("engine.Connection.json", side_effect=ValueError("chat template refused the messages")), \
                mock.patch("engine.threading.Thread") as start:
            with self.assertRaisesRegex(ValueError, "images"):
                self.engine.start(self.chat["id"], body, self.settings())
            start.assert_not_called()
        self.assertEqual(before, self.store.snapshot(self.chat["id"]))
        self.assertEqual(body["content"], "Describe this")
        self.assertEqual(len(body["attachments"]), 1)

    def test_image_admission_includes_followup_and_retry_history(self):
        body = {"content": "Describe this", "attachments": [{"name": "picture.png", "data": "data:image/png;base64,aGVsbG8="}]}
        self.reply(text_event("A picture"))
        with mock.patch("engine.Connection.json", return_value={"tokens": [1], "count": 1}) as probe:
            run = self.engine.start(self.chat["id"], body, self.settings(["echo"]))
            self.wait_run(run["id"], {"completed"})
        args = probe.call_args.args
        self.assertEqual(args[0], "/tokenize")
        self.assertEqual(args[1], FakeUpstream.requests[0])
        self.assertEqual(args[1]["messages"][0]["content"][-1]["image_url"]["url"], body["attachments"][0]["data"])
        snapshot = self.store.snapshot(self.chat["id"])
        user = next(m for m in snapshot["messages"] if m["role"] == "user")
        for next_body in ({"content": "And its colour?"}, {"regenerate": True, "parent_id": user["id"]}):
            with self.subTest(body=next_body), mock.patch("engine.Connection.json", side_effect=ValueError("no images")):
                with self.assertRaisesRegex(ValueError, "images"):
                    self.engine.start(self.chat["id"], next_body, self.settings())
                self.assertEqual(snapshot, self.store.snapshot(self.chat["id"]))

    def test_image_preflight_connection_failure_keeps_draft(self):
        before = self.store.snapshot(self.chat["id"])
        with mock.patch("engine.Connection.json", side_effect=TimeoutError("timeout")):
            with self.assertRaisesRegex(ValueError, "Try again"):
                self.engine.start(self.chat["id"], {"attachments": [{"name": "picture.png", "data": "data:image/png;base64,aGVsbG8="}]}, self.settings())
        self.assertEqual(before, self.store.snapshot(self.chat["id"]))

    def test_text_attachment_does_not_probe_images(self):
        self.reply(text_event("Read it"))
        with mock.patch("engine.Connection.json") as probe:
            run = self.engine.start(self.chat["id"], {"content": "Read", "attachments": [{"name": "notes.txt", "text": "hello"}]}, self.settings())
            self.wait_run(run["id"], {"completed"})
            probe.assert_not_called()
        self.assertIn("hello", FakeUpstream.requests[0]["messages"][0]["content"][1]["text"])

    def settings(self, tools=None):
        return {"model": "fake-model", "max_tokens": 32, "tools": tools or []}

    def reply(self, *events):
        with FakeUpstream.lock:
            FakeUpstream.replies.append(list(events))

    def wait_run(self, rid, statuses, timeout=3):
        end = time.monotonic() + timeout
        with self.store.changed:
            while True:
                run = self.store.run(rid)
                if run["status"] in statuses:
                    return run
                remaining = end - time.monotonic()
                if remaining <= 0:
                    self.fail("run did not reach %r; current state is %r" % (statuses, run["status"]))
                self.store.changed.wait(remaining)

    def wait_approval(self, rid, timeout=3):
        return self.wait_run(rid, {"approval"}, timeout)

    def wait_content(self, message_id, content, timeout=3):
        end = time.monotonic() + timeout
        with self.store.changed:
            while True:
                message = self.store.message(message_id)
                if message["content"] == content:
                    return message
                remaining = end - time.monotonic()
                if remaining <= 0:
                    self.fail("message did not reach expected content")
                self.store.changed.wait(remaining)

    def snapshot(self):
        return self.store.snapshot(self.chat["id"])

    def start(self, body=None, settings=None):
        return self.engine.start(self.chat["id"], body or {"content": "question"},
                                 settings or self.settings())

    def test_stream_accumulation_and_final_usage(self):
        self.reply(text_event("Hello ", "thinking "), text_event("world", "done", {
            "prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}))
        run = self.start()
        self.assertEqual(self.wait_run(run["id"], {"completed", "failed"})["status"], "completed")
        snapshot = self.snapshot()
        answer = next(m for m in snapshot["messages"] if m["role"] == "assistant")
        self.assertEqual(answer["content"], "Hello world")
        self.assertEqual(answer["reasoning"], "thinking done")
        self.assertEqual(answer["usage"]["total_tokens"], 9)
        self.assertEqual(len(FakeUpstream.requests), 1)

    def test_approved_tool_runs_once_then_model_finishes(self):
        self.reply(tool_event())
        self.reply(text_event("Finished"))
        run = self.start({"content": "use echo"}, self.settings(["echo"]))
        self.wait_approval(run["id"])
        call = next(call for message in self.snapshot()["messages"]
                    for call in message.get("tool_calls", []) if call["status"] == "approval")
        self.engine.approve(run["id"], call["id"], True, False)
        self.assertEqual(self.wait_run(run["id"], {"completed", "failed"})["status"], "completed")
        self.assertEqual(self.tools.calls, [("echo", {"text": "hello"})])
        self.assertEqual(len(FakeUpstream.requests), 2)
        second_history = FakeUpstream.requests[1]["messages"]
        self.assertEqual(second_history[-1]["role"], "tool")
        self.assertIn("hello", second_history[-1]["content"])

    def test_denial_is_recorded_without_tool_execution(self):
        self.reply(tool_event())
        self.reply(text_event("Declined"))
        run = self.start({"content": "use echo"}, self.settings(["echo"]))
        self.wait_approval(run["id"])
        call = next(call for message in self.snapshot()["messages"]
                    for call in message.get("tool_calls", []) if call["status"] == "approval")
        self.engine.approve(run["id"], call["id"], False, False)
        self.assertEqual(self.wait_run(run["id"], {"completed", "failed"})["status"], "completed")
        self.assertEqual(self.tools.calls, [])
        record = next(call for message in self.snapshot()["messages"]
                      for call in message.get("tool_calls", []) if call["status"] == "denied")
        self.assertEqual(record["status"], "denied")

    def test_event_replay_does_not_start_another_generation(self):
        gate = threading.Event()
        FakeUpstream.gate = gate
        self.reply(text_event("partial"), "WAIT", text_event("done"))
        run = self.start()
        self.assertTrue(FakeUpstream.stream_opened.wait(2))
        self.wait_content(run["message_id"], "partial")
        first = self.store.events(run["id"], 0)
        self.assertTrue(first)
        replay = self.store.events(run["id"], 0)
        self.assertEqual(replay, first)
        self.assertEqual(len(FakeUpstream.requests), 1)
        gate.set()
        self.assertEqual(self.wait_run(run["id"], {"completed", "failed"})["status"], "completed")
        final_events = self.store.events(run["id"], 0)
        self.assertEqual(final_events[:len(first)], first)
        self.assertEqual(len(FakeUpstream.requests), 1)

    def test_regeneration_adds_sibling_and_preserves_old_reply(self):
        self.reply(text_event("old answer"))
        self.reply(text_event("new answer"))
        first = self.start({"content": "same question"})
        self.assertEqual(self.wait_run(first["id"], {"completed", "failed"})["status"], "completed")
        original = next(m for m in self.snapshot()["messages"] if m["role"] == "assistant")
        user = next(m for m in self.snapshot()["messages"] if m["role"] == "user")
        second = self.start({"regenerate": True, "parent_id": user["id"]})
        self.assertEqual(self.wait_run(second["id"], {"completed", "failed"})["status"], "completed")
        assistants = [m for m in self.snapshot()["messages"] if m["role"] == "assistant"]
        self.assertEqual(len(assistants), 2)
        self.assertEqual(original["content"], "old answer")
        self.assertEqual([m["content"] for m in assistants], ["old answer", "new answer"])
        self.assertEqual({m["parent_id"] for m in assistants}, {user["id"]})

    def test_cancel_stops_a_blocked_upstream_stream(self):
        gate = threading.Event()
        FakeUpstream.gate = gate
        self.reply(text_event("started"), "WAIT", text_event("unreachable"))
        run = self.start()
        self.assertTrue(FakeUpstream.stream_opened.wait(2))
        self.engine.cancel(run["id"])
        interrupted = True
        try:
            status = self.wait_run(run["id"], {"cancelled", "failed"}, timeout=1)["status"]
        except AssertionError:
            interrupted = False
            gate.set()
            status = self.wait_run(run["id"], {"cancelled", "failed"}, timeout=3)["status"]
        self.assertEqual(status, "cancelled")
        self.assertTrue(interrupted, "cancel did not unblock the active upstream read")
        self.assertEqual(len(FakeUpstream.requests), 1)

    def test_truncated_upstream_stream_fails_run(self):
        self.reply(text_event("partial"), "TRUNCATE")
        run = self.start()
        current = self.wait_run(run["id"], {"failed", "completed"})
        self.assertEqual(current["status"], "failed")
        answer = next(m for m in self.snapshot()["messages"] if m["role"] == "assistant")
        self.assertEqual(answer["status"], "failed")
        self.assertEqual(answer["content"], "partial")

    def test_followup_after_cancelled_orphan_call_omits_tool_calls(self):
        settings = self.settings()
        cancelled = self.store.start(self.chat["id"], {"content": "old request"}, settings)
        answer = self.store.message(cancelled["message_id"])
        answer.update(status="cancelled", tool_calls=[{
            "id": "orphan-call", "name": "echo", "arguments": "{}", "status": "cancelled"}])
        self.store.publish(cancelled["id"], "message", answer)
        self.store.publish(cancelled["id"], "state", dict(self.store.run(cancelled["id"]), status="cancelled"))
        self.reply(text_event("new answer"))
        followup = self.start({"content": "continue"}, settings)
        self.assertEqual(self.wait_run(followup["id"], {"completed", "failed"})["status"], "completed")
        history = FakeUpstream.requests[0]["messages"]
        self.assertFalse(any("tool_calls" in message for message in history))

    def test_restart_marks_inflight_run_interrupted(self):
        run = self.store.start(self.chat["id"], {"content": "pending"}, self.settings())
        message = self.store.message(run["message_id"])
        self.store.close()
        reopened = Store(self.db_path)
        try:
            self.assertEqual(reopened.run(run["id"])["status"], "interrupted")
            self.assertEqual(reopened.message(message["id"])["status"], "interrupted")
            self.assertEqual(reopened.events(run["id"], 0)[-1]["kind"], "state")
        finally:
            reopened.close()
        self.store = None


if __name__ == "__main__":
    unittest.main()
