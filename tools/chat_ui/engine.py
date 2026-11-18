"""One owner for model/tool turns, approvals and cancellation."""

from concurrent.futures import ThreadPoolExecutor
import json
import http.client
import threading
import time

from store import Store, encoded, identity
from upstream import Cancelled, Connection


TERMINAL = {"completed", "cancelled", "failed", "interrupted"}


class Control:
    def __init__(self):
        self.cancel = threading.Event()
        self.changed = threading.Condition()
        self.pending = {}
        self.allowed = set()
        self.connection = None

    def stop(self):
        self.cancel.set()
        with self.changed:
            if self.connection:
                self.connection.cancel()
            self.changed.notify_all()


class Engine:
    def __init__(self, store, upstream, tools, key=None, max_rounds=16):
        self.store, self.upstream, self.tools, self.key = store, upstream, tools, key
        self.max_rounds = max_rounds
        self.lock = threading.RLock()
        self.controls = {}
        self.threads = set()

    def start(self, chat_id, body, settings):
        body = dict(body)
        with self.store.lock:
            chat = self.store.chat(chat_id)
            parent = body.setdefault("parent_id", chat["active_leaf"])
            if parent and self.store.message(parent)["chat_id"] != chat_id:
                raise ValueError("The parent belongs to another conversation.")
            messages = self._history(parent, settings.get("system", ""))
        if not body.get("regenerate"):
            messages.append({"role": "user", "content": self._content(body)})
        if any(isinstance(m["content"], list) and any(p.get("type") == "image_url" for p in m["content"]) for m in messages):
            try:
                available = self.tools.list_tools() if settings.get("tools") else []
                Connection(self.upstream, timeout=30, key=self.key).json("/tokenize", self._payload(messages, settings, available))
            except (OSError, http.client.HTTPException) as exc:
                raise ValueError("Could not reach the server to check this message's images. Try again. Your draft has been kept.") from exc
            except ValueError as exc:
                raise ValueError("The server could not process this message with images. Use an image-capable model and server, or remove the images. Your draft has been kept.") from exc
        with self.lock:
            if len(self.controls) >= 8:
                raise ValueError("Eight replies are already running. Wait for one to finish.")
            run = self.store.start(chat_id, body, settings)
            control = Control()
            self.controls[run["id"]] = control
            thread = threading.Thread(target=self._run, args=(run, control), daemon=True)
            self.threads.add(thread)
            thread.start()
            return run

    def cancel(self, rid):
        self.store.run(rid)
        with self.lock:
            control = self.controls.get(rid)
            if control:
                control.stop()

    def approve(self, rid, call_id, allow, remember):
        with self.lock:
            control = self.controls.get(rid)
        if not control:
            raise ValueError("This reply is no longer active")
        with control.changed:
            pending = control.pending.get(call_id)
            if not pending or pending["decision"] is not None:
                raise ValueError("This tool call is not awaiting approval")
            pending["decision"] = allow
            if allow and remember:
                control.allowed.add(pending["name"])
            control.changed.notify_all()

    def shutdown(self):
        with self.lock:
            for control in self.controls.values():
                control.stop()
            threads = list(self.threads)
        for thread in threads:
            thread.join(timeout=2)

    def _state(self, rid, status, **extra):
        self.store.publish(rid, "state", dict(self.store.run(rid), status=status, **extra))

    @staticmethod
    def _content(message):
        text = message.get("content", "")
        attachments = message.get("attachments", [])
        if not attachments:
            return text
        parts = [{"type": "text", "text": text}] if text else []
        for attachment in attachments:
            if "text" in attachment:
                parts.append({"type": "text", "text": f"\nFile: {attachment['name']}\n{attachment['text']}"})
            else:
                parts.append({"type": "image_url", "image_url": {"url": attachment["data"]}})
        return parts

    def _history(self, leaf, system):
        messages = [{"role": "system", "content": system}] if system else []
        chain = self.store.path(leaf)
        excluded = set()
        for user, reply in zip(chain, chain[1:]):
            if user["role"] != "user" or not any(a.get("data", "").startswith("data:image/") for a in user.get("attachments", [])):
                continue
            if reply["role"] != "assistant" or reply.get("status") != "failed" or any(reply.get(k) for k in ("content", "reasoning", "tool_calls")):
                continue
            if not reply.get("run_id"):
                continue
            run = self.store.run(reply["run_id"])
            if run["status"] == "failed" and "chat template refused the messages" in (run.get("error") or ""):
                excluded.update((user["id"], reply["id"]))
        chain = [m for m in chain if m["id"] not in excluded]
        resolved = {m["tool_call_id"] for m in chain if m["role"] == "tool"}
        for message in chain:
            role, text = message["role"], message["content"]
            entry = {"role": role, "content": text}
            if role == "user":
                entry["content"] = self._content(message)
            if role == "tool":
                entry["tool_call_id"] = message["tool_call_id"]
            if role == "assistant":
                if message.get("reasoning"):
                    entry["reasoning_content"] = message["reasoning"]
                calls = [c for c in message.get("tool_calls", []) if c["id"] in resolved]
                if calls:
                    entry["tool_calls"] = [{"id": c["id"], "type": "function", "function": {
                        "name": c["name"], "arguments": c["arguments"]}} for c in calls]
                if not calls and not text and not message.get("reasoning"):
                    continue
            messages.append(entry)
        return messages

    @staticmethod
    def _payload(messages, settings, available):
        payload = {k: settings[k] for k in ("model", "temperature", "top_p", "top_k", "max_tokens") if k in settings and settings[k] is not None}
        payload.update(messages=messages,
                       stream=True, stream_options={"include_usage": True})
        selected = [t for t in available if t["name"] in settings.get("tools", [])]
        if selected:
            payload["tools"] = [{"type": "function", "function": {"name": t["name"],
                                "description": t["description"], "parameters": t["inputSchema"]}} for t in selected]
        return payload

    def _model(self, rid, message, settings, available, control):
        payload = self._payload(self._history(message["parent_id"], settings.get("system", "")), settings, available)
        conn = Connection(self.upstream, timeout=120, key=self.key)
        with control.changed:
            control.connection = conn
        if control.cancel.is_set():
            conn.cancel()
            raise Cancelled()
        calls, fragments, chunks, thought = {}, {}, [], []
        last, chars, first = time.monotonic(), 0, True
        opening = True

        def flush():
            nonlocal last
            if chunks or thought or fragments:
                data = dict(message_id=message["id"], content="".join(chunks), reasoning="".join(thought))
                if fragments:
                    data["tool_calls"] = [fragments[k] for k in sorted(fragments)]
                self.store.publish(rid, "delta", data)
                chunks.clear()
                thought.clear()
                fragments.clear()
                last = time.monotonic()

        try:
            for event in conn.events(payload):
                if control.cancel.is_set():
                    raise Cancelled()
                metrics = {k: event[k] for k in ("usage", "timings", "context") if isinstance(event.get(k), dict)}
                if opening:
                    request_id = event.get("id")
                    if isinstance(request_id, str) and 1 <= len(request_id) <= 128 and all("!" <= c <= "~" for c in request_id):
                        metrics["request"] = {"id": request_id}
                    opening = False
                if metrics:
                    self.store.publish(rid, "metrics", dict(message_id=message["id"], **metrics))
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    content, reasoning = delta.get("content") or "", delta.get("reasoning_content") or ""
                    if not isinstance(content, str) or not isinstance(reasoning, str):
                        raise ValueError("Invalid streamed text")
                    if content:
                        chunks.append(content)
                    if reasoning:
                        thought.append(reasoning)
                    chars += len(content) + len(reasoning)
                    if chars > 16 * 1024 * 1024:
                        raise ValueError("The reply exceeded the 16 MiB safety limit")
                    for part in delta.get("tool_calls", []):
                        index = part.get("index", 0)
                        if type(index) is not int or not 0 <= index < 16:
                            raise ValueError("Too many tool calls in one model reply")
                        function = part.get("function", {})
                        name, arguments = function.get("name", ""), function.get("arguments", "")
                        if not isinstance(name, str) or not isinstance(arguments, str):
                            raise ValueError("Invalid streamed tool arguments")
                        call = calls.setdefault(index, dict(id=identity(), name_chars=0, argument_chars=0))
                        if call["argument_chars"] + len(arguments) > 1024 * 1024 or call["name_chars"] + len(name) > 256:
                            raise ValueError("Tool arguments are too large")
                        call["name_chars"] += len(name)
                        call["argument_chars"] += len(arguments)
                        fragment = fragments.setdefault(index, dict(id=call["id"], index=index, name="", arguments=""))
                        fragment["name"] += name
                        fragment["arguments"] += arguments
                    if choice.get("finish_reason"):
                        message["finish_reason"] = choice["finish_reason"]
                if (chunks or thought or fragments) and (first or time.monotonic() - last >= 0.05):
                    flush()
                    first = False
        finally:
            flush()
            with control.changed:
                control.connection = None
        latest = self.store.message(message["id"])
        latest["status"] = "completed"
        for call in latest["tool_calls"]:
            call["status"] = "pending"
        if "finish_reason" in message:
            latest["finish_reason"] = message["finish_reason"]
        self.store.publish(rid, "message", latest)
        return latest

    def _tool(self, rid, message_id, call, allowed, control):
        call = dict(call)

        def update(status, **values):
            call.update(status=status, **values)
            self.store.publish(rid, "tool", dict(message_id=message_id, call=dict(call)))

        started = time.monotonic()
        try:
            if call["name"] not in allowed:
                raise ValueError("The model requested a tool that is not enabled")
            args = json.loads(call["arguments"])
            if not isinstance(args, dict):
                raise ValueError("Tool arguments must be a JSON object")
            spec = allowed[call["name"]]
            with control.changed:
                if spec.get("approval", True) and call["name"] not in control.allowed:
                    control.pending[call["id"]] = dict(name=call["name"], decision=None)
                    update("approval")
                    self._state(rid, "approval")
                    deadline = time.monotonic() + 1800
                    while control.pending[call["id"]]["decision"] is None and not control.cancel.is_set():
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise ValueError("Tool approval expired after 30 minutes")
                        control.changed.wait(min(remaining, 10))
                    pending = control.pending.pop(call["id"])
                    if not any(v["decision"] is None for v in control.pending.values()):
                        self._state(rid, "running")
                    if control.cancel.is_set():
                        raise Cancelled()
                    if not pending["decision"]:
                        update("denied")
                        return {"isError": True, "content": [{"type": "text", "text": "The user declined this tool call."}]}
            if control.cancel.is_set():
                raise Cancelled()
            started = time.monotonic()
            update("running")
            result = self.tools.call_tool(call["name"], args, control.cancel)
            duration = (time.monotonic() - started) * 1000
            if control.cancel.is_set():
                update("cancelled", duration_ms=duration, result=result, error="Stopped waiting. An external action may already have occurred.")
                raise Cancelled()
            update("failed" if result.get("isError") else "completed", result=result, duration_ms=duration)
            return result
        except Cancelled:
            if call["status"] != "cancelled":
                update("cancelled", error="Cancelled. An external action may already have occurred.")
            return {"isError": True, "content": [{"type": "text", "text": "Cancelled"}]}
        except Exception as exc:
            update("failed", error=str(exc)[:1000], duration_ms=(time.monotonic() - started) * 1000)
            return {"isError": True, "content": [{"type": "text", "text": str(exc)[:1000]}]}
        finally:
            with control.changed:
                expired = control.pending.pop(call["id"], None)
                if expired and not control.cancel.is_set() and not any(v["decision"] is None for v in control.pending.values()):
                    self._state(rid, "running")

    def _run(self, run, control):
        rid, message_id = run["id"], run["message_id"]
        try:
            settings = run["settings"]
            available = self.tools.list_tools() if settings.get("tools") else []
            allowed = {t["name"]: t for t in available if t["name"] in settings.get("tools", [])}
            missing = set(settings.get("tools", [])) - set(allowed)
            if missing:
                raise ValueError("Some selected tools are unavailable. Update the enabled tools and retry.")
            for _ in range(self.max_rounds):
                message = self._model(rid, self.store.message(message_id), settings, available, control)
                calls = message["tool_calls"]
                if control.cancel.is_set():
                    raise Cancelled()
                if not calls:
                    self._state(rid, "completed")
                    return
                with ThreadPoolExecutor(max_workers=4, thread_name_prefix="chat-tool") as pool:
                    results = list(pool.map(lambda call: self._tool(rid, message_id, call, allowed, control), calls))
                parent = message_id
                for call, result in zip(calls, results):
                    content = encoded(result)
                    if len(content) > 65536:
                        content = encoded({"isError": result.get("isError", False), "truncated": True,
                                           "preview": content[:60000], "notice": "The complete tool result is available in the chat activity panel."})
                    tool_message = Store.new_message(run["chat_id"], parent, "tool", rid, content=content,
                                                     tool_call_id=call["id"], status="completed")
                    self.store.publish(rid, "message", tool_message)
                    parent = tool_message["id"]
                if control.cancel.is_set():
                    raise Cancelled()
                next_message = Store.new_message(run["chat_id"], parent, "assistant", rid)
                message_id = next_message["id"]
                self.store.publish(rid, "message", next_message)
                self._state(rid, "running", message_id=message_id)
            raise ValueError(f"Stopped after {self.max_rounds} model/tool rounds. Review the activity before continuing.")
        except Exception as exc:
            cancelled = control.cancel.is_set() or isinstance(exc, Cancelled)
            status = "cancelled" if cancelled else "failed"
            message = self.store.message(message_id)
            message["status"] = status
            for call in message.get("tool_calls", []):
                if call["status"] in ("preparing", "pending"):
                    call["status"] = status
            self.store.publish(rid, "message", message)
            self._state(rid, status, error=None if cancelled else str(exc)[:1000])
        finally:
            with self.lock:
                self.controls.pop(rid, None)
                self.threads.discard(threading.current_thread())
