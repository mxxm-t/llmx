"""Bounded memory-only conversations using the regular store and engine."""

import threading
import time

from engine import Engine
from store import Store


class DisabledTools:
    def list_tools(self):
        return []

    def call_tool(self, _name, _arguments, _cancel=None):
        raise ValueError("Tools are unavailable in temporary chats")


class TemporaryChats:
    TTL_SECONDS = 3600
    MAX_CHATS = 16

    def __init__(self, upstream, key=None):
        self.store = Store(":memory:", id_prefix="tmp_")
        self.engine = Engine(self.store, upstream, DisabledTools(), key)
        self.lock = threading.RLock()
        self.accessed = {}
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._reap, name="temporary-chats", daemon=True)
        self.thread.start()

    @staticmethod
    def check_settings(settings):
        if settings.get("tools"):
            raise ValueError("Tools are unavailable in temporary chats")

    def create(self, values):
        self.check_settings(values.get("settings", {}))
        with self.lock:
            self.expire()
            if self.stopped.is_set():
                raise ValueError("The chat service is stopping")
            if len(self.accessed) >= self.MAX_CHATS:
                raise ValueError("Sixteen temporary chats are already open. Close one before starting another.")
            chat = self.store.create_chat(values)
            chat = self.store.update_chat(chat["id"], {"temporary": True})
            self.accessed[chat["id"]] = time.monotonic()
            return chat

    def access(self, key, run=False):
        with self.lock:
            self.expire()
            value = self.store.run(key) if run else self.store.chat(key)
            self.accessed[value["chat_id"] if run else key] = time.monotonic()
            return self.store, self.engine

    def delete(self, key):
        with self.lock:
            self.store.delete_chat(key)
            self.accessed.pop(key, None)

    def expire(self):
        with self.lock:
            now = time.monotonic()
            for key, accessed in list(self.accessed.items()):
                with self.store.lock:
                    try:
                        updated = self.store._idle(key)
                    except ValueError:
                        self.accessed[key] = now
                        continue
                    if now - accessed >= self.TTL_SECONDS and time.time() - updated >= self.TTL_SECONDS:
                        self.delete(key)

    def _reap(self):
        while not self.stopped.wait(60):
            self.expire()

    def close(self):
        with self.lock:
            if self.stopped.is_set():
                return
            self.stopped.set()
        self.thread.join()
        self.engine.shutdown()
        with self.engine.lock:
            threads = list(self.engine.threads)
        for thread in threads:
            thread.join()
        with self.lock:
            self.accessed.clear()
            self.store.close()
