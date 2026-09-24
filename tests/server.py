import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baseline
import common
import f32
import moe
from common import run as cli

# The server of docs/SERVER.md against the CLI on the same file: a greedy request through /v1/generate gives the text `generate --temp 0` gives, alone and while three other requests decode beside it; a streamed request arrives as events with the same ids; a seeded request repeats; a bad body and a request past the context are refused; a client that goes away mid-stream leaves the server with nothing active; a chat turn renders.
# The synthetic F32 model (16-token context) needs no download; the real Q8_0 fixture, when it is on disk, repeats the checks with room to stream.


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class Server:
    def __init__(self, model, *extra):
        self.port = free_port()
        # The device and cache flags go on the command, not the executable path, which device_args would not recognise; the server then runs where the CLI it is compared with runs.
        args = ["serve", model, "--host", "127.0.0.1", "--port", str(self.port), "--max-seqs", "8"] + list(extra)
        # The server logs a line per request on stderr, so it goes to a file: a pipe nobody reads would fill and block the server.
        self.log = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
        self.proc = subprocess.Popen([common.exe_path()] + common.device_args(args, "f32"), stdout=subprocess.DEVNULL, stderr=self.log,
                                     text=True, encoding="utf-8")
        deadline = time.time() + 120
        while time.time() < deadline:
            if self.proc.poll() is not None:
                self.log.seek(0)
                raise AssertionError("server exited early: " + self.log.read())
            try:
                self.get("/v1/health")
                return
            except (urllib.error.URLError, ConnectionError, OSError):
                time.sleep(0.1)
        raise AssertionError("server did not come up")

    def get(self, path):
        with urllib.request.urlopen("http://127.0.0.1:%d%s" % (self.port, path), timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))

    def post(self, path, body, timeout=300):
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request("http://127.0.0.1:%d%s" % (self.port, path), data=data,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode("utf-8"))

    def stream(self, path, body):
        """Every SSE data payload, parsed, in order; the end marker is None."""
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request("http://127.0.0.1:%d%s" % (self.port, path), data=data,
                                     headers={"Content-Type": "application/json"})
        events = []
        with urllib.request.urlopen(req, timeout=300) as r:
            for raw in r:
                line = raw.decode("utf-8").rstrip("\n")
                if line.startswith("data: "):
                    payload = line[6:]
                    events.append(None if payload == "[DONE]" else json.loads(payload))
        return events

    def close(self):
        self.proc.kill()
        self.proc.wait()
        self.log.close()


def cli_greedy_text(model, prompt, n, flags=()):
    """What `generate --temp 0` prints between its pp and tg lines, with the f32 cache sides and the flags the server under test is started with."""
    rc, out = cli(["generate", model, prompt, "-n", str(n), "--temp", "0"] + list(flags), cache="f32")
    assert rc == 0, out
    lines = out.split("\n")
    assert lines[0].startswith("pp:") and lines[-2].startswith("tg:"), out
    return "\n".join(lines[1:-2])


def check_server(model, prompts, n, long_n, chat, prefix=None, flags=()):
    srv = Server(model, *flags)
    try:
        health = srv.get("/v1/health")
        assert health["status"] == "ok" and health["active"] == 0, health
        models = srv.get("/v1/models")
        assert models["object"] == "list" and models["data"][0]["object"] == "model", models
        assert models["data"][0]["id"] and models["data"][0]["context_length"] > 0, models

        # Greedy through the server gives the CLI's text, and the ids are kept for the checks that follow.
        expected = {}
        for prompt in prompts:
            want = cli_greedy_text(model, prompt, n, flags)
            status, reply = srv.post("/v1/generate", {"prompt": prompt, "max_tokens": n, "temperature": 0})
            assert status == 200, reply
            assert reply["text"] == want, (prompt, reply["text"], want)
            assert reply["finish"] in ("eos", "length", "stop"), reply
            expected[prompt] = reply["ids"]

        # The same requests four at a time give the same ids each.
        results = {}
        def worker(p):
            results[p] = srv.post("/v1/generate", {"prompt": p, "max_tokens": n, "temperature": 0})
        threads = [threading.Thread(target=worker, args=(p,)) for p in prompts[:4]]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for p in prompts[:4]:
            status, reply = results[p]
            assert status == 200 and reply["ids"] == expected[p], (p, reply, expected[p])

        # A stream: token events, then the end marker, with the same ids.
        events = srv.stream("/v1/generate", {"prompt": prompts[0], "max_tokens": n, "temperature": 0, "stream": True})
        ids = [e["id"] for e in events if e and "id" in e]
        assert ids == expected[prompts[0]], (ids, expected[prompts[0]])
        assert events[-1] is None and events[-2].get("done") is True, events[-2:]

        # A seeded sampled request is reproducible.
        a = srv.post("/v1/generate", {"prompt": prompts[0], "max_tokens": n, "temperature": 1.0, "seed": 7})[1]
        b = srv.post("/v1/generate", {"prompt": prompts[0], "max_tokens": n, "temperature": 1.0, "seed": 7})[1]
        assert a["ids"] == b["ids"], (a, b)

        # Refusals: a bad body, and a request past the context.
        assert srv.post("/v1/generate", {"prompt": ""})[0] == 400
        assert srv.post("/v1/generate", {"prompt": "a", "max_tokens": 10 ** 9})[0] == 413

        # A client that leaves mid-stream: open the socket, start a request, close after the first bytes, and the server ends with nothing active.
        s = socket.create_connection(("127.0.0.1", srv.port))
        body = json.dumps({"prompt": prompts[0], "max_tokens": long_n, "temperature": 0, "stream": True}).encode()
        s.sendall(b"POST /v1/generate HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n" % len(body) + body)
        s.recv(64)
        s.close()
        deadline = time.time() + 60
        while time.time() < deadline and srv.get("/v1/health")["active"] != 0:
            time.sleep(0.2)
        assert srv.get("/v1/health")["active"] == 0, "a cancelled request stayed active"

        # /v1/chat renders through the template and answers; the synthetic model's 16-token context has no room for a rendered turn.
        if chat:
            status, reply = srv.post("/v1/chat", {"messages": [{"role": "user", "content": prompts[0]}],
                                                  "max_tokens": 2, "temperature": 0})
            assert status == 200 and reply["tokens"] >= 1, reply

        # The compatible routes: /v1/completions gives the native route's greedy text in the standard shape, whole and streamed with the finish chunk then the end marker; /v1/chat/completions renders the same template as /v1/chat, its first chunk carries the role and its usage counts add up; the standard refusals have the standard shape.
        # An absent max_tokens means no cap, checked on the synthetic model only, since the real model's uncapped reply would run long.
        if not chat:
            status, reply = srv.post("/v1/completions", {"prompt": prompts[0], "temperature": 0})
            limit = models["data"][0]["context_length"]
            assert status == 200 and reply["choices"][0]["finish_reason"] in ("stop", "length"), reply
            assert reply["usage"]["completion_tokens"] >= 1 and reply["usage"]["total_tokens"] <= limit, reply
        status, reply = srv.post("/v1/completions", {"prompt": prompts[0], "max_tokens": n, "temperature": 0})
        assert status == 200 and reply["object"] == "text_completion", reply
        assert reply["choices"][0]["text"] == cli_greedy_text(model, prompts[0], n, flags), reply
        assert reply["choices"][0]["finish_reason"] in ("stop", "length"), reply
        assert reply["usage"]["total_tokens"] == reply["usage"]["prompt_tokens"] + reply["usage"]["completion_tokens"], reply
        # The timings a client shows speed from: prompt tokens prefilled and reused, and generation after the first token.
        t = reply["timings"]
        assert t["prompt_n"] + t["cache_n"] == reply["usage"]["prompt_tokens"], reply
        assert t["predicted_n"] == max(reply["usage"]["completion_tokens"] - 1, 0) and t["prompt_ms"] >= 0, reply
        events = srv.stream("/v1/completions", {"prompt": prompts[0], "max_tokens": n, "temperature": 0, "stream": True,
                                                "stream_options": {"include_usage": True}})
        assert events[-1] is None and events[-2]["usage"]["completion_tokens"] == len(expected[prompts[0]]), events[-3:]
        assert events[-3]["choices"][0]["finish_reason"] in ("stop", "length") and "timings" in events[-3], events[-3]
        assert "".join(e["choices"][0]["text"] for e in events[:-2]) == reply["choices"][0]["text"], events
        status, err = srv.post("/v1/completions", {"prompt": prompts[0], "n": 2})
        assert status == 400 and err["error"]["message"], err
        if chat:
            status, reply = srv.post("/v1/chat/completions",
                                     {"messages": [{"role": "user", "content": [{"type": "text", "text": prompts[0]}]}],
                                      "max_completion_tokens": 2, "temperature": 0})
            assert status == 200 and reply["object"] == "chat.completion", reply
            assert reply["choices"][0]["message"]["role"] == "assistant" and reply["usage"]["completion_tokens"] >= 1, reply
            events = srv.stream("/v1/chat/completions", {"messages": [{"role": "user", "content": prompts[0]}],
                                                         "max_tokens": 2, "temperature": 0, "stream": True})
            assert events[0]["choices"][0]["delta"]["role"] == "assistant", events[0]
            assert events[-1] is None and events[-2]["choices"][0]["finish_reason"] in ("stop", "length"), events[-2:]

        # Prefix reuse: a long prompt, then the same prompt with a different ending; the second forks the first's full blocks, prefills only what follows, and its greedy text equals the CLI's for the whole.
        if prefix:
            first = prefix + " The first"
            second = prefix + " The second"
            status, a = srv.post("/v1/generate", {"prompt": first, "max_tokens": n, "temperature": 0})
            assert status == 200 and a["reused_tokens"] == 0, a
            status, b = srv.post("/v1/generate", {"prompt": second, "max_tokens": n, "temperature": 0})
            assert status == 200 and b["reused_tokens"] > 0, b
            assert b["text"] == cli_greedy_text(model, second, n, flags), (b["text"],)
            health = srv.get("/v1/health")
            assert health["prefix_hits"] >= 1 and health["donors"] >= 1, health
        return len(prompts)
    finally:
        srv.close()


# The ids each prompt gets alone, then four at a time, where a pass holds one request's prompt rows beside another's decode rows.
# Ids rather than text: a synthetic model's greedy bytes need not be UTF-8.
def check_mixed(model, prompts, n, flags):
    srv = Server(model, *flags)
    try:
        alone = {}
        for p in prompts:
            status, reply = srv.post("/v1/generate", {"prompt": p, "max_tokens": n, "temperature": 0})
            assert status == 200, reply
            alone[p] = reply["ids"]
        for group in (prompts[:4], prompts[2:]):
            results = {}
            def worker(p):
                results[p] = srv.post("/v1/generate", {"prompt": p, "max_tokens": n, "temperature": 0})
            threads = [threading.Thread(target=worker, args=(p,)) for p in group]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            for p in group:
                status, reply = results[p]
                assert status == 200 and reply["ids"] == alone[p], (p, reply, alone[p])
        return len(prompts)
    finally:
        srv.close()


def check_limits(model):
    """The serving limits: a KV budget below the context bounds a request,
    and a full queue refuses with 503 rather than waiting."""
    srv = Server(model, "--max-seqs", "1", "--max-queue", "1", "--ctx-size", "512")
    try:
        assert srv.post("/v1/generate", {"prompt": "a", "max_tokens": 600})[0] == 413
        results = {}
        def worker(i):
            results[i] = srv.post("/v1/generate", {"prompt": "The capital of France is", "max_tokens": 300,
                                                   "temperature": 0})
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
            time.sleep(0.2)
        for t in threads:
            t.join()
        codes = sorted(status for status, _ in results.values())
        assert codes == [200, 200, 503], codes
        assert [r for s, r in results.values() if s == 503][0]["error"], results
    finally:
        srv.close()


def check_uncapped(model):
    """Uncapped requests share a small pool: each reserves its prompt and grows, a request is paused when the pool runs out and resumes from its history, and every one runs to its own end."""
    srv = Server(model, "--max-seqs", "4", "--ctx-size", "1024")
    try:
        results = {}
        prompts = ["The capital of France is", "Once upon a time", "def fib(n):"]
        def worker(p):
            results[p] = srv.post("/v1/completions", {"prompt": p, "temperature": 0}, timeout=600)
        threads = [threading.Thread(target=worker, args=(p,)) for p in prompts]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for p in prompts:
            status, reply = results[p]
            assert status == 200 and reply["choices"][0]["finish_reason"] in ("stop", "length"), (p, reply)
            assert reply["usage"]["total_tokens"] <= 1024, (p, reply)
        health = srv.get("/v1/health")
        assert health["active"] == 0 and health["pauses"] >= 1, health
    finally:
        srv.close()


def run():
    if os.environ.get("LLMX_CACHE_TYPE", "f32") != "f32":
        print("server: SKIP - the greedy comparison with the CLI is made with f32 caches (LLMX_CACHE_TYPE=%s)"
              % os.environ["LLMX_CACHE_TYPE"])
        return True
    with tempfile.TemporaryDirectory(prefix="llmx_server_") as directory:
        model = os.path.join(directory, "tiny-f32.gguf")
        f32.write_model(model, f32.tensors(True))
        # The synthetic model's context is 16 tokens: prompt plus tokens stay inside it.
        n = check_server(model, ["a", "ab", "abc", "abcdefg"], 6, 14, chat=False)
        print("server: synthetic F32 model, %d prompts greedy-equal to the CLI alone and four at a time, a stream, "
              "a seeded repeat, refusals, a cancelled stream, the compatible completions  [ok]" % n)
        # On a device, the synthetic mixture of experts with its routed layers on the host and prompts from extent 3 streamed: four at a time, a pass holds streamed prompt rows beside host decode rows.
        # Experts on the host are a placement of one device, so a list of several skips this.
        if os.environ.get("LLMX_DEVICE", "cpu") != "cpu" and "," not in os.environ.get("LLMX_DEVICE", ""):
            routed = os.path.join(directory, "tiny-moe.gguf")
            f32.write_model(routed, moe.tensors(), config=moe.CONFIG, arch="qwen3moe")
            n = check_mixed(routed, ["a", "ab", "abc", "abcdefg", "abcd", "b"], 6, ("--cpu-moe", "--moe-stream-from", "3"))
            print("server: synthetic MoE model, experts on the host and long prompts streamed, %d prompts alone and four at a time  [ok]" % n)
    real = baseline.find_fixture(baseline.BASELINE_MODELS[0])
    if real:
        with open(os.path.join(os.path.dirname(__file__), "data", "baseline_perplexity.json"), encoding="utf-8") as f:
            excerpt = json.load(f)["text"]
        n = check_server(real, ["The capital of France is", "Once upon a time", "def fib(n):", "The three laws of"],
                         16, 4000, chat=True, prefix=excerpt)
        check_limits(real)
        check_uncapped(real)
        print("server: %s, %d prompts greedy-equal to the CLI alone and four at a time, a stream, a seeded repeat, "
              "refusals, a cancelled stream, a chat turn, the compatible routes, a reused prefix, the limits, uncapped requests sharing a pool  [ok]"
              % (os.path.basename(real), n))
    else:
        print("server: SKIP real-model pass - fixture model not on disk")
    return True
