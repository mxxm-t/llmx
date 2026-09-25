import json
import os
import socket
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

# The server of docs/SERVER.md against the CLI on the same file: a greedy request through /v1/generate gives the text `generate --temp 0` gives, alone and while three other requests decode beside it; a streamed request arrives as events with the same ids; a seeded request repeats, and a compatible request's seed of -1 samples as no seed; a bad body, a number its field cannot hold, a sampling field outside the CLI's range and a request past the context are refused; a client that goes away mid-stream, during a whole reply, while its prompt is read or while it waits in the queue leaves the server with nothing active and its blocks free, and one that shuts only its sending side gets no answer; a chat turn renders; a conversation growing past half a small pool reuses its history on every follow-up; a follow-up short of room consumes the turn it repeats and leaves an unrelated donor in place.
# The synthetic F32 model (16-token context) needs no download; the real Q8_0 fixture, when it is on disk, repeats the checks with room to stream.
# The synthetic model's file name holds a byte that is not UTF-8 on Linux and characters beyond ASCII that several Windows code pages cannot map elsewhere, and every reply naming the model must still be UTF-8.


class Server:
    def __init__(self, model, *extra):
        # The device and cache flags go on the command, not the executable path, which device_args would not recognise; the server then runs where the CLI it is compared with runs.
        args = ["serve", model, "--max-seqs", "8"] + list(extra)
        self.proc, self.port, self.log = common.start_server([common.exe_path()] + common.device_args(args, "f32"))

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

    def open(self, path, body):
        """A POST on a socket of its own, sent whole and left open, for a client that leaves when the test says."""
        s = socket.create_connection(("127.0.0.1", self.port))
        data = json.dumps(body).encode()
        s.sendall(b"POST %s HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n"
                  % (path.encode(), len(data)) + data)
        return s

    def wait(self, done, what, seconds=5):
        """/v1/health once `done` holds for it, within `seconds`; `what` names the failure otherwise."""
        deadline = time.time() + seconds
        while True:
            health = self.get("/v1/health")
            if done(health):
                return health
            assert time.time() < deadline, "%s: %s" % (what, health)
            time.sleep(0.05)

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
        assert models["data"][0]["context_length"] > 0, models
        # The model's name is its file name, with each byte that belongs to no UTF-8 character replaced by its own U+FFFD; every reply decodes as UTF-8 only if it is.
        # Python reads such a byte of a file name as a lone surrogate from U+DC80 to U+DCFF.
        name = "".join("\ufffd" if "\udc80" <= ch <= "\udcff" else ch for ch in os.path.basename(model))
        assert models["data"][0]["id"] == name and health["model"] == name, (models, health, name)

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
        # A number its field cannot hold is refused before any cast, and -1, no cap or no seed on the compatible routes, is refused on the native route as a cap or a seed.
        for field, value in (("max_tokens", 3e9), ("top_k", 1e10), ("seed", -1)):
            status, err = srv.post("/v1/generate", dict({"prompt": "a", "max_tokens": 2}, **{field: value}))
            assert status == 400 and field in err["error"], (field, status, err)
        assert srv.post("/v1/generate", {"prompt": "a", "max_tokens": -1})[0] == 400
        # A sampling field outside the range the CLI's flag takes is refused, the penalty's synonym on the compatible route too, and the ends of each range are accepted.
        # A top_k of -1 is refused here on the native route only, since the compatible routes take it as no top-k.
        for field, value in (("temperature", -0.5), ("temperature", "0.5"), ("temperature", 1e39), ("top_k", -1), ("top_p", 1.5),
                             ("top_p", -0.1), ("penalty", 0.5)):
            status, err = srv.post("/v1/generate", dict({"prompt": "a", "max_tokens": 2}, **{field: value}))
            assert status == 400 and field in err["error"], (field, value, status, err)
        for field in ("penalty", "repetition_penalty"):
            status, err = srv.post("/v1/completions", {"prompt": "a", "max_tokens": 2, field: 0.9})
            assert status == 400 and field in err["error"]["message"], (field, status, err)
        status, reply = srv.post("/v1/generate", {"prompt": "a", "max_tokens": 2, "temperature": 0, "top_k": 0, "top_p": 1, "penalty": 1})
        assert status == 200 and reply["ids"], (status, reply)

        # A client that leaves mid-stream: open the socket, start a request, close after the first bytes, and the server ends with nothing active.
        s = srv.open("/v1/generate", {"prompt": prompts[0], "max_tokens": long_n, "temperature": 0, "stream": True})
        s.recv(64)
        s.close()
        srv.wait(lambda h: h["active"] == 0, "a cancelled request stayed active", 60)

        # /v1/chat renders through the template and answers; the synthetic model's 16-token context has no room for a rendered turn.
        if chat:
            status, reply = srv.post("/v1/chat", {"messages": [{"role": "user", "content": prompts[0]}],
                                                  "max_tokens": 2, "temperature": 0})
            assert status == 200 and reply["tokens"] >= 1, reply

        # The compatible routes: /v1/completions gives the native route's greedy text in the standard shape, whole and streamed with the finish chunk then the end marker; /v1/chat/completions renders the same template as /v1/chat, its first chunk carries the role and its usage counts add up; the standard refusals have the standard shape.
        # An absent max_tokens, or -1, means no cap, checked on the synthetic model only, since the real model's uncapped reply would run long.
        if not chat:
            limit = models["data"][0]["context_length"]
            uncapped = []
            for cap in ({}, {"max_tokens": -1}):
                status, reply = srv.post("/v1/completions", dict({"prompt": prompts[0], "temperature": 0}, **cap))
                assert status == 200 and reply["choices"][0]["finish_reason"] in ("stop", "length"), reply
                assert reply["usage"]["completion_tokens"] >= 1 and reply["usage"]["total_tokens"] <= limit, reply
                uncapped.append((reply["choices"][0]["text"], reply["usage"]["completion_tokens"]))
            assert uncapped[0] == uncapped[1], uncapped
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
        # Clients send a seed of -1 for a random one, and the compatible routes sample it as they sample a request with no seed; a seed below -1 is still refused.
        sampled = [srv.post("/v1/completions", dict({"prompt": prompts[0], "max_tokens": n, "temperature": 1.0}, **seed))
                   for seed in ({}, {"seed": -1})]
        assert all(status == 200 for status, _ in sampled), sampled
        assert sampled[0][1]["choices"][0]["text"] == sampled[1][1]["choices"][0]["text"], sampled
        status, err = srv.post("/v1/completions", {"prompt": prompts[0], "max_tokens": 2, "seed": -2})
        assert status == 400 and "seed" in err["error"]["message"], err
        # Clients send a top_k of -1 for no top-k, and the compatible routes sample it as top_k 0, which keeps every token; a top_k below -1 is still refused.
        sampled = [srv.post("/v1/completions", {"prompt": prompts[0], "max_tokens": n, "temperature": 1.0, "seed": 7, "top_k": k})
                   for k in (0, -1)]
        assert all(status == 200 for status, _ in sampled), sampled
        assert sampled[0][1]["choices"][0]["text"] == sampled[1][1]["choices"][0]["text"], sampled
        status, err = srv.post("/v1/completions", {"prompt": prompts[0], "max_tokens": 2, "top_k": -2})
        assert status == 400 and "top_k" in err["error"]["message"], err
        # A body past the size limit is refused while it is read, still in the compatible route's error shape.
        s = socket.create_connection(("127.0.0.1", srv.port), timeout=30)
        s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n" % (65 << 20))
        raw = b""
        while True:
            part = s.recv(4096)
            if not part:
                break
            raw += part
        s.close()
        head, _, payload = raw.partition(b"\r\n\r\n")
        err = json.loads(payload)
        assert head.startswith(b"HTTP/1.1 413") and isinstance(err["error"], dict) and err["error"]["message"], raw
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
    """The serving limits: a KV budget below the context bounds a request, and a full queue refuses with 503 rather than waiting."""
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


def check_paused_prefill(model):
    """A long uncapped prompt read one token a pass is still prefilling when an earlier uncapped request has to grow and the pool has no room, so it is paused part-way; resumed, its greedy text is the CLI's for the whole prompt."""
    flags = ("--ubatch", "1")
    srv = Server(model, "--ctx-size", "1280", *flags)
    try:
        # 464 tokens in a pool of 1280: in blocks of 64 or 128 both requests fit at admission, the short one reaches its first growth step before this prompt is read, and that step does not fit beside it.
        with open(os.path.join(os.path.dirname(__file__), "data", "wiki.test.raw"), encoding="utf-8") as f:
            long_prompt = f.read()[:1800]
        # The short request streams and is queued first, so the long one is the latest admitted and the one paused.
        s = srv.open("/v1/completions", {"prompt": "Once upon a time", "temperature": 0, "stream": True})
        s.recv(64)
        result = []
        failure = []
        # A stop string ends the long request a few tokens in, enough to tell which prompt it resumed from.
        def worker():
            try:
                result.append(srv.post("/v1/completions", {"prompt": long_prompt, "temperature": 0, "stop": [" ."]}, timeout=900))
            except Exception as e:
                failure.append(e)
        later = threading.Thread(target=worker)
        later.start()
        # Once the long request is paused the short one leaves, so the long one resumes now rather than after the short one's whole reply.
        while later.is_alive() and srv.get("/v1/health")["pauses"] == 0:
            time.sleep(0.05)
        s.close()
        later.join()
        # A post that failed in the worker raises its own error here rather than leaving result empty.
        if failure:
            raise failure[0]
        status, reply = result[0]
        assert status == 200, reply
        want = cli_greedy_text(model, long_prompt, reply["usage"]["completion_tokens"], flags)
        assert reply["choices"][0]["text"] == want, (reply["choices"][0]["text"], want)
        health = srv.get("/v1/health")
        assert health["active"] == 0 and health["pauses"] >= 1, health
    finally:
        srv.close()


def check_conversation(model, text):
    """A conversation whose history grows past half of a small pool: each follow-up repeats the last turn's prompt and reply and adds half of `text`, forks that turn's history however full the pool is, so its `reused_tokens` is larger than the last turn's and the server's `prefix_tokens` grows on every follow-up, and gives the CLI's greedy text for its whole prompt."""
    pool, n, turns = 1024, 16, 6
    srv = Server(model, "--ctx-size", str(pool))
    try:
        words = text.split(" ")
        halves = [" ".join(words[:len(words) // 2]), " ".join(words[len(words) // 2:])]
        prompt, reused, prefix = halves[0], 0, 0
        for turn in range(turns):
            status, reply = srv.post("/v1/generate", {"prompt": prompt, "max_tokens": n, "temperature": 0})
            assert status == 200, reply
            assert reply["text"] == cli_greedy_text(model, prompt, n), (turn, reply["text"])
            health = srv.get("/v1/health")
            if turn:
                assert reply["reused_tokens"] > reused and health["prefix_tokens"] > prefix, (turn, reply["prompt_tokens"], reply["reused_tokens"], reused, health)
            reused, prefix = reply["reused_tokens"], health["prefix_tokens"]
            prompt += reply["text"] + " " + halves[(turn + 1) % 2]
        assert reply["prompt_tokens"] > pool // 2, reply
        return turns
    finally:
        srv.close()


def check_unrelated_donor(model):
    """A follow-up turn beside an unrelated donor, on a pool with room for the follow-up once the turn it repeats is consumed but not beside both donors: the follow-up consumes that turn's history rather than evicting the unrelated donor, so a later prompt repeating the unrelated request's history still reuses it, with the CLI's greedy text."""
    pool, n = 1024, 16
    srv = Server(model, "--ctx-size", str(pool))
    try:
        with open(os.path.join(os.path.dirname(__file__), "data", "wiki.test.raw"), encoding="utf-8") as f:
            text = f.read()
        # In a pool of 1024 tokens in blocks of 64 or 128, the unrelated request keeps about 200 tokens and the first turn about 280, and the follow-up's prompt and max_tokens, about 590, fit beside one of them but not beside both.
        unrelated, first = text[4000:4750], text[:1100]
        replies = []
        for prompt in (unrelated, first):
            status, reply = srv.post("/v1/generate", {"prompt": prompt, "max_tokens": n, "temperature": 0})
            assert status == 200 and reply["reused_tokens"] == 0, reply
            replies.append(reply)
        status, reply = srv.post("/v1/generate", {"prompt": first + replies[1]["text"] + " " + text[1100:2200], "max_tokens": n, "temperature": 0})
        assert status == 200 and reply["reused_tokens"] > 0, reply
        # One donor went to make room for the follow-up, so the pool was short; the first turn's is the one it consumed.
        health = srv.get("/v1/health")
        assert health["donors"] == 2, health
        later = unrelated + replies[0]["text"] + " " + text[6000:6200]
        status, reply = srv.post("/v1/generate", {"prompt": later, "max_tokens": n, "temperature": 0})
        assert status == 200 and reply["reused_tokens"] > 0, (reply, srv.get("/v1/health"))
        assert reply["text"] == cli_greedy_text(model, later, n), (reply["text"],)
    finally:
        srv.close()


# A client that leaves is noticed within seconds wherever its request is, though nothing written to it fails: a whole reply while it is generated, a streamed prompt while it is read, a request waiting for the one slot, and a whole reply whose client shuts only its sending side, which then gets no answer.
# The server runs one slot and reads prompts one token a pass, so a second request queues and a long prompt stays in its prefill; the pool is POOL tokens.
# Every request left behind would run for thousands of passes, a whole reply of LONG tokens or a prompt of about 6400, far past the seconds its departure has to be noticed in, so a server that notices nothing fails here on any device.
POOL = 8192
LONG = POOL - 64


def leave_whole(srv):
    s = srv.open("/v1/generate", {"prompt": "Once upon a time", "max_tokens": LONG, "temperature": 0})
    srv.wait(lambda h: h["active"] == 1, "the whole reply did not start")
    s.close()
    return "a whole reply whose client left"


def leave_prefill(srv):
    with open(os.path.join(os.path.dirname(__file__), "data", "wiki.test.raw"), encoding="utf-8") as f:
        long_prompt = f.read()[:28000]
    # About 6400 tokens, one a pass: the stream has sent only its head when the client leaves.
    s = srv.open("/v1/completions", {"prompt": long_prompt, "max_tokens": 8, "temperature": 0, "stream": True})
    s.recv(64)
    srv.wait(lambda h: h["active"] == 1, "the long prompt did not start")
    s.close()
    return "a streamed prompt whose client left while it was read"


def leave_queued(srv):
    busy = srv.open("/v1/generate", {"prompt": "Once upon a time", "max_tokens": LONG, "temperature": 0, "stream": True})
    busy.recv(64)
    srv.wait(lambda h: h["active"] == 1, "the busy request did not start")
    s = srv.open("/v1/generate", {"prompt": "The capital of France is", "max_tokens": 8, "temperature": 0})
    srv.wait(lambda h: h["queued"] == 1, "the second request did not queue")
    s.close()
    # It leaves the queue while the slot's request, whose client stays, goes on.
    health = srv.wait(lambda h: h["queued"] == 0, "a queued request whose client left stayed queued")
    assert health["active"] == 1, health
    busy.close()
    return "a queued request whose client left"


def leave_half(srv):
    """A client that shuts its sending side and reads on is taken as gone: its request ends and the connection closes with no answer, not even an error."""
    s = srv.open("/v1/generate", {"prompt": "Once upon a time", "max_tokens": LONG, "temperature": 0})
    srv.wait(lambda h: h["active"] == 1, "the whole reply did not start")
    s.shutdown(socket.SHUT_WR)
    s.settimeout(5)
    raw = b""
    try:
        part = s.recv(4096)
        while part:
            raw += part
            part = s.recv(4096)
    except socket.timeout:
        raise AssertionError("a client that shut its sending side: the connection stayed open")
    s.close()
    assert raw == b"", raw
    return "a whole reply whose client shut its sending side"


def settled(srv, what):
    """Nothing active or queued, and a request reaching the whole pool starts, which it can only once no request holds blocks, donors giving theirs up; its first token is enough."""
    srv.wait(lambda h: h["active"] == 0 and h["queued"] == 0, what + " stayed")
    s = srv.open("/v1/generate", {"prompt": "a", "max_tokens": POOL - 1, "temperature": 0, "stream": True})
    s.settimeout(5)
    raw = b""
    try:
        while b"\"id\"" not in raw:
            part = s.recv(4096)
            assert part, raw
            raw += part
    except socket.timeout:
        raise AssertionError(what + ": the pool's blocks did not come back")
    s.close()
    srv.wait(lambda h: h["active"] == 0, "a request reaching the whole pool stayed after its client left")


def check_departed(model):
    srv = Server(model, "--max-seqs", "1", "--ctx-size", str(POOL), "--ubatch", "1")
    try:
        for leave in (leave_whole, leave_prefill, leave_queued, leave_half):
            settled(srv, leave(srv))
    finally:
        srv.close()


def run():
    if common.f32_cache_skip("server"):
        return True
    with tempfile.TemporaryDirectory(prefix="llmx_server_") as directory:
        # Every reply that names the model carries its file name, so the synthetic model's holds a byte that is not UTF-8 where the file system takes one (Linux), and characters beyond ASCII elsewhere.
        # On Windows a name read in the system code page instead of as UTF-8 fails only where one of its UTF-8 bytes has no mapping there, so U+00E1 brings 0xA1 for code page 1257, U+00E0 brings 0xA0 for 932, and U+4E2D breaks 936, 949 and 950.
        name = "tiny-f32-\udcff.gguf" if sys.platform.startswith("linux") else "tiny-f32-\u00e1\u00e9\u00e0\u4e2d.gguf"
        model = os.path.join(directory, name)
        f32.write_model(model, f32.tensors(True))
        # The synthetic model's context is 16 tokens: prompt plus tokens stay inside it.
        n = check_server(model, ["a", "ab", "abc", "abcdefg"], 6, 14, chat=False)
        print("server: synthetic F32 model, %d prompts greedy-equal to the CLI alone and four at a time, a stream, "
              "a seeded repeat, refusals, a cancelled stream, the compatible completions, its file name as UTF-8 in every reply  [ok]" % n)
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
        check_paused_prefill(real)
        turns = check_conversation(real, excerpt)
        check_unrelated_donor(real)
        check_departed(real)
        print("server: %s, %d prompts greedy-equal to the CLI alone and four at a time, a stream, a seeded repeat, "
              "refusals, a cancelled stream, a chat turn, the compatible routes, a reused prefix, the limits, uncapped requests sharing a pool, "
              "a prompt paused while prefilling, a %d-turn conversation past half the pool reusing its history on every follow-up, "
              "a follow-up consuming the turn it repeats while an unrelated donor stays, clients leaving a whole reply, a prefill and the queue, and one shutting its sending side  [ok]"
              % (os.path.basename(real), n, turns))
    else:
        print("server: SKIP real-model pass - fixture model not on disk")
    return True
