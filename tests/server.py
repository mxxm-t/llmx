import http.client
import json
import math
import os
import re
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
import mxfp4
import server_mix_tool
from tokenizer import build_byte_vocab

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import server_load  # noqa: E402

# The server of docs/SERVER.md against the CLI on the same file, on synthetic dense and MoE models that need no download and on the real Q8_0 fixture when it is on disk; AGENTS.md (Tests, Server) lists what each check holds.


class Server:
    def __init__(self, model, *extra):
        # The device and cache flags go on the command, not the executable path, which device_args would not recognise; the server then runs where the CLI it is compared with runs.
        # Eight sequences unless the check names its own count, since a flag given twice is refused.
        args = ["serve", model] + ([] if "--max-seqs" in extra else ["--max-seqs", "8"]) + list(extra)
        command = common.device_args(args, "f32")
        self.requested_dtype = command[command.index("--dtype") + 1] if "--dtype" in command else "auto"
        self.proc, self.port, self.log = common.start_server([common.exe_path()] + command)

    def get(self, path):
        with urllib.request.urlopen("http://127.0.0.1:%d%s" % (self.port, path), timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))

    def post(self, path, body, timeout=300):
        """`body` as JSON, or as it is when it is bytes."""
        data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        req = urllib.request.Request("http://127.0.0.1:%d%s" % (self.port, path), data=data,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode("utf-8"))

    def stream(self, path, body):
        """Every SSE data payload, parsed, in order; the end marker is None."""
        return [None if payload == "[DONE]" else json.loads(payload) for payload in self.stream_raw(path, body)]

    def raw(self, path, body):
        """A whole reply's body as the server wrote it, for comparing bytes."""
        req = urllib.request.Request("http://127.0.0.1:%d%s" % (self.port, path), data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.read().decode("utf-8")

    def stream_raw(self, path, body):
        """Every SSE data payload of a stream as the server wrote it, in order."""
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request("http://127.0.0.1:%d%s" % (self.port, path), data=data,
                                     headers={"Content-Type": "application/json"})
        payloads = []
        with urllib.request.urlopen(req, timeout=300) as r:
            for raw in r:
                line = raw.decode("utf-8").rstrip("\n")
                if line.startswith("data: "):
                    payloads.append(line[6:])
        return payloads

    def open(self, path, body):
        """A POST on a socket of its own, sent whole and left open, for a client that leaves when the test says."""
        s = socket.create_connection(("127.0.0.1", self.port))
        data = json.dumps(body).encode()
        s.sendall(b"POST %s HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n"
                  % (path.encode(), len(data)) + data)
        return s

    def together(self, path, bodies, timeout=600):
        """Each body's status and parsed reply, the requests arriving at once: each is sent whole on a socket of its own but for its last byte, then the last bytes back to back.
        Client threads that each open a connection and send their request can arrive far enough apart that one request ends before the next is admitted."""
        socks, lasts = [], []
        try:
            for body in bodies:
                s = socket.create_connection(("127.0.0.1", self.port), timeout=timeout)
                socks.append(s)
                data = json.dumps(body).encode("utf-8")
                request = b"POST %s HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n" % (path.encode(), len(data)) + data
                s.sendall(request[:-1])
                lasts.append(request[-1:])
            for s, last in zip(socks, lasts):
                s.sendall(last)
            replies = []
            for s in socks:
                r = http.client.HTTPResponse(s)
                r.begin()
                replies.append((r.status, json.loads(r.read().decode("utf-8"))))
            return replies
        finally:
            for s in socks:
                s.close()

    def oversized(self, path):
        """The head and the parsed body of the reply to a POST announcing a body past the size limit, which the server refuses while it reads."""
        s = socket.create_connection(("127.0.0.1", self.port), timeout=30)
        s.sendall(b"POST %s HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n" % (path.encode(), 65 << 20))
        raw = b""
        while True:
            part = s.recv(4096)
            if not part:
                break
            raw += part
        s.close()
        head, _, payload = raw.partition(b"\r\n\r\n")
        return head, json.loads(payload)

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
        common.stop_server(self.proc, self.log)


def cli_greedy_text(model, prompt, n, flags=()):
    """What `generate --temp 0` prints between its pp and tg lines, as text, with the f32 cache sides and the flags the server under test is started with."""
    p = common.run_process(["generate", model, prompt, "-n", str(n), "--temp", "0"] + list(flags), cache="f32")
    assert p.returncode == 0, p.stderr
    return common.generate_text(p.stdout).decode("utf-8")


def repaired(raw):
    """The bytes `raw` as the server writes text into JSON: each valid UTF-8 character as it is, and U+FFFD for each byte that starts none."""
    out, i = [], 0
    while i < len(raw):
        for n in (1, 2, 3, 4):
            try:
                ch = raw[i:i + n].decode("utf-8")
            except UnicodeDecodeError:
                continue
            if len(ch) == 1:
                out.append(ch)
                i += n
                break
        else:
            out.append("\ufffd")
            i += 1
    return "".join(out)


def cli_tokenize(model, text):
    """The ids `llmx tokenize` prints for `text`."""
    p = common.run_process(["tokenize", model, text])
    assert p.returncode == 0, p.stderr
    return common.parse_ids(common.cli_stdout(p.stdout).decode("ascii"))


def cli_detokenize(model, ids):
    """The bytes `llmx detokenize` prints for `ids`, without the line feed that ends them, repaired as the server repairs text."""
    p = common.run_process(["detokenize", model, ",".join(map(str, ids))])
    assert p.returncode == 0, p.stderr
    out = common.cli_stdout(p.stdout)
    assert out.endswith(b"\n"), out
    return repaired(out[:-1])


def check_tokenize(srv, model, texts, replies, vocab, chat):
    """/v1/tokenize and /v1/detokenize against `llmx tokenize` and `llmx detokenize`, the load tool's count and the prompts the generating routes read; AGENTS.md lists the cases."""
    counts = {}
    for text in texts:
        want = cli_tokenize(model, text)
        counts[text] = len(want)
        # add_special is not read, whatever its value.
        for extra in ({}, {"add_special": False}, {"add_special": True}, {"add_special": 1}):
            status, reply = srv.post("/v1/tokenize", dict({"text": text}, **extra))
            assert status == 200 and reply == {"tokens": want, "count": len(want)}, (text, extra, status, reply, want)
        status, reply = srv.post("/v1/detokenize", {"tokens": want})
        assert status == 200 and reply == {"text": text}, (text, status, reply)
        if any(ord(ch) > 127 for ch in text):
            for ids in [[i] for i in dict.fromkeys(want)] + [want, want[::-1]]:
                status, reply = srv.post("/v1/detokenize", {"tokens": ids})
                assert status == 200 and reply["text"] == cli_detokenize(model, ids), (text, ids, reply)
    # `vocab` is the output rows /v1/models counts, and the edge the route checks is the tokenizer's, so the CLI must take the id below it and refuse the one at it.
    status, reply = srv.post("/v1/detokenize", {"tokens": [vocab - 1]})
    assert status == 200 and reply["text"] == cli_detokenize(model, [vocab - 1]), reply
    assert common.run_process(["detokenize", model, str(vocab)]).returncode != 0, vocab
    for prompt, generated in replies.items():
        status, reply = srv.post("/v1/tokenize", {"text": prompt})
        assert status == 200 and reply["count"] == generated["prompt_tokens"], (prompt, reply, generated)
        status, reply = srv.post("/v1/detokenize", {"tokens": generated["ids"]})
        assert status == 200 and reply["text"] == generated["text"], (prompt, reply, generated)
    count = server_load.tokenize_route(server_load.Target("http://127.0.0.1:%d" % srv.port), 30)
    assert count is not None and {text: count(text) for text in counts} == counts, counts
    if chat:
        # The chat fixture's cases carried over from the Jinja2 goldens, under the template the fixture pins for this file, as transformers renders them.
        with open(os.path.join(os.path.dirname(__file__), "data", "baseline_chat_template.json"), encoding="utf-8") as f:
            [pinned] = [t for t in json.load(f)["templates"] if any(o.split()[-1] == os.path.basename(model) for o in t["origin"])]
        cases = [case for case in pinned["cases"] if case["name"].startswith("legacy-") and case["generate"]]
        assert len(cases) == 6, [case["name"] for case in cases]
        for case in cases:
            want = cli_tokenize(model, case["expected"])
            status, reply = srv.post("/v1/tokenize", {"messages": case["messages"]})
            assert status == 200 and reply == {"tokens": want, "count": len(want)}, (case, reply, want)
        last = cases[-1]
        status, turn = srv.post("/v1/chat", {"messages": last["messages"], "max_tokens": 4, "temperature": 0})
        assert status == 200 and turn["prompt_tokens"] == len(cli_tokenize(model, last["expected"])), turn
        status, plain = srv.post("/v1/generate", {"prompt": last["expected"], "max_tokens": 4, "temperature": 0})
        assert status == 200 and plain["ids"] == turn["ids"], (plain, turn)
    # Refusals in the native error shape: a body that is not JSON or not an object, a text that is not a string, both text and messages or neither, messages that are not a non-empty array, tokens that are not an array, an id that is not a whole number or lies outside the vocabulary, 2^32 past a valid id included, and a body past the size limit.
    refused = [("/v1/tokenize", b"{"), ("/v1/tokenize", b"[]"), ("/v1/tokenize", {}), ("/v1/tokenize", {"text": 5}),
               ("/v1/tokenize", {"text": "a", "messages": [{"role": "user", "content": "a"}]}),
               ("/v1/tokenize", {"messages": []}), ("/v1/tokenize", {"messages": "a"}),
               ("/v1/detokenize", b"{"), ("/v1/detokenize", {}), ("/v1/detokenize", {"tokens": "97"})]
    refused += [("/v1/detokenize", {"tokens": [97, bad]}) for bad in (vocab, -1, 1.5, "97", True, None, 2 ** 32 + 97, 1e300)]
    for path, body in refused:
        status, err = srv.post(path, body)
        assert status == 400 and isinstance(err["error"], str) and err["error"], (path, body, status, err)
    for path in ("/v1/tokenize", "/v1/detokenize"):
        head, err = srv.oversized(path)
        assert head.startswith(b"HTTP/1.1 413") and isinstance(err["error"], str), (path, head, err)


def cli_reply(model, prompt, n, flags):
    """`generate`'s reply with the f32 cache sides: its bytes and the token count of its tg line."""
    p = common.run_process(["generate", model, prompt, "-n", str(n)] + list(flags), cache="f32")
    assert p.returncode == 0, p.stderr
    count = re.findall(rb"^tg: (\d+) tok", common.cli_stdout(p.stdout), re.M)[-1]
    return common.generate_text(p.stdout), int(count)


def cli_chat_reply(model, line, n, flags):
    """`chat`'s reply to one line with an empty system message, as a reply carries it, with the f32 cache sides."""
    p = common.run_process(["chat", model, "--system", "", "-n", str(n)] + list(flags), input=(line + "\n").encode(), cache="f32")
    assert p.returncode == 0, p.stderr
    out = common.cli_stdout(p.stdout)
    banner = b"Chat ready (type your message; Ctrl+C to quit)\n"
    assert out.startswith(banner) and out.endswith(b"\n"), out
    return repaired(out[len(banner):-1])


def post_ok(srv, path, body):
    status, reply = srv.post(path, body)
    assert status == 200, (path, body, reply)
    return reply


def refuses_ignore_eos(srv):
    """A non-boolean ignore_eos is refused with 400 on every generating route, in the route's error shape."""
    for path in ("/v1/generate", "/v1/chat", "/v1/completions", "/v1/chat/completions"):
        body = {"messages": [{"role": "user", "content": "a"}]} if "chat" in path else {"prompt": "a"}
        for value in ("true", 1, 0, None, [True], {}):
            status, err = srv.post(path, dict(body, max_tokens=2, ignore_eos=value))
            message = err["error"]["message"] if "completions" in path else err["error"]
            assert status == 400 and "ignore_eos" in message, (path, value, status, err)


def alone_and_together(srv, bodies):
    """Each /v1/generate body's ids, alone and then all at once, which must be the same."""
    alone = [post_ok(srv, "/v1/generate", body)["ids"] for body in bodies]
    results = {}
    def worker(i):
        results[i] = srv.post("/v1/generate", bodies[i])
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(bodies))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for i, body in enumerate(bodies):
        status, reply = results[i]
        assert status == 200 and reply["ids"] == alone[i], (body, reply, alone[i])


# ignore_eos on the synthetic model, whose token b below 256 is the byte b, so the bytes the CLI prints are its ids.
def check_ignore_eos_synthetic(directory):
    """Without ignore_eos a greedy reply ends at the model's end token, and with it the reply runs to its limit, through the CLI and the server alike, greedy and seeded; uncapped, it runs to the context, as the CLI asked for the room its prompt leaves does; requests with and without it give their own ids side by side; a non-boolean value is refused."""
    weights = f32.tensors(True)
    prompt, n = "ab", 12
    ids = list(cli_reply(f32.write_model(os.path.join(directory, "tiny-plain.gguf"), weights), prompt, n, ("--temp", "0"))[0])
    # The end token is the first id from the third on that the reply has not given before, so the reply ends just before it.
    k = next(i for i in range(2, n) if ids[i] not in ids[:i])
    end = ids[k]
    model = f32.write_model(os.path.join(directory, "tiny-end.gguf"), weights, eos_id=end)
    greedy, seeded = ("--temp", "0"), ("--temp", "1", "--seed", "7")
    early, count = cli_reply(model, prompt, n, greedy)
    assert list(early) == ids[:k] and count == k, (early, ids, k)
    full, count = cli_reply(model, prompt, n, greedy + ("--ignore-eos",))
    assert count == len(full) == n and list(full[:k]) == ids[:k] and end not in full, (full, end)
    drawn, count = cli_reply(model, prompt, n, seeded + ("--ignore-eos",))
    assert count == len(drawn) == n and end not in drawn, (drawn, end)
    srv = Server(model)
    try:
        base = {"prompt": prompt, "max_tokens": n}
        for body, want, finish in (({"temperature": 0}, ids[:k], "eos"), ({"temperature": 0, "ignore_eos": False}, ids[:k], "eos"),
                                   ({"temperature": 0, "ignore_eos": True}, list(full), "length"),
                                   ({"temperature": 1, "seed": 7, "ignore_eos": True}, list(drawn), "length")):
            reply = post_ok(srv, "/v1/generate", dict(base, **body))
            assert reply["ids"] == want and reply["finish"] == finish, (body, reply, want)
        events = srv.stream("/v1/generate", dict(base, temperature=0, ignore_eos=True, stream=True))
        assert [e["id"] for e in events if e and "id" in e] == list(full) and events[-2]["finish"] == "length", events[-2:]
        for body, tokens, finish in (({}, k, "stop"), ({"ignore_eos": True}, n, "length")):
            reply = post_ok(srv, "/v1/completions", dict(base, temperature=0, **body))
            assert reply["usage"]["completion_tokens"] == tokens and reply["choices"][0]["finish_reason"] == finish, (body, reply)
        # Uncapped, the reply ends at the end token without ignore_eos and at the 16-token context with it.
        limit = srv.get("/v1/models")["data"][0]["context_length"]
        reply = post_ok(srv, "/v1/completions", {"prompt": prompt, "temperature": 0})
        assert reply["usage"]["completion_tokens"] == k and reply["choices"][0]["finish_reason"] == "stop", reply
        reply = post_ok(srv, "/v1/completions", {"prompt": prompt, "temperature": 0, "ignore_eos": True})
        assert reply["usage"]["total_tokens"] == limit and reply["choices"][0]["finish_reason"] == "length", reply
        # The CLI asked for the room the prompt leaves gives the server's ids and fills the context too.
        room = limit - reply["usage"]["prompt_tokens"]
        filled, count = cli_reply(model, prompt, room, greedy + ("--ignore-eos",))
        capped = post_ok(srv, "/v1/generate", dict(base, max_tokens=room, temperature=0, ignore_eos=True))
        assert count == room and capped["ids"] == list(filled) and capped["finish"] == "length", (filled, capped)
        alone_and_together(srv, [dict(base, prompt=p, temperature=0, ignore_eos=on) for p, on in (("a", True), ("ab", False), ("ab", True), ("abc", True))])
        refuses_ignore_eos(srv)
    finally:
        srv.close()
    return k, n


# A turn in Qwen3's chat format answered without thinking, so the greedy reply is short and ends at the model's end token.
IGNORE_EOS_LINE = "Say hello in one short sentence. /no_think"
IGNORE_EOS_PROMPT = "<|im_start|>user\n" + IGNORE_EOS_LINE + "<|im_end|>\n<|im_start|>assistant\n"


def check_ignore_eos_real(model):
    """On a real model a greedy reply that ends early at the end token runs to its limit with ignore_eos, through generate and chat on the CLI and all four routes, greedy and seeded, a stop text still ending it; with and without it side by side each request gives its own ids."""
    n, prompt = 48, IGNORE_EOS_PROMPT
    greedy, seeded = ("--temp", "0"), ("--temp", "0.8", "--seed", "11")
    early, k = cli_reply(model, prompt, n, greedy)
    full, count = cli_reply(model, prompt, n, greedy + ("--ignore-eos",))
    assert k < n and count == n and full.startswith(early), (early, k, full, count)
    drawn, count = cli_reply(model, prompt, n, seeded + ("--ignore-eos",))
    assert count == n, (drawn, count)
    early, full, drawn = repaired(early), repaired(full), repaired(drawn)
    # A stop text reached only past the masked end token still ends the reply.
    stop = full[len(early):len(early) + 4]
    assert stop and stop not in early, (early, full)
    stopped = repaired(cli_reply(model, prompt, n, greedy + ("--ignore-eos", "--stop", stop))[0])
    assert full.startswith(stopped) and stop in stopped and len(stopped) < len(full), (stopped, full)
    chat_early = cli_chat_reply(model, IGNORE_EOS_LINE, n, greedy)
    chat_full = cli_chat_reply(model, IGNORE_EOS_LINE, n, greedy + ("--ignore-eos",))
    srv = Server(model)
    try:
        base = {"prompt": prompt, "max_tokens": n}
        for body, text, tokens, finish in (({"temperature": 0}, early, k, "eos"), ({"temperature": 0, "ignore_eos": True}, full, n, "length"),
                                           ({"temperature": 0.8, "seed": 11, "ignore_eos": True}, drawn, n, "length"),
                                           ({"temperature": 0, "ignore_eos": True, "stop": [stop]}, stopped, None, "stop")):
            reply = post_ok(srv, "/v1/generate", dict(base, **body))
            assert reply["text"] == text and reply["finish"] == finish and (tokens is None or reply["tokens"] == tokens), (body, reply, text)
        reply = post_ok(srv, "/v1/completions", dict(base, temperature=0, ignore_eos=True))
        assert reply["choices"][0]["text"] == full and reply["choices"][0]["finish_reason"] == "length", reply
        messages = [{"role": "system", "content": ""}, {"role": "user", "content": IGNORE_EOS_LINE}]
        for body, text, finish in (({}, chat_early, "eos"), ({"ignore_eos": True}, chat_full, "length")):
            reply = post_ok(srv, "/v1/chat", dict({"messages": messages, "max_tokens": n, "temperature": 0}, **body))
            assert reply["text"] == text and reply["finish"] == finish and (finish == "eos") == (reply["tokens"] < n), (body, reply, text)
        reply = post_ok(srv, "/v1/chat/completions", {"messages": messages, "max_tokens": n, "temperature": 0, "ignore_eos": True})
        assert reply["choices"][0]["message"] == split_reply(chat_full) and reply["choices"][0]["finish_reason"] == "length", (reply, chat_full)
        alone_and_together(srv, [dict(base, temperature=0, ignore_eos=on) for on in (False, True)] +
                           [dict(base, temperature=0.8, seed=11, ignore_eos=True), dict(base, prompt="The capital of France is", temperature=0, ignore_eos=True)])
    finally:
        srv.close()
    return k, n


def check_dtype(model):
    """Explicit CPU policies are reported by health and reach the same generation as the CLI."""
    for requested, effective, how in (("f16", "f16", "native"), ("f32", "f32", "native"),
                                     ("bf16", "bf16", "emulated"), ("auto", "f16", "native")):
        flags = ("--device", "cpu", "--threads", "1", "--dtype", requested)
        srv = Server(model, *flags)
        try:
            record = srv.get("/v1/health")["dtype"]
            paths = common.CPU_DTYPE_PATHS[effective]
            assert record == {"requested": requested, "declared": "bf16", "effective": effective,
                              "devices": [{"device": "cpu", "effective": effective, "how": how, "paths": paths}]}, record
            reply = post_ok(srv, "/v1/generate", {"prompt": "abc", "max_tokens": 3, "temperature": 0})
            want, count = cli_reply(model, "abc", 3, flags + ("--temp", "0"))
            assert reply["text"] == repaired(want) and len(reply["ids"]) == count, (requested, reply, want)
        finally:
            srv.close()


def check_server(model, prompts, n, long_n, chat, texts, prefix=None, flags=()):
    srv = Server(model, *flags)
    try:
        health = srv.get("/v1/health")
        assert health["status"] == "ok" and health["active"] == 0, health
        dtype = health["dtype"]
        assert set(dtype) == {"requested", "declared", "effective", "devices"} and dtype["requested"] == srv.requested_dtype, dtype
        assert dtype["declared"] in ("f32", "f16", "bf16") and dtype["effective"] in ("f32", "f16", "bf16"), dtype
        assert dtype["devices"], dtype
        for device in dtype["devices"]:
            assert set(device) == {"device", "how", "paths", "effective"} and device["device"] and device["paths"], device
            assert device["how"] in ("native", "emulated", "fallback"), device
        models = srv.get("/v1/models")
        assert models["object"] == "list" and models["data"][0]["object"] == "model", models
        assert models["data"][0]["context_length"] > 0, models
        # The model's name is its file name, with each byte that belongs to no UTF-8 character replaced by its own U+FFFD; every reply decodes as UTF-8 only if it is.
        # Python reads such a byte of a file name as a lone surrogate from U+DC80 to U+DCFF.
        name = "".join("\ufffd" if "\udc80" <= ch <= "\udcff" else ch for ch in os.path.basename(model))
        assert models["data"][0]["id"] == name and health["model"] == name, (models, health, name)

        # Greedy through the server gives the CLI's text, and the replies are kept for the checks that follow.
        expected, replies = {}, {}
        for prompt in prompts:
            want = cli_greedy_text(model, prompt, n, flags)
            status, reply = srv.post("/v1/generate", {"prompt": prompt, "max_tokens": n, "temperature": 0})
            assert status == 200, reply
            assert reply["text"] == want, (prompt, reply["text"], want)
            assert reply["finish"] in ("eos", "length", "stop"), reply
            expected[prompt] = reply["ids"]
            replies[prompt] = reply

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

        check_tokenize(srv, model, texts, replies, models["data"][0]["vocab"], chat)

        # A client that leaves mid-stream once its reply has begun, and the server ends with nothing active.
        common.leave_mid_stream(srv.port, {"prompt": prompts[0], "max_tokens": long_n, "temperature": 0}, 64)
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
        check_logprobs(srv, prompts, n, chat)
        # A body past the size limit is refused while it is read, still in the compatible route's error shape.
        head, err = srv.oversized("/v1/chat/completions")
        assert head.startswith(b"HTTP/1.1 413") and isinstance(err["error"], dict) and err["error"]["message"], (head, err)
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


# A model whose next token follows from its last alone, since its one layer adds nothing to the embedding: after a line feed it writes "qr</think>o", then its end of text.
# Its template writes a turn's reasoning in angle brackets apart from its reply, and the reply three times over, so a render tells a turn split into "o" and "qr" from one left whole or split the other way.
REASONING_CONFIG = {"block_count": 1, "embedding_length": 16, "feed_forward_length": 8, "attention.head_count": 2,
                    "attention.head_count_kv": 1, "attention.key_length": 8, "context_length": 512}
REASONING_REPLY = b"qr</think>o"
REASONING_TEMPLATE = ("{% if messages[-1]['role'] != 'user' %}{{ raise_exception('The last message must be the user\\'s.') }}{% endif %}"
                      "{{ messages|length }}{% for m in messages %}|{{ m.role }}:"
                      "{% if m.reasoning_content is defined %}<{{ m.reasoning_content }}>{% endif %}{{ m.content * 3 }}{% endfor %}"
                      "{% if add_generation_prompt %}|assistant:\n{% endif %}")
# The same without the reasoning, as a template that knows none is written.
PLAIN_TEMPLATE = ("{{ messages|length }}{% for m in messages %}|{{ m.role }}:{{ m.content * 3 }}{% endfor %}"
                  "{% if add_generation_prompt %}|assistant:\n{% endif %}")
# The same opening the reply inside <think>, as the Qwen 3.5 templates do, so the reply's reasoning is "qr" and its content "o", unless the request sets enable_thinking false, which closes the <think> in the prompt as those templates do.
OPEN_TEMPLATE = PLAIN_TEMPLATE.replace("|assistant:\n", "|assistant:{% if enable_thinking is defined and not enable_thinking %}<think>\n\n</think>\n\n"
                                                        "{% else %}<think>\n{% endif %}")


def split_reply(text, opened=False):
    """The message the compatible chat route gives for reply `text`, written from the rule rather than the server's code: a reply inside <think>, opened by the template (`opened`) or by the reply after any newlines, gives the text up to the first </think> as reasoning_content, without the newlines around it, and the rest as content, without the newlines opening it; a cut-off reply is all reasoning; any other reply is all content."""
    if not opened:
        body = text.lstrip("\n")
        if not body.startswith("<think>"):
            return {"role": "assistant", "content": text}
        text = body[len("<think>"):]
    reasoning, closed, rest = text.partition("</think>")
    return {"role": "assistant", "reasoning_content": reasoning.strip("\n"), "content": rest.lstrip("\n") if closed else ""}


def reasoning_tensors():
    """The weights of that model: each token of the chain, from the line feed, is one dimension of the embedding, and the head maps that dimension to the token after it."""
    cfg, vocab, eos = REASONING_CONFIG, f32.VOCAB, f32.VOCAB - 1
    width, ff, hd = cfg["embedding_length"], cfg["feed_forward_length"], cfg["attention.key_length"]
    q, kv = cfg["attention.head_count"] * hd, cfg["attention.head_count_kv"] * hd
    chain = [ord("\n")] + list(REASONING_REPLY) + [eos]
    assert len(set(chain)) == len(chain) and len(chain) - 1 <= width
    embedding, head = [0.0] * (width * vocab), [0.0] * (width * vocab)
    for dim, (token, following) in enumerate(zip(chain, chain[1:])):
        embedding[token * width + dim] = 1.0
        head[following * width + dim] = 1.0
    tensors = [("token_embd.weight", [width, vocab], embedding), ("output.weight", [width, vocab], head)]
    tensors += [(name, [size], [1.0] * size) for name, size in (("output_norm.weight", width), ("blk.0.attn_norm.weight", width),
                                                                ("blk.0.ffn_norm.weight", width), ("blk.0.attn_q_norm.weight", hd),
                                                                ("blk.0.attn_k_norm.weight", hd))]
    tensors += [(name, shape, [0.0] * (shape[0] * shape[1]))
                for name, shape in (("blk.0.attn_q.weight", [width, q]), ("blk.0.attn_k.weight", [width, kv]),
                                    ("blk.0.attn_v.weight", [width, kv]), ("blk.0.attn_output.weight", [q, width]),
                                    ("blk.0.ffn_gate.weight", [width, ff]), ("blk.0.ffn_up.weight", [width, ff]),
                                    ("blk.0.ffn_down.weight", [ff, width]))]
    return [(name, None, shape, values) for name, shape, values in tensors]


def reasoning_render(messages):
    """What the template renders for `messages`, written out here, with the assistant's header after them."""
    text = "%d" % len(messages)
    for m in messages:
        text += "|%s:%s%s" % (m["role"], "<%s>" % m["reasoning_content"] if "reasoning_content" in m else "", m["content"] * 3)
    return text + "|assistant:\n"


def check_reasoning(model):
    """chat records its reply as a turn split at </think>, and the server reads an assistant message a client sends back the same way, whether its reasoning stays in the content, comes as reasoning_content or is null: all render the second turn alike.
    With a template that names no reasoning_content, both keep the turn whole instead.
    Under the Qwen 3.8 template of the chat fixture, which shows earlier reasoning only from reasoning_content, chat's two turns are the reference's renders with the reply split, the second beginning with the first, and the server's second turn is the same.
    The render shows in the prompt's token count, a token a byte in this vocabulary: chat's for each turn, which starts no prefix of the last so each is read whole, and the server's for the same conversation, which /v1/tokenize counts alike."""
    f32.write_model(model, reasoning_tensors(), REASONING_TEMPLATE, f32.VOCAB - 1, config=REASONING_CONFIG)
    reply = REASONING_REPLY.decode()
    first = [{"role": "system", "content": ""}, {"role": "user", "content": "a"}]
    second = first + [{"role": "assistant", "content": "o", "reasoning_content": "qr"}, {"role": "user", "content": "b"}]
    p = common.run_process(["chat", model, "--system", "", "--temp", "0", "-n", "16", "--verbose"], input=b"a\nb\n", timeout=60)
    assert p.returncode == 0, p.stderr
    assert common.cli_stdout(p.stdout).endswith(b"\n" + REASONING_REPLY + b"\n" + REASONING_REPLY + b"\n"), p.stdout
    counts = [int(x) for x in re.findall(rb"Processing (\d+) prompt tokens", p.stderr)]
    assert counts == [len(reasoning_render(first)), len(reasoning_render(second))], (counts, reasoning_render(second))
    srv = Server(model)
    try:
        status, got = srv.post("/v1/chat", {"messages": first, "max_tokens": 16, "temperature": 0})
        assert status == 200 and got["text"] == reply and got["prompt_tokens"] == counts[0], got
        raw = first + [{"role": "assistant", "content": reply}, {"role": "user", "content": "b"}]
        null = first + [{"role": "assistant", "content": reply, "reasoning_content": None}, {"role": "user", "content": "b"}]
        for messages in (raw, null):
            status, got = srv.post("/v1/chat", {"messages": messages, "max_tokens": 16, "temperature": 0})
            assert status == 200 and got["text"] == reply and got["prompt_tokens"] == counts[1], (messages, got)
        status, got = srv.post("/v1/chat/completions", {"messages": second, "max_tokens": 16, "temperature": 0})
        assert status == 200 and got["choices"][0]["message"]["content"] == reply and "reasoning_content" not in got["choices"][0]["message"], got
        assert got["usage"]["prompt_tokens"] == counts[1], got
        # /v1/tokenize renders messages as the chat routes do, so it counts the same prompt for each form of the turn.
        for messages in (raw, null, second):
            status, got = srv.post("/v1/tokenize", {"messages": messages})
            assert status == 200 and got["count"] == counts[1], (messages, got)
        # A reasoning_content that is not a string is refused, and so is a conversation the template raises on, with its message, by the chat routes and by /v1/tokenize.
        bad = first + [{"role": "assistant", "content": "o", "reasoning_content": 5}, {"role": "user", "content": "b"}]
        for path in ("/v1/chat", "/v1/tokenize"):
            status, err = srv.post(path, {"messages": bad, "max_tokens": 4})
            assert status == 400 and "reasoning_content" in err["error"], (path, err)
        raised = first + [{"role": "assistant", "content": "o"}]
        status, err = srv.post("/v1/chat/completions", {"messages": raised, "max_tokens": 4})
        assert status == 400 and err["error"]["message"].endswith("The last message must be the user's."), err
        status, err = srv.post("/v1/tokenize", {"messages": raised})
        assert status == 400 and err["error"].endswith("The last message must be the user's."), err
    finally:
        srv.close()
    f32.write_model(model, reasoning_tensors(), PLAIN_TEMPLATE, f32.VOCAB - 1, config=REASONING_CONFIG)
    whole = first + [{"role": "assistant", "content": reply}, {"role": "user", "content": "b"}]
    p = common.run_process(["chat", model, "--system", "", "--temp", "0", "-n", "16", "--verbose"], input=b"a\nb\n", timeout=60)
    assert p.returncode == 0, p.stderr
    counts = [int(x) for x in re.findall(rb"Processing (\d+) prompt tokens", p.stderr)]
    assert counts == [len(reasoning_render(first)), len(reasoning_render(whole))], (counts, reasoning_render(whole))
    srv = Server(model)
    try:
        status, got = srv.post("/v1/chat", {"messages": whole, "max_tokens": 16, "temperature": 0})
        assert status == 200 and got["text"] == reply and got["prompt_tokens"] == counts[1], got
    finally:
        srv.close()
    # A template that opens the reply inside <think>: the compatible chat route gives the text before </think> as reasoning_content and the rest as content, whole and streamed, while the native route keeps the text whole; the plain template above gave the text whole as content.
    f32.write_model(model, reasoning_tensors(), OPEN_TEMPLATE, f32.VOCAB - 1, config=REASONING_CONFIG)
    srv = Server(model)
    try:
        body = {"messages": first, "max_tokens": 16, "temperature": 0}
        status, got = srv.post("/v1/chat/completions", body)
        message = got["choices"][0]["message"]
        assert status == 200 and message == split_reply(reply, opened=True) == {"role": "assistant", "reasoning_content": "qr", "content": "o"}, got
        events = srv.stream("/v1/chat/completions", dict(body, stream=True))
        deltas = [e["choices"][0]["delta"] for e in events if e and e.get("choices")]
        assert deltas[0]["role"] == "assistant" and events[-1] is None, events
        assert "".join(d.get("reasoning_content", "") for d in deltas) == "qr" and "".join(d.get("content", "") for d in deltas) == "o", deltas
        status, got = srv.post("/v1/chat", body)
        assert status == 200 and got["text"] == reply, got
        # chat_template_kwargs reach the template: with enable_thinking false the prompt closes the <think>, so the whole reply is content; the native chat route and /v1/tokenize render the same prompt.
        off = dict(body, chat_template_kwargs={"enable_thinking": False})
        status, got = srv.post("/v1/chat/completions", off)
        assert status == 200 and got["choices"][0]["message"] == split_reply(reply) == {"role": "assistant", "content": reply}, got
        closed = len(reasoning_render(first).replace("|assistant:\n", "|assistant:<think>\n\n</think>\n\n"))
        assert got["usage"]["prompt_tokens"] == closed, (got, closed)
        status, got = srv.post("/v1/tokenize", {"messages": first, "chat_template_kwargs": {"enable_thinking": False}})
        assert status == 200 and got["count"] == closed, got
        for bad, text in (({"chat_template_kwargs": [1]}, "chat_template_kwargs must be an object"),
                          ({"chat_template_kwargs": {"messages": []}}, "chat_template_kwargs cannot set messages"),
                          ({"chat_template_kwargs": {"x": {"y": 1}}}, "chat_template_kwargs values must be")):
            status, err = srv.post("/v1/chat/completions", dict(body, **bad))
            assert status == 400 and text in err["error"]["message"], (bad, err)
    finally:
        srv.close()
    with open(os.path.join(os.path.dirname(__file__), "data", "baseline_chat_template.json"), encoding="utf-8") as f:
        fixture = json.load(f)
    [qwen38] = [t for t in fixture["templates"] if t["name"] == "Qwen3.8"]
    assert qwen38["split_turns"] and fixture["turn_before"] == first and fixture["turn_after"] == second[-1:]
    expected = [qwen38["first"]["expected"], qwen38["turns"][fixture["turn_texts"].index(reply)]["expected"]]
    assert expected[1].startswith(expected[0]), expected
    f32.write_model(model, reasoning_tensors(), qwen38["template"], f32.VOCAB - 1, config=REASONING_CONFIG)
    p = common.run_process(["chat", model, "--system", "", "--temp", "0", "-n", "16", "--verbose"], input=b"a\nb\n", timeout=60)
    assert p.returncode == 0, p.stderr
    assert common.cli_stdout(p.stdout).endswith(b"\n" + REASONING_REPLY + b"\n" + REASONING_REPLY + b"\n"), p.stdout
    counts = [int(x) for x in re.findall(rb"Processing (\d+) prompt tokens", p.stderr)]
    assert counts == [len(x.encode()) for x in expected], (counts, expected)
    srv = Server(model)
    try:
        for messages in (whole, second):
            status, got = srv.post("/v1/chat", {"messages": messages, "max_tokens": 16, "temperature": 0})
            assert status == 200 and got["text"] == reply and got["prompt_tokens"] == counts[1], (messages, got)
            status, got = srv.post("/v1/tokenize", {"messages": messages})
            assert status == 200 and got["count"] == counts[1], (messages, got)
    finally:
        srv.close()


# The tokens each position of a logprobs request lists beside its sampled one.
TOP = 5


def volatile(text):
    """A reply with its id and timings blanked, the only bytes in which two runs of one compatible request may differ."""
    text = re.sub(r'"id":"(chat)?cmpl-[0-9]+"', '"id":""', text)
    return re.sub(r'"timings":\{[^{}]*\}', '"timings":{}', text)


def strip_logprobs(value):
    """A parsed reply or event without the fields logprobs add."""
    if isinstance(value, dict):
        return {k: strip_logprobs(v) for k, v in value.items() if k not in ("logprobs", "top_logprobs", "logprob")}
    if isinstance(value, list):
        return [strip_logprobs(v) for v in value]
    return value


def check_top(chosen, top, what):
    """A position's most likely tokens under greedy: TOP of them, most likely first, the first being the sampled token with its value."""
    values = [t["logprob"] for t in top]
    assert len(top) == TOP and values == sorted(values, reverse=True) and all(v <= 0 for v in values), (what, top)
    assert chosen["logprob"] == max(values), (what, chosen, top)


def logprob_cases(prompt, n, chat):
    """Each logprobs route: its path, a greedy body that asks for none, the fields that ask for them and fields that ask for none by name."""
    ask = {"logprobs": True, "top_logprobs": TOP}
    cases = [("/v1/generate", {"prompt": prompt, "max_tokens": n, "temperature": 0}, ask, {"logprobs": False, "top_logprobs": 0}),
             ("/v1/completions", {"prompt": prompt, "max_tokens": n, "temperature": 0}, {"logprobs": TOP}, {"logprobs": None})]
    if chat:
        messages = [{"role": "user", "content": prompt}]
        cases += [("/v1/chat", {"messages": messages, "max_tokens": n, "temperature": 0}, ask, {"logprobs": None}),
                  ("/v1/chat/completions", {"messages": messages, "max_tokens": n, "temperature": 0}, ask, {"logprobs": False})]
    return cases


def logprob_values(path, reply):
    """The sampled tokens' values of a whole reply, in order, whatever the route's shape."""
    if path in ("/v1/generate", "/v1/chat"):
        return reply["logprobs"]
    if path == "/v1/completions":
        return reply["choices"][0]["logprobs"]["token_logprobs"]
    return [e["logprob"] for e in reply["choices"][0]["logprobs"]["content"]]


def check_logprob_shape(path, reply, ids):
    """A whole reply's logprobs in its route's shape, one entry per sampled token, greedy's token the most likely at each."""
    if path in ("/v1/generate", "/v1/chat"):
        assert len(reply["logprobs"]) == len(reply["top_logprobs"]) == len(ids), reply
        for i, (value, top) in enumerate(zip(reply["logprobs"], reply["top_logprobs"])):
            check_top({"logprob": value}, top, (path, i))
            assert top[0]["id"] == ids[i], (path, i, top, ids[i])
        return
    choice = reply["choices"][0]
    count = reply["usage"]["completion_tokens"]
    if path == "/v1/completions":
        lp = choice["logprobs"]
        text = choice["text"]
        assert len(lp["tokens"]) == len(lp["token_logprobs"]) == len(lp["top_logprobs"]) == len(lp["text_offset"]) == count, lp
        for i, (value, top) in enumerate(zip(lp["token_logprobs"], lp["top_logprobs"])):
            # The listed tokens and the sampled one, which greedy's always is.
            assert len(top) <= TOP + 1 and lp["tokens"][i] in top and top[lp["tokens"][i]] == value == max(top.values()), (i, value, top)
        offsets = lp["text_offset"]
        assert offsets == sorted(offsets) and (not offsets or offsets[0] == 0) and all(o <= len(text) for o in offsets), offsets
        if not any(t.startswith("bytes:") for t in lp["tokens"]):
            assert "".join(lp["tokens"]) == text, (lp["tokens"], text)
            assert all(text[o:].startswith(t) for o, t in zip(offsets, lp["tokens"])), (offsets, lp["tokens"])
        return
    assert set(choice["logprobs"]) == {"content", "refusal"} and choice["logprobs"]["refusal"] is None, choice["logprobs"]
    content = choice["logprobs"]["content"]
    assert len(content) == count, content
    for i, entry in enumerate(content):
        assert set(entry) == {"token", "logprob", "bytes", "top_logprobs"}, entry
        assert all(set(t) == {"token", "logprob", "bytes"} for t in entry["top_logprobs"]), entry
        check_top(entry, entry["top_logprobs"], (path, i))
        assert entry["top_logprobs"][0]["bytes"] == entry["bytes"], entry
    # The tokens are the whole reply, its reasoning included, so they give the message it splits into.
    whole = b"".join(bytes(e["bytes"]) for e in content)
    try:
        assert split_reply(whole.decode("utf-8")) == choice["message"], (whole, choice["message"])
    except UnicodeDecodeError:
        pass  # a reply ending inside a character holds bytes its text replaced


def stream_logprobs(path, events):
    """A stream's logprobs gathered as a whole reply holds them: the native values and top lists, the completions route's lists, or the chat route's content."""
    tokens = [e for e in events if e and "choices" not in e and "id" in e]
    if path in ("/v1/generate", "/v1/chat"):
        return {"logprobs": [e["logprob"] for e in tokens], "top_logprobs": [e["top_logprobs"] for e in tokens]}
    chunks = [e for e in events if e and e.get("choices")]
    lists = [c["choices"][0]["logprobs"] for c in chunks]
    # A chunk that carries no token carries null.
    carried = [lp for lp in lists if lp is not None]
    if path == "/v1/completions":
        assert all(len(lp["tokens"]) == 1 for lp in carried), carried
        return {k: [v for lp in carried for v in lp[k]] for k in ("tokens", "token_logprobs", "top_logprobs", "text_offset")}
    assert all(len(lp["content"]) == 1 and lp["refusal"] is None for lp in carried), carried
    return {"content": [e for lp in carried for e in lp["content"]], "refusal": None}


def check_logprobs(srv, prompts, n, chat):
    """Logprobs on each route, whole and streamed.
    A reply that does not ask for them, or asks for none by name, is byte-identical to the reply of the request that never names them, apart from a compatible reply's id and timings.
    A reply that asks has that reply's ids and text, repeats byte for byte, lists each position's most likely tokens with greedy's token first, and the stream carries the whole reply's values.
    Every route gives one prompt the same values, and each prompt the values it gets alone when four run at once."""
    by_route = {}
    for path, base, ask, off in logprob_cases(prompts[0], n, chat):
        plain = volatile(srv.raw(path, base))
        assert volatile(srv.raw(path, dict(base, **off))) == plain, (path, off)
        first = volatile(srv.raw(path, dict(base, **ask)))
        assert volatile(srv.raw(path, dict(base, **ask))) == first, (path, "values differ run to run")
        reply = json.loads(first)
        assert strip_logprobs(reply) == json.loads(plain), (path, reply)
        check_logprob_shape(path, reply, reply.get("ids"))
        by_route[path] = logprob_values(path, reply)

        # Streamed, the same bytes without logprobs, and with them the whole reply's values chunk by chunk.
        plain_events = [volatile(p) for p in srv.stream_raw(path, dict(base, stream=True))]
        assert [volatile(p) for p in srv.stream_raw(path, dict(base, stream=True, **off))] == plain_events, (path, "stream", off)
        events = [None if p == "[DONE]" else json.loads(volatile(p)) for p in srv.stream_raw(path, dict(base, stream=True, **ask))]
        assert [strip_logprobs(e) for e in events] == [None if p == "[DONE]" else json.loads(p) for p in plain_events], (path, "stream")
        gathered = stream_logprobs(path, events)
        if path in ("/v1/generate", "/v1/chat"):
            assert gathered == {"logprobs": reply["logprobs"], "top_logprobs": reply["top_logprobs"]}, (path, gathered)
        else:
            assert gathered == reply["choices"][0]["logprobs"], (path, gathered, reply["choices"][0]["logprobs"])
    # The completions route reads the prompt /v1/generate reads and the chat routes render the same one, so their values are the same.
    assert by_route["/v1/completions"] == by_route["/v1/generate"], by_route
    if chat:
        assert by_route["/v1/chat/completions"] == by_route["/v1/chat"], by_route

    # The most a request may list, and the refusals, in each route's error shape.
    status, reply = srv.post("/v1/generate", {"prompt": prompts[0], "max_tokens": 2, "temperature": 0, "logprobs": True, "top_logprobs": 20})
    assert status == 200 and all(len(t) == 20 for t in reply["top_logprobs"]), (status, reply)
    for body, field in (({"logprobs": True, "top_logprobs": 21}, "top_logprobs"), ({"logprobs": True, "top_logprobs": -1}, "top_logprobs"),
                        ({"logprobs": "yes"}, "logprobs"), ({"top_logprobs": 3}, "top_logprobs")):
        status, err = srv.post("/v1/generate", dict({"prompt": "a", "max_tokens": 2}, **body))
        assert status == 400 and field in err["error"], (body, status, err)
    for body in ({"logprobs": 21}, {"logprobs": True}, {"logprobs": 1.5}):
        status, err = srv.post("/v1/completions", dict({"prompt": "a", "max_tokens": 2}, **body))
        assert status == 400 and "logprobs" in err["error"]["message"], (body, status, err)
    status, err = srv.post("/v1/chat/completions", {"messages": [{"role": "user", "content": "a"}], "max_tokens": 2, "top_logprobs": 2})
    assert status == 400 and "top_logprobs" in err["error"]["message"], (status, err)

    # A seeded draw: the completions route's map holds the sampled token's text at every position, listed or not, with one token listed and with none, and every route gives the draw's values, the native shape with no top_logprobs when none are asked for.
    drawn = {"prompt": prompts[0], "max_tokens": n, "temperature": 1.5, "penalty": 1.3, "seed": 11}
    status, native = srv.post("/v1/generate", dict(drawn, logprobs=True, top_logprobs=1))
    assert status == 200, native
    status, bare = srv.post("/v1/generate", dict(drawn, logprobs=True, top_logprobs=0))
    assert status == 200 and bare["ids"] == native["ids"] and bare["logprobs"] == native["logprobs"] and "top_logprobs" not in bare, bare
    for k in (1, 0):
        status, reply = srv.post("/v1/completions", dict(drawn, logprobs=k))
        assert status == 200, (k, reply)
        lp = reply["choices"][0]["logprobs"]
        assert lp["token_logprobs"] == native["logprobs"], (k, lp["token_logprobs"], native["logprobs"])
        for i, (token, value, top) in enumerate(zip(lp["tokens"], lp["token_logprobs"], lp["top_logprobs"])):
            assert top.get(token) == value and len(top) <= k + 1, (k, i, token, value, top)

    # Four at a time, where a pass holds one request's prompt rows beside another's decode rows, each request's values are the ones it gets alone.
    ask = {"max_tokens": n, "temperature": 0, "logprobs": True, "top_logprobs": TOP}
    alone = {p: srv.post("/v1/generate", dict(ask, prompt=p))[1] for p in prompts[:4]}
    results = {}
    def worker(p):
        results[p] = srv.post("/v1/generate", dict(ask, prompt=p))
    threads = [threading.Thread(target=worker, args=(p,)) for p in prompts[:4]]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for p in prompts[:4]:
        status, reply = results[p]
        assert status == 200 and reply == alone[p], (p, reply, alone[p])


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


def check_mxfp4(directory):
    """MXFP4 server replies against HF at the first token and against runs alone thereafter, without reuse or pauses."""
    with open(os.path.join(os.path.dirname(__file__), "data", "baseline_mxfp4.json"), encoding="utf-8") as f:
        goldens = json.load(f)
    assert [x["name"] for x in goldens["fixtures"]] == [x[0] for x in mxfp4.VARIANTS]
    worst = agreement = 0.0
    used_bounds = set()
    for (name, tied, routed), golden in zip(mxfp4.VARIANTS, goldens["fixtures"]):
        config, weights, packed = mxfp4.fixture(tied, routed)
        assert config == golden["config"] and mxfp4.weight_hash(weights) == golden["weights_sha256"]
        assert mxfp4.packed_hash(packed) == golden["packed_sha256"]
        model = mxfp4.write_fixture(os.path.join(directory, "server-mxfp4-" + name + ".gguf"), config, weights, packed, routed)
        if mxfp4.device_skip(model):
            print("server: SKIP MXFP4 - selected device has no MXFP4 kernel")
            return
        srv = Server(model)
        try:
            bodies, alone = [], []
            for case in golden["cases"]:
                body = dict(prompt=case["text"], max_tokens=3, temperature=0, ignore_eos=True, logprobs=True, top_logprobs=5)
                reply = post_ok(srv, "/v1/generate", body)
                scores = case["logits"]
                out, bounds = common.run_hf(["logits", model, case["text"], "--top", str(len(scores))])
                logits = dict(zip(*common.parse_logits(out)))
                common.hf_logit_error("server CLI " + name, logits, scores, bound=bounds["logit"])
                agreement = max(agreement, common.check_logprob_row(reply["top_logprobs"][0], logits))
                used_bounds.add(2 * bounds["logit"])
                top = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
                assert reply["ids"][0] == top[0], (name, case["text"], reply["ids"], top)
                observed = [entry["id"] for entry in reply["top_logprobs"][0]]
                assert common.top5_overlap(observed, top, [scores[i] for i in top], margin=2 * bounds["logit"]) == 5, (name, observed, top[:5])
                shift = max(scores)
                normalizer = math.log(math.fsum(math.exp(x - shift) for x in scores))
                for entry in reply["top_logprobs"][0]:
                    error = abs(entry["logprob"] - (scores[entry["id"]] - shift - normalizer))
                    assert error < 2 * bounds["logit"], (name, case["text"], entry, error)
                    worst = max(worst, error)
                assert len(reply["top_logprobs"][0]) == 5 and len(reply["ids"]) == 3, reply
                assert reply["logprobs"][0] == reply["top_logprobs"][0][0]["logprob"], reply
                bodies.append(body)
                alone.append(reply)
            results = {}
            barrier = threading.Barrier(len(bodies))
            def worker(i):
                barrier.wait(timeout=30)
                results[i] = srv.post("/v1/generate", bodies[i])
            threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(bodies))]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            for i, body in enumerate(bodies):
                status, reply = results[i]
                assert status == 200, reply
                for key in ("ids", "logprobs", "top_logprobs"):
                    assert reply[key] == alone[i][key], (name, i, key, reply, alone[i])
                events = srv.stream("/v1/generate", dict(body, stream=True))
                assert [e["id"] for e in events if e and "id" in e] == alone[i]["ids"]
                assert stream_logprobs("/v1/generate", events) == {k: alone[i][k] for k in ("logprobs", "top_logprobs")}
                assert events[-1] is None and events[-2].get("done") is True, events[-2:]
            health = srv.wait(lambda h: h["active"] == 0, "MXFP4 request stayed active")
            assert health["prefix_hits"] == 0 and health["pauses"] == 0, health
        finally:
            srv.close()
    print("server: MXFP4 dense tied/untied and MoE, first-token HF logprobs (max error %.8f, witnessed bounds %s), "
          "server/CLI logprob error %.9f, concurrent and streamed replies exact, no donor hits or pauses  [ok]"
          % (worst, sorted(used_bounds), agreement))


def check_seeded(model):
    """Seeded requests on each of the sampler's four paths, four at a time and the other settings at their defaults, give the text `generate` gives alone with the same settings and seed, as the server writes it: the default top-k with its nucleus, the default top-k without one, a nucleus over the whole vocabulary, and every token kept."""
    cases = [({}, []), ({"top_k": 40, "top_p": 1.0}, ["--topk", "40", "--topp", "1"]),
             ({"top_k": 0, "top_p": 0.95}, ["--topk", "0", "--topp", "0.95"]),
             ({"top_k": 0, "top_p": 1.0}, ["--topk", "0", "--topp", "1"])]
    prompt, n, seed = "Once upon a time", 16, 7
    srv = Server(model)
    results = {}
    try:
        def worker(i):
            results[i] = srv.post("/v1/generate", dict({"prompt": prompt, "max_tokens": n, "seed": seed}, **cases[i][0]))
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(cases))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        srv.close()
    for i, (fields, sampling) in enumerate(cases):
        status, reply = results[i]
        assert status == 200, reply
        want = repaired(cli_reply(model, prompt, n, ["--seed", str(seed)] + sampling)[0])
        assert reply["text"] == want, (fields, reply["text"], want)


def check_stream_reuse(directory):
    """With the experts on the host and prompts of 100 tokens or more streamed, a 120-token prompt that forks a finished prompt's first block gives the ids and values it gives on a server of its own: its path follows the whole prompt, not the 56 tokens the fork leaves to read."""
    model = os.path.join(directory, "tiny-moe-192.gguf")
    f32.write_model(model, moe.tensors(), config=dict(moe.CONFIG, context_length=192), arch="qwen3moe")
    flags = ("--cpu-moe", "--moe-stream-from", "100")
    # 64 shared tokens, one block of the device's cache, then 56 of each prompt's own.
    shared = "".join(chr(ord("a") + i % 26) for i in range(64))
    tail = "".join(chr(ord("A") + i * 7 % 26) for i in range(56))
    first, second = shared + tail[::-1], shared + tail
    body = lambda p: {"prompt": p, "max_tokens": 8, "temperature": 0, "logprobs": True, "top_logprobs": TOP}
    srv = Server(model, *flags)
    try:
        status, alone = srv.post("/v1/generate", body(second))
        assert status == 200 and alone["reused_tokens"] == 0, alone
    finally:
        srv.close()
    srv = Server(model, *flags)
    try:
        for p in (first, second):
            status, reply = srv.post("/v1/generate", body(p))
            assert status == 200, reply
        assert reply["reused_tokens"] == 64, reply
        for key in ("ids", "logprobs", "top_logprobs"):
            assert reply[key] == alone[key], (key, reply[key], alone[key])
    finally:
        srv.close()


def check_limits(model):
    """The serving limits: a KV budget below the context bounds a request, and a full queue refuses with 503 rather than waiting, while the tokenize routes, which pass no queue, still answer and count a text past the context."""
    srv = Server(model, "--max-seqs", "1", "--max-queue", "1", "--ctx-size", "512")
    try:
        assert srv.post("/v1/generate", {"prompt": "a", "max_tokens": 600})[0] == 413
        long_text = " ".join(["the"] * 600)
        long_ids = cli_tokenize(model, long_text)
        assert len(long_ids) > 512, len(long_ids)
        results = {}
        def worker(i):
            results[i] = srv.post("/v1/generate", {"prompt": "The capital of France is", "max_tokens": 300,
                                                   "temperature": 0})
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
            time.sleep(0.2)
        # The one slot and the one queue place are held from before the tokenize requests to after them.
        full = srv.get("/v1/health")
        tokenized = srv.post("/v1/tokenize", {"text": long_text})
        detokenized = srv.post("/v1/detokenize", {"tokens": long_ids})
        after = srv.get("/v1/health")
        for t in threads:
            t.join()
        codes = sorted(status for status, _ in results.values())
        assert codes == [200, 200, 503], codes
        assert [r for s, r in results.values() if s == 503][0]["error"], results
        assert full["active"] == 1 and full["queued"] == 1 and after["active"] == 1 and after["queued"] == 1, (full, after)
        assert tokenized == (200, {"tokens": long_ids, "count": len(long_ids)}), tokenized
        assert detokenized == (200, {"text": long_text}), detokenized
    finally:
        srv.close()


def check_unencodable(directory):
    """A model whose vocabulary lacks the byte token `q`: /v1/tokenize refuses a text holding it with the 400 and the message the generating routes give, as a text and as messages, and `llmx tokenize` fails on it."""
    tokens = build_byte_vocab() + ["<|endoftext|>"]
    tokens[ord("q")] = "qq"
    model = f32.write_model(os.path.join(directory, "tiny-f32-no-q.gguf"), f32.tensors(True), tokens=tokens)
    p = common.run_process(["tokenize", model, "q"])
    assert p.returncode != 0 and b"not in vocab" in p.stderr, (p.returncode, p.stderr)
    srv = Server(model)
    try:
        messages = [{"role": "user", "content": "q"}]
        replies = [srv.post(path, body) for path, body in (("/v1/tokenize", {"text": "q"}),
                                                           ("/v1/generate", {"prompt": "q", "max_tokens": 1}),
                                                           ("/v1/tokenize", {"messages": messages}),
                                                           ("/v1/chat", {"messages": messages, "max_tokens": 1}))]
        assert all(status == 400 for status, _ in replies) and len({err["error"] for _, err in replies}) == 1, replies
    finally:
        srv.close()


def same_choice(alone, got, what):
    """A compatible reply's choice against the same request's alone: its text, its finish and every token's values, the first differing token named."""
    a, b = alone["logprobs"], got["logprobs"]
    first = next((i for i, pair in enumerate(zip(zip(a["tokens"], a["token_logprobs"], a["top_logprobs"]),
                                                 zip(b["tokens"], b["token_logprobs"], b["top_logprobs"]))) if pair[0] != pair[1]), None)
    assert first is None and got == alone, "%s: token %s of %d differs from the request alone" % (what, first, len(a["tokens"]))


def check_uncapped(model):
    """Uncapped requests share a small pool: each reserves its prompt and grows, a request is paused when the pool runs out and resumes from its history, and every one runs to its own end with the tokens and log-probabilities it gives alone, where it never pauses."""
    srv = Server(model, "--max-seqs", "4", "--ctx-size", "1024")
    try:
        prompts = ["The capital of France is", "Once upon a time", "def fib(n):"]
        body = lambda p: {"prompt": p, "temperature": 0, "logprobs": TOP}
        alone = {}
        for p in prompts:
            status, reply = srv.post("/v1/completions", body(p), timeout=600)
            assert status == 200, reply
            alone[p] = reply["choices"][0]
        before = srv.get("/v1/health")
        results = {}
        def worker(p):
            results[p] = srv.post("/v1/completions", body(p), timeout=600)
        threads = [threading.Thread(target=worker, args=(p,)) for p in prompts]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for p in prompts:
            status, reply = results[p]
            assert status == 200 and reply["choices"][0]["finish_reason"] in ("stop", "length"), (p, reply)
            assert reply["usage"]["total_tokens"] <= 1024, (p, reply)
            same_choice(alone[p], reply["choices"][0], "an uncapped request paused beside others, %r" % p)
        health = srv.get("/v1/health")
        assert health["active"] == 0 and health["queued"] == 0 and health["paused"] == 0, health
        assert health["pauses"] > before["pauses"], (before, health)
        # Resumes take device donors back, promote host donors or recompute missing rows; count only this concurrent group.
        assert any(health[key] > before[key] for key in ("recomputed", "taken_back", "host_hits")), (before, health)
    finally:
        srv.close()


def check_paused_prefill(model):
    """A long uncapped prompt read one token a pass is still prefilling when an earlier uncapped request has to grow and the pool has no room, so it is paused part-way; resumed, its greedy text is the CLI's for the whole prompt and its values those it gives alone."""
    flags = ("--ubatch", "1")
    # 464 tokens in a pool of 1280: in blocks of 64 or 128 both requests fit at admission, the short one reaches its first growth step before this prompt is read, and that step does not fit beside it.
    with open(os.path.join(os.path.dirname(__file__), "data", "wiki.test.raw"), encoding="utf-8") as f:
        long_prompt = f.read()[:1800]
    # A stop string ends the long request a few tokens in, enough to tell which prompt it resumed from.
    body = {"prompt": long_prompt, "temperature": 0, "stop": [" ."], "logprobs": TOP}
    # Alone on a server of its own, so no donor of this run's is there for the paused request to fork.
    srv = Server(model, "--ctx-size", "1280", *flags)
    try:
        status, alone = srv.post("/v1/completions", body, timeout=900)
        assert status == 200, alone
    finally:
        srv.close()
    srv = Server(model, "--ctx-size", "1280", *flags)
    try:
        # The short request streams and is queued first, so the long one is the latest admitted and the one paused.
        s = srv.open("/v1/completions", {"prompt": "Once upon a time", "temperature": 0, "stream": True})
        s.recv(64)
        result = []
        failure = []
        def worker():
            try:
                result.append(srv.post("/v1/completions", body, timeout=900))
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
        same_choice(alone["choices"][0], reply["choices"][0], "a prompt paused while prefilling")
        health = srv.get("/v1/health")
        assert health["active"] == 0 and health["pauses"] >= 1, health
    finally:
        srv.close()


def check_conversation(model, text):
    """A conversation whose history grows past half of a small pool: each follow-up repeats the last turn's prompt and reply and adds half of `text`, and forks the last turn's prompt however full the pool is but never its reply, whose rows decode computed, so its `reused_tokens` stays within the last turn's prompt and, once the last turn's prompt passes 512 tokens, where a device's tile takes one split whatever the length, grows with the server's `prefix_tokens` on every follow-up, and it gives the CLI's greedy text for its whole prompt."""
    pool, n, turns = 1024, 16, 6
    srv = Server(model, "--ctx-size", str(pool))
    try:
        words = text.split(" ")
        halves = [" ".join(words[:len(words) // 2]), " ".join(words[len(words) // 2:])]
        prompt, reused, prefix, last = halves[0], 0, 0, 0
        for turn in range(turns):
            status, reply = srv.post("/v1/generate", {"prompt": prompt, "max_tokens": n, "temperature": 0})
            assert status == 200, reply
            assert reply["text"] == cli_greedy_text(model, prompt, n), (turn, reply["text"])
            health = srv.get("/v1/health")
            if turn:
                assert reply["reused_tokens"] <= last, (turn, reply["reused_tokens"], last)
            if last >= 512:
                assert reply["reused_tokens"] > reused and health["prefix_tokens"] > prefix, (turn, reply["prompt_tokens"], reply["reused_tokens"], reused, health)
            reused, prefix, last = reply["reused_tokens"], health["prefix_tokens"], reply["prompt_tokens"]
            prompt += reply["text"] + " " + halves[(turn + 1) % 2]
        assert reply["prompt_tokens"] > pool // 2, reply
        return turns
    finally:
        srv.close()


def check_unrelated_donor(model):
    """A follow-up turn beside an unrelated donor, on a pool with room for the follow-up once the turn it repeats is consumed but not beside both donors: the follow-up consumes that turn's history rather than evicting the unrelated donor, so a later prompt repeating the unrelated request's history still reuses it, with the CLI's greedy text."""
    pool, n = 1344, 16
    srv = Server(model, "--ctx-size", str(pool))
    try:
        with open(os.path.join(os.path.dirname(__file__), "data", "wiki.test.raw"), encoding="utf-8") as f:
            text = f.read()
        # Every prompt passes 449 tokens, so on a device each takes the tile split the others do and a fork may take its rows.
        # The unrelated request keeps 495 tokens and the first turn 528, and the follow-up's prompt and max_tokens, about 805, of which it shares 512 with the first turn, fit beside one of them but not beside both: 11 blocks of 128 and 21 of 64 against 12 and 22.
        unrelated, first = text[4000:6000], text[:2000]
        replies = []
        for prompt in (unrelated, first):
            status, reply = srv.post("/v1/generate", {"prompt": prompt, "max_tokens": n, "temperature": 0})
            assert status == 200 and reply["reused_tokens"] == 0, reply
            replies.append(reply)
        status, reply = srv.post("/v1/generate", {"prompt": first + replies[1]["text"] + " " + text[2000:3100], "max_tokens": n, "temperature": 0})
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


def check_reprefill(model):
    """A chat reply read again as prompt rows (docs/SPECULATIVE.md, section 2, Idle re-prefill): after a first turn of about 500 tokens and a 160-token reply without reasoning, the server reads what the next turn begins with while idle, so a follow-up turn on /v1/chat/completions reuses tokens past the first turn's prompt, into its reply, and gives the reply the same request gives on a fresh server; returns the tokens it reused and the first turn's prompt."""
    with open(os.path.join(os.path.dirname(__file__), "data", "wiki.test.raw"), encoding="utf-8") as f:
        text = f.read()[:2000]
    first = {"messages": [{"role": "user", "content": text + "\n\nSummarize this."}], "max_tokens": 160, "temperature": 0,
             "ignore_eos": True, "chat_template_kwargs": {"enable_thinking": False}}
    srv = Server(model)
    try:
        a = post_ok(srv, "/v1/chat/completions", first)
        srv.wait(lambda h: h["reprefills"] >= 1, "the first turn's reply was not read again", 60)
        messages = first["messages"] + [{"role": "assistant", "content": a["choices"][0]["message"]["content"]}, {"role": "user", "content": "Shorter."}]
        follow = dict(first, messages=messages, max_tokens=32)
        b = post_ok(srv, "/v1/chat/completions", follow)
        reused, prompt = b["timings"]["cache_n"], a["usage"]["prompt_tokens"]
        assert reused > prompt, (reused, prompt, srv.get("/v1/health"))
    finally:
        srv.close()
    fresh = Server(model)
    try:
        c = post_ok(fresh, "/v1/chat/completions", follow)
        assert c["timings"]["cache_n"] == 0, c
    finally:
        fresh.close()
    assert b["choices"][0]["message"] == c["choices"][0]["message"], (b["choices"][0], c["choices"][0])
    return reused, prompt


def check_host_tier(model):
    """Donors kept in host memory (docs/SPECULATIVE.md, section 2, Host tier): two conversations of about 500 tokens alternate on a 1024-token pool that holds one finished turn beside the next request, so each turn evicts the other conversation's donor; with the host tier each follow-up promotes its own and reuses its prefix, with the CLI's greedy text, and without it reuses nothing. Returns the tokens the follow-ups reused."""
    with open(os.path.join(os.path.dirname(__file__), "data", "wiki.test.raw"), encoding="utf-8") as f:
        text = f.read()
    n = 16
    reused = {}
    for host in ("1073741824", "0"):
        srv = Server(model, "--ctx-size", "1024", "--host-cache-bytes", host)
        try:
            first = {}
            for key, part in (("a", text[:2000]), ("b", text[4000:6000])):
                first[key] = part + post_ok(srv, "/v1/generate", {"prompt": part, "max_tokens": n, "temperature": 0})["text"]
            total = 0
            for key, more in (("a", text[2000:2400]), ("b", text[6000:6400])):
                prompt = first[key] + " " + more
                reply = post_ok(srv, "/v1/generate", {"prompt": prompt, "max_tokens": n, "temperature": 0})
                assert reply["text"] == cli_greedy_text(model, prompt, n), (host, key, reply["text"])
                total += reply["reused_tokens"]
            health = srv.get("/v1/health")
            if host != "0":
                assert total > 0 and health["host_hits"] >= 2 and health["host_bytes_moved"] > 0, (total, health)
            else:
                assert total == 0 and health["host_donors"] == 0, (total, health)
            reused[host] = total
        finally:
            srv.close()
    return reused["1073741824"]


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
    if not server_mix_tool.run():
        return False
    if common.f32_cache_skip("server"):
        return common.SKIPPED
    with tempfile.TemporaryDirectory(prefix="llmx_server_") as directory:
        # Every reply that names the model carries its file name, so the synthetic model's holds a byte that is not UTF-8 where the file system takes one (Linux), and characters beyond ASCII elsewhere.
        # On Windows a name read in the system code page instead of as UTF-8 fails only where one of its UTF-8 bytes has no mapping there, so U+00E1 brings 0xA1 for code page 1257, U+00E0 brings 0xA0 for 932, and U+4E2D breaks 936, 949 and 950.
        name = "tiny-f32-\udcff.gguf" if sys.platform.startswith("linux") else "tiny-f32-\u00e1\u00e9\u00e0\u4e2d.gguf"
        model = os.path.join(directory, name)
        f32.write_model(model, f32.tensors(True))
        check_dtype(model)
        # The synthetic model's context is 16 tokens: prompt plus tokens stay inside it.
        # One token a byte and no special token: text beyond ASCII splits inside its characters, a special token's text reads as its bytes, and the longest text runs past the 16-token context, which the route counts rather than refuses.
        texts = ["a", "h\u00e9llo w\u00f6rld", "\u65e5\u672c\u8a9e", "\U0001f600", "<|endoftext|>", "<|im_start|>user\nhi<|im_end|>", ""]
        n = check_server(model, ["a", "ab", "abc", "abcdefg"], 6, 14, chat=False, texts=texts)
        print("server: synthetic F32 model, %d prompts greedy-equal to the CLI alone and four at a time, a stream, "
              "a seeded repeat, refusals, the tokenize routes, a cancelled stream, the compatible completions, logprobs on the generate and completions routes, its file name as UTF-8 in every reply  [ok]" % n)
        check_unencodable(directory)
        print("server: synthetic F32 model without the byte token q, a text holding it refused alike by /v1/tokenize, /v1/generate and /v1/chat  [ok]")
        check_reasoning(os.path.join(directory, "reasoning.gguf"))
        print("server: an assistant turn with its reasoning in the content, in reasoning_content or null beside it renders as chat renders its own reply, "
              "and whole in both under a template without reasoning; two chat turns under a Qwen 3.8 template are the reference's renders; "
              "/v1/tokenize counts each as the chat routes read it; a bad reasoning_content and a conversation the template raises on answer 400  [ok]")
        # The synthetic mixture of experts, each prompt's ids alone equal to its ids four at a time, where a pass routes one request's prompt rows beside another's decode rows.
        # On a device its routed layers run on the host with prompts from extent 3 streamed, so a pass holds streamed prompt rows beside host decode rows; those flags need a device, so the CPU runs the model without them.
        # Experts on the host are a placement of one device, so a list of several skips this.
        device = os.environ.get("LLMX_DEVICE", "cpu")
        if "," not in device:
            routed = os.path.join(directory, "tiny-moe.gguf")
            f32.write_model(routed, moe.tensors(), config=moe.CONFIG, arch="qwen3moe")
            host = device != "cpu"
            n = check_mixed(routed, ["a", "ab", "abc", "abcdefg", "abcd", "b"], 6,
                            ("--cpu-moe", "--moe-stream-from", "3") if host else ())
            print("server: synthetic MoE model, %s, %d prompts alone and four at a time  [ok]"
                  % ("experts on the host and long prompts streamed" if host else "on the CPU", n))
            if host:
                check_stream_reuse(directory)
                print("server: synthetic MoE model, experts on the host and long prompts streamed, a prompt forking a finished prompt's block giving its values alone  [ok]")
        check_mxfp4(directory)
        k, n = check_ignore_eos_synthetic(directory)
        print("server: ignore_eos on the synthetic model, a greedy reply that ends at its end token after %d tokens running to %d through the CLI "
              "and the server, greedy and seeded, uncapped to the context, which the CLI given the room also fills, beside requests without it, and refused unless a boolean  [ok]" % (k, n))
    real = baseline.find_fixture(baseline.BASELINE_MODELS[0])
    if real:
        with open(os.path.join(os.path.dirname(__file__), "data", "baseline_perplexity.json"), encoding="utf-8") as f:
            excerpt = json.load(f)["text"]
        # The tokenizer golden's texts: whitespace, digits, text beyond ASCII, emoji, special tokens and a soft hyphen.
        with open(baseline.GOLDEN, encoding="utf-8") as f:
            texts = [case["text"] for case in json.load(f)["cases"]] + [""]
        n = check_server(real, ["The capital of France is", "Once upon a time", "def fib(n):", "The three laws of"],
                         16, 4000, chat=True, texts=texts, prefix=excerpt)
        check_seeded(real)
        check_limits(real)
        check_uncapped(real)
        check_paused_prefill(real)
        turns = check_conversation(real, excerpt)
        promoted = check_host_tier(real)
        print("server: %s, two conversations alternating on a pool that holds one, each follow-up promoting its donor from host memory (%d tokens reused) with the CLI's greedy text, none without the host tier  [ok]"
              % (os.path.basename(real), promoted))
        reused, first = check_reprefill(real)
        print("server: %s, a chat follow-up reusing %d tokens of a %d-token first turn and its reply, read again while idle, with its reply on a fresh server  [ok]"
              % (os.path.basename(real), reused, first))
        check_unrelated_donor(real)
        check_departed(real)
        k, n = check_ignore_eos_real(real)
        print("server: %s, %d prompts greedy-equal to the CLI alone and four at a time, a stream, a seeded repeat, "
              "seeded requests equal to the CLI on four sampler paths, refusals, the tokenize routes, a cancelled stream, a chat turn, the compatible routes, logprobs on every route, a reused prefix, the limits, uncapped requests paused and resumed with their values alone, "
              "a prompt paused while prefilling with its values alone, a %d-turn conversation past half the pool reusing its history on every follow-up, "
              "a follow-up consuming the turn it repeats while an unrelated donor stays, clients leaving a whole reply, a prefill and the queue, and one shutting its sending side  [ok]"
              % (os.path.basename(real), n, turns))
        print("server: ignore_eos on %s, a greedy reply that ends at its end token after %d tokens running to %d through generate, chat "
              "and the four routes, greedy and seeded, a stop text still ending it, beside requests without it  [ok]" % (os.path.basename(real), k, n))
    else:
        print("server: SKIP real-model pass - fixture model not on disk")
    return True
