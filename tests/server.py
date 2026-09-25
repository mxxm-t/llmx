import json
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
from tokenizer import build_byte_vocab

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import server_load  # noqa: E402

# The server of docs/SERVER.md against the CLI on the same file: a greedy request through /v1/generate gives the text `generate --temp 0` gives, alone and while three other requests decode beside it; a streamed request arrives as events with the same ids; a seeded request repeats, and a compatible request's seed of -1 samples as no seed; a bad body, a number its field cannot hold, a sampling field outside the CLI's range and a request past the context are refused; a client that goes away mid-stream, during a whole reply, while its prompt is read or while it waits in the queue leaves the server with nothing active and its blocks free, and one that shuts only its sending side gets no answer; a chat turn renders; /v1/tokenize and /v1/detokenize give the ids and text `llmx tokenize` and `llmx detokenize` give, while the queue is full too, and a text the tokenizer cannot encode is refused there as the generating routes refuse it; a conversation growing past half a small pool reuses its history on every follow-up; a follow-up short of room consumes the turn it repeats and leaves an unrelated donor in place.
# The synthetic F32 model (16-token context) needs no download; the real Q8_0 fixture, when it is on disk, repeats the checks with room to stream.
# The synthetic model's file name holds a byte that is not UTF-8 on Linux and characters beyond ASCII that several Windows code pages cannot map elsewhere, and every reply naming the model must still be UTF-8.
# The synthetic MoE model gives each prompt the same ids alone and four at a time, on the CPU as it is and on a device with its experts on the host.
# With ignore_eos a reply that would end at the model's end token runs to its limit, through the CLI and the server alike, on the synthetic model given an end token and on the real fixture.


class Server:
    def __init__(self, model, *extra):
        # The device and cache flags go on the command, not the executable path, which device_args would not recognise; the server then runs where the CLI it is compared with runs.
        args = ["serve", model, "--max-seqs", "8"] + list(extra)
        self.proc, self.port, self.log = common.start_server([common.exe_path()] + common.device_args(args, "f32"))

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
        with open(os.path.join(os.path.dirname(__file__), "data", "baseline_chat_template.json"), encoding="utf-8") as f:
            cases = [case for case in json.load(f)["cases"] if case["generate"]]
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
        assert reply["choices"][0]["message"]["content"] == chat_full and reply["choices"][0]["finish_reason"] == "length", reply
        alone_and_together(srv, [dict(base, temperature=0, ignore_eos=on) for on in (False, True)] +
                           [dict(base, temperature=0.8, seed=11, ignore_eos=True), dict(base, prompt="The capital of France is", temperature=0, ignore_eos=True)])
    finally:
        srv.close()
    return k, n


def check_server(model, prompts, n, long_n, chat, texts, prefix=None, flags=()):
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
        # One token a byte and no special token: text beyond ASCII splits inside its characters, a special token's text reads as its bytes, and the longest text runs past the 16-token context, which the route counts rather than refuses.
        texts = ["a", "h\u00e9llo w\u00f6rld", "\u65e5\u672c\u8a9e", "\U0001f600", "<|endoftext|>", "<|im_start|>user\nhi<|im_end|>", ""]
        n = check_server(model, ["a", "ab", "abc", "abcdefg"], 6, 14, chat=False, texts=texts)
        print("server: synthetic F32 model, %d prompts greedy-equal to the CLI alone and four at a time, a stream, "
              "a seeded repeat, refusals, the tokenize routes, a cancelled stream, the compatible completions, its file name as UTF-8 in every reply  [ok]" % n)
        check_unencodable(directory)
        print("server: synthetic F32 model without the byte token q, a text holding it refused alike by /v1/tokenize, /v1/generate and /v1/chat  [ok]")
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
        check_limits(real)
        check_uncapped(real)
        check_paused_prefill(real)
        turns = check_conversation(real, excerpt)
        check_unrelated_donor(real)
        check_departed(real)
        k, n = check_ignore_eos_real(real)
        print("server: %s, %d prompts greedy-equal to the CLI alone and four at a time, a stream, a seeded repeat, "
              "refusals, the tokenize routes, a cancelled stream, a chat turn, the compatible routes, a reused prefix, the limits, uncapped requests sharing a pool, "
              "a prompt paused while prefilling, a %d-turn conversation past half the pool reusing its history on every follow-up, "
              "a follow-up consuming the turn it repeats while an unrelated donor stays, clients leaving a whole reply, a prefill and the queue, and one shutting its sending side  [ok]"
              % (os.path.basename(real), n, turns))
        print("server: ignore_eos on %s, a greedy reply that ends at its end token after %d tokens running to %d through generate, chat "
              "and the four routes, greedy and seeded, a stop text still ending it, beside requests without it  [ok]" % (os.path.basename(real), k, n))
    else:
        print("server: SKIP real-model pass - fixture model not on disk")
    return True
