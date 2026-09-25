"""A server under load: the latency and throughput figures a serving runtime is judged by.

Standard library only.
Every request streams its reply, so the arrival of every token is timed, and samples greedily, so the work of a request is fixed.

Load:

    closed loop  --concurrency C ...: at each level C users each send a request and send their next when its reply ends, --num-prompts requests in all (C by default, one a user), repeated --rounds times; the table shows the round with the most output tokens per second among the rounds in which no request failed (among all rounds when every one has a failure), and the notes below it list the failures of every round
    open loop    --rate R ...: --num-prompts requests (100 by default) arrive at R a second as a Poisson process drawn from --seed, each sent at its time whatever is still running, as a reference serving benchmark sends them; inf sends them all at once

With neither flag the closed loop runs at 1, 2, 4, 8, 16, 32 and 64 users; with --rate alone only the rates run.
An open level queues on the server whatever it sends beyond what the server runs, and a server that caps its queue refuses the rest, which count as failures: start `llmx serve` with --max-queue at least --num-prompts (it runs 16 requests and queues 64 by default).

Workload:

    prompts  by default the eight short fixed prompts in turn; with --input-len N every request its own prompt of N tokens, the word "the" and then random words from a fixed list; with --input-len-range A:B the lengths uniform from A to B
    replies  by default up to --tokens N tokens (64), a reply ending at the end of text if the model emits it first; with --output-len N in its place every request asks for exactly N tokens and to ignore the end of text, and a reply that ends short is counted as short

The prompt lengths are drawn from --seed alone, before any prompt is built, so the n-th request of every level and round has the same length on every server.
The words are drawn from --seed and the level and round, so the requests of a run share no prefix beyond the first word but by chance, and a rerun with the same seed repeats them.
The defaults are the earlier load of this tool, and the earlier flags keep their meaning.

A prompt's length counts every token the server reads for it, a start token it adds included.
Where the server has a tokenize route (POST /tokenize, or POST /v1/tokenize as llmx serve has), each prompt is counted there and a word added or dropped until the count is exact.
Otherwise two requests for one token, on the word list once and twice, show whether every word is one token and what a prompt costs beyond its words; when every word is one token, a prompt of k words has a known length.
Where a reply reports its prompt tokens, the achieved lengths are checked against the target and the mismatches reported.

--output-len sends the cap with ignore_eos set on every route: max_tokens on llmx's /v1/generate and on the OpenAI route, n_predict on the reference server's /completion.
llmx's routes honour ignore_eos; a server that does not can still end a reply at its end of text, which the short count shows.

The APIs: --api llmx speaks /v1/generate, --api openai /v1/completions (with the model id /v1/models lists, or --model), and --api completion the reference server's /completion route, so the same load compares servers on the same model and card.
A reply's tokens are the count it reports (usage on the OpenAI route, tokens_predicted on /completion, the done event on /v1/generate), so an event carrying several tokens counts them all and an event carrying only text held back at the end, a character split across tokens, counts none; a reply that reports no count has one token an event.

Figures per level, over the completed requests:

    tok/s        output tokens per second, from the level's first send to its last reply's end
    req/s        completed requests per second over the same span
    ttft         time to first token, from a request's send, its connection included, to its first streamed token: mean, median, 90th and 99th percentile
    itl          inter-token latency, the gap between consecutive token events of a request, over all requests: median and 99th percentile
    all tok/s    prompt and output tokens per second over the level's span
    tpot         time per output token after the first, per request (end - first token) / (tokens - 1), the end being the reply's last event: mean, median and 99th percentile
    e2e          end-to-end latency, from a request's send to its reply's last event: median and 99th percentile
    in, out      mean prompt and output tokens of a completed request, as the replies report them or as the prompts were counted
    done, fail   completed and failed requests; a request fails on an HTTP error, a refused or dropped connection, an error event, a stream that ends before its last event, or a timeout
    short        completed requests with fewer tokens than they asked for
    reused       mean prompt tokens a completed request took from the server's prefix cache, from the replies (timings.cache_n) or from the difference of /v1/health's prefix_tokens around the level; - where the server reports neither

Every prompt starts with the word "the", so a server that matches a prompt against its cache token by token reuses that word and any start token, one or two tokens a request.
--api completion sends cache_prompt false, as this tool always has, so the reference server's /completion reuses nothing; the OpenAI route leaves every server's prefix cache at its default.

Below the table each level has notes: the failures of every round by reason, the replies that ended short, the prompts whose counted length misses its target, the sends of an open level that fell behind their arrival times at the 99th percentile by over 10 ms or a twentieth of the mean gap between arrivals, whichever is longer (the load was lighter than asked), and the requests that took over 100 ms to connect, as a burst past the server's listen backlog does.

--warmup requests at the longest prompt and the full reply length run before any level, so the first level does not carry the server's first pass or its first growth of scratch.
--timeout S fails a request after S seconds with nothing arriving (600 by default, the limit the earlier tool put on every read), and --total-timeout S a request that takes over S seconds in all (21600 by default); the calibration and the warm-up wait at least 600.
--json writes every request's record (send time and its lag behind the schedule, connect time, time to first token, gaps, tokens, failure reason) and every level's figures, rewritten after each level.
--self-test checks the stream reading and the figures on made-up arrival times exactly, then runs every API in both loads and the whole tool against an in-process server, and needs no model.

Exit status: 0 when every timed request completed, 3 when any failed (after the table, the notes and --json are written), 1 when the server cannot be reached or the warm-up fails, 2 on bad flags.

    python tools/server_load.py --url http://127.0.0.1:8080 --concurrency 1 4 8 16 --tokens 64
    python tools/server_load.py --input-len 128 --output-len 128 --concurrency 1 2 4 8 16 32 64 --num-prompts 128 --json closed.json
    llmx serve model.gguf --max-queue 256
    python tools/server_load.py --input-len-range 64:1024 --output-len 128 --rate 1 2 4 8 inf --num-prompts 200 --seed 1 --json open.json
"""
import argparse
import contextlib
import datetime
import http.client
import io
import json
import math
import os
import random
import socket
import sys
import tempfile
import threading
import time
import urllib.parse

PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a small village,",
    "def fibonacci(n):",
    "The three laws of thermodynamics state that",
    "In 1969, the first humans landed on the Moon.",
    "A recipe for bread needs flour, water,",
    "The quick brown fox",
    "Electric cars differ from petrol cars in",
]

# Common words that are one token each, with a leading space or without, in common byte-pair vocabularies; the calibration checks it on the server under test.
WORDS = (
    "the of and to in is was for on that with as it by at from his he be are this which an or had not but have they were "
    "one all their has been who its first new more after also other two when there into some time only can may would over "
    "such about three most years than people them what between then so under up these out during her she him we you our "
    "made used city world state school water house long many day part year work number way place game might each life both "
    "well much still very small great good old high few same last must before even down here where how back four five "
    "every never while through").split()

DEFAULT_LEVELS = [1, 2, 4, 8, 16, 32, 64]

# An open level starts this long after its request threads are released, so their wake-up is over before the first send.
LEAD = 0.05
# The notes flag an open level whose sends lag their arrival times at the 99th percentile by more than LAG_NOTE or a twentieth of the mean gap between arrivals, whichever is longer, and requests that take longer than CONNECT_NOTE to connect.
LAG_NOTE = 0.010
CONNECT_NOTE = 0.100


class Failure(Exception):
    """A request that did not complete, with the reason."""


class Target:
    """The server's address: host, port and a base path the routes are appended to."""

    def __init__(self, url):
        u = urllib.parse.urlsplit(url)
        if u.scheme not in ("http", "https") or not u.hostname:
            raise ValueError("--url must look like http://host:port")
        self.https = u.scheme == "https"
        self.host = u.hostname
        self.port = u.port or (443 if self.https else 80)
        self.base = u.path.rstrip("/")

    def connection(self, timeout):
        cls = http.client.HTTPSConnection if self.https else http.client.HTTPConnection
        return cls(self.host, self.port, timeout=timeout)

    def call(self, method, path, body=None, timeout=60):
        """A whole request: the status and the parsed JSON reply, or None for a reply that is not JSON."""
        conn = self.connection(timeout)
        try:
            data = None if body is None else json.dumps(body).encode("utf-8")
            conn.request(method, self.base + path, body=data,
                         headers={"Content-Type": "application/json"} if data is not None else {})
            r = conn.getresponse()
            raw = r.read()
        finally:
            conn.close()
        try:
            return r.status, json.loads(raw.decode("utf-8"))
        except ValueError:
            return r.status, None


class Api:
    """How one API asks for tokens, and how its streamed events read."""

    def __init__(self, name, model=None):
        self.name = name
        self.model = model

    def request(self, prompt, tokens, force, stream=True):
        """The route and body of a greedy request for `tokens` tokens; `force` asks the server to ignore the end of text."""
        if self.name == "llmx":
            path, body = "/v1/generate", {"prompt": prompt, "max_tokens": tokens, "temperature": 0}
        elif self.name == "openai":
            path, body = "/v1/completions", {"prompt": prompt, "max_tokens": tokens, "temperature": 0}
            if self.model:
                body["model"] = self.model
            if stream:
                body["stream_options"] = {"include_usage": True}
        else:
            path, body = "/completion", {"prompt": prompt, "n_predict": tokens, "temperature": 0, "cache_prompt": False}
        if stream:
            body["stream"] = True
        if force:
            body["ignore_eos"] = True
        return path, body

    def prompt_count(self, target, prompt, timeout):
        """The tokens the server reads for a prompt, from a whole reply of one token, or None when the reply does not say."""
        path, body = self.request(prompt, 1, False, stream=False)
        status, reply = target.call("POST", path, body, timeout)
        if status != 200 or not isinstance(reply, dict):
            return None
        if self.name == "llmx":
            n = reply.get("prompt_tokens")
        elif self.name == "openai":
            n = (reply.get("usage") or {}).get("prompt_tokens")
        else:
            n = reply.get("tokens_evaluated")
            t = reply.get("timings") or {}
            if n is None and isinstance(t.get("prompt_n"), int):
                n = t["prompt_n"] + t.get("cache_n", 0)
        return n if isinstance(n, int) else None

    def read(self, event, rec):
        """Whether a streamed event carries a token and whether it is the reply's last, with the counts it reports written into the record."""
        if not isinstance(event, dict):
            return False, False
        if "error" in event:
            raise Failure("error event: " + json.dumps(event["error"])[:200])
        if self.name == "llmx":
            if "id" in event:
                return True, False
            if event.get("done"):
                rec["finish"] = event.get("finish")
                for key, field in (("tokens", "output_reported"), ("prompt_tokens", "input_reported"),
                                   ("reused_tokens", "reused")):
                    if isinstance(event.get(key), int):
                        rec[field] = event[key]
                return False, True
            # A held piece of text after the last token, with no id.
            return False, False
        if self.name == "openai":
            usage = event.get("usage")
            if isinstance(usage, dict):
                if isinstance(usage.get("prompt_tokens"), int):
                    rec["input_reported"] = usage["prompt_tokens"]
                if isinstance(usage.get("completion_tokens"), int):
                    rec["output_reported"] = usage["completion_tokens"]
            timings = event.get("timings")
            if isinstance(timings, dict) and isinstance(timings.get("cache_n"), int):
                rec["reused"] = timings["cache_n"]
            choices = event.get("choices")
            if not choices:
                return False, False
            c = choices[0]
            text = c.get("text")
            if text is None:
                text = (c.get("delta") or {}).get("content")
            finish = c.get("finish_reason")
            if finish is not None:
                rec["finish"] = finish
            # A chunk carries a token unless it is the finish chunk with no text of its own; a chunk of held text with no token behind it is dropped by the reply's count.
            return bool(text) or finish is None, finish is not None
        # The reference server's /completion: every event a token, the last one flagged stop and carrying the counts.
        if not event.get("stop"):
            return True, False
        if isinstance(event.get("tokens_predicted"), int):
            rec["output_reported"] = event["tokens_predicted"]
        if isinstance(event.get("tokens_evaluated"), int):
            rec["input_reported"] = event["tokens_evaluated"]
        timings = event.get("timings")
        if isinstance(timings, dict) and isinstance(timings.get("cache_n"), int):
            rec["reused"] = timings["cache_n"]
        rec["finish"] = event.get("stop_type") or ("eos" if event.get("stopped_eos") else
                                                   "length" if event.get("stopped_limit") else "stop")
        return bool(event.get("content")), True


def percentile(values, q):
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def new_record(index, spec):
    return {"index": index, "ok": False, "error": None, "input_target": spec["input_target"],
            "input_counted": spec["input_counted"], "output_target": spec["tokens"], "input_reported": None,
            "output_reported": None, "reused": None, "finish": None, "connect": None, "scheduled": None, "lag": None}


def parse_stream(api, lines, rec, arrivals):
    """Reads a streamed reply from its lines, (arrival time, line) pairs in order: every token event's time is appended to `arrivals`, the counts the reply reports are written into the record, and the time of the reply's last event is returned.
    Raises Failure on an error event and on a stream that ends before its last event."""
    ended, end = False, None
    for now, raw in lines:
        line = raw.strip()
        if line.startswith(b"error:"):
            # The reference server sends an error that ends a stream as an event of that name.
            raise Failure("error event: " + line[6:].strip().decode("utf-8", "replace")[:200])
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if payload == b"[DONE]":
            if ended or api.name == "openai":
                return now
            break
        token, last = api.read(json.loads(payload.decode("utf-8")), rec)
        if token:
            arrivals.append(now)
        if last:
            ended, end = True, now
            if api.name == "completion":
                # No [DONE] follows /completion's stop event, so the reply ends here, whatever the server does with the stream after it.
                return now
    if ended:
        return end
    raise Failure("the stream ended before its last event")


def finish_record(rec, sent, arrivals, end, t0):
    """A request's figures from its send time, its token arrival times and its end, all on the perf_counter clock; `t0` is its level's start."""
    if rec["output_reported"] is not None and len(arrivals) > rec["output_reported"]:
        # Events past the reply's own count carry text the server held back, the rest of a character split across tokens, and no token of their own.
        arrivals = arrivals[:rec["output_reported"]]
    tokens = rec["output_reported"] if rec["output_reported"] is not None else len(arrivals)
    rec["sent"] = sent - t0
    rec["token_events"] = len(arrivals)
    rec["output_tokens"] = tokens
    rec["input_tokens"] = rec["input_reported"] if rec["input_reported"] is not None else rec["input_counted"]
    rec["ttft"] = arrivals[0] - sent if arrivals and tokens > 0 else None
    rec["tpot"] = (end - arrivals[0]) / (tokens - 1) if tokens > 1 and arrivals else None
    rec["itl"] = [b - a for a, b in zip(arrivals, arrivals[1:])]
    rec["e2e"] = end - sent
    rec["short"] = rec["ok"] and tokens < rec["output_target"]
    return rec


def send(target, api, spec, index, limits, t0):
    """One streamed request, timed; a failure is recorded with its reason, never raised.
    `limits` is (idle, total): the seconds the request may wait with nothing arriving, and the seconds it may take in all."""
    idle, total = limits
    rec = new_record(index, spec)
    path, body = api.request(spec["prompt"], spec["tokens"], spec["force"])
    data = json.dumps(body).encode("utf-8")
    arrivals = []
    conn = target.connection(min(idle, total))
    sent = time.perf_counter()
    deadline = sent + total
    end = None
    sock = None

    def wait(read):
        # Every blocking read waits at most `idle`, and never past the request's deadline.
        left = deadline - time.perf_counter()
        if left <= 0:
            raise Failure("the reply took over %gs" % total)
        sock.settimeout(min(idle, left))
        try:
            return read()
        except socket.timeout:
            raise Failure("the reply took over %gs" % total if left < idle else "nothing arrived for %gs" % idle)

    def lines(r):
        while True:
            raw = wait(r.readline)
            now = time.perf_counter()
            if not raw:
                return
            yield now, raw

    try:
        try:
            conn.connect()
        except socket.timeout:
            raise Failure("no connection within %gs" % min(idle, total))
        rec["connect"] = time.perf_counter() - sent
        # The response takes the socket over from a connection it closes, so the reads set their timeouts on the socket itself.
        sock = conn.sock
        conn.request("POST", target.base + path, body=data, headers={"Content-Type": "application/json"})
        r = wait(conn.getresponse)
        if r.status != 200:
            raise Failure("HTTP %d %s" % (r.status, wait(lambda: r.read(200)).decode("utf-8", "replace").strip()))
        end = parse_stream(api, lines(r), rec, arrivals)
        rec["ok"] = True
    except Failure as e:
        rec["error"] = str(e)
    except http.client.IncompleteRead:
        rec["error"] = "the connection closed before the reply ended"
    except Exception as e:
        # Anything else a request meets, a refused connection or a reply that does not parse, fails that request and no other.
        rec["error"] = "%s: %s" % (type(e).__name__, e)
    finally:
        conn.close()
    return finish_record(rec, sent, arrivals, end if end is not None else time.perf_counter(), t0)


TOKENIZE_PATHS = ("/tokenize", "/v1/tokenize")


def tokenize_route(target, timeout):
    """A counter through the server's tokenize route, the first of TOKENIZE_PATHS it answers on, or None when it has none: the body carries the field names the common tokenize routes read, with the start token counted."""
    def counter(path):
        def count(text):
            status, reply = target.call("POST", path, {"content": text, "prompt": text, "text": text, "add_special": True,
                                                       "add_special_tokens": True}, timeout)
            if status == 200 and isinstance(reply, dict) and isinstance(reply.get("tokens"), list):
                return len(reply["tokens"])
            return None
        return count
    for path in TOKENIZE_PATHS:
        count = counter(path)
        if count(WORDS[0]) is not None:
            return count
    return None


class Lengths:
    """Prompts of a token count, built from WORDS, and how the server under test counts them."""

    def __init__(self, source=None, base=0, per_word=1.0, uniform=None, probes=None, count=None):
        self.source, self.base, self.per_word, self.uniform, self.probes, self.count = (
            source, base, per_word, uniform, probes, count)

    @classmethod
    def calibrate(cls, target, api, timeout):
        """How the server counts prompts: through its tokenize route, or from two probe requests."""
        count = tokenize_route(target, timeout)
        counter = count or (lambda text: api.prompt_count(target, text, timeout))
        once, twice = counter(" ".join(WORDS)), counter(" ".join(WORDS + WORDS))
        if once is None or twice is None or twice <= once:
            # Nothing reports a count: a prompt of N tokens is N words, unverified.
            return cls(count=count)
        # A word list read twice costs its words once more: the difference is what the words cost, the rest is what a prompt costs beyond them.
        return cls("tokenize route" if count else "two probe requests", 2 * once - twice, (twice - once) / len(WORDS),
                   twice - once == len(WORDS), [once, twice], count)

    def smallest(self):
        return self.base + 1

    def make(self, rng, n):
        """A prompt of n tokens and its count, or None when nothing counts it; the count can miss n where a tokenize route counts words the calibration did not see."""
        k = max(1, int(round((n - self.base) / self.per_word)))
        words = [WORDS[0]] + [rng.choice(WORDS) for _ in range(k - 1)]
        if self.count:
            c = self.count(" ".join(words))
            for _ in range(64):
                if c is None or c == n:
                    break
                if c < n:
                    words.append(rng.choice(WORDS))
                elif len(words) > 1:
                    words.pop()
                else:
                    break
                c = self.count(" ".join(words))
            return " ".join(words), c
        return " ".join(words), (self.base + k if self.uniform else None)

    def describe(self):
        if self.source is None:
            return "not counted: the server has no tokenize route and its replies report no prompt count, so a prompt of N tokens is N words"
        if self.uniform:
            return "counted by the server's %s: every word one token, %d more a prompt" % (self.source, self.base)
        return ("counted by the server's %s: the words average %.3f tokens here, so %s" %
                (self.source, self.per_word, "each prompt is trimmed to its length there" if self.count else
                 "prompt lengths are approximate"))

    def json(self):
        return {"source": self.source, "probes": self.probes, "per_word": self.per_word, "base": self.base,
                "uniform": self.uniform}


class Workload:
    """The prompts and reply lengths of every request."""

    def __init__(self, args, lengths):
        self.seed = args.seed
        self.tokens = args.output_len if args.output_len is not None else args.tokens
        self.force = args.output_len is not None
        self.range = None
        if args.input_len is not None:
            self.range = (args.input_len, args.input_len)
        elif args.input_len_range is not None:
            self.range = args.input_len_range
        self.lengths = lengths

    def targets(self, n):
        """The prompt lengths of n requests, from the seed alone: the i-th request of every level and round has the same length whatever the server counts."""
        rng = random.Random("%d/lengths" % self.seed)
        return [rng.randint(*self.range) for _ in range(n)]

    def specs(self, key, n, longest=False):
        """n requests, the same for the same seed and key; `longest` gives every one the longest prompt length."""
        if self.range is None:
            prompts = [(PROMPTS[i % len(PROMPTS)], None, None) for i in range(n)]
        else:
            targets = [self.range[1]] * n if longest else self.targets(n)
            # The words come from a stream of their own, so how many a prompt takes on one server moves no other prompt's length.
            rng = random.Random("%d/words/%s" % (self.seed, key))
            prompts = [self.lengths.make(rng, t) + (t,) for t in targets]
        return [{"prompt": prompt, "input_target": target, "input_counted": counted, "tokens": self.tokens,
                 "force": self.force} for prompt, counted, target in prompts]

    def describe(self):
        if self.range is None:
            prompts = "the eight fixed prompts in turn"
        elif self.range[0] == self.range[1]:
            prompts = "prompts of %d tokens" % self.range[0]
        else:
            prompts = "prompts of %d to %d tokens, uniform" % self.range
        replies = ("%d tokens a reply, end of text ignored" % self.tokens if self.force else
                   "up to %d tokens a reply, ending at the end of text" % self.tokens)
        return "%s; %s; seed %d" % (prompts, replies, self.seed)


def health(target):
    """The server's prefix counters from /v1/health, or None where it has none."""
    try:
        status, h = target.call("GET", "/v1/health", timeout=10)
    except (OSError, http.client.HTTPException):
        return None
    if status == 200 and isinstance(h, dict) and isinstance(h.get("prefix_hits"), int):
        return h
    return None


def closed_level(target, api, specs, concurrency, limits):
    """specs sent by `concurrency` users, each sending its next request when its reply ends: the records and the level's span."""
    records = [None] * len(specs)
    lock = threading.Lock()
    queue = list(range(len(specs)))
    queue.reverse()

    def user():
        while True:
            with lock:
                if not queue:
                    return
                i = queue.pop()
            records[i] = send(target, api, specs[i], i, limits, t0)

    threads = [threading.Thread(target=user) for _ in range(min(concurrency, len(specs)))]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return records, time.perf_counter() - t0


def arrivals(rate, n, seed):
    """Send offsets in seconds of n requests arriving as a Poisson process at `rate` a second; the first at 0."""
    rng = random.Random("%d/arrivals/%r" % (seed, rate))
    out, t = [], 0.0
    for _ in range(n):
        out.append(t)
        if math.isfinite(rate):
            t += rng.expovariate(rate)
    return out


def open_level(target, api, specs, offsets, limits):
    """specs sent at their offsets whatever is still running: the records and the level's span, each record with its send's lag behind its offset."""
    records = [None] * len(specs)
    go = threading.Event()

    def one(i):
        go.wait()
        delay = t0 + offsets[i] - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        records[i] = send(target, api, specs[i], i, limits, t0)

    # Every request's thread is running before the level starts, each sleeping until its time, so starting threads, which a burst would otherwise wait on, delays no send.
    threads = [threading.Thread(target=one, args=(i,)) for i in range(len(specs))]
    for t in threads:
        t.start()
    t0 = time.perf_counter() + LEAD
    go.set()
    for t in threads:
        t.join()
    for rec, offset in zip(records, offsets):
        rec["scheduled"] = offset
        rec["lag"] = rec["sent"] - offset
    return records, time.perf_counter() - t0


def summarize(records, wall, before=None, after=None):
    """A level's figures from its records and its span, with the server's prefix counters around it when it has them."""
    done = [r for r in records if r["ok"]]
    ttft = [r["ttft"] for r in done if r["ttft"] is not None]
    tpot = [r["tpot"] for r in done if r["tpot"] is not None]
    itl = [g for r in done for g in r["itl"]]
    e2e = [r["e2e"] for r in done]
    out = sum(r["output_tokens"] for r in done)
    inputs = [r["input_tokens"] for r in done]
    known = bool(done) and all(n is not None for n in inputs)
    s = {"requests": len(records), "completed": len(done), "failed": len(records) - len(done),
         "short": sum(1 for r in done if r["short"]), "duration": wall,
         "input_tokens": sum(inputs) if known else None, "output_tokens": out,
         "output_tok_s": out / wall if wall > 0 else float("nan"),
         "total_tok_s": (sum(inputs) + out) / wall if known and wall > 0 else float("nan"),
         "req_s": len(done) / wall if wall > 0 else float("nan"),
         "input_mean": mean(inputs) if known else float("nan"), "output_mean": mean([r["output_tokens"] for r in done]),
         "ttft_mean": mean(ttft), "ttft_p50": percentile(ttft, 0.5), "ttft_p90": percentile(ttft, 0.9),
         "ttft_p99": percentile(ttft, 0.99),
         "tpot_mean": mean(tpot), "tpot_p50": percentile(tpot, 0.5), "tpot_p99": percentile(tpot, 0.99),
         "itl_mean": mean(itl), "itl_p50": percentile(itl, 0.5), "itl_p99": percentile(itl, 0.99),
         "e2e_mean": mean(e2e), "e2e_p50": percentile(e2e, 0.5), "e2e_p99": percentile(e2e, 0.99),
         "prefix_hits": None, "prefix_tokens": None, "reused_mean": None, "prefix_source": None}
    reused = [r["reused"] for r in done if r["reused"] is not None]
    if reused:
        s["prefix_hits"], s["prefix_tokens"], s["prefix_source"] = sum(1 for n in reused if n > 0), sum(reused), "replies"
        s["reused_mean"] = mean(reused)
    elif before is not None and after is not None:
        s["prefix_hits"] = after["prefix_hits"] - before["prefix_hits"]
        s["prefix_tokens"] = after.get("prefix_tokens", 0) - before.get("prefix_tokens", 0)
        s["reused_mean"] = s["prefix_tokens"] / len(done) if done else None
        s["prefix_source"] = "health"
    # The prompt counts the replies report, against the lengths asked, and the prompts whose own count missed its length.
    reported = [r for r in done if r["input_reported"] is not None and r["input_target"] is not None]
    s["input_checked"] = len(reported)
    s["input_mismatched"] = sum(1 for r in reported if r["input_reported"] != r["input_target"])
    s["input_off"] = sum(1 for r in records if r["input_counted"] is not None and r["input_target"] is not None and
                         r["input_counted"] != r["input_target"])
    connects = [r["connect"] for r in records if r["connect"] is not None]
    s["connect_p50"], s["connect_p99"] = percentile(connects, 0.5), percentile(connects, 0.99)
    s["connect_max"] = max(connects) if connects else None
    s["connect_slow"] = sum(1 for c in connects if c > CONNECT_NOTE)
    lags = [r["lag"] for r in records if r["lag"] is not None]
    s["lag_p50"], s["lag_p99"], s["lag_max"] = ((percentile(lags, 0.5), percentile(lags, 0.99), max(lags)) if lags else
                                                (None, None, None))
    s["errors"] = {}
    for r in records:
        if not r["ok"]:
            s["errors"][r["error"]] = s["errors"].get(r["error"], 0) + 1
    return s


def best_round(summaries):
    """The round a closed level shows: the most output tokens per second among the rounds in which no request failed, or among all when every round has a failure."""
    return max(range(len(summaries)), key=lambda k: (summaries[k]["failed"] == 0, summaries[k]["output_tok_s"]))


HEAD = "%-10s %5s %9s %8s %10s %10s %10s %10s" + " %9s %10s %10s %10s %10s %10s %10s %10s %6s %6s %5s %4s %5s %6s"
COLUMNS = ("tok/s", "req/s", "ttft p50", "ttft p99", "itl p50", "itl p99", "all tok/s", "ttft mean", "ttft p90",
           "tpot mean", "tpot p50", "tpot p99", "e2e p50", "e2e p99", "in", "out", "done", "fail", "short", "reused")


def header(level_name):
    return HEAD % (("api", level_name) + COLUMNS)


def row(api, level, s):
    ms = lambda v: v * 1e3
    reused = "-" if s["reused_mean"] is None else "%.1f" % s["reused_mean"]
    # The first eight columns keep the earlier table's layout.
    return ("%-10s %5s %9.1f %8.2f %8.1f ms %8.1f ms %8.2f ms %8.2f ms" % (
                api, level, s["output_tok_s"], s["req_s"], ms(s["ttft_p50"]), ms(s["ttft_p99"]), ms(s["itl_p50"]),
                ms(s["itl_p99"])) +
            " %9.1f %8.1f ms %8.1f ms %8.2f ms %8.2f ms %8.2f ms %8.1f ms %8.1f ms %6.1f %6.1f %5d %4d %5d %6s" % (
                s["total_tok_s"], ms(s["ttft_mean"]), ms(s["ttft_p90"]), ms(s["tpot_mean"]), ms(s["tpot_p50"]),
                ms(s["tpot_p99"]), ms(s["e2e_p50"]), ms(s["e2e_p99"]), s["input_mean"], s["output_mean"],
                s["completed"], s["failed"], s["short"], reused))


def lag_limit(rate):
    """The send lag at the 99th percentile past which an open level at `rate` is noted."""
    return max(LAG_NOTE, 0.05 / rate) if math.isfinite(rate) else LAG_NOTE


def notes(label, s, lag=None):
    """The lines a level adds below the table: its failures by reason, short replies, prompt lengths off their target, sends behind their schedule by more than `lag` and slow connections."""
    out = []
    for reason, n in sorted(s["errors"].items(), key=lambda kv: -kv[1]):
        out.append("%s: %d failed: %s" % (label, n, reason))
    if s["short"]:
        out.append("%s: %d of %d replies ended before their length" % (label, s["short"], s["completed"]))
    if s["input_mismatched"]:
        out.append("%s: %d of %d replies report a prompt count off its target" % (label, s["input_mismatched"],
                                                                                 s["input_checked"]))
    if s["input_off"]:
        out.append("%s: %d prompts are off their length by the tokenize route's count, the nearest the word list reached" %
                   (label, s["input_off"]))
    if lag is not None and s["lag_p99"] is not None and s["lag_p99"] > lag:
        out.append("%s: requests went out behind their arrival times, by %.1f ms at the median, %.1f ms at the 99th "
                   "percentile and %.1f ms at most, so the load was lighter than asked" %
                   (label, s["lag_p50"] * 1e3, s["lag_p99"] * 1e3, s["lag_max"] * 1e3))
    if s["connect_slow"]:
        out.append("%s: %d requests took over %d ms to connect, %.0f ms at most, which their time to first token includes; "
                   "a burst past the server's listen backlog does this" %
                   (label, s["connect_slow"], CONNECT_NOTE * 1e3, s["connect_max"] * 1e3))
    return out


def clean(value):
    """The value with every NaN and infinity as null, which JSON can hold."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def write_json(path, doc):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(clean(doc), f, indent=1)
    os.replace(tmp, path)


def length_range(text):
    try:
        lo, hi = (int(x) for x in text.split(":"))
    except ValueError:
        raise argparse.ArgumentTypeError("expected LO:HI, two whole numbers")
    if not 1 <= lo <= hi:
        raise argparse.ArgumentTypeError("expected 1 <= LO <= HI")
    return lo, hi


def positive(text):
    n = int(text)
    if n < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return n


def rate(text):
    r = float(text)
    if not r > 0:
        raise argparse.ArgumentTypeError("a rate must be above 0, or inf")
    return r


def parse(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("--api", choices=["llmx", "openai", "completion"], default="llmx")
    p.add_argument("--model", default=None, help="the model id --api openai sends; by default the first /v1/models lists")
    p.add_argument("--concurrency", type=positive, nargs="+", default=None,
                   help="closed-loop levels, users at once (default 1 2 4 8 16 32 64 unless --rate is given alone)")
    p.add_argument("--rate", type=rate, nargs="+", default=None,
                   help="open-loop levels, requests a second arriving as a Poisson process; inf sends all at once")
    p.add_argument("--num-prompts", type=positive, default=None,
                   help="requests a level: the closed loop's default is one a user, the open loop's 100")
    p.add_argument("--tokens", type=positive, default=None,
                   help="tokens a request asks for, a reply ending at the end of text (default 64)")
    p.add_argument("--output-len", type=positive, default=None,
                   help="tokens every request asks for with the end of text ignored; replaces --tokens")
    p.add_argument("--input-len", type=positive, default=None, help="prompt tokens of every request")
    p.add_argument("--input-len-range", type=length_range, default=None, metavar="LO:HI",
                   help="prompt tokens uniform from LO to HI")
    p.add_argument("--seed", type=int, default=0, help="seeds the prompt lengths, the prompts and the arrivals (default 0)")
    p.add_argument("--rounds", type=positive, default=2,
                   help="repeats of each closed-loop level; the table shows the round with the most output tok/s among "
                        "those in which no request failed (default 2)")
    p.add_argument("--warmup", type=int, default=1, help="requests before any level, one at a time (default 1)")
    p.add_argument("--timeout", type=float, default=600,
                   help="seconds a timed request may go with nothing arriving before it fails (default 600)")
    p.add_argument("--total-timeout", type=float, default=21600,
                   help="seconds a timed request may take in all before it fails (default 21600); the calibration and "
                        "the warm-up wait at least 600 on both limits")
    p.add_argument("--json", default=None, metavar="PATH", help="write every request's record and every level's figures")
    p.add_argument("--self-test", action="store_true", help="check the figures against made-up times and an in-process server, and exit")
    args = p.parse_args(argv)
    if args.input_len is not None and args.input_len_range is not None:
        p.error("--input-len and --input-len-range are two ways to say one thing; give one")
    if args.output_len is not None and args.tokens is not None:
        p.error("--output-len replaces --tokens; give one")
    if args.tokens is None:
        args.tokens = 64
    if args.warmup < 0:
        p.error("--warmup must be 0 or more")
    if not args.timeout > 0 or not args.total_timeout > 0:
        p.error("--timeout and --total-timeout must be above 0")
    return args


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args = parse(argv)
    if args.self_test:
        return 0 if self_test() else 1
    target = Target(args.url)
    levels = args.concurrency if args.concurrency is not None else ([] if args.rate else DEFAULT_LEVELS)
    rates = args.rate or []
    limits = (args.timeout, args.total_timeout)
    # The calls before the timed levels, the calibration and the warm-up, carry a server's first pass and are not what the limits bound.
    setup = (max(args.timeout, 600), max(args.total_timeout, 600))
    try:
        model = args.model
        if args.api == "openai" and model is None:
            status, models = target.call("GET", "/v1/models", timeout=30)
            if status == 200 and isinstance(models, dict) and models.get("data"):
                model = models["data"][0].get("id")
        api = Api(args.api, model)
        lengths = None
        if args.input_len is not None or args.input_len_range is not None:
            lengths = Lengths.calibrate(target, api, setup[0])
            want = args.input_len if args.input_len is not None else args.input_len_range[0]
            if want < lengths.smallest():
                sys.exit("error: a prompt here is at least %d tokens, above the %d asked" % (lengths.smallest(), want))
    except (OSError, http.client.HTTPException) as e:
        sys.exit("error: cannot reach %s: %s" % (args.url, e))
    work = Workload(args, lengths)
    doc = {"tool": "tools/server_load.py", "schema": 2, "started": datetime.datetime.now().isoformat(timespec="seconds"),
           "url": args.url, "api": args.api, "model": model,
           "args": dict(vars(args), rate=[r if math.isfinite(r) else "inf" for r in rates]), "workload": work.describe(),
           "lengths": lengths.json() if lengths else None, "warmup": [], "levels": [], "requests": 0, "failed": 0}
    if args.input_len is not None or args.input_len_range is not None or args.output_len is not None:
        print("workload: " + work.describe())
        if lengths:
            print("prompts: " + lengths.describe())
        sys.stdout.flush()

    # The warm-up runs at the longest prompt and the full reply length, one request at a time.
    for i, spec in enumerate(work.specs("warmup", args.warmup, longest=True)):
        rec = send(target, api, spec, i, setup, time.perf_counter())
        doc["warmup"].append(rec)
        if not rec["ok"]:
            if args.json:
                write_json(args.json, doc)
            sys.exit("error: the warm-up request failed: %s" % rec["error"])

    reasons = set()

    def account(s):
        doc["requests"] += s["requests"]
        doc["failed"] += s["failed"]
        reasons.update(s["errors"])

    pending = []
    if levels:
        print(header("conc"))
        sys.stdout.flush()
    for c in levels:
        n = args.num_prompts if args.num_prompts is not None else c
        rounds = []
        for k in range(args.rounds):
            specs = work.specs("closed/%d/%d" % (c, k), n)
            before = health(target)
            records, wall = closed_level(target, api, specs, c, limits)
            s = summarize(records, wall, before, health(target))
            account(s)
            rounds.append({"round": k, "summary": s, "records": records})
        best = best_round([rd["summary"] for rd in rounds])
        print(row(args.api, str(c), rounds[best]["summary"]))
        sys.stdout.flush()
        for k, rd in enumerate(rounds):
            label = "conc %d" % c if k == best else "conc %d, round %d of %d, not the one shown" % (c, k + 1, len(rounds))
            pending += notes(label, rd["summary"])
        doc["levels"].append({"mode": "closed", "concurrency": c, "requests": n, "best": best,
                              "summary": rounds[best]["summary"], "rounds": rounds})
        if args.json:
            write_json(args.json, doc)
    for line in pending:
        print(line)
    pending = []
    if rates:
        if levels:
            print()
        print(header("rate"))
        sys.stdout.flush()
    for r in rates:
        n = args.num_prompts if args.num_prompts is not None else 100
        specs = work.specs("open/%r" % r, n)
        before = health(target)
        records, wall = open_level(target, api, specs, arrivals(r, n, args.seed), limits)
        s = summarize(records, wall, before, health(target))
        account(s)
        print(row(args.api, "%g" % r, s))
        sys.stdout.flush()
        pending += notes("rate %g" % r, s, lag_limit(r))
        doc["levels"].append({"mode": "open", "rate": r if math.isfinite(r) else "inf", "requests": n, "summary": s,
                              "records": records})
        if args.json:
            write_json(args.json, doc)
    for line in pending:
        print(line)
    if any(reason.startswith("HTTP 503") for reason in reasons):
        print("HTTP 503 is a server refusing work past its queue: `llmx serve` runs --max-seqs requests and queues "
              "--max-queue more (16 and 64 by default), so give --max-queue at least --num-prompts")
    sys.stdout.flush()
    if doc["failed"]:
        print("error: %d of %d timed requests failed" % (doc["failed"], doc["requests"]), file=sys.stderr)
        return 3
    return 0


class FakeServer:
    """A server for --self-test that speaks the three APIs, counts a prompt's tokens as its words, and sends each streamed token at a known time after the request arrives: the first after `ttft` seconds and one every `itl` after it.
    With `tokenize` one of TOKENIZE_PATHS it has a tokenize route there, reading the field that path's servers read.
    Streamed requests take their behaviour from `plan` in arrival order: ok, 503, cut (the connection drops after two tokens), short (half the tokens, then the end of text) or stall (silence after the first token until the server closes).
    Every streamed reply reports 4 reused prompt tokens, which /v1/health counts, and `ignore_eos` keeps the field of that name each streamed request sent, None for none, in arrival order."""

    REUSED = 4

    def __init__(self, ttft, itl, tokenize):
        import http.server
        fake = self
        self.ttft, self.itl, self.tokenize = ttft, itl, tokenize
        self.plan = []
        self.lock = threading.Lock()
        self.closing = threading.Event()
        self.hits = 0
        self.reused = 0
        self.ignore_eos = []

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            disable_nagle_algorithm = True

            def log_message(self, *args):
                pass

            def reply(self, status, obj):
                data = json.dumps(obj).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(data)
                self.close_connection = True

            def chunk(self, obj):
                data = b"data: " + (obj if isinstance(obj, bytes) else json.dumps(obj).encode("utf-8")) + b"\n\n"
                self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))

            def do_GET(self):
                if self.path == "/v1/health":
                    with fake.lock:
                        return self.reply(200, {"status": "ok", "prefix_hits": fake.hits, "prefix_tokens": fake.reused})
                if self.path == "/v1/models":
                    return self.reply(200, {"object": "list", "data": [{"id": "fake"}]})
                self.reply(404, {"error": "no such route"})

            def do_POST(self):
                arrived = time.perf_counter()
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))).decode("utf-8"))
                api = {"/v1/generate": "llmx", "/v1/completions": "openai", "/completion": "completion"}.get(self.path)
                if self.path == fake.tokenize:
                    field = {"/tokenize": "content", "/v1/tokenize": "text"}[self.path]
                    if not isinstance(body.get(field), str):
                        return self.reply(400, {"error": field + " must be a string"})
                    return self.reply(200, {"tokens": list(range(len(body[field].split())))})
                if api is None:
                    return self.reply(404, {"error": "no such route"})
                prompt = len(body["prompt"].split())
                tokens = body.get("max_tokens", body.get("n_predict"))
                if not body.get("stream"):
                    return self.reply(200, {"llmx": {"text": "x", "prompt_tokens": prompt, "tokens": 1},
                                            "openai": {"usage": {"prompt_tokens": prompt, "completion_tokens": 1}},
                                            "completion": {"content": "x", "tokens_evaluated": prompt,
                                                           "tokens_predicted": 1}}[api])
                with fake.lock:
                    behaviour = fake.plan.pop(0) if fake.plan else "ok"
                    fake.ignore_eos.append(body.get("ignore_eos"))
                if behaviour == "503":
                    return self.reply(503, {"error": "queue full"})
                with fake.lock:
                    fake.hits += 1
                    fake.reused += FakeServer.REUSED
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                emit = tokens // 2 if behaviour == "short" else tokens
                try:
                    for j in range(emit):
                        if behaviour == "cut" and j == 2:
                            return
                        if behaviour == "stall" and j == 1:
                            fake.closing.wait()
                            return
                        wait = arrived + fake.ttft + j * fake.itl - time.perf_counter()
                        if wait > 0:
                            time.sleep(wait)
                        self.chunk({"llmx": {"id": 100 + j, "text": " w"},
                                    "openai": {"choices": [{"index": 0, "text": " w", "finish_reason": None}]},
                                    "completion": {"content": " w", "stop": False}}[api])
                    finish = "length" if emit == tokens else "eos"
                    if api == "llmx":
                        self.chunk({"done": True, "finish": finish, "tokens": emit})
                        self.chunk(b"[DONE]")
                    elif api == "openai":
                        self.chunk({"choices": [{"index": 0, "text": "", "finish_reason": "length" if emit == tokens else "stop"}],
                                    "timings": {"cache_n": FakeServer.REUSED, "prompt_n": prompt - FakeServer.REUSED}})
                        self.chunk({"choices": [], "usage": {"prompt_tokens": prompt, "completion_tokens": emit}})
                        self.chunk(b"[DONE]")
                    else:
                        self.chunk({"content": "", "stop": True, "tokens_predicted": emit, "tokens_evaluated": prompt,
                                    "stop_type": "limit" if emit == tokens else "eos",
                                    "timings": {"cache_n": FakeServer.REUSED, "prompt_n": prompt - FakeServer.REUSED}})
                    self.wfile.write(b"0\r\n\r\n")
                except OSError:
                    pass

        class Server(http.server.ThreadingHTTPServer):
            daemon_threads = True
            request_queue_size = 64

        self.httpd = Server(("127.0.0.1", 0), Handler)
        self.url = "http://127.0.0.1:%d" % self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()

    def close(self):
        self.closing.set()
        self.httpd.shutdown()
        self.httpd.server_close()


def made_up(api_name, events, sent=1.0, tokens=5):
    """A request's record from made-up (seconds after its send, line) pairs, read as send reads a stream."""
    rec = new_record(0, {"prompt": "", "input_target": 12, "input_counted": 12, "tokens": tokens, "force": True})
    arrivals_, end = [], None
    try:
        end = parse_stream(Api(api_name), ((sent + t, line) for t, line in events), rec, arrivals_)
        rec["ok"] = True
    except Failure as e:
        rec["error"] = str(e)
    return finish_record(rec, sent, arrivals_, end if end is not None else sent + events[-1][0], 0.0)


def exact_checks(check):
    """The stream reading and the figures on made-up times, where every value is known exactly."""
    def near(label, value, want):
        check(label, value is not None and abs(value - want) < 1e-9, "= %r, want %r" % (value, want))

    def gaps(label, rec, want):
        check(label + " itl", len(rec["itl"]) == len(want) and all(abs(a - b) < 1e-9 for a, b in zip(rec["itl"], want)),
              "= %r, want %r" % (rec["itl"], want))

    def d(obj):
        return b"data: " + json.dumps(obj).encode("utf-8") + b"\n"

    blank = b"\n"
    times = [0.10, 0.15, 0.20, 0.35, 0.40]

    # /v1/generate: five tokens, a held piece of text with no id, then the done event and [DONE] in one write.
    ev = [(t, line) for t in times for line in (d({"id": 7, "text": " w"}), blank)]
    ev += [(0.41, d({"text": "é"})), (0.42, d({"done": True, "finish": "length", "tokens": 5})), (0.42, blank),
           (0.42, b"data: [DONE]\n")]
    r = made_up("llmx", ev)
    check("llmx stream", (r["ok"], r["token_events"], r["output_tokens"], r["finish"], r["short"]) ==
          (True, 5, 5, "length", False), r)
    near("llmx ttft", r["ttft"], 0.10)
    gaps("llmx", r, [0.05, 0.05, 0.15, 0.05])
    near("llmx tpot", r["tpot"], (0.42 - 0.10) / 4)
    near("llmx e2e", r["e2e"], 0.42)

    # llmx's /v1/completions: the same tokens, a held piece after the last one with no finish reason, the finish chunk with its timings, the usage chunk, and [DONE] 10 ms later.
    ev = [(t, d({"choices": [{"index": 0, "text": " w", "finish_reason": None}]})) for t in times]
    ev += [(0.41, d({"choices": [{"index": 0, "text": "é", "finish_reason": None}]})),
           (0.42, d({"choices": [{"index": 0, "text": "", "finish_reason": "length"}], "timings": {"cache_n": 3}})),
           (0.42, d({"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 5}})), (0.43, b"data: [DONE]\n")]
    r = made_up("openai", ev)
    check("openai stream", (r["ok"], r["token_events"], r["output_tokens"], r["input_tokens"], r["reused"]) ==
          (True, 5, 5, 12, 3), r)
    near("openai ttft", r["ttft"], 0.10)
    gaps("openai", r, [0.05, 0.05, 0.15, 0.05])
    near("openai tpot", r["tpot"], (0.43 - 0.10) / 4)
    near("openai e2e", r["e2e"], 0.43)

    # A server that puts the finish reason on the last token's chunk.
    ev = [(0.1, d({"choices": [{"index": 0, "text": " a", "finish_reason": None}]})),
          (0.2, d({"choices": [{"index": 0, "text": " b", "finish_reason": None}]})),
          (0.3, d({"choices": [{"index": 0, "text": " c", "finish_reason": "length"}]})),
          (0.3, d({"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 3}})), (0.31, b"data: [DONE]\n")]
    r = made_up("openai", ev, tokens=3)
    check("openai finish on a token", (r["ok"], r["token_events"], r["output_tokens"]) == (True, 3, 3), r)
    near("openai finish on a token tpot", r["tpot"], (0.31 - 0.1) / 2)

    # A reply with no usage chunk counts its token events, and its finish chunk with no text is not one.
    ev = [(0.1, d({"choices": [{"index": 0, "text": " a", "finish_reason": None}]})),
          (0.2, d({"choices": [{"index": 0, "text": " b", "finish_reason": None}]})),
          (0.3, d({"choices": [{"index": 0, "text": "", "finish_reason": "length"}]})), (0.3, b"data: [DONE]\n")]
    r = made_up("openai", ev, tokens=2)
    check("openai without usage", (r["ok"], r["token_events"], r["output_tokens"], r["short"]) == (True, 2, 2, False), r)

    # llmx's /v1/completions reply of no token: its one empty chunk is not a token.
    ev = [(0.1, d({"choices": [{"index": 0, "text": "", "finish_reason": None}]})),
          (0.1, d({"choices": [{"index": 0, "text": "", "finish_reason": "stop"}]})),
          (0.1, d({"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 0}})), (0.1, b"data: [DONE]\n")]
    r = made_up("openai", ev)
    check("openai no token", (r["ok"], r["token_events"], r["output_tokens"], r["ttft"], r["tpot"], r["short"]) ==
          (True, 0, 0, None, None, True), r)

    # The reference server's /completion: events of 1, 3 and 1 tokens, the stop event reporting 5, then the stream held open with more on it.
    ev = [(0.10, d({"content": "a", "stop": False})), (0.30, d({"content": "bcd", "stop": False})),
          (0.40, d({"content": "e", "stop": False})),
          (0.42, d({"content": "", "stop": True, "tokens_predicted": 5, "tokens_evaluated": 12, "stop_type": "limit",
                    "timings": {"cache_n": 2, "prompt_n": 10}})),
          (0.90, d({"content": "late", "stop": False}))]
    r = made_up("completion", ev)
    check("completion stream", (r["ok"], r["token_events"], r["output_tokens"], r["input_tokens"], r["reused"],
                                r["finish"]) == (True, 3, 5, 12, 2, "limit"), r)
    near("completion ttft", r["ttft"], 0.10)
    gaps("completion", r, [0.2, 0.1])
    near("completion tpot", r["tpot"], (0.42 - 0.10) / 4)
    near("completion e2e", r["e2e"], 0.42)

    # Error events in each API's shape fail the request with their message, the tokens before them kept in the record.
    for name, line in (("completion", b'error: {"code": 500, "message": "boom", "type": "server_error"}\n'),
                       ("llmx", d({"error": "boom"})),
                       ("openai", d({"error": {"message": "boom", "type": "invalid_request_error"}}))):
        first = {"llmx": {"id": 1, "text": "a"}, "openai": {"choices": [{"index": 0, "text": "a", "finish_reason": None}]},
                 "completion": {"content": "a", "stop": False}}[name]
        r = made_up(name, [(0.1, d(first)), (0.2, line)])
        check(name + " error event", not r["ok"] and "boom" in (r["error"] or "") and r["token_events"] == 1, r)
        # A stream that ends after a token, with no last event.
        r = made_up(name, [(0.1, d(first)), (0.2, blank)])
        check(name + " cut stream", not r["ok"] and r["error"] == "the stream ended before its last event", r)
    r = made_up("llmx", [(0.1, d({"id": 1, "text": "a"})), (0.2, b"data: [DONE]\n")])
    check("llmx [DONE] with no done event", not r["ok"] and r["error"] == "the stream ended before its last event", r)

    # The level's arithmetic: two completed requests and a failed one on a level of 0.5 s.
    spec = {"input_target": 10, "input_counted": 10, "tokens": 4, "force": True, "prompt": ""}
    a = new_record(0, spec)
    a["ok"] = True
    finish_record(a, 1.0, [1.1, 1.12, 1.15, 1.2], 1.21, 1.0)
    b = new_record(1, spec)
    b.update(ok=True, output_reported=2, input_reported=11, reused=3)
    # A third arrival past the reply's count of two is dropped.
    finish_record(b, 1.05, [1.25, 1.26, 1.29], 1.3, 1.0)
    c = new_record(2, dict(spec, input_counted=9))
    c["error"] = "HTTP 503"
    finish_record(c, 1.1, [], 1.15, 1.0)
    s = summarize([a, b, c], 0.5)
    for key, want in (("completed", 2), ("failed", 1), ("short", 1), ("output_tokens", 6), ("input_tokens", 21),
                      ("input_checked", 1), ("input_mismatched", 1), ("input_off", 1), ("prefix_hits", 1),
                      ("prefix_tokens", 3)):
        check(key, s[key] == want, "= %r, want %r" % (s[key], want))
    for key, want in (("output_tok_s", 12.0), ("total_tok_s", 54.0), ("req_s", 4.0), ("ttft_mean", 0.15),
                      ("ttft_p50", 0.15), ("ttft_p90", 0.19), ("ttft_p99", 0.199), ("tpot_mean", (0.11 / 3 + 0.05) / 2),
                      ("tpot_p50", (0.11 / 3 + 0.05) / 2), ("itl_p50", 0.025), ("itl_p99", 0.0494),
                      ("e2e_p50", 0.23), ("e2e_p99", 0.2496), ("input_mean", 10.5), ("output_mean", 3.0),
                      ("reused_mean", 3.0)):
        near(key, s[key], want)
    check("dropped arrival", b["token_events"] == 2 and len(b["itl"]) == 1, b)
    check("errors", s["errors"] == {"HTTP 503": 1}, repr(s["errors"]))
    lines = notes("x", s)
    check("notes", any("1 failed: HTTP 503" in n for n in lines) and any("1 prompts are off" in n for n in lines), lines)
    # A send lag of 20 ms is noted at 8 requests a second and at inf, not at 0.5 a second, where arrivals are 2 s apart.
    lagged = dict(s, lag_p50=0.005, lag_p99=0.02, lag_max=0.021)
    check("lag notes", [any("behind their arrival times" in n for n in notes("x", lagged, lag_limit(r)))
                        for r in (8.0, float("inf"), 0.5)] == [True, True, False] and
          not any("behind" in n for n in notes("x", lagged)))

    # The best round: a failure-free round wins over a faster round with a failure.
    rounds = [{"failed": 1, "output_tok_s": 10.0}, {"failed": 0, "output_tok_s": 9.0}, {"failed": 0, "output_tok_s": 8.0}]
    check("best round", best_round(rounds) == 1 and best_round(rounds[:1]) == 0 and
          best_round([{"failed": 2, "output_tok_s": 1.0}, {"failed": 1, "output_tok_s": 2.0}]) == 1)

    # Arrivals: inf all at once, and a seeded Poisson process.
    check("arrivals", arrivals(float("inf"), 3, 0) == [0.0, 0.0, 0.0])
    rng = random.Random("7/arrivals/2.0")
    want = [0.0]
    for _ in range(3):
        want.append(want[-1] + rng.expovariate(2.0))
    check("poisson", arrivals(2.0, 4, 7) == want[:4])

    # Prompt lengths come from the seed alone: every round and level, and a server whose prompts cost a start token more, get the same lengths, from different words.
    args = argparse.Namespace(seed=0, output_len=8, tokens=64, input_len=None, input_len_range=(64, 1024))
    w0, w1 = Workload(args, Lengths("two probe requests", 0, 1.0, True)), Workload(args, Lengths("two probe requests", 1, 1.0, True))
    r0, r1, r2 = w0.specs("closed/8/0", 8), w0.specs("closed/8/1", 8), w1.specs("closed/8/0", 8)
    targets = [sp["input_target"] for sp in r0]
    check("lengths across rounds and servers", targets == [sp["input_target"] for sp in r1] ==
          [sp["input_target"] for sp in r2] == [sp["input_target"] for sp in w0.specs("open/4.0", 8)] and
          len(set(targets)) > 1, targets)
    check("words across rounds", all(x["prompt"] != y["prompt"] for x in r0 for y in r1) and w0.specs("closed/8/0", 8) == r0)
    check("counted lengths", all(sp["input_counted"] == sp["input_target"] == len(sp["prompt"].split()) for sp in r0) and
          all(sp["input_counted"] == sp["input_target"] == len(sp["prompt"].split()) + 1 for sp in r2))
    # A tokenize route that counts every word twice cannot reach an odd length: the prompt keeps the nearest count and says so.
    twice = Lengths("tokenize route", 0, 2.0, True, count=lambda text: 2 * len(text.split()))
    prompt, counted = twice.make(random.Random(1), 13)
    check("unreachable length", counted is not None and counted != 13 and counted == 2 * len(prompt.split()), counted)


def fake_checks(name, check):
    """The checks against FakeServer on one API: counts, events, failure reasons and ordering, with time bounds a busy machine cannot break."""
    ttft, itl, n = 0.05, 0.03, 5
    # A lower bound holds on any machine: the server writes no token before its time, and the client reads none before the server writes it.
    # The upper bound only catches a figure taken from the wrong end, such as a stream's close.
    loose = 2.0

    def within(label, values, lo):
        check(label, bool(values) and all(v is not None and lo - 0.001 <= v <= lo + loose for v in values),
              "= %r, want %.3f to %.3f" % (values, lo, lo + loose))

    # The reference server's tokenize route is /tokenize and llmx's /v1/tokenize; the OpenAI route's server here has none.
    route = {"completion": "/tokenize", "llmx": "/v1/tokenize"}.get(name)
    fake = FakeServer(ttft, itl, tokenize=route)
    try:
        target = Target(fake.url)
        api = Api(name, "fake" if name == "openai" else None)
        lengths = Lengths.calibrate(target, api, 10)
        check(name + " lengths", lengths.uniform and lengths.base == 0 and
              lengths.source == ("tokenize route" if route else "two probe requests"), lengths.json())
        args = argparse.Namespace(seed=3, output_len=n, tokens=64, input_len=12, input_len_range=None)
        work = Workload(args, lengths)
        specs = work.specs("closed/2/0", 4)
        check(name + " prompts", all(len(sp["prompt"].split()) == 12 and sp["input_counted"] == 12 for sp in specs) and
              len(set(sp["prompt"] for sp in specs)) == 4 and work.specs("closed/2/0", 4) == specs)
        before = health(target)
        records, wall = closed_level(target, api, specs, 2, (10, 30))
        s = summarize(records, wall, before, health(target))
        label = name + " closed"
        check(label + " counts", (s["completed"], s["failed"], s["short"], s["output_tokens"]) == (4, 0, 0, 4 * n), s)
        check(label + " events", all(r["token_events"] == n and len(r["itl"]) == n - 1 for r in records),
              [r["token_events"] for r in records])
        check(label + " input", (s["input_tokens"], s["input_mismatched"], s["input_off"]) == (48, 0, 0), s)
        check(label + " reused", (s["prefix_hits"], s["prefix_tokens"], s["reused_mean"], s["prefix_source"]) ==
              (4, 4 * FakeServer.REUSED, float(FakeServer.REUSED), "health" if name == "llmx" else "replies"), s)
        within(label + " ttft", [r["ttft"] for r in records], ttft)
        within(label + " e2e", [r["e2e"] for r in records], ttft + (n - 1) * itl)
        within(label + " connect", [r["connect"] for r in records], 0.0)
        check(label + " itl order", all(g >= 0 for r in records for g in r["itl"]))
        # Two users over four requests: two after two.
        within(label + " span", [wall], 2 * (ttft + (n - 1) * itl))
        # --output-len fixes the lengths by asking every request to ignore the end of text, and a --tokens request asks nothing.
        # The --tokens request comes from a workload built without --output-len, so the check covers the flag's reading as well as the request.
        check(label + " ignore_eos", fake.ignore_eos == [True] * 4, fake.ignore_eos)
        capped = Workload(argparse.Namespace(seed=3, output_len=None, tokens=n, input_len=12, input_len_range=None), lengths)
        send(target, api, capped.specs("closed/2/0", 1)[0], 0, (10, 30), time.perf_counter())
        check(name + " --tokens ignore_eos", fake.ignore_eos[4:] == [None], fake.ignore_eos)

        offsets = arrivals(40.0, 5, 3)
        records, wall = open_level(target, api, work.specs("open/40.0", 5), offsets, (10, 30))
        s = summarize(records, wall)
        label = name + " open"
        check(label + " counts", (s["completed"], s["failed"]) == (5, 0), s)
        check(label + " schedule", all(r["scheduled"] == o and r["lag"] >= -0.001 for r, o in zip(records, offsets)) and
              s["lag_max"] is not None, [(r["scheduled"], r["lag"]) for r in records])
        within(label + " ttft", [r["ttft"] for r in records], ttft)
        within(label + " span", [wall], offsets[-1] + ttft + (n - 1) * itl)

        if name == "llmx":
            # Failures are counted by reason: a refusal and a dropped stream; a short reply completes and is counted short.
            fake.plan = ["ok", "503", "cut", "short", "ok"]
            records, wall = closed_level(target, api, work.specs("failures", 5), 3, (10, 30))
            s = summarize(records, wall)
            label = name + " failures"
            check(label, (s["completed"], s["failed"], s["short"]) == (3, 2, 1), s)
            reasons = sorted(s["errors"])
            dropped = ("the connection closed before the reply ended", "the stream ended before its last event")
            check(label + " reasons", len(reasons) == 2 and any(r.startswith("HTTP 503") for r in reasons) and
                  any(r in dropped for r in reasons), reasons)
            short = [r for r in records if r["short"]]
            check(label + " short", len(short) == 1 and short[0]["output_tokens"] == n // 2 and short[0]["finish"] == "eos",
                  short)
            # A stream that falls silent after its first token fails on either limit, whichever is shorter, with that limit's reason.
            for limits, reason in (((0.25, 30), "nothing arrived for 0.25s"), ((30, 0.25), "the reply took over 0.25s")):
                fake.plan = ["stall"]
                r = send(target, api, work.specs("stall", 1)[0], 0, limits, time.perf_counter())
                check(label + " " + reason, r["error"] == reason and 0.25 - 0.005 <= r["e2e"] <= 0.25 + loose, r)
    finally:
        fake.close()


def main_checks(check):
    """The whole tool against FakeServer: a closed level of two rounds whose first has a refusal shows the clean round, lists the refusal below the table, writes it to --json and exits 3."""
    fake = FakeServer(0.02, 0.01, tokenize=None)
    fake.plan = ["503"]
    out, err = io.StringIO(), io.StringIO()
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = main(["--url", fake.url, "--concurrency", "2", "--rounds", "2", "--warmup", "0", "--output-len", "3",
                       "--json", path])
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    finally:
        fake.close()
        os.remove(path)
    text = out.getvalue()
    level = doc["levels"][0]
    check("main exit", rc == 3 and "error: 1 of 4 timed requests failed" in err.getvalue(), (rc, err.getvalue()))
    check("main round", level["best"] == 1 and [rd["summary"]["failed"] for rd in level["rounds"]] == [1, 0] and
          doc["failed"] == 1 and doc["requests"] == 4, (level["best"], doc["failed"]))
    check("main notes", "conc 2, round 1 of 2, not the one shown: 1 failed: HTTP 503" in text and
          "give --max-queue at least --num-prompts" in text, text)


def self_test():
    """The stream reading and the figures on made-up times, exactly, then every API in both loads against FakeServer, with failures, timeouts and short replies, then the whole tool; True when every check holds."""
    failed = []

    def check(name, ok, detail=""):
        if not ok:
            failed.append("%s %s" % (name, detail))

    exact_checks(check)
    for name in ("llmx", "openai", "completion"):
        fake_checks(name, check)
    main_checks(check)
    for line in failed:
        print("  self-test: " + line)
    print("server_load self-test: streams and figures on made-up times; closed and open loads on llmx, openai and "
          "completion against a server with known token times; prompts counted through /tokenize and /v1/tokenize; failures by reason, timeouts, a short reply and the "
          "exit status  [%s]" % ("ok" if not failed else "FAILED"))
    return not failed


if __name__ == "__main__":
    sys.exit(main())
