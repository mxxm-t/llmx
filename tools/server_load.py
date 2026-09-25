"""A server under load: the latency and throughput figures a serving runtime is judged by.

Standard library only.
Every request streams its reply, so the arrival of every token is timed, and samples greedily, so the work of a request is fixed.

Load:

    closed loop  --concurrency C ...: at each level C users each send a request and send their next when its reply ends, --num-prompts requests in all (C by default, one a user), repeated --rounds times; the round with the most output tokens per second is reported
    open loop    --rate R ...: --num-prompts requests (100 by default) arrive at R a second as a Poisson process drawn from --seed, each sent when it arrives whatever is still running, as a reference serving benchmark sends them; inf sends them all at once

With neither flag the closed loop runs at 1, 2, 4, 8, 16, 32 and 64 users; with --rate alone only the rates run.

Workload:

    prompts  by default the eight short fixed prompts in turn; with --input-len N every request its own prompt of N tokens, the word "the" and then random words from a fixed list, drawn from --seed per level and round, so no two requests of a run share a prefix (a rerun with the same seed repeats them); with --input-len-range A:B the lengths uniform from A to B, drawn from --seed
    replies  by default up to --tokens N tokens (64), a reply ending at the end of text if the model emits it first; with --output-len N in its place every request asks for exactly N tokens and to ignore the end of text, and a reply that ends short is counted as short

The defaults are the earlier load of this tool, and the earlier flags keep their meaning.

A prompt's length counts every token the server reads for it, a start token it adds included.
Where the server has a tokenize route (POST /tokenize), each prompt is counted there and a word added or dropped until the count is exact.
Otherwise two requests for one token, on the word list once and twice, show whether every word is one token and what a prompt costs beyond its words; when every word is one token, a prompt of k words has a known length.
Where a reply reports its prompt tokens, the achieved lengths are checked against the target and the mismatches reported.

--output-len sends the cap with ignore_eos set on every route: max_tokens on llmx's /v1/generate and on the OpenAI route, n_predict on the reference server's /completion.
A server that does not honour ignore_eos can still end a reply at its end of text, which the short count shows.

The APIs: --api llmx speaks /v1/generate, --api openai /v1/completions (with the model id /v1/models lists, or --model), and --api completion the reference server's /completion route, so the same load compares servers on the same model and card.

Figures per level, over the completed requests:

    tok/s        output tokens per second, from the level's first send to its last reply's end
    req/s        completed requests per second over the same span
    ttft         time to first token, from a request's send to its first streamed token: mean, median, 90th and 99th percentile
    itl          inter-token latency, the gap between consecutive token events of a request, over all requests: median and 99th percentile
    all tok/s    prompt and output tokens per second over the level's span
    tpot         time per output token after the first, per request (last token - first token) / (tokens - 1): mean, median and 99th percentile
    e2e          end-to-end latency, from a request's send to the end of its reply: median and 99th percentile
    in, out      mean prompt and output tokens of a completed request, as the replies report them or as the prompts were counted
    done, fail   completed and failed requests; a request fails on an HTTP error, a refused or dropped connection, an error event, a stream that ends before its last event, or no reply within --timeout, and every failure is counted and its reason listed
    short        completed requests with fewer tokens than they asked for
    hits         requests that reused a cached prefix, from the replies (timings.cache_n) or from the difference of the server's /v1/health counters around the level

--warmup requests at the longest prompt and the full reply length run before any level, so the first level does not carry the server's first pass or its first growth of scratch.
--json writes every request's record (send time, time to first token, gaps, tokens, failure reason) and every level's figures, rewritten after each level.
--self-test checks the figures against an in-process server whose tokens arrive at known times, on all three APIs and both loads, with failures, a timeout and short replies, and needs no model.

    python tools/server_load.py --url http://127.0.0.1:8080 --concurrency 1 4 8 16 --tokens 64
    python tools/server_load.py --input-len 128 --output-len 128 --concurrency 1 2 4 8 16 32 64 --num-prompts 128 --json closed.json
    python tools/server_load.py --input-len-range 64:1024 --output-len 128 --rate 1 2 4 8 inf --num-prompts 200 --seed 1 --json open.json
"""
import argparse
import datetime
import http.client
import json
import math
import os
import random
import socket
import sys
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
            # A chunk carries a token unless it is the finish chunk with no text of its own.
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
            "output_reported": None, "reused": None, "finish": None}


def finish_record(rec, sent, arrivals, end, t0):
    """A request's figures from its send time, its token arrival times and its end, all on the perf_counter clock; `t0` is its level's start."""
    tokens = rec["output_reported"] if rec["output_reported"] is not None else len(arrivals)
    rec["sent"] = sent - t0
    rec["token_events"] = len(arrivals)
    rec["output_tokens"] = tokens
    rec["input_tokens"] = rec["input_reported"] if rec["input_reported"] is not None else rec["input_counted"]
    rec["ttft"] = arrivals[0] - sent if arrivals and tokens > 0 else None
    rec["tpot"] = (arrivals[-1] - arrivals[0]) / (tokens - 1) if tokens > 1 and len(arrivals) > 1 else None
    rec["itl"] = [b - a for a, b in zip(arrivals, arrivals[1:])]
    rec["e2e"] = end - sent
    rec["short"] = rec["ok"] and tokens < rec["output_target"]
    return rec


def send(target, api, spec, index, timeout, t0):
    """One streamed request, timed; a failure is recorded with its reason, never raised."""
    rec = new_record(index, spec)
    path, body = api.request(spec["prompt"], spec["tokens"], spec["force"])
    data = json.dumps(body).encode("utf-8")
    arrivals = []
    conn = target.connection(timeout)
    sent = time.perf_counter()
    deadline = sent + timeout
    end = None
    try:
        conn.request("POST", target.base + path, body=data, headers={"Content-Type": "application/json"})
        sock = conn.sock
        # Every blocking read waits at most for what is left of the request's time.
        sock.settimeout(max(deadline - time.perf_counter(), 0.001))
        r = conn.getresponse()
        if r.status != 200:
            raise Failure("HTTP %d %s" % (r.status, r.read(200).decode("utf-8", "replace").strip()))
        ended = False
        while True:
            left = deadline - time.perf_counter()
            if left <= 0:
                raise socket.timeout()
            sock.settimeout(left)
            raw = r.readline()
            now = time.perf_counter()
            if not raw:
                break
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                ended = ended or api.name == "openai"
                break
            token, last = api.read(json.loads(payload.decode("utf-8")), rec)
            if token:
                arrivals.append(now)
            ended = ended or last
        end = time.perf_counter()
        if not ended:
            raise Failure("the stream ended before its last event")
        rec["ok"] = True
    except socket.timeout:
        rec["error"] = "no reply within the %gs timeout" % timeout
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


def tokenize_route(target, timeout):
    """A counter through the server's tokenize route, or None when it has none: the body carries the field names both common tokenize routes read, with the start token counted."""
    def count(text):
        status, reply = target.call("POST", "/tokenize", {"content": text, "prompt": text, "add_special": True,
                                                          "add_special_tokens": True}, timeout)
        if status == 200 and isinstance(reply, dict) and isinstance(reply.get("tokens"), list):
            return len(reply["tokens"])
        return None
    return count if count(WORDS[0]) is not None else None


class Lengths:
    """Prompts of a token count, built from WORDS, and how the server under test counts them."""

    def __init__(self, target, api, timeout):
        self.count = tokenize_route(target, timeout)
        counter = self.count or (lambda text: api.prompt_count(target, text, timeout))
        self.source = "tokenize route" if self.count else "two probe requests"
        once, twice = counter(" ".join(WORDS)), counter(" ".join(WORDS + WORDS))
        if once is None or twice is None or twice <= once:
            # Nothing reports a count: a prompt of N tokens is N words, unverified.
            self.source, self.base, self.per_word, self.uniform = None, 0, 1.0, None
            self.probes = None
        else:
            # A word list read twice costs its words once more: the difference is what the words cost, the rest is what a prompt costs beyond them.
            self.per_word = (twice - once) / len(WORDS)
            self.base = 2 * once - twice
            self.uniform = twice - once == len(WORDS)
            self.probes = [once, twice]

    def smallest(self):
        return self.base + 1

    def make(self, rng, n):
        """A prompt of n tokens and its count, or None when nothing counts it."""
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

    def specs(self, key, n, longest=False):
        """n requests, the same for the same seed and key; `longest` gives every one the longest prompt length."""
        rng = random.Random("%d/%s" % (self.seed, key))
        out = []
        for i in range(n):
            if self.range is None:
                prompt, target, counted = PROMPTS[i % len(PROMPTS)], None, None
            else:
                target = self.range[1] if longest else rng.randint(*self.range)
                prompt, counted = self.lengths.make(rng, target)
            out.append({"prompt": prompt, "input_target": target, "input_counted": counted, "tokens": self.tokens,
                        "force": self.force})
        return out

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


def closed_level(target, api, specs, concurrency, timeout):
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
            records[i] = send(target, api, specs[i], i, timeout, t0)

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


def open_level(target, api, specs, offsets, timeout):
    """specs sent at their offsets whatever is still running: the records and the level's span."""
    records = [None] * len(specs)

    def one(i):
        records[i] = send(target, api, specs[i], i, timeout, t0)

    threads = []
    t0 = time.perf_counter()
    for i, offset in enumerate(offsets):
        delay = t0 + offset - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        t = threading.Thread(target=one, args=(i,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    for rec, offset in zip(records, offsets):
        rec["scheduled"] = offset
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
         "prefix_hits": None, "prefix_tokens": None, "prefix_source": None}
    reused = [r["reused"] for r in done if r["reused"] is not None]
    if reused:
        s["prefix_hits"], s["prefix_tokens"], s["prefix_source"] = sum(1 for n in reused if n > 0), sum(reused), "replies"
    elif before is not None and after is not None:
        s["prefix_hits"] = after["prefix_hits"] - before["prefix_hits"]
        s["prefix_tokens"] = after.get("prefix_tokens", 0) - before.get("prefix_tokens", 0)
        s["prefix_source"] = "health"
    # The prompt counts the replies report, against the lengths asked.
    reported = [r for r in done if r["input_reported"] is not None and r["input_target"] is not None]
    s["input_checked"] = len(reported)
    s["input_mismatched"] = sum(1 for r in reported if r["input_reported"] != r["input_target"])
    s["errors"] = {}
    for r in records:
        if not r["ok"]:
            s["errors"][r["error"]] = s["errors"].get(r["error"], 0) + 1
    return s


HEAD = "%-10s %5s %9s %8s %10s %10s %10s %10s" + " %9s %10s %10s %10s %10s %10s %10s %10s %6s %6s %5s %4s %5s %5s"
COLUMNS = ("tok/s", "req/s", "ttft p50", "ttft p99", "itl p50", "itl p99", "all tok/s", "ttft mean", "ttft p90",
           "tpot mean", "tpot p50", "tpot p99", "e2e p50", "e2e p99", "in", "out", "done", "fail", "short", "hits")


def header(level_name):
    return HEAD % (("api", level_name) + COLUMNS)


def row(api, level, s):
    ms = lambda v: v * 1e3
    hits = "-" if s["prefix_hits"] is None else str(s["prefix_hits"])
    # The first eight columns keep the earlier table's layout.
    return ("%-10s %5s %9.1f %8.2f %8.1f ms %8.1f ms %8.2f ms %8.2f ms" % (
                api, level, s["output_tok_s"], s["req_s"], ms(s["ttft_p50"]), ms(s["ttft_p99"]), ms(s["itl_p50"]),
                ms(s["itl_p99"])) +
            " %9.1f %8.1f ms %8.1f ms %8.2f ms %8.2f ms %8.2f ms %8.1f ms %8.1f ms %6.1f %6.1f %5d %4d %5d %5s" % (
                s["total_tok_s"], ms(s["ttft_mean"]), ms(s["ttft_p90"]), ms(s["tpot_mean"]), ms(s["tpot_p50"]),
                ms(s["tpot_p99"]), ms(s["e2e_p50"]), ms(s["e2e_p99"]), s["input_mean"], s["output_mean"],
                s["completed"], s["failed"], s["short"], hits))


def notes(label, s):
    """The lines a level adds below the table: its failures by reason, short replies, and prompt counts off their target."""
    out = []
    for reason, n in sorted(s["errors"].items(), key=lambda kv: -kv[1]):
        out.append("%s: %d failed: %s" % (label, n, reason))
    if s["short"]:
        out.append("%s: %d of %d replies ended before their length" % (label, s["short"], s["completed"]))
    if s["input_mismatched"]:
        out.append("%s: %d of %d replies report a prompt count off its target" % (label, s["input_mismatched"],
                                                                                 s["input_checked"]))
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
    p.add_argument("--seed", type=int, default=0, help="seeds the prompts, their lengths and the arrivals (default 0)")
    p.add_argument("--rounds", type=positive, default=2,
                   help="repeats of each closed-loop level; the best by output tok/s is reported (default 2)")
    p.add_argument("--warmup", type=int, default=1, help="requests before any level, one at a time (default 1)")
    p.add_argument("--timeout", type=float, default=600,
                   help="seconds a timed request may take before it fails (default 600); the calibration and warm-up wait at least 600")
    p.add_argument("--json", default=None, metavar="PATH", help="write every request's record and every level's figures")
    p.add_argument("--self-test", action="store_true", help="check the figures against an in-process server and exit")
    args = p.parse_args(argv)
    if args.input_len is not None and args.input_len_range is not None:
        p.error("--input-len and --input-len-range are two ways to say one thing; give one")
    if args.output_len is not None and args.tokens is not None:
        p.error("--output-len replaces --tokens; give one")
    if args.tokens is None:
        args.tokens = 64
    if args.warmup < 0:
        p.error("--warmup must be 0 or more")
    if not args.timeout > 0:
        p.error("--timeout must be above 0")
    return args


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args = parse(argv)
    if args.self_test:
        return 0 if self_test() else 1
    target = Target(args.url)
    levels = args.concurrency if args.concurrency is not None else ([] if args.rate else DEFAULT_LEVELS)
    rates = args.rate or []
    # The calls before the timed levels, the calibration and the warm-up, are not what --timeout bounds.
    setup = max(args.timeout, 600)
    try:
        model = args.model
        if args.api == "openai" and model is None:
            status, models = target.call("GET", "/v1/models", timeout=30)
            if status == 200 and isinstance(models, dict) and models.get("data"):
                model = models["data"][0].get("id")
        api = Api(args.api, model)
        lengths = None
        if args.input_len is not None or args.input_len_range is not None:
            lengths = Lengths(target, api, setup)
            want = args.input_len if args.input_len is not None else args.input_len_range[0]
            if want < lengths.smallest():
                sys.exit("error: a prompt here is at least %d tokens, above the %d asked" % (lengths.smallest(), want))
    except (OSError, http.client.HTTPException) as e:
        sys.exit("error: cannot reach %s: %s" % (args.url, e))
    work = Workload(args, lengths)
    doc = {"tool": "tools/server_load.py", "schema": 2, "started": datetime.datetime.now().isoformat(timespec="seconds"),
           "url": args.url, "api": args.api, "model": model,
           "args": dict(vars(args), rate=[r if math.isfinite(r) else "inf" for r in rates]), "workload": work.describe(),
           "lengths": lengths.json() if lengths else None, "warmup": [], "levels": []}
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

    pending = []
    if levels:
        print(header("conc"))
        sys.stdout.flush()
    for c in levels:
        n = args.num_prompts if args.num_prompts is not None else c
        rounds, best = [], None
        for k in range(args.rounds):
            specs = work.specs("closed/%d/%d" % (c, k), n)
            before = health(target)
            records, wall = closed_level(target, api, specs, c, args.timeout)
            s = summarize(records, wall, before, health(target))
            rounds.append({"round": k, "summary": s, "records": records})
            if best is None or s["output_tok_s"] > rounds[best]["summary"]["output_tok_s"]:
                best = k
        s = rounds[best]["summary"]
        print(row(args.api, str(c), s))
        sys.stdout.flush()
        pending += notes("conc %d" % c, s)
        doc["levels"].append({"mode": "closed", "concurrency": c, "requests": n, "best": best, "summary": s,
                              "rounds": rounds})
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
        records, wall = open_level(target, api, specs, arrivals(r, n, args.seed), args.timeout)
        s = summarize(records, wall, before, health(target))
        print(row(args.api, "%g" % r, s))
        sys.stdout.flush()
        pending += notes("rate %g" % r, s)
        doc["levels"].append({"mode": "open", "rate": r if math.isfinite(r) else "inf", "requests": n, "summary": s, "records": records})
        if args.json:
            write_json(args.json, doc)
    for line in pending:
        print(line)
    return 0


class FakeServer:
    """A server for --self-test that speaks the three APIs, counts a prompt's tokens as its words, and sends each streamed token at a known time after the request arrives: the first after `ttft` seconds and one every `itl` after it.
    Streamed requests take their behaviour from `plan` in arrival order: ok, 503, cut (the connection drops after two tokens), short (half the tokens, then the end of text) or stall (a second's silence after the first token).
    Every streamed reply reports 4 reused prompt tokens, and /v1/health counts them."""

    REUSED = 4

    def __init__(self, ttft, itl, tokenize):
        import http.server
        fake = self
        self.ttft, self.itl, self.tokenize = ttft, itl, tokenize
        self.plan = []
        self.lock = threading.Lock()
        self.hits = 0
        self.reused = 0

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
                if self.path == "/tokenize" and fake.tokenize:
                    return self.reply(200, {"tokens": list(range(len(body["content"].split())))})
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
                            time.sleep(1.0)
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

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.url = "http://127.0.0.1:%d" % self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def self_test():
    """The figures on made-up arrival times, exactly, then against FakeServer on every API in both loads, with failures, a timeout and short replies; True when every check holds."""
    failed = []

    def check(name, ok, detail=""):
        if not ok:
            failed.append("%s %s" % (name, detail))

    # Exact arithmetic: two completed requests and a failed one on a level of 0.5 s.
    spec = {"input_target": 10, "input_counted": 10, "tokens": 4, "force": True, "prompt": ""}
    a = new_record(0, spec)
    a["ok"] = True
    finish_record(a, 1.0, [1.1, 1.12, 1.15, 1.2], 1.21, 1.0)
    b = new_record(1, spec)
    b.update(ok=True, output_reported=2, input_reported=11, reused=3)
    finish_record(b, 1.05, [1.25, 1.26], 1.3, 1.0)
    c = new_record(2, spec)
    c["error"] = "HTTP 503"
    finish_record(c, 1.1, [], 1.15, 1.0)
    s = summarize([a, b, c], 0.5)
    for key, want in (("completed", 2), ("failed", 1), ("short", 1), ("output_tokens", 6), ("input_tokens", 21),
                      ("input_checked", 1), ("input_mismatched", 1), ("prefix_hits", 1), ("prefix_tokens", 3)):
        check(key, s[key] == want, "= %r, want %r" % (s[key], want))
    for key, want in (("output_tok_s", 12.0), ("total_tok_s", 54.0), ("req_s", 4.0), ("ttft_mean", 0.15),
                      ("ttft_p50", 0.15), ("ttft_p90", 0.19), ("ttft_p99", 0.199), ("tpot_mean", (0.1 / 3 + 0.01) / 2),
                      ("tpot_p50", (0.1 / 3 + 0.01) / 2), ("itl_p50", 0.025), ("itl_p99", 0.0494),
                      ("e2e_p50", 0.23), ("e2e_p99", 0.2496), ("input_mean", 10.5), ("output_mean", 3.0)):
        check(key, abs(s[key] - want) < 1e-9, "= %r, want %r" % (s[key], want))
    check("errors", s["errors"] == {"HTTP 503": 1}, repr(s["errors"]))
    check("arrivals", arrivals(float("inf"), 3, 0) == [0.0, 0.0, 0.0])
    rng = random.Random("7/arrivals/2.0")
    want = [0.0]
    for _ in range(3):
        want.append(want[-1] + rng.expovariate(2.0))
    check("poisson", arrivals(2.0, 4, 7) == want[:4])

    # Against the fake server: the first token 50 ms after a request arrives, one every 30 ms after it, 5 tokens, so 170 ms a request.
    # The slack of each timing check is under one gap, so a figure taken from the wrong token misses.
    # A machine busy with other work can wake a thread late and put a level past its slack, so each API's checks get three attempts and pass on the first clean one; a fault in the tool fails all three.
    for name in ("llmx", "openai", "completion"):
        for _ in range(3):
            misses = fake_checks(name)
            if not misses:
                break
        failed += misses
    for line in failed:
        print("  self-test: " + line)
    print("server_load self-test: exact figures on made-up times; closed and open loads on llmx, openai and completion "
          "against a server with known token times; failures by reason, a timeout and a short reply  [%s]"
          % ("ok" if not failed else "FAILED"))
    return not failed


def fake_checks(name):
    """The checks of self_test against FakeServer on one API: what missed, empty when all held."""
    ttft, itl, n = 0.05, 0.03, 5
    slack = 0.025
    failed = []

    def check(label, ok, detail=""):
        if not ok:
            failed.append("%s %s" % (label, detail))

    def near(label, value, want, above):
        check(label, value is not None and want - 0.002 <= value <= want + above,
              "= %s, want %.4f (-0.002/+%.3f)" % (value, want, above))

    fake = FakeServer(ttft, itl, tokenize=name == "completion")
    try:
        target = Target(fake.url)
        api = Api(name, "fake" if name == "openai" else None)
        lengths = Lengths(target, api, 10)
        check(name + " lengths", lengths.uniform and lengths.base == 0 and
              lengths.source == ("tokenize route" if name == "completion" else "two probe requests"), lengths.json())
        args = argparse.Namespace(seed=3, output_len=n, tokens=64, input_len=12, input_len_range=None)
        work = Workload(args, lengths)
        specs = work.specs("closed/2/0", 4)
        check(name + " prompts", all(len(sp["prompt"].split()) == 12 and sp["input_counted"] == 12 for sp in specs) and
              len(set(sp["prompt"] for sp in specs)) == 4 and work.specs("closed/2/0", 4) == specs)
        before = health(target)
        records, wall = closed_level(target, api, specs, 2, 10)
        s = summarize(records, wall, before, health(target))
        label = name + " closed"
        check(label + " counts", (s["completed"], s["failed"], s["short"], s["output_tokens"]) == (4, 0, 0, 4 * n), s)
        check(label + " events", all(r["token_events"] == n for r in records), [r["token_events"] for r in records])
        check(label + " input", (s["input_tokens"], s["input_mismatched"]) == (48, 0), (s["input_tokens"], s["input_mismatched"]))
        check(label + " hits", (s["prefix_hits"], s["prefix_tokens"], s["prefix_source"]) ==
              (4, 4 * FakeServer.REUSED, "health" if name == "llmx" else "replies"),
              (s["prefix_hits"], s["prefix_tokens"], s["prefix_source"]))
        near(label + " ttft p50", s["ttft_p50"], ttft, slack)
        near(label + " itl p50", s["itl_p50"], itl, slack / 2)
        near(label + " tpot p50", s["tpot_p50"], itl, slack / 2)
        near(label + " e2e p50", s["e2e_p50"], ttft + (n - 1) * itl, 2 * slack)
        # Two users over four requests of 170 ms: two after two.
        near(label + " span", wall, 2 * (ttft + (n - 1) * itl), 4 * slack)

        offsets = arrivals(40.0, 5, 3)
        records, wall = open_level(target, api, work.specs("open/40.0", 5), offsets, 10)
        s = summarize(records, wall)
        label = name + " open"
        check(label + " counts", (s["completed"], s["failed"]) == (5, 0), s)
        check(label + " schedule", all(r["scheduled"] == o and r["sent"] >= o - 0.001 for r, o in zip(records, offsets)))
        near(label + " ttft p50", s["ttft_p50"], ttft, slack)
        near(label + " span", wall, offsets[-1] + ttft + (n - 1) * itl, 4 * slack)

        if name == "llmx":
            # Failures are counted by reason: a refusal, a dropped stream and a stall past the timeout; a short reply completes and is counted short.
            fake.plan = ["ok", "503", "cut", "short", "ok", "stall"]
            records, wall = closed_level(target, api, work.specs("failures", 6), 3, 0.4)
            s = summarize(records, wall)
            label = name + " failures"
            check(label, (s["completed"], s["failed"], s["short"]) == (3, 3, 1), s)
            reasons = sorted(s["errors"])
            dropped = ("the connection closed before the reply ended", "the stream ended before its last event")
            check(label + " reasons", len(reasons) == 3 and any(r.startswith("HTTP 503") for r in reasons) and
                  "no reply within the 0.4s timeout" in reasons and any(r in dropped for r in reasons), reasons)
            short = [r for r in records if r["short"]]
            check(label + " short", len(short) == 1 and short[0]["output_tokens"] == n // 2 and short[0]["finish"] == "eos",
                  short)
            check(label + " timeout", all(r["e2e"] < 0.4 + slack for r in records if r["error"] and "timeout" in r["error"]))
    finally:
        fake.close()
    return failed


if __name__ == "__main__":
    sys.exit(main())
