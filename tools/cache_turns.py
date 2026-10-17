"""A long chat against a running `llmx serve`, at production's proportions: an unrelated long task after every reply, no idle moment of five seconds, a cut, a restart (docs/CACHE-PLAN.md, Production since 2026-10-11).

`drive` writes OUT/events.jsonl, per turn the rows a restore point held (the shared prefix with the turn before's history, in whole blocks), the rows reused and lost, the losses the server counted since the turn before ended
(`reasons` where the turn lost rows, `losses_elsewhere` where it lost none), the disk bytes written and read, the read's wait and the time to first token; per task and other reply their time to first token and the p99 of their gaps.
It also writes OUT/turns.jsonl, each turn's request and reply, which `verify` sends to a fresh server and holds every reply equal to (P9).
"""
import argparse
import http.client
import json
import os
import subprocess
import sys
import threading
import time

LOSSES = ("job_lost", "consumed_lost", "not_to_host", "host_lost", "boundary_lost", "read_failed", "read_refused", "disk.lost_before_written")
BLOCK = 64


class Server:
    def __init__(self, port):
        self.port = port

    def call(self, method, path, body=None, timeout=7200):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        c.request(method, path, json.dumps(body).encode() if body is not None else None, {"Content-Type": "application/json"})
        r = c.getresponse()
        data = r.read()
        c.close()
        return r.status, json.loads(data.decode("utf-8", "replace")) if data else None

    def health(self):
        return self.call("GET", "/v1/health", timeout=30)[1]

    def counters(self):
        h = self.health()
        out = {k: v for k, v in h["reuse"]["moves"]["since_start"].items() if v}
        for group, values in (("reread", h["reread"]["since_start"]), ("disk", h["reuse"]["disk"]["since_start"]), ("background", h["pacing"].get("background", {}).get("since_start", {}))):
            out.update({group + "." + k: v for k, v in values.items() if isinstance(v, (int, float)) and v})
        return out

    def ready(self):
        while True:
            try:
                if self.health():
                    return
            except Exception:
                pass
            time.sleep(2)

    def stream(self, body):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=7200)
        start = time.time()
        c.request("POST", "/v1/chat/completions", json.dumps(dict(body, stream=True)).encode(), {"Content-Type": "application/json"})
        r = c.getresponse()
        buf, text, arrivals, last = b"", "", [], None
        while True:
            chunk = r.read1(65536)
            if not chunk:
                break
            buf += chunk
            while b"\n\n" in buf:
                event, buf = buf.split(b"\n\n", 1)
                line = event.decode("utf-8", "replace")
                if not line.startswith("data: ") or line[6:] == "[DONE]":
                    continue
                e = json.loads(line[6:])
                if e.get("choices"):
                    d = e["choices"][0].get("delta", {})
                    if d.get("content") or d.get("reasoning_content"):
                        arrivals.append(time.time() - start)
                    text += d.get("content") or ""
                    if e["choices"][0].get("finish_reason"):
                        last = e
        c.close()
        gaps = sorted(b - a for a, b in zip(arrivals, arrivals[1:]))
        p99 = round(gaps[min(len(gaps) - 1, int(0.99 * len(gaps)))] * 1000, 1) if gaps else None
        return text, (arrivals[0] if arrivals else None), p99, (last or {}).get("timings")


def delta(a, b):
    return {k: round(b.get(k, 0) - a.get(k, 0), 3) for k in sorted(set(a) | set(b)) if b.get(k, 0) != a.get(k, 0)}


def shared(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def drive(args):
    server = Server(args.port)
    text_source = open(args.text, encoding="utf-8").read()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    lock = threading.Lock()

    def log(kind, name="events.jsonl", **fields):
        fields.update(kind=kind, t=round(time.time() - t0, 3))
        with lock:
            with open(os.path.join(args.out, name), "a") as f:
                f.write(json.dumps(fields) + "\n")

    def words(offset, tokens):
        s = text_source[offset % len(text_source):] + text_source
        return s[:tokens * 4]

    def tokens_of(messages):
        st, got = server.call("POST", "/v1/tokenize", {"messages": messages})
        return got["tokens"] if st == 200 else None

    def task(turn):
        body = {"messages": [{"role": "user", "content": "### Task:\nGenerate a concise title for the text below.\n\n" + words(600000 + turn * 70000, args.task_tokens)}], "max_tokens": 64, "temperature": 0}
        _, ttft, p99, timings = server.stream(body)
        log("task", turn=turn, reused=(timings or {}).get("cache_n"), ttft=ttft, gap_p99_ms=p99)

    def other(turn):
        _, ttft, p99, _ = server.stream({"messages": [{"role": "user", "content": "Write a long story about a lighthouse keeper, part %d." % turn}], "max_tokens": 256, "temperature": 0})
        log("other", turn=turn, ttft=ttft, gap_p99_ms=p99)

    messages = []
    for k in range(args.early):
        messages.append({"role": "user", "content": "Summarize this passage in two sentences.\n\n" + words(k * 20000, 4500)})
        messages.append({"role": "assistant", "content": "The passage describes events and people in some detail; it gives dates, places and outcomes."})
    server.ready()
    previous = None
    # The losses of a turn's history happen between the turn before's reply and the turn (in the task's window), so they are read from the turn before's end.
    since_last = server.counters()
    for turn in range(args.turns):
        messages.append({"role": "user", "content": "Summarize this passage in two sentences.\n\n" + words(400000 + turn * 13000, 2500)})
        ids = tokens_of(messages)
        held = shared(previous, ids) // BLOCK * BLOCK if previous and ids else 0
        body = {"messages": list(messages), "max_tokens": args.reply, "temperature": 0}
        text, ttft, p99, timings = server.stream(body)
        after = server.counters()
        reused = (timings or {}).get("cache_n") or 0
        since = delta(since_last, after)
        losses = {k: v for k, v in since.items() if k in LOSSES or k.startswith("disk.room_wait")}
        lost = max(0, held - reused)
        log("turn", turn=turn, rendered=len(ids) if ids else None, held=held, reused=reused, lost=lost, cut=bool(ids and len(ids) > args.ctx), ttft=ttft, gap_p99_ms=p99,
            disk_bytes_written=since.get("disk.bytes_written", 0), disk_bytes_read=since.get("disk.bytes_read", 0), disk_wait_ms=since.get("disk.wait_ms", 0),
            reasons=losses if lost else {}, losses_elsewhere={} if lost else losses, moves=since)
        log("turn", name="turns.jsonl", turn=turn, request=body, reply=text)
        since_last = after
        messages.append({"role": "assistant", "content": text})
        previous = tokens_of(messages + [{"role": "user", "content": "x"}])
        if args.delay >= 0:
            time.sleep(args.delay)
            threads = [threading.Thread(target=task, args=(turn,))] + ([threading.Thread(target=other, args=(turn,))] if args.other else [])
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        if turn == args.restart_after:
            carried = delta(since_last, server.counters())
            rc = subprocess.call(args.restart, shell=True)
            server.ready()
            # The new server counts from zero: what the old one counted since the turn before ended starts the next turn's window.
            since_last = {k: -v for k, v in carried.items()}
            log("restarted", rc=rc)
        else:
            time.sleep(3)
    log("final", health=server.health())


def verify(args):
    server = Server(args.port)
    failed = 0
    for line in open(args.turns_jsonl, encoding="utf-8"):
        rec = json.loads(line)
        st, got = server.call("POST", "/v1/chat/completions", rec["request"])
        text = ((got or {}).get("choices") or [{}])[0].get("message", {}).get("content") or ""
        same = st == 200 and text == rec["reply"]
        failed += not same
        print("turn %d: %s" % (rec["turn"], "same" if same else "DIFFERS"), flush=True)
    print("P9: %s" % ("every reply equal" if not failed else "%d replies differ" % failed))
    return 1 if failed else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    d = sub.add_parser("drive", help="the chat with its tasks, a cut and a restart")
    d.add_argument("--port", type=int, required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--early", type=int, default=12, help="earlier turns of about 4.5k tokens the chat starts with")
    d.add_argument("--turns", type=int, default=7)
    d.add_argument("--restart-after", type=int, default=-1, help="the turn after whose task the server restarts, -1 for none")
    d.add_argument("--restart", default="", help="a shell command that stops the server cleanly and starts it again on the same port and directories")
    d.add_argument("--delay", type=float, default=2.0, help="seconds from a reply's end to its task, 0 at once, below 0 no task")
    d.add_argument("--other", action="store_true", help="another user's reply beside each task")
    d.add_argument("--ctx", type=int, default=65536, help="the pool's tokens, which tell a cut turn")
    d.add_argument("--task-tokens", type=int, default=14000)
    d.add_argument("--reply", type=int, default=384)
    d.add_argument("--text", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "data", "wiki.test.raw"))
    v = sub.add_parser("verify", help="every recorded turn against a fresh server")
    v.add_argument("--port", type=int, required=True)
    v.add_argument("turns_jsonl")
    args = ap.parse_args()
    if args.command == "drive":
        drive(args)
        return 0
    return verify(args)


if __name__ == "__main__":
    sys.exit(main())
