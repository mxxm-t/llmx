"""Many users at once through `llmx serve`, every request checked against
what it gives alone: the check that a placement, a layer split above all,
holds under whatever mix of prompts and decodes a server meets.

Standard library only. Every request is greedy. The requests mix short and
long prompts, cut from a text file, with short and long replies. Phases:

    alone     each request by itself, one after another: its reference ids
    together  every request at once, so prompts and decodes share passes
    skewed    the same requests arriving at staggered times, long prompts
              landing while others decode, and some clients leaving
              mid-stream
    cli       the first requests through `llmx generate --temp 0` on the
              same devices, whose prompt runs as one transaction and, on a
              split, pipelined over the stages

Every request that runs to its end must give its ids alone, the CLI its
text; a client that left must leave nothing active.

    python tools/server_mix_check.py --model M.gguf --text wiki.txt \\
        --device vulkan:0,vulkan:1,vulkan:2 --layer-shares 1,1,1
"""
import argparse
import json
import os
import random
import subprocess
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))
import common  # noqa: E402


def post(port, body, timeout=1800):
    req = urllib.request.Request("http://127.0.0.1:%d/v1/generate" % port, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def health(port):
    with urllib.request.urlopen("http://127.0.0.1:%d/v1/health" % port, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def requests_from(text, count, rng):
    """`count` requests: prompts of a sentence, a paragraph or pages, cut from `text` at random offsets, with replies of 8 to 128 tokens."""
    out = []
    for i in range(count):
        chars = [120, 1500, 6000, 12000][i % 4]
        start = rng.randrange(0, max(1, len(text) - chars))
        # A leading '-' would read as a flag to `llmx generate`.
        prompt = text[start:start + chars].lstrip("-")
        out.append({"prompt": prompt, "max_tokens": [128, 32, 64, 8][(i // 4) % 4], "temperature": 0})
    return out


def run_together(port, reqs, delays=None, leavers=()):
    results, threads = {}, []
    def worker(i):
        if delays:
            time.sleep(delays[i])
        if i in leavers:
            common.leave_mid_stream(port, reqs[i], 600)
            return
        try:
            results[i] = post(port, reqs[i])["ids"]
        except Exception as e:  # reported below as a mismatch
            results[i] = "error: %s" % e
    for i in range(len(reqs)):
        threads.append(threading.Thread(target=worker, args=(i,)))
        threads[-1].start()
    for t in threads:
        t.join()
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--exe", default=common.EXE)
    p.add_argument("--model", required=True)
    p.add_argument("--text", required=True, help="text the prompts are cut from")
    p.add_argument("--device", default="cpu")
    p.add_argument("--layer-shares")
    p.add_argument("--requests", type=int, default=16)
    p.add_argument("--max-seqs", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cli", type=int, default=4,
                   help="requests also checked against the CLI; the first four cover every prompt length, the last two several ubatch chunks")
    args = p.parse_args()
    common.EXE = os.path.abspath(args.exe)

    rng = random.Random(args.seed)
    with open(args.text, encoding="utf-8", errors="replace") as f:
        text = f.read()
    reqs = requests_from(text, args.requests, rng)
    flags = ["--device", args.device] + (["--layer-shares", args.layer_shares] if args.layer_shares else [])
    proc, port, log = common.start_server([common.EXE, "serve", args.model, "--max-seqs", str(args.max_seqs)] + flags,
                                          wait=1800)
    failures = []
    try:
        replies = [post(port, r) for r in reqs]
        alone = [r["ids"] for r in replies]
        print("alone: %d requests, %d tokens" % (len(reqs), sum(len(a) for a in alone)), flush=True)

        got = run_together(port, reqs)
        bad = [i for i in range(len(reqs)) if got.get(i) != alone[i]]
        print("together: %d of %d differ" % (len(bad), len(reqs)), flush=True)
        failures += ["together %d" % i for i in bad]

        # Short requests first with the longest replies, the long prompts landing while they decode, and every fourth client leaving once its stream has begun.
        delays = [0.0 if len(r["prompt"]) <= 1500 else 0.5 + rng.random() * 2 for r in reqs]
        leavers = set(range(3, len(reqs), 4))
        got = run_together(port, reqs, delays, leavers)
        bad = [i for i in range(len(reqs)) if i not in leavers and got.get(i) != alone[i]]
        print("skewed: %d of %d differ, %d clients left early" % (len(bad), len(reqs) - len(leavers), len(leavers)), flush=True)
        failures += ["skewed %d" % i for i in bad]
        deadline = time.time() + 120
        while time.time() < deadline and health(port)["active"] != 0:
            time.sleep(0.2)
        if health(port)["active"] != 0:
            failures.append("clients that left stayed active")
        if post(port, reqs[0])["ids"] != alone[0]:
            failures.append("the first request differs after the load")
    finally:
        common.stop_server(proc, log)

    # The CLI prints the reply's text between its pp and tg lines, which must be the text the server gave the request alone.
    for i in range(min(args.cli, len(reqs))):
        r = reqs[i]
        out = subprocess.run([common.EXE, "generate", args.model, r["prompt"], "-n", str(r["max_tokens"]), "--temp", "0"] + flags,
                             capture_output=True)
        try:
            same = out.returncode == 0 and common.generate_text(out.stdout).decode("utf-8", errors="replace") == replies[i]["text"]
        except AssertionError:  # output outside the pp and tg frame
            same = False
        if not same:
            failures.append("cli %d" % i)
    print("cli: %d requests checked" % min(args.cli, len(reqs)), flush=True)

    print("FAIL: " + ", ".join(failures) if failures else "all requests match")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
