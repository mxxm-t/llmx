"""Many users at once through `llmx serve`, every request checked against what it gives alone: the check that a placement, a layer split above all, holds under whatever mix of prompts and decodes a server meets.

Standard library only.
Every request is greedy, or with --sampled drawn at the sampler's defaults (temperature 0.8, top-k 40, top-p 0.95) with a seed of its own.
The requests mix short and long prompts, cut from a text file, with short and long replies.
Phases:

    alone     each request by itself, one after another: its reference ids
    together  every request at once, so prompts and decodes share passes
    skewed    the same requests arriving at staggered times, long prompts landing while others decode, and some clients leaving mid-stream
    cli       the first requests through `llmx generate` with the same settings on the same devices, whose prompt runs as one transaction and, on a split, pipelined over the stages

Every request that runs to its end must give its ids alone, the CLI its text; a client that left must leave nothing active.
--logprobs asks every request of these phases for its log-probabilities and top five too, which must equal alone's as its ids do.
--ids writes every phase's ids, with --logprobs beside their values, so two builds can be compared byte for byte.
--passes N serves with N passes in flight, which a layer split takes above one.
--generating-share X serves with that share for generating requests, whose rows then go in passes of their own between a prompt pass's pieces.
--tensor-width W groups the devices in tensor groups of W, the server and the CLI alike.
--drafter lookup|embedded|PATH serves with drafts, so every phase holds drafting to the replies without them: alone, together and skewed as the server gives them, and the CLI's run without drafts; it prints the drafts the server fed and kept, and fails where it fed none.
--fresh-phases starts each capped phase and the final repeat on a fresh server and requires zero prefix reuse and pauses, isolating batching from history reuse.
--ctx-size sets the server pool; leave enough room for every active request in this mode.

--uncapped runs other phases on a pool too small for its requests: 12 uncapped greedy requests through /v1/completions, streamed with logprobs 5, with --max-seqs 6 and --ctx-size 4096 unless given.
Each runs alone, where it never pauses, then all at once, where requests are paused and resumed; each must give its tokens and every log-probability alone.
It reports the pauses, the tokens resumes recomputed, the wall time of the run together and its inter-token p50 and p99.

    python tools/server_mix_check.py --model M.gguf --text wiki.txt --device vulkan:0,vulkan:1,vulkan:2 --layer-shares 1,1,1
"""
import argparse
import contextlib
import http.client
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


def requests_from(text, count, rng, logprobs=False, sampled=False):
    """`count` requests: prompts of a sentence, a paragraph or pages, cut from `text` at random offsets, with replies of 8 to 128 tokens, greedy or with `sampled` drawn at the defaults with seeds 1 to `count`, and with `logprobs` each asking for its log-probabilities and top five."""
    out = []
    for i in range(count):
        chars = [120, 1500, 6000, 12000][i % 4]
        start = rng.randrange(0, max(1, len(text) - chars))
        # A leading '-' would read as a flag to `llmx generate`.
        prompt = text[start:start + chars].lstrip("-")
        out.append({"prompt": prompt, "max_tokens": [128, 32, 64, 8][(i // 4) % 4], "temperature": 0})
        if sampled:
            out[-1].update({"temperature": 0.8, "top_k": 40, "top_p": 0.95, "seed": i + 1})
        if logprobs:
            out[-1].update({"logprobs": True, "top_logprobs": 5})
    return out


def cli_sampling(req):
    """The `llmx generate` flags that draw as the server draws `req`."""
    if req["temperature"] == 0:
        return ["--temp", "0"]
    return ["--temp", str(req["temperature"]), "--topk", str(req["top_k"]), "--topp", str(req["top_p"]), "--seed", str(req["seed"])]


def answer(reply):
    """What a phase compares of a reply: its ids, and with log-probabilities asked their values and top five beside them."""
    return [reply["ids"], reply["logprobs"], reply["top_logprobs"]] if "logprobs" in reply else reply["ids"]


def run_together(port, reqs, delays=None, leavers=()):
    """Every request at once, each after its delay: a finished request's answer, None for a client in `leavers` that left as planned, and an error either way failed."""
    results, threads = {}, []
    def worker(i):
        if delays:
            time.sleep(delays[i])
        try:
            if i in leavers:
                common.leave_mid_stream(port, reqs[i], 600)
                results[i] = None
            else:
                results[i] = answer(post(port, reqs[i]))
        except Exception as e:  # reported below as a mismatch
            results[i] = "error: %s" % e
    for i in range(len(reqs)):
        threads.append(threading.Thread(target=worker, args=(i,)))
        threads[-1].start()
    for t in threads:
        t.join()
    return results


def stream_logprobs(port, prompt):
    """An uncapped greedy stream through /v1/completions: each token's text, log-probability and top five, and its arrival time."""
    body = json.dumps({"prompt": prompt, "temperature": 0, "logprobs": 5, "stream": True}).encode()
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=7200)
    c.request("POST", "/v1/completions", body, {"Content-Type": "application/json"})
    tokens, arrivals = [], []
    for raw in c.getresponse():
        line = raw.decode("utf-8").rstrip("\n")
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        lp = json.loads(line[6:])["choices"][0].get("logprobs")
        if lp:
            tokens.append([lp["tokens"][0], lp["token_logprobs"][0], lp["top_logprobs"][0]])
            arrivals.append(time.perf_counter())
    c.close()
    return tokens, arrivals


def percentile(values, q):
    s = sorted(values)
    if not s:
        return float("nan")
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def uncapped(args, text, flags):
    """The uncapped phases: each request alone, then all at once on the small pool, compared token by token and value by value."""
    rng = random.Random(7 if args.seed is None else args.seed)
    prompts = []
    for i in range(args.requests or 12):
        chars = [300, 900, 1800, 2600][i % 4]
        start = rng.randrange(0, max(1, len(text) - chars))
        prompts.append(text[start:start + chars].lstrip("-"))
    proc, port, log = common.start_server([common.EXE, "serve", args.model, "--max-seqs", str(args.max_seqs or 6),
                                           "--ctx-size", str(args.ctx_size or 4096)] + flags, wait=1800)
    try:
        alone = [stream_logprobs(port, p)[0] for p in prompts]
        before = health(port)
        together, arrivals = {}, {}
        def worker(i):
            together[i], arrivals[i] = stream_logprobs(port, prompts[i])
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(prompts))]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.perf_counter() - t0
        after = health(port)
    finally:
        proc.kill()
        proc.wait()
        log.close()
    differ = 0
    for i in range(len(prompts)):
        a, b = alone[i], together[i]
        first = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), None if len(a) == len(b) else min(len(a), len(b)))
        differ += first is not None
        print("request %d: %d tokens alone, %d together, %s" % (i, len(a), len(b), "the same" if first is None else "differs from token %d" % first))
    gaps = [(y - x) * 1e3 for i in arrivals for x, y in zip(arrivals[i], arrivals[i][1:])]
    print("uncapped: %d of %d differ; %d pauses, %s tokens recomputed; together %.1f s, %d tokens, inter-token p50 %.2f ms, p99 %.2f ms, longest gap %.0f ms"
          % (differ, len(prompts), after["pressure"]["since_start"]["pauses"] - before["pressure"]["since_start"]["pauses"],
             after["pressure"]["since_start"]["recomputed_tokens"] - before["pressure"]["since_start"]["recomputed_tokens"],
             wall, sum(len(t) for t in together.values()), percentile(gaps, 0.5), percentile(gaps, 0.99), max(gaps or [0])), flush=True)
    if args.ids:
        with open(args.ids, "w", encoding="utf-8") as f:
            json.dump({"alone": alone, "together": [together[i] for i in range(len(prompts))]}, f)
    print("FAIL: %d requests differ from alone" % differ if differ else "all requests match")
    return 1 if differ else 0


@contextlib.contextmanager
def serving(args, flags):
    """One capped-phase server, drained by the caller and always stopped on exit."""
    command = [common.EXE, "serve", args.model, "--max-seqs", str(args.max_seqs or 8)] + flags
    if args.ctx_size is not None:
        command += ["--ctx-size", str(args.ctx_size)]
    proc, port, log = common.start_server(command, wait=1800)
    try:
        yield port
    finally:
        common.stop_server(proc, log)


def check_phase(port, name, fresh, failures):
    """A finished phase has no requests left; a fresh phase must not have reused or paused one."""
    deadline = time.monotonic() + 120
    state = health(port)
    while any(state["requests"]["now"].values()) and time.monotonic() < deadline:
        time.sleep(0.2)
        state = health(port)
    if any(state["requests"]["now"].values()):
        failures.append(name + " left requests active, queued or paused")
    if fresh:
        print("fresh phase %s: %s" % (name, json.dumps(state, sort_keys=True)), flush=True)
        if state["reuse"]["since_start"]["forks"] or state["reuse"]["since_start"]["tokens"] or state["pressure"]["since_start"]["pauses"]:
            failures.append(name + " reused a donor or paused a request")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--exe", default=common.EXE)
    p.add_argument("--model", required=True)
    p.add_argument("--text", required=True, help="text the prompts are cut from")
    p.add_argument("--device", default="cpu")
    p.add_argument("--layer-shares")
    p.add_argument("--tensor-width", type=int, help="the devices form tensor groups of this many, on every server and CLI comparison")
    p.add_argument("--cache-type", choices=["f16", "f32"], help="KV cache type for both sides, on every server and CLI comparison; omitted uses the runtime default")
    p.add_argument("--requests", type=int, help="16, or 12 with --uncapped")
    p.add_argument("--max-seqs", type=int, help="8, or 6 with --uncapped")
    p.add_argument("--ctx-size", type=int, help="server KV pool; 4096 with --uncapped, otherwise the model default")
    p.add_argument("--seed", type=int, help="1, or 7 with --uncapped")
    p.add_argument("--uncapped", action="store_true", help="the uncapped phases in place of the others")
    p.add_argument("--logprobs", action="store_true", help="the capped phases compare log-probabilities and the top five beside the ids")
    p.add_argument("--sampled", action="store_true", help="the capped phases draw every request at the defaults with a seed of its own, in place of greedy")
    p.add_argument("--passes", type=int, help="passes in flight, the server's own number when not given")
    p.add_argument("--generating-share", type=float, help="the share of each device's time generating requests keep while prompts are read, the server's own when not given")
    p.add_argument("--drafter", default="off",
                   help="the server's drafter: off, lookup, embedded or a file of MTP blocks beside the model; the CLI runs without drafts, so its text holds the server's drafts to the reply without them")
    p.add_argument("--fresh-phases", action="store_true", help="fresh server for each capped phase; fail on prefix reuse or pauses")
    p.add_argument("--cli", type=int, default=4,
                   help="requests also checked against the CLI; the first four cover every prompt length, the last two several ubatch chunks")
    p.add_argument("--ids", metavar="PATH", help="write the ids of every phase as JSON, the skewed phase's clients that left as null; with --uncapped each token's text and values")
    args = p.parse_args()
    if args.fresh_phases and args.uncapped:
        p.error("--fresh-phases cannot be combined with --uncapped")
    if args.uncapped and args.sampled:
        p.error("--sampled draws the capped phases, which --uncapped replaces")
    common.EXE = os.path.abspath(args.exe)

    with open(args.text, encoding="utf-8", errors="replace") as f:
        text = f.read()
    flags = ["--device", args.device] + (["--layer-shares", args.layer_shares] if args.layer_shares else [])
    if args.tensor_width:
        flags += ["--tensor-width", str(args.tensor_width)]
    if args.cache_type:
        flags += ["--cache-type-k", args.cache_type, "--cache-type-v", args.cache_type]
    server_flags = flags + (["--passes", str(args.passes)] if args.passes else []) + (["--drafter", args.drafter] if args.drafter != "off" else [])
    if args.generating_share is not None:
        server_flags += ["--generating-share", str(args.generating_share)]
    if args.uncapped:
        return uncapped(args, text, server_flags)
    rng = random.Random(1 if args.seed is None else args.seed)
    reqs = requests_from(text, args.requests or 16, rng, args.logprobs, args.sampled)
    failures = []
    phases = {}
    shared_server = contextlib.nullcontext(None) if args.fresh_phases else serving(args, server_flags)
    with shared_server as shared_port:
        def phase():
            return serving(args, server_flags) if args.fresh_phases else contextlib.nullcontext(shared_port)

        with phase() as port:
            replies = [post(port, r) for r in reqs]
            alone = [answer(r) for r in replies]
            print("alone: %d requests, %d tokens" % (len(reqs), sum(len(r["ids"]) for r in replies)), flush=True)
            check_phase(port, "alone", args.fresh_phases, failures)

        with phase() as port:
            got = run_together(port, reqs)
            phases["alone"], phases["together"] = alone, [got.get(i) for i in range(len(reqs))]
            bad = [i for i in range(len(reqs)) if got.get(i) != alone[i]]
            print("together: %d of %d differ" % (len(bad), len(reqs)), flush=True)
            failures += ["together %d" % i for i in bad]
            check_phase(port, "together", args.fresh_phases, failures)

        with phase() as port:
            # Short requests first with the longest replies, the long prompts landing while others decode, and every fourth client leaving once its stream has begun.
            delays = [0.0 if len(r["prompt"]) <= 1500 else 0.5 + rng.random() * 2 for r in reqs]
            leavers = set(range(3, len(reqs), 4))
            got = run_together(port, reqs, delays, leavers)
            phases["skewed"] = [None if i in leavers else got.get(i) for i in range(len(reqs))]
            bad = [i for i in range(len(reqs)) if i not in leavers and got.get(i) != alone[i]]
            stuck = [i for i in sorted(leavers) if i not in got or got[i] is not None]
            print("skewed: %d of %d differ, %d clients left early, %d of them failed" % (len(bad), len(reqs) - len(leavers), len(leavers), len(stuck)), flush=True)
            failures += ["skewed leaver %d: %s" % (i, got.get(i, "no result")) for i in stuck]
            failures += ["skewed %d" % i for i in bad]
            check_phase(port, "skewed", args.fresh_phases, failures)

        with phase() as port:
            if answer(post(port, reqs[0])) != alone[0]:
                failures.append("the first request differs on a fresh server" if args.fresh_phases else "the first request differs after the load")
            check_phase(port, "repeat" if args.fresh_phases else "recovery", args.fresh_phases, failures)
            # Drafts that were never fed would hold nothing to the replies without them.
            if args.drafter != "off":
                state = health(port)
                fed, kept = state["drafting"]["since_start"]["drafted"], state["drafting"]["since_start"]["kept"]
                print("drafts: %d fed, %d kept%s" % (fed, kept, " in the last phase" if args.fresh_phases else ""), flush=True)
                if not fed:
                    failures.append("no draft was fed")

    # The CLI prints the reply's text between its pp and tg lines, which must be the text the server gave the request alone.
    for i in range(min(args.cli, len(reqs))):
        r = reqs[i]
        command = [common.EXE, "generate", args.model, r["prompt"], "-n", str(r["max_tokens"])] + cli_sampling(r) + flags
        out = subprocess.run(command, capture_output=True)
        # A server just stopped gives its devices' memory back over some seconds: a run refused for want of room waits for it, up to a minute.
        for _ in range(30):
            if not (out.returncode and b"does not fit" in out.stderr):
                break
            time.sleep(2)
            out = subprocess.run(command, capture_output=True)
        if out.returncode:
            last = out.stderr.decode("utf-8", errors="replace").strip().splitlines()[-1:]
            failures.append("cli %d did not run (exit %d%s)" % (i, out.returncode, ": " + last[0] if last else ""))
            continue
        try:
            same = common.generate_text(out.stdout).decode("utf-8", errors="replace") == replies[i]["text"]
        except ValueError:  # output outside the pp and tg frame
            same = False
        if not same:
            failures.append("cli %d" % i)
    print("cli: %d requests checked" % min(args.cli, len(reqs)), flush=True)
    if args.ids:
        with open(args.ids, "w", encoding="utf-8") as f:
            json.dump(phases, f)

    print("FAIL: " + ", ".join(failures) if failures else "all requests match")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
