"""Aggregate decode throughput of a server under N concurrent requests.

Standard library only. Every request asks for the same number of tokens
with sampling off, so the work per request is fixed; the figure is decoded
tokens per second summed over the requests, measured from the first
request's send to the last reply. With --api completion the same load goes
to a server that speaks the reference's /completion route, so the two can
be compared on the same model, card and load.

    python tools/server_load.py --url http://127.0.0.1:8080 --concurrency 1 4 8 16 --tokens 64
"""
import argparse
import json
import sys
import threading
import time
import urllib.request

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


def one(url, api, prompt, tokens, out, i):
    if api == "llmx":
        body = {"prompt": prompt, "max_tokens": tokens, "temperature": 0}
        path = "/v1/generate"
    else:
        body = {"prompt": prompt, "n_predict": tokens, "temperature": 0, "cache_prompt": False}
        path = "/completion"
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url + path, data=data, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        reply = json.loads(r.read().decode("utf-8"))
    t1 = time.perf_counter()
    if api == "llmx":
        n = reply["tokens"]
    else:
        n = reply.get("tokens_predicted", tokens)
    out[i] = (n, t1 - t0)


def measure(url, api, concurrency, tokens, rounds):
    best = 0.0
    for _ in range(rounds):
        out = [None] * concurrency
        threads = [threading.Thread(target=one, args=(url, api, PROMPTS[i % len(PROMPTS)], tokens, out, i))
                   for i in range(concurrency)]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.perf_counter() - t0
        total = sum(n for n, _ in out)
        best = max(best, total / wall)
    return best


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("--api", choices=["llmx", "completion"], default="llmx")
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8, 16])
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--rounds", type=int, default=2, help="repeats per concurrency level; the best is reported")
    args = p.parse_args()
    # A warm-up so the first level does not carry the model's first pass.
    measure(args.url, args.api, 1, 8, 1)
    for c in args.concurrency:
        tps = measure(args.url, args.api, c, args.tokens, args.rounds)
        print("%s concurrency %2d: %8.1f tok/s aggregate (%d tokens each, best of %d)"
              % (args.api, c, tps, args.tokens, args.rounds))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
