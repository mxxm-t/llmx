"""A server under N concurrent streaming requests: the latency and
throughput figures a serving runtime is judged by.

Standard library only. Every request asks for the same number of tokens
with sampling off, so the work per request is fixed, and streams its
reply so the arrival of every token is timed. Per concurrency level the
tool reports, over all requests of the level:

    ttft     time to first token, from the request's send to its first
             streamed token, median and 99th percentile
    itl      inter-token latency, the gap between consecutive tokens of a
             request, median and 99th percentile
    tok/s    decoded tokens per second summed over the requests, from the
             first send to the last reply
    req/s    completed requests per second over the same span

With --api completion the same load goes to a server that speaks the
reference's /completion route, so the two can be compared on the same
model, card and load; --api openai speaks /v1/chat/completions.

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


def request_for(api, prompt, tokens):
    if api == "llmx":
        return "/v1/generate", {"prompt": prompt, "max_tokens": tokens, "temperature": 0, "stream": True}
    if api == "openai":
        return "/v1/completions", {"prompt": prompt, "max_tokens": tokens, "temperature": 0, "stream": True}
    return "/completion", {"prompt": prompt, "n_predict": tokens, "temperature": 0, "cache_prompt": False,
                           "stream": True}


def token_of(api, event):
    """Whether a streamed event carries a token: the reference's stream ends
    with an event that has no content but the final stop flag."""
    if api == "llmx":
        return "id" in event
    if api == "openai":
        return bool(event.get("choices")) and event["choices"][0].get("text", "") != ""
    return bool(event.get("content", "")) or not event.get("stop", False)


def one(url, api, prompt, tokens, out, i):
    """A streamed request: the arrival time of every token."""
    path, body = request_for(api, prompt, tokens)
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url + path, data=data, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    arrivals = []
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8").rstrip("\n")
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            if token_of(api, event):
                arrivals.append(time.perf_counter())
    out[i] = (t0, arrivals, time.perf_counter())


def percentile(values, q):
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def measure(url, api, concurrency, tokens, rounds):
    """The best round by aggregate tokens per second, with its latencies."""
    best = None
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
        total = sum(len(a) for _, a, _ in out)
        ttft = [a[0] - s for s, a, _ in out if a]
        itl = [b - a for _, arr, _ in out for a, b in zip(arr, arr[1:])]
        result = {"tok_s": total / wall, "req_s": concurrency / wall,
                  "ttft_p50": percentile(ttft, 0.5), "ttft_p99": percentile(ttft, 0.99),
                  "itl_p50": percentile(itl, 0.5), "itl_p99": percentile(itl, 0.99), "tokens": total}
        if best is None or result["tok_s"] > best["tok_s"]:
            best = result
    return best


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("--api", choices=["llmx", "openai", "completion"], default="llmx")
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8, 16])
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--rounds", type=int, default=2, help="repeats per concurrency level; the best is reported")
    args = p.parse_args()
    # A warm-up so the first level does not carry the model's first pass.
    measure(args.url, args.api, 1, 8, 1)
    print("%-10s %5s %9s %8s %10s %10s %10s %10s" % ("api", "conc", "tok/s", "req/s", "ttft p50", "ttft p99",
                                                     "itl p50", "itl p99"))
    for c in args.concurrency:
        r = measure(args.url, args.api, c, args.tokens, args.rounds)
        print("%-10s %5d %9.1f %8.2f %8.1f ms %8.1f ms %8.2f ms %8.2f ms"
              % (args.api, c, r["tok_s"], r["req_s"], r["ttft_p50"] * 1e3, r["ttft_p99"] * 1e3,
                 r["itl_p50"] * 1e3, r["itl_p99"] * 1e3))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
