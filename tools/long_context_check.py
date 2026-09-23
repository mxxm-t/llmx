"""Long-context end-to-end check: one 16k-token summarization prompt, greedy,
with no generation cap, run on two backends and compared by the hash of what
came back.

This is the case the short gates do not reach. The HF baseline scores windows
of at most a few hundred tokens and the server component sends short prompts,
so nothing else here exercises a prompt that fills thousands of KV blocks,
crosses many block boundaries in one pass, and then decodes from that history
until the model stops on its own. Greedy sampling makes the whole run a single
deterministic function of the weights and the prompt, so two backends that
agree on every kernel agree on the exact token sequence, and one hash says so.

Usage:
  python tools/long_context_check.py --exe build/Release/llmx.exe \\
      --model <model.gguf> --device vulkan:0 [--baseline cpu] [--tokens 16384]

It starts `llmx serve` once per backend on a port the OS chooses, sends the
prompt to /v1/generate with temperature 0 and a generation cap large enough
that the model reaches its own end of text, and reports prompt tokens,
generated tokens, prefill and decode throughput and the SHA-256 of the
generated text. It exits non-zero if the two backends' hashes differ.
"""

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, os.pardir, "tests", "data", "wiki.test.raw")
INSTRUCTION = (
    "Read the following encyclopedia extract and write a single paragraph "
    "summarising what it is about.\n\n"
)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def build_prompt(port, want_tokens):
    """Text whose prompt is as close to `want_tokens` tokens as a few probes
    get. The count comes from a running server rather than the `tokenize`
    command, because a prompt this long does not fit a command line: 16k
    tokens of this corpus is about 68,000 characters against the 32,767 a
    Windows command line takes. Each probe asks for one token, so it costs one
    prefill."""
    raw = open(CORPUS, encoding="utf-8").read().lstrip("\n ")

    def count(chars):
        text = INSTRUCTION + raw[:chars]
        got = run_once(port, text, 1)
        return got.get("prompt_tokens", 0), text

    chars = min(len(raw), want_tokens * 4)
    best = None
    for _ in range(6):
        n, text = count(chars)
        if best is None or abs(n - want_tokens) < abs(best[0] - want_tokens):
            best = (n, text)
        if n == want_tokens or n == 0:
            break
        # The corpus is uniform enough that one linear step lands within a
        # few tokens; the loop is there for the rounding.
        step = int(chars * (want_tokens / n - 1.0))
        if step == 0:
            break
        chars = max(1000, min(len(raw), chars + step))
    return best


def serve(exe, model, device, ctx, extra):
    port = free_port()
    cmd = [exe, "serve", model, "--host", "127.0.0.1", "--port", str(port),
           "--ctx-size", str(ctx)] + extra
    if device:
        cmd += ["--device", device]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    deadline = time.time() + 600
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit("serve exited:\n" + (proc.stdout.read() or "")[-2000:])
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/health", timeout=2) as r:
                if json.load(r).get("status") == "ok":
                    return proc, port
        except Exception:
            time.sleep(0.25)
    proc.kill()
    raise SystemExit("serve did not become healthy")


# Seconds to wait for a reply; None waits as long as it takes. An uncapped greedy reply on the CPU after a 16k-token prompt can run for hours.
TIMEOUT = None


def run_once(port, prompt, max_tokens):
    body = json.dumps({"prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0.0, "seed": 0}).encode("utf-8")
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    start = time.time()
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        out = json.load(r)
    out["wall_s"] = time.time() - start
    return out


def report(name, got):
    print(f"{name:10s} prompt {got.get('prompt_tokens', 0)} tokens, generated "
          f"{got.get('tokens', 0)}, finish {got.get('finish')!r}, "
          f"{got['wall_s']:.1f} s wall, sha256 {got['sha256'][:16]}", flush=True)


def arm(exe, model, device, prompt, ctx, max_tokens, extra):
    proc, port = serve(exe, model, device, ctx, extra)
    try:
        got = run_once(port, prompt, max_tokens)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    text = got.get("text", "")
    got["sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", required=True, help="the backend under test, e.g. vulkan:0")
    ap.add_argument("--baseline", default="cpu", help="the backend it is compared against")
    ap.add_argument("--tokens", type=int, default=16384, help="target prompt tokens")
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="generation cap; 0 lets it run until the model stops or the context is full")
    ap.add_argument("--ctx-size", type=int, default=0,
                    help="KV budget; 0 uses the prompt plus the cap plus a margin")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=0, help="seconds to wait for each reply; 0 waits as long as it takes")
    ap.add_argument("--skip-baseline", action="store_true",
                    help="run the device arm only and print its hash")
    args = ap.parse_args()

    global TIMEOUT
    TIMEOUT = args.timeout or None
    extra = ["--threads", str(args.threads)] if args.threads else []
    # No cap means the generation is bounded only by what the KV pool holds
    # after the prompt, so the model stops when it stops. The finish reason
    # in the report says which happened.
    max_tokens = args.max_tokens
    ctx = args.ctx_size or (args.tokens + (max_tokens or args.tokens) + 512)
    if not max_tokens:
        max_tokens = ctx - args.tokens - 512

    # The device arm opens first and sizes the prompt, since a probe is a
    # whole prefill and this is the backend that does one quickly.
    proc, port = serve(args.exe, args.model, args.device, ctx, extra)
    try:
        n, prompt = build_prompt(port, args.tokens)
        print(f"prompt: {n} tokens, {len(prompt)} characters", flush=True)
        device_got = run_once(port, prompt, max_tokens)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    device_got["sha256"] = hashlib.sha256(device_got.get("text", "").encode("utf-8")).hexdigest()
    report(args.device, device_got)

    if args.skip_baseline:
        return 0
    base_got = arm(args.exe, args.model, None if args.baseline == "cpu" else args.baseline,
                   prompt, ctx, max_tokens, extra)
    report(args.baseline, base_got)

    same = base_got["sha256"] == device_got["sha256"]
    print(f"\n{args.baseline} vs {args.device}: {'IDENTICAL' if same else 'DIFFERENT'}")
    print(f"  {args.baseline:10s} {base_got['sha256']}")
    print(f"  {args.device:10s} {device_got['sha256']}")
    if not same:
        ta, tb = base_got.get("text", ""), device_got.get("text", "")
        for i, (x, y) in enumerate(zip(ta, tb)):
            if x != y:
                print(f"  first difference at character {i}:")
                print(f"    {args.baseline}: {ta[max(0,i-60):i+60]!r}")
                print(f"    {args.device}: {tb[max(0,i-60):i+60]!r}")
                break
        else:
            print(f"  one is a prefix of the other: {len(ta)} against {len(tb)} characters")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
