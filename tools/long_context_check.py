"""Long-context end-to-end check: one 16k-token summarization prompt, greedy,
on the backend under test, checked for repeatability and against a second
backend's reading of the same tokens.

This is the case the short gates do not reach. The HF baseline scores windows
of at most a few hundred tokens and the server component sends short prompts,
so nothing else here exercises a prompt that fills thousands of KV blocks,
crosses many block boundaries in one pass, and then decodes from that
history.

Two backends with different activation precision need not produce the same
text: greedy decoding follows the top logit, and after a long prompt two
candidates can sit within the rounding the precisions differ by, where either
choice is correct. A hash across backends therefore parts at the first such
near-tie. The check is two parts instead:

  1. Repeatability: the backend under test answers twice, from two fresh
     servers, and must give the same tokens both times.
  2. Accuracy: the baseline backend reads the prompt followed by the tokens
     the device generated, and at every generated position the device's
     token must be the baseline's top choice or within --margin logits of
     it. A kernel that is wrong rather than differently rounded puts some
     token far below the baseline's choice.

Usage:
  python tools/long_context_check.py --exe build/Release/llmx.exe \\
      --model <model.gguf> --device vulkan:0 [--baseline cpu] [--tokens 16384] [--max-tokens 512]

It starts `llmx serve` for each device run on a port the OS chooses, sends the
prompt to /v1/generate with temperature 0, and reports prompt tokens,
generated tokens, wall time and the SHA-256 of the generated text; then it
runs `llmx logits --last` on the baseline over the prompt and the generated
tokens. It exits non-zero if either part fails.
"""

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, os.pardir, "tests", "data", "wiki.test.raw")
INSTRUCTION = (
    "Read the following encyclopedia extract and write a single paragraph "
    "summarising what it is about.\n\n"
)
TOP = 20   # candidates the baseline reports per position


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
        # The corpus is uniform enough that one linear step lands within a few tokens; the loop is there for the rounding.
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
    # The server logs a line per request; to a file, since a pipe nobody reads fills and blocks it.
    log = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    deadline = time.time() + 600
    while time.time() < deadline:
        if proc.poll() is not None:
            log.seek(0)
            raise SystemExit("serve exited:\n" + log.read()[-2000:])
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/health", timeout=2) as r:
                if json.load(r).get("status") == "ok":
                    return proc, port
        except Exception:
            time.sleep(0.25)
    proc.kill()
    raise SystemExit("serve did not become healthy")


def stop(proc):
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()


# Seconds to wait for a reply; None waits as long as it takes.
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
    out["sha256"] = hashlib.sha256(out.get("text", "").encode("utf-8")).hexdigest()
    return out


def report(name, got):
    print(f"{name:12s} prompt {got.get('prompt_tokens', 0)} tokens, generated "
          f"{got.get('tokens', 0)}, finish {got.get('finish')!r}, "
          f"{got['wall_s']:.1f} s wall, sha256 {got['sha256'][:16]}", flush=True)


def baseline_logits(exe, model, baseline, prompt, reply_ids, extra):
    """The baseline's top candidates at each position that predicts a generated token: the prompt followed by every generated token but the last."""
    with tempfile.TemporaryDirectory(prefix="llmx_long_") as d:
        text_path, ids_path = os.path.join(d, "prompt.txt"), os.path.join(d, "reply.ids")
        with open(text_path, "w", encoding="utf-8", newline="") as f:
            f.write(prompt)
        with open(ids_path, "w", encoding="utf-8") as f:
            f.write(" ".join(str(i) for i in reply_ids[:-1]))
        cmd = [exe, "logits", model, text_path, "--file", "--then-ids", ids_path,
               "--last", str(len(reply_ids)), "--top", str(TOP), "--device", baseline] + extra
        out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if out.returncode != 0:
        raise SystemExit("baseline logits failed:\n" + out.stderr[-2000:])
    rows = []
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        cands = [(int(parts[i]), float(parts[i + 1])) for i in range(1, len(parts) - 1, 2)]
        rows.append((int(parts[0]), cands))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", required=True, help="the backend under test, e.g. vulkan:0")
    ap.add_argument("--baseline", default="cpu", help="the backend that reads the device's tokens")
    ap.add_argument("--tokens", type=int, default=16384, help="target prompt tokens")
    ap.add_argument("--max-tokens", type=int, default=512, help="tokens the device generates")
    ap.add_argument("--margin", type=float, default=0.5,
                    help="how far below the baseline's top logit a generated token may sit")
    ap.add_argument("--ctx-size", type=int, default=0,
                    help="KV budget; 0 uses the prompt plus the generation plus a margin")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=0, help="seconds to wait for each reply; 0 waits as long as it takes")
    args = ap.parse_args()
    args.exe = os.path.abspath(args.exe)

    global TIMEOUT
    TIMEOUT = args.timeout or None
    extra = ["--threads", str(args.threads)] if args.threads else []
    ctx = args.ctx_size or (args.tokens + args.max_tokens + 512)

    # 1. The device twice, each from a fresh server; the first run also sizes the prompt.
    runs = []
    prompt = None
    for i in range(2):
        proc, port = serve(args.exe, args.model, args.device, ctx, extra)
        try:
            if prompt is None:
                n, prompt = build_prompt(port, args.tokens)
                print(f"prompt: {n} tokens, {len(prompt)} characters", flush=True)
            got = run_once(port, prompt, args.max_tokens)
        finally:
            stop(proc)
        report(f"{args.device} #{i + 1}", got)
        runs.append(got)
    repeat_ok = runs[0].get("ids") == runs[1].get("ids")
    print(f"\nrepeatability: {'SAME' if repeat_ok else 'DIFFERENT'} tokens on two runs of {args.device}", flush=True)

    # 2. The baseline reads the prompt and the device's tokens.
    reply = runs[0].get("ids", [])
    if not reply:
        raise SystemExit("the device generated nothing")
    start = time.time()
    rows = baseline_logits(args.exe, args.model, args.baseline, prompt, reply,
                           ["--threads", str(args.threads)] if args.threads else [])
    if len(rows) != len(reply):
        raise SystemExit(f"baseline gave {len(rows)} positions for {len(reply)} generated tokens")
    agree, worst, worst_at, beyond = 0, 0.0, -1, []
    for j, (pos, cands) in enumerate(rows):
        top_logit = cands[0][1]
        mine = next((l for t, l in cands if t == reply[j]), None)
        gap = float("inf") if mine is None else top_logit - mine
        if gap == 0.0 or cands[0][0] == reply[j]:
            agree += 1
        if gap > worst:
            worst, worst_at = gap, j
        if gap > args.margin:
            beyond.append((j, gap))
    print(f"accuracy: {args.baseline} read {len(reply)} generated tokens in {time.time() - start:.0f} s; "
          f"its top choice {agree}/{len(reply)}, largest gap {worst:.3f} logits at token {worst_at}, "
          f"{len(beyond)} beyond {args.margin}", flush=True)
    for j, gap in beyond[:10]:
        print(f"  token {j}: {gap:.3f} below the baseline's top", flush=True)
    accuracy_ok = not beyond
    print(f"\n{'PASS' if repeat_ok and accuracy_ok else 'FAIL'}: repeatability {'ok' if repeat_ok else 'failed'}, "
          f"accuracy {'ok' if accuracy_ok else 'failed'}")
    return 0 if repeat_ok and accuracy_ok else 1


if __name__ == "__main__":
    sys.exit(main())
