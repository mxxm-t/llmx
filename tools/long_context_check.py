"""Long-context end-to-end check: one 16k-token summarization request, greedy,
on the backend under test, checked for repeatability and against a second
backend's reading of the same tokens.

The request is one user message, an instruction to summarize followed by an
encyclopedia extract, sent through the model's chat template as the chat
routes render it, so an instruct model answers it. As raw text the model
continued the article instead and fell into loops, where the two backends
compared near-ties that say nothing about either.

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
      --model <model.gguf> --device vulkan:0 [--baseline cpu] [--dtype auto|f16|bf16|f32] [--tokens 16384] [--max-tokens 512] [--cli]

It starts `llmx serve` for each device run on a port the OS chooses, sizes
the message with /v1/tokenize, sends it to /v1/chat with temperature 0, and
reports prompt tokens, generated tokens, wall time and the SHA-256 of the
generated text; then `llmx logits --chat --last` reads the rendered prompt
and the generated tokens on the device, whose rows it prints beside the
baseline's at a position past the margin, and then on the baseline. The
device reads first, so its work is done before the baseline's long reading.
It exits non-zero if either part fails.

With --cli each device run is a fresh `llmx generate --chat --file` at
temperature 0 instead, whose `--verbose` output gives the prompt's token
count and the generated ids, for a model the server does not take, such as
one whose layers keep a recurrent state; the message is sized with one-token
`generate --chat` runs, since `llmx tokenize` renders no template.
The dtype selection applies to prompt sizing, both fresh generations and both
backends' scoring passes; each backend still reports its own resolved policy.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "tests"))
import common

CORPUS = os.path.join(HERE, os.pardir, "tests", "data", "wiki.test.raw")
INSTRUCTION = (
    "Read the following encyclopedia extract and write a single paragraph "
    "summarising what it is about.\n\n"
)
TOP = 20   # candidates the baseline reports per position


def messages(text):
    """The request: `text` as the one user message."""
    return [{"role": "user", "content": text}]


def post(port, route, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{route}", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.load(r)


def build_prompt(port, want_tokens):
    """The message whose rendered prompt is as close to `want_tokens` tokens
    as a few probes get. The count comes from a running server's
    /v1/tokenize, which renders the messages as /v1/chat does, rather than
    the `tokenize` command, which renders no template."""
    raw = open(CORPUS, encoding="utf-8").read().lstrip("\n ")

    def count(chars):
        text = INSTRUCTION + raw[:chars]
        return post(port, "/v1/tokenize", {"messages": messages(text)})["count"], text

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
    """(process, port, log) of `llmx serve` on `device`, once it answers."""
    cmd = [exe, "serve", model, "--ctx-size", str(ctx)] + extra
    if device:
        cmd += ["--device", device]
    return common.start_server(cmd, wait=600)


# Seconds to wait for a reply; None waits as long as it takes.
TIMEOUT = None


def run_once(port, prompt, max_tokens):
    start = time.time()
    out = post(port, "/v1/chat", {"messages": messages(prompt), "max_tokens": max_tokens,
                                  "temperature": 0.0, "seed": 0})
    out["wall_s"] = time.time() - start
    out["sha256"] = hashlib.sha256(out.get("text", "").encode("utf-8")).hexdigest()
    return out


def generate_cli(exe, model, device, prompt, max_tokens, extra):
    """One fresh `llmx generate --chat` of the message, greedy, as the server run's reply: its prompt tokens, ids and wall time."""
    with tempfile.TemporaryDirectory(prefix="llmx_long_") as d:
        path = os.path.join(d, "prompt.txt")
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(prompt)
        cmd = [exe, "generate", model, "--file", path, "--chat", "-n", str(max_tokens), "--temp", "0", "--verbose", "--device", device] + extra
        start = time.time()
        out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=TIMEOUT)
    if out.returncode != 0:
        raise SystemExit("generate failed:\n" + out.stderr[-2000:])
    lines = out.stdout.splitlines()
    counts = [int(l.split()[-1]) for l in lines if l.startswith("prompt tokens: ")]
    ids = next((l[len("ids:"):].strip() for l in reversed(lines) if l.startswith("ids:")), None)
    if not counts or ids is None:
        raise SystemExit("generate --verbose printed no prompt count or ids:\n" + out.stdout[-2000:])
    got = {"prompt_tokens": counts[-1], "ids": [int(i) for i in ids.split(",")] if ids else []}
    got["tokens"] = len(got["ids"])
    got["finish"] = "length" if got["tokens"] >= max_tokens else "stop"
    got["wall_s"] = time.time() - start
    got["sha256"] = hashlib.sha256(ids.encode("ascii")).hexdigest()
    return got


def build_prompt_cli(exe, model, device, want_tokens, extra):
    """The message closest to `want_tokens` tokens, as build_prompt finds it, counted by one-token `generate --chat` runs, which render the template."""
    raw = open(CORPUS, encoding="utf-8").read().lstrip("\n ")

    def count(chars):
        text = INSTRUCTION + raw[:chars]
        return generate_cli(exe, model, device, text, 1, extra)["prompt_tokens"], text

    chars = min(len(raw), want_tokens * 4)
    best = None
    for _ in range(6):
        n, text = count(chars)
        if best is None or abs(n - want_tokens) < abs(best[0] - want_tokens):
            best = (n, text)
        if n == want_tokens or n == 0:
            break
        step = int(chars * (want_tokens / n - 1.0))
        if step == 0:
            break
        chars = max(1000, min(len(raw), chars + step))
    return best


def report(name, got):
    print(f"{name:12s} prompt {got.get('prompt_tokens', 0)} tokens, generated "
          f"{got.get('tokens', 0)}, finish {got.get('finish')!r}, "
          f"{got['wall_s']:.1f} s wall, sha256 {got['sha256'][:16]}", flush=True)


def baseline_logits(exe, model, baseline, prompt, reply_ids, extra):
    """The baseline's top candidates at each position that predicts a generated token: the rendered prompt followed by every generated token but the last."""
    with tempfile.TemporaryDirectory(prefix="llmx_long_") as d:
        text_path, ids_path = os.path.join(d, "prompt.txt"), os.path.join(d, "reply.ids")
        with open(text_path, "w", encoding="utf-8", newline="") as f:
            f.write(prompt)
        with open(ids_path, "w", encoding="utf-8") as f:
            f.write(" ".join(str(i) for i in reply_ids[:-1]))
        cmd = [exe, "logits", model, "--file", text_path, "--chat", "--then-ids", ids_path,
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
    ap.add_argument("--tensor-width", type=int, default=1, help="split every layer of the device under test across this many devices of its --device list (docs/TENSOR-SPLIT.md)")
    ap.add_argument("--dtype", choices=("auto", "f16", "bf16", "f32", "int8"), default="auto",
                    help="activation dtype for generation and scoring on both backends")
    ap.add_argument("--tokens", type=int, default=16384, help="target prompt tokens")
    ap.add_argument("--max-tokens", type=int, default=512, help="tokens the device generates")
    ap.add_argument("--margin", type=float, default=0.5,
                    help="how far below the baseline's top logit a generated token may sit")
    ap.add_argument("--ctx-size", type=int, default=0,
                    help="KV budget; 0 uses the prompt plus the generation plus a margin")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=0, help="seconds to wait for each reply; 0 waits as long as it takes")
    ap.add_argument("--cli", action="store_true", help="run the device through two fresh `llmx generate` runs rather than `llmx serve`")
    args = ap.parse_args()
    args.exe = os.path.abspath(args.exe)

    global TIMEOUT
    TIMEOUT = args.timeout or None
    extra = ["--dtype", args.dtype]
    if args.threads:
        extra += ["--threads", str(args.threads)]
    ctx = args.ctx_size or (args.tokens + args.max_tokens + 512)
    # The device under test's flags: the baseline reads on one device.
    device_extra = extra + (["--tensor-width", str(args.tensor_width)] if args.tensor_width > 1 else [])

    # 1. The device twice, each from a fresh server, or a fresh process with --cli; the first run also sizes the message.
    runs = []
    prompt = None
    if args.cli:
        n, prompt = build_prompt_cli(args.exe, args.model, args.device, args.tokens, device_extra)
        print(f"prompt: {n} tokens, {len(prompt)} characters", flush=True)
    for i in range(2):
        if args.cli:
            got = generate_cli(args.exe, args.model, args.device, prompt, args.max_tokens, device_extra)
        else:
            proc, port, log = serve(args.exe, args.model, args.device, ctx, device_extra)
            try:
                if prompt is None:
                    n, prompt = build_prompt(port, args.tokens)
                    print(f"prompt: {n} tokens, {len(prompt)} characters", flush=True)
                got = run_once(port, prompt, args.max_tokens)
            finally:
                common.stop_server(proc, log)
        report(f"{args.device} #{i + 1}", got)
        runs.append(got)
    repeat_ok = runs[0].get("ids") == runs[1].get("ids")
    print(f"\nrepeatability: {'SAME' if repeat_ok else 'DIFFERENT'} tokens on two runs of {args.device}", flush=True)

    reply = runs[0].get("ids", [])
    if not reply:
        raise SystemExit("the device generated nothing")
    # The device reads its own tokens too, for a position past the margin, where both readings tell a near-tie from a wrong kernel; it does so now, so the device is free while the baseline reads.
    device_rows = baseline_logits(args.exe, args.model, args.device, prompt, reply, device_extra)
    print(f"{args.device} read its own {len(reply)} tokens; the baseline reads them next", flush=True)

    # 2. The baseline reads the prompt and the device's tokens.
    start = time.time()
    rows = baseline_logits(args.exe, args.model, args.baseline, prompt, reply, extra)
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
    # A position past the margin with both readings of it: the baseline's top candidates and the device's own.
    for j, gap in beyond[:10]:
        print(f"  token {j}: {gap:.3f} below the baseline's top; the device took {reply[j]}", flush=True)
        for name, found in ((args.baseline, rows), (args.device, device_rows)):
            if found and len(found) == len(reply):
                print(f"    {name:10s} " + ", ".join(f"{t} {l:.3f}" for t, l in found[j][1][:3]), flush=True)
    accuracy_ok = not beyond
    print(f"\n{'PASS' if repeat_ok and accuracy_ok else 'FAIL'}: repeatability {'ok' if repeat_ok else 'failed'}, "
          f"accuracy {'ok' if accuracy_ok else 'failed'}")
    return 0 if repeat_ok and accuracy_ok else 1


if __name__ == "__main__":
    sys.exit(main())
