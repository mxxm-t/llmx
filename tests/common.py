import json
import math
import os
import re
import socket
import subprocess
import sys
import struct
import tempfile
import time
import urllib.request

# Shared helpers for synthetic tests, the optional real-model HF baseline and the tools in tools/ that run the CLI or a server.

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(ROOT, "llmx.exe" if os.name == "nt" else "llmx")


def exe_path():
    if not os.path.exists(EXE):
        raise SystemExit("Executable not found: %s. Build first and pass --exe to tests/run_tests.py." % EXE)
    return EXE


# Commands that take --device.
# LLMX_DEVICE is test configuration, like LLMX_BASELINE_GGUF: it never reaches the binary except as this flag.
DEVICE_COMMANDS = {"generate", "chat", "logits", "perplexity", "bench", "serve"}


def device_args(args, cache=None):
    args = list(args)
    if not args or args[0] not in DEVICE_COMMANDS:
        return args
    # The synthetic bench, `bench` without --model, times one device's kernels on a model of its own, so it takes only --device and --threads: it gets neither layer shares nor cache types.
    synthetic = args[0] == "bench" and "--model" not in args
    # A command that names its own device keeps it, and the configured layer shares, which belong to the configured devices, stay off it too.
    if "--device" not in args:
        device = os.environ.get("LLMX_DEVICE")
        if device:
            args += ["--device", device]
        # LLMX_LAYER_SHARES, set by run_tests.py --layer-shares, fixes each listed device's proportion of the layers, so a split the fit would not choose on this machine is tested anyway.
        shares = os.environ.get("LLMX_LAYER_SHARES")
        if shares and "--layer-shares" not in args and not synthetic:
            args += ["--layer-shares", shares]
    # LLMX_CACHE_TYPE, set by run_tests.py --cache-type, runs the same commands with both cache sides stored as that type; test configuration like LLMX_DEVICE, reaching the binary only as flags.
    # `cache` is a component asking for a type because its fixtures need it, which an explicit LLMX_CACHE_TYPE overrides.
    want = os.environ.get("LLMX_CACHE_TYPE") or cache
    if want and "--cache-type-k" not in args and not synthetic:
        args += ["--cache-type-k", want, "--cache-type-v", want]
    return args


# How far from the reference's 5th logit two tokens may sit and still trade places at the top-5 boundary: near ties there reorder with the rounding of any backend that sums in another order.
TOP5_TIE_MARGIN = 0.1


def top5_overlap(ids, ref_ids, ref_logits, margin=TOP5_TIE_MARGIN):
    """The reference's top-5 tokens found in `ids[:5]`, a boundary swap counting as agreement.

    A reference top-5 token missing from `ids[:5]` is forgiven only when the reference puts it within `margin` of its own 5th logit,
    and only against a token `ids[:5]` holds instead that the reference puts within `margin` below that logit; a strong token that
    vanishes, or one pulled in from further down, still counts as a miss.
    """
    top, fifth = set(ref_ids[:5]), ref_logits[4]
    ref = dict(zip(ref_ids, ref_logits))
    got = set(ids[:5])
    missing = [t for t in top - got if ref[t] - fifth <= margin]
    extra = [t for t in got - top if t in ref and fifth - ref[t] <= margin]
    return len(got & top) + min(len(missing), len(extra))


def run_process(args, input=None, cache=None, text=False, cwd=None, timeout=None):
    """Run the llmx CLI with the configured device flags and `input` on stdin, returning the finished process; its output is bytes unless `text`."""
    return subprocess.run([exe_path()] + device_args(args, cache), input=input, capture_output=True,
                          encoding="utf-8" if text else None, cwd=cwd or ROOT, timeout=timeout)


def run(args, cwd=None, cache=None):
    """Run the llmx CLI, returning (returncode, stdout_text)."""
    p = run_process(args, cache=cache, text=True, cwd=cwd)
    # A failure's diagnostic is on stderr; hand it back with the output so a caller can tell a missing device kernel from a wrong answer.
    return p.returncode, p.stdout if p.returncode == 0 else p.stdout + p.stderr


def generate_text(stdout):
    """The bytes `llmx generate` wrote between its `pp:` and `tg:` lines, from its raw stdout, without the line feed that ends the text.
    The lines `--verbose` adds, the prompt token count before them and the cache line after, may frame them; any other output fails the assertion."""
    # Windows text-mode stdout writes every line feed as CR LF, generated ones included, so each pair is read back as the line feed it was.
    out = stdout.replace(b"\r\n", b"\n")
    frame = re.fullmatch(rb"(?:prompt tokens: \d+\n)?pp: [^\n]*\n(.*)\ntg: [^\n]*\n(?:kv: [^\n]*\n)?", out, re.S)
    assert frame, out
    return frame[1]


def free_port():
    """A loopback port the system has just handed out and nothing holds."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def start_server(command, wait=120):
    """Start `command`, an `llmx serve` command line, on a free loopback port and return (process, port, log) once /v1/health answers ok.
    The server logs a line per request, so its output goes to a temporary file: a pipe nobody reads would fill and block it."""
    port = free_port()
    # The log names the model by its file name, which on Linux can hold bytes that are not UTF-8, so reading it back replaces them rather than raising in place of the real error.
    log = tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(list(command) + ["--host", "127.0.0.1", "--port", str(port)],
                            stdout=log, stderr=subprocess.STDOUT)
    # Anything that ends the wait without a healthy server, a health reply that is not UTF-8 or not JSON included, stops the server before it goes up to the caller.
    try:
        deadline = time.time() + wait
        while time.time() < deadline:
            if proc.poll() is not None:
                log.seek(0)
                raise RuntimeError("server exited early: " + log.read())
            try:
                with urllib.request.urlopen("http://127.0.0.1:%d/v1/health" % port, timeout=30) as r:
                    if json.load(r).get("status") == "ok":
                        return proc, port, log
            except OSError:
                pass  # not listening yet
            time.sleep(0.1)
        raise RuntimeError("server did not come up")
    except BaseException:
        stop_server(proc, log)
        raise


def stop_server(proc, log):
    """Stop a server start_server started and close its log.
    The server has no shutdown of its own to wait for, so it is killed."""
    proc.kill()
    proc.wait()
    log.close()


def leave_mid_stream(port, body, after_bytes):
    """A client that leaves mid-stream: `body` goes to /v1/generate as a stream, and the socket closes once `after_bytes` of the reply have arrived, or the server has closed it first."""
    data = json.dumps(dict(body, stream=True)).encode()
    s = socket.create_connection(("127.0.0.1", port), timeout=600)
    s.sendall(b"POST /v1/generate HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n"
              % len(data) + data)
    got = 0
    while got < after_bytes:
        chunk = s.recv(4096)
        if not chunk:
            break
        got += len(chunk)
    s.close()


def device_lacks_kernel(rc, out):
    """True when the selected device refused the model for want of a kernel,
    which a test reports as skipped rather than failed."""
    return (rc != 0 and bool(os.environ.get("LLMX_DEVICE")) and
            ("unsupported matrix type" in out or "unsupported embedding type" in out))


def write_bin(path, floats):
    """Write a list of floats as little-endian f32."""
    with open(path, "wb") as f:
        for x in floats:
            f.write(struct.pack("<f", x))


def read_bin_floats(path):
    with open(path, "rb") as f:
        data = f.read()
    return struct.unpack("<%df" % (len(data) // 4), data)


def max_err(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


# The components that check exact f32 arithmetic against independently generated fixtures run the CLI through this, so they keep their f32 cache sides now that the runtime stores f16 by default.
def run_f32_cache(args, cwd=None):
    return run(args, cwd, cache="f32")


def f32_cache_skip(component):
    """True, after reporting the skip, when LLMX_CACHE_TYPE asks for a cache other than f32.
    The exact comparisons need f32 caches; an f16 cache rounds keys and values and is checked by the real-model gate."""
    cache = os.environ.get("LLMX_CACHE_TYPE", "f32")
    if cache == "f32":
        return False
    print("%s: SKIP - its exact comparisons are made with f32 caches (LLMX_CACHE_TYPE=%s)" % (component, cache))
    return True


def parse_logits(out):
    """The `id logit` rows `llmx logits` prints, as a list of ids and a list of logits in its order."""
    ids, values = [], []
    for line in out.splitlines():
        p = line.split()
        if len(p) == 2 and p[0].isdigit():
            ids.append(int(p[0]))
            values.append(float(p[1]))
    return ids, values


def perplexity_fields(out):
    """The `name: value` lines `llmx perplexity` prints; a line without a colon or a repeated name raises ValueError."""
    fields = {}
    for line in out.strip().splitlines():
        if ":" not in line:
            raise ValueError("malformed PPL line")
        key, value = line.split(":", 1)
        if key in fields:
            raise ValueError("duplicate PPL field: " + key)
        fields[key] = value.strip()
    return fields


def hf_logit_error(name, got, expected):
    """The largest distance of `got`, one position's {id: logit} over the whole vocabulary, from that position's logits in a tiny model's HF fixture, which must be within 2e-5."""
    assert set(got) == set(range(len(expected))), "missing %s logits" % name
    assert all(math.isfinite(v) for v in got.values()), "non-finite %s logits" % name
    error = max(abs(got[i] - value) for i, value in enumerate(expected))
    assert error < 2e-5, "%s/HF logit error: %.8f" % (name, error)
    return error


def parse_ids(out):
    """The token IDs `llmx tokenize` prints."""
    return [int(t) for t in out.replace(",", " ").split()]


# The real-model HF gates, tests/baseline.py and tests/baseline_8b.py, check their outputs with these validators, each at its own model's vocabulary, context and bounds.
# A validator raises ValueError on the first rule an output breaks and returns what it measured otherwise.

def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_ids(output, expected):
    ids = parse_ids(output)
    require(ids == expected, "token IDs differ from HF")
    return {"tokens": len(ids)}


def check_logits(output, case, vocab, bounds):
    """`llmx logits --top 10` for a reference case: the prompt's exact token count, then ten unique IDs below `vocab` with finite logits sorted from the top, none beyond `bounds["max_abs_logit"]`.
    HF's top-1 must lead, and the top-5 overlap as top5_overlap counts it must reach `bounds["top5_overlap"]`."""
    lines = output.strip().splitlines()
    require(len(lines) == 11 and lines[0] == "tokens: " + str(case["n_tokens"]),
            "wrong logit count or prompt token count")
    pairs = [line.split() for line in lines[1:]]
    require(all(len(pair) == 2 for pair in pairs), "malformed logits")
    ids = [int(pair[0]) for pair in pairs]
    values = [float(pair[1]) for pair in pairs]
    require(len(set(ids)) == 10 and all(0 <= token < vocab for token in ids),
            "duplicate or invalid logit token IDs")
    require(all(math.isfinite(value) and abs(value) <= bounds["max_abs_logit"] for value in values),
            "non-finite or implausible logits")
    require(all(a >= b for a, b in zip(values, values[1:])), "logits not sorted")
    require(ids[0] == case["top_ids"][0], "top-1 %d, HF %d" % (ids[0], case["top_ids"][0]))
    overlap = top5_overlap(ids, case["top_ids"], case["top_logits"])
    require(overlap >= bounds["top5_overlap"],
            "top-5 overlap %d/5 below bound %d" % (overlap, bounds["top5_overlap"]))
    return {"top1": ids[0], "top5_overlap": overlap, "top_ids": ids, "top_logits": values}


def check_ppl(output, case, total_tokens, context, bounds):
    """`llmx perplexity` for a reference case of a `total_tokens` text: exactly its fields, every count exact (the context size is `context` for the continuous case), and a finite mean NLL within `bounds["continuous_nll"]` of HF's, or `bounds["window_nll"]` in windows."""
    fields = perplexity_fields(output)
    counts = {"tokens": total_tokens, "used tokens": case["used_tokens"],
              "scored tokens": case["n_scored"], "chunks": case["chunks"],
              "context size": case["context_size"] or context}
    require(set(fields) == set(counts) | {"mean NLL", "perplexity"}, "missing or unexpected PPL fields")
    for key, expected in counts.items():
        require(int(fields[key]) == expected, "PPL %s %s, expected %d" % (key, fields[key], expected))
    nll, ppl = float(fields["mean NLL"]), float(fields["perplexity"])
    require(math.isfinite(nll) and math.isfinite(ppl) and nll >= 0 and ppl >= 1,
            "invalid NLL/PPL")
    bound = bounds["window_nll"] if case["context_size"] else bounds["continuous_nll"]
    delta = abs(nll - case["mean_nll"])
    require(delta <= bound, "mean NLL %.6f vs HF %.6f: difference %.9g exceeds bound %.9g"
            % (nll, case["mean_nll"], delta, bound))
    # The CLI prints six significant digits.
    require(math.isclose(ppl, math.exp(nll), rel_tol=2e-5), "inconsistent NLL/PPL")
    return dict(counts, mean_nll=nll, perplexity=ppl, hf_mean_nll=case["mean_nll"],
                absolute_nll_delta=delta, bound=bound)


# Every perplexity case is scored both ways: in batched passes, the prompt path, and one token at a time, the decode path.
# On a device they are different kernels, and scoring only one way once left one set of them without an HF check at all.
PPL_MODES = ("batched", "per-token")


def ppl_cases(doc):
    """A perplexity fixture's cases: the whole text in one window, then its windowed cases."""
    return [dict(doc, context_size=0, max_chunks=0, chunks=1, used_tokens=doc["n_tokens"])] + doc["chunk_cases"]


def ppl_command(model, path, case, mode, ubatch=None):
    """The `llmx perplexity` command scoring `case` of the text in the file `path` in `mode`, one of PPL_MODES."""
    args = ["perplexity", model, "--file", path, "--threads", "6"]
    if ubatch:
        args += ["--ubatch", str(ubatch)]
    if mode == "per-token":
        args.append("--per-token")
    if case["context_size"]:
        args += ["--ctx-size", str(case["context_size"])]
    if case["max_chunks"]:
        args += ["--chunks", str(case["max_chunks"])]
    return args


def check_hf_fixture(name, model, cases, perplexity, text, ubatches, placements=((),)):
    """A tiny F32 model against its HF fixture at 1 and 4 threads, with f32 caches.
    Every case's 257 logits at every ubatch and placement must be within the fixture bound (a placement other than the empty one runs at 4 threads only), then the windowed NLL of `text` for every perplexity case within 1e-5.
    Returns the largest logit error and the number of logit comparisons."""
    worst, count = 0.0, 0
    for threads in (1, 4):
        for ubatch in ubatches:
            for case, placement in ((c, p) for c in cases for p in placements if threads == 4 or not p):
                rc, out = run_f32_cache(["logits", model, case["text"], "--top", "257",
                                         "--threads", str(threads), "--ubatch", str(ubatch)] + list(placement))
                assert rc == 0, "%s logits failed: %s" % (name, out)
                worst = max(worst, hf_logit_error(name, dict(zip(*parse_logits(out))), case["logits"]))
                count += 1
        for case in perplexity:
            rc, out = run_f32_cache(["perplexity", model, text, "--threads", str(threads), "-c", str(case["context"])])
            assert rc == 0, "%s PPL failed: %s" % (name, out)
            error = abs(float(perplexity_fields(out)["mean NLL"]) - case["mean_nll"])
            assert math.isfinite(error) and error < 1e-5, "%s/HF NLL error: %.8f" % (name, error)
    return worst, count
