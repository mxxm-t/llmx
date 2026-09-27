import json
import heapq
import itertools
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


# What a component returns when it compared nothing, which tests/run_tests.py reports as SKIP, neither a pass nor a failure.
SKIPPED = "skipped"
REQUIRED_DEVICE_TYPES = {}


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
    # LLMX_LOAD_MODE, set by run_tests.py --load-mode, reads every model's weights that way; test configuration again, reaching the binary only as the flag.
    mode = os.environ.get("LLMX_LOAD_MODE")
    if mode and "--load-mode" not in args and not synthetic:
        args += ["--load-mode", mode]
    return args


# How far from the reference's 5th logit two tokens may sit and still trade places at the top-5 boundary: near ties there reorder with the rounding of any backend that sums in another order.
# It also bounds llmx's own gap between the reference's 5th and 6th tokens when those two trade places.
TOP5_TIE_MARGIN = 0.1


def top5_overlap(ids, ref_ids, ref_logits, logits=(), margin=TOP5_TIE_MARGIN):
    """The reference's top-5 tokens found in `ids[:5]`, a boundary swap counting as agreement.

    A reference top-5 token missing from `ids[:5]` is forgiven only when the reference puts it within `margin` of its own 5th logit, and only against a token `ids[:5]` holds instead that the reference puts within `margin` below that logit; a strong token that vanishes, or one pulled in from further down, still counts as a miss.
    A swap of the reference's 5th and 6th tokens is also forgiven when `logits`, llmx's values for `ids`, puts the two within `margin`.
    Quantized weights can bring two tokens the reference keeps apart that close, and then any change of arithmetic reorders them.
    """
    top, fifth = set(ref_ids[:5]), ref_logits[4]
    ref = dict(zip(ref_ids, ref_logits))
    got = set(ids[:5])
    missing = [t for t in top - got if ref[t] - fifth <= margin]
    extra = [t for t in got - top if t in ref and fifth - ref[t] <= margin]
    forgiven = min(len(missing), len(extra))
    own = dict(zip(ids, logits))
    out, into = ref_ids[4], ref_ids[5]
    if (out not in got and into in got and not (out in missing and into in extra)
            and out in own and abs(own[into] - own[out]) <= margin):
        forgiven = min(len(top - got), forgiven + 1)
    return len(got & top) + forgiven



def _ranked_device_logits(row):
    require(len(row) >= 6 and all(math.isfinite(v) for v in row), "device comparison needs finite full-vocabulary rows with at least six tokens")
    return heapq.nlargest(10, range(len(row)), key=lambda i: (row[i], -i))


def check_device_rows(cpu_rows, device_rows, token_ids, max_logit_gap=None, *, calibration=False):
    """The quantization plan's CPU/device criterion over every position of one token sequence, in either execution path.
    Calibration retains the existing-type control's errors without applying candidate acceptance; the candidate uses that measured maximum as its limit.
    This supplements independent HF correctness and does not define a new HF bound."""
    require(len(token_ids) >= 2, "device comparison needs at least two token IDs")
    require(max_logit_gap is None or (math.isfinite(max_logit_gap) and max_logit_gap >= 0), "invalid calibrated logit gap")
    require(not calibration or max_logit_gap is None, "calibration measures its limit rather than accepting one")
    nll = [[], []]
    top1_failures, top5_failures = [], []
    rows, vocab, worst, checked, overlap = 0, None, 0.0, 0, 5
    for pos, (cpu, device) in enumerate(itertools.zip_longest(cpu_rows, device_rows)):
        require(cpu is not None and device is not None and pos < len(token_ids), "missing or extra logit row")
        top, got = _ranked_device_logits(cpu), _ranked_device_logits(device)
        if vocab is None:
            vocab = len(cpu)
            require(all(type(i) is int and 0 <= i < vocab for i in token_ids), "invalid token ID")
        require(len(cpu) == len(device) == vocab, "logit row width changed")
        if cpu[top[0]] - cpu[top[1]] > TOP5_TIE_MARGIN:
            if got[0] != top[0]:
                top1_failures.append(pos)
                require(calibration, "device top-1 differs outside the CPU tie margin at position %d" % pos)
            checked += 1
        current = top5_overlap(got, top, [cpu[i] for i in top], [device[i] for i in got])
        overlap = min(overlap, current)
        if current != 5:
            top5_failures.append(pos)
            require(calibration, "device top-5 differs outside the tie margin at position %d" % pos)
        worst = max(worst, max(abs(a - b) for a, b in zip(cpu, device)))
        if pos + 1 < len(token_ids):
            for result, row, best in zip(nll, (cpu, device), (top[0], got[0])):
                peak = row[best]
                result.append(peak - row[token_ids[pos + 1]] + math.log(math.fsum(math.exp(x - peak) for x in row)))
        rows += 1
    require(rows == len(token_ids), "missing logit rows")
    means = [math.fsum(values) / len(values) for values in nll]
    delta = abs(means[0] - means[1])
    require(all(math.isfinite(v) for v in means), "nonfinite device comparison NLL")
    require(calibration or delta <= 0.01, "device mean NLL differs from CPU by %.9g, bound 0.01" % delta)
    require(math.isfinite(worst) and (max_logit_gap is None or worst <= max_logit_gap),
            "device logit gap %.9g exceeds calibrated bound %s" % (worst, max_logit_gap))
    return dict(rows=rows, scored=rows - 1, top1_checked=checked, min_top5_overlap=overlap,
                max_logit_gap=worst, cpu_nll=means[0], device_nll=means[1], nll_delta=delta,
                top1_fail_positions=top1_failures, top5_fail_positions=top5_failures,
                criteria_passed=not top1_failures and not top5_failures and delta <= 0.01)


def check_device_greedy(cpu_rows, device_rows, cpu_ids, device_ids, *, calibration=False):
    """Both captures contain 64 finite argmax steps; their IDs must agree before the first CPU top-2 gap within TOP5_TIE_MARGIN.
    After that tie their histories may differ, so their subsequent logits are not compared.
    Calibration reports any earlier disagreement without applying candidate acceptance."""
    require(len(cpu_ids) == len(device_ids) == 64, "greedy comparison needs exactly 64 IDs per backend")
    first_tie, first_divergence, rows, vocab = None, None, 0, None
    for pos, (cpu, device) in enumerate(itertools.zip_longest(cpu_rows, device_rows)):
        require(cpu is not None and device is not None and pos < 64, "missing or extra greedy row")
        top, got = _ranked_device_logits(cpu), _ranked_device_logits(device)
        if vocab is None:
            vocab = len(cpu)
            require(all(type(i) is int and 0 <= i < vocab for i in cpu_ids + device_ids), "invalid greedy token ID")
        require(len(cpu) == len(device) == vocab, "greedy row width changed")
        require(cpu_ids[pos] == top[0] and device_ids[pos] == got[0], "greedy ID is not its row's argmax")
        if first_tie is None and cpu[top[0]] - cpu[top[1]] <= TOP5_TIE_MARGIN:
            first_tie = pos
        if cpu_ids[pos] != device_ids[pos]:
            if first_divergence is None:
                first_divergence = pos
            require(calibration or first_tie is not None, "greedy tokens differ before a CPU near-tie at step %d" % pos)
        rows += 1
    require(rows == 64, "missing greedy rows")
    return dict(matched_prefix=64 if first_divergence is None else first_divergence,
                checked_steps=64 if first_tie is None else first_tie,
                first_near_tie=first_tie, first_divergence=first_divergence,
                agrees_until_tie=first_divergence is None or (first_tie is not None and first_divergence >= first_tie))


def run_process(args, input=None, cache=None, text=False, timeout=None):
    """Run the llmx CLI from the repository root with the configured device flags and `input` on stdin, returning the finished process; its output is bytes unless `text`."""
    return subprocess.run([exe_path()] + device_args(args, cache), input=input, capture_output=True,
                          encoding="utf-8" if text else None, cwd=ROOT, timeout=timeout)


def run(args, cache=None):
    """Run the llmx CLI, returning (returncode, stdout_text)."""
    p = run_process(args, cache=cache, text=True)
    # A failure's diagnostic is on stderr; hand it back with the output so a caller can tell a missing device kernel from a wrong answer.
    return p.returncode, p.stdout if p.returncode == 0 else p.stdout + p.stderr


def cli_stdout(raw):
    """The bytes the CLI wrote to stdout, from the raw bytes its pipe carried.
    Windows text-mode stdout writes every line feed as CR LF, generated ones included, so there each pair is read back as the line feed it was; elsewhere the bytes arrive as written."""
    return raw.replace(b"\r\n", b"\n") if os.name == "nt" else raw


def generate_text(stdout):
    """The bytes `llmx generate` wrote between its `pp:` and `tg:` lines, from its raw stdout, without the line feed that ends the text.
    The lines `--verbose` adds, the prompt token count before them and the cache line after, may frame them; any other output raises ValueError."""
    out = cli_stdout(stdout)
    frame = re.fullmatch(rb"(?:prompt tokens: \d+\n)?pp: [^\n]*\n(.*)\ntg: [^\n]*\n(?:kv: [^\n]*\n)?", out, re.S)
    if not frame:
        raise ValueError("generate output outside the pp and tg frame: %r" % out)
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
    """True for a selected device's unsupported type, unless the runner requires that type, in which case the component fails."""
    if rc == 0 or not os.environ.get("LLMX_DEVICE"):
        return False
    match = re.search(r"inference: (?:embedding|head|layer [0-9]+'s (?:mixer|feed-forward part)) needs tensor [^\r\n]+ of type (?:[A-Z][A-Z0-9_]* \(([0-9]+)\)|([0-9]+)), "
                      r"which the backend of device [0-9]+ does not support", out)
    if not match:
        match = re.search(r"unsupported (?:matrix|embedding) type ([0-9]+)\b", out)
    if not match:
        return False
    type_id = int(next(value for value in match.groups() if value is not None))
    if type_id in REQUIRED_DEVICE_TYPES:
        raise RuntimeError("required device type %s (%d) refused: %s"
                           % (REQUIRED_DEVICE_TYPES[type_id], type_id, out.strip()))
    return True


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


# The components that check exact f32 arithmetic against independently generated fixtures run the CLI through this, which asks for f32 on both cache sides where the runtime would store f16.
def run_f32_cache(args):
    return run(args, cache="f32")


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


def tokenize_failures(model, cases, ids=None):
    """The tokenizer golden's cases whose text `llmx tokenize` on `model` does not turn into their `ids`, each as (text escaped to ASCII, the ids wanted, what the CLI gave).
    `ids`, when given, maps each position of a model that holds only part of a vocabulary to the id it stands for."""
    failures = []
    for case in cases:
        rc, out = run(["tokenize", model, case["text"]])
        if rc != 0:
            got = "exit %d: %s" % (rc, out.strip())
        else:
            got = [ids[p] for p in parse_ids(out)] if ids else parse_ids(out)
        if got != case["ids"]:
            # The Windows console is not UTF-8, so the text is escaped rather than crashing the report on the cases most likely to fail.
            failures.append((case["text"].encode("unicode_escape").decode("ascii"), case["ids"], got))
    return failures


# The real-model HF gates, tests/baseline.py and tests/baseline_8b.py, check their outputs with these validators, each at its own model's vocabulary, context and bounds.
# A validator raises ValueError on the first rule an output breaks and returns what it measured otherwise.

def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_ids(output, expected):
    ids = parse_ids(output)
    require(ids == expected, "token IDs differ from HF")
    return {"tokens": len(ids)}


def check_logits(output, case, vocab, bounds, require_top1=True):
    """`llmx logits --top 10` for a reference case: the prompt's exact token count, then ten unique IDs below `vocab` with finite logits sorted from the top, none beyond `bounds["max_abs_logit"]`.
    HF's top-1 must lead unless the caller counts the top-1 matches over its cases, and the top-5 overlap as top5_overlap counts it must reach `bounds["top5_overlap"]`."""
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
    if require_top1:
        require(ids[0] == case["top_ids"][0], "top-1 %d, HF %d" % (ids[0], case["top_ids"][0]))
    overlap = top5_overlap(ids, case["top_ids"], case["top_logits"], values)
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
# On a device they are different kernels, so scoring only one way would leave one set of them without an HF check.
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


# The directories the dead-code and docs checks read, beside the files at the root; build trees and other untracked output at the root are left out.
TREE_DIRS = ("src", "tests", "tools", "docs", "cmake", "docker", ".github")
# Files read as text; any other file is only known to exist.
TEXT_SUFFIXES = (".hpp", ".cpp", ".h", ".comp", ".glsl", ".py", ".md", ".txt", ".yml", ".yaml", ".cmake", ".in", ".bat")


def tree_files():
    """The paths from the root, with '/' separators, of the files a commit of this tree would hold: git's tracked files still on disk and the new ones it does not ignore.
    A copy without .git, such as an export, is walked instead, build output and caches left out."""
    if os.path.exists(os.path.join(ROOT, ".git")):
        r = subprocess.run(["git", "-C", ROOT, "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                           capture_output=True)
        if r.returncode:
            raise RuntimeError("git ls-files failed in %s: %s" % (ROOT, r.stderr.decode(errors="replace").strip()))
        names = sorted(set(r.stdout.decode("utf-8").split("\0")) - {""})
        return [n for n in names if os.path.isfile(os.path.join(ROOT, n))]
    names = [n for n in os.listdir(ROOT) if os.path.isfile(os.path.join(ROOT, n))]
    for top in TREE_DIRS:
        for directory, dirs, files in os.walk(os.path.join(ROOT, top)):
            dirs[:] = sorted(d for d in dirs if d != "__pycache__")
            rel = os.path.relpath(directory, ROOT).replace(os.sep, "/")
            names += [rel + "/" + f for f in files if not f.endswith(".pyc")]
    return names


def read_tree():
    """(texts, paths): every text file of the tree by its path from the root, and the set of every file path, the files at the root and under TREE_DIRS only.
    The evidence under docs/benchmarks/ and the test data are dated or generated, so their files are paths only; line endings are read as line feeds, so a checkout with CR LF reads the same."""
    texts, paths = {}, set()
    for path in tree_files():
        if "/" in path and not path.startswith(tuple(d + "/" for d in TREE_DIRS)):
            continue
        paths.add(path)
        if path.endswith(TEXT_SUFFIXES) and not path.startswith(("docs/benchmarks/", "tests/data/")):
            with open(os.path.join(ROOT, path), encoding="utf-8", errors="replace") as f:
                texts[path] = f.read()
    return texts, paths


CPP_BLANKS = re.compile(r"""//[^\n]*|/\*.*?\*/|(?:u8|u|U|L)?R"([^(\s]*)\(.*?\)\1"|(?:u8|u|U|L)?"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])+'""", re.S)


def cpp_code(text, keep_strings=False):
    """C++ or GLSL `text` with its comments blanked and, unless `keep_strings`, its string and character literals emptied, every line feed kept so that line numbers hold."""
    def blank(m):
        s = m.group(0)
        keep = "\n" * s.count("\n")
        if s.startswith("/"):
            return " " + keep
        return s if keep_strings else ('""' + keep if not s.startswith("'") else "' '" + keep)
    return CPP_BLANKS.sub(blank, text)


# The findings the dead-code and docs checks still report, each with how often it occurs and why it is still there (AGENTS.md, Dead code and stale docs).
KNOWN_FINDINGS = os.path.join(ROOT, "tests", "data", "known_findings.txt")


def known_findings(text, checks):
    """The entries of the known-findings list `text`: (check, file, name) -> (times it occurs, reason).
    A line is `check | file | name | times | reason`; `#` starts a comment line.
    An entry whose check is not one of `checks`, every check of both components, is refused, as are one without a reason or a count, and one listed twice."""
    entries = {}
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = [f.strip() for f in line.split(" | ")]
        if len(fields) != 5 or not all(fields) or not re.fullmatch(r"[1-9]\d*", fields[3]):
            raise ValueError("known findings line %d: expected 'check | file | name | times | reason': %s" % (n, line))
        if fields[0] not in checks:
            raise ValueError("known findings line %d names no check the components have: %s" % (n, fields[0]))
        key = tuple(fields[:3])
        if key in entries:
            raise ValueError("known findings line %d lists %s again" % (n, " | ".join(key)))
        entries[key] = (int(fields[3]), fields[4])
    return entries


def load_known_findings():
    """The known-findings list, each line's check held to the checks of both components, so a mistyped one fails every job rather than staying unsettled."""
    import dead_code
    import docs_check
    with open(KNOWN_FINDINGS, encoding="utf-8") as f:
        return known_findings(f.read(), dead_code.SOURCE_CHECKS + dead_code.LINKED_CHECKS + docs_check.CHECKS)


def settle_findings(found, checks, listed, say=print):
    """Hold `found`, (check, file, name) -> [where each occurrence was seen], to the entries of `listed` whose check is one of `checks`.
    Says each finding the list lacks, each that occurs another number of times than listed, and each entry of those checks that no longer occurs, and returns True when there are none."""
    mine = {k: v for k, v in listed.items() if k[0] in checks}
    new = sorted(k for k in found if k not in mine)
    recount = sorted(k for k in found if k in mine and len(found[k]) != mine[k][0])
    gone = sorted(k for k in mine if k not in found)
    for k in new:
        say("  new finding: %s | %s | %s  (%s)" % (k + ("; ".join(found[k]),)))
    for k in recount:
        say("  occurs %d times, listed %d: %s | %s | %s  (%s)" % ((len(found[k]), mine[k][0]) + k + ("; ".join(found[k]),)))
    for k in gone:
        say("  listed but no longer found, so its line in tests/data/known_findings.txt goes: %s | %s | %s" % k)
    return not new and not recount and not gone
