"""The deepseek41 module's names held to the step 2 writer's files (docs/DEEPSEEK4.md, step 4).

Both layouts are written from the committed golden by tests/fixture_deepseek4.py, standard library only.
The gate (step 7, piece 7): `llmx logits` on the fork layout reads the golden's 200 ids through --then-ids, and every position's 128 logits must be the reference's within 2e-5 and every target's NLL, taken from those rows, within 1e-5.
It runs as the reference's 11 runs do, each prefix whole at 1, 2, 3, 64, 127, 128, 129 and 200 tokens and the decode path from the first token (--per-token), and the whole sequence again at ubatch 1, 3 and 16, and at one and three threads.
No selection of the golden sits within 2e-5 of a tie (margin_floor), so no run needs forced mode while the logits hold.
The upstream layout is refused by its key.
On a configured device, which has none of the model's ops yet, the file is refused at load naming the head's op, and the component reports SKIP.
Its entries are kept in f32 only, so either cache side given as f16 is refused by its flag before the model is built.
`serve` refuses the model by name until step 11, `generate --drafter lookup` gives the ids it gives with drafts off (`common.check_drafts`), and `info` lists the fork file's keys and tensors, and llmx-arch-deepseek4-test, found beside the executable, holds the file to the one its own C++ list builds.
"""
import json
import math
import os
import subprocess
import tempfile

import common
import fixture_deepseek4

GOLDEN = os.path.join(common.ROOT, "tests", "data", "deepseek41_reference.json")
UPSTREAM = "inference: unsupported metadata deepseek41.ple.layers"
DEVICE = "inference: the head needs hc_collapse_norm, which the backend of its device does not implement"
SERVE = "server: deepseek41 is not served until its checkpoints, pause and prefix reuse are built"
F32 = ["--cache-type-k", "f32", "--cache-type-v", "f32"]
LOGIT_BOUND = 2e-5
NLL_BOUND = 1e-5


def tool_path():
    return os.path.join(os.path.dirname(common.exe_path()), "llmx-arch-deepseek4-test" + (".exe" if os.name == "nt" else ""))


def refused(path, text, flags=()):
    """`logits` on `path` with `flags` exits 1 with `text` as its error, its stdout at most the prompt's token count printed before the first pass."""
    p = common.run_process(["logits", path, "a"] + list(flags), text=True)
    assert p.returncode == 1 and text in p.stderr and all(line.startswith("tokens: ") for line in p.stdout.splitlines()), \
        "deepseek4: logits on %s gave exit %d, stdout %r, stderr %r" % (os.path.basename(path), p.returncode, p.stdout, p.stderr)


def rows(path, directory, ids, flags=()):
    """The logits rows `logits` prints for `ids`, read through --then-ids after an empty text, every position with all 128 ids, as a list of id-to-logit maps."""
    text = os.path.join(directory, "empty.txt")
    open(text, "w").close()
    tail = os.path.join(directory, "sequence.ids")
    with open(tail, "w") as f:
        f.write(" ".join(str(i) for i in ids))
    args = ["logits", path, "--file", text, "--then-ids", tail, "--last", str(len(ids)), "--top", "128"] + F32 + list(flags)
    p = common.run_process(args, text=True)
    assert p.returncode == 0, "deepseek4: logits %s gave exit %d, stderr %r" % (" ".join(flags), p.returncode, p.stderr[-400:])
    lines = p.stdout.splitlines()
    assert lines[0] == "tokens: %d" % len(ids), lines[0]
    result = []
    for position, line in enumerate(lines[1:]):
        fields = line.split()
        assert int(fields[0]) == position and len(fields) == 1 + 2 * 128, "deepseek4: row %d of %s: %r" % (position, flags, line[:80])
        result.append({int(i): float(v) for i, v in zip(fields[1::2], fields[2::2])})
    assert len(result) == len(ids), (flags, len(result))
    return result


def nll(row, target):
    peak = max(row.values())
    return peak + math.log(sum(math.exp(v - peak) for v in row.values())) - row[target]


def held(name, got, document):
    """The largest logit and NLL errors of `got`, rows from position 0, against the golden, each asserted within its bound."""
    sequence, logits = document["sequence"], document["logits"]
    logit_error = max(abs(row[i] - logits[t][i]) for t, row in enumerate(got) for i in range(len(logits[t])))
    nll_error = max([abs(nll(got[t], sequence[t + 1]) - document["nll"][t]) for t in range(len(got) - 1)] or [0.0])
    assert logit_error <= LOGIT_BOUND, "deepseek4: %s logits differ from the reference's by %.3g (bound %g)" % (name, logit_error, LOGIT_BOUND)
    assert nll_error <= NLL_BOUND, "deepseek4: %s NLL differs from the reference's by %.3g (bound %g)" % (name, nll_error, NLL_BOUND)
    return logit_error, nll_error


def gate(path, directory, document):
    """The reference's runs, then the whole sequence over passes and threads; returns the largest errors."""
    sequence = document["sequence"]
    closest = min(margins["minimum"] for margins in document["margins"].values())
    assert closest >= document["margin_floor"], "deepseek4: a selection sits %.3g from a tie, which needs forced mode" % closest
    runs = [("whole %d" % length, sequence[:length], []) for length in sorted(set(r["length"] for r in document["runs"] if r["kind"] == "whole"))]
    runs += [("decode", sequence, ["--per-token"]), ("ubatch 1", sequence, ["--ubatch", "1"]), ("ubatch 3", sequence, ["--ubatch", "3"]),
             ("ubatch 16", sequence, ["--ubatch", "16"]), ("threads 1", sequence, ["--threads", "1"]), ("threads 3", sequence, ["--threads", "3"])]
    worst = [0.0, 0.0]
    whole = None
    for name, ids, flags in runs:
        got = rows(path, directory, ids, flags)
        if len(ids) == len(sequence) and not flags:
            whole = got
        errors = held(name, got, document)
        worst = [max(a, b) for a, b in zip(worst, errors)]
    mean = sum(nll(whole[t], sequence[t + 1]) for t in range(len(sequence) - 1)) / (len(sequence) - 1)
    assert abs(mean - document["mean_nll"]) <= NLL_BOUND, "deepseek4: mean NLL %.9f against the reference's %.9f" % (mean, document["mean_nll"])
    print("deepseek4: %d runs, every position's 128 logits within %.3g of the reference (bound %g), each target's NLL within %.3g (bound %g), mean NLL %.9f against %.9f, no selection nearer a tie than %.3g"
          % (len(runs), worst[0], LOGIT_BOUND, worst[1], NLL_BOUND, mean, document["mean_nll"], closest))


def run(require=False):
    with open(GOLDEN, encoding="utf-8") as f:
        document = json.load(f)
    args, state, engram = fixture_deepseek4.fixture_from_document(document)
    with tempfile.TemporaryDirectory(prefix="llmx_deepseek4_") as directory:
        paths = {layout: fixture_deepseek4.write_deepseek41(os.path.join(directory, layout + ".gguf"), args, state, engram, layout)
                 for layout in fixture_deepseek4.LAYOUTS}
        fork = str(paths["fork"])
        # No device backend has the model's ops yet, so a configured device refuses the file at load by the first op it lacks, the head's, and nothing is compared.
        if any(d != "cpu" for d in os.environ.get("LLMX_DEVICE", "cpu").split(",")):
            refused(fork, DEVICE, F32)
            print("deepseek4: SKIP - the device refuses the model at load (%s), so nothing was compared" % DEVICE)
            return common.SKIPPED
        gate(fork, directory, document)
        refused(str(paths["upstream"]), UPSTREAM)
        print("deepseek4: the upstream layout is refused by its key")
        refused(fork, "--cache-type-k: this model's entries are kept in f32 only, not f16", ["--cache-type-k", "f16", "--cache-type-v", "f32"])
        refused(fork, "--cache-type-v: this model's entries are kept in f32 only, not f16", ["--cache-type-k", "f32", "--cache-type-v", "f16"])
        print("deepseek4: an f16 cache side refused by its flag")
        p = common.run_process(["serve", fork, "--host", "127.0.0.1", "--port", str(common.free_port())], text=True, timeout=120)
        assert p.returncode == 1 and SERVE in p.stderr, "deepseek4: serve gave exit %d, stderr %r" % (p.returncode, p.stderr[-400:])
        print("deepseek4: serve refuses the model by name until step 11")
        common.check_drafts("deepseek4", fork)
        print("deepseek4: generate with --drafter lookup at 1, 4, 8 and 16 drafts, greedy and seeded, gives the ids it gives with drafts off")
        metadata, tensors = fixture_deepseek4.map_deepseek41(args, state, engram, "fork")
        p = common.run_process(["info", fork], text=True)
        assert p.returncode == 0, "deepseek4: info failed: %s" % p.stderr
        listed = set(line.split(" = ", 1)[0].strip() for line in p.stdout.splitlines() if " = " in line)
        names = set(line.split()[1] for line in p.stdout.splitlines() if line.startswith("  ") and " shape=" in line)
        assert set(metadata) <= listed, "deepseek4: info lists no %s" % sorted(set(metadata) - listed)
        assert names == set(t[0] for t in tensors), "deepseek4: info's tensors differ from the writer's: %s" % sorted(names ^ set(t[0] for t in tensors))
        print("deepseek4: info lists the fork file's %d keys and %d tensors" % (len(metadata), len(tensors)))
        tool = tool_path()
        if not os.path.exists(tool):
            assert not require, "deepseek4: %s not found beside the executable" % tool
            print("deepseek4: the C++ list not compared - %s not found beside the executable" % tool)
            return True
        p = subprocess.run([tool, GOLDEN, "--gguf", fork], capture_output=True, encoding="utf-8", errors="replace", timeout=600)
        assert p.returncode == 0, "deepseek4: the writer's fork file is not the test's: %s%s" % (p.stdout, p.stderr)
        print("deepseek4: the writer's fork file holds the arch-deepseek4 test's keys and tensors")
        for line in p.stdout.splitlines():
            if line.startswith(("arch-deepseek4: layer ", "arch-deepseek4: a 512-row", "arch-deepseek4: the writer's file over", "arch-deepseek4: a 12", "arch-deepseek4: one CPU", "arch-deepseek4: a fork", "arch-deepseek4: two CPUs", "arch-deepseek4: an abort")):
                print("deepseek4: " + line.split(": ", 1)[1])
    return True
