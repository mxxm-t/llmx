"""The tiny inkling2_loop fixture held to HF's Inkling2LoopForCausalLM (docs/INKLING2.md, 6. Correctness).

The fixture of tests/fixture_inkling2.py is written from the committed golden's seed, its file held to the reference's SHA-256.
`llmx logits` reads the golden's sequence through --then-ids, and every position's logits must be HF's within 2e-5 and each target's NLL within 1e-5: whole at lengths either side of the window, the relative extent and the log-scale floor, token by token, at ubatch 1, 3, 7 and 16 and at one and three threads.
Greedy `generate` after the golden's prompt gives HF's ids, and no router or greedy pick of the golden sits within 2e-5 of a tie.
On a configured device whose backend lacks one of the model's ops the file is refused at load by the op's name, and the component reports SKIP.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile

import common
import fixture_inkling2
from minimax import nll, rows

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import check_device  # noqa: E402

GOLDEN = os.path.join(common.ROOT, "tests", "data", "inkling2_reference.json")
DEVICE = "which the backend of its device does not implement"
F32 = ["--cache-type-k", "f32", "--cache-type-v", "f32"]
LOGIT_BOUND = 2e-5
NLL_BOUND = 1e-5
CUT_ROWS = 320   # a pass a group of two GPU backends cuts: two parts of at least 128 rows each (kBesideRows)


def held(label, got, golden):
    """The largest logit and NLL errors of `got` against the golden's rows from position 0, each asserted within its bound."""
    sequence, logits = golden["sequence"], golden["logits"]
    logit_error = max(abs(row[i] - logits[t][i]) for t, row in enumerate(got) for i in range(len(logits[t])))
    nll_error = max([abs(nll(got[t], sequence[t + 1]) - golden["nll"][t]) for t in range(len(got) - 1)] or [0.0])
    assert logit_error <= LOGIT_BOUND, "inkling2: %s logits differ from HF's by %.3g (bound %g)" % (label, logit_error, LOGIT_BOUND)
    assert nll_error <= NLL_BOUND, "inkling2: %s NLL differs from HF's by %.3g (bound %g)" % (label, nll_error, NLL_BOUND)
    return logit_error, nll_error


def gate(path, directory, golden, floor):
    config, sequence = golden["config"], golden["sequence"]
    vocab = config["unpadded_vocab_size"]
    closest = min(golden["router_margin"]["minimum"], golden["greedy_margin"])
    assert closest >= floor, "inkling2: a router or greedy pick sits %.3g from a tie" % closest
    window, extent, base = config["sliding_window_size"], config["rel_extent"], config["log_scaling_n_floor"]
    lengths = sorted({1, 2, window, window + 1, extent, extent + 1, base, base + 1, len(sequence)})
    runs = [("whole %d" % n, sequence[:n], []) for n in lengths]
    runs += [("decode", sequence, ["--per-token"]), ("threads 1", sequence, ["--threads", "1"]), ("threads 3", sequence, ["--threads", "3"])]
    runs += [("ubatch %d" % u, sequence, ["--ubatch", str(u)]) for u in (1, 3, golden["chunk"], 16)]
    worst, whole = [0.0, 0.0], None
    for label, ids, flags in runs:
        got = rows(path, directory, ids, vocab, flags)
        if len(ids) == len(sequence) and not flags:
            whole = got
        worst = [max(a, b) for a, b in zip(worst, held(label, got, golden))]
    mean = sum(nll(whole[t], sequence[t + 1]) for t in range(len(sequence) - 1)) / (len(sequence) - 1)
    assert abs(mean - golden["mean_nll"]) <= NLL_BOUND, "inkling2: mean NLL %.9f against HF's %.9f" % (mean, golden["mean_nll"])
    p = common.run_process(["generate", path, golden["prompt"], "-n", str(len(golden["greedy"])), "--temp", "0", "--ignore-eos", "--verbose"] + F32)
    assert p.returncode == 0, "inkling2: generate failed: %s" % p.stderr.decode("utf-8", "replace")
    line = re.search(rb"^ids:(.*)$", common.cli_stdout(p.stdout), re.M)
    ids = [int(i) for i in line[1].replace(b",", b" ").split()]
    assert ids == golden["greedy"], "inkling2: greedy ids %s, HF's %s" % (ids, golden["greedy"])
    for flags in (["--yarn", "2"], ["--yarn", "2", "--yarn-original-ctx", "64"]):
        p = common.run_process(["logits", path, "a"] + flags + F32, text=True)
        assert p.returncode == 1 and "this model has none" in p.stderr, "inkling2: %s was not refused: %s" % (" ".join(flags), p.stderr[-400:])
    print("inkling2: %d runs from 1 to %d tokens (window %d, extent %d, log floor %d), every logit within %.3g of HF's (bound %g), each target's NLL within %.3g (bound %g), "
          "mean NLL %.9f against %.9f, %d greedy ids HF's, no router or greedy pick nearer a tie than %.3g"
          % (len(runs), len(sequence), window, extent, base, worst[0], LOGIT_BOUND, worst[1], NLL_BOUND, mean, golden["mean_nll"], len(ids), closest))


def grouped(path, directory, golden, require):
    """The golden's sequence on a tensor group of two CPU backends, which llmx-model-logits forms since the command line lists a device once: every batched and per-token row HF's within the bounds."""
    tool = os.path.join(os.path.dirname(common.exe_path()), "llmx-model-logits" + (".exe" if os.name == "nt" else ""))
    if not os.path.exists(tool):
        assert not require, "inkling2: llmx-model-logits is not beside the executable, and --require-tools asks for it"
        print("inkling2: SKIP the tensor group - llmx-model-logits is not beside the executable")
        return
    sequence, logits = golden["sequence"], golden["logits"]
    ids, prefix = os.path.join(directory, "group.ids"), os.path.join(directory, "group")
    with open(ids, "w") as f:
        f.write(" ".join(str(i) for i in sequence))
    cmd = [tool, path, ids, prefix, "cpu,cpu", "f32", "3", "0", "-", "2"]
    p = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=600)
    assert p.returncode == 0, "inkling2: the group of two failed: %s\n%s" % (" ".join(cmd), p.stderr[-800:])
    vocab = os.path.getsize(prefix + ".batched.bin") // (4 * len(sequence))
    worst = [0.0, 0.0]
    for phase in ("batched", "decode"):
        got = [list(r) for r in check_device.captured_rows(prefix, phase, len(sequence), vocab)]
        worst = [max(a, b) for a, b in zip(worst, held("group of two " + phase, [dict(enumerate(r[:golden["config"]["unpadded_vocab_size"]])) for r in got], golden))]
    print("inkling2: a tensor group of two CPU backends, batched and per token, every logit within %.3g of HF's (bound %g), each target's NLL within %.3g (bound %g), %d positions of %d logits"
          % (worst[0], LOGIT_BOUND, worst[1], NLL_BOUND, len(logits), len(logits[0])))


def cut(path, directory, golden, require):
    """A pass of CUT_ROWS rows on a tensor group of two of the configured devices, which cuts it in two parts (Model::cut_record): its first rows HF's within the bounds, and every row the per-token rows' within the logit bound."""
    devices = list(dict.fromkeys(d for d in os.environ.get("LLMX_DEVICE", "cpu").split(",") if d != "cpu"))
    if len(devices) < 2:
        print("inkling2: SKIP the cut pass - it needs two GPU devices configured (a CPU group never cuts, and a collective refuses one device twice)")
        return
    tool = os.path.join(os.path.dirname(common.exe_path()), "llmx-model-logits" + (".exe" if os.name == "nt" else ""))
    if not os.path.exists(tool):
        assert not require, "inkling2: llmx-model-logits is not beside the executable, and --require-tools asks for it"
        print("inkling2: SKIP the cut pass - llmx-model-logits is not beside the executable")
        return
    group = ",".join(devices[:2])
    sequence = (golden["sequence"] * (CUT_ROWS // len(golden["sequence"]) + 1))[:CUT_ROWS]
    ids, prefix = os.path.join(directory, "cut.ids"), os.path.join(directory, "cut")
    with open(ids, "w") as f:
        f.write(" ".join(str(i) for i in sequence))
    cmd = [tool, path, ids, prefix, group, "f32", str(CUT_ROWS), "0", "-", "2", "f32"]
    p = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=600)
    assert p.returncode == 0, "inkling2: the cut pass failed: %s\n%s" % (" ".join(cmd), p.stderr[-800:])
    assert "cuts a pass" in p.stderr, "inkling2: the group of %s recorded the %d-row pass whole, so the cut was not reached:\n%s" % (group, CUT_ROWS, p.stderr[-800:])
    vocab = os.path.getsize(prefix + ".batched.bin") // (4 * CUT_ROWS)
    batched, decode = ([list(r) for r in check_device.captured_rows(prefix, phase, CUT_ROWS, vocab)] for phase in ("batched", "decode"))
    unpadded, n = golden["config"]["unpadded_vocab_size"], len(golden["sequence"])
    worst = held("cut pass", [dict(enumerate(r[:unpadded])) for r in batched[:n]], golden)
    apart = max(abs(a - b) for x, y in zip(batched, decode) for a, b in zip(x[:unpadded], y[:unpadded]))
    assert apart <= LOGIT_BOUND, "inkling2: the cut pass's rows differ from the per-token rows by %.3g (bound %g)" % (apart, LOGIT_BOUND)
    print("inkling2: a %d-row pass cut on the group of %s, its first %d rows within %.3g of HF's logits and %.3g of the NLL, every row within %.3g of the per-token rows (bound %g)"
          % (CUT_ROWS, group, n, worst[0], worst[1], apart, LOGIT_BOUND))


def run(require=False):
    if common.tensor_width_skip("inkling2") or common.f32_cache_skip("inkling2"):
        return common.SKIPPED
    with open(GOLDEN, encoding="utf-8") as f:
        document = json.load(f)
    golden = document["fixture"]
    assert golden["config"] == fixture_inkling2.CONFIG, "inkling2: the golden's config is not the fixture's"
    state = fixture_inkling2.make_state(golden["config"], golden["seed"])
    assert fixture_inkling2.state_sha256(state) == golden["state_sha256"], "inkling2: the fixture's weights are not the reference's"
    with tempfile.TemporaryDirectory(prefix="llmx_inkling2_") as directory:
        path = str(fixture_inkling2.write_inkling2(os.path.join(directory, "inkling2.gguf"), golden["config"], state))
        with open(path, "rb") as f:
            assert hashlib.sha256(f.read()).hexdigest() == golden["file_sha256"], "inkling2: the fixture's file is not the reference's"
        if any(d != "cpu" for d in os.environ.get("LLMX_DEVICE", "cpu").split(",")):
            p = common.run_process(["logits", path, "a"] + F32, text=True)
            if p.returncode == 1 and DEVICE in p.stderr:
                print("inkling2: SKIP - the device refuses the model at load (%s), so nothing was compared" % p.stderr.strip().splitlines()[-1])
                return common.SKIPPED
        gate(path, directory, golden, document["margin_floor"])
        grouped(path, directory, golden, require)
        cut(path, directory, golden, require)
    return True
