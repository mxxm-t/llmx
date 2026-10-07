"""The tensor split's numerics (docs/TENSOR-SPLIT.md, step 2): tiny models whose every split falls whole at widths 2 and 4, against goldens HF made from the same weights (tools/gen_baseline.py tensor-split, tests/data/baseline_tensor_split.json).

Groups of CPU backends are formed by llmx-model-logits, since the command line lists a device once: every row of the batched and per-token captures at the goldens' positions within the bound of the precision its path computes in, the windowed NLL, and the device-reference criterion of tests/common.py against the same model on one device, ranking, NLL, calibration and greedy agreement.
The qwen35 fixture, whose linear-attention layers keep a recurrent state, is held the same way: each member runs its K heads and the V heads that read them over its own state.
"""
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import common
import decode_probe
import f32
import qwen35
from tokenizer import build_byte_vocab

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import check_device  # noqa: E402

# Feed-forward width, heads, KV heads, the qwen35 K and V heads and the vocabulary all divide by 2 and 4; KV heads are replicated at width 4.
CONFIG = {"block_count": 2, "embedding_length": 40, "feed_forward_length": 32,
          "attention.head_count": 4, "attention.head_count_kv": 2,
          "attention.key_length": 16, "context_length": 16}
VOCAB = 260
TOKENS = build_byte_vocab() + ["<|endoftext|>"] + ["<|reserved_%d|>" % i for i in range(VOCAB - 257)]
QWEN35 = {"name": "split", "v_heads": 12, "tied": False, "mtp": False, "v_head": 8, "vocab": VOCAB,
          "config": {"embedding_length": 40, "feed_forward_length": 32, "attention.head_count": 4, "attention.head_count_kv": 2,
                     "attention.key_length": 16, "attention.value_length": 16, "rope.dimension_count": 8,
                     "ssm.state_size": 12, "ssm.group_count": 4}}
WIDTHS = ((2, "cpu,cpu"), (4, "cpu,cpu,cpu,cpu"))


def groups():
    """The one device and the groups the captures run on: CPU backends, or with run_tests.py --device and --tensor-width (LLMX_DEVICE, LLMX_TENSOR_WIDTH) that list's first device and its devices as groups of that width."""
    device, width = os.environ.get("LLMX_DEVICE"), os.environ.get("LLMX_TENSOR_WIDTH")
    if device and width and int(width) > 1:
        return device.split(",")[0], ((int(width), device),)
    return "cpu", WIDTHS
GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "baseline_tensor_split.json")


def golden():
    """The committed goldens, checked against the shapes and the weights this module builds."""
    with open(GOLDEN, encoding="utf-8") as f:
        doc = json.load(f)
    assert doc["config"] == CONFIG and doc["vocab"] == VOCAB and doc["qwen35"]["spec"] == QWEN35, "tensor-split: fixture shapes changed"
    for fixture in doc["qwen3"]:
        assert f32.weight_hash(f32.tensors(fixture["tied"], config=CONFIG, vocab=VOCAB)) == fixture["weights_sha256"], "tensor-split: qwen3 weights changed"
    assert f32.weight_hash(qwen35.hashed(qwen35.raw_weights(QWEN35))) == doc["qwen35"]["weights_sha256"], "tensor-split: qwen35 weights changed"
    return doc


def tool_path():
    return os.path.join(os.path.dirname(common.exe_path()), "llmx-model-logits" + (".exe" if os.name == "nt" else ""))


def capture(tool, model, ids, prefix, devices, width):
    """Every row of the batched and per-token passes and 64 greedy steps of `model` over `devices` as groups of `width`, and the capture's metadata."""
    cmd = [tool, model, ids, prefix, devices, "f32", "3", "0", "-", str(width)]
    result = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=600)
    assert result.returncode == 0, "tensor-split: capture failed: %s\n%s" % (" ".join(cmd), result.stderr)
    meta = json.loads(result.stdout)
    rows = {phase: [list(r) for r in check_device.captured_rows(prefix, phase, 64 if phase == "greedy" else len(meta["tokens"]), VOCAB)]
            for phase in ("batched", "decode", "greedy")}
    return meta, rows


def held_to_hf(label, meta, rows, fixture):
    """Each captured row at a golden text's last position within the bound of its path's precision, and the batched rows' NLL of the longest text, one window, within the HF NLL bound; the largest logit and NLL errors."""
    worst = 0.0
    for phase in ("batched", "decode"):
        for case in fixture["cases"]:
            row = rows[phase][len(case["text"]) - 1]
            worst = max(worst, common.hf_logit_error("%s %s %r" % (label, phase, case["text"]), dict(enumerate(row)), case["logits"],
                                                     meta["dtype"], meta["matrix_paths"][phase]))
    text = f32.TEXTS[-1].encode("ascii")
    total = 0.0
    for pos in range(len(text) - 1):
        row = rows["batched"][pos]
        shift = max(row)
        total += shift + math.log(math.fsum(math.exp(v - shift) for v in row)) - row[text[pos + 1]]
    nll = total / (len(text) - 1)
    want = next(p["mean_nll"] for p in fixture["perplexity"] if p["context"] == 16)
    bound = common.hf_bounds(meta["dtype"], meta["matrix_paths"]["batched"])["nll"]
    assert abs(nll - want) < bound, "%s: NLL %.8f against HF %.8f, bound %g" % (label, nll, want, bound)
    return worst, abs(nll - want)


def drafts(tool, model, directory, devices, width):
    """The embedded drafter's pick and two drafts after each of four prompts of `model` on `devices`, as llmx-decode-probe prints them with f32 caches: per prompt the pick, the drafted ids and each draft row's logits."""
    out = []
    for prompt in ("a", "ab", "hello", "The quick br"):
        p = decode_probe.probe(tool, model, directory, {"prompt": prompt, "ids": [], "tokens": [0, 1], "draft": 2, "cache": "f32"}, devices, width)
        assert p.returncode == 0, "tensor-split: llmx-decode-probe failed on %s at width %d: %s%s" % (devices, width, p.stdout, p.stderr)
        pick = re.search(r"^draft pick (\d+), (\d+) drafts$", p.stdout, re.M)
        rows = re.findall(r"^draft (\d+) (\d+):((?: \S+)+)$", p.stdout, re.M)
        assert pick and len(rows) == 2, "tensor-split: no drafts on %s at width %d: %s" % (devices, width, p.stdout)
        out.append((int(pick.group(1)), [int(r[1]) for r in rows], [[float(v) for v in r[2].split()] for r in rows]))
    return out


def check_drafter(directory, single, widths, require):
    """The qwen35 fixture with an MTP block as an embedded drafter over a group (docs/TENSOR-SPLIT.md, step 6): from four prompts, the pick and the two drafts must be one device's, whose drafts the qwen35 component holds to HF on its own fixture, and every draft row's logits within the F32 bound of one device's.
    Returns the largest difference, or None with a skip line where the tool is not beside the executable."""
    tool = decode_probe.tool_path()
    if not os.path.exists(tool):
        assert not require, "tensor-split: llmx-decode-probe is not beside the executable, and --require-tools asks for it"
        print("tensor-split: SKIP the drafter over a group - llmx-decode-probe is not beside the executable")
        return None
    spec = dict(QWEN35, name="split-mtp", mtp=True)
    model = qwen35.write_fixture(directory, dict(spec, config=dict(spec["config"], context_length=128)), tokens=TOKENS)
    one = drafts(tool, model, directory, single if single == "cpu" else single.split(":", 1)[1], 1)
    worst = 0.0
    for width, devices in widths:
        listed = ",".join(d if d == "cpu" else d.split(":", 1)[1] for d in devices.split(","))
        for (pick, ids, rows), (want_pick, want_ids, want_rows) in zip(drafts(tool, model, directory, listed, width), one):
            assert pick == want_pick and ids == want_ids, "tensor-split: a group of %d drafts %s after %d, one device %s after %d" % (width, ids, pick, want_ids, want_pick)
            worst = max(worst, max(abs(a - b) for row, want in zip(rows, want_rows) for a, b in zip(row, want)))
    assert worst < common.F32_HF_LOGIT_BOUND, "tensor-split: a group's draft logits %.8f from one device's" % worst
    return worst


def run(require=False):
    tool = tool_path()
    if not os.path.exists(tool):
        assert not require, "tensor-split: %s not found beside the executable" % tool
        print("tensor-split: SKIP - %s not found beside the executable" % tool)
        return common.SKIPPED
    if common.f32_cache_skip("tensor-split"):
        return common.SKIPPED
    doc = golden()
    single, widths = groups()
    worst_logit = worst_nll = 0.0
    with tempfile.TemporaryDirectory(prefix="llmx_tensor_split_") as directory:
        ids = os.path.join(directory, "ids.txt")
        Path(ids).write_text(" ".join(str(b) for b in f32.TEXTS[-1].encode("ascii")), encoding="ascii")
        # A context past the 13 tokens and 64 greedy steps the captures take; the logits at the goldens' positions do not depend on it.
        for fixture in doc["qwen3"]:
            name = "tied" if fixture["tied"] else "untied"
            model = f32.write_model(os.path.join(directory, "split-%s.gguf" % name), f32.tensors(fixture["tied"], config=CONFIG, vocab=VOCAB),
                                    config=dict(CONFIG, context_length=128), tokens=TOKENS)
            one_meta, one = capture(tool, model, ids, os.path.join(directory, name + "-1"), single, 1)
            held_to_hf("qwen3 %s, one device" % name, one_meta, one, fixture)
            for width, devices in widths:
                meta, rows = capture(tool, model, ids, os.path.join(directory, "%s-%d" % (name, width)), devices, width)
                logit, nll = held_to_hf("qwen3 %s, width %d" % (name, width), meta, rows, fixture)
                worst_logit, worst_nll = max(worst_logit, logit), max(worst_nll, nll)
                for phase in ("batched", "decode"):
                    common.check_device_rows(one[phase], rows[phase], list(f32.TEXTS[-1].encode("ascii")))
                common.check_device_greedy(one["greedy"], rows["greedy"], one_meta["greedy"], meta["greedy"])
        spec = QWEN35
        model = qwen35.write_fixture(directory, dict(spec, config=dict(spec["config"], context_length=128)), tokens=TOKENS)
        one_meta, one = capture(tool, model, ids, os.path.join(directory, "qwen35-1"), single, 1)
        held_to_hf("qwen35, one device", one_meta, one, doc["qwen35"])
        for width, devices in widths:
            meta, rows = capture(tool, model, ids, os.path.join(directory, "qwen35-%d" % width), devices, width)
            logit, nll = held_to_hf("qwen35, width %d" % width, meta, rows, doc["qwen35"])
            worst_logit, worst_nll = max(worst_logit, logit), max(worst_nll, nll)
            for phase in ("batched", "decode"):
                common.check_device_rows(one[phase], rows[phase], list(f32.TEXTS[-1].encode("ascii")))
            common.check_device_greedy(one["greedy"], rows["greedy"], one_meta["greedy"], meta["greedy"])
        drafted = check_drafter(directory, single, widths, require)
    print("tensor-split: the qwen3 fixtures, tied and untied, and the qwen35 fixture on %s against HF, max logit error %.8f and NLL error %.8f, and against %s by the device-reference criterion%s  [ok]"
          % (" and ".join("%s as width %d" % (devices, width) for width, devices in widths), worst_logit, worst_nll, single,
             "" if drafted is None else "; the embedded drafter's picks and drafts those of %s, draft logits within %.8f" % (single, drafted)))
    return True


if __name__ == "__main__":
    run()
