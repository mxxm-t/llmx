"""The tiny DSpine, DFlash2 and DFlash sidecars beside the tiny qwen35 target and their goldens (docs/SPECULATIVE.md, section 11; docs/TESTS.md).

The sidecar's K and V heads differ from the target's, so its KV layers take a shape of their own (ModelPlan::draft_kv).
`python tests/fixture_dspine.py --exe LLMX [--variant dflash2|dflash] OUT.json` writes the golden: the target's taps from llmx-route-probe, then tools/dspine_reference.py over them, for seeds until every case has no near-tie.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile

import common
import qwen35
import spec_decode

# The sidecar: 2 layers of 2 query heads and 1 KV head 8 wide beside a target of 2 KV heads 40 wide, a block of 4 rows, 3 candidates a row, a window of 3 that cuts context and block rows alike.
SIDECAR = {"block_count": 2, "block_size": 4, "selector_top_k": 3, "attention.head_count": 2, "attention.head_count_kv": 1, "attention.key_length": 8,
           "attention.value_length": 8, "feed_forward_length": 16, "attention.sliding_window": 3, "rope.freq_base": 10000.0,
           "attention.layer_norm_rms_epsilon": 1e-6, "target_layers": [1, 2]}
CODES, ROUTES, MASK = 6, 5, 1
# DFlash2's own keys: two taps, a coefficient a channel (the target's width of 37 is prime), a selector of rank 5.
DFLASH2 = {"conv_kernel_size": 2, "conv_group_size": 1, "selector_rank": 5}
# A DFlash sidecar has no candidates and no attention.causal, its windowed first layer causal and its second over all positions not (model/arch/dflash.hpp, read_config).
VARIANTS = ("dspine", "dflash2", "dflash")
TARGET = "hv3"
# Prompts of N context rows, N + block_size within the target's context of 16.
PROMPTS = ["abc", "the cat s", "hello world"]
# A gap below this between the K-th and the next logit, or between a score row's top two, makes the generator try the next seed.
GAP = 100 * common.F32_HF_LOGIT_BOUND


def target_spec():
    return next(f for f in qwen35.FIXTURES if f["name"] == TARGET)


def tensors(seed, width, vocab, variant="dspine"):
    """The sidecar's tensors as (name, shape, values) in GGUF order, the fastest dimension first."""
    import numpy as np
    rng = np.random.default_rng(seed)
    E, T, D, H, KV, F = width, len(SIDECAR["target_layers"]), SIDECAR["attention.key_length"], SIDECAR["attention.head_count"], SIDECAR["attention.head_count_kv"], SIDECAR["feed_forward_length"]
    out = []

    def matrix(name, n_in, n_out):
        out.append((name, [n_in, n_out], (rng.standard_normal((n_out, n_in)) / np.sqrt(n_in)).astype(np.float32)))

    def norm(name, n):
        out.append((name, [n], (1 + 0.1 * rng.standard_normal(n)).astype(np.float32)))

    matrix("fc.weight", T * E, E)
    norm("enc.output_norm.weight", E)
    norm("output_norm.weight", E)
    dflash2, spine = variant == "dflash2", variant == "dspine"
    S, CK = DFLASH2["selector_rank"], DFLASH2["conv_kernel_size"]
    if dflash2:
        matrix("selector_hidden.weight", E, S)
        for name in ("selector_predecessor.weight", "selector_successor.weight"):
            out.append((name, [S, vocab], rng.standard_normal((vocab, S)).astype(np.float32)))
    elif spine:
        out.append(("dspine_codebook.weight", [CODES, vocab], (rng.standard_normal((vocab, CODES)) / np.sqrt(CODES)).astype(np.float32)))
        for name in ("dspine_gate.weight", "dspine_pred_gate.weight", "dspine_prev.weight"):
            matrix(name, CODES, ROUTES)
        out.append(("dspine_gate.bias", [ROUTES], (0.1 * rng.standard_normal(ROUTES)).astype(np.float32)))
    for l in range(SIDECAR["block_count"]):
        p = "blk.%d." % l
        if dflash2:
            for side in ("attn", "ffn"):
                out.append((p + side + "_conv_base", [E, CK, 2], (0.5 + 0.3 * rng.standard_normal((2, CK, E))).astype(np.float32)))
                matrix(p + side + "_conv_proj.weight", E, 2 * CK * E // DFLASH2["conv_group_size"])
        norm(p + "attn_norm.weight", E)
        matrix(p + "attn_q.weight", E, H * D)
        matrix(p + "attn_k.weight", E, KV * D)
        matrix(p + "attn_v.weight", E, KV * D)
        norm(p + "attn_q_norm.weight", D)
        norm(p + "attn_k_norm.weight", D)
        matrix(p + "attn_output.weight", H * D, E)
        norm(p + "ffn_norm.weight", E)
        matrix(p + "ffn_gate.weight", E, F)
        matrix(p + "ffn_up.weight", E, F)
        matrix(p + "ffn_down.weight", F, E)
        if spine:
            matrix(p + "dspine_read.weight", E, CODES)
            matrix(p + "dspine_write.weight", ROUTES, E)
    return out


def write_sidecar(path, target, seed, variant="dspine"):
    """The sidecar of `variant` and `seed` beside the target file `target`, with the target's tokenizer; returns its SHA-256."""
    t = spec_decode.GGUF(target)
    md = {k: v for k, v in t.metadata.items() if k.startswith("tokenizer.")}
    md["general.architecture"] = (8, "dflash")
    md["tokenizer.ggml.mask_token_id"] = (4, MASK)
    width = t.value("qwen35.embedding_length")
    md["dflash.embedding_length"] = (4, width)
    for key, value in list(SIDECAR.items()) + (list(DFLASH2.items()) if variant == "dflash2" else []):
        if key == "selector_top_k" and variant == "dflash":
            continue
        if key == "target_layers":
            md["dflash." + key] = (9, (5, value))
        elif isinstance(value, float):
            md["dflash." + key] = (6, value)
        else:
            md["dflash." + key] = (4, value)
    md["dflash.attention.sliding_window_pattern"] = (9, (7, [True] * SIDECAR["block_count"] if variant != "dflash" else [True, False]))
    if variant != "dflash":
        md["dflash.attention.causal"] = (7, variant == "dspine")
    md["dflash.rope.dimension_sections"] = (9, (5, [SIDECAR["attention.key_length"] // 2, 0, 0, 0]))
    vocab = next(x.shape[1] for x in t.tensors if x.name == "token_embd.weight")
    assert vocab == len(t.value("tokenizer.ggml.tokens")[1]), "fixture_dspine: the target's embedding rows are not its token list"
    spec_decode.write_gguf(path, md, [(name, shape, 0, values.tobytes()) for name, shape, values in tensors(seed, width, vocab, variant)])
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def taps(probe, target, directory, prompt):
    """The residual entering each tapped layer at every prompt position, [N, T, E], from llmx-route-probe on the CPU in one pass."""
    import numpy as np
    ids = os.path.join(directory, "prompt.ids")
    with open(ids, "w") as f:
        f.write(" ".join(str(i) for i in prompt))
    prefix = os.path.join(directory, "taps")
    layers = SIDECAR["target_layers"]
    p = subprocess.run([probe, target, ids, prefix, str(len(prompt)), str(len(prompt)), "cpu", ",".join(str(l) for l in layers)],
                       capture_output=True, encoding="utf-8", errors="replace", timeout=300)
    if p.returncode:
        raise SystemExit("llmx-route-probe gave exit %d: %s" % (p.returncode, p.stderr[-400:]))
    records = {r[0]: np.array(r[3], np.float32).reshape(r[1], r[2]) for r in qwen35.read_streams(prefix + ".streams")}
    return np.stack([records[l] for l in layers], 1)


def near_tie(case):
    """Whether a case's candidates or picks rest on a gap below GAP."""
    import numpy as np
    k = case["cand"].shape[1]
    order = np.sort(case["logits"], 1)[:, ::-1]
    if (order[:, k - 1] - order[:, k] < GAP).any():
        return True
    if "score" not in case:
        return False
    top = np.sort(case["score"], -1)[..., ::-1]
    return bool((top[..., 0] - top[..., 1] < GAP).any())


def generate(exe, out, variant="dspine"):
    import numpy as np
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
    import dspine_reference
    probe = os.path.join(os.path.dirname(os.path.abspath(exe)), "llmx-route-probe" + (".exe" if os.name == "nt" else ""))
    with tempfile.TemporaryDirectory(prefix="llmx_dspine_") as directory:
        target = qwen35.write_fixture(directory, target_spec())
        encoded = []
        for text in PROMPTS:
            p = subprocess.run([exe, "tokenize", target, text], capture_output=True, encoding="utf-8", errors="replace", timeout=60)
            encoded.append([int(i) for i in p.stdout.strip().split(",")])
            assert 0 < len(encoded[-1]) and len(encoded[-1]) + SIDECAR["block_size"] <= 16, (text, encoded[-1])
        tapped = [taps(probe, target, directory, prompt) for prompt in encoded]
        anchors = []
        for prompt in encoded:
            empty = os.path.join(directory, "empty.txt")
            open(empty, "w").close()
            ids = os.path.join(directory, "anchor.ids")
            with open(ids, "w") as f:
                f.write(" ".join(str(i) for i in prompt))
            p = subprocess.run([exe, "logits", target, "--file", empty, "--then-ids", ids, "--top", "1", "--device", "cpu", "--dtype", "f32",
                                "--cache-type-k", "f32", "--cache-type-v", "f32"], capture_output=True, encoding="utf-8", errors="replace", timeout=300)
            if p.returncode or not common.parse_logits(p.stdout)[0]:
                raise SystemExit("llmx logits gave no anchor for %s: %s" % (prompt, p.stderr.strip()))
            anchors.append(common.parse_logits(p.stdout)[0][0])
        sidecar = os.path.join(directory, "dspine.gguf")
        for seed in range(1, 200):
            digest = write_sidecar(sidecar, target, seed, variant)
            side, tgt = dspine_reference.Gguf(sidecar), dspine_reference.Gguf(target)
            cases = []
            for prompt, ids, tap, anchor in zip(PROMPTS, encoded, tapped, anchors):
                r = dspine_reference.block(side, tgt, tap, anchor)
                head = "output.weight" if "output.weight" in tgt.t else "token_embd.weight"
                r["logits"] = r["final"][1:] @ tgt.rows(head).T
                cases.append(dict(r, prompt=prompt, ids=ids, anchor=anchor))
            if not any(near_tie(c) for c in cases):
                break
        else:
            raise SystemExit("no seed below 200 gives cases without a near-tie")
    doc = {"target": TARGET, "variant": variant, "seed": seed, "sha256": digest, "gap": GAP, "cases": [
        {"prompt": c["prompt"], "ids": c["ids"], "anchor": c["anchor"], "drafts": [int(x) for x in c["drafts"]], "cand": c["cand"].tolist(),
         "logits": np.round(c["logits"].astype(np.float64), 7).tolist(), "ctx_k": np.round(c["ctx_k"].astype(np.float64), 7).tolist(),
         "ctx_v": np.round(c["ctx_v"].astype(np.float64), 7).tolist()} for c in cases]}
    with open(out, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    print("seed %d, drafts %s" % (seed, [c["drafts"] for c in doc["cases"]]))


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--exe", required=True)
    a.add_argument("--variant", choices=VARIANTS, default="dspine")
    a.add_argument("out")
    args = a.parse_args()
    generate(args.exe, args.out, args.variant)
