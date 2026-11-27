"""Numpy forward of the DFly2 drafter of DeepSeek-V4.1-Flash (docs/SPECULATIVE.md, section 12), written from the model description, and its tiny golden.

The forward reads its weights by their GGUF names (tests/fixture_deepseek4.py, map_dfly2) and computes in float64.
The target's side (context latents, their picks, the final hidden before the anchor, the embedding and the head) is an input, so the golden checks the drafter alone.
`python tools/dfly2_reference.py OUT_DIR` writes the tiny fixture's weights, inputs, intermediates and outputs as .npy files.
"""
import json
import math
import os
import sys

if __name__ == "__main__" and sys.argv[1:2] in ([], ["-h"], ["--help"]):
    print(__doc__)
    sys.exit(0)

import numpy as np  # noqa: E402

# The tiny shape: the release's structure (three layers, mHC of four, grouped output, a block of six) at small widths.
# Widths, vocabulary, rope and clamp are those of the tiny DeepSeek-V4.1 target (tests/data/deepseek41_dspark.json), whose ratio-1 layers 5 to 7 lend the context.
TINY = {"hidden": 64, "heads": 4, "head_dim": 32, "rope_dim": 8, "q_lora": 32, "o_groups": 2, "o_lora": 16, "experts": 6, "experts_used": 3,
        "expert_ff": 32, "shared": 1, "layers": 3, "hc": 4, "sinkhorn": 20, "hc_eps": 1e-6, "eps": 1e-20, "vocab": 128, "block": 6, "mask_id": 127,
        "target_layers": [5, 6, 7], "ctx": 40, "picks": 16, "max_block_slots": 16, "correction_ff": 64, "rope_theta": 160000.0, "rope_factor": 16.0,
        "rope_orig": 65536, "beta_fast": 32.0, "beta_slow": 1.0, "route_scale": 1.5, "swiglu_limit": 1.0, "seed": 20261126}


def rms_norm(x, w, eps):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w


def silu(x):
    return x / (1.0 + np.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def inv_freq(a):
    """DeepSeek-V4's YaRN frequencies over the rope channels."""
    d, theta = a["rope_dim"], a["rope_theta"]
    base = theta ** (np.arange(0, d, 2, dtype=np.float64) / d)
    extra, inter = 1.0 / base, 1.0 / (a["rope_factor"] * base)

    def turning(rot):
        return d * math.log(a["rope_orig"] / (rot * 2 * math.pi)) / (2 * math.log(theta))
    low, high = max(math.floor(turning(a["beta_fast"])), 0), min(math.ceil(turning(a["beta_slow"])), d - 1)
    if low == high:
        high += 0.001
    keep = 1 - np.clip((np.arange(d // 2) - low) / (high - low), 0, 1)
    return inter * (1 - keep) + extra * keep


def rope(x, pos, a, inverse=False):
    """Rotate the last rope_dim channels of x [..., rows, (heads,) D] in adjacent pairs at positions pos [rows]."""
    ang = np.asarray(pos, dtype=np.float64)[:, None] * inv_freq(a)
    cos, sin = np.cos(ang), np.sin(ang)
    if inverse:
        sin = -sin
    if x.ndim == 3:
        cos, sin = cos[:, None, :], sin[:, None, :]
    d = a["rope_dim"]
    re, im = x[..., -d::2], x[..., -d + 1::2]
    out = x.copy()
    out[..., -d::2] = re * cos - im * sin
    out[..., -d + 1::2] = re * sin + im * cos
    return out


def sparse_read(q, latent, idx, sink, scale):
    """Per-head-sink attention of q [rows, H, D] over latent rows idx [rows, K] (-1 unused); a row with no key reads zero."""
    out = np.zeros_like(q)
    for r in range(q.shape[0]):
        keys = latent[idx[r][idx[r] >= 0]]
        if len(keys) == 0:
            continue
        logits = q[r] @ keys.T * scale
        top = np.maximum(logits.max(-1, keepdims=True), sink[:, None])
        p = np.exp(logits - top)
        p /= p.sum(-1, keepdims=True) + np.exp(sink[:, None] - top)
        out[r] = p @ keys
    return out


def grouped_output(read, pos, w, pre, a):
    """Inverse rope, then each output group's low-rank projection, then the shared up-projection."""
    rows, g = read.shape[0], a["o_groups"]
    x = rope(read, pos, a, inverse=True).reshape(rows, g, -1)
    wa = w[pre + "attn_output_a.weight"].reshape(g, a["o_lora"], -1)
    return np.einsum("lgd,grd->lgr", x, wa).reshape(rows, -1) @ w[pre + "attn_output_b.weight"].T


def query(x, pos, w, pre, a):
    q = rms_norm(x @ w[pre + "attn_q_a.weight"].T, w[pre + "attn_q_a_norm.weight"], a["eps"]) @ w[pre + "attn_q_b.weight"].T
    return rope(q.reshape(x.shape[0], a["heads"], a["head_dim"]), pos, a)


def attention(x, pos, ctx, idx, w, pre, a):
    """A draft layer's attention: latent MQA of the block over its context picks and every block row, non-causal."""
    block = rope(rms_norm(x @ w[pre + "attn_kv.weight"].T, w[pre + "attn_kv_a_norm.weight"], a["eps"]), pos, a)
    read = sparse_read(query(x, pos, w, pre, a), np.concatenate([ctx, block]), idx, w[pre + "attn_sinks.weight"], a["head_dim"] ** -0.5)
    return grouped_output(read, pos, w, pre, a)


def swiglu(x, gate, up, down, limit):
    g, u = x @ gate.T, x @ up.T
    return (silu(np.minimum(g, limit)) * np.clip(u, -limit, limit)) @ down.T


def moe(x, w, pre, a):
    """sqrt(softplus) router, bias for the choice only, renormalised weights times the route scale, clamped experts and a clamped shared expert."""
    scores = np.sqrt(np.logaddexp(0.0, x @ w[pre + "ffn_gate_inp.weight"].T))
    choice = scores + w[pre + "exp_probs_b.bias"]
    out = swiglu(x, w[pre + "ffn_gate_shexp.weight"], w[pre + "ffn_up_shexp.weight"], w[pre + "ffn_down_shexp.weight"], a["swiglu_limit"])
    picks = []
    for r in range(x.shape[0]):
        top = np.argsort(-choice[r], kind="stable")[:a["experts_used"]]
        weight = scores[r, top] / (scores[r, top].sum() + 1e-20) * a["route_scale"]
        for e, s in zip(top, weight):
            out[r] += s * swiglu(x[r:r + 1], w[pre + "ffn_gate_exps.weight"][e], w[pre + "ffn_up_exps.weight"][e], w[pre + "ffn_down_exps.weight"][e], a["swiglu_limit"])[0]
        picks.append(top)
    return out, np.array(picks)


def hc_mixes(h, kind, w, pre, a):
    """(pre, post, comb) of the copies h [rows, hc, D] for one sublayer, with sinkhorn's column-first normalisation."""
    n = a["hc"]
    flat = h.reshape(h.shape[0], -1)
    m = flat @ w[pre + "hc_%s_fn.weight" % kind].T / np.sqrt(np.mean(flat * flat, -1, keepdims=True) + a["eps"])
    s, b, eps = w[pre + "hc_%s_scale.weight" % kind], w[pre + "hc_%s_base.weight" % kind], a["hc_eps"]
    p = sigmoid(m[:, :n] * s[0] + b[:n]) + eps
    q = 2 * sigmoid(m[:, n:2 * n] * s[1] + b[n:2 * n])
    c = (m[:, 2 * n:] * s[2] + b[2 * n:]).reshape(-1, n, n)
    c = np.exp(c - c.max(-1, keepdims=True))
    c = c / c.sum(-1, keepdims=True) + eps
    c = c / (c.sum(-2, keepdims=True) + eps)
    for _ in range(a["sinkhorn"] - 1):
        c = c / (c.sum(-1, keepdims=True) + eps)
        c = c / (c.sum(-2, keepdims=True) + eps)
    return p, q, c


def hc_pre(h, p):
    return np.einsum("rk,rkd->rd", p, h)


def hc_post(x, h, q, c):
    return q[:, :, None] * x[:, None, :] + np.einsum("rjk,rjd->rkd", c, h)


def draft(w, inp, a, trace):
    """The block's final-normed hidden rows [block, D]; trace collects the intermediates."""
    pos, emb = inp["positions"], w["token_embd.weight"]
    x = emb[np.r_[inp["anchor"], np.full(a["block"] - 1, a["mask_id"])]]
    fused = np.concatenate([rms_norm(emb[inp["anchor"]], w["fusion.enorm.weight"], a["eps"]),
                            rms_norm(inp["prev_hidden"], w["fusion.hnorm.weight"], a["eps"])]) @ w["fusion.eh_proj.weight"].T
    x[0] += w["fusion.gate.weight"] * fused
    trace["block_input"] = x.copy()
    copies, collapsed = np.repeat(x[:, None, :], a["hc"], axis=1), x
    for i in range(a["layers"]):
        pre = "blk.%d." % i
        pa, qa, ca = hc_mixes(copies, "attn", w, pre, a)
        out = attention(rms_norm(collapsed, w[pre + "attn_norm.weight"], a["eps"]), pos, inp["ctx_latent"][i], inp["ctx_index"][i], w, pre, a)
        trace["attn_out.%d" % i] = out
        hidden = hc_post(out, copies, qa, ca)
        pf, qf, cf = hc_mixes(hidden, "ffn", w, pre, a)
        out, trace["moe_picks.%d" % i] = moe(rms_norm(hc_pre(hidden, pa), w[pre + "ffn_norm.weight"], a["eps"]), w, pre, a)
        trace["ffn_out.%d" % i] = out
        copies = hc_post(out, hidden, qf, cf)
        collapsed = hc_pre(copies, pf)
        trace["collapsed.%d" % i] = collapsed
    return rms_norm(collapsed, w["output_norm.weight"], a["eps"])


def correct(h, prev, w, inp, a, trace):
    """Corrected hidden rows h' [block, D] given each slot's previous token; slot k's block keys read slots 0..k only."""
    pos, c, sc = inp["positions"], "corr.", a["head_dim"] ** -0.5
    hn = rms_norm(h, w[c + "hidden_norm.weight"], a["eps"])
    en = rms_norm(w["token_embd.weight"][prev], w[c + "embed_norm.weight"], a["eps"])
    x = np.concatenate([hn, en], -1)
    delta = (silu(x @ w[c + "ffn_gate.weight"].T) * (x @ w[c + "ffn_up.weight"].T)) @ w[c + "ffn_down.weight"].T
    q = rope((rms_norm(x @ w[c + "attn_q_a.weight"].T, w[c + "attn_q_a_norm.weight"], a["eps"]) @ w[c + "attn_q_b.weight"].T).reshape(len(h), a["heads"], -1), pos, a)
    ctx = inp["ctx_latent"][-1]
    idx = np.where(inp["ctx_index"][-1] < len(ctx), inp["ctx_index"][-1], -1)
    read = sparse_read(q, ctx, idx, w[c + "attn_sinks.weight"], sc)
    keys = rope(rms_norm(x @ w[c + "attn_kv.weight"].T, w[c + "attn_kv_a_norm.weight"], a["eps"]), pos, a)
    gate = np.tanh(x @ w[c + "block_gate.weight"].T + w[c + "block_gate_bias.weight"][:, :len(h)].T)
    for k in range(len(h)):
        logits = np.einsum("hd,jd->hj", q[k], keys[:k + 1]) * sc + w[c + "block_rel_bias.weight"][:, k - np.arange(k + 1)]
        top = np.maximum(logits.max(-1, keepdims=True), w[c + "block_sinks.weight"][:, None])
        p = np.exp(logits - top)
        p /= p.sum(-1, keepdims=True) + np.exp(w[c + "block_sinks.weight"][:, None] - top)
        read[k] += gate[k][:, None] * (p @ keys[:k + 1])
    trace["corr_read"] = read
    return h + delta + grouped_output(read, pos, w, c, a)


def forward(w, inp, a):
    """Drafts of one block: slot 0 holds the anchor, slot k > 0 predicts the token k after it, and the correction runs serially over the drafts."""
    trace = {}
    h = draft(w, inp, a, trace)
    trace["draft_hidden"] = h
    prev = np.full(a["block"], inp["anchor"])
    logits = np.zeros((a["block"], a["vocab"]))
    for k in range(a["block"]):
        if k >= 2:
            prev[k] = int(np.argmax(logits[k - 1]))
        logits[k] = (correct(h[:k + 1], prev[:k + 1], w, dict(inp, positions=inp["positions"][:k + 1]), a, {})[k]) @ w["output.weight"].T
    trace["corrected"] = correct(h, prev, w, inp, a, trace)
    trace["prev_tokens"] = prev
    trace["logits"] = logits
    trace["drafts"] = np.argmax(logits[1:], -1)
    return trace


def shapes(a):
    d, h, hd, n, ff, e = a["hidden"], a["heads"], a["head_dim"], a["hc"], a["expert_ff"], a["experts"]
    attn = {"attn_q_a.weight": (a["q_lora"], d), "attn_q_a_norm.weight": (a["q_lora"],), "attn_q_b.weight": (h * hd, a["q_lora"]),
            "attn_kv.weight": (hd, d), "attn_kv_a_norm.weight": (hd,), "attn_sinks.weight": (h,),
            "attn_output_a.weight": (a["o_groups"] * a["o_lora"], h * hd // a["o_groups"]), "attn_output_b.weight": (d, a["o_groups"] * a["o_lora"])}
    out = {"token_embd.weight": (a["vocab"], d), "output.weight": (a["vocab"], d), "output_norm.weight": (d,), "fusion.eh_proj.weight": (d, 2 * d),
           "fusion.enorm.weight": (d,), "fusion.hnorm.weight": (d,), "fusion.gate.weight": (d,)}
    for i in range(a["layers"]):
        p = "blk.%d." % i
        out.update({p + k: v for k, v in attn.items()})
        out.update({p + "attn_norm.weight": (d,), p + "ffn_norm.weight": (d,), p + "ffn_gate_inp.weight": (e, d), p + "exp_probs_b.bias": (e,),
                    p + "ffn_gate_exps.weight": (e, ff, d), p + "ffn_up_exps.weight": (e, ff, d), p + "ffn_down_exps.weight": (e, d, ff),
                    p + "ffn_gate_shexp.weight": (ff * a["shared"], d), p + "ffn_up_shexp.weight": (ff * a["shared"], d),
                    p + "ffn_down_shexp.weight": (d, ff * a["shared"])})
        for kind in ("attn", "ffn"):
            out.update({p + "hc_%s_fn.weight" % kind: ((2 + n) * n, n * d), p + "hc_%s_base.weight" % kind: ((2 + n) * n,), p + "hc_%s_scale.weight" % kind: (3,)})
    c, cf = "corr.", a["correction_ff"]
    out.update({c + k: v for k, v in attn.items() if k != "attn_q_a.weight" and k != "attn_kv.weight"})
    out.update({c + "attn_q_a.weight": (a["q_lora"], 2 * d), c + "attn_kv.weight": (hd, 2 * d), c + "hidden_norm.weight": (d,), c + "embed_norm.weight": (d,),
                c + "ffn_gate.weight": (cf, 2 * d), c + "ffn_up.weight": (cf, 2 * d), c + "ffn_down.weight": (d, cf),
                c + "block_gate.weight": (h, 2 * d), c + "block_gate_bias.weight": (h, a["max_block_slots"]),
                c + "block_rel_bias.weight": (h, a["max_block_slots"]), c + "block_sinks.weight": (h,)})
    return out


def tiny_state(a):
    """Seeded weights: norms near one, mHC scales and sinks of order one, projections at fan-in scale, every gate open."""
    rng = np.random.default_rng(a["seed"])
    w = {}
    for name, shape in sorted(shapes(a).items()):
        if "norm" in name:
            v = 1.0 + 0.1 * rng.standard_normal(shape)
        elif len(shape) == 1 or "bias" in name or name.endswith("scale.weight"):
            v = 0.5 * rng.standard_normal(shape)
        else:
            v = rng.standard_normal(shape) / math.sqrt(shape[-1])
        w[name] = v.astype(np.float32).astype(np.float64)
    return w


def tiny_inputs(a, case=0):
    """One block at anchor position 100: per layer a target-roped context latent and each slot's picks, then the six block rows; the last layer's slot 3 picks nothing."""
    rng = np.random.default_rng(a["seed"] + 1 + case)
    start = 100
    lat, idx = [], []
    for i in range(a["layers"]):
        lat.append(rope(rng.standard_normal((a["ctx"], a["head_dim"])), np.arange(start - a["ctx"], start), a).astype(np.float32).astype(np.float64))
        picks = np.stack([rng.permutation(a["ctx"])[:a["picks"]] for _ in range(a["block"])])
        picks[:, -2:] = -1
        if i == a["layers"] - 1:
            picks[3] = -1
        idx.append(np.concatenate([picks, np.tile(a["ctx"] + np.arange(a["block"]), (a["block"], 1))], 1).astype(np.int32))
    return {"anchor": 7, "positions": np.arange(start, start + a["block"]), "prev_hidden": rng.standard_normal(a["hidden"]).astype(np.float32).astype(np.float64),
            "ctx_latent": lat, "ctx_index": idx}


def filled(salt, n, shift):
    """n exact F32 values in [-1000, 1000] / 2^shift from an integer hash, which tests/arch_deepseek4.cpp (filled) mirrors."""
    k = np.arange(n, dtype=np.uint64) + np.uint64(salt * 1000003)
    return ((k * np.uint64(2654435761)) % np.uint64(2001)).astype(np.float64) / 2.0 ** shift - 1000.0 / 2.0 ** shift


def paired_inputs(a, case):
    """A block as the runtime reads it after 128 target tokens: anchor at 128, each stage's window rows of its target layer at positions 0 to 127 and the 128 rows of the one pool, and one picks row over every slot.
    Every value comes from filled() by the case's salt, so the native check writes the same state; a pool row is a latent then an index key, which the drafter does not read."""
    W, P, D, salt, shift = 128, 128, a["head_dim"], case["salt"], case["shift"]
    pool = filled(salt + 3, P * 2 * D, shift).reshape(P, 2 * D)[:, :D]
    lat, idx = [], []
    for i in range(a["layers"]):
        lat.append(np.concatenate([filled(salt + i, W * D, shift).reshape(W, D), pool]))
        picks = np.array(case["picks"][i])
        row = np.concatenate([np.arange(W), np.where(picks >= 0, W + picks, -1), W + P + np.arange(a["block"])])
        idx.append(np.tile(row, (a["block"], 1)).astype(np.int32))
    return {"anchor": case["anchor"], "positions": np.arange(W, W + a["block"]), "prev_hidden": filled(salt + 4, a["hidden"], shift), "ctx_latent": lat, "ctx_index": idx}


def paired_cases(a):
    """The runtime-shaped cases: their salts, anchors, value scales and picks, the last stage short of four picks and two stages of two."""
    out = []
    for n, (salt, anchor, shift) in enumerate(PAIRED):
        picks = [[(j * 37 + 11 * i + 5 * n) % 128 if j < a["picks"] - (4 if i == a["layers"] - 1 else 2) else -1 for j in range(a["picks"])] for i in range(a["layers"])]
        out.append({"salt": salt, "anchor": anchor, "shift": shift, "picks": picks})
    return out


# Salts, anchors and scales: the first case's wide context gives every draft a margin above 0.5, the second's narrow one keeps the sinks and the block rows in each softmax.
PAIRED = ((25, 120, 7), (11, 111, 10))


def write_golden(out_dir, a=TINY):
    os.makedirs(out_dir, exist_ok=True)
    w, inp = tiny_state(a), tiny_inputs(a)
    trace = forward(w, inp, a)
    np.savez(os.path.join(out_dir, "weights.npz"), **{k: v.astype(np.float32) for k, v in w.items()})
    np.savez(os.path.join(out_dir, "inputs.npz"), anchor=np.int32(inp["anchor"]), positions=inp["positions"].astype(np.int32),
             prev_hidden=inp["prev_hidden"].astype(np.float32), **{"ctx_latent.%d" % i: v.astype(np.float32) for i, v in enumerate(inp["ctx_latent"])},
             **{"ctx_index.%d" % i: v for i, v in enumerate(inp["ctx_index"])})
    for k, v in trace.items():
        np.save(os.path.join(out_dir, k + ".npy"), v.astype(np.float32) if v.dtype.kind == "f" else v.astype(np.int32))
    with open(os.path.join(out_dir, "args.json"), "w") as f:
        json.dump(a, f, indent=1)
    return trace


def margin(logits):
    """The least gap between a draft row's top two logits."""
    return float(margins(logits).min())


def trunk_weights(state):
    """The tiny trunk's embedding and head (tests/fixture_deepseek4.py state, (shape, flat) by name) under the drafter's names, as F32 values."""
    return {mine: np.array(state[theirs][1], dtype=np.float32).astype(np.float64).reshape(state[theirs][0]) for mine, theirs in (("token_embd.weight", "embed.weight"), ("output.weight", "head.weight"))}


def margins(logits):
    """Each draft row's gap between its top two logits."""
    top = np.sort(logits[1:], -1)
    return top[:, -1] - top[:, -2]


def paired(w, trunk, a=TINY):
    """The runtime-shaped cases with their drafts, previous tokens, per-draft margins and logits, through the trunk's embedding and head."""
    out = []
    for case in paired_cases(a):
        t = forward(dict(w, **trunk), paired_inputs(a, case), a)
        out.append(dict(case, drafts=t["drafts"].tolist(), prev_tokens=t["prev_tokens"].tolist(), margins=[round(float(m), 6) for m in margins(t["logits"])],
                        logits=[[float("%.9g" % v) for v in row] for row in t["logits"]]))
    return out


def document(cases, trunk, a=TINY):
    """The committed golden (tests/data/deepseek41_dfly2.json): the args; for each input case its drafts, the previous tokens the correction read, the least draft margin and the logits; and the runtime-shaped cases."""
    w, out = tiny_state(a), []
    for case in cases:
        t = forward(w, tiny_inputs(a, case), a)
        out.append({"case": case, "drafts": t["drafts"].tolist(), "prev_tokens": t["prev_tokens"].tolist(), "least_draft_margin": round(margin(t["logits"]), 4),
                    "logits": [[float("%.7g" % v) for v in row] for row in t["logits"]]})
    return {"_comment": "Written by tools/dfly2_reference.py --document; case 0 matched AngelSpec e02ea36's torch modules in float64 to 2.5e-6 with the same drafts. "
            "The paired cases are the runtime's layout (docs/SPECULATIVE.md, section 12), which llmx-arch-deepseek4-test --dfly2 writes into a saved sequence.",
            "args": a, "cases": out, "paired": paired(w, trunk, a)}


if __name__ == "__main__":
    if sys.argv[1] == "--document":
        tests = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests")
        sys.path.insert(0, tests)
        import fixture_deepseek4
        with open(os.path.join(tests, "data", "deepseek41_reference.json"), encoding="utf-8") as f:
            _, state, _ = fixture_deepseek4.fixture_from_document(json.load(f))
        golden = document([int(c) for c in sys.argv[3:]], trunk_weights(state))
        with open(sys.argv[2], "w", encoding="utf-8", newline="\n") as f:
            json.dump(golden, f, indent=None, separators=(",", ":"))
            f.write("\n")
    else:
        t = write_golden(sys.argv[1])
        print("drafts", t["drafts"].tolist(), "prev", t["prev_tokens"].tolist(), "least margin %.4g" % margin(t["logits"]))
