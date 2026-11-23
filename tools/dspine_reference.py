"""A DSpine drafter's block in F32 numpy, ported from the official dspine_mlx code (model.py, bridge.py, propose.py RefinementTail.greedy), reading a dflash sidecar GGUF.

Usage: python3 dspine_reference.py SIDECAR.gguf TARGET.gguf IN.npz OUT.npz
IN.npz holds taps [N, T, E] f32 (the residual entering each of the sidecar's target_layers at positions 0..N-1) and anchor (the token at position N).
OUT.npz gets ctx_k and ctx_v [L, N, KV, D] (each layer's context K after its norm and rope, and V), cand [B-1, K] (ascending ids), score [B-1, K, K], drafts [B-1], logits_top [B-1, K] and final [B, E].
The norms are read as the GGUF stores them, so a converter that adds one to a norm is checked separately against the original weights (docs/TESTS.md).
"""
import struct
import sys

np = None


def load_numpy():
    global np
    if np is None:
        import numpy
        np = numpy


def bf16(raw):
    return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


class Gguf:
    def __init__(self, path):
        load_numpy()
        self.f = f = open(path, "rb")
        rd = lambda fmt: struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]
        rs = lambda: f.read(rd("Q")).decode()
        T = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

        def rv(t):
            if t == 8:
                return rs()
            if t == 9:
                et, n = rd("I"), rd("Q")
                return [rv(et) for _ in range(n)]
            return rd(T[t])

        assert f.read(4) == b"GGUF"
        rd("I")
        nt, nk = rd("Q"), rd("Q")
        self.kv = {}
        for _ in range(nk):
            k = rs()
            self.kv[k] = rv(rd("I"))
        self.t = {}
        for _ in range(nt):
            n = rs()
            ne = [rd("Q") for _ in range(rd("I"))]
            self.t[n] = (ne, rd("I"), rd("Q"))
        al = self.kv.get("general.alignment", 32)
        self.base = (f.tell() + al - 1) // al * al

    def rows(self, name, lo=None, hi=None):
        ne, ty, off = self.t[name]
        n0 = ne[0]
        nrows = int(np.prod(ne[1:])) if len(ne) > 1 else 1
        lo, hi = lo or 0, nrows if hi is None else hi
        if not 0 <= lo <= hi <= nrows:
            raise SystemExit("%s: rows %d to %d of %d" % (name, lo, hi, nrows))
        bs, es = {0: (4, 1), 1: (2, 1), 30: (2, 1), 8: (34, 32)}.get(ty, (0, 0))
        if not bs:
            raise SystemExit("%s: tensor type %d" % (name, ty))
        rb = n0 // es * bs
        self.f.seek(self.base + off + lo * rb)
        raw = self.f.read((hi - lo) * rb)
        if len(raw) != (hi - lo) * rb:
            raise SystemExit("%s: the file ends inside the tensor" % name)
        if ty == 0:
            return np.frombuffer(raw, np.float32).reshape(hi - lo, n0).copy()
        if ty == 1:
            return np.frombuffer(raw, np.float16).astype(np.float32).reshape(hi - lo, n0)
        if ty == 30:
            return bf16(raw).reshape(hi - lo, n0).copy()
        b = np.frombuffer(raw, np.uint8).reshape(-1, 34)
        d = b[:, :2].copy().view(np.float16).astype(np.float32)
        q = b[:, 2:].view(np.int8).astype(np.float32)
        return (d * q).reshape(hi - lo, n0)

    def vector(self, name):
        return self.rows(name).reshape(-1)


def rmsnorm(x, w, eps):
    return x / np.sqrt(np.mean(x * x, -1, keepdims=True) + eps) * w


def rope(x, pos, theta):
    # NeoX halves over the whole head, x [n, heads, D]
    D = x.shape[-1]
    inv = theta ** (-np.arange(0, D, 2, dtype=np.float64) / D)
    ang = pos[:, None].astype(np.float64) * inv[None, :]
    cos, sin = np.cos(ang).astype(np.float32)[:, None, :], np.sin(ang).astype(np.float32)[:, None, :]
    a, b = x[..., : D // 2], x[..., D // 2 :]
    return np.concatenate([a * cos - b * sin, a * sin + b * cos], -1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def block(side, tgt, taps, anchor):
    """The drafter's block after context rows `taps` [N, T, E] and the token `anchor` at position N, as a dict of the arrays OUT.npz holds."""
    load_numpy()
    m = lambda k: side.kv["dflash." + k]
    L, B, W, K = m("block_count"), m("block_size"), m("attention.sliding_window"), m("selector_top_k")
    H, KV, D, THETA = m("attention.head_count"), m("attention.head_count_kv"), m("attention.key_length"), float(m("rope.freq_base"))
    EPS, MASK = float(m("attention.layer_norm_rms_epsilon")), side.kv["tokenizer.ggml.mask_token_id"]
    w = lambda name: side.rows(name)
    v = lambda name: side.vector(name)
    N, T, E = taps.shape
    C = side.t["dspine_codebook.weight"][0][0]
    if side.t["dspine_codebook.weight"][0][1:] != tgt.t["token_embd.weight"][0][1:]:
        raise SystemExit("dspine_codebook.weight: a row a token of the target's vocabulary")
    R = side.t["dspine_gate.bias"][0][0]
    sqc = np.float32(np.sqrt(C))
    ids = np.array([anchor] + [MASK] * (B - 1))
    bpos, kpos_ctx = np.arange(N, N + B), np.arange(N)
    ctx = rmsnorm(taps.reshape(N, -1).astype(np.float32) @ w("fc.weight").T, v("enc.output_norm.weight"), EPS)
    x = np.concatenate([tgt.rows("token_embd.weight", int(i), int(i) + 1) for i in ids])
    packed_w = np.concatenate([w("dspine_gate.weight"), w("dspine_pred_gate.weight"), w("dspine_prev.weight")])
    codebook = lambda t: side.rows("dspine_codebook.weight", int(t), int(t) + 1)[0]
    gate_b = v("dspine_gate.bias")
    residual, ctx_k, ctx_v = x, [], []
    x_in = rmsnorm(x, v("blk.0.attn_norm.weight"), EPS)
    for l in range(L):
        p = "blk.%d." % l
        q = rmsnorm((x_in @ w(p + "attn_q.weight").T).reshape(B, H, D), v(p + "attn_q_norm.weight"), EPS)
        kb = rmsnorm((x_in @ w(p + "attn_k.weight").T).reshape(B, KV, D), v(p + "attn_k_norm.weight"), EPS)
        vb = (x_in @ w(p + "attn_v.weight").T).reshape(B, KV, D)
        kc = rmsnorm((ctx @ w(p + "attn_k.weight").T).reshape(N, KV, D), v(p + "attn_k_norm.weight"), EPS)
        vc = (ctx @ w(p + "attn_v.weight").T).reshape(N, KV, D)
        q, kb, kc = rope(q, bpos, THETA), rope(kb, bpos, THETA), rope(kc, kpos_ctx, THETA)
        ctx_k.append(kc)
        ctx_v.append(vc)
        keys, vals = np.concatenate([kc, kb]), np.concatenate([vc, vb])
        kpos = np.concatenate([kpos_ctx, bpos])
        mask = (kpos[None, :] <= bpos[:, None]) & (kpos[None, :] > bpos[:, None] - W)
        out = np.empty((B, H, D), np.float32)
        for h in range(H):
            s = (q[:, h] @ keys[:, h // (H // KV)].T) / np.sqrt(D)
            s = np.where(mask, s, -np.inf)
            s = np.exp(s - s.max(-1, keepdims=True))
            out[:, h] = (s / s.sum(-1, keepdims=True)) @ vals[:, h // (H // KV)]
        hcur = residual + out.reshape(B, -1) @ w(p + "attn_output.weight").T
        mm = rmsnorm(hcur, v(p + "ffn_norm.weight"), EPS)
        g = mm @ w(p + "ffn_gate.weight").T
        raw = hcur + (g * sigmoid(g) * (mm @ w(p + "ffn_up.weight").T)) @ w(p + "ffn_down.weight").T
        # code bridge
        rms = np.sqrt(np.maximum(np.mean(raw * raw, -1), np.float32(1.1754943508222875e-38)))
        proj = (raw / (rms[:, None] + EPS)) @ w(p + "dspine_read.weight").T
        codes = sqc * proj / (np.sqrt(np.sum(proj * proj, -1, keepdims=True)) + EPS)
        codes[0] = sqc * codebook(anchor)
        pk = codes @ packed_w.T
        qg, sg, vg = pk[:, :R], pk[:, R : 2 * R], pk[:, 2 * R :]
        msg = np.zeros((B, R), np.float32)
        msg[1:] = sigmoid(qg[1:] + sg[:-1] + gate_b) * vg[:-1]
        delta = msg @ w(p + "dspine_write.weight").T
        delta[0] = 0
        outl = raw + rms[:, None] * delta
        if l == L - 1:
            stash = (raw, rms, qg, w(p + "dspine_write.weight"))
            final = rmsnorm(outl, v("output_norm.weight"), EPS)
        else:
            x_in = rmsnorm(outl, v("blk.%d.attn_norm.weight" % (l + 1)), EPS)
        residual = outl
    # candidates: the target's head over rows 1..B-1, top K, ascending ids
    rows = final[1:]
    head_name = "output.weight" if "output.weight" in tgt.t else "token_embd.weight"
    V = tgt.t[head_name][0][1]
    logits = np.concatenate([rows @ tgt.rows(head_name, lo, min(lo + 16384, V)).T for lo in range(0, V, 16384)], 1)
    cand = np.sort(np.argsort(-logits, axis=1, kind="stable")[:, :K], axis=1)
    raw, rms, qg, write = stash[0][1:], stash[1][1:], stash[2][1:], stash[3]
    n = B - 1
    head = np.stack([np.concatenate([tgt.rows(head_name, int(c), int(c) + 1) for c in cand[t]]) for t in range(n)])
    sid = np.concatenate([np.full((1, K), anchor), cand[:-1]])
    e = sqc * np.stack([[codebook(i) for i in r] for r in sid])
    gate = sigmoid(qg[:, None, :] + e @ w("dspine_pred_gate.weight").T + gate_b)
    msg = gate * (e @ w("dspine_prev.weight").T)
    refined = raw[:, None, :] + rms[:, None, None] * (msg @ write.T)
    score = np.einsum("tje,tke->tjk", refined * v("output_norm.weight"), head)
    idx = score.argmax(-1)
    drafts, j = [], idx[0, 0]
    for t in range(n):
        drafts.append(int(cand[t, j]))
        if t + 1 < n:
            j = idx[t + 1, j]
    return dict(ctx_k=np.stack(ctx_k), ctx_v=np.stack(ctx_v), cand=cand, score=score, drafts=np.array(drafts),
                logits_top=np.take_along_axis(logits, cand, 1), final=final)


def main():
    if len(sys.argv) == 2 and sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    load_numpy()
    inp = np.load(sys.argv[3])
    out = block(Gguf(sys.argv[1]), Gguf(sys.argv[2]), inp["taps"], int(inp["anchor"]))
    np.savez(sys.argv[4], **out)
    print("drafts", list(out["drafts"]))


if __name__ == "__main__":
    main()
