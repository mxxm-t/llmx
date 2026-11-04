"""F32 CPU forms of the kernels DeepSeek's inference/model.py imports from its kernel.py at 1bae0e10.

Each follows its TileLang kernel's arithmetic in F32: the model formats' quantize-dequantize (FP8 E4M3 with an E8M0 scale per 32, FP4 E2M1 with an E8M0 scale per 32 or an E4M3 scale per 16) with the kernel's scale rules, sparse attention with its sink and empty-row convention, and the Sinkhorn split in the kernel's order.
Casts round to nearest, ties to even, as the CUDA conversions TileLang emits do.
The GEMMs that take quantized activations are not reached by an F32 model and refuse.
The attention rounds no activation: the kernel's BF16 q and kv inputs and its BF16 cast of the probabilities before PV (acc_s_cast) are left out, and a one-shot maximum over the row replaces its online softmax (the plan's section 4.5).
"""

if __name__ == "__main__":
    # A module tools/deepseek4_reference.py imports, not a command; this answers --help without torch.
    import argparse
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    raise SystemExit("deepseek4_kernel.py is imported by tools/deepseek4_reference.py and runs nothing itself")

import torch

FP8_MAX = 448.0
FP4_MAX = 6.0
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# Midpoints between neighbouring E2M1 magnitudes; a tie goes to the even code, so up at 0.75, 1.75 and 3.5 and down elsewhere.
_MIDPOINTS = ((0.25, False), (0.75, True), (1.25, False), (1.75, True), (2.5, False), (3.5, True), (5.0, False))


def _f32(value):
    return torch.tensor(value, dtype=torch.float32)


def ceil_log2(x):
    """kernel.py's fast_log2_ceil: the exponent field less 127, plus one when any mantissa bit is set."""
    bits = x.contiguous().view(torch.int32)
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & ((1 << 23) - 1)
    return exponent - 127 + (mantissa != 0).to(torch.int32)


def pow2(e):
    """kernel.py's fast_pow2: (e + 127) << 23 read as f32."""
    return ((e + 127) << 23).to(torch.int32).view(torch.float32)


def round_scale(amax, max_inv):
    return pow2(ceil_log2(amax * max_inv))


def to_e2m1(x):
    """Round f32 values already clamped to [-6, 6] to E2M1, ties to even, keeping the sign of a zero."""
    magnitude = x.abs()
    index = torch.zeros_like(magnitude, dtype=torch.int64)
    for midpoint, up in _MIDPOINTS:
        index += (magnitude >= midpoint if up else magnitude > midpoint).to(torch.int64)
    values = torch.tensor(_E2M1, dtype=torch.float32)[index]
    return torch.where(torch.signbit(x), -values, values)


def _blocks(x, block_size):
    n = x.size(-1)
    if n % block_size:
        raise ValueError("width %d is not whole blocks of %d" % (n, block_size))
    return x.float().reshape(-1, n // block_size, block_size)


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    """FP8 E4M3 quantize-dequantize in place with a power-of-two scale per block: amax at least 1e-4, s = 2^ceil(log2(amax * (1/448)))."""
    if not inplace or scale_fmt is None:
        raise NotImplementedError("act_quant: only the in-place power-of-two form is reached by the F32 model")
    xb = _blocks(x, block_size)
    amax = torch.maximum(xb.abs().amax(-1, keepdim=True), _f32(1e-4))
    s = round_scale(amax, _f32(1.0 / FP8_MAX))
    q = torch.clamp(xb / s, -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float() * s
    x.copy_(q.reshape(x.shape))
    return x


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    """FP4 E2M1 quantize-dequantize in place: an E8M0 scale 2^ceil(log2(amax * (1/6))) with amax at least 6 * 2^-126, or an E4M3 scale E4M3(amax / 6) with amax at least 6 * 2^-9."""
    if not inplace:
        raise NotImplementedError("fp4_act_quant: only the in-place form is reached by the model")
    xb = _blocks(x, block_size)
    amax = xb.abs().amax(-1, keepdim=True)
    if scale_dtype == torch.float8_e4m3fn:
        amax = torch.maximum(amax, _f32(6 * 2.0 ** -9))
        s = (amax / FP4_MAX).to(torch.float8_e4m3fn).float()
    elif scale_dtype == torch.float8_e8m0fnu:
        amax = torch.maximum(amax, _f32(6 * 2.0 ** -126))
        s = round_scale(amax, _f32(1.0 / FP4_MAX))
    else:
        raise ValueError("fp4_act_quant: unknown scale type %s" % scale_dtype)
    q = to_e2m1(torch.clamp(xb / s, -FP4_MAX, FP4_MAX)) * s
    x.copy_(q.reshape(x.shape))
    return x


def fp8_gemm(*_args, **_kwargs):
    raise NotImplementedError("fp8_gemm: the F32 model holds no FP8 linear")


def fp4_gemm(*_args, **_kwargs):
    raise NotImplementedError("fp4_gemm: the F32 model holds no FP4 linear")


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """Attention of each query row over the kv rows its indices name, -1 naming none; the max starts at -1e30 and the sink joins the denominator after it, so a row with no index gives zeros."""
    b, m, h, d = q.shape
    valid = topk_idxs != -1
    rows = torch.where(valid, topk_idxs, 0).long()
    gathered = torch.gather(kv.float().unsqueeze(1).expand(b, m, -1, d), 2, rows.unsqueeze(-1).expand(-1, -1, -1, d))
    gathered = torch.where(valid.unsqueeze(-1), gathered, 0.0)
    scores = torch.einsum("bmhd,bmtd->bmht", q.float(), gathered) * _f32(softmax_scale)
    scores = torch.where(valid.unsqueeze(2), scores, -torch.inf)
    peak = torch.maximum(scores.amax(-1, keepdim=True) if scores.size(-1) else torch.full((b, m, h, 1), -torch.inf), _f32(-1e30))
    p = torch.exp(scores - peak)
    denominator = p.sum(-1, keepdim=True) + torch.exp(attn_sink.float().view(1, 1, h, 1) - peak)
    return (torch.einsum("bmht,bmtd->bmhd", p, gathered) / denominator).to(q.dtype)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """pre, post and the doubly stochastic comb from each row's mixes, in kernel.py's order."""
    hc = hc_mult
    m = mixes.float()
    eps = _f32(eps)
    pre = torch.sigmoid(m[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(m[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = (m[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]).unflatten(-1, (hc, hc))
    comb = torch.exp(comb - comb.amax(-1, keepdim=True))
    comb = comb / comb.sum(-1, keepdim=True) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


NAMES = ("act_quant", "fp4_act_quant", "fp4_gemm", "fp8_gemm", "hc_split_sinkhorn", "sparse_attn")
