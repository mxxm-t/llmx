"""The tiny deepseek41 fixture: its weights from a seed, and its GGUF in both layouts, torch-free.

The state maps DeepSeek's parameter names (inference/model.py at 1bae0e10) to `(shape, values)` flattened row-major.
tools/deepseek4_reference.py loads the same state into DeepSeek's own model, so the suite rebuilds the reference's weights from the golden's seed alone.
"""

import hashlib
import json
import math
import os
import struct
import tempfile
import zlib
from pathlib import Path

try:
    from .spec_decode import ARRAY, F32, MXFP4, STRING, write_gguf
except ImportError:
    from spec_decode import ARRAY, F32, MXFP4, STRING, write_gguf


GGUF_UINT32 = 4
GGUF_INT32 = 5
GGUF_FLOAT32 = 6
GGUF_BOOL = 7
GGUF_UINT64 = 10
LAYOUTS = ("fork", "upstream")

# The reduced ModelArgs, every layer kind of V4.1 within 256 tokens (docs/ASSETS.md, The deepseek41 tiny reference).
TINY_ARGS = {
    "max_batch_size": 1, "max_seq_len": 256, "temperature": 0.0, "dtype": "bf16", "expert_dtype": None,
    "vocab_size": 128, "dim": 64, "moe_inter_dim": 32, "n_layers": 8, "n_mtp_layers": 0, "n_heads": 4,
    "n_routed_experts": 4, "n_shared_experts": 1, "n_activated_experts": 2, "score_func": "sqrtsoftplus",
    "gate_temp": 1.0, "norm_topk_prob": True, "route_scale": 1.5, "swiglu_limit": 1.0,
    "q_lora_rank": 32, "head_dim": 32, "rope_head_dim": 8, "norm_eps": 1e-20, "o_groups": 2, "o_lora_rank": 16,
    "window_size": 128, "compress_ratios": [0, 0, 2, 2, 2, 1, 1, 1],
    "kv_source_layers": [2, 4, 5], "index_source_layers": [2, 4, 5, 7],
    "compress_rope_theta": 160000.0, "original_seq_len": 65536, "rope_theta": 10000.0, "rope_factor": 16.0,
    "beta_fast": 32, "beta_slow": 1,
    "index_n_heads": 8, "index_head_dim": 32, "index_topk": 16,
    "candidate_source_layer": 5, "candidate_topk_blocks": 4, "candidate_block_size": 8,
    "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-6,
    "engram_layer_ids": [1, 4], "engram_max_ngram_size": 4, "engram_vocab_size": 40,
    "engram_n_heads": 2, "engram_head_dim": 32, "engram_pad_id": 2, "engram_compressed_vocab_size": 99092,
}

# The tiny DSpark drafter (docs/DEEPSEEK4.md, section 4.7): three window-only stages with wider experts of their own, taps on a ratio-2 Engram layer and either side of it, a block of 5 and a Markov head of rank 16.
DSPARK_ARGS = dict(TINY_ARGS, n_mtp_layers=3, compress_ratios=TINY_ARGS["compress_ratios"] + [0, 0, 0], dspark_block_size=5,
                   dspark_noise_token_id=127, dspark_target_layer_ids=[3, 4, 6], dspark_markov_rank=16, dspark_n_routed_experts=6,
                   dspark_n_activated_experts=3)

# E2M1 magnitudes by code; code c | 8 is the negative of code c.
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _mask64(x):
    return x & 0xFFFFFFFFFFFFFFFF


class _SplitMix64:
    def __init__(self, seed):
        self.state = _mask64(seed)

    def next(self):
        self.state = _mask64(self.state + 0x9E3779B97F4A7C15)
        z = self.state
        z = _mask64((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9)
        z = _mask64((z ^ (z >> 27)) * 0x94D049BB133111EB)
        return z ^ (z >> 31)


def _stream(seed, name):
    return _SplitMix64(seed * 1000003 + zlib.crc32(name.encode("utf-8")))


def _uniform(seed, name, count, amp, base=0.0):
    """count values base + amp * k / 2^23 for k uniform in [-2^23, 2^23), each exact in f32 for a power-of-two amp."""
    rng = _stream(seed, name)
    return [base + amp * (((rng.next() >> 40) - (1 << 23)) / float(1 << 23)) for _ in range(count)]


def _bf16_near_one(seed, name, count):
    """1 + j / 64 for j in [-16, 16]: BF16 values whose products are exact in f32, as engram_q and engram_k are BF16."""
    rng = _stream(seed, name)
    return [1.0 + ((rng.next() >> 32) % 33 - 16) / 64.0 for _ in range(count)]


def _linear_amp(fan_in):
    return 2.0 ** -math.ceil(math.log2(fan_in) / 2)


def layer_roles(args, layer):
    """Which optional parts a block has, as model.py builds them."""
    ratio = args["compress_ratios"][layer]
    kv_source = layer in args["kv_source_layers"]
    index_source = layer in args["index_source_layers"]
    return {"ratio": ratio, "kv_source": kv_source, "index_source": index_source,
            "owns_index_keys": index_source and kv_source, "engram": layer in args["engram_layer_ids"]}


def parameter_shapes(args, engram_rows):
    """Every parameter of DeepSeek's Transformer for these args, in model.py's names; engram_rows gives each Engram layer's table rows."""
    d, hc, h, hd = args["dim"], args["hc_mult"], args["n_heads"], args["head_dim"]
    q, g, orank = args["q_lora_rank"], args["o_groups"], args["o_lora_rank"]
    e, inter, ih, ihd = args["n_routed_experts"], args["moe_inter_dim"], args["index_n_heads"], args["index_head_dim"]
    mix = (2 + hc) * hc
    shapes = {"embed.weight": (args["vocab_size"], d), "norm.weight": (d,), "head.weight": (args["vocab_size"], d)}
    stages = args.get("n_mtp_layers", 0) if args.get("dspark_block_size") else 0
    for layer in range(args["n_layers"] + stages):
        roles = layer_roles(args, layer)
        p = "layers.%d." % layer if layer < args["n_layers"] else "mtp.%d." % (layer - args["n_layers"])
        e = args["n_routed_experts"] if layer < args["n_layers"] else args["dspark_n_routed_experts"]
        shapes.update({
            p + "attn.attn_sink": (h,), p + "attn.wq_a.weight": (q, d), p + "attn.q_norm.weight": (q,),
            p + "attn.wq_b.weight": (h * hd, q), p + "attn.wkv.weight": (hd, d), p + "attn.kv_norm.weight": (hd,),
            p + "attn.wo_a.weight": (g * orank, h * hd // g), p + "attn.wo_b.weight": (d, g * orank),
            p + "ffn.gate.weight": (e, d), p + "ffn.gate.bias": (e,),
            p + "attn_norm.weight": (d,), p + "ffn_norm.weight": (d,),
            p + "hc_attn_fn": (mix, hc * d), p + "hc_ffn_fn": (mix, hc * d), p + "hc_attn_base": (mix,),
            p + "hc_ffn_base": (mix,), p + "hc_attn_scale": (3,), p + "hc_ffn_scale": (3,),
        })
        if roles["kv_source"]:
            shapes[p + "attn.compressor.norm.weight"] = (hd,)
            shapes[p + "attn.compressor.wkv.weight"] = (hd, d)
            if roles["ratio"] > 1:
                shapes[p + "attn.compressor.wgate.weight"] = (hd, d)
        if roles["index_source"]:
            shapes[p + "attn.indexer.wq_b.weight"] = (ih * ihd, q)
            shapes[p + "attn.indexer.weights_proj.weight"] = (ih, d)
            if roles["owns_index_keys"]:
                shapes[p + "attn.indexer.wk.weight"] = (ihd, hd)
                shapes[p + "attn.indexer.k_norm.weight"] = (ihd,)
        for expert in ["experts.%d" % i for i in range(e)] + ["shared_experts"]:
            shapes[p + "ffn.%s.w1.weight" % expert] = (inter, d)
            shapes[p + "ffn.%s.w2.weight" % expert] = (d, inter)
            shapes[p + "ffn.%s.w3.weight" % expert] = (inter, d)
        if roles["engram"]:
            rows, ehd = engram_rows[args["engram_layer_ids"].index(layer)], args["engram_head_dim"]
            cols = (args["engram_max_ngram_size"] - 1) * args["engram_n_heads"] * ehd
            shapes[p + "engram.embed.weight"] = (rows, ehd)
            shapes[p + "engram.embed.scale"] = (rows, ehd // 32)
            shapes[p + "engram.wkv.weight"] = (d * (hc + 1), cols)
            shapes[p + "engram.q_weight"] = (hc, d)
            shapes[p + "engram.k_weight"] = (hc, d)
    if stages:
        # The stages share the trunk's embedding and head, which DeepSeek's state lists under every stage too.
        rank, last = args["dspark_markov_rank"], "mtp.%d." % (stages - 1)
        for stage in range(stages):
            shapes["mtp.%d.embed.weight" % stage] = shapes["embed.weight"]
            shapes["mtp.%d.head.weight" % stage] = shapes["head.weight"]
        shapes.update({"mtp.0.main_proj.weight": (d, d * len(args["dspark_target_layer_ids"])), "mtp.0.main_norm.weight": (d,),
                       last + "norm.weight": (d,), last + "markov_head.embed.weight": (args["vocab_size"], rank),
                       last + "markov_head.head.weight": (args["vocab_size"], rank), last + "confidence_head.proj.weight": (1, d + rank)})
    return shapes


def _engram_table(seed, name, rows, width, zero_rows):
    """E2M1 values (no -0) with one power-of-two scale per 32, as the checkpoint's FP8 rows hold them; zero_rows hold code 0."""
    rng = _stream(seed, name)
    values, scales = [], []
    zero = set(zero_rows)
    for row in range(rows):
        for _ in range(width // 32):
            scales.append(2.0 ** -(1 + (rng.next() >> 40) % 3))
            codes = [(rng.next() >> 40) % 15 for _ in range(32)]
            codes = [c + 1 if c >= 8 else c for c in codes]
            if row in zero:
                codes = [0] * 32
            values.extend(math.copysign(E2M1[c & 7], -1.0 if c & 8 else 1.0) for c in codes)
    return values, scales


def make_state(args, engram_rows, seed, zero_rows, drafter_seed=None):
    """The tiny state: name to (shape, values); an Engram table's weight holds E2M1 values and its scale the block scales, as model.py's FP8 table and E8M0 scale do.
    With `drafter_seed` the drafter's own parameters (mtp.*) are drawn from it, so a search over it leaves the trunk as it is."""
    state = {}
    trunk_seed = seed
    for name, shape in parameter_shapes(args, engram_rows).items():
        count = math.prod(shape)
        seed = drafter_seed if drafter_seed is not None and name.startswith("mtp.") else trunk_seed
        if name.endswith("engram.embed.weight"):
            layer = int(name.split(".")[1])
            values, scales = _engram_table(seed, name, shape[0], shape[1], zero_rows.get(layer, ()))
            state[name] = (shape, values)
            state[name[:-len("weight")] + "scale"] = ((shape[0], shape[1] // 32), scales)
        elif name.endswith("engram.embed.scale"):
            continue
        elif name.endswith(("engram.q_weight", "engram.k_weight")):
            state[name] = (shape, _bf16_near_one(seed, name, count))
        elif name.startswith("mtp.") and name.split(".", 2)[2] in ("embed.weight", "head.weight"):
            continue
        elif name.endswith("norm.weight"):
            state[name] = (shape, _uniform(seed, name, count, 2.0 ** -3, 1.0))
        elif name.endswith(("hc_attn_scale", "hc_ffn_scale")):
            state[name] = (shape, _uniform(seed, name, count, 2.0 ** -2, 1.0))
        elif name.endswith(("hc_attn_base", "hc_ffn_base")):
            state[name] = (shape, _uniform(seed, name, count, 2.0 ** -2))
        elif name.endswith("attn_sink") or name == "embed.weight":
            state[name] = (shape, _uniform(seed, name, count, 1.0))
        elif name.endswith("gate.bias"):
            state[name] = (shape, _uniform(seed, name, count, 2.0 ** -4))
        elif name == "head.weight":
            state[name] = (shape, _uniform(seed, name, count, 2.0 ** -3))
        else:
            state[name] = (shape, _uniform(seed, name, count, _linear_amp(shape[-1])))
    for name in parameter_shapes(args, engram_rows):
        if name.startswith("mtp.") and name.split(".", 2)[2] in ("embed.weight", "head.weight"):
            state[name] = state[name.split(".", 2)[2]]
    return state


def state_sha256(state):
    """SHA-256 over the names in order, each with its shape and its values as little-endian f32."""
    digest = hashlib.sha256()
    for name in sorted(state):
        shape, values = state[name]
        header = json.dumps([name, list(shape)], separators=(",", ":")).encode("utf-8")
        digest.update(struct.pack("<Q", len(header)) + header)
        digest.update(struct.pack("<%df" % len(values), *values))
    return digest.hexdigest()


def _f32_tensor(name, shape, values):
    return name, list(reversed(shape)), F32, struct.pack("<%df" % len(values), *values)


def _mxfp4_rows(values, scales):
    """MXFP4 blocks holding exactly these E2M1 values times their block scales: the scale's E8M0 byte, then the E2M1 codes in Q4_0's nibble order."""
    out = bytearray()
    for block, scale in enumerate(scales):
        exponent = math.frexp(scale)[1] - 1
        if scale != 2.0 ** exponent:
            raise ValueError("an Engram block scale is not a power of two")
        codes = []
        for value in values[block * 32:(block + 1) * 32]:
            magnitude = E2M1.index(abs(value))
            codes.append(magnitude | (8 if value < 0 else 0))
        out.append(exponent + 127)
        out.extend(codes[j] | codes[j + 16] << 4 for j in range(16))
    return bytes(out)


def _metadata(args, engram, layout):
    n = args["n_layers"]
    a = "deepseek41."
    meta = {
        "general.architecture": (STRING, "deepseek41"),
        a + "block_count": (GGUF_UINT32, n),
        a + "context_length": (GGUF_UINT32, args["max_seq_len"]),
        a + "embedding_length": (GGUF_UINT32, args["dim"]),
        a + "attention.head_count": (GGUF_UINT32, args["n_heads"]),
        a + "attention.head_count_kv": (GGUF_UINT32, 1),
        a + "rope.scaling.type": (STRING, "yarn"),
        a + "rope.scaling.factor": (GGUF_FLOAT32, args["rope_factor"]),
        a + "rope.scaling.original_context_length": (GGUF_UINT32, args["original_seq_len"]),
        a + "rope.scaling.yarn_beta_fast": (GGUF_FLOAT32, float(args["beta_fast"])),
        a + "rope.scaling.yarn_beta_slow": (GGUF_FLOAT32, float(args["beta_slow"])),
        a + "rope.freq_base": (GGUF_FLOAT32, args["rope_theta"]),
        a + "attention.layer_norm_rms_epsilon": (GGUF_FLOAT32, args["norm_eps"]),
        a + "expert_count": (GGUF_UINT32, args["n_routed_experts"]),
        a + "expert_used_count": (GGUF_UINT32, args["n_activated_experts"]),
        a + "expert_gating_func": (GGUF_UINT32, 4),
        a + "attention.key_length": (GGUF_UINT32, args["head_dim"]),
        a + "attention.value_length": (GGUF_UINT32, args["head_dim"]),
        a + "rope.dimension_count": (GGUF_UINT32, args["rope_head_dim"]),
        a + "attention.q_lora_rank": (GGUF_UINT32, args["q_lora_rank"]),
        a + "attention.sliding_window": (GGUF_UINT32, args["window_size"]),
        a + "expert_feed_forward_length": (GGUF_UINT32, args["moe_inter_dim"]),
        a + "expert_shared_count": (GGUF_UINT32, args["n_shared_experts"]),
        a + "expert_weights_scale": (GGUF_FLOAT32, args["route_scale"]),
        a + "expert_weights_norm": (GGUF_BOOL, args["norm_topk_prob"]),
        a + "swiglu_clamp_exp": (ARRAY, (GGUF_FLOAT32, [args["swiglu_limit"]] * n)),
        a + "swiglu_clamp_shexp": (ARRAY, (GGUF_FLOAT32, [args["swiglu_limit"]] * n)),
        a + "attention.indexer.head_count": (GGUF_UINT32, args["index_n_heads"]),
        a + "attention.indexer.key_length": (GGUF_UINT32, args["index_head_dim"]),
        a + "attention.indexer.top_k": (GGUF_UINT32, args["index_topk"]),
        a + "attention.output_group_count": (GGUF_UINT32, args["o_groups"]),
        a + "attention.output_lora_rank": (GGUF_UINT32, args["o_lora_rank"]),
        a + "attention.compress_ratios": (ARRAY, (GGUF_INT32, list(args["compress_ratios"]))),
        a + "attention.compress_rope_freq_base": (GGUF_FLOAT32, args["compress_rope_theta"]),
        a + "hyper_connection.count": (GGUF_UINT32, args["hc_mult"]),
        a + "hyper_connection.sinkhorn_iterations": (GGUF_UINT32, args["hc_sinkhorn_iters"]),
        a + "hyper_connection.epsilon": (GGUF_FLOAT32, args["hc_eps"]),
        a + "hash_layer_count": (GGUF_UINT32, 0),
        "tokenizer.ggml.tokens": (ARRAY, (STRING, list(engram["tokens"]))),
        "tokenizer.ggml.bos_token_id": (GGUF_UINT32, 0),
        "tokenizer.ggml.eos_token_id": (GGUF_UINT32, 1),
        "tokenizer.ggml.padding_token_id": (GGUF_UINT32, 1),
        "tokenizer.ggml.add_bos_token": (GGUF_BOOL, False),
        "tokenizer.ggml.add_eos_token": (GGUF_BOOL, False),
    }
    layers = list(args["engram_layer_ids"])
    primes = [list(row) for row in engram["primes"]]
    token_map = list(engram["token_map"])
    if len(token_map) != args["vocab_size"]:
        raise ValueError("the token map must hold one id for each of the fixture's tokens")
    multipliers = [m for row in engram["multipliers"] for m in row]
    if layout == "fork":
        meta.update({
            a + "attention.kv_source_layer_ids": (ARRAY, (GGUF_UINT32, list(args["kv_source_layers"]))),
            a + "attention.index_source_layer_ids": (ARRAY, (GGUF_UINT32, list(args["index_source_layers"]))),
            a + "attention.candidate_source_layer_id": (GGUF_UINT32, args["candidate_source_layer"]),
            a + "attention.candidate_block_size": (GGUF_UINT32, args["candidate_block_size"]),
            a + "attention.candidate_topk_blocks": (GGUF_UINT32, args["candidate_topk_blocks"]),
            a + "engram.layer_ids": (ARRAY, (GGUF_INT32, layers)),
            a + "engram.head_dim": (GGUF_UINT32, args["engram_head_dim"]),
            a + "engram.n_heads": (GGUF_UINT32, args["engram_n_heads"]),
            a + "engram.max_ngram_size": (GGUF_UINT32, args["engram_max_ngram_size"]),
            a + "engram.pad_token_id": (GGUF_UINT32, args["engram_pad_id"]),
            a + "engram.compressed_vocab_size": (GGUF_UINT32, args["engram_compressed_vocab_size"]),
            a + "engram.num_embeddings": (ARRAY, (GGUF_UINT64, [sum(row) for row in primes])),
            a + "engram.primes": (ARRAY, (GGUF_UINT64, [p for row in primes for p in row])),
            a + "engram.offsets": (ARRAY, (GGUF_UINT64, [sum(row[:i]) for row in primes for i in range(len(row))])),
            a + "engram.multipliers": (ARRAY, (GGUF_UINT64, multipliers)),
            a + "engram.token_map": (ARRAY, (GGUF_INT32, token_map)),
        })
    else:
        sizes = [p for row in primes for p in row]
        meta.update({
            a + "attention.kv_source_layers": (ARRAY, (GGUF_UINT32, list(args["kv_source_layers"]))),
            a + "attention.indexer.source_layers": (ARRAY, (GGUF_UINT32, list(args["index_source_layers"]))),
            a + "attention.indexer.candidate_source_layer": (GGUF_INT32, args["candidate_source_layer"]),
            a + "attention.indexer.candidate_top_k_blocks": (GGUF_UINT32, args["candidate_topk_blocks"]),
            a + "attention.indexer.candidate_block_size": (GGUF_UINT32, args["candidate_block_size"]),
            a + "ple.layers": (ARRAY, (GGUF_UINT32, layers)),
            a + "ple.ngram_size": (GGUF_UINT32, args["engram_max_ngram_size"]),
            a + "ple.heads_per_ngram": (GGUF_UINT32, args["engram_n_heads"]),
            a + "embedding_length_per_layer_input": (GGUF_UINT32, args["engram_head_dim"]),
            a + "ple.layer_multipliers": (ARRAY, (GGUF_UINT64, multipliers)),
            a + "ple.head_offsets": (ARRAY, (GGUF_UINT64, [sum(sizes[:i]) for i in range(len(sizes))])),
            a + "ple.head_vocab_sizes": (ARRAY, (GGUF_UINT64, sizes)),
            a + "engram.pad_token_id": (GGUF_UINT32, args["engram_pad_id"]),
            a + "engram.token_map": (ARRAY, (GGUF_INT32, token_map)),
        })
    return meta


# model.py's names to the GGUF's, one table for each layout; a name absent from a layout's table is mapped by code below.
_ROOT = {"embed.weight": "token_embd.weight", "norm.weight": "output_norm.weight", "head.weight": "output.weight"}
_LAYER = {
    "attn_norm.weight": "attn_norm.weight", "ffn_norm.weight": "ffn_norm.weight",
    "hc_attn_fn": "hc_attn_fn.weight", "hc_attn_base": "hc_attn_base.weight", "hc_attn_scale": "hc_attn_scale.weight",
    "hc_ffn_fn": "hc_ffn_fn.weight", "hc_ffn_base": "hc_ffn_base.weight", "hc_ffn_scale": "hc_ffn_scale.weight",
    "attn.attn_sink": "attn_sinks.weight", "attn.wq_a.weight": "attn_q_a.weight", "attn.q_norm.weight": "attn_q_a_norm.weight",
    "attn.wq_b.weight": "attn_q_b.weight", "attn.wkv.weight": "attn_kv.weight", "attn.kv_norm.weight": "attn_kv_a_norm.weight",
    "attn.wo_a.weight": "attn_output_a.weight", "attn.wo_b.weight": "attn_output_b.weight",
    "attn.compressor.wkv.weight": "attn_compressor_kv.weight", "attn.compressor.wgate.weight": "attn_compressor_gate.weight",
    "attn.compressor.norm.weight": "attn_compressor_norm.weight",
    "attn.indexer.wq_b.weight": "indexer.attn_q_b.weight", "attn.indexer.weights_proj.weight": "indexer.proj.weight",
    "ffn.gate.weight": "ffn_gate_inp.weight", "ffn.gate.bias": "exp_probs_b.bias",
    "ffn.shared_experts.w1.weight": "ffn_gate_shexp.weight", "ffn.shared_experts.w2.weight": "ffn_down_shexp.weight",
    "ffn.shared_experts.w3.weight": "ffn_up_shexp.weight",
}
_LAYOUT_LAYER = {
    "fork": {"attn.indexer.wk.weight": "indexer_compressor_kv.weight", "attn.indexer.k_norm.weight": "indexer_compressor_norm.weight",
             "engram.wkv.weight": "engram_wkv.weight", "engram.q_weight": "engram_q.weight", "engram.k_weight": "engram_k.weight"},
    "upstream": {"attn.indexer.wk.weight": "indexer.attn_k.weight", "attn.indexer.k_norm.weight": "indexer.k_norm.weight",
                 "engram.wkv.weight": "engram_kv.weight"},
}
_EXPERTS = {"w1": "ffn_gate_exps.weight", "w2": "ffn_down_exps.weight", "w3": "ffn_up_exps.weight"}


def gguf_name(name, layout):
    """The GGUF tensor a model.py parameter is written to in a layout, or None where it is written combined (experts, the Engram gate and tables)."""
    if name in _ROOT:
        return _ROOT[name]
    _, layer, rest = name.split(".", 2)
    table = dict(_LAYER, **_LAYOUT_LAYER[layout])
    return "blk.%s.%s" % (layer, table[rest]) if rest in table else None


def map_deepseek41(args, state, engram, layout):
    """Return `(metadata, tensors)` of the tiny file in a layout, for `spec_decode.write_gguf`; every state entry is written once."""
    if layout not in LAYOUTS:
        raise ValueError("unknown deepseek41 layout %r" % (layout,))
    used, tensors = set(), []
    for name in sorted(state):
        target = gguf_name(name, layout)
        if target is not None:
            tensors.append(_f32_tensor(target, *state[name]))
            used.add(name)
    tables = []
    for layer in range(args["n_layers"]):
        p = "layers.%d." % layer
        for proj, target in _EXPERTS.items():
            names = [p + "ffn.experts.%d.%s.weight" % (e, proj) for e in range(args["n_routed_experts"])]
            shape = state[names[0]][0]
            values = [v for n in names for v in state[n][1]]
            tensors.append(_f32_tensor("blk.%d.%s" % (layer, target), (len(names),) + tuple(shape), values))
            used.update(names)
        if layer not in args["engram_layer_ids"]:
            continue
        weight, scale = state[p + "engram.embed.weight"], state[p + "engram.embed.scale"]
        used.update((p + "engram.embed.weight", p + "engram.embed.scale"))
        if layout == "fork":
            tensors.append(("blk.%d.engram_embed.weight" % layer, list(reversed(weight[0])), MXFP4, _mxfp4_rows(weight[1], scale[1])))
        else:
            q, k = state[p + "engram.q_weight"], state[p + "engram.k_weight"]
            product = [a * b for a, b in zip(q[1], k[1])]
            if struct.unpack("<%df" % len(product), struct.pack("<%df" % len(product), *product)) != tuple(product):
                raise ValueError("engram_q times engram_k is not exact in f32")
            tensors.append(_f32_tensor("blk.%d.engram_gate.weight" % layer, q[0], product))
            used.update((p + "engram.q_weight", p + "engram.k_weight"))
            tables.extend(weight[1][i] * scale[1][i // 32] for i in range(len(weight[1])))
    if layout == "upstream":
        rows = len(tables) // args["engram_head_dim"]
        tensors.append(_f32_tensor("per_layer_token_embd.weight", (rows, args["engram_head_dim"]), tables))
    if used != set(state):
        raise ValueError("unmapped tensors remain: %r" % sorted(set(state) - used))
    names = [t[0] for t in tensors]
    if len(names) != len(set(names)):
        raise ValueError("two parameters map to one GGUF tensor")
    return _metadata(args, engram, layout), tensors


def write_deepseek41(path, args, state, engram, layout):
    """Write the tiny file in a layout, replacing `path` only once it is complete."""
    metadata, tensors = map_deepseek41(args, state, engram, layout)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, staged_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    staged = Path(staged_name)
    try:
        write_gguf(staged, metadata, tensors)
        os.replace(staged, path)
    except BaseException:
        try:
            staged.unlink()
        except FileNotFoundError:
            pass
        raise
    return path


# The sidecar's keys, the trunk's that a stage shares with it and its own (docs/DEEPSEEK4.md, section 4.7).
_DFLASH_SHARED = ("context_length", "embedding_length", "attention.head_count", "attention.head_count_kv", "rope.scaling.type", "rope.scaling.factor",
                  "rope.scaling.original_context_length", "rope.scaling.yarn_beta_fast", "rope.scaling.yarn_beta_slow", "rope.freq_base",
                  "attention.layer_norm_rms_epsilon", "expert_gating_func", "attention.key_length", "attention.value_length", "rope.dimension_count",
                  "attention.q_lora_rank", "attention.sliding_window", "expert_feed_forward_length", "expert_shared_count", "expert_weights_scale",
                  "expert_weights_norm", "attention.indexer.head_count", "attention.indexer.key_length", "attention.indexer.top_k",
                  "attention.output_group_count", "attention.output_lora_rank", "attention.compress_rope_freq_base", "hyper_connection.count",
                  "hyper_connection.sinkhorn_iterations", "hyper_connection.epsilon", "hash_layer_count")
_DFLASH_GLOBALS = {"mtp.0.main_proj.weight": "fc.weight", "mtp.0.main_norm.weight": "enc.output_norm.weight", "norm.weight": "output_norm.weight",
                   "markov_head.embed.weight": "markov_w1.weight", "markov_head.head.weight": "markov_w2.weight",
                   "confidence_head.proj.weight": "conf_proj.weight"}


def map_dflash(args, state, engram):
    """Return `(metadata, tensors)` of the tiny `dflash` sidecar, as the converter writes DeepSeek-V4.1-Flash-DSpark: its stages as blocks 0 to 2 and its globals by their own names."""
    trunk = _metadata(args, engram, "fork")
    stages, n = args["n_mtp_layers"], args["n_layers"]
    meta = {key: value for key, value in trunk.items() if key.startswith("tokenizer.")}
    meta.update({"dflash." + key: trunk["deepseek41." + key] for key in _DFLASH_SHARED})
    meta.update({
        "general.architecture": (STRING, "dflash"),
        "dflash.block_count": (GGUF_UINT32, stages),
        "dflash.expert_count": (GGUF_UINT32, args["dspark_n_routed_experts"]),
        "dflash.expert_used_count": (GGUF_UINT32, args["dspark_n_activated_experts"]),
        "dflash.swiglu_clamp_exp": (ARRAY, (GGUF_FLOAT32, [args["swiglu_limit"]] * stages)),
        "dflash.swiglu_clamp_shexp": (ARRAY, (GGUF_FLOAT32, [args["swiglu_limit"]] * stages)),
        "dflash.attention.compress_ratios": (ARRAY, (GGUF_INT32, list(args["compress_ratios"][n:]))),
        "dflash.block_size": (GGUF_UINT32, args["dspark_block_size"]),
        "dflash.target_layers": (ARRAY, (GGUF_INT32, list(args["dspark_target_layer_ids"]))),
        "tokenizer.ggml.mask_token_id": (GGUF_UINT32, args["dspark_noise_token_id"]),
    })
    tensors = []
    last = "mtp.%d." % (stages - 1)
    for name in sorted(state):
        if not name.startswith("mtp."):
            continue
        stage, rest = name.split(".", 2)[1:]
        if rest in ("embed.weight", "head.weight") or ".experts." in rest:
            continue
        if name in _DFLASH_GLOBALS:
            tensors.append(_f32_tensor(_DFLASH_GLOBALS[name], *state[name]))
        elif name.startswith(last) and rest in _DFLASH_GLOBALS:
            tensors.append(_f32_tensor(_DFLASH_GLOBALS[rest], *state[name]))
        else:
            tensors.append(_f32_tensor("blk.%s.%s" % (stage, _LAYER[rest]), *state[name]))
    for stage in range(stages):
        for proj, target in _EXPERTS.items():
            names = ["mtp.%d.ffn.experts.%d.%s.weight" % (stage, e, proj) for e in range(args["dspark_n_routed_experts"])]
            values = [v for name in names for v in state[name][1]]
            tensors.append(_f32_tensor("blk.%d.%s" % (stage, target), (len(names),) + tuple(state[names[0]][0]), values))
    return meta, tensors


def write_dflash(path, args, state, engram):
    """Write the tiny sidecar, replacing `path` only once it is complete."""
    metadata, tensors = map_dflash(args, state, engram)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(path.name + ".tmp")
    write_gguf(staged, metadata, tensors)
    os.replace(staged, path)
    return path


def fixture_from_document(document, **overrides):
    """The args, state and Engram constants a committed golden names, rebuilt from its seed; `overrides` replace args, as the group-legal shape's four output groups do."""
    args = dict(document["args"], **overrides)
    engram = document["engram"]
    zero_rows = {int(layer): rows for layer, rows in engram["zero_rows"].items()}
    state = make_state(args, [sum(row) for row in engram["primes"]], document["seed"], zero_rows)
    return args, state, engram


def dspark_fixture(document):
    """The trunk's args, the DSpark args and their state, as the DSpark golden (document["dspark"]) names them, rebuilt from the trunk's seed and the drafter's."""
    args, _, engram = fixture_from_document(document)
    dspark = document["dspark"]["args"]
    zero_rows = {int(layer): rows for layer, rows in engram["zero_rows"].items()}
    state = make_state(dspark, [sum(row) for row in engram["primes"]], document["seed"], zero_rows, document["dspark"]["seed"])
    return args, dspark, state, engram


# The model formats' rounding points (docs/DEEPSEEK4.md, section 4.5): E4M3 values under a power-of-two scale per block, and E2M1 values.
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def e4m3_margin(y):
    """How far a scaled value lies from E4M3's nearest rounding midpoint, in units of the code spacing there; a value on a code is 0.5 away."""
    a = abs(y)
    if a == 0:
        return 0.5
    spacing = 2.0 ** (max(math.floor(math.log2(a)), -6) - 3)
    f = a / spacing
    return abs(f - math.floor(f) - 0.5)


def e2m1_margin(y):
    """How far a scaled value lies from E2M1's nearest rounding midpoint, in units of the spacing of the two codes around it."""
    a = abs(y)
    for lo, hi in zip(E2M1_VALUES, E2M1_VALUES[1:]):
        if a <= hi:
            return abs(a - (lo + hi) / 2) / (hi - lo)
    return 0.5


def power_margin(t, codes_per_octave):
    """How far amax over the format's largest value, `t`, lies from the power of two where the block's scale doubles, in code spacings of the block's largest value."""
    return codes_per_octave * abs(t / 2.0 ** round(math.log2(t)) - 1)


def fp8_margins(blocks):
    """The least value margin and the least scale margin over blocks of F32 values coded to FP8 E4M3 with s = 2^ceil(log2(max(amax, 1e-4) / 448))."""
    value, scale = 0.5, math.inf
    for block in blocks:
        t = _f32(max(max(abs(v) for v in block), 1e-4) * _f32(1.0 / 448.0))
        s = 2.0 ** math.ceil(math.log2(t))
        value = min(value, min(e4m3_margin(v / s) for v in block))
        scale = min(scale, power_margin(t, 14.0))
    return value, scale


def fp4_margins(blocks, e4m3_scale):
    """The least value margin and the least scale margin over blocks coded to FP4 E2M1, under an E4M3 scale of amax / 6 or an E8M0 one of 2^ceil(log2(amax / 6))."""
    value, scale = 0.5, math.inf
    for block in blocks:
        amax = max(abs(v) for v in block)
        if e4m3_scale:
            t = _f32(max(amax, 6 * 2.0 ** -9) / 6.0)
            scale = min(scale, e4m3_margin(t))
            s = _e4m3(t)
        else:
            t = _f32(max(amax, 6 * 2.0 ** -126) * _f32(1.0 / 6.0))
            scale = min(scale, power_margin(t, 3.0))
            s = 2.0 ** math.ceil(math.log2(t))
        value = min(value, min(e2m1_margin(min(max(v / s, -6.0), 6.0)) for v in block))
    return value, scale


def _f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def _e4m3(x):
    """x rounded to E4M3, ties to even, for a positive normal x up to 448."""
    spacing = 2.0 ** (max(math.floor(math.log2(x)), -6) - 3)
    return round(x / spacing) * spacing


def search_drafter_seed(measure, seeds):
    """The drafter seed with the largest least margin over its window and block rows, the lower seed on a tie, and each seed's margins; measure(seed) gives {"window": ..., "block": ...}."""
    results = {seed: measure(seed) for seed in seeds}
    return max(seeds, key=lambda seed: (min(results[seed]["window"], results[seed]["block"]), -seed)), results
