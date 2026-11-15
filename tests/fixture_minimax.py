"""The tiny minimax-m3 fixtures: their weights from a seed, and their GGUF, torch-free.

The state maps the parameter names of transformers' MiniMaxM3VLForCausalLM (5.23.0) to `(shape, values)` flattened row-major.
tools/minimax_reference.py loads the same state into HF's model, so the suite rebuilds the reference's weights from the golden's seed alone.
"""

import math
import os
import struct
import tempfile
from pathlib import Path

try:
    from .spec_decode import ARRAY, F32, STRING, write_gguf
    from .fixture_deepseek4 import _linear_amp, _stream, _uniform, state_sha256
except ImportError:
    from spec_decode import ARRAY, F32, STRING, write_gguf
    from fixture_deepseek4 import _linear_amp, _stream, _uniform, state_sha256

GGUF_UINT32 = 4
GGUF_FLOAT32 = 6
GGUF_BOOL = 7
EOS = 1

# HF's text config of each fixture: "small" crosses its first sparse drop at position 12 (block 4, 3 kept), "wide" has the real head and index widths and block, and drops first at 256.
_COMMON = {
    "vocab_size": 128, "rms_norm_eps": 1e-6, "routed_scaling_factor": 2.0, "swiglu_alpha": 1.702, "swiglu_limit": 7.0,
    "index_local_blocks": 1, "tie_word_embeddings": False, "max_position_embeddings": 512, "bos_token_id": 0, "eos_token_id": EOS,
}
FIXTURES = {
    "small": dict(_COMMON, hidden_size=64, intermediate_size=32, shared_intermediate_size=32, dense_intermediate_size=96, num_hidden_layers=3,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=16, rotary_dim=8, num_local_experts=8, num_experts_per_tok=3,
                  mlp_layer_types=["dense", "sparse", "sparse"], layer_types=["minimax_m3_sparse"] * 3, index_n_heads=2, index_head_dim=16,
                  index_block_size=4, index_topk_blocks=3, rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5}),
    "wide": dict(_COMMON, hidden_size=128, intermediate_size=32, shared_intermediate_size=32, dense_intermediate_size=64, num_hidden_layers=2,
                 num_attention_heads=4, num_key_value_heads=2, head_dim=128, rotary_dim=64, num_local_experts=4, num_experts_per_tok=2,
                 mlp_layer_types=["dense", "sparse"], layer_types=["minimax_m3_sparse"] * 2, index_n_heads=2, index_head_dim=128,
                 index_block_size=128, index_topk_blocks=2, rope_parameters={"rope_type": "default", "rope_theta": 10000000.0, "partial_rotary_factor": 0.5}),
}


def _norm_offsets(seed, name, count):
    """Gemma norm offsets w, each a multiple of 2^-19 below 1/8 in magnitude, so the file's 1 + w is exact in f32."""
    rng = _stream(seed, name)
    return [((rng.next() >> 47) - (1 << 16)) / float(1 << 19) for _ in range(count)]


def parameter_shapes(config):
    """Every HF parameter and persistent buffer of the text model, by name, with its shape."""
    E, V, D, H, HK = config["hidden_size"], config["vocab_size"], config["head_dim"], config["num_attention_heads"], config["num_key_value_heads"]
    I, IH, X, Fe, F = config["index_head_dim"], config["index_n_heads"], config["num_local_experts"], config["intermediate_size"], config["dense_intermediate_size"]
    shapes = {"model.embed_tokens.weight": (V, E), "model.norm.weight": (E,), "lm_head.weight": (V, E)}
    for layer, kind in enumerate(config["mlp_layer_types"]):
        p = "model.layers.%d." % layer
        shapes.update({p + "self_attn.q_proj.weight": (H * D, E), p + "self_attn.k_proj.weight": (HK * D, E), p + "self_attn.v_proj.weight": (HK * D, E),
                       p + "self_attn.o_proj.weight": (E, H * D), p + "self_attn.q_norm.weight": (D,), p + "self_attn.k_norm.weight": (D,),
                       p + "self_attn.indexer.q_proj.weight": (IH * I, E), p + "self_attn.indexer.k_proj.weight": (I, E),
                       p + "self_attn.indexer.q_norm.weight": (I,), p + "self_attn.indexer.k_norm.weight": (I,),
                       p + "input_layernorm.weight": (E,), p + "post_attention_layernorm.weight": (E,)})
        if kind == "dense":
            shapes.update({p + "mlp.gate_up_proj.weight": (2 * F, E), p + "mlp.down_proj.weight": (E, F)})
        else:
            S = config["shared_intermediate_size"]
            shapes.update({p + "mlp.gate.weight": (X, E), p + "mlp.gate.e_score_correction_bias": (X,), p + "mlp.experts.gate_up_proj": (X, 2 * Fe, E),
                           p + "mlp.experts.down_proj": (X, E, Fe), p + "mlp.shared_experts.gate_up_proj.weight": (2 * S, E),
                           p + "mlp.shared_experts.down_proj.weight": (E, S)})
    return shapes


def make_state(config, seed):
    """The fixture's state: name to (shape, values), each value exact in f32."""
    state = {}
    for name, shape in parameter_shapes(config).items():
        count = math.prod(shape)
        if name.endswith("norm.weight") or name.endswith("layernorm.weight"):
            state[name] = (shape, _norm_offsets(seed, name, count))
        elif name.endswith("e_score_correction_bias"):
            state[name] = (shape, _uniform(seed, name, count, 2.0 ** -3))
        elif name in ("model.embed_tokens.weight", "lm_head.weight"):
            state[name] = (shape, _uniform(seed, name, count, 1.0 if name.startswith("model.") else 2.0 ** -3))
        else:
            state[name] = (shape, _uniform(seed, name, count, _linear_amp(shape[-1])))
    return state


def _byte_char(b):
    """GPT-2's byte-level character of byte `b`, so token b of the fixture's vocabulary is the one byte b."""
    printable = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    if b in printable:
        return chr(b)
    return chr(256 + [c for c in range(256) if c not in printable].index(b))


def tokens(config):
    """The vocabulary: token i is byte i, so ASCII text without merges is one token a byte."""
    return [_byte_char(i) for i in range(config["vocab_size"])]


def _f32_tensor(name, shape, values):
    return name, list(reversed(shape)), F32, struct.pack("<%df" % len(values), *values)


def _metadata(config):
    a = "minimax-m3."
    layers = config["mlp_layer_types"]
    dense = layers.index("sparse")
    if any(kind != "dense" for kind in layers[:dense]) or any(kind != "sparse" for kind in layers[dense:]):
        raise ValueError("minimax-m3 keeps its dense layers first")
    rotary = int(config["head_dim"] * config["rope_parameters"]["partial_rotary_factor"])
    if rotary != config["rotary_dim"]:
        raise ValueError("the rotary width HF's rope computes is not rotary_dim")
    return {
        "general.architecture": (STRING, "minimax-m3"),
        a + "block_count": (GGUF_UINT32, config["num_hidden_layers"]),
        a + "leading_dense_block_count": (GGUF_UINT32, dense),
        a + "context_length": (GGUF_UINT32, config["max_position_embeddings"]),
        a + "embedding_length": (GGUF_UINT32, config["hidden_size"]),
        a + "feed_forward_length": (GGUF_UINT32, config["dense_intermediate_size"]),
        a + "attention.head_count": (GGUF_UINT32, config["num_attention_heads"]),
        a + "attention.head_count_kv": (GGUF_UINT32, config["num_key_value_heads"]),
        a + "attention.key_length": (GGUF_UINT32, config["head_dim"]),
        a + "attention.value_length": (GGUF_UINT32, config["head_dim"]),
        a + "rope.dimension_count": (GGUF_UINT32, rotary),
        a + "rope.freq_base": (GGUF_FLOAT32, config["rope_parameters"]["rope_theta"]),
        a + "attention.layer_norm_rms_epsilon": (GGUF_FLOAT32, config["rms_norm_eps"]),
        a + "expert_count": (GGUF_UINT32, config["num_local_experts"]),
        a + "expert_used_count": (GGUF_UINT32, config["num_experts_per_tok"]),
        a + "expert_gating_func": (GGUF_UINT32, 2),
        a + "expert_feed_forward_length": (GGUF_UINT32, config["intermediate_size"]),
        a + "expert_shared_count": (GGUF_UINT32, 1),
        a + "expert_weights_scale": (GGUF_FLOAT32, config["routed_scaling_factor"]),
        a + "expert_weights_norm": (GGUF_BOOL, True),
        a + "attention.indexer.head_count": (GGUF_UINT32, config["index_n_heads"]),
        a + "attention.indexer.key_length": (GGUF_UINT32, config["index_head_dim"]),
        a + "attention.indexer.top_k": (GGUF_UINT32, config["index_topk_blocks"]),
        a + "attention.indexer.block_size": (GGUF_UINT32, config["index_block_size"]),
        a + "attention.indexer.local_blocks": (GGUF_UINT32, config["index_local_blocks"]),
        "tokenizer.ggml.tokens": (ARRAY, (STRING, tokens(config))),
        "tokenizer.ggml.bos_token_id": (GGUF_UINT32, config["bos_token_id"]),
        "tokenizer.ggml.eos_token_id": (GGUF_UINT32, config["eos_token_id"]),
        "tokenizer.ggml.add_bos_token": (GGUF_BOOL, False),
        "tokenizer.ggml.add_eos_token": (GGUF_BOOL, False),
    }


# The one-matrix tensors by their HF suffix, as the converter names them; norms take the Gemma 1 + w it folds in.
_MATRICES = {"self_attn.q_proj.weight": "attn_q", "self_attn.k_proj.weight": "attn_k", "self_attn.v_proj.weight": "attn_v",
             "self_attn.o_proj.weight": "attn_output", "self_attn.indexer.q_proj.weight": "indexer.q_proj",
             "self_attn.indexer.k_proj.weight": "indexer.k_proj", "mlp.down_proj.weight": "ffn_down", "mlp.gate.weight": "ffn_gate_inp",
             "mlp.shared_experts.down_proj.weight": "ffn_down_shexp"}
_NORMS = {"input_layernorm.weight": "attn_norm", "post_attention_layernorm.weight": "ffn_norm", "self_attn.q_norm.weight": "attn_q_norm",
          "self_attn.k_norm.weight": "attn_k_norm", "self_attn.indexer.q_norm.weight": "indexer.q_norm", "self_attn.indexer.k_norm.weight": "indexer.k_norm"}
# A fused gate and up matrix, gate rows first, and the halves' names.
_FUSED = {"mlp.gate_up_proj.weight": ("ffn_gate", "ffn_up"), "mlp.shared_experts.gate_up_proj.weight": ("ffn_gate_shexp", "ffn_up_shexp"),
          "mlp.experts.gate_up_proj": ("ffn_gate_exps", "ffn_up_exps")}


def map_minimax(config, state):
    """Return `(metadata, tensors)` of the fixture's F32 file, for `spec_decode.write_gguf`; every state entry is written once."""
    tensors, used = [], set()
    one_plus = lambda values: [1.0 + v for v in values]
    tensors.append(_f32_tensor("token_embd.weight", *state["model.embed_tokens.weight"]))
    tensors.append(_f32_tensor("output.weight", *state["lm_head.weight"]))
    shape, values = state["model.norm.weight"]
    tensors.append(_f32_tensor("output_norm.weight", shape, one_plus(values)))
    used.update(("model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"))
    for layer in range(config["num_hidden_layers"]):
        p, out = "model.layers.%d." % layer, "blk.%d." % layer
        for suffix, target in _MATRICES.items():
            if p + suffix in state:
                tensors.append(_f32_tensor(out + target + ".weight", *state[p + suffix]))
                used.add(p + suffix)
        for suffix, target in _NORMS.items():
            shape, values = state[p + suffix]
            tensors.append(_f32_tensor(out + target + ".weight", shape, one_plus(values)))
            used.add(p + suffix)
        for suffix, (gate, up) in _FUSED.items():
            if p + suffix not in state:
                continue
            shape, values = state[p + suffix]
            rows, cols = shape[-2], shape[-1]
            stacks = math.prod(shape[:-2])
            half = rows // 2 * cols
            halves = ([], [])
            for s in range(stacks):
                block = values[s * rows * cols:(s + 1) * rows * cols]
                halves[0].extend(block[:half])
                halves[1].extend(block[half:])
            for name, part in zip((gate, up), halves):
                tensors.append(_f32_tensor(out + name + ".weight", tuple(shape[:-2]) + (rows // 2, cols), part))
            used.add(p + suffix)
        if p + "mlp.experts.down_proj" in state:
            tensors.append(_f32_tensor(out + "ffn_down_exps.weight", *state[p + "mlp.experts.down_proj"]))
            tensors.append(_f32_tensor(out + "exp_probs_b.bias", *state[p + "mlp.gate.e_score_correction_bias"]))
            used.update((p + "mlp.experts.down_proj", p + "mlp.gate.e_score_correction_bias"))
    if used != set(state):
        raise ValueError("unmapped tensors remain: %r" % sorted(set(state) - used))
    names = [t[0] for t in tensors]
    if len(names) != len(set(names)):
        raise ValueError("two parameters map to one GGUF tensor")
    return _metadata(config), tensors


def write_minimax(path, config, state):
    """Write the fixture's file, replacing `path` only once it is complete."""
    metadata, tensors = map_minimax(config, state)
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
