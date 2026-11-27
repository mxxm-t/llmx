"""The tiny inkling2_loop fixture: its weights from a seed, its GGUF in the layout of the real files, torch-free.

The state maps the parameter names of transformers' Inkling2LoopForCausalLM (5.24.0) to `(shape, values)` flattened row-major.
tools/inkling2_reference.py loads the same state into HF's model; `map_inkling2` holds the file layouts measured on the real file (docs/INKLING2.md, 1. The files) once.
"""

import math
import os
import struct
import tempfile
from pathlib import Path

try:
    from .spec_decode import ARRAY, F32, STRING, write_gguf
    from .fixture_deepseek4 import _linear_amp, _uniform, state_sha256
    from .fixture_minimax import _byte_char
except ImportError:
    from spec_decode import ARRAY, F32, STRING, write_gguf
    from fixture_deepseek4 import _linear_amp, _uniform, state_sha256
    from fixture_minimax import _byte_char

GGUF_UINT32 = 4
GGUF_FLOAT32 = 6
GGUF_BOOL = 7
EOS = 1
SLIDING = "hybrid_sliding"

# Twelve physical layers in blocks of three, the loop over layers 3 to 8 run three times, full attention closing each block: 24 execution layers, every layer kind of the real model.
# The window (5), the relative extent (7) and the log-scale floor (8) are crossed within a few dozen tokens, and the clip thresholds are small enough that some rows clip.
CONFIG = {
    "vocab_size": 300, "unpadded_vocab_size": 288, "hidden_size": 20, "num_hidden_layers": 12,
    "num_attention_heads": 6, "num_key_value_heads": 2, "head_dim": 4, "swa_num_attention_heads": 6, "swa_num_key_value_heads": 2, "swa_head_dim": 4,
    "sliding_window_size": 5, "d_rel": 3, "rel_extent": 7, "log_scaling_n_floor": 8, "log_scaling_alpha": 0.1,
    "layer_types": [SLIDING if (i + 1) % 3 else "hybrid" for i in range(12)],
    "max_position_embeddings": 4096, "rms_norm_eps": 1e-6, "conv_kernel_size": 4,
    "moe_intermediate_size": 8, "n_routed_experts": 6, "num_experts_per_tok": 2, "route_scale": 8.0,
    "qk_l2_clip_threshold": 1.5, "rel_l2_clip_threshold": 0.75, "n_layers_per_attn_res_block": 3,
    "mlp_gate_softcap": 4.0, "mlp_up_softcap": 25.0, "final_logit_softcapping": 30.0,
    "loop_layer_index_start": 3, "loop_layer_index_end": 9, "loop_num_passes": 3, "loop_gate_rank": 3,
    "bos_token_id": 0, "eos_token_id": EOS, "tie_word_embeddings": False,
}


def _global(config, layer):
    return config["layer_types"][layer] != SLIDING


def _loop(config, layer):
    return config["loop_layer_index_start"] <= layer < config["loop_layer_index_end"]


def parameter_shapes(config):
    """Every HF parameter of the text model, by name, with its shape."""
    E, V, X, F, R, K = (config[k] for k in ("hidden_size", "vocab_size", "n_routed_experts", "moe_intermediate_size", "d_rel", "conv_kernel_size"))
    loops = config["loop_layer_index_end"] - config["loop_layer_index_start"]
    shapes = {"model.embed_tokens.weight": (V, E), "model.embed_tokens.embed_norm.weight": (E,), "model.norm.weight": (E,),
              "model.final_input.query": (E,), "model.message_input.query": (loops, 4, E), "lm_head.weight": (config["unpadded_vocab_size"], E)}
    for layer in range(config["num_hidden_layers"]):
        p, glob = "model.layers.%d." % layer, _global(config, layer)
        H, HK, D = (config["num_attention_heads"], config["num_key_value_heads"], config["head_dim"]) if glob else \
            (config["swa_num_attention_heads"], config["swa_num_key_value_heads"], config["swa_head_dim"])
        extent = config["rel_extent"] if glob else config["sliding_window_size"]
        a = p + "self_attn."
        shapes.update({p + "mlp_input.query": (E,), a + "q_proj.linear.weight": (H * D, E), a + "q_proj.norm.scale": (H, D),
                       a + "k_proj.linear.weight": (HK * D, E), a + "v_proj.linear.weight": (HK * D, E),
                       a + "o_proj.gate_proj.weight": (H * D, E), a + "o_proj.linear.weight": (E, H * D), a + "sink_proj.weight": (H, E),
                       a + "relative_position_bias.proj": (R, extent), a + "relative_position_bias.linear.weight": (H * R, E),
                       a + "relative_position_bias.norm.scale": (H, R), p + "mlp.experts.gate_up_proj": (X, 2 * F, E),
                       p + "mlp.experts.down_proj": (X, E, F), p + "mlp.gate.weight": (X, E), p + "mlp.gate.e_score_correction_bias": (X,),
                       p + "post_attention_layernorm.weight": (E,), p + "post_attention_convolution.conv1d.weight": (E, 1, K),
                       p + "post_mlp_convolution.conv1d.weight": (E, 1, K)})
        if layer > 0:
            shapes[p + "attention_input.query"] = (3, E) if glob else (E,)
        if glob:
            for kv in "kv":
                shapes.update({a + kv + "_proj.sconv.static_weight": (K, HK * D), a + kv + "_proj.sconv.weight_proj.weight": (HK * K, E)})
            for qkv in "qkv":
                shapes[p + qkv + "_input_layernorm.weight"] = (E,)
        else:
            shapes.update({a + "k_proj.sconv.conv1d.weight": (HK * D, 1, K), a + "v_proj.sconv.conv1d.weight": (HK * D, 1, K),
                           p + "input_layernorm.weight": (E,)})
        if _loop(config, layer):
            for gate in ("q", "k", "v", "mlp"):
                g = p + gate + "_message_gate."
                shapes.update({g + "down_proj.weight": (config["loop_gate_rank"], 2 * E), g + "forget_gate_proj.weight": (E, config["loop_gate_rank"]),
                               g + "message_gate_proj.weight": (E, config["loop_gate_rank"])})
    return shapes


def make_state(config, seed):
    """The fixture's state: name to (shape, values), each value exact in f32; nothing HF initialises to a constant is left constant."""
    state = {}
    for name, shape in parameter_shapes(config).items():
        count = math.prod(shape)
        if name.endswith("norm.weight") or name.endswith("layernorm.weight") or name.endswith(".scale"):
            values = _uniform(seed, name, count, 0.5, 1.0)
        elif name.endswith(".query"):
            values = _uniform(seed, name, count, 1.0)
        elif name.endswith("e_score_correction_bias"):
            values = _uniform(seed, name, count, 2.0 ** -3)
        elif name.endswith("static_weight") or name.endswith("conv1d.weight") or name.endswith("relative_position_bias.proj"):
            values = _uniform(seed, name, count, 0.5)
        elif name == "model.embed_tokens.weight":
            values = _uniform(seed, name, count, 1.0)
        else:
            values = _uniform(seed, name, count, _linear_amp(shape[-1]))
        state[name] = (shape, values)
    return state


def tokens(config):
    """The vocabulary: token i below 256 is byte i, so ASCII text without merges is one token a byte; the rest are named placeholders."""
    return [_byte_char(i) if i < 256 else "<|extra_%d|>" % i for i in range(config["vocab_size"])]


def _f32(name, shape, values):
    return name, list(reversed(shape)), F32, struct.pack("<%df" % len(values), *values)


def _metadata(config):
    a = "inkling2_loop."
    n = config["num_hidden_layers"]
    heads = [config["num_attention_heads"] if _global(config, i) else config["swa_num_attention_heads"] for i in range(n)]
    kv_heads = [config["num_key_value_heads"] if _global(config, i) else config["swa_num_key_value_heads"] for i in range(n)]
    u32 = lambda v: (GGUF_UINT32, v)
    f32 = lambda v: (GGUF_FLOAT32, v)
    return {
        "general.architecture": (STRING, "inkling2_loop"),
        a + "block_count": u32(n), a + "context_length": u32(config["max_position_embeddings"]),
        a + "embedding_length": u32(config["hidden_size"]), a + "feed_forward_length": u32(config["moe_intermediate_size"]),
        a + "attention.head_count": (ARRAY, (GGUF_UINT32, heads)), a + "attention.head_count_kv": (ARRAY, (GGUF_UINT32, kv_heads)),
        a + "attention.layer_norm_rms_epsilon": f32(config["rms_norm_eps"]),
        a + "expert_count": u32(config["n_routed_experts"]), a + "expert_used_count": u32(config["num_experts_per_tok"]),
        a + "attention.key_length": u32(config["head_dim"]), a + "attention.value_length": u32(config["head_dim"]),
        a + "attention.key_length_swa": u32(config["swa_head_dim"]), a + "attention.value_length_swa": u32(config["swa_head_dim"]),
        a + "loop.count": u32(config["loop_num_passes"]), a + "loop.start": u32(config["loop_layer_index_start"]),
        a + "loop.end": u32(config["loop_layer_index_end"]), a + "loop.gate_rank": u32(config["loop_gate_rank"]), a + "loop.varying": (GGUF_BOOL, False),
        a + "vocab_size": u32(config["vocab_size"]), a + "vocab_size_unpadded": u32(config["unpadded_vocab_size"]),
        a + "expert_feed_forward_length": u32(config["moe_intermediate_size"]), a + "rope.dimension_count": u32(0),
        a + "attention.sliding_window": u32(config["sliding_window_size"]),
        a + "attention.sliding_window_pattern": (ARRAY, (GGUF_BOOL, [not _global(config, i) for i in range(n)])),
        a + "shortconv.l_cache": u32(config["conv_kernel_size"]), a + "expert_weights_scale": f32(config["route_scale"]),
        a + "activation.situ_beta": f32(config["mlp_gate_softcap"]), a + "activation.situ_linear_beta": f32(config["mlp_up_softcap"]),
        a + "attn_res.block_size": u32(config["n_layers_per_attn_res_block"]), a + "final_logit_softcapping": f32(config["final_logit_softcapping"]),
        a + "attention.relative_length": u32(config["d_rel"]), a + "attention.relative_extent": u32(config["rel_extent"]),
        a + "attention.qk_clip": f32(config["qk_l2_clip_threshold"]), a + "attention.relative_clip": f32(config["rel_l2_clip_threshold"]),
        a + "attention.log_scale_floor": u32(config["log_scaling_n_floor"]), a + "attention.log_scale_alpha": f32(config["log_scaling_alpha"]),
        "tokenizer.ggml.model": (STRING, "gpt2"), "tokenizer.ggml.tokens": (ARRAY, (STRING, tokens(config))),
        "tokenizer.ggml.bos_token_id": u32(config["bos_token_id"]), "tokenizer.ggml.eos_token_id": u32(config["eos_token_id"]),
        "tokenizer.ggml.add_bos_token": (GGUF_BOOL, False), "tokenizer.ggml.add_eos_token": (GGUF_BOOL, False),
    }


# HF suffix to file name, values unchanged (a shape [a, b] is written ne = [b, a]).
_PLAIN = {"mlp_input.query": "ffn_res_query", "attention_input.query": "attn_res_query", "self_attn.q_proj.linear.weight": "attn_q",
          "self_attn.q_proj.norm.scale": "attn_q_clip", "self_attn.k_proj.linear.weight": "attn_k", "self_attn.v_proj.linear.weight": "attn_v",
          "self_attn.o_proj.gate_proj.weight": "attn_gate", "self_attn.o_proj.linear.weight": "attn_output", "self_attn.sink_proj.weight": "attn_sink_proj",
          "self_attn.relative_position_bias.linear.weight": "attn_rel",
          "self_attn.relative_position_bias.norm.scale": "attn_rel_clip", "mlp.experts.gate_up_proj": "ffn_gate_up_exps",
          "mlp.experts.down_proj": "ffn_down_exps", "mlp.gate.weight": "ffn_gate_inp", "input_layernorm.weight": "attn_norm",
          "q_input_layernorm.weight": "attn_q_inp_norm", "k_input_layernorm.weight": "attn_k_inp_norm", "v_input_layernorm.weight": "attn_v_inp_norm",
          "post_attention_layernorm.weight": "ffn_norm", "self_attn.k_proj.sconv.conv1d.weight": "attn_k_conv",
          "self_attn.v_proj.sconv.conv1d.weight": "attn_v_conv", "post_attention_convolution.conv1d.weight": "attn_post_conv",
          "post_mlp_convolution.conv1d.weight": "ffn_post_conv", "self_attn.k_proj.sconv.weight_proj.weight": "attn_k_conv_proj",
          "self_attn.v_proj.sconv.weight_proj.weight": "attn_v_conv_proj"}
for _gate, _file in (("q", "q"), ("k", "k"), ("v", "v"), ("mlp", "ffn")):
    _PLAIN.update({_gate + "_message_gate.down_proj.weight": "loop_%s_ctrl_down" % _file,
                   _gate + "_message_gate.forget_gate_proj.weight": "loop_%s_ctrl_forget" % _file,
                   _gate + "_message_gate.message_gate_proj.weight": "loop_%s_ctrl_gain" % _file})
# The full-attention static taps: HF's [K, C] with tap 0 on the current token, written as the conv1d layout [C, K] with the current token last.
_STATIC = {"self_attn.k_proj.sconv.static_weight": "attn_k_conv", "self_attn.v_proj.sconv.static_weight": "attn_v_conv"}
# HF's relative position bias table [R, extent], written transposed (ne [R, extent]) as the real file stores it.
_TRANSPOSED = {"self_attn.relative_position_bias.proj": "attn_rel_bias"}


def map_inkling2(config, state):
    """Return `(metadata, tensors)` of the fixture's F32 file, for `spec_decode.write_gguf`; every state entry is written once."""
    tensors, used = [], set()
    for hf, gg in (("model.embed_tokens.weight", "token_embd"), ("model.embed_tokens.embed_norm.weight", "token_embd_norm"),
                   ("model.norm.weight", "output_norm"), ("model.final_input.query", "output_res_query")):
        tensors.append(_f32(gg + ".weight", *state[hf]))
        used.add(hf)
    # The head padded to the full vocabulary as the real files are, with rows no token has; these dominate any hidden row, so reading one puts a padded id on top.
    (rows, width), values = state["lm_head.weight"]
    pad = []
    for j in range(config["vocab_size"] - rows):
        sign = 100.0 if j % 2 == 0 else -100.0
        pad += [sign] * width if j < 2 else [sign if i == (j - 2) // 2 else 0.0 for i in range(width)]
    tensors.append(_f32("output.weight", (config["vocab_size"], width), values + pad))
    used.add("lm_head.weight")
    (loops, four, width), values = state["model.message_input.query"]
    for i in range(loops):
        layer = config["loop_layer_index_start"] + i
        tensors.append(_f32("blk.%d.loop_msg_query.weight" % layer, (four, width), values[i * four * width:(i + 1) * four * width]))
    used.add("model.message_input.query")
    for layer in range(config["num_hidden_layers"]):
        p, out = "model.layers.%d." % layer, "blk.%d." % layer
        for suffix, target in _PLAIN.items():
            if p + suffix in state:
                shape, values = state[p + suffix]
                if suffix.endswith("conv1d.weight"):
                    shape = (shape[0], shape[2])
                tensors.append(_f32(out + target + ".weight", shape, values))
                used.add(p + suffix)
        for suffix, target in _STATIC.items():
            if p + suffix in state:
                (K, C), values = state[p + suffix]
                tensors.append(_f32(out + target + ".weight", (C, K), [values[(K - 1 - t) * C + c] for c in range(C) for t in range(K)]))
                used.add(p + suffix)
        for suffix, target in _TRANSPOSED.items():
            if p + suffix in state:
                (rows, columns), values = state[p + suffix]
                tensors.append(_f32(out + target + ".weight", (columns, rows), [values[r * columns + c] for c in range(columns) for r in range(rows)]))
                used.add(p + suffix)
        tensors.append(_f32(out + "exp_probs_b.bias", *state[p + "mlp.gate.e_score_correction_bias"]))
        used.add(p + "mlp.gate.e_score_correction_bias")
    if used != set(state):
        raise ValueError("unmapped tensors remain: %r" % sorted(set(state) - used))
    names = [t[0] for t in tensors]
    if len(names) != len(set(names)):
        raise ValueError("two parameters map to one GGUF tensor")
    return _metadata(config), tensors


def write_inkling2(path, config, state):
    """Write the fixture's file, replacing `path` only once it is complete."""
    metadata, tensors = map_inkling2(config, state)
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

