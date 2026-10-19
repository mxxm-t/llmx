"""Independent tiny Qwen4Exp HF-state to GGUF mapping and fixture writer.

The input is a plain Python mapping from HF tensor names to ``(shape, values)``.
Values are flattened in HF row-major order, so this module has no torch or
safetensors dependency and writes only the float32 reference fixture format.
"""

import math
import os
import struct
import tempfile
from pathlib import Path

try:
    from .spec_decode import ARRAY, F32, STRING, write_gguf
except ImportError:
    from spec_decode import ARRAY, F32, STRING, write_gguf


GGUF_UINT32 = 4
GGUF_INT32 = 5
GGUF_FLOAT32 = 6
GGUF_BOOL = 7
GGUF_UINT64 = 10
GGUF_ARRAY = ARRAY


def _f32(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Qwen4Exp fixture contains a non-finite weight")
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError("Qwen4Exp fixture weight is outside float32") from error


def _tensor(state, name, shape):
    try:
        got_shape, values = state[name]
    except KeyError as error:
        raise ValueError("missing HF tensor " + name) from error
    got_shape = tuple(got_shape)
    expected = tuple(shape)
    if got_shape != expected:
        raise ValueError("%s has HF shape %s, expected %s" % (name, got_shape, expected))
    values = list(values)
    if len(values) != math.prod(expected):
        raise ValueError("%s has %d values, expected %d" % (name, len(values), math.prod(expected)))
    return values


def _expect(expected, name, shape):
    if name in expected:
        raise ValueError("duplicate expected HF tensor " + name)
    expected[name] = tuple(shape)


def _value(state, expected, name, shape):
    _expect(expected, name, shape)
    return _tensor(state, name, shape)


def _name_and_values(name, shape, values):
    return name, list(shape), F32, struct.pack("<%df" % len(values), *(_f32(v) for v in values))


def _norm(values):
    return [_f32(1.0 + value) for value in values]


def _v_order(k_heads, v_heads):
    if k_heads < 1 or v_heads < 1 or v_heads % k_heads:
        raise ValueError("linear value head count must be a positive multiple of key heads")
    per_key = v_heads // k_heads
    return [(index % k_heads) * per_key + index // k_heads for index in range(v_heads)]


def _head_rows(values, head_width, order):
    return [value for source in order for value in values[source * head_width:(source + 1) * head_width]]


def _head_columns(values, rows, head_width, order):
    cols = len(order) * head_width
    return [value for row in range(rows) for source in order
            for value in values[row * cols + source * head_width:row * cols + (source + 1) * head_width]]


def _metadata(config, state, expected):
    hidden = config["hidden_size"]
    layers = config["num_hidden_layers"]
    layer_types = list(config["layer_types"])
    if len(layer_types) != layers or any(kind not in ("linear_attention", "qwen_sparse_attention") for kind in layer_types):
        raise ValueError("unsupported Qwen4Exp layer_types")
    eos_id = config.get("eos_token_id")
    if isinstance(eos_id, (tuple, list)):
        raise ValueError("Qwen4Exp tiny fixture requires a scalar eos_token_id")
    if not isinstance(eos_id, int) or not 0 <= eos_id < config["vocab_size"]:
        raise ValueError("eos_token_id is outside vocab_size")

    # Scalar values use GGUF's standard unsigned, float and string types.
    meta = {
        "general.architecture": (STRING, "qwen4exp"),
        "qwen4exp.block_count": (GGUF_UINT32, layers),
        "qwen4exp.context_length": (GGUF_UINT32, config["max_position_embeddings"]),
        "qwen4exp.embedding_length": (GGUF_UINT32, hidden),
        "qwen4exp.attention.head_count": (GGUF_UINT32, config["num_attention_heads"]),
        "qwen4exp.attention.head_count_kv": (GGUF_UINT32, config["num_key_value_heads"]),
        "qwen4exp.attention.key_length": (GGUF_UINT32, config["head_dim"]),
        "qwen4exp.attention.value_length": (GGUF_UINT32, config["head_dim"]),
        "qwen4exp.attention.layer_norm_rms_epsilon": (GGUF_FLOAT32, config["rms_norm_eps"]),
        "qwen4exp.rope.freq_base": (GGUF_FLOAT32, config["rope_parameters"]["rope_theta"]),
        "qwen4exp.rope.dimension_count": (GGUF_UINT32, int(config["head_dim"] * config["rope_parameters"]["partial_rotary_factor"])),
        "qwen4exp.rope.dimension_sections": (GGUF_ARRAY, (GGUF_INT32, (list(config["rope_parameters"]["mrope_section"])[:4] + [0] * 4)[:4]),),
        "qwen4exp.expert_count": (GGUF_UINT32, config["num_experts"]),
        "qwen4exp.expert_used_count": (GGUF_UINT32, config["num_experts_per_tok"]),
        "qwen4exp.expert_feed_forward_length": (GGUF_UINT32, config["moe_intermediate_size"]),
        "qwen4exp.expert_shared_feed_forward_length": (GGUF_UINT32, config["shared_expert_intermediate_size"]),
        "qwen4exp.hyper_connection.count": (GGUF_UINT32, config["hc_count"]),
        "qwen4exp.hyper_connection.low_rank": (GGUF_UINT32, config["hc_lowrank"]),
        "qwen4exp.attention.recurrent_layers": (GGUF_ARRAY, (GGUF_BOOL, [kind == "linear_attention" for kind in layer_types])),
        "qwen4exp.attention.compress_ratios": (GGUF_ARRAY, (GGUF_INT32, [0 if kind == "linear_attention" else config["indexer_compress_ratio"] for kind in layer_types])),
        "qwen4exp.attention.indexer.head_count": (GGUF_UINT32, config["indexer_n_heads"]),
        "qwen4exp.attention.indexer.key_length": (GGUF_UINT32, config["indexer_head_dim"]),
        "qwen4exp.attention.indexer.top_k": (GGUF_UINT32, config["indexer_budget"]),
        "qwen4exp.ssm.conv_kernel": (GGUF_UINT32, config["linear_conv_kernel_dim"]),
        "qwen4exp.ssm.inner_size": (GGUF_UINT32, config["linear_num_value_heads"] * config["linear_value_head_dim"]),
        "qwen4exp.ssm.state_size": (GGUF_UINT32, config["linear_key_head_dim"]),
        "qwen4exp.ssm.time_step_rank": (GGUF_UINT32, config["linear_num_value_heads"]),
        "qwen4exp.ssm.group_count": (GGUF_UINT32, config["linear_num_key_heads"]),
        "tokenizer.ggml.eos_token_id": (GGUF_UINT32, eos_id),
        "tokenizer.ggml.tokens": (GGUF_ARRAY, (STRING, _fixture_tokens(config["vocab_size"], eos_id))),
    }
    if config.get("full_attention_interval") is not None:
        metadata["qwen4exp.full_attention_interval"] = (GGUF_UINT32, config["full_attention_interval"])

    ple_layers = [one_based - 1 for one_based in config.get("ple_layer_ids", ())]
    if ple_layers:
        ple = config
        ngram_size = ple["ngram_size"]
        head_count = (ngram_size - 1) * ple["heads_per_ngram"]
        multipliers = _value(state, expected, _ple_name(ple_layers[0], "layer_multipliers"), (ngram_size,))
        offsets = _value(state, expected, _ple_name(ple_layers[0], "ngram_heads_offsets"), (head_count,))
        sizes = _value(state, expected, _ple_name(ple_layers[0], "ngram_heads_vocab_sizes"), (head_count,))
        for name, values in (("layer_multipliers", multipliers), ("head_offsets", offsets), ("head_vocab_sizes", sizes)):
            if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 or v > 0xffffffffffffffff for v in values):
                raise ValueError("PLE %s must contain uint64 values" % name)
        if len(set(offsets)) != len(offsets) or any(v <= 0 for v in sizes):
            raise ValueError("invalid PLE table offsets or vocab sizes")
        if any(value <= 0 or value % 2 == 0 for value in multipliers):
            raise ValueError("PLE layer multipliers must be positive odd uint64 values")
        row_dim = ple["ple_embed_dim"] // head_count
        offsets_and_sizes = sorted(zip(offsets, sizes))
        cursor = 0
        for offset, size in offsets_and_sizes:
            if offset != cursor:
                raise ValueError("PLE head offsets must tile the embedding table without gaps")
            cursor += size
        divisor = ple.get("make_ngram_vocab_size_divisible_by", 1)
        padded_rows = (cursor + divisor - 1) // divisor * divisor
        table_name = _ple_name(ple_layers[0], "ngram_embedding.weight")
        table_shape, _ = state[table_name]
        if len(table_shape) != 2 or tuple(table_shape) != (padded_rows, row_dim):
            raise ValueError("PLE embedding table does not match padded head vocabulary rows")
        meta.update({
            "qwen4exp.ple.layers": (GGUF_ARRAY, (GGUF_UINT32, ple_layers)),
            "qwen4exp.ple.ngram_size": (GGUF_UINT32, ngram_size),
            "qwen4exp.ple.heads_per_ngram": (GGUF_UINT32, ple["heads_per_ngram"]),
            "qwen4exp.ple.conv_kernel": (GGUF_UINT32, ple["ple_conv_kernel_size"]),
            "qwen4exp.ple.eos_token_id": (GGUF_UINT32, eos_id),
            "qwen4exp.embedding_length_per_layer_input": (GGUF_UINT32, row_dim),
            "qwen4exp.ple.layer_multipliers": (GGUF_ARRAY, (GGUF_UINT64, multipliers)),
            "qwen4exp.ple.head_offsets": (GGUF_ARRAY, (GGUF_UINT64, offsets)),
            "qwen4exp.ple.head_vocab_sizes": (GGUF_ARRAY, (GGUF_UINT64, sizes)),
        })
    elif any(".ple." in name for name in state):
        raise ValueError("HF PLE tensors are present but ple_layer_ids is empty")
    return meta


def _fixture_tokens(vocab_size, eos_id):
    # Reuse tests/f32.py's byte vocabulary, capped for the tiny reference.
    try:
        from f32 import build_byte_vocab
        base = build_byte_vocab()
    except ImportError:
        base = ["<byte-%02x>" % index for index in range(256)]
    if len(base) < vocab_size - 1:
        base.extend("<unused-%d>" % index for index in range(len(base), vocab_size - 1))
    tokens = base[:vocab_size - 1]
    if eos_id != vocab_size - 1:
        tokens[eos_id] = "<|endoftext|>"
        tokens.append("<qwen4exp-fixture-eos-slot>")
    else:
        tokens.append("<|endoftext|>")
    return tokens


def _ple_name(layer, suffix):
    return "model.layers.%d.ple.ple_embedding.%s" % (layer, suffix)


def state_from_document(document):
    """Read the JSON-safe state dictionary in a tiny HF reference document.

    This adapter is intentionally only structural: integer hash tensors stay
    Python integers, and validation of names and shapes remains in the mapper.
    """
    if not isinstance(document, dict) or not isinstance(document.get("state_dict"), dict):
        raise ValueError("reference document must contain a state_dict object")
    state = {}
    for name, entry in document["state_dict"].items():
        if not isinstance(name, str) or not isinstance(entry, dict) or set(entry) != {"shape", "values"}:
            raise ValueError("malformed state_dict entry %r" % (name,))
        shape = entry["shape"]
        values = entry["values"]
        if not isinstance(shape, list) or not shape or any(not isinstance(dim, int) or dim <= 0 for dim in shape):
            raise ValueError("%s has an invalid JSON tensor shape" % name)
        if not isinstance(values, list) or any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in values):
            raise ValueError("%s has invalid JSON tensor values" % name)
        if len(values) != math.prod(shape):
            raise ValueError("%s has %d JSON values for shape %s" % (name, len(values), shape))
        state[name] = (tuple(shape), values)
    return state


def _expected_state(config):
    hidden = config["hidden_size"]
    layers = config["num_hidden_layers"]
    hc = config["hc_count"]
    lowrank = config["hc_lowrank"]
    expected = {}
    _expect(expected, "model.embed_tokens.weight", (config["vocab_size"], hidden))
    if not config.get("tie_word_embeddings", False):
        _expect(expected, "lm_head.weight", (config["vocab_size"], hidden))
    _expect(expected, "model.hyper_connection_mixer.hc_norm.weight", (hidden * hc,))
    _expect(expected, "model.hyper_connection_mixer.input_mix_weight_down.weight", (lowrank, hidden * hc))
    _expect(expected, "model.hyper_connection_mixer.input_mix_weight_up.weight", (hidden * hc, lowrank))

    kh = config["linear_num_key_heads"]
    vh = config["linear_num_value_heads"]
    kd = config["linear_key_head_dim"]
    vd = config["linear_value_head_dim"]
    qk = 2 * kh * kd
    ple_layers = set(config.get("ple_layer_ids", ()))
    for index, kind in enumerate(config["layer_types"]):
        prefix = "model.layers.%d." % index
        for path in ("attn_hyper_connection", "mlp_hyper_connection"):
            _expect(expected, prefix + path + ".hc_norm.weight", (hidden * hc,))
            _expect(expected, prefix + path + ".input_mix_weight_down.weight", (lowrank, hidden * hc))
            _expect(expected, prefix + path + ".input_mix_weight_up.weight", (hidden * hc, lowrank))
            _expect(expected, prefix + path + ".block_inject_weight.weight", (hc, hidden * hc))
        if kind == "linear_attention":
            _expect(expected, prefix + "linear_attn.in_proj_qkv.weight", (qk + vh * vd, hidden))
            _expect(expected, prefix + "linear_attn.in_proj_z.weight", (vh * vd, hidden))
            _expect(expected, prefix + "linear_attn.in_proj_b.weight", (vh, hidden))
            _expect(expected, prefix + "linear_attn.in_proj_a.weight", (vh, hidden))
            _expect(expected, prefix + "linear_attn.conv1d.weight", (qk + vh * vd, 1, config["linear_conv_kernel_dim"]))
            _expect(expected, prefix + "linear_attn.dt_bias", (vh,))
            _expect(expected, prefix + "linear_attn.A_log", (vh,))
            _expect(expected, prefix + "linear_attn.norm.weight", (vd,))
            _expect(expected, prefix + "linear_attn.out_proj.weight", (hidden, vh * vd))
            if index + 1 in ple_layers:
                total_ple_heads = (config["ngram_size"] - 1) * config["heads_per_ngram"]
                key_width = hidden * total_ple_heads
                _expect(expected, prefix + "ple.key_proj.weight", (key_width, config["ple_embed_dim"]))
                _expect(expected, prefix + "ple.value_proj.weight", (hidden, config["ple_embed_dim"]))
                for suffix in ("norm_key", "norm_query", "norm_conv"):
                    _expect(expected, prefix + "ple." + suffix + ".weight", (key_width,))
                _expect(expected, prefix + "ple.conv1d.weight", (key_width, 1, config["ple_conv_kernel_size"]))
                _expect(expected, _ple_name(index, "layer_multipliers"), (config["ngram_size"],))
                _expect(expected, _ple_name(index, "ngram_heads_offsets"), (total_ple_heads,))
                _expect(expected, _ple_name(index, "ngram_heads_vocab_sizes"), (total_ple_heads,))
                _expect(expected, _ple_name(index, "ngram_embedding.weight"), (-1, config["ple_embed_dim"] // total_ple_heads))
        elif kind == "qwen_sparse_attention":
            heads = config["num_attention_heads"]
            kv_heads = config["num_key_value_heads"]
            dim = config["head_dim"]
            idx_dim = config["indexer_head_dim"]
            idx_q_dim = config["indexer_n_heads"] * idx_dim
            idx_k_dim = config["indexer_kv_heads"] * idx_dim
            _expect(expected, prefix + "self_attn.q_proj.weight", (2 * heads * dim, hidden))
            _expect(expected, prefix + "self_attn.k_proj.weight", (kv_heads * dim, hidden))
            _expect(expected, prefix + "self_attn.v_proj.weight", (kv_heads * dim, hidden))
            _expect(expected, prefix + "self_attn.o_proj.weight", (hidden, heads * dim))
            for suffix in ("q_norm", "k_norm"):
                _expect(expected, prefix + "self_attn." + suffix + ".weight", (dim,))
            _expect(expected, prefix + "self_attn.indexer.index_qk_proj.weight", (idx_q_dim + idx_k_dim, hidden))
            for suffix in ("q_layernorm", "k_layernorm"):
                _expect(expected, prefix + "self_attn.indexer." + suffix + ".weight", (idx_dim,))
        else:
            raise ValueError("unsupported Qwen4Exp layer kind " + str(kind))

        experts = config["num_experts"]
        ff = config["moe_intermediate_size"]
        shared = config["shared_expert_intermediate_size"]
        _expect(expected, prefix + "mlp.gate.weight", (experts, hidden))
        _expect(expected, prefix + "mlp.experts.gate_up_proj", (experts, 2 * ff, hidden))
        _expect(expected, prefix + "mlp.experts.down_proj", (experts, hidden, ff))
        _expect(expected, prefix + "mlp.shared_expert.gate_proj.weight", (shared, hidden))
        _expect(expected, prefix + "mlp.shared_expert.up_proj.weight", (shared, hidden))
        _expect(expected, prefix + "mlp.shared_expert.down_proj.weight", (hidden, shared))
        _expect(expected, prefix + "mlp.shared_expert_gate.weight", (1, hidden))
    return expected


def map_qwen4exp(config, state):
    """Map complete HF Qwen4ExpForCausalLM state to independent GGUF parts.

    Return `(metadata, tensors)` suitable for `spec_decode.write_gguf`.
    Every HF key must be accounted for; no unrecognized or omitted tensor is
    accepted. The output tensor layout is the converter's GGUF input-first
    dimension order and all floating tensors are stored as F32.
    """
    if not isinstance(config, dict) or not isinstance(state, dict):
        raise TypeError("config and state must be dictionaries")
    # Reference documents store the exact constructor kwargs, which omit this
    # class-level discriminator; accept it only when supplied by a full HF config.
    if config.get("model_type", "qwen4_exp_text") != "qwen4_exp_text":
        raise ValueError("expected the pinned Qwen4Exp text config")
    if config.get("vision_config") is not None or config.get("mtp_num_hidden_layers", 0):
        raise ValueError("the independent Qwen4Exp fixture writer supports text-only CausalLM without MTP")
    ple_ids = tuple(config.get("ple_layer_ids", ()))
    if len(ple_ids) > 1:
        raise ValueError("the tiny Qwen4Exp fixture supports exactly one PLE table layer")
    if config.get("tie_word_embeddings", False):
        raise ValueError("the tiny Qwen4Exp fixture requires an independent output head")
    if config.get("hidden_act", "silu") != "silu":
        raise ValueError("the tiny Qwen4Exp fixture supports SiLU expert activation")
    if (config.get("output_gate_type") or config.get("hidden_act", "silu")) != "sigmoid":
        raise ValueError("the qwen4exp GGUF fixture requires sigmoid GDN output gating")
    if config.get("attention_bias", False):
        raise ValueError("the tiny Qwen4Exp fixture does not accept attention bias")
    linear_key_heads = config.get("linear_num_key_heads", 0)
    linear_value_heads = config.get("linear_num_value_heads", 0)
    if linear_key_heads < 1 or linear_value_heads < 1 or linear_value_heads % linear_key_heads:
        raise ValueError("linear value heads must be divisible by key heads")
    ratio = config.get("indexer_compress_ratio", 0)
    budget = config.get("indexer_budget", 0)
    if ratio < 1 or budget < 1 or config.get("indexer_kv_heads") != 1 or budget % ratio:
        raise ValueError("QSA indexer requires one KV head and a divisible budget")
    if any(not isinstance(layer, int) or not 1 <= layer <= config["num_hidden_layers"]
           or config["layer_types"][layer - 1] != "linear_attention" for layer in ple_ids):
        raise ValueError("ple_layer_ids must name an in-range one-based linear-attention layer")
    expected = _expected_state(config)
    missing = sorted(set(expected) - set(state))
    extra = sorted(set(state) - set(expected))
    if missing or extra:
        raise ValueError("HF state keys differ; missing=%s extra=%s" % (missing, extra))
    for name, shape in expected.items():
        actual, values = state[name]
        if shape[0] == -1:
            if len(actual) != 2 or actual[1] != shape[1] or math.prod(actual) != len(values):
                raise ValueError("%s has invalid HF table shape %s" % (name, tuple(actual)))
        else:
            _tensor(state, name, shape)

    metadata = _metadata(config, state, {})
    tensors = []
    used = set()

    def add(hf_name, gguf_name, gguf_shape, values=None, mark_used=True):
        if mark_used and hf_name in used:
            raise ValueError("HF tensor was mapped twice: " + hf_name)
        used.add(hf_name)
        if values is None:
            values = state[hf_name][1]
        tensors.append(_name_and_values(gguf_name, gguf_shape, values))

    def add_direct(hf_name, gguf_name):
        shape = state[hf_name][0]
        add(hf_name, gguf_name, tuple(reversed(shape)))

    def add_norm(hf_name, gguf_name):
        add(hf_name, gguf_name, tuple(reversed(state[hf_name][0])), _norm(state[hf_name][1]))

    add_direct("model.embed_tokens.weight", "token_embd.weight")
    if "lm_head.weight" in state:
        add_direct("lm_head.weight", "output.weight")
    hc_prefix = "model.hyper_connection_mixer."
    for field, target in (("hc_norm", "output_hc_norm"),
                          ("input_mix_weight_down", "output_hc_down"),
                          ("input_mix_weight_up", "output_hc_up")):
        source = hc_prefix + field + ".weight"
        (add_norm if field == "hc_norm" else add_direct)(source, target + ".weight")

    kh = config["linear_num_key_heads"]
    vh = config["linear_num_value_heads"]
    kd = config["linear_key_head_dim"]
    vd = config["linear_value_head_dim"]
    order = _v_order(kh, vh)
    qk_dim = 2 * kh * kd
    for index, kind in enumerate(config["layer_types"]):
        prefix = "model.layers.%d." % index
        block = "blk.%d." % index
        for source, stem in (("attn_hyper_connection", "hc_attn"), ("mlp_hyper_connection", "hc_ffn")):
            add_norm(prefix + source + ".hc_norm.weight", block + stem + "_norm.weight")
            add_direct(prefix + source + ".input_mix_weight_down.weight", block + stem + "_down.weight")
            add_direct(prefix + source + ".input_mix_weight_up.weight", block + stem + "_up.weight")
            add_direct(prefix + source + ".block_inject_weight.weight", block + stem + "_inject.weight")
        if kind == "linear_attention":
            names = {
                "in_proj_qkv.weight": "attn_qkv.weight",
                "in_proj_z.weight": "attn_gate.weight",
                "in_proj_b.weight": "ssm_beta.weight",
                "in_proj_a.weight": "ssm_alpha.weight",
                "conv1d.weight": "ssm_conv1d.weight",
                "dt_bias": "ssm_dt.bias",
                "A_log": "ssm_a",
                "norm.weight": "ssm_norm.weight",
                "out_proj.weight": "ssm_out.weight",
            }
            p = prefix + "linear_attn."
            qkv_name = p + "in_proj_qkv.weight"
            qkv = list(state[qkv_name][1])
            hidden = config["hidden_size"]
            qk_count = qk_dim * hidden
            qkv = qkv[:qk_count] + _head_rows(qkv[qk_count:], vd * hidden, order)
            add(qkv_name, block + names["in_proj_qkv.weight"], tuple(reversed(state[qkv_name][0])), qkv)
            for source, target in (("in_proj_z.weight", "in_proj_z.weight"),
                                   ("in_proj_b.weight", "in_proj_b.weight"),
                                   ("in_proj_a.weight", "in_proj_a.weight")):
                hf_name = p + source
                width = vd if source == "in_proj_z.weight" else 1
                add(hf_name, block + names[target], tuple(reversed(state[hf_name][0])), _head_rows(state[hf_name][1], width * hidden, order))
            conv_name = p + "conv1d.weight"
            conv_shape = state[conv_name][0]
            conv = list(state[conv_name][1])
            taps = conv_shape[2]
            channels = conv_shape[0]
            channel_values = [conv[c * taps:(c + 1) * taps] for c in range(channels)]
            reordered_channels = channel_values[:qk_dim] + [channel_values[qk_dim + source * vd + offset]
                for source in order for offset in range(vd)]
            conv = [v for row in reordered_channels for v in row]
            add(conv_name, block + names["conv1d.weight"], (taps, channels), conv)
            for suffix in ("dt_bias", "A_log"):
                hf_name = p + suffix
                values = list(state[hf_name][1])
                values = [values[source] for source in order]
                if suffix == "A_log":
                    values = [_f32(-math.exp(float(v))) for v in values]
                add(hf_name, block + names[suffix], tuple(reversed(state[hf_name][0])), values)
            hf_name = p + "norm.weight"
            add(hf_name, block + names["norm.weight"], tuple(reversed(state[hf_name][0])))
            hf_name = p + "out_proj.weight"
            out_shape = state[hf_name][0]
            add(hf_name, block + names["out_proj.weight"], tuple(reversed(out_shape)),
                _head_columns(state[hf_name][1], out_shape[0], vd, order))
            if index + 1 in config.get("ple_layer_ids", ()):
                ple_prefix = prefix + "ple."
                for source, target in (("key_proj", "ple_key"), ("value_proj", "ple_value")):
                    hf_name = ple_prefix + source + ".weight"
                    add_direct(hf_name, block + target + ".weight")
                for source, target in (("norm_key", "ple_norm_key"), ("norm_query", "ple_norm_query"), ("norm_conv", "ple_norm_conv")):
                    hf_name = ple_prefix + source + ".weight"
                    add_norm(hf_name, block + target + ".weight")
                hf_name = ple_prefix + "conv1d.weight"
                shape = state[hf_name][0]
                add(hf_name, block + "ple_conv1d.weight", (shape[2], shape[0]), state[hf_name][1])
                table_hf = _ple_name(index, "ngram_embedding.weight")
                add_direct(table_hf, "per_layer_token_embd.weight")
                for constant in ("layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes"):
                    used.add(_ple_name(index, constant))
        else:
            p = prefix + "self_attn."
            for source, target in (("q_proj", "attn_q"), ("k_proj", "attn_k"), ("v_proj", "attn_v"), ("o_proj", "attn_output")):
                add_direct(p + source + ".weight", block + target + ".weight")
            for source, target in (("q_norm", "attn_q_norm"), ("k_norm", "attn_k_norm")):
                add_norm(p + source + ".weight", block + target + ".weight")
            indexer = p + "indexer."
            fused = indexer + "index_qk_proj.weight"
            fused_shape = state[fused][0]
            qrows = config["indexer_n_heads"] * config["indexer_head_dim"]
            values = list(state[fused][1])
            row_size = fused_shape[1]
            add(fused, block + "indexer.q_proj.weight", (row_size, qrows), values[:qrows * row_size])
            add(fused, block + "indexer.k_proj.weight", (row_size, fused_shape[0] - qrows), values[qrows * row_size:], mark_used=False)
            for source, target in (("q_layernorm", "q_norm"), ("k_layernorm", "k_norm")):
                add_norm(indexer + source + ".weight", block + "indexer." + target + ".weight")

        mlp = prefix + "mlp."
        add_direct(mlp + "gate.weight", block + "ffn_gate_inp.weight")
        experts, rows, width = state[mlp + "experts.gate_up_proj"][0]
        fused_name = mlp + "experts.gate_up_proj"
        fused_values = list(state[fused_name][1])
        half = rows // 2 * width
        for which, target in ((0, "ffn_gate_exps.weight"), (1, "ffn_up_exps.weight")):
            split = [value for expert in range(experts)
                     for value in fused_values[expert * rows * width + which * half:expert * rows * width + (which + 1) * half]]
            add(fused_name, block + target, (width, rows // 2, experts), split, mark_used=(which == 0))
        down = mlp + "experts.down_proj"
        add_direct(down, block + "ffn_down_exps.weight")
        for source, target in (("shared_expert.gate_proj.weight", "ffn_gate_shexp.weight"),
                               ("shared_expert.up_proj.weight", "ffn_up_shexp.weight"),
                               ("shared_expert.down_proj.weight", "ffn_down_shexp.weight")):
            add_direct(mlp + source, block + target)
        gate = mlp + "shared_expert_gate.weight"
        add(gate, block + "ffn_gate_inp_shexp.weight", (hidden,), state[gate][1])

    if used != set(state):
        raise ValueError("unmapped HF tensors remain: " + repr(sorted(set(state) - used)))
    names = [tensor[0] for tensor in tensors]
    if len(names) != len(set(names)):
        raise ValueError("multiple HF tensors map to the same GGUF tensor")
    return metadata, tensors


def write_qwen4exp(path, config, state):
    """Write an independent F32 text-only Qwen4Exp GGUF fixture."""
    metadata, tensors = map_qwen4exp(config, state)
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
