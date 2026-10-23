"""Standard-library checks for the independent Qwen4Exp GGUF fixture writer."""

import math
import os
import hashlib
import json
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spec_decode
import qwen4exp
import common
import server as server_test


CONFIG = {
    "model_type": "qwen4_exp_text",
    "hidden_size": 4,
    "vocab_size": 4,
    "num_hidden_layers": 2,
    "layer_types": ["linear_attention", "qwen_sparse_attention"],
    "hc_count": 2,
    "hc_lowrank": 1,
    "output_gate_type": "sigmoid",
    "tie_word_embeddings": False,
    "max_position_embeddings": 16,
    "num_attention_heads": 1,
    "num_key_value_heads": 1,
    "head_dim": 4,
    "rms_norm_eps": 1e-6,
    "rope_parameters": {"rope_theta": 10000.0, "partial_rotary_factor": 0.5, "mrope_section": [1, 0]},
    "num_experts": 2,
    "num_experts_per_tok": 1,
    "moe_intermediate_size": 2,
    "shared_expert_intermediate_size": 3,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 4,
    "linear_key_head_dim": 1,
    "linear_value_head_dim": 1,
    "linear_conv_kernel_dim": 4,
    "indexer_n_heads": 1,
    "indexer_kv_heads": 1,
    "indexer_head_dim": 2,
    "indexer_budget": 4,
    "indexer_compress_ratio": 2,
    "ple_layer_ids": [1],
    "ngram_size": 3,
    "heads_per_ngram": 1,
    "ple_embed_dim": 4,
    "ple_conv_kernel_size": 4,
    "ngram_vocab_size_base": 3,
    "make_ngram_vocab_size_divisible_by": 4,
    "split_ngram_parts": 1,
    "eos_token_id": 3,
    "bos_token_id": 1,
    "pad_token_id": 0,
}
GGUF_BOOL = 7
GGUF_UINT32 = 4
GGUF_INT32 = 5
GGUF_UINT64 = 10


def state_fixture():
    state = {}
    seed = 1

    def add(name, shape, values=None):
        nonlocal seed
        count = math.prod(shape)
        if values is None:
            values = [seed + i / 16 for i in range(count)]
            seed += 10
        state[name] = (tuple(shape), list(values))

    hidden, streams, lowrank = CONFIG["hidden_size"], CONFIG["hc_count"], CONFIG["hc_lowrank"]
    vocab = CONFIG["vocab_size"]
    add("model.embed_tokens.weight", (vocab, hidden))
    add("lm_head.weight", (vocab, hidden))
    add("model.hyper_connection_mixer.hc_norm.weight", (hidden * streams,), [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0])
    add("model.hyper_connection_mixer.input_mix_weight_down.weight", (lowrank, hidden * streams))
    add("model.hyper_connection_mixer.input_mix_weight_up.weight", (hidden * streams, lowrank))

    for layer, kind in enumerate(CONFIG["layer_types"]):
        p = "model.layers.%d." % layer
        for branch in ("attn", "mlp"):
            name = p + branch + "_hyper_connection."
            add(name + "hc_norm.weight", (hidden * streams,), [0.125] * (hidden * streams))
            add(name + "input_mix_weight_down.weight", (lowrank, hidden * streams))
            add(name + "input_mix_weight_up.weight", (hidden * streams, lowrank))
            add(name + "block_inject_weight.weight", (streams, hidden * streams))
        if kind == "linear_attention":
            p += "linear_attn."
            add(p + "in_proj_qkv.weight", (8, hidden), list(range(32)))
            add(p + "in_proj_z.weight", (4, hidden), list(range(16)))
            add(p + "in_proj_b.weight", (4, hidden), list(range(16, 32)))
            add(p + "in_proj_a.weight", (4, hidden), list(range(32, 48)))
            add(p + "conv1d.weight", (8, 1, 4), list(range(32)))
            add(p + "dt_bias", (4,), [10, 20, 30, 40])
            add(p + "A_log", (4,), [0.0, math.log(2.0), math.log(3.0), math.log(4.0)])
            add(p + "norm.weight", (1,), [1.25])
            add(p + "out_proj.weight", (hidden, 4), list(range(16)))
            ple = "model.layers.%d.ple." % layer
            add(ple + "key_proj.weight", (8, 4))
            add(ple + "value_proj.weight", (4, 4))
            for norm in ("norm_key", "norm_query", "norm_conv"):
                add(ple + norm + ".weight", (8,), [0.25] * 8)
            add(ple + "conv1d.weight", (8, 1, 4), list(range(32, 64)))
            e = "model.layers.%d.ple.ple_embedding." % layer
            add(e + "layer_multipliers", (3,), [4294967311, 4294967357, 4294967371])
            add(e + "ngram_heads_offsets", (2,), [0, 3])
            add(e + "ngram_heads_vocab_sizes", (2,), [3, 5])
            add(e + "ngram_embedding.weight", (8, 2), list(range(16)))
        else:
            p += "self_attn."
            add(p + "q_proj.weight", (8, hidden))
            add(p + "k_proj.weight", (4, hidden))
            add(p + "v_proj.weight", (4, hidden))
            add(p + "o_proj.weight", (hidden, 4))
            add(p + "q_norm.weight", (4,), [0.25] * 4)
            add(p + "k_norm.weight", (4,), [0.25] * 4)
            add(p + "indexer.index_qk_proj.weight", (4, hidden), list(range(16)))
            add(p + "indexer.q_layernorm.weight", (2,), [0.25, 0.5])
            add(p + "indexer.k_layernorm.weight", (2,), [0.75, 1.0])
        mlp = "model.layers.%d.mlp." % layer
        add(mlp + "gate.weight", (2, hidden))
        add(mlp + "experts.gate_up_proj", (2, 4, hidden), list(range(32)))
        add(mlp + "experts.down_proj", (2, hidden, 2))
        add(mlp + "shared_expert.gate_proj.weight", (3, hidden))
        add(mlp + "shared_expert.up_proj.weight", (3, hidden))
        add(mlp + "shared_expert.down_proj.weight", (hidden, 3))
        add(mlp + "shared_expert_gate.weight", (1, hidden))
    return state


def _assert_inverse_source(test, config, state, metadata, tensors):
    by_name = {name: (shape, type_id, payload) for name, shape, type_id, payload in tensors}

    def decoded(name):
        shape, type_id, payload = by_name[name]
        return spec_decode.decode(type_id, payload, math.prod(shape))

    def put(name, values):
        restored[name] = list(values)

    def unorder(values, width, order):
        chunks = [None] * len(order)
        for output_index, source_index in enumerate(order):
            chunks[source_index] = values[output_index * width:(output_index + 1) * width]
        return [value for chunk in chunks for value in chunk]

    restored = {}
    hidden = config["hidden_size"]
    key_heads = config["linear_num_key_heads"]
    value_heads = config["linear_num_value_heads"]
    key_dim = config["linear_key_head_dim"]
    value_dim = config["linear_value_head_dim"]
    order = qwen4exp._v_order(key_heads, value_heads)
    qk_count = 2 * key_heads * key_dim * hidden
    put("model.embed_tokens.weight", decoded("token_embd.weight"))
    put("lm_head.weight", decoded("output.weight"))
    put("model.hyper_connection_mixer.hc_norm.weight", [x - 1.0 for x in decoded("output_hc_norm.weight")])
    put("model.hyper_connection_mixer.input_mix_weight_down.weight", decoded("output_hc_down.weight"))
    put("model.hyper_connection_mixer.input_mix_weight_up.weight", decoded("output_hc_up.weight"))

    for index, kind in enumerate(config["layer_types"]):
        source = "model.layers.%d." % index
        block = "blk.%d." % index
        for branch, stem in (("attn", "hc_attn"), ("mlp", "hc_ffn")):
            prefix = source + branch + "_hyper_connection."
            put(prefix + "hc_norm.weight", [x - 1.0 for x in decoded(block + stem + "_norm.weight")])
            put(prefix + "input_mix_weight_down.weight", decoded(block + stem + "_down.weight"))
            put(prefix + "input_mix_weight_up.weight", decoded(block + stem + "_up.weight"))
            put(prefix + "block_inject_weight.weight", decoded(block + stem + "_inject.weight"))
        if kind == "linear_attention":
            raw = decoded(block + "attn_qkv.weight")
            put(source + "linear_attn.in_proj_qkv.weight",
                raw[:qk_count] + unorder(raw[qk_count:], value_dim * hidden, order))
            for suffix, target, width in (("in_proj_z.weight", "attn_gate.weight", value_dim * hidden),
                                         ("in_proj_b.weight", "ssm_beta.weight", hidden),
                                         ("in_proj_a.weight", "ssm_alpha.weight", hidden)):
                put(source + "linear_attn." + suffix, unorder(decoded(block + target), width, order))
            conv = decoded(block + "ssm_conv1d.weight")
            taps = config["linear_conv_kernel_dim"]
            rows = [conv[i * taps:(i + 1) * taps] for i in range(len(conv) // taps)]
            qk_channels = 2 * key_heads * key_dim
            original = rows[:qk_channels] + [None] * (value_heads * value_dim)
            for output_head, source_head in enumerate(order):
                for offset in range(value_dim):
                    original[qk_channels + source_head * value_dim + offset] = rows[qk_channels + output_head * value_dim + offset]
            put(source + "linear_attn.conv1d.weight", [value for row in original for value in row])
            put(source + "linear_attn.dt_bias", unorder(decoded(block + "ssm_dt.bias"), 1, order))
            put(source + "linear_attn.A_log", [math.log(-value) for value in unorder(decoded(block + "ssm_a"), 1, order)])
            put(source + "linear_attn.norm.weight", decoded(block + "ssm_norm.weight"))
            output = decoded(block + "ssm_out.weight")
            original_output = []
            for row in range(hidden):
                original_output.extend(unorder(output[row * value_heads * value_dim:(row + 1) * value_heads * value_dim], value_dim, order))
            put(source + "linear_attn.out_proj.weight", original_output)
            ple = source + "ple."
            for source_name, target in (("key_proj.weight", "ple_key.weight"), ("value_proj.weight", "ple_value.weight")):
                put(ple + source_name, decoded(block + target))
            for norm in ("norm_key", "norm_query", "norm_conv"):
                put(ple + norm + ".weight", [x - 1.0 for x in decoded(block + "ple_" + norm + ".weight")])
            put(ple + "conv1d.weight", decoded(block + "ple_conv1d.weight"))
            put(ple + "ple_embedding.ngram_embedding.weight", decoded("per_layer_token_embd.weight"))
            for source_name, metadata_name in (("layer_multipliers", "layer_multipliers"),
                                               ("ngram_heads_offsets", "head_offsets"),
                                               ("ngram_heads_vocab_sizes", "head_vocab_sizes")):
                put(ple + "ple_embedding." + source_name, metadata["qwen4exp.ple." + metadata_name][1][1])
        else:
            for source_name, target in (("q_proj", "attn_q"), ("k_proj", "attn_k"),
                                        ("v_proj", "attn_v"), ("o_proj", "attn_output")):
                put(source + "self_attn." + source_name + ".weight", decoded(block + target + ".weight"))
            for norm in ("q_norm", "k_norm"):
                put(source + "self_attn." + norm + ".weight", [x - 1.0 for x in decoded(block + "attn_" + norm + ".weight")])
            put(source + "self_attn.indexer.index_qk_proj.weight",
                decoded(block + "indexer.q_proj.weight") + decoded(block + "indexer.k_proj.weight"))
            for source_name, target in (("q_layernorm", "q_norm"), ("k_layernorm", "k_norm")):
                put(source + "self_attn.indexer." + source_name + ".weight",
                    [x - 1.0 for x in decoded(block + "indexer." + target + ".weight")])

        mlp = source + "mlp."
        put(mlp + "gate.weight", decoded(block + "ffn_gate_inp.weight"))
        gate = decoded(block + "ffn_gate_exps.weight")
        up = decoded(block + "ffn_up_exps.weight")
        experts = config["num_experts"]
        ff_width = config["moe_intermediate_size"] * hidden
        fused = []
        for expert in range(experts):
            fused.extend(gate[expert * ff_width:(expert + 1) * ff_width])
            fused.extend(up[expert * ff_width:(expert + 1) * ff_width])
        put(mlp + "experts.gate_up_proj", fused)
        put(mlp + "experts.down_proj", decoded(block + "ffn_down_exps.weight"))
        for source_name, target in (("shared_expert.gate_proj.weight", "ffn_gate_shexp.weight"),
                                    ("shared_expert.up_proj.weight", "ffn_up_shexp.weight"),
                                    ("shared_expert.down_proj.weight", "ffn_down_shexp.weight")):
            put(mlp + source_name, decoded(block + target))
        put(mlp + "shared_expert_gate.weight", decoded(block + "ffn_gate_inp_shexp.weight"))

    test.assertEqual(set(restored), set(state))
    for name, (_, source_values) in state.items():
        error = max((abs(float(a) - float(b)) for a, b in zip(restored[name], source_values)), default=0.0)
        if name.endswith("linear_attn.A_log"):
            test.assertLessEqual(error, 2e-6, name)
        elif "norm" in name and name.endswith(".weight"):
            test.assertLessEqual(error, 2e-7, name)
        else:
            test.assertEqual(restored[name], source_values, name)


def _canonical_reference_hash(state):
    digest = hashlib.sha256()
    integer_suffixes = ("layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes")
    for name in sorted(state):
        shape, values = state[name]
        dtype = "torch.int64" if name.endswith(integer_suffixes) else "torch.float32"
        header = json.dumps([name, dtype, list(shape)], separators=(",", ":")).encode("utf-8")
        digest.update(struct.pack("<Q", len(header)))
        digest.update(header)
        pack = "<%dq" % len(values) if dtype == "torch.int64" else "<%df" % len(values)
        digest.update(struct.pack(pack, *values))
    return digest.hexdigest()


class Qwen4ExpFixtureTests(unittest.TestCase):
    require_native = False

    def _check_server_hf_reply(self, reply, logits, label, prompt_tokens):
        top = sorted(range(len(logits)), key=lambda i: (-logits[i], i))
        shift = max(logits)
        log_z = shift + math.log(math.fsum(math.exp(value - shift) for value in logits))
        expected_logprob = logits[top[0]] - log_z
        self.assertEqual(reply["ids"], [top[0]], label)
        self.assertEqual(reply["prompt_tokens"], prompt_tokens, label)
        self.assertEqual(len(reply["logprobs"]), 1, label)
        self.assertLess(abs(reply["logprobs"][0] - expected_logprob), 2e-5,
                        (label, reply, expected_logprob))
        observed = reply["top_logprobs"][0]
        self.assertEqual([entry["id"] for entry in observed], top[:5],
                         (label, observed, top[:5]))
        for entry in observed:
            wanted = logits[entry["id"]] - log_z
            self.assertLess(abs(entry["logprob"] - wanted), 2e-5,
                            (label, entry, wanted))

    def setUp(self):
        self.state = state_fixture()
        self.metadata, self.tensors = qwen4exp.map_qwen4exp(CONFIG, self.state)
        self.by_name = {name: (shape, type_id, payload) for name, shape, type_id, payload in self.tensors}

    def decoded(self, name):
        shape, type_id, payload = self.by_name[name]
        return spec_decode.decode(type_id, payload, math.prod(shape))

    def test_every_weight_roundtrips_to_its_hf_source(self):
        restored = {}

        def put(hf_name, values, subtract_one=False):
            restored[hf_name] = [value - 1.0 for value in values] if subtract_one else list(values)

        def unorder(values, width, order):
            source = [None] * len(order)
            for output_index, source_index in enumerate(order):
                source[source_index] = values[output_index * width:(output_index + 1) * width]
            return [value for chunk in source for value in chunk]

        put("model.embed_tokens.weight", self.decoded("token_embd.weight"))
        put("lm_head.weight", self.decoded("output.weight"))
        put("model.hyper_connection_mixer.hc_norm.weight", self.decoded("output_hc_norm.weight"), True)
        put("model.hyper_connection_mixer.input_mix_weight_down.weight", self.decoded("output_hc_down.weight"))
        put("model.hyper_connection_mixer.input_mix_weight_up.weight", self.decoded("output_hc_up.weight"))
        for index, kind in enumerate(CONFIG["layer_types"]):
            source = "model.layers.%d." % index
            block = "blk.%d." % index
            for branch in ("attn", "mlp"):
                from_name = source + branch + "_hyper_connection."
                to_name = block + ("hc_attn" if branch == "attn" else "hc_ffn")
                put(from_name + "hc_norm.weight", self.decoded(to_name + "_norm.weight"), True)
                put(from_name + "input_mix_weight_down.weight", self.decoded(to_name + "_down.weight"))
                put(from_name + "input_mix_weight_up.weight", self.decoded(to_name + "_up.weight"))
                put(from_name + "block_inject_weight.weight", self.decoded(to_name + "_inject.weight"))
            if kind == "linear_attention":
                order = [0, 2, 1, 3]
                raw = self.decoded(block + "attn_qkv.weight")
                qk_count = 4 * CONFIG["hidden_size"]
                v_width = CONFIG["linear_value_head_dim"] * CONFIG["hidden_size"]
                put(source + "linear_attn.in_proj_qkv.weight", raw[:qk_count] + unorder(raw[qk_count:], v_width, order))
                for suffix, width in (("in_proj_z.weight", 1), ("in_proj_b.weight", 1), ("in_proj_a.weight", 1)):
                    name = source + "linear_attn." + suffix
                    put(name, unorder(self.decoded(block + ("attn_gate.weight" if suffix == "in_proj_z.weight" else "ssm_beta.weight" if suffix == "in_proj_b.weight" else "ssm_alpha.weight")), width * CONFIG["hidden_size"], order))
                conv = self.decoded(block + "ssm_conv1d.weight")
                taps = CONFIG["linear_conv_kernel_dim"]
                conv_rows = [conv[i * taps:(i + 1) * taps] for i in range(8)]
                original_conv_rows = conv_rows[:4] + [None] * 4
                for output_index, source_head in enumerate(order):
                    original_conv_rows[4 + source_head] = conv_rows[4 + output_index]
                put(source + "linear_attn.conv1d.weight", [value for row in original_conv_rows for value in row])
                put(source + "linear_attn.dt_bias", unorder(self.decoded(block + "ssm_dt.bias"), 1, order))
                put(source + "linear_attn.A_log", [math.log(-value) for value in unorder(self.decoded(block + "ssm_a"), 1, order)])
                put(source + "linear_attn.norm.weight", self.decoded(block + "ssm_norm.weight"))
                output = self.decoded(block + "ssm_out.weight")
                original_output = []
                for row in range(CONFIG["hidden_size"]):
                    output_row = output[row * 4:(row + 1) * 4]
                    original_row = [None] * 4
                    for output_head, source_head in enumerate(order):
                        original_row[source_head] = output_row[output_head]
                    original_output.extend(original_row)
                put(source + "linear_attn.out_proj.weight", original_output)
                ple = source + "ple."
                for hf_suffix, gguf_suffix in (("key_proj.weight", "ple_key.weight"), ("value_proj.weight", "ple_value.weight")):
                    put(ple + hf_suffix, self.decoded(block + gguf_suffix))
                for norm in ("norm_key", "norm_query", "norm_conv"):
                    put(ple + norm + ".weight", self.decoded(block + "ple_" + norm + ".weight"), True)
                put(ple + "conv1d.weight", self.decoded(block + "ple_conv1d.weight"))
                put(ple + "ple_embedding.ngram_embedding.weight", self.decoded("per_layer_token_embd.weight"))
                for source_name, metadata_name in (("layer_multipliers", "layer_multipliers"),
                                                    ("ngram_heads_offsets", "head_offsets"),
                                                    ("ngram_heads_vocab_sizes", "head_vocab_sizes")):
                    _, values = self.metadata["qwen4exp.ple." + metadata_name][1]
                    restored[ple + "ple_embedding." + source_name] = list(values)
            else:
                for source_suffix, target_suffix in (("q_proj", "attn_q"), ("k_proj", "attn_k"),
                                                     ("v_proj", "attn_v"), ("o_proj", "attn_output")):
                    put(source + "self_attn." + source_suffix + ".weight", self.decoded(block + target_suffix + ".weight"))
                for norm in ("q_norm", "k_norm"):
                    put(source + "self_attn." + norm + ".weight", self.decoded(block + "attn_" + norm + ".weight"), True)
                q = self.decoded(block + "indexer.q_proj.weight")
                k = self.decoded(block + "indexer.k_proj.weight")
                put(source + "self_attn.indexer.index_qk_proj.weight", q + k)
                for source_norm, target_norm in (("q_layernorm", "q_norm"), ("k_layernorm", "k_norm")):
                    put(source + "self_attn.indexer." + source_norm + ".weight",
                        self.decoded(block + "indexer." + target_norm + ".weight"), True)
            mlp = source + "mlp."
            put(mlp + "gate.weight", self.decoded(block + "ffn_gate_inp.weight"))
            gate = self.decoded(block + "ffn_gate_exps.weight")
            up = self.decoded(block + "ffn_up_exps.weight")
            fused = []
            gate_rows = 2 * CONFIG["hidden_size"]
            for expert in range(2):
                fused.extend(gate[expert * gate_rows:(expert + 1) * gate_rows])
                fused.extend(up[expert * gate_rows:(expert + 1) * gate_rows])
            put(mlp + "experts.gate_up_proj", fused)
            put(mlp + "experts.down_proj", self.decoded(block + "ffn_down_exps.weight"))
            for source_name, target_name in (("shared_expert.gate_proj.weight", "ffn_gate_shexp.weight"),
                                             ("shared_expert.up_proj.weight", "ffn_up_shexp.weight"),
                                             ("shared_expert.down_proj.weight", "ffn_down_shexp.weight")):
                put(mlp + source_name, self.decoded(block + target_name))
            put(mlp + "shared_expert_gate.weight", self.decoded(block + "ffn_gate_inp_shexp.weight"))

        self.assertEqual(set(restored), set(self.state))
        for name, (_, source_values) in self.state.items():
            if name.endswith("linear_attn.A_log"):
                error = max(abs(a - b) for a, b in zip(restored[name], source_values))
                self.assertLessEqual(error, 2e-6, name)
            else:
                self.assertEqual(restored[name], source_values, name)

    def test_metadata_matches_text_graph_and_uint64_hash_contract(self):
        self.assertEqual(self.metadata["general.architecture"], (spec_decode.STRING, "qwen4exp"))
        self.assertEqual(self.metadata["qwen4exp.attention.recurrent_layers"], (spec_decode.ARRAY, (GGUF_BOOL, [True, False])))
        self.assertEqual(self.metadata["qwen4exp.attention.compress_ratios"], (spec_decode.ARRAY, (GGUF_INT32, [0, 2])))
        self.assertEqual(self.metadata["qwen4exp.ple.layers"], (spec_decode.ARRAY, (GGUF_UINT32, [0])))
        for name, expected in (("layer_multipliers", [4294967311, 4294967357, 4294967371]),
                               ("head_offsets", [0, 3]), ("head_vocab_sizes", [3, 5])):
            actual = self.metadata["qwen4exp.ple." + name]
            self.assertEqual(actual, (spec_decode.ARRAY, (GGUF_UINT64, expected)))
        self.assertEqual(self.metadata["qwen4exp.embedding_length_per_layer_input"], (GGUF_UINT32, 2))
        self.assertEqual(self.metadata["qwen4exp.rope.dimension_sections"], (spec_decode.ARRAY, (GGUF_INT32, [1, 0, 0, 0])))
        self.assertEqual(len(self.metadata["tokenizer.ggml.tokens"][1][1]), CONFIG["vocab_size"])
        metadata, _ = qwen4exp.map_qwen4exp(dict(CONFIG, full_attention_interval=None), self.state)
        self.assertNotIn("qwen4exp.full_attention_interval", metadata)

    def test_ple_metadata_converts_one_based_layer_id_to_zero_based(self):
        config = dict(CONFIG, num_hidden_layers=3,
                      layer_types=["linear_attention", "qwen_sparse_attention", "linear_attention"],
                      ple_layer_ids=[3])
        prefix = "model.layers.2.ple.ple_embedding."
        state = {
            prefix + "layer_multipliers": ((3,), [11, 13, 17]),
            prefix + "ngram_heads_offsets": ((2,), [0, 3]),
            prefix + "ngram_heads_vocab_sizes": ((2,), [3, 5]),
            prefix + "ngram_embedding.weight": ((8, 2), list(range(16))),
        }
        metadata = qwen4exp._metadata(config, state, {})
        self.assertEqual(metadata["qwen4exp.ple.layers"], (spec_decode.ARRAY, (GGUF_UINT32, [2])))

    def test_json_state_adapter_preserves_large_integer_hashes(self):
        document = {"state_dict": {
            "model.layers.0.ple.ple_embedding.layer_multipliers": {
                "shape": [2], "values": [4294967311, 4294967357],
            },
            "model.embed_tokens.weight": {"shape": [1, 2], "values": [0.25, -0.5]},
        }}
        state = qwen4exp.state_from_document(document)
        self.assertEqual(state["model.layers.0.ple.ple_embedding.layer_multipliers"], ((2,), [4294967311, 4294967357]))
        self.assertEqual(state["model.embed_tokens.weight"], ((1, 2), [0.25, -0.5]))
        for invalid in ({}, {"state_dict": {"weight": {"shape": [2], "values": [1]}}},
                        {"state_dict": {"weight": {"shape": [1], "values": [True]}}}):
            with self.assertRaises(ValueError):
                qwen4exp.state_from_document(invalid)

    def test_linear_attention_head_reorder_and_transforms(self):
        # HF value heads are grouped by key head; GGUF tiles [key0,key1,key0,key1].
        order = [0, 2, 1, 3]
        raw_qkv = self.state["model.layers.0.linear_attn.in_proj_qkv.weight"][1]
        expected_qkv = raw_qkv[:16] + [v for head in order for v in raw_qkv[16 + head * 4:20 + head * 4]]
        self.assertEqual(self.decoded("blk.0.attn_qkv.weight"), expected_qkv)
        z = self.state["model.layers.0.linear_attn.in_proj_z.weight"][1]
        self.assertEqual(self.decoded("blk.0.attn_gate.weight"), [v for head in order for v in z[head * 4:(head + 1) * 4]])
        a = self.decoded("blk.0.ssm_a")
        expected_a = [-1.0, -3.0, -2.0, -4.0]
        self.assertEqual(a, expected_a)
        conv = self.state["model.layers.0.linear_attn.conv1d.weight"][1]
        conv_rows = [conv[i * 4:(i + 1) * 4] for i in range(8)]
        expected_conv = [v for row in conv_rows[:4] + [conv_rows[4 + head] for head in order] for v in row]
        self.assertEqual(self.decoded("blk.0.ssm_conv1d.weight"), expected_conv)
        projection = self.state["model.layers.0.linear_attn.out_proj.weight"][1]
        expected_projection = [v for row in range(4) for head in order for v in projection[row * 4 + head:row * 4 + head + 1]]
        self.assertEqual(self.decoded("blk.0.ssm_out.weight"), expected_projection)
        self.assertEqual(self.by_name["blk.0.ssm_conv1d.weight"][0], [4, 8])

    def test_indexer_split_norm_offset_and_expert_stacks(self):
        qpart = self.by_name["blk.1.indexer.q_proj.weight"]
        kpart = self.by_name["blk.1.indexer.k_proj.weight"]
        self.assertEqual(qpart[0], [4, 2])
        self.assertEqual(kpart[0], [4, 2])
        self.assertEqual(self.decoded("blk.1.indexer.q_proj.weight"), list(range(8)))
        self.assertEqual(self.decoded("blk.1.indexer.k_proj.weight"), list(range(8, 16)))
        self.assertEqual(self.decoded("blk.1.indexer.q_norm.weight"), [1.25, 1.5])
        gate_up = self.decoded("blk.0.ffn_gate_exps.weight")
        up = self.decoded("blk.0.ffn_up_exps.weight")
        self.assertEqual(gate_up, list(range(8)) + list(range(16, 24)))
        self.assertEqual(up, list(range(8, 16)) + list(range(24, 32)))
        self.assertEqual(self.by_name["blk.0.ffn_gate_exps.weight"][0], [4, 2, 2])
        self.assertEqual(len(self.by_name), len(self.tensors))

    def test_hyper_connection_and_ple_weights(self):
        self.assertEqual(self.decoded("output_hc_norm.weight"), [1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875, 2.0])
        self.assertEqual(self.decoded("blk.0.hc_attn_norm.weight"), [1.125] * 8)
        self.assertEqual(self.by_name["blk.0.ple_conv1d.weight"][0], [4, 8])
        self.assertEqual(self.decoded("blk.0.ple_conv1d.weight"), list(range(32, 64)))
        self.assertEqual(self.by_name["per_layer_token_embd.weight"][0], [2, 8])
        self.assertEqual(self.decoded("per_layer_token_embd.weight"), list(range(16)))

    def test_complete_weights_write_and_independently_decode(self):
        with tempfile.TemporaryDirectory(prefix="qwen4exp_gguf_") as directory:
            path = Path(directory) / "tiny.gguf"
            qwen4exp.write_qwen4exp(path, CONFIG, self.state)
            gguf = spec_decode.GGUF(path)
            self.assertEqual(gguf.value("general.architecture"), "qwen4exp")
            self.assertEqual(len(gguf.tensors), len(self.tensors))
            decoded_names = set()
            for tensor in gguf.tensors:
                decoded_names.add(tensor.name)
                want = self.decoded(tensor.name)
                got = gguf.decode(tensor, numpy=False)
                self.assertEqual(got, want, tensor.name)
            self.assertEqual(decoded_names, set(self.by_name))

    def test_pinned_hf_artifact_hash_inverse_and_independent_gguf(self):
        path = Path(__file__).resolve().parent / "data" / "qwen4exp_reference.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(document["model_revision"], "de4b8e4d43b917e7706784d8bb445c9af86a3540")
        self.assertEqual(document["transformers_release_commit"], "93c8b7b485963a10800c91f55304db6be211c2bd")
        config = document["config"]
        self.assertEqual(config["output_gate_type"], "sigmoid")
        state = qwen4exp.state_from_document(document)
        self.assertEqual(_canonical_reference_hash(state), "1dfc001dbd33a54e03123bd468079437c1776aa94908339e42bad6f71133bc1a")
        self.assertEqual(document["weights_canonical_sha256"], _canonical_reference_hash(state))
        metadata, tensors = qwen4exp.map_qwen4exp(config, state)
        _assert_inverse_source(self, config, state, metadata, tensors)

        ple_prefix = "model.layers.0.ple.ple_embedding."
        for source, target in (("layer_multipliers", "layer_multipliers"),
                               ("ngram_heads_offsets", "head_offsets"),
                               ("ngram_heads_vocab_sizes", "head_vocab_sizes")):
            source_values = state[ple_prefix + source][1]
            self.assertEqual(metadata["qwen4exp.ple." + target],
                             (spec_decode.ARRAY, (GGUF_UINT64, source_values)))
        for changed in ("model.embed_tokens.weight", ple_prefix + "layer_multipliers"):
            mutated = dict(state)
            shape, values = state[changed]
            values = list(values)
            values[0] += 1
            mutated[changed] = (shape, values)
            self.assertNotEqual(_canonical_reference_hash(mutated), document["weights_canonical_sha256"])

        with tempfile.TemporaryDirectory(prefix="qwen4exp_hf_artifact_") as directory:
            output = Path(directory) / "reference.gguf"
            qwen4exp.write_qwen4exp(output, config, state)
            independent = spec_decode.GGUF(output)
            expected = {name: (shape, type_id, payload) for name, shape, type_id, payload in tensors}
            self.assertEqual(len(independent.tensors), len(expected))
            for tensor in independent.tensors:
                shape, type_id, payload = expected[tensor.name]
                values = spec_decode.decode(type_id, payload, math.prod(shape))
                self.assertEqual(independent.decode(tensor, numpy=False), values, tensor.name)

    def test_native_reads_emitted_gguf_and_matches_hf_lookup_rows(self):
        tool = Path(common.EXE).parent / ("llmx-arch-qwen4exp-test.exe" if os.name == "nt" else "llmx-arch-qwen4exp-test")
        if not tool.is_file():
            if self.require_native:
                self.fail("llmx-arch-qwen4exp-test is required beside --exe")
            self.skipTest("llmx-arch-qwen4exp-test is not beside --exe")
        reference = Path(__file__).resolve().parent / "data" / "qwen4exp_reference.json"
        document = json.loads(reference.read_text(encoding="utf-8"))
        state = qwen4exp.state_from_document(document)
        with tempfile.TemporaryDirectory(prefix="qwen4exp_native_") as directory:
            model = Path(directory) / "reference.gguf"
            qwen4exp.write_qwen4exp(model, document["config"], state)
            result = subprocess.run([str(tool), "--gguf", str(model), "--reference", str(reference)],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_native_model_matches_full_hf_graph(self):
        tool = Path(common.EXE).parent / ("llmx-qwen4exp-model-test.exe" if os.name == "nt" else "llmx-qwen4exp-model-test")
        if not tool.is_file():
            if self.require_native:
                self.fail("llmx-qwen4exp-model-test is required beside --exe")
            self.skipTest("llmx-qwen4exp-model-test is not beside --exe")
        reference = Path(__file__).resolve().parent / "data" / "qwen4exp_reference.json"
        document = json.loads(reference.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory(prefix="qwen4exp_model_") as directory:
            model = Path(directory) / "reference.gguf"
            qwen4exp.write_qwen4exp(model, document["config"], qwen4exp.state_from_document(document))
            command = [str(tool), "--gguf", str(model), "--reference", str(reference)]
            if os.environ.get("LLMX_DEVICE"):
                command += ["--device", os.environ["LLMX_DEVICE"]]
            result = subprocess.run(command, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            group = [str(tool), "--gguf", str(model), "--reference", str(reference), "--tensor-width", "2"]
            result = subprocess.run(group, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_native_server_lookup_storage(self):
        if not Path(common.EXE).is_file():
            if self.require_native:
                self.fail("llmx executable is required for the server fixture")
            self.skipTest("llmx executable is unavailable for the server fixture")
        reference = Path(__file__).resolve().parent / "data" / "qwen4exp_reference.json"
        document = json.loads(reference.read_text(encoding="utf-8"))
        state = qwen4exp.state_from_document(document)
        metadata, tensors = qwen4exp.map_qwen4exp(document["config"], state)
        table = next(tensor for tensor in tensors if tensor[0] == "per_layer_token_embd.weight")
        layers = metadata["qwen4exp.ple.layers"][1][1]
        with tempfile.TemporaryDirectory(prefix="qwen4exp_lookup_health_") as directory:
            model = Path(directory) / "reference.gguf"
            qwen4exp.write_qwen4exp(model, document["config"], state)
            for storage in ("device", "file"):
                with self.subTest(storage=storage):
                    srv = server_test.Server(str(model), "--max-seqs", "1", "--kv-tokens", "512",
                                             "--lookup-storage", storage, "--host-cache-bytes", "0")
                    try:
                        topology = srv.get("/v1/health")["topology"]
                        self.assertEqual(topology["lookups"], [
                            {"layer": layer, "rows": table[1][1], "row_bytes": table[1][0] * 4,
                             "bytes": len(table[3]), "type": "F32", "storage": storage,
                             "members": topology["layers"][layer]["mixer_members"]}
                            for layer in layers])
                    finally:
                        srv.close()

    def test_native_server_two_histories_and_followup_match_hf(self):
        if not Path(common.EXE).is_file():
            if self.require_native:
                self.fail("llmx executable is required for the server fixture")
            self.skipTest("llmx executable is unavailable for the server fixture")
        reference = Path(__file__).resolve().parent / "data" / "qwen4exp_reference.json"
        document = json.loads(reference.read_text(encoding="utf-8"))
        cases = document["cases"]
        first = cases[0]
        second = cases[2]
        first_ids = first["input_ids"]
        second_ids = second["input_ids"][:-1]  # exclude its terminal EOS from the text prompt
        self.assertNotIn(document["config"]["eos_token_id"], first_ids)
        self.assertNotIn(document["config"]["eos_token_id"], second_ids)
        # The tiny fixture's byte IDs 0..30 map to control characters, not printable token text.
        first_prompt = "".join(chr(token) for token in first_ids)
        second_prompt = "".join(chr(token) for token in second_ids)
        followup_prompt = first_prompt + chr(first["decode_token"])

        def logits_for(case, ids):
            return case["full_logits"][len(ids) - 1]

        with tempfile.TemporaryDirectory(prefix="qwen4exp_server_") as directory:
            model = Path(directory) / "reference.gguf"
            qwen4exp.write_qwen4exp(model, document["config"],
                                    qwen4exp.state_from_document(document))
            srv = server_test.Server(str(model), "--max-seqs", "2", "--kv-tokens", "512",
                                     "--dtype", "f32", "--cache-type-k", "f32", "--cache-type-v", "f32",
                                     "--host-cache-bytes", "0")
            try:
                for prompt, ids in ((first_prompt, first_ids), (second_prompt, second_ids),
                                    (followup_prompt, first_ids + [first["decode_token"]])):
                    status, tokenized = srv.post("/v1/tokenize", {"text": prompt})
                    self.assertEqual(status, 200, tokenized)
                    self.assertEqual(tokenized["tokens"], ids, (prompt, tokenized, ids))

                body_a = {"prompt": first_prompt, "max_tokens": 1, "temperature": 0,
                          "ignore_eos": True, "logprobs": True, "top_logprobs": 5}
                body_b = {"prompt": second_prompt, "max_tokens": 1, "temperature": 0,
                          "ignore_eos": True, "logprobs": True, "top_logprobs": 5}
                alone_a = server_test.post_ok(srv, "/v1/generate", body_a)
                alone_b = server_test.post_ok(srv, "/v1/generate", body_b)
                self._check_server_hf_reply(alone_a, logits_for(first, first_ids), "first", len(first_ids))
                self._check_server_hf_reply(alone_b, logits_for(second, second_ids), "second", len(second_ids))

                together = srv.together("/v1/generate", [body_a, body_b])
                self.assertEqual([status for status, _ in together], [200, 200], together)
                for index, (status, reply) in enumerate(together):
                    self.assertEqual(status, 200, reply)
                    expected = alone_a if index == 0 else alone_b
                    for key in ("ids", "logprobs", "top_logprobs"):
                        self.assertEqual(reply[key], expected[key], (index, key, reply, expected))

                repeated = server_test.post_ok(srv, "/v1/generate", body_a)
                for key in ("ids", "logprobs", "top_logprobs"):
                    self.assertEqual(repeated[key], alone_a[key], (key, repeated, alone_a))

                followup = server_test.post_ok(
                    srv, "/v1/generate",
                    {"prompt": followup_prompt, "max_tokens": 1, "temperature": 0,
                     "ignore_eos": True, "logprobs": True, "top_logprobs": 5})
                self.assertEqual(followup["prompt_tokens"], len(first_ids) + 1)
                self._check_server_hf_reply(followup, first["decode_logits"], "followup", len(first_ids) + 1)
            finally:
                srv.close()

    def test_native_server_edited_prompt_reuses_checkpoint_matches_fresh(self):
        if not Path(common.EXE).is_file():
            if self.require_native:
                self.fail("llmx executable is required for the server fixture")
            self.skipTest("llmx executable is unavailable for the server fixture")
        reference = Path(__file__).resolve().parent / "data" / "qwen4exp_reference.json"
        document = json.loads(reference.read_text(encoding="utf-8"))
        case = document["persistence_case"]
        original_ids = case["input_ids"]
        self.assertEqual(len(original_ids), 269)
        eos = document["config"]["eos_token_id"]
        edited_ids = original_ids[:257] + [(token + 1) % eos for token in original_ids[257:]]
        self.assertNotEqual(original_ids[257], edited_ids[257])
        prompts = ["".join(chr(token) for token in ids) for ids in (original_ids, edited_ids)]
        flags = ("--kv-tokens", "512", "--max-seqs", "1", "--ubatch", "16",
                 "--threads", "1", "--dtype", "f32", "--cache-type-k", "f32",
                 "--cache-type-v", "f32", "--host-cache-bytes", "0")
        bodies = [{"prompt": prompt, "max_tokens": 1, "temperature": 0,
                   "ignore_eos": True, "logprobs": True, "top_logprobs": 5} for prompt in prompts]
        with tempfile.TemporaryDirectory(prefix="qwen4exp_server_edit_") as directory:
            model = Path(directory) / "reference.gguf"
            config = dict(document["config"])
            config.update(case["config_overrides"])
            qwen4exp.write_qwen4exp(model, config, qwen4exp.state_from_document(document))
            srv = server_test.Server(str(model), *flags)
            try:
                for prompt, ids in zip(prompts, (original_ids, edited_ids)):
                    status, tokenized = srv.post("/v1/tokenize", {"text": prompt})
                    self.assertEqual(status, 200, tokenized)
                    self.assertEqual(tokenized["tokens"], ids)
                original = server_test.post_ok(srv, "/v1/generate", bodies[0])
                self._check_server_hf_reply(original, case["full_logits"], "before edit", len(original_ids))
                edited = server_test.post_ok(srv, "/v1/generate", bodies[1])
                # The 269-token donor keeps its state at 256; the edit first differs at 257.
                self.assertEqual(edited["reused_tokens"], 256, edited)
            finally:
                srv.close()
            fresh = server_test.Server(str(model), *flags)
            try:
                status, tokenized = fresh.post("/v1/tokenize", {"text": prompts[1]})
                self.assertEqual(status, 200, tokenized)
                self.assertEqual(tokenized["tokens"], edited_ids)
                alone = server_test.post_ok(fresh, "/v1/generate", bodies[1])
                self.assertEqual(alone["reused_tokens"], 0, alone)
                self.assertEqual(edited["prompt_tokens"], len(edited_ids))
                self.assertEqual(alone["prompt_tokens"], len(edited_ids))
                for key in ("ids", "logprobs", "top_logprobs"):
                    self.assertEqual(edited[key], alone[key], (key, edited, alone))
            finally:
                fresh.close()

    def test_native_server_edited_chat_reuses_checkpoint_matches_fresh(self):
        if not Path(common.EXE).is_file():
            if self.require_native:
                self.fail("llmx executable is required for the server fixture")
            self.skipTest("llmx executable is unavailable for the server fixture")
        reference = Path(__file__).resolve().parent / "data" / "qwen4exp_reference.json"
        document = json.loads(reference.read_text(encoding="utf-8"))
        eos = document["config"]["eos_token_id"]
        roles = ("system", "user", "assistant", "user")
        markers = {"system": 1, "user": 2, "assistant": 3}
        contents = [[(i * 7 + 5) % eos for i in range(size)] for size in (8, 210, 16, 26)]
        edited_contents = [list(ids) for ids in contents]
        edited_contents[-1][16:] = [(token + 1) % eos for token in contents[-1][16:]]
        messages, rendered = [], []
        for variant in (contents, edited_contents):
            messages.append([{"role": role, "content": "".join(chr(token) for token in ids)}
                             for role, ids in zip(roles, variant)])
            rendered.append([token for role, ids in zip(roles, variant)
                             for token in [markers[role], *ids, 4]] + [3])
        self.assertEqual([len(ids) for ids in rendered], [269, 269])
        shared = next(i for i, (before, after) in enumerate(zip(*rendered)) if before != after)
        self.assertEqual(shared, 257)
        template = ("{% for message in messages %}"
                    "{% if message['role'] == 'system' %}\x01"
                    "{% elif message['role'] == 'user' %}\x02{% else %}\x03{% endif %}"
                    "{{ message['content'] }}\x04{% endfor %}"
                    "{% if add_generation_prompt %}\x03{% endif %}")
        flags = ("--kv-tokens", "512", "--max-seqs", "1", "--ubatch", "16",
                 "--threads", "1", "--dtype", "f32", "--cache-type-k", "f32",
                 "--cache-type-v", "f32", "--host-cache-bytes", "0")
        bodies = [{"messages": turns, "max_tokens": 1, "temperature": 0,
                   "ignore_eos": True, "logprobs": True, "top_logprobs": 5} for turns in messages]

        def render_ids(turns):
            return [token for message in turns
                    for token in [markers[message["role"]], *map(ord, message["content"]), 4]] + [3]

        cut_messages = messages[0][:2]
        cut_ids = render_ids(cut_messages)
        self.assertEqual(len(cut_ids), 223)
        cut_body = dict(bodies[0], messages=cut_messages)
        followup_user = messages[1][-1]

        with tempfile.TemporaryDirectory(prefix="qwen4exp_server_chat_edit_") as directory:
            model = Path(directory) / "reference.gguf"
            config = dict(document["config"])
            config.update(document["persistence_case"]["config_overrides"])
            metadata, tensors = qwen4exp.map_qwen4exp(config, qwen4exp.state_from_document(document))
            metadata["tokenizer.chat_template"] = (spec_decode.STRING, template)
            spec_decode.write_gguf(model, metadata, tensors)
            srv = server_test.Server(str(model), *flags)
            try:
                for turns, ids in zip(messages, rendered):
                    status, tokenized = srv.post("/v1/tokenize", {"messages": turns})
                    self.assertEqual(status, 200, tokenized)
                    self.assertEqual(tokenized["tokens"], ids)
                original = server_test.post_ok(srv, "/v1/chat", bodies[0])
                self.assertEqual(original["prompt_tokens"], 269)
                edited = server_test.post_ok(srv, "/v1/chat", bodies[1])
                self.assertNotEqual(original["top_logprobs"], edited["top_logprobs"])
                # This template preserves the 256-row checkpoint before the edit at 257.
                self.assertEqual(edited["reused_tokens"], 256, edited)
                status, tokenized = srv.post("/v1/tokenize", {"messages": cut_messages})
                self.assertEqual(status, 200, tokenized)
                self.assertEqual(tokenized["tokens"], cut_ids)
                cut = server_test.post_ok(srv, "/v1/chat", cut_body)
                self.assertEqual(cut["prompt_tokens"], len(cut_ids))
                followup_turns = cut_messages + [{"role": "assistant", "content": cut["text"]}, followup_user]
                cut_followup = dict(bodies[1], messages=followup_turns)
                followup_ids = render_ids(followup_turns)
                self.assertNotEqual(followup_ids, cut_ids)
                status, tokenized = srv.post("/v1/tokenize", {"messages": cut_followup["messages"]})
                self.assertEqual(status, 200, tokenized)
                self.assertEqual(tokenized["tokens"], followup_ids)
                continued = server_test.post_ok(srv, "/v1/chat", cut_followup)
                self.assertEqual(continued["prompt_tokens"], len(followup_ids))
                self.assertNotEqual(cut["top_logprobs"], continued["top_logprobs"])
            finally:
                srv.close()
            fresh_cases = (("edited", messages[1], rendered[1], bodies[1], edited),
                           ("cut", cut_messages, cut_ids, cut_body, cut),
                           ("follow-up", cut_followup["messages"], followup_ids, cut_followup, continued))
            for label, turns, expected_ids, body, warmed in fresh_cases:
                fresh = server_test.Server(str(model), *flags)
                try:
                    status, tokenized = fresh.post("/v1/tokenize", {"messages": turns})
                    self.assertEqual(status, 200, (label, tokenized))
                    self.assertEqual(tokenized["tokens"], expected_ids, label)
                    alone = server_test.post_ok(fresh, "/v1/chat", body)
                    self.assertEqual(alone["reused_tokens"], 0, (label, alone))
                    self.assertEqual(alone["prompt_tokens"], len(expected_ids), label)
                    self.assertEqual(warmed["prompt_tokens"], len(expected_ids), label)
                    # Each response gets an independent fresh server; this is not an HF oracle.
                    for key in ("ids", "logprobs", "top_logprobs"):
                        self.assertEqual(warmed[key], alone[key], (label, key, warmed, alone))
                finally:
                    fresh.close()

    def test_native_server_tensor_group_overlaps_two_hf_histories(self):
        if not Path(common.EXE).is_file():
            if self.require_native:
                self.fail("llmx executable is required for the server fixture")
            self.skipTest("llmx executable is unavailable for the server fixture")

        reference = Path(__file__).resolve().parent / "data" / "qwen4exp_reference.json"
        document = json.loads(reference.read_text(encoding="utf-8"))
        case = document["persistence_case"]
        ids = case["input_ids"]
        prefix_tokens = case["prefix_tokens"]
        self.assertEqual((len(ids), prefix_tokens), (269, 257))
        self.assertTrue(all(type(token) is int and 0 <= token < document["config"]["eos_token_id"]
                            for token in ids))
        prompts = ["".join(chr(token) for token in ids[:prefix_tokens]),
                   "".join(chr(token) for token in ids)]
        wanted_logits = [case["prefix_logits"], case["full_logits"]]

        # Add a zero shared-expert channel so the unchanged HF fixture fits a width-two group.
        config = dict(document["config"])
        state = qwen4exp.state_from_document(document)
        hidden = config["hidden_size"]
        width = config["shared_expert_intermediate_size"]
        self.assertEqual(width, 7)
        for layer in range(config["num_hidden_layers"]):
            prefix = "model.layers.%d.mlp.shared_expert." % layer
            for proj in ("gate_proj", "up_proj"):
                name = prefix + proj + ".weight"
                shape, values = state[name]
                self.assertEqual(shape, (width, hidden), name)
                state[name] = ((8, hidden), list(values) + [0.0] * hidden)
            name = prefix + "down_proj.weight"
            shape, values = state[name]
            self.assertEqual(shape, (hidden, width), name)
            rows = [values[row * width:(row + 1) * width] + [0.0] for row in range(hidden)]
            state[name] = ((hidden, 8), [value for row in rows for value in row])
        config["shared_expert_intermediate_size"] = 8
        config.update(case["config_overrides"])

        configured = os.environ.get("LLMX_DEVICE")
        if not configured or len(configured.split(",")) < 2:
            self.skipTest("width-two server group needs two configured devices; duplicate CPU entries are refused")
        group_device = ",".join(configured.split(",")[:2])

        with tempfile.TemporaryDirectory(prefix="qwen4exp_server_group_") as directory:
            model = Path(directory) / "reference-padded.gguf"
            qwen4exp.write_qwen4exp(model, config, state)

            # First prove the zero-padding preserves the original HF math on one device.
            single = server_test.Server(str(model), "--device", group_device.split(",", 1)[0],
                                        "--max-seqs", "1", "--kv-tokens", "512", "--ubatch", "1",
                                        "--threads", "1", "--dtype", "f32", "--cache-type-k", "f32",
                                        "--cache-type-v", "f32", "--host-cache-bytes", "0")
            try:
                single_replies = []
                for prompt, logits in zip(prompts, wanted_logits):
                    body = {"prompt": prompt, "max_tokens": 1, "temperature": 0,
                            "ignore_eos": True, "logprobs": True, "top_logprobs": 5}
                    reply = server_test.post_ok(single, "/v1/generate", body)
                    self._check_server_hf_reply(reply, logits, "padded-single", len(prompt))
                    single_replies.append(reply)
            finally:
                single.close()

            group_flags = ("--device", group_device, "--max-seqs", "2", "--kv-tokens", "1024",
                           "--ubatch", "1", "--tensor-width", "2", "--threads", "1", "--dtype", "f32",
                           "--cache-type-k", "f32", "--cache-type-v", "f32", "--host-cache-bytes", "0")
            srv = server_test.Server(str(model), *group_flags)
            try:
                bodies = [{"prompt": prompt, "max_tokens": 1, "temperature": 0,
                           "ignore_eos": True, "logprobs": True, "top_logprobs": 5}
                          for prompt in prompts]
                alone = [server_test.post_ok(srv, "/v1/generate", body) for body in bodies]
                for index, reply in enumerate(alone):
                    self._check_server_hf_reply(reply, wanted_logits[index], "group-alone-%d" % index,
                                                len(prompts[index]))
                    self.assertEqual(reply["ids"], single_replies[index]["ids"])

                srv.close()
                srv = None
                srv = server_test.Server(str(model), *group_flags)
                done = threading.Event()
                overlap = threading.Event()
                monitor_errors = []

                def monitor_active():
                    while not done.is_set():
                        try:
                            health = srv.get("/v1/health")
                            if health["requests"]["now"]["active"] >= 2:
                                overlap.set()
                                return
                        except Exception as error:
                            monitor_errors.append(repr(error))
                            return
                        done.wait(0.002)

                monitor = threading.Thread(target=monitor_active, name="qwen4exp-active-witness")
                monitor.start()
                try:
                    together = srv.together("/v1/generate", bodies)
                finally:
                    done.set()
                    monitor.join(timeout=35)
                self.assertFalse(monitor.is_alive(), "active-request witness did not stop")
                self.assertFalse(monitor_errors, monitor_errors)
                self.assertTrue(overlap.is_set(), "health never witnessed two active requests")
                self.assertEqual([status for status, _ in together], [200, 200], together)
                for index, (status, reply) in enumerate(together):
                    self.assertEqual(status, 200, reply)
                    self._check_server_hf_reply(reply, wanted_logits[index], "group-together-%d" % index,
                                                len(prompts[index]))
                    for key in ("ids", "logprobs", "top_logprobs"):
                        self.assertEqual(reply[key], alone[index][key],
                                         ("together", index, key, reply, alone[index]))
            finally:
                if srv is not None:
                    srv.close()

    def test_native_server_restores_qwen4exp_history_from_disk(self):
        if os.name == "nt":
            self.skipTest("SIGTERM disk persistence is POSIX-only")
        if not Path(common.EXE).is_file():
            if self.require_native:
                self.fail("llmx executable is required for the server fixture")
            self.skipTest("llmx executable is unavailable for the server fixture")

        reference = Path(__file__).resolve().parent / "data" / "qwen4exp_reference.json"
        document = json.loads(reference.read_text(encoding="utf-8"))
        case = document["persistence_case"]
        ids = case["input_ids"]
        prefix_tokens = case["prefix_tokens"]
        self.assertEqual(len(ids), 269)
        self.assertEqual(prefix_tokens, 257)
        self.assertTrue(all(type(token) is int and 0 <= token < document["config"]["eos_token_id"]
                            for token in ids))
        prefix_prompt = "".join(chr(token) for token in ids[:prefix_tokens])
        full_prompt = "".join(chr(token) for token in ids)

        with tempfile.TemporaryDirectory(prefix="qwen4exp_disk_") as directory:
            model = Path(directory) / "reference.gguf"
            config = dict(document["config"])
            config.update(case["config_overrides"])
            qwen4exp.write_qwen4exp(model, config, qwen4exp.state_from_document(document))
            cache = os.path.join(directory, "disk")
            os.makedirs(cache)
            flags = ("--kv-tokens", "1024", "--max-seqs", "1", "--ubatch", "16",
                     "--dtype", "f32", "--cache-type-k", "f32", "--cache-type-v", "f32") + \
                    server_test.disk_flags(cache, "--disk-cache-keep", host=64 << 20, disk=256 << 20)

            srv = None
            try:
                srv = server_test.Server(str(model), *flags)
                status, tokenized = srv.post("/v1/tokenize", {"text": prefix_prompt})
                self.assertEqual(status, 200, tokenized)
                self.assertEqual(tokenized["tokens"], ids[:prefix_tokens])
                prefix_reply = server_test.post_ok(
                    srv, "/v1/generate",
                    {"prompt": prefix_prompt, "max_tokens": 1, "temperature": 0,
                     "ignore_eos": True, "logprobs": True, "top_logprobs": 5})
                self._check_server_hf_reply(prefix_reply, case["prefix_logits"],
                                            "disk prefix", prefix_tokens)
                srv.wait(lambda health: health["reuse"]["disk"]["now"]["writing"],
                         "Qwen4Exp disk tier did not become writable", 60)
                self.assertEqual(server_test.terminate(srv), 0)
                srv = None

                kept = server_test.servers_in(cache)
                self.assertEqual(len(kept), 1, kept)
                self.assertTrue(os.path.exists(os.path.join(cache, kept[0], "kept")), kept)

                srv = server_test.Server(str(model), *flags)
                health = srv.wait(
                    lambda value: value["reuse"]["disk"]["now"]["entries"] >= 1,
                    "Qwen4Exp disk entries were not adopted", 60)
                disk_now = health["reuse"]["disk"]["now"]
                self.assertGreaterEqual(disk_now["segments"], 1, health)
                self.assertGreaterEqual(disk_now["states"], 1, health)
                self.assertEqual(disk_now["segments"] + disk_now["states"], disk_now["entries"], health)
                self.assertEqual(health["reuse"]["disk"]["since_start"]["errors"], 0, health)

                status, tokenized = srv.post("/v1/tokenize", {"text": full_prompt})
                self.assertEqual(status, 200, tokenized)
                self.assertEqual(tokenized["tokens"], ids)
                restored = server_test.post_ok(
                    srv, "/v1/generate",
                    {"prompt": full_prompt, "max_tokens": 1, "temperature": 0,
                     "ignore_eos": True, "logprobs": True, "top_logprobs": 5})
                health = srv.get("/v1/health")
                self.assertGreaterEqual(restored["reused_tokens"], 128, (restored, health))
                self._check_server_hf_reply(restored, case["full_logits"],
                                            "restored continuation", len(ids))
                self.assertGreaterEqual(health["reuse"]["disk"]["since_start"]["hits"], 1, health)
                self.assertEqual(health["reuse"]["disk"]["since_start"]["errors"], 0, health)

                srv.close()
                srv = None

                crash_cache = os.path.join(directory, "crash-disk")
                os.makedirs(crash_cache)
                crash_flags = ("--kv-tokens", "1024", "--max-seqs", "1", "--ubatch", "16",
                               "--dtype", "f32", "--cache-type-k", "f32", "--cache-type-v", "f32") + \
                              server_test.disk_flags(crash_cache, "--disk-cache-keep",
                                                     host=64 << 20, disk=256 << 20)
                srv = server_test.Server(str(model), *crash_flags)
                status, tokenized = srv.post("/v1/tokenize", {"text": prefix_prompt})
                self.assertEqual(status, 200, tokenized)
                self.assertEqual(tokenized["tokens"], ids[:prefix_tokens])
                before_crash = server_test.post_ok(
                    srv, "/v1/generate",
                    {"prompt": prefix_prompt, "max_tokens": 1, "temperature": 0,
                     "ignore_eos": True, "logprobs": True, "top_logprobs": 5})
                self._check_server_hf_reply(before_crash, case["prefix_logits"],
                                            "crash-arm prefix", prefix_tokens)

                def published(health):
                    now = health["reuse"]["disk"]["now"]
                    return (now["ready"] and now["entries"] >= 1 and now["segments"] >= 1 and
                            now["states"] >= 1 and now["in_flight"] == 0)

                health = srv.wait(published,
                                  "fresh Qwen4Exp cache did not publish complete segment/state entries", 60)
                disk_now = health["reuse"]["disk"]["now"]
                self.assertGreater(disk_now["bytes"], 0, health)
                self.assertEqual(disk_now["segments"] + disk_now["states"], disk_now["entries"], health)
                self.assertGreaterEqual(health["reuse"]["disk"]["since_start"]["bytes_written"], 1, health)

                child = srv.proc
                child.send_signal(signal.SIGKILL)
                killed_status = child.wait(timeout=60)
                srv.log.close()
                srv = None
                self.assertEqual(killed_status, -signal.SIGKILL)

                srv = server_test.Server(str(model), *crash_flags)
                health = srv.wait(
                    lambda value: published(value),
                    "published Qwen4Exp entries were not adopted after SIGKILL", 60)
                disk_now = health["reuse"]["disk"]["now"]
                self.assertEqual(disk_now["segments"] + disk_now["states"], disk_now["entries"], health)
                self.assertEqual(health["reuse"]["disk"]["since_start"]["errors"], 0, health)

                status, tokenized = srv.post("/v1/tokenize", {"text": full_prompt})
                self.assertEqual(status, 200, tokenized)
                self.assertEqual(tokenized["tokens"], ids)
                recovered = server_test.post_ok(
                    srv, "/v1/generate",
                    {"prompt": full_prompt, "max_tokens": 1, "temperature": 0,
                     "ignore_eos": True, "logprobs": True, "top_logprobs": 5})
                self.assertGreaterEqual(recovered["reused_tokens"], 128, recovered)
                self._check_server_hf_reply(recovered, case["full_logits"],
                                            "crash-recovered continuation", len(ids))
                health = srv.get("/v1/health")
                self.assertGreaterEqual(health["reuse"]["disk"]["since_start"]["hits"], 1, health)
                self.assertEqual(health["reuse"]["disk"]["since_start"]["errors"], 0, health)
            finally:
                if srv is not None:
                    srv.close()

    def test_unknown_missing_or_bad_shape_refused(self):
        bad = dict(self.state)
        bad["model.layers.0.unrecognized.weight"] = ((1,), [0.0])
        with self.assertRaisesRegex(ValueError, "state keys differ"):
            qwen4exp.map_qwen4exp(CONFIG, bad)
        bad = dict(self.state)
        del bad["model.layers.0.ple.ple_embedding.ngram_embedding.weight"]
        with self.assertRaisesRegex(ValueError, "state keys differ"):
            qwen4exp.map_qwen4exp(CONFIG, bad)
        bad = dict(self.state)
        bad["model.layers.1.self_attn.q_proj.weight"] = ((7, 4), [0.0] * 28)
        with self.assertRaisesRegex(ValueError, "shape"):
            qwen4exp.map_qwen4exp(CONFIG, bad)

    def test_refusal_happens_before_output_is_replaced(self):
        with tempfile.TemporaryDirectory(prefix="qwen4exp_refuse_") as directory:
            path = Path(directory) / "existing.gguf"
            path.write_bytes(b"preserve this file")
            bad = dict(self.state)
            bad["unexpected.tensor"] = ((1,), [2.0])
            with self.assertRaises(ValueError):
                qwen4exp.write_qwen4exp(path, CONFIG, bad)
            self.assertEqual(path.read_bytes(), b"preserve this file")

    def test_only_sigmoid_gdn_gate_is_representable_in_gguf(self):
        omitted = dict(CONFIG)
        del omitted["output_gate_type"]
        for config in (omitted, dict(CONFIG, output_gate_type=None), dict(CONFIG, output_gate_type="silu"),
                       dict(CONFIG, output_gate_type="tanh")):
            with self.subTest(output_gate_type=config.get("output_gate_type", "omitted")):
                with self.assertRaisesRegex(ValueError, "requires sigmoid GDN output gating"):
                    qwen4exp.map_qwen4exp(config, self.state)
        with tempfile.TemporaryDirectory(prefix="qwen4exp_gate_refuse_") as directory:
            path = Path(directory) / "existing.gguf"
            path.write_bytes(b"preserve this file")
            with self.assertRaisesRegex(ValueError, "requires sigmoid GDN output gating"):
                qwen4exp.write_qwen4exp(path, omitted, self.state)
            self.assertEqual(path.read_bytes(), b"preserve this file")

    def test_refuses_ambiguous_or_unsupported_configs_and_nonfinite_weights(self):
        for change, message in (({"eos_token_id": [3, 2]}, "scalar eos"),
                                ({"ple_layer_ids": [1, 2]}, "exactly one PLE"),
                                ({"vision_config": {"x": 1}}, "text-only"),
                                ({"layer_types": ["linear_attention", "full_attention"]}, "unsupported Qwen4Exp layer kind")):
            config = dict(CONFIG, **change)
            with self.assertRaisesRegex(ValueError, message):
                qwen4exp.map_qwen4exp(config, self.state)
        bad = dict(self.state)
        shape, values = bad["lm_head.weight"]
        bad["lm_head.weight"] = (shape, [float("nan")] + values[1:])
        with self.assertRaisesRegex(ValueError, "non-finite"):
            qwen4exp.map_qwen4exp(CONFIG, bad)


if __name__ == "__main__":
    unittest.main()
