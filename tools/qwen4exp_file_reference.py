"""Bounded file-exact Qwen4Exp weights around the pinned HF model's own forward."""

import argparse
from contextlib import contextmanager
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import spec_decode
import qwen4exp_reference as reference
from gen_layered_reference import loaded_modules, peak_gib


SDPA_SOURCE_SHA256 = "53c7229daca9ade4c5df874194448938c1edc925abbc71809f9750dd66381e6f"


def environment():
    import torch
    import transformers
    import numpy
    import safetensors
    modules = {name: importlib.import_module(path) for name, path in (
        ("cache_utils.py", "transformers.cache_utils"),
        ("modeling_qwen4_exp.py", "transformers.models.qwen4_exp.modeling_qwen4_exp"),
        ("configuration_qwen4_exp.py", "transformers.models.qwen4_exp.configuration_qwen4_exp"))}
    hashes = {name: reference._sha256_file(Path(module.__file__)) for name, module in modules.items()}
    optional = tuple(name for name in ("fla", "flash_linear_attention", "causal_conv1d", "kernels")
                     if importlib.util.find_spec(name) is not None)
    reference._check_environment(torch.__version__, transformers.__version__, numpy.__version__,
                                 safetensors.__version__, hashes, optional)
    return torch, transformers


class Shards:
    """Header-only shard index; every payload read uses the independent packed span decoder."""

    def __init__(self, paths, max_decoded_bytes):
        if not paths or len(set(map(str, paths))) != len(paths):
            raise ValueError("reference requires distinct GGUF shard paths")
        self.files = sorted((spec_decode.GGUF(path) for path in paths), key=lambda file: file.value("split.no", 0))
        self.tensors = {}
        self.max_decoded_bytes = max_decoded_bytes
        self.seen = set()
        self.largest_decode = 0
        count = self.files[0].value("split.count", 1)
        if count != len(self.files):
            raise ValueError("incomplete GGUF shard set")
        numbers = set()
        baseline = {key: value for key, value in self.files[0].metadata.items() if key != "split.no"}
        for file in self.files:
            if any(key != "split.no" and (key not in baseline or value != baseline[key]) for key, value in file.metadata.items()):
                raise ValueError("GGUF shard metadata differs")
            number = file.value("split.no", 0)
            if number in numbers or not 0 <= number < count:
                raise ValueError("repeated or invalid GGUF shard number")
            numbers.add(number)
            for tensor in file.tensors:
                if tensor.name in self.tensors:
                    raise ValueError("duplicate GGUF tensor " + tensor.name)
                self.tensors[tensor.name] = file, tensor
        total = self.files[0].value("split.tensors.count", len(self.tensors))
        if total != len(self.tensors):
            raise ValueError("GGUF shard tensor count differs")
        if self.value("general.architecture") != "qwen4exp":
            raise ValueError("file-exact Qwen4Exp reference needs qwen4exp GGUF")

    def value(self, key, default=None):
        return self.files[0].value(key, default)

    def read(self, name, first=0, count=None):
        file, tensor = self.tensors[name]
        count = tensor.count - first if count is None else count
        if count * 4 > self.max_decoded_bytes:
            raise ValueError("decoded span exceeds reference memory limit: " + name)
        values = file.decode_span(tensor, first, count, numpy=True, tensor_sources=self.tensors)
        self.largest_decode = max(self.largest_decode, values.nbytes)
        self.seen.add(name)
        return values

    def embedding(self, name, ids):
        import numpy as np
        _, tensor = self.tensors[name]
        columns, rows = tensor.shape
        flat = ids.detach().cpu().numpy().reshape(-1)
        if flat.size and (flat.min() < 0 or flat.max() >= rows):
            raise ValueError("embedding row outside GGUF tensor " + name)
        unique, inverse = np.unique(flat, return_inverse=True)
        if unique.size * columns * 4 > self.max_decoded_bytes:
            raise ValueError("selected embedding rows exceed reference memory limit")
        packed = np.empty((len(unique), columns), dtype=np.float32)
        start = 0
        while start < len(unique):
            end = start + 1
            while end < len(unique) and unique[end] == unique[end - 1] + 1:
                end += 1
            packed[start:end] = self.read(name, int(unique[start]) * columns, (end - start) * columns).reshape(-1, columns)
            start = end
        return packed, inverse.reshape(tuple(ids.shape))

    def labels(self):
        files = []
        for file in self.files:
            path = Path(file.path)
            print("hash begin " + path.name, file=sys.stderr, flush=True)
            files.append({"file": path.name, "bytes": path.stat().st_size, "sha256": reference._sha256_file(path)})
            print("hash end " + path.name, file=sys.stderr, flush=True)
        types = {}
        for name in sorted(self.seen):
            kind = spec_decode.type_name(self.tensors[name][1].type)
            types[kind] = types.get(kind, 0) + 1
        return {"files": files, "types": types, "source_tensors": len(self.tensors),
                "decoded_source_tensors": len(self.seen), "largest_decoded_span_bytes": self.largest_decode}


class ExpertWeight:
    """HF's original expert loop indexes one decoded expert at a time."""

    def __init__(self, owner, layer, gate_up):
        self.owner, self.layer, self.gate_up = owner, layer, gate_up

    def __getitem__(self, expert):
        import numpy as np
        owner, c = self.owner, self.owner.config
        expert = int(expert)
        if not 0 <= expert < c.num_experts:
            raise ValueError("expert index outside reference configuration")
        count = c.hidden_size * c.moe_intermediate_size
        prefix = "blk.%d." % self.layer
        if self.gate_up:
            gate = owner.source.read(prefix + "ffn_gate_exps.weight", expert * count, count)
            up = owner.source.read(prefix + "ffn_up_exps.weight", expert * count, count)
            values = np.concatenate((gate.reshape(-1, c.hidden_size), up.reshape(-1, c.hidden_size)), axis=0)
        else:
            values = owner.source.read(prefix + "ffn_down_exps.weight", expert * count, count).reshape(c.hidden_size, -1)
        return owner.torch.from_numpy(values)


class FileReference:
    """Original HF forward with module-resident ordinary weights, row-selected PLE and one expert per indexing operation."""

    def __init__(self, paths, config, max_tokens=128, max_decoded_bytes=3 << 30, attention="eager"):
        if attention not in ("eager", "sdpa"):
            raise ValueError("reference attention must be eager or sdpa")
        self.torch, _ = environment()
        self.attention_source_sha256 = {}
        if attention == "sdpa":
            module = importlib.import_module("transformers.integrations.sdpa_attention")
            actual = reference._sha256_file(Path(module.__file__))
            if actual != SDPA_SOURCE_SHA256:
                raise ValueError("pinned SDPA integration source hash differs")
            self.attention_source_sha256["integrations/sdpa_attention.py"] = actual
        from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
        from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM
        self.config = Qwen4ExpTextConfig(**config.get("text_config", config))
        self.config._attn_implementation = attention
        self.config._experts_implementation = "eager"
        if (self.config.output_gate_type or self.config.hidden_act) != "sigmoid":
            raise ValueError("qwen4exp GGUF requires sigmoid GDN output gating")
        if max_tokens <= 0 or max_decoded_bytes <= 0:
            raise ValueError("reference limits must be positive")
        self.max_tokens = max_tokens
        self.source = Shards(paths, max_decoded_bytes)
        c = self.config
        expected = {"embedding_length": c.hidden_size, "attention.head_count": c.num_attention_heads,
                    "attention.head_count_kv": c.num_key_value_heads, "attention.key_length": c.head_dim,
                    "attention.layer_norm_rms_epsilon": c.rms_norm_eps, "rope.freq_base": c.rope_parameters["rope_theta"],
                    "rope.dimension_count": int(c.head_dim * c.rope_parameters.get("partial_rotary_factor", 1)),
                    "expert_count": c.num_experts, "expert_used_count": c.num_experts_per_tok,
                    "expert_feed_forward_length": c.moe_intermediate_size, "expert_shared_feed_forward_length": c.shared_expert_intermediate_size,
                    "hyper_connection.count": c.hc_count, "hyper_connection.low_rank": c.hc_lowrank,
                    "ssm.conv_kernel": c.linear_conv_kernel_dim, "ssm.state_size": c.linear_key_head_dim,
                    "ssm.group_count": c.linear_num_key_heads, "ssm.time_step_rank": c.linear_num_value_heads,
                    "ssm.inner_size": c.linear_num_value_heads * c.linear_value_head_dim,
                    "attention.indexer.head_count": c.indexer_n_heads, "attention.indexer.key_length": c.indexer_head_dim,
                    "attention.indexer.top_k": c.indexer_budget}
        for key, value in expected.items():
            if self.source.value("qwen4exp." + key) != (spec_decode.to_f32(value) if isinstance(value, float) else value):
                raise ValueError("HF configuration disagrees with GGUF metadata: " + key)
        if self.source.value("qwen4exp.block_count") - self.source.value("qwen4exp.nextn_predict_layers", 0) != c.num_hidden_layers:
            raise ValueError("HF layer count disagrees with GGUF metadata")
        recurrent = self.source.value("qwen4exp.attention.recurrent_layers")
        interval = self.source.value("qwen4exp.full_attention_interval")
        kinds = recurrent[1][:c.num_hidden_layers] if recurrent is not None else None
        if interval is not None:
            if not isinstance(interval, int) or interval <= 0:
                raise ValueError("invalid GGUF full-attention interval")
            periodic = [(index + 1) % interval != 0 for index in range(c.num_hidden_layers)]
            if kinds is not None and kinds != periodic:
                raise ValueError("GGUF layer kinds contradict full-attention interval")
            kinds = periodic
        if kinds != [kind == "linear_attention" for kind in c.layer_types]:
            raise ValueError("HF layer kinds disagree with GGUF metadata")
        ratios = self.source.value("qwen4exp.attention.compress_ratios")[1][:c.num_hidden_layers]
        if ratios != [0 if kind == "linear_attention" else c.indexer_compress_ratio for kind in c.layer_types]:
            raise ValueError("HF compression ratios disagree with GGUF metadata")
        sections = list(c.rope_parameters.get("mrope_section", [11, 11, 10]))
        if self.source.value("qwen4exp.rope.dimension_sections")[1] != (sections + [0] * 4)[:4]:
            raise ValueError("HF rotary sections disagree with GGUF metadata")
        ple_layers = self.source.value("qwen4exp.ple.layers", (4, []))[1]
        if len(c.ple_layer_ids) > 1 or ple_layers != [layer - 1 for layer in c.ple_layer_ids]:
            raise ValueError("HF PLE layer IDs disagree with GGUF metadata")
        if ple_layers:
            eos = c.eos_token_id[0] if isinstance(c.eos_token_id, list) else c.eos_token_id
            for key, value in (("ngram_size", c.ngram_size), ("heads_per_ngram", c.heads_per_ngram),
                               ("conv_kernel", c.ple_conv_kernel_size), ("eos_token_id", eos)):
                if self.source.value("qwen4exp.ple." + key) != value:
                    raise ValueError("HF PLE configuration disagrees with GGUF metadata: " + key)
            image = self.source.value("qwen4exp.ple.image_token_id")
            if image is not None and image != config.get("image_token_id"):
                raise ValueError("HF image token disagrees with GGUF PLE metadata")
        self.head_order = sorted(range(c.linear_num_value_heads),
                                 key=lambda index: (index % c.linear_num_key_heads) * (c.linear_num_value_heads // c.linear_num_key_heads) + index // c.linear_num_key_heads)
        with self.torch.device("meta"):
            self.model = Qwen4ExpForCausalLM(c)
        self.model.eval().requires_grad_(False)
        text = self.model.model
        text.rotary_emb = type(text.rotary_emb)(config=c)
        self.modules = [("model.embed_tokens", text.embed_tokens)]
        self.modules += [("model.layers.%d" % i, layer) for i, layer in enumerate(text.layers)]
        self.modules += [("model.hyper_connection_mixer", text.hyper_connection_mixer), ("lm_head", self.model.lm_head)]
        self.ple_embeddings = []
        self.ple_buffers = {}
        self.mapping_roundtrip = {"norm_max_abs": 0.0, "a_max_relative": 0.0, "limit_f32_units": 8}
        for i, layer in enumerate(text.layers):
            experts = layer.mlp.experts
            for name, gate_up in (("gate_up_proj", True), ("down_proj", False)):
                delattr(experts, name)
                setattr(experts, name, ExpertWeight(self, i, gate_up))
            if layer.ple is not None:
                from transformers.models.qwen4_exp.modeling_qwen4_exp import _build_layer_multipliers
                ngram = layer.ple.ple_embedding
                multipliers = _build_layer_multipliers(ngram.unigram_vocab_size, ngram.ngram_size, ngram.ple_layer_index, ngram.seed).tolist()
                for field, key, values in (("layer_multipliers", "layer_multipliers", multipliers),
                                           ("ngram_heads_offsets", "head_offsets", ngram.head_offsets),
                                           ("ngram_heads_vocab_sizes", "head_vocab_sizes", ngram.head_vocab_sizes)):
                    if self.source.value("qwen4exp.ple." + key) != (10, values):
                        raise ValueError("GGUF PLE constants disagree with independent HF construction: " + key)
                    self.ple_buffers["model.layers.%d.ple.ple_embedding.%s" % (i, field)] = values
                embedding = ngram.ngram_embedding
                delattr(embedding, "weight")
                embedding.weight = self.torch.empty(0, c.ple_embed_dim // ((c.ngram_size - 1) * c.heads_per_ngram))
                self.ple_embeddings.append(embedding)
        self.expected = {}
        for prefix, module in self.modules:
            for name, tensor in module.state_dict().items():
                self.expected[prefix + "." + name] = tuple(tensor.shape)
        # Check every ordinary parameter and each omitted storage's shape without reading its payload.
        for name, shape in self.expected.items():
            self._parts(name, shape, validate=True)
        for i in range(c.num_hidden_layers):
            for name, shape in (("ffn_gate_exps.weight", (c.hidden_size, c.moe_intermediate_size, c.num_experts)),
                                ("ffn_up_exps.weight", (c.hidden_size, c.moe_intermediate_size, c.num_experts)),
                                ("ffn_down_exps.weight", (c.moe_intermediate_size, c.hidden_size, c.num_experts))):
                self._shape("blk.%d.%s" % (i, name), shape)
        if self.ple_embeddings:
            columns = c.ple_embed_dim // ((c.ngram_size - 1) * c.heads_per_ngram)
            _, table = self.source.tensors["per_layer_token_embd.weight"]
            self._shape("per_layer_token_embd.weight", (columns, table.shape[1]))
            if table.shape[1] != self.ple_embeddings[0].num_embeddings or self.source.value("qwen4exp.embedding_length_per_layer_input") != columns:
                raise ValueError("HF PLE table geometry disagrees with GGUF metadata")
            offsets = self.source.value("qwen4exp.ple.head_offsets")[1]
            sizes = self.source.value("qwen4exp.ple.head_vocab_sizes")[1]
            if any(offset + size > table.shape[1] for offset, size in zip(offsets, sizes)):
                raise ValueError("PLE metadata exceeds the source table")

    def _shape(self, name, shape):
        if name not in self.source.tensors:
            raise ValueError("missing GGUF tensor " + name)
        tensor = self.source.tensors[name][1]
        if tuple(tensor.shape) != tuple(shape):
            raise ValueError("unexpected GGUF tensor shape: " + name)
        if tensor.type not in (spec_decode.F32, spec_decode.F16, spec_decode.BF16, spec_decode.Q8_0):
            raise ValueError("unsupported file-exact tensor type: " + name)

    def _parts(self, name, shape, validate=False):
        import numpy as np
        kind = "plain"
        match = re.fullmatch(r"model\.layers\.(\d+)\.(.+)", name)
        if name == "model.embed_tokens.weight":
            source = "token_embd.weight"
        elif name == "lm_head.weight":
            source = "output.weight" if "output.weight" in self.source.tensors else "token_embd.weight"
        elif name.startswith("model.hyper_connection_mixer."):
            suffix = name.removeprefix("model.hyper_connection_mixer.")
            source = {"hc_norm.weight": "output_hc_norm.weight", "input_mix_weight_down.weight": "output_hc_down.weight",
                      "input_mix_weight_up.weight": "output_hc_up.weight"}[suffix]
            kind = "norm" if suffix == "hc_norm.weight" else kind
        elif match:
            index, suffix = match.groups()
            prefix = "blk." + index + "."
            for branch, target in (("attn_hyper_connection.", "hc_attn"), ("mlp_hyper_connection.", "hc_ffn")):
                if suffix.startswith(branch):
                    field = suffix[len(branch):]
                    source = prefix + target + {"hc_norm.weight": "_norm.weight", "input_mix_weight_down.weight": "_down.weight",
                                                "input_mix_weight_up.weight": "_up.weight", "block_inject_weight.weight": "_inject.weight"}[field]
                    kind = "norm" if field == "hc_norm.weight" else kind
                    break
            else:
                fields = {
                    "linear_attn.in_proj_qkv.weight": ("attn_qkv.weight", "qkv"), "linear_attn.in_proj_z.weight": ("attn_gate.weight", "heads"),
                    "linear_attn.in_proj_a.weight": ("ssm_alpha.weight", "heads"), "linear_attn.in_proj_b.weight": ("ssm_beta.weight", "heads"),
                    "linear_attn.conv1d.weight": ("ssm_conv1d.weight", "conv"), "linear_attn.dt_bias": ("ssm_dt.bias", "heads"),
                    "linear_attn.A_log": ("ssm_a", "a"), "linear_attn.norm.weight": ("ssm_norm.weight", "plain"),
                    "linear_attn.out_proj.weight": ("ssm_out.weight", "columns"),
                    "self_attn.q_proj.weight": ("attn_q.weight", "plain"), "self_attn.k_proj.weight": ("attn_k.weight", "plain"),
                    "self_attn.v_proj.weight": ("attn_v.weight", "plain"), "self_attn.o_proj.weight": ("attn_output.weight", "plain"),
                    "self_attn.q_norm.weight": ("attn_q_norm.weight", "norm"), "self_attn.k_norm.weight": ("attn_k_norm.weight", "norm"),
                    "self_attn.indexer.q_layernorm.weight": ("indexer.q_norm.weight", "norm"),
                    "self_attn.indexer.k_layernorm.weight": ("indexer.k_norm.weight", "norm"),
                    "mlp.gate.weight": ("ffn_gate_inp.weight", "plain"), "mlp.shared_expert_gate.weight": ("ffn_gate_inp_shexp.weight", "vector"),
                    "mlp.shared_expert.gate_proj.weight": ("ffn_gate_shexp.weight", "plain"),
                    "mlp.shared_expert.up_proj.weight": ("ffn_up_shexp.weight", "plain"),
                    "mlp.shared_expert.down_proj.weight": ("ffn_down_shexp.weight", "plain"),
                    "ple.key_proj.weight": ("ple_key.weight", "plain"), "ple.value_proj.weight": ("ple_value.weight", "plain"),
                    "ple.norm_key.weight": ("ple_norm_key.weight", "norm"), "ple.norm_query.weight": ("ple_norm_query.weight", "norm"),
                    "ple.norm_conv.weight": ("ple_norm_conv.weight", "norm"), "ple.conv1d.weight": ("ple_conv1d.weight", "ple_conv"),
                }
                if suffix == "self_attn.indexer.index_qk_proj.weight":
                    q = self.config.indexer_n_heads * self.config.indexer_head_dim
                    names = (prefix + "indexer.q_proj.weight", prefix + "indexer.k_proj.weight")
                    self._shape(names[0], (shape[1], q))
                    self._shape(names[1], (shape[1], shape[0] - q))
                    return None if validate else np.concatenate([self.source.read(part).reshape(-1, shape[1]) for part in names])
                if suffix.startswith("ple.ple_embedding."):
                    values = self.ple_buffers[name]
                    if len(values) != shape[0]:
                        raise ValueError("unexpected PLE integer metadata length")
                    return None if validate else np.asarray(values, dtype=np.int64)
                source, kind = fields[suffix]
                source = prefix + source
        else:
            raise ValueError("unmapped HF Qwen4Exp parameter " + name)
        stored_shape = tuple(reversed(shape))
        if kind in ("conv", "ple_conv"):
            stored_shape = (shape[2], shape[0])
        elif kind == "vector":
            stored_shape = (shape[1],)
        elif kind == "norm" and len(shape) == 1:
            tensor = self.source.tensors[source][1]
            if np.prod(tensor.shape) == shape[0]:
                stored_shape = tuple(tensor.shape)
        self._shape(source, stored_shape)
        if validate:
            return None
        values = self.source.read(source).reshape(shape if kind not in ("conv", "ple_conv") else (shape[0], shape[2]))
        c, order = self.config, self.head_order
        if kind == "norm":
            stored = values
            values = stored - np.float32(1)
            error = np.abs((values + np.float32(1)) - stored)
            self.mapping_roundtrip["norm_max_abs"] = max(self.mapping_roundtrip["norm_max_abs"], float(error.max()))
            if not np.all(error <= 8 * np.finfo(np.float32).eps * np.maximum(1, np.abs(stored))):
                raise ValueError("norm-offset inverse exceeds declared float32 bound")
        elif kind in ("heads", "a"):
            values = values.reshape(c.linear_num_value_heads, -1)[order].reshape(shape)
            if kind == "a":
                stored = values
                if not np.all(np.isfinite(stored) & (stored < 0)):
                    raise ValueError("negative-exp A tensor needs finite negative values")
                values = np.log(-stored.astype(np.float64)).astype(np.float32)
                error = np.abs((-np.exp(values) - stored) / stored)
                self.mapping_roundtrip["a_max_relative"] = max(self.mapping_roundtrip["a_max_relative"], float(error.max()))
                if not np.all(error <= 8 * np.finfo(np.float32).eps):
                    raise ValueError("log-exp A inverse exceeds declared float32 bound")
        elif kind in ("qkv", "conv"):
            first = 2 * c.linear_num_key_heads * c.linear_key_head_dim
            tail = values[first:].reshape(c.linear_num_value_heads, c.linear_value_head_dim, -1)[order]
            values = np.concatenate((values[:first], tail.reshape(-1, values.shape[1]))).reshape(shape)
        elif kind == "columns":
            values = values.reshape(shape[0], c.linear_num_value_heads, c.linear_value_head_dim)[:, order].reshape(shape)
        return np.ascontiguousarray(values.reshape(shape))

    def load(self, module, prefix):
        state = {name: self.torch.from_numpy(self._parts(prefix + name, tuple(value.shape)))
                 for name, value in module.state_dict().items()}
        module.load_state_dict(state, strict=True, assign=True)
        if any(t.is_meta for t in list(module.parameters()) + list(module.buffers())):
            raise ValueError("reference module still has meta weights: " + prefix)

    def release(self, module):
        module.to("meta")

    def _validate_input_ids(self, input_ids, previous=0, minimum=1):
        if input_ids.device.type != "cpu" or input_ids.dtype != self.torch.long or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("reference takes one CPU int64 token row")
        if input_ids.shape[1] < minimum or previous + input_ids.shape[1] > self.max_tokens:
            raise ValueError("reference token limit exceeded")
        if input_ids.min() < 0 or input_ids.max() >= self.config.vocab_size:
            raise ValueError("reference token ID outside vocabulary")

    @contextmanager
    def _forward_scope(self, on_load=None, include_head=True):
        torch = self.torch
        hooks = []
        try:
            for embedding in self.ple_embeddings:
                def before(module, args):
                    values, ids = self.source.embedding("per_layer_token_embd.weight", args[0])
                    module.weight = torch.from_numpy(values)
                    return (torch.from_numpy(ids),)
                def after(module, args, output):
                    module.weight = torch.empty(0, module.embedding_dim)
                hooks.append(embedding.register_forward_pre_hook(before))
                hooks.append(embedding.register_forward_hook(after))
            modules = self.modules if include_head else self.modules[:-1]
            with loaded_modules(modules, self.load, self.release, on_load), torch.inference_mode():
                yield
        finally:
            for hook in hooks:
                hook.remove()
            for embedding in self.ple_embeddings:
                embedding.weight = torch.empty(0, embedding.embedding_dim)

    def forward(self, input_ids, past_key_values=None, use_cache=False, on_load=None, logits_to_keep=0):
        previous = past_key_values.get_seq_length() if past_key_values is not None else 0
        self._validate_input_ids(input_ids, previous)
        with self._forward_scope(on_load):
            return self.model(input_ids=input_ids, past_key_values=past_key_values, use_cache=use_cache,
                              logits_to_keep=logits_to_keep, return_dict=True)

    def sequence_nll(self, input_ids, rows=256, on_load=None):
        """Score every next-token target with bounded rows from the original HF head."""
        if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
            raise ValueError("NLL row count must be a positive integer")
        self._validate_input_ids(input_ids, minimum=2)
        torch = self.torch
        with self._forward_scope(on_load, include_head=False):
            hidden = self.model.model(input_ids=input_ids, use_cache=False, return_dict=True).last_hidden_state[0, :-1]
        targets = input_ids[0, 1:]
        head = self.model.lm_head
        try:
            self.load(head, "lm_head.")
            if on_load:
                on_load("lm_head", head)
            parts = []
            with torch.inference_mode():
                for start in range(0, hidden.shape[0], rows):
                    logits = head(hidden[start:start + rows]).double()
                    if not torch.isfinite(logits).all():
                        raise ValueError("nonfinite file-exact HF output")
                    chunk_targets = targets[start:start + rows]
                    parts.append(torch.logsumexp(logits, dim=-1) - logits.gather(1, chunk_targets[:, None]).squeeze(1))
                    del logits
            return torch.cat(parts)
        finally:
            self.release(head)


def score_nll(torch, logits, input_ids):
    """Next-token NLL with float64 scoring of the original logits, as gen_baseline.gen_perplexity does."""
    if logits.ndim != 2 or input_ids.ndim != 1 or logits.shape[0] != input_ids.numel():
        raise ValueError("NLL scoring needs one logit row per input token")
    values = logits[:-1].double()
    return torch.logsumexp(values, dim=-1) - values.gather(1, input_ids[1:, None]).squeeze(1)


def output_values(torch, logits, input_ids, top=0):
    """Summarize native all/tail logits without treating omitted positions as scored targets."""
    if logits.ndim != 2 or input_ids.ndim != 1 or not 0 < logits.shape[0] <= input_ids.numel():
        raise ValueError("output needs all or tail logit rows of one token sequence")
    if not 0 <= top <= logits.shape[1]:
        raise ValueError("top count outside vocabulary")
    start = input_ids.numel() - logits.shape[0]
    nll = score_nll(torch, logits, input_ids[start:])
    mean = nll.mean() if nll.numel() else None
    perplexity = mean.exp() if mean is not None else None
    if not torch.isfinite(logits).all() or not torch.isfinite(nll).all() or (perplexity is not None and not torch.isfinite(perplexity)):
        raise ValueError("nonfinite file-exact HF output")
    document = {"positions": list(range(start, input_ids.numel())), "output_coverage": "tail" if start else "all",
                "scoring_scope": "tail next-token targets" if start else "whole-sequence next-token targets",
                "nll_target_positions": list(range(start + 1, input_ids.numel())), "n_scored": nll.numel(),
                "nll": nll.tolist(), "mean_nll": mean.item() if mean is not None else None,
                "perplexity": perplexity.item() if perplexity is not None else None}
    if top:
        targets = input_ids[start + 1:]
        document.update(nll_target_ids=targets.tolist(),
                        nll_target_logits=logits[:-1].gather(1, targets[:, None]).squeeze(1).tolist())
        rankings = []
        for row in logits:
            cutoff = torch.topk(row, top).values[-1]
            above = torch.nonzero(row > cutoff, as_tuple=False).flatten().tolist()
            tied = torch.nonzero(row == cutoff, as_tuple=False).flatten()[:top - len(above)].tolist()
            rankings.append(sorted(above + tied, key=lambda token: (-row[token].item(), token)))
        document.update(top_ids=rankings, top_logits=[[row[token].item() for token in ranking]
                                                     for row, ranking in zip(logits, rankings)])
    else:
        document["logits"] = logits.tolist()
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--gguf", required=True, nargs="+", type=Path)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input-ids", help="JSON list of token IDs")
    inputs.add_argument("--input-ids-file", type=Path, help="file containing a JSON list of token IDs")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-decoded-mib", type=int, default=3072)
    parser.add_argument("--attention", choices=("eager", "sdpa"), default="eager")
    parser.add_argument("--last", type=int, default=0, help="native HF tail logit rows; 0 keeps all positions")
    parser.add_argument("--top", type=int, default=0, help="compact top candidates per output row; 0 keeps all logits")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists: " + str(args.output))
    if not args.output.parent.is_dir():
        parser.error("output parent directory does not exist")
    if args.threads <= 0 or args.max_tokens <= 0 or args.max_decoded_mib <= 0:
        parser.error("thread and memory/token limits must be positive")
    if args.last < 0 or args.top < 0:
        parser.error("--last and --top must be nonnegative")
    try:
        ids = json.loads(args.input_ids_file.read_text(encoding="utf-8") if args.input_ids_file else args.input_ids)
    except (OSError, ValueError) as error:
        parser.error("cannot read input IDs: " + str(error))
    if not isinstance(ids, list) or not ids or any(not isinstance(token, int) or isinstance(token, bool) for token in ids):
        parser.error("--input-ids must be a nonempty JSON list of integer token IDs")
    if len(ids) > args.max_tokens:
        parser.error("input exceeds --max-tokens")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    vocab = config.get("text_config", config)["vocab_size"]
    if min(ids) < 0 or max(ids) >= vocab:
        parser.error("input token ID outside configured vocabulary")
    if args.top > vocab:
        parser.error("--top exceeds configured vocabulary")
    runner = FileReference(args.gguf, config, args.max_tokens, args.max_decoded_mib << 20, attention=args.attention)
    torch = runner.torch
    torch.set_num_threads(args.threads)
    ids = torch.tensor([ids], dtype=torch.long)
    def progress(name, module):
        print("HF loaded " + name, file=sys.stderr, flush=True)
    result = runner.forward(ids, on_load=progress, logits_to_keep=args.last).logits[0]
    values = output_values(torch, result, ids[0], args.top)
    document = {"weights": runner.source.labels(), "config_sha256": reference._sha256_file(args.config),
                "model_revision": reference.MODEL_REVISION, "transformers_release_commit": reference.TRANSFORMERS_COMMIT,
                "transformers_source_sha256": reference.SOURCE_SHA256, "torch_version": torch.__version__,
                "attention_source_sha256": runner.attention_source_sha256,
                "transformers_version": reference.TRANSFORMERS_VERSION, "dtype": "float32", "device": "cpu", "attention": args.attention,
                "experts_implementation": "eager",
                "operation_mode": "original HF full forward with native logits_to_keep",
                "nll_scoring": "float64 logsumexp minus next-token logit",
                "mapping": "tests/spec_decode.py; GGUF V-head order, norm offsets and negative-exp A restored for original HF modules",
                "mapping_limits": "A_log uses float64 log then float32 rounding; HF exp and norm offset restoration may round",
                "mapping_roundtrip": runner.mapping_roundtrip,
                "threads": args.threads, "input_ids": ids[0].tolist(), "last": args.last, "top": args.top, **values,
                "max_tokens": args.max_tokens, "max_decoded_bytes": args.max_decoded_mib << 20, "peak_rss_gib": peak_gib()}
    staged = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output.parent, delete=False) as out:
            staged = Path(out.name)
            json.dump(document, out, separators=(",", ":"), allow_nan=False)
            out.write("\n")
        os.link(staged, args.output)
    finally:
        if staged is not None:
            staged.unlink()


if __name__ == "__main__":
    main()
