"""Generate a tiny Qwen4-Exp reference from the pinned Transformers model."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import tempfile


MODEL_REPO = "Qwen/Qwen3.8-Flash-Next"
MODEL_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
TRANSFORMERS_VERSION = "5.16.1"
TRANSFORMERS_COMMIT = "93c8b7b485963a10800c91f55304db6be211c2bd"
TORCH_VERSION = "2.11.0+cpu"
NUMPY_VERSION = "2.2.6"
SAFETENSORS_VERSION = "0.8.0"
SOURCE_SHA256 = {
    "cache_utils.py": "4b284431cb3a881b6e6f8b8c6430df6f2efdcb3366a2484c7984ae88c612c61a",
    "modeling_qwen4_exp.py": "77fec77d87f2a0eb23b95fa04276fb5779698a7c7f523cf5061e49c118bcc459",
    "configuration_qwen4_exp.py": "26b47995740e3bc596b44b2011ee6c3d971d46136438b00dd5fad9557bec4254",
}
SEED = 380_2027
OUTPUT_NAME = "qwen4exp_reference.json"
CONFIG_NAME = "qwen4exp_config.json"
WEIGHTS_NAME = "qwen4exp_weights.safetensors"
PLE_NONZERO_PARAMETER = "model.layers.0.ple.conv1d.weight"

# The reduced config retains both layer kinds, four residual streams, routed and shared experts, PLE, and QSA.
# Thirteen tokens reach the oldest PLE convolution tap and leave a one-token QSA tail.
TINY_CONFIG = {
    "vocab_size": 32,
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 8,
    "max_position_embeddings": 32,
    "initializer_range": 0.02,
    "rms_norm_eps": 1e-6,
    "use_cache": True,
    "tie_word_embeddings": False,
    "hidden_act": "silu",
    "output_gate_type": "sigmoid",
    "rope_parameters": {
        "rope_theta": 10000.0,
        "rope_type": "default",
        "partial_rotary_factor": 0.5,
        "mrope_section": [1, 1, 0],
    },
    "attention_bias": False,
    "attention_dropout": 0.0,
    "hc_count": 4,
    "hc_lowrank": 4,
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 8,
    "linear_value_head_dim": 8,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 4,
    "moe_intermediate_size": 8,
    "shared_expert_intermediate_size": 7,
    "num_experts_per_tok": 2,
    "num_experts": 4,
    "norm_topk_prob": True,
    "layer_types": ["linear_attention", "qwen_sparse_attention"],
    "ple_layer_ids": [1],
    "ple_embed_dim": 8,
    "ple_conv_kernel_size": 4,
    "ngram_size": 3,
    "heads_per_ngram": 2,
    "ngram_vocab_size_base": 17,
    "make_ngram_vocab_size_divisible_by": 4,
    "seed": 3802027,
    "split_ngram_parts": 1,
    "indexer_n_heads": 4,
    "indexer_kv_heads": 1,
    "indexer_head_dim": 8,
    "indexer_budget": 4,
    "indexer_compress_ratio": 2,
    "bos_token_id": 1,
    "eos_token_id": 31,
    "pad_token_id": 0,
}
EOS = TINY_CONFIG["eos_token_id"]
CASES = (
    {"name": "ordinary_partial_block", "input_ids": [2, 5, 7, 11, 13, 17, 19, 23, 3, 6, 10, 14, 18], "decode_token": 22},
    {"name": "eos_in_history", "input_ids": [3, 8, 12, 16, EOS, 14, 18, 22, 24, 26, 28, 29, 30], "decode_token": 25},
    {"name": "terminal_eos", "input_ids": [4, 9, 15, 21, 27, 6, 20, 25, 2, 7, 11, 17, EOS], "decode_token": None},
)
SLICE_LENGTHS = (3, 2, 4, 4)


def selection_counts(config=TINY_CONFIG, sequence_length=13):
    """Describe budgeted complete blocks plus each query's uncompressed tail."""
    ratio = config["indexer_compress_ratio"]
    budget = config["indexer_budget"]
    if ratio <= 0 or budget <= 0 or budget % ratio:
        raise ValueError("QSA budget must be a positive multiple of its compression ratio")
    block_topk = budget // ratio
    counts = []
    for visible in range(1, sequence_length + 1):
        complete, tail = divmod(visible, ratio)
        counts.append((min(block_topk, complete) * ratio + tail, visible))
    return counts


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_environment(torch_version, transformers_version, numpy_version,
                       safetensors_version, source_hashes, optional_modules=()):
    """Refuse unpinned reference dependencies before creating output files."""
    expected = (
        ("Transformers", transformers_version, TRANSFORMERS_VERSION),
        ("Torch", torch_version, TORCH_VERSION),
        ("NumPy", numpy_version, NUMPY_VERSION),
        ("safetensors", safetensors_version, SAFETENSORS_VERSION),
    )
    for name, found, wanted in expected:
        if found != wanted:
            raise RuntimeError("Qwen4-Exp reference requires %s %s, found %s" % (name, wanted, found))
    if source_hashes != SOURCE_SHA256:
        raise RuntimeError("installed Qwen4-Exp source SHA-256 does not match the pinned Transformers source")
    accelerated = sorted(set(optional_modules) & {"fla", "flash_linear_attention", "causal_conv1d", "kernels"})
    if accelerated:
        raise RuntimeError("Qwen4-Exp reference refuses optional accelerated modules: %s" % ", ".join(accelerated))


def _require_finite(value, name):
    """Refuse nonfinite reference logits without imposing a tolerance."""
    if isinstance(value, (list, tuple)):
        for child in value:
            _require_finite(child, name)
    elif not math.isfinite(float(value)):
        raise RuntimeError("HF Qwen4-Exp produced a non-finite %s value" % name)


def _publish_staged(stage, output_dir):
    """Replace reference outputs only after all three staged files exist."""
    names = (CONFIG_NAME, WEIGHTS_NAME, OUTPUT_NAME)
    staged = [Path(stage) / name for name in names]
    missing = [path.name for path in staged if not path.is_file()]
    if missing:
        raise RuntimeError("Qwen4-Exp staging is incomplete: %s" % ", ".join(missing))
    for source in staged:
        os.replace(source, Path(output_dir) / source.name)


def _canonical_weight_hash(state_dict, torch):
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        header = json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode("utf-8")
        digest.update(struct.pack("<Q", len(header)))
        digest.update(header)
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _max_abs(left, right):
    return float((left.float() - right.float()).abs().max().item())


def _topk_margins(scores, k, dim):
    """Compute margins from the exact score tensors HF passes to topk."""
    values = scores.detach().float().cpu().movedim(dim, -1).reshape(-1, scores.shape[dim]).tolist()
    margins = []
    for row in values:
        ranked = sorted(row, reverse=True)
        if len(ranked) > k:
            margins.append(float(ranked[k - 1] - ranked[k]))
    return margins


class _TopKTrace:
    """Observe top-k inputs without changing HF operations."""

    def __init__(self):
        from torch.utils._python_dispatch import TorchDispatchMode

        self.owner_stack = []
        self.records = []
        owner_stack = self.owner_stack
        records = self.records

        class Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                output = func(*args, **kwargs)
                if str(func) == "aten.topk.default" and owner_stack and args:
                    k = int(args[1]) if len(args) > 1 else int(kwargs["k"])
                    dim = int(args[2]) if len(args) > 2 else int(kwargs.get("dim", -1))
                    records.append((owner_stack[-1], args[0].detach(), k, dim))
                return output

        self.mode = Mode()
        self.handles = []

    def attach(self, model):
        for name, module in model.named_modules():
            if name.endswith(".indexer"):
                label = "indexer"
            elif name.endswith(".mlp.gate"):
                label = "router"
            else:
                continue

            def before(_module, _inputs, label=label):
                self.owner_stack.append(label)

            def after(_module, _inputs, output, label=label):
                if not self.owner_stack or self.owner_stack[-1] != label:
                    raise RuntimeError("Qwen4-Exp top-k trace stack is inconsistent")
                self.owner_stack.pop()
                return output

            self.handles.append(module.register_forward_pre_hook(before))
            self.handles.append(module.register_forward_hook(after, always_call=True))

    def margins(self):
        result = {"indexer": [], "router": []}
        for owner, scores, k, dim in self.records:
            result[owner].extend(_topk_margins(scores, k, dim))
        return result

    def calls(self, owner):
        calls = {}
        for name, scores, k, dim in self.records:
            if name == owner:
                key = (tuple(scores.shape), k, dim)
                calls[key] = calls.get(key, 0) + 1
        return ", ".join("%d calls shape=%s k=%d dim=%d" % (count, shape, k, dim)
                         for (shape, k, dim), count in sorted(calls.items())) or "no top-k calls"

    def close(self):
        for handle in self.handles:
            handle.remove()


def _capture_indexer_trace(model, torch):
    """Capture HF's indexer projections and its final selected-token mask."""
    records = {
        "query_projection_pre_norm_rope": [],
        "raw_index_keys_pre_pool_norm_rope": [],
        "selected_token_ids": [],
        "mask_calls": [],
    }
    handles = []
    indexers = []
    for name, module in model.named_modules():
        if name.endswith(".indexer"):
            indexers.append((name, module))
    if len(indexers) != 1:
        raise RuntimeError("tiny HF fixture requires exactly one QSA indexer")
    name, indexer = indexers[0]
    projection = getattr(indexer, "index_qk_proj", None)
    if projection is None:
        raise RuntimeError("pinned HF QSA indexer lacks index_qk_proj")
    query_width = indexer.index_n_heads * indexer.index_head_dim
    key_width = indexer.index_kv_heads * indexer.index_head_dim

    def capture_projection(_module, _inputs, output):
        if not hasattr(output, "detach") or output.ndim != 3 or output.shape[0] != 1:
            raise RuntimeError("pinned HF index_qk_proj output layout changed")
        if output.shape[-1] != query_width + key_width or indexer.index_kv_heads != 1:
            raise RuntimeError("pinned HF QSA query/key projection widths changed")
        query, raw_keys = torch.split(output.detach(), [query_width, key_width], dim=-1)
        query = query.reshape(1, output.shape[1], indexer.index_n_heads, indexer.index_head_dim)
        raw_keys = raw_keys.reshape(1, output.shape[1], indexer.index_kv_heads, indexer.index_head_dim).squeeze(2)
        records["query_projection_pre_norm_rope"].extend(query[0].cpu().tolist())
        records["raw_index_keys_pre_pool_norm_rope"].extend(raw_keys[0].cpu().tolist())

    def capture_mask(_module, _inputs, output):
        if not hasattr(output, "detach") or output.ndim != 4 or output.shape[0:2] != (1, 1):
            raise RuntimeError("pinned HF indexer selected-mask layout changed")
        mask = output.detach().cpu()
        selected = mask == 0 if mask.is_floating_point() else mask.bool()
        rows = [torch.nonzero(selected[0, 0, query], as_tuple=False).flatten().tolist()
                for query in range(selected.shape[-2])]
        records["selected_token_ids"].extend(rows)
        records["mask_calls"].append({"module": name, "selected_ids": rows})
        return output

    handles.append(projection.register_forward_hook(capture_projection))
    handles.append(indexer.register_forward_hook(capture_mask))
    return records, handles


def _forward(model, input_ids, torch, cache=None, use_cache=True, ngram_ids=None):
    ids = torch.tensor([input_ids], dtype=torch.long)
    handles = []
    captured = []
    if ngram_ids is not None:
        def capture(module, args):
            indices = args[0]
            if indices.dtype != torch.long or indices.shape[:2] != (1, len(input_ids)):
                raise RuntimeError("HF n-gram lookup changed its integer type or row shape")
            captured.append(indices[0].detach().cpu().tolist())
        for name, module in model.named_modules():
            if name.endswith(".ple_embedding.ngram_embedding"):
                handles.append(module.register_forward_pre_hook(capture))
        if len(handles) != 1:
            for handle in handles:
                handle.remove()
            raise RuntimeError("tiny HF fixture requires exactly one n-gram embedding")
    try:
        result = model(input_ids=ids, past_key_values=cache, use_cache=use_cache, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    if ngram_ids is not None:
        if len(captured) != 1:
            raise RuntimeError("HF did not call its n-gram embedding exactly once")
        ngram_ids.extend(captured[0])
    return result.logits[0].detach().cpu(), result.past_key_values


def _persistence_reference(model, torch):
    """Run independent uncached prefix and full inputs with the same frozen weights."""
    overrides = {"max_position_embeddings": 512}
    config = type(model.config)(**dict(TINY_CONFIG, **overrides))
    config._attn_implementation = "eager"
    reference = type(model)(config).cpu().float().eval()
    reference.load_state_dict(model.state_dict(), strict=True)
    input_ids = [(i * 7 + i // 13) % EOS for i in range(269)]
    prefix_tokens = 257
    with torch.no_grad():
        prefix, _ = _forward(reference, input_ids[:prefix_tokens], torch, use_cache=False)
        full, _ = _forward(reference, input_ids, torch, use_cache=False)
    if prefix.shape != (prefix_tokens, TINY_CONFIG["vocab_size"]) or full.shape != (len(input_ids), TINY_CONFIG["vocab_size"]):
        raise RuntimeError("HF persistence reference changed its logit shape")
    _require_finite(prefix.tolist(), "persistence prefix logit")
    _require_finite(full.tolist(), "persistence full logit")
    return {
        "config_overrides": overrides,
        "input_ids": input_ids,
        "prefix_tokens": prefix_tokens,
        "prefix_logits": prefix[-1].tolist(),
        "full_logits": full[-1].tolist(),
        "max_abs_prefix_vs_full_prefix": _max_abs(prefix[-1], full[prefix_tokens - 1]),
    }


def _capture_intermediate_outputs(model):
    """Capture pinned HF module inputs and outputs without changing their values."""
    captured = {}
    handles = []
    modules = dict(model.named_modules())
    required = (
        "model.layers.0.ple",
        "model.layers.0.ple.ple_embedding.ngram_embedding",
        "model.layers.1.attn_hyper_connection",
    )
    missing = [name for name in required if name not in modules]
    if missing:
        raise RuntimeError("pinned HF intermediate modules are missing: %s" % ", ".join(missing))

    def tensor_values(value):
        if not hasattr(value, "detach"):
            raise RuntimeError("pinned HF intermediate is not a tensor")
        return value.detach().cpu().tolist()

    def ple_before(_module, inputs):
        if len(inputs) < 2:
            raise RuntimeError("pinned HF PLE input layout changed")
        captured["ple_input"] = tensor_values(inputs[0])

    def ple_after(_module, _inputs, output):
        captured["ple_output"] = tensor_values(output)

    def embeddings_after(_module, _inputs, output):
        captured["ngram_embeddings"] = tensor_values(output)

    def attention_hc_after(_module, _inputs, output):
        if not isinstance(output, tuple) or len(output) != 3:
            raise RuntimeError("pinned HF attention hyper-connection output is not a triple")
        captured["attention_hyper_connection"] = {
            "mixed_input": tensor_values(output[0]),
            "original_streams": tensor_values(output[1]),
            "injection_gates": tensor_values(output[2]),
        }

    handles.append(modules[required[0]].register_forward_pre_hook(ple_before))
    handles.append(modules[required[0]].register_forward_hook(ple_after))
    handles.append(modules[required[1]].register_forward_hook(embeddings_after))
    handles.append(modules[required[2]].register_forward_hook(attention_hc_after))
    return captured, handles


def _case_reference(model, spec, torch, trace):
    input_ids = spec["input_ids"]
    full_ngram = []
    intermediates, intermediate_handles = _capture_intermediate_outputs(model)
    full_indexer, full_indexer_handles = _capture_indexer_trace(model, torch)
    try:
        full, _ = _forward(model, input_ids, torch, use_cache=False, ngram_ids=full_ngram)
    finally:
        for handle in intermediate_handles:
            handle.remove()
        for handle in full_indexer_handles:
            handle.remove()
    expected_intermediates = {
        "ple_input", "ngram_embeddings", "ple_output", "attention_hyper_connection",
    }
    if set(intermediates) != expected_intermediates:
        raise RuntimeError("pinned HF did not produce every requested intermediate output")

    sliced_parts = []
    sliced_ngram = []
    cache = None
    offset = 0
    sliced_indexer, sliced_indexer_handles = _capture_indexer_trace(model, torch)
    try:
        for length in SLICE_LENGTHS:
            part, cache = _forward(model, input_ids[offset:offset + length], torch, cache, ngram_ids=sliced_ngram)
            sliced_parts.append(part)
            offset += length
    finally:
        for handle in sliced_indexer_handles:
            handle.remove()
    if offset != len(input_ids):
        raise RuntimeError("slice plan does not cover the fixture input")
    sliced = torch.cat(sliced_parts, dim=0)

    cached_rows = []
    cached_ngram = []
    cache = None
    cached_indexer, cached_indexer_handles = _capture_indexer_trace(model, torch)
    try:
        for token in input_ids:
            row, cache = _forward(model, [token], torch, cache, ngram_ids=cached_ngram)
            cached_rows.append(row[-1])
    finally:
        for handle in cached_indexer_handles:
            handle.remove()
    cached = torch.stack(cached_rows)

    decode = None
    decode_ngram = []
    decode_indexer = None
    if spec["decode_token"] is not None:
        decode_indexer, handles = _capture_indexer_trace(model, torch)
        try:
            decode, _ = _forward(model, [spec["decode_token"]], torch, cache, ngram_ids=decode_ngram)
        finally:
            for handle in handles:
                handle.remove()
        decode = decode[-1]

    if full_ngram != sliced_ngram or full_ngram != cached_ngram:
        raise RuntimeError("HF n-gram lookup IDs differ across full, sliced and token-by-token calls")

    _require_finite(full.tolist(), "full-forward logit")
    _require_finite(sliced.tolist(), "sliced-forward logit")
    _require_finite(cached.tolist(), "cached logit")
    if decode is not None:
        _require_finite(decode.tolist(), "decode logit")

    margin = trace.margins()
    selected = [call["selected_ids"] for call in cached_indexer["mask_calls"]]
    if decode_indexer is not None:
        selected.extend(call["selected_ids"] for call in decode_indexer["mask_calls"])
    indexed = [row for call in selected for row in call]
    expected_counts = selection_counts(sequence_length=len(input_ids))
    for query, (selected_ids, (expected, visible)) in enumerate(zip(indexed, expected_counts)):
        if len(selected_ids) != expected or any(token >= visible for token in selected_ids):
            raise RuntimeError("HF indexer mask at query %d does not match its block budget and tail" % query)
    if spec["name"] == "ordinary_partial_block":
        final_ids = indexed[len(input_ids) - 1]
        if len(input_ids) - 1 not in final_ids:
            raise RuntimeError("HF indexer did not retain the partial trailing block")
        if len(final_ids) >= len(input_ids):
            raise RuntimeError("HF indexer failed to truncate its candidate blocks")
    for label, capture, count in (
        ("full", full_indexer, len(input_ids)),
        ("sliced", sliced_indexer, len(input_ids)),
        ("cached", cached_indexer, len(input_ids)),
    ):
        for key in ("query_projection_pre_norm_rope", "raw_index_keys_pre_pool_norm_rope", "selected_token_ids"):
            if len(capture[key]) != count:
                raise RuntimeError("HF %s QSA capture %s has %d rows, expected %d" % (
                    label, key, len(capture[key]), count))
    if (full_indexer["selected_token_ids"] != sliced_indexer["selected_token_ids"] or
            full_indexer["selected_token_ids"] != cached_indexer["selected_token_ids"]):
        raise RuntimeError("HF QSA selected-token membership differs across full, sliced and cached calls")
    if decode_indexer is not None:
        for key in ("query_projection_pre_norm_rope", "raw_index_keys_pre_pool_norm_rope", "selected_token_ids"):
            if len(decode_indexer[key]) != 1:
                raise RuntimeError("HF decode QSA capture %s does not have one row" % key)
    for owner in ("router", "indexer"):
        if not margin[owner]:
            raise RuntimeError("HF %s trace has no candidate boundary (%s)" % (owner, trace.calls(owner)))
        minimum = min(margin[owner])
        if minimum <= 0:
            raise RuntimeError("HF %s has a tied top-k boundary (minimum margin %.9g; %s)" % (
                owner, minimum, trace.calls(owner)))

    return {
        "name": spec["name"],
        "input_ids": input_ids,
        "decode_token": spec["decode_token"],
        "eos_positions": [i for i, token in enumerate(input_ids) if token == EOS],
        "full_logits": full.tolist(),
        "sliced_logits": sliced.tolist(),
        "cached_logits": cached.tolist(),
        "decode_logits": None if decode is None else decode.tolist(),
        "ngram_ids": full_ngram,
        "decode_ngram_ids": decode_ngram,
        "ple_intermediates": {
            "layer": 0,
            "input": intermediates["ple_input"],
            "ngram_embeddings": intermediates["ngram_embeddings"],
            "output": intermediates["ple_output"],
        },
        "attention_hyper_connection": {
            "layer": 1,
            **intermediates["attention_hyper_connection"],
        },
        "max_abs_full_vs_cached": _max_abs(full, cached),
        "max_abs_sliced_vs_cached": _max_abs(sliced, cached),
        "full_equals_cached_bits": bool(torch.equal(full, cached)),
        "sliced_equals_cached_bits": bool(torch.equal(sliced, cached)),
        "indexer_selected_ids": selected,
        "qsa_indexer": {
            "full": {key: full_indexer[key] for key in (
                "query_projection_pre_norm_rope", "raw_index_keys_pre_pool_norm_rope", "selected_token_ids")},
            "sliced": {key: sliced_indexer[key] for key in (
                "query_projection_pre_norm_rope", "raw_index_keys_pre_pool_norm_rope", "selected_token_ids")},
            "cached": {key: cached_indexer[key] for key in (
                "query_projection_pre_norm_rope", "raw_index_keys_pre_pool_norm_rope", "selected_token_ids")},
            "decode": None if decode_indexer is None else {key: decode_indexer[key] for key in (
                "query_projection_pre_norm_rope", "raw_index_keys_pre_pool_norm_rope", "selected_token_ids")},
        },
        "minimum_router_topk_margin": min(margin["router"]),
        "minimum_indexer_topk_margin": min(margin["indexer"]),
    }


def pinned_environment():
    """torch, transformers, numpy, safetensors and the pinned sources' hashes, refused unless they are the pinned ones and no accelerated module is installed."""
    import numpy
    import torch
    import transformers
    import safetensors

    modeling = importlib.import_module("transformers.models.qwen4_exp.modeling_qwen4_exp")
    configuration = importlib.import_module("transformers.models.qwen4_exp.configuration_qwen4_exp")
    cache = importlib.import_module("transformers.cache_utils")
    source_hashes = {
        "cache_utils.py": _sha256_file(Path(cache.__file__)),
        "modeling_qwen4_exp.py": _sha256_file(Path(modeling.__file__)),
        "configuration_qwen4_exp.py": _sha256_file(Path(configuration.__file__)),
    }
    optional_modules = tuple(
        name for name in ("fla", "flash_linear_attention", "causal_conv1d", "kernels")
        if importlib.util.find_spec(name) is not None
    )
    _check_environment(
        torch.__version__, transformers.__version__, numpy.__version__,
        safetensors.__version__, source_hashes, optional_modules,
    )
    return torch, transformers, numpy, safetensors, source_hashes


def generate(output_dir):
    """Write the pinned HF model, its deterministic weights, and its logits."""
    from safetensors.torch import load_file, save_file
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM

    torch, transformers, numpy, safetensors, source_hashes = pinned_environment()
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    config = Qwen4ExpTextConfig(**TINY_CONFIG)
    config._attn_implementation = "eager"
    model = Qwen4ExpForCausalLM(config).cpu().float().eval()

    generator = torch.Generator(device="cpu").manual_seed(SEED + 1)
    parameters = dict(model.named_parameters())
    if PLE_NONZERO_PARAMETER not in parameters:
        raise RuntimeError("pinned HF model lacks %s" % PLE_NONZERO_PARAMETER)
    ple_conv = parameters[PLE_NONZERO_PARAMETER]
    with torch.no_grad():
        if torch.count_nonzero(ple_conv).item() != 0:
            raise RuntimeError("HF PLE convolution initialization changed; review its fixture weight rule")
        ple_conv.copy_(torch.randn(ple_conv.shape, generator=generator) * 0.01)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="qwen4exp_reference_", dir=out) as temporary:
        stage = Path(temporary)
        config_path = stage / CONFIG_NAME
        weights_path = stage / WEIGHTS_NAME
        config_path.write_text(json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        save_file(
            {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()},
            str(weights_path), metadata={"format": "pt"},
        )
        state_hash = _canonical_weight_hash(model.state_dict(), torch)
        state = load_file(str(weights_path), device="cpu")
        model.load_state_dict(state, strict=True)
        if _canonical_weight_hash(model.state_dict(), torch) != state_hash:
            raise RuntimeError("reloaded safetensors checkpoint changed its weights")
        del state

        all_cases = []
        for spec in CASES:
            trace = _TopKTrace()
            trace.attach(model)
            try:
                with torch.no_grad(), trace.mode:
                    record = _case_reference(model, spec, torch, trace)
            finally:
                trace.close()
            all_cases.append(record)

        ordinary = next(case for case in all_cases if case["name"] == "ordinary_partial_block")
        final_selection = ordinary["indexer_selected_ids"][len(CASES[0]["input_ids"]) - 1][0]
        if final_selection[-1] != len(CASES[0]["input_ids"]) - 1:
            raise RuntimeError("HF indexer partial-block witness is absent from the final selected ids")

        doc = {
            "_comment": "Generated by tools/qwen4exp_reference.py using pinned Transformers Qwen4ExpForCausalLM. Logits, QSA projections, raw index keys and selected-token masks are captured from HF.",
            "fixture_kind": "synthetic",
            "model_repo": MODEL_REPO,
            "model_revision": MODEL_REVISION,
            "transformers_version": transformers.__version__,
            "transformers_release_commit": TRANSFORMERS_COMMIT,
            "transformers_source_sha256": source_hashes,
            "torch_version": torch.__version__,
            "numpy_version": numpy.__version__,
            "safetensors_version": safetensors.__version__,
            "dtype": "float32",
            "device": "cpu",
            "attention": "eager",
            "optional_accelerated_modules": [],
            "seed": SEED,
            "nondefault_weight_initialization": {
                PLE_NONZERO_PARAMETER: "normal samples, standard deviation 0.01, from CPU torch.Generator seed+1; HF initializes this PLE convolution to zero"
            },
            "config_file": CONFIG_NAME,
            "config_sha256": _sha256_file(config_path),
            "weights_file": WEIGHTS_NAME,
            "weights_file_sha256": _sha256_file(weights_path),
            "weights_canonical_sha256": state_hash,
            "state_dict": {
                name: {"shape": list(value.shape), "values": value.detach().cpu().reshape(-1).tolist()}
                for name, value in sorted(model.state_dict().items())
            },
            "config": TINY_CONFIG,
            "layers": TINY_CONFIG["layer_types"],
            "slice_lengths": list(SLICE_LENGTHS),
            "qsa_witness": {
                "sequence_length": len(CASES[0]["input_ids"]),
                "compress_ratio": TINY_CONFIG["indexer_compress_ratio"],
                "indexer_budget": TINY_CONFIG["indexer_budget"],
                "visible_complete_blocks": len(CASES[0]["input_ids"]) // TINY_CONFIG["indexer_compress_ratio"],
                "visible_partial_tokens": len(CASES[0]["input_ids"]) % TINY_CONFIG["indexer_compress_ratio"],
                "selected_ids_at_final_query": final_selection,
                "selected_count": len(final_selection),
                "visible_count": len(CASES[0]["input_ids"]),
            },
            "cases": all_cases,
            "persistence_case": _persistence_reference(model, torch),
        }
        staged_artifact = stage / OUTPUT_NAME
        staged_artifact.write_text(json.dumps(doc, separators=(",", ":")) + "\n", encoding="utf-8")
        _publish_staged(stage, out)

    artifact = out / OUTPUT_NAME
    print("wrote %s (%d cases, %d parameters)" % (
        artifact, len(all_cases), sum(parameter.numel() for parameter in model.parameters())))
    return artifact
