"""Generate the tiny deepseek41 reference with DeepSeek's own inference/model.py at 1bae0e10 and the F32 kernel port.

It runs offline in the environment of tools/deepseek4_requirements.txt on a copy of DeepSeek's code checked against its pins (docs/ASSETS.md, The deepseek41 tiny reference).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import struct
import sys
import tempfile
import types

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS.parent / "tests"))
import fixture_deepseek4 as fixture  # noqa: E402

REPO = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"
# Every file of the revision the reference reads, by its git blob oid and SHA-256.
SOURCE = {
    "config.json": ("09917a9139b22d5bf8be52132787f435147b1020", "8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879"),
    "encoding/encoding.py": ("3c7c357dc9ae3ed37adacb8572a2e9498b7d1bcb", "13231c592e958fb39e2af6001dae69d90e7a36b265702b9b390236fb387599e2"),
    "inference/config.json": ("7a915cc69e21abbc6d7fb939cef5d09c72aa0d24", "2e84f45cf1dac8c7fcbb200e96667d4b913275690668ed496f24c7747207a809"),
    "inference/convert.py": ("61bd2943648920ce599c4c86dc9bf6316e152fe8", "035028340479145594a81d6084a8424e57363adf83c0d5983914783d95614d76"),
    "inference/engram.py": ("a36bc6c6b8cbf2d3dbac850863c23b55a7ecb5c7", "11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897"),
    "inference/generate.py": ("84d4873c4c59ea743e104920a77984f022bfb9f5", "8668d67f7d108e32b90d50cb0d8606889ceb2219bfe95741d84e22f70768e9f0"),
    "inference/image_processor.py": ("a50311ece615b72be2c9d4b649a05619c27681e7", "482759e3bcc4e9bb5ee582b244cc563f5d0e163d8b48dda91ebb7106e62f9272"),
    "inference/kernel.py": ("6fa7bd5aafbe1bc62887080963cc9fdd838d89f5", "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455"),
    "inference/model.py": ("46cc8310bb64852f3c36c9cc5dd13d47acabf6ee", "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d"),
    "inference/requirements.txt": ("0ac24b86509757adc25e0d7518b16fffbf81a5b1", "be82bcd5775ccf8361ddd22cac53ef7b3b83434697fc3c61c75870450fb9437e"),
    "inference/vision.py": ("77af0bdef9f4649edebac2876cd1df50a147ce98", "5d49edc196a4ef22384abe76d35a40098cbe1e74b586c8f66a2edff4f076b26c"),
    "tokenizer.json": ("6a15814dd25c934028034531da744c689f87ff21", "c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b"),
    "tokenizer_config.json": ("f3dad388a2bbfd6a8605bd02754acd86d9ca5112", "6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547"),
}
REQUIREMENTS = TOOLS / "deepseek4_requirements.txt"
PYTHON_VERSION = "3.12.13"
# Modules whose presence would put another implementation in place of DeepSeek's code or the port.
FORBIDDEN_MODULES = ("tilelang", "deep_gemm", "flash_mla", "transformers")
TOKENIZER_VOCAB = 129280
COMPRESSED_VOCAB = 99092
# The full token map's SHA-256, its ids as little-endian int32, so a file's engram.token_map is held to the reference's.
TOKEN_MAP_SHA256 = "c60a86322ec17b4142bfef3c57a8d81fb428550cdf487f88a4320cb59fe46481"

# The weights' seed, the first from 411017 whose least margin over 60 seeds was largest (3.8e-5), and the prompt's.
SEED = 411027
SEQUENCE_SEED = 411017
SEQUENCE_LENGTH = 200
WHOLE_LENGTHS = (1, 2, 3, 64, 127, 128, 129, 200)
DECODE_FROM = (1, 128, 129)
# The Engram layer and position whose six table rows are zero, so its key row is zero and its dot +0.
ZERO_KEY = (1, 5)
# The token whose raw id the sequence holds at this position, and the pad id in text.
PAD_POSITION = 9
SINKHORN_POSITIONS = (0, 1, 127, 128, 199)
# Every selection's margin at its cut must clear the tiny models' per-logit bound, and every cached row equal the whole prompt's within half of it.
MARGIN_FLOOR = 2e-5
EQUAL_BOUND = 1e-5
OUTPUT_NAME = "deepseek41_reference.json"
GGUF_NAMES = {layout: "deepseek41-tiny-%s.gguf" % layout for layout in fixture.LAYOUTS}


def sequence():
    """The 200 prompt ids: ids 3 to 127 from the fixture's generator, with the pad id 2 at PAD_POSITION."""
    rng = fixture._stream(SEQUENCE_SEED, "sequence")
    ids = [3 + (rng.next() >> 40) % 125 for _ in range(SEQUENCE_LENGTH)]
    ids[PAD_POSITION] = 2
    return ids


def text_sha256(path):
    """SHA-256 of a text file of this tree with LF line ends, whatever a checkout gave it."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def git_blob_oid(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def read_requirements(path=REQUIREMENTS):
    """The pinned name and version of each line of the hash-pinned requirements."""
    pins = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        spec, _, digest = line.partition(" --hash=sha256:")
        name, _, version = spec.partition("==")
        if not version or len(digest) != 64:
            raise RuntimeError("deepseek4 requirements line is not name==version --hash=sha256:...: %s" % line)
        pins[name.lower()] = version
    return pins


def check_environment(python_version, installed, pins):
    """Refuse a Python or any distribution other than the pinned ones."""
    if python_version != PYTHON_VERSION:
        raise RuntimeError("deepseek4 reference requires Python %s, found %s" % (PYTHON_VERSION, python_version))
    for name, version in sorted(pins.items()):
        if installed.get(name) != version:
            raise RuntimeError("deepseek4 reference requires %s %s, found %s" % (name, version, installed.get(name)))
    extra = sorted(set(installed) - set(pins) - {"pip"})
    if extra:
        raise RuntimeError("deepseek4 reference refuses unpinned distributions: %s" % ", ".join(extra))


def installed_distributions():
    return {dist.metadata["Name"].lower().replace("_", "-"): dist.version for dist in importlib.metadata.distributions()}


def check_sources(directory):
    """Every pinned file of the revision, held to its blob oid and SHA-256; returns the files' bytes."""
    files = {}
    for relative, (oid, sha256) in sorted(SOURCE.items()):
        path = Path(directory) / relative
        if not path.is_file():
            raise RuntimeError("deepseek4 source lacks %s" % relative)
        data = path.read_bytes()
        if git_blob_oid(data) != oid or hashlib.sha256(data).hexdigest() != sha256:
            raise RuntimeError("deepseek4 source %s is not the pinned file at %s" % (relative, REVISION[:8]))
        files[relative] = data
    return files


def kernel_imports(source):
    """The names a module's source imports from `kernel`."""
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module == "kernel":
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import) and any(alias.name == "kernel" for alias in node.names):
            raise RuntimeError("deepseek4 source imports kernel as a module")
    return names


def check_imports(loaded_modules, dist_initialized):
    """Refuse a run where DeepSeek's code reached another implementation or a process group."""
    present = sorted(name for name in FORBIDDEN_MODULES if name in loaded_modules)
    if present:
        raise RuntimeError("deepseek4 reference refuses modules in sys.modules: %s" % ", ".join(present))
    if dist_initialized:
        raise RuntimeError("deepseek4 reference refuses an initialized process group")


def publish_staged(stage, output_dir):
    """Replace the outputs only once every staged file exists."""
    names = (OUTPUT_NAME,) + tuple(GGUF_NAMES[layout] for layout in fixture.LAYOUTS)
    staged = [Path(stage) / name for name in names]
    missing = [path.name for path in staged if not path.is_file()]
    if missing:
        raise RuntimeError("deepseek4 staging is incomplete: %s" % ", ".join(missing))
    for source in staged:
        os.replace(source, Path(output_dir) / source.name)


def check_margins(margins, floor=MARGIN_FLOOR):
    """Refuse a selection whose margin at its cut is below the floor."""
    for kind, values in sorted(margins.items()):
        if values and min(values) < floor:
            raise RuntimeError("deepseek4 %s selection margin %.3g is below the floor %.3g; choose another seed" % (kind, min(values), floor))


def _f32_value(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _short(value):
    """The shortest decimal that reads back as the same f32."""
    value = _f32_value(float(value))
    for digits in range(1, 10):
        text = "%.*g" % (digits, value)
        if _f32_value(float(text)) == value:
            return float(text)
    return value


class _Adapter:
    """The tokenizer surface engram.build_compressed_token_map reads: the Rust tokenizer and its size with added tokens."""

    def __init__(self, path):
        from tokenizers import Tokenizer
        self.backend_tokenizer = Tokenizer.from_file(str(path))

    def __len__(self):
        return self.backend_tokenizer.get_vocab_size(with_added_tokens=True)


class _Trace:
    """Observes DeepSeek's forward: replaces top-k with llmx's total order (score, then lower index) and records every selection, the Sinkhorn split, the n-gram hash and Engram's gate."""

    def __init__(self, torch, model_module, kernel_port):
        from torch.overrides import TorchFunctionMode
        self.torch = torch
        self.owner = None
        self.engram_layer = None
        self.block = None
        self.position = 0
        self.in_expert = False
        self.records = {}
        self.margins = {"index": [], "candidate": [], "router": []}
        self.clamped = 0
        self.bf16_casts = 0
        self.skipped_activation_qdq = 0
        trace = self

        class Mode(TorchFunctionMode):
            def __torch_function__(self, func, types_, args=(), kwargs=None):
                return trace.dispatch(func, args, kwargs or {})

        self.mode = Mode()
        self.handles = []
        self.model_module = model_module
        self.port = kernel_port
        self.original_select = model_module.select_candidate_blocks

    def add(self, key, value):
        self.records.setdefault(key, []).append(value)

    def dispatch(self, func, args, kwargs):
        torch = self.torch
        if func in (torch.Tensor.topk, torch.topk):
            return self.topk(*args, **kwargs)
        if func is torch.Tensor.to and (any(a is torch.bfloat16 for a in args[1:]) or kwargs.get("dtype") is torch.bfloat16):
            x = args[0]
            if not torch.equal(func(*args, **kwargs).float(), x.float()):
                raise RuntimeError("a BF16 cast in DeepSeek's code would round an F32 value")
            self.bf16_casts += 1
            return x
        out = func(*args, **kwargs)
        if func is torch.clamp and self.in_expert:
            self.clamped += int((out != args[0]).sum())
        if self.engram_layer is not None and func is torch.copysign:
            self.add(("engram_dot", self.engram_layer), args[1][0].tolist())
        if self.engram_layer is not None and func is torch.sigmoid:
            self.add(("engram_gate", self.engram_layer), out[0].tolist())
        return out

    def topk(self, x, k, dim=-1, largest=True, sorted=True):
        torch = self.torch
        if not largest or dim not in (-1, x.dim() - 1):
            raise RuntimeError("deepseek4 top-k outside the last dimension")
        if torch.isnan(x).any():
            raise RuntimeError("deepseek4 top-k over a NaN score")
        values, indices = torch.sort(x, dim=-1, descending=True, stable=True)
        if values.size(-1) > k and k > 0 and self.owner is not None:
            cut, next_ = values[..., k - 1].reshape(-1), values[..., k].reshape(-1)
            finite = torch.isfinite(cut) & torch.isfinite(next_)
            self.margins[self.owner[0]].extend((cut - next_)[finite].tolist())
        return torch.return_types.topk((values[..., :k], indices[..., :k]))

    def attach(self, model):
        def pre(owner):
            def hook(_module, inputs):
                self.owner = owner
                if owner[0] == "index":
                    self.position = inputs[3]
            return hook

        def index_after(layer):
            def hook(_module, inputs, output):
                offset = inputs[4]
                rows = self.records.setdefault(("index", layer), {})
                for i, row in enumerate(output[0].tolist()):
                    rows[self.position + i] = sorted(int(v) - offset for v in row if v >= 0)
                self.owner = None
            return hook

        def router_after(layer):
            def hook(_module, _inputs, output):
                self.add(("router", layer), [sorted(row) for row in output[1].tolist()])
                self.owner = None
            return hook

        def engram_pre(layer):
            def hook(_module, _inputs):
                self.engram_layer = layer
            return hook

        def engram_after(_module, _inputs, _output):
            self.engram_layer = None

        def block_pre(layer):
            def hook(_module, _inputs):
                self.block = layer
            return hook

        for layer, block in enumerate(model.layers):
            self.handles.append(block.register_forward_pre_hook(block_pre(layer)))
            if block.attn.indexer is not None:
                self.handles.append(block.attn.indexer.register_forward_pre_hook(pre(("index", layer))))
                self.handles.append(block.attn.indexer.register_forward_hook(index_after(layer)))
            self.handles.append(block.ffn.gate.register_forward_pre_hook(pre(("router", layer))))
            self.handles.append(block.ffn.gate.register_forward_hook(router_after(layer)))
            for expert in [e for e in block.ffn.experts if e is not None] + [block.ffn.shared_experts]:
                self.handles.append(expert.register_forward_pre_hook(lambda _m, _i: setattr(self, "in_expert", True)))
                self.handles.append(expert.register_forward_hook(lambda _m, _i, _o: setattr(self, "in_expert", False)))
            if block.engram is not None:
                self.handles.append(block.engram.register_forward_pre_hook(engram_pre(layer)))
                self.handles.append(block.engram.register_forward_hook(engram_after))
        self.handles.append(model.engram_hash.register_forward_hook(
            lambda _m, _i, output: self.add(("hash",), output[0].tolist())))

        def select(logits, compress_lens, topk_blocks, block_size):
            layer = self.owner[1]
            self.owner = ("candidate", layer)
            mask = self.original_select(logits, compress_lens, topk_blocks, block_size)
            self.owner = ("index", layer)
            rows = self.records.setdefault(("candidate", layer), {})
            for i, row in enumerate(mask[0].reshape(-1, mask.size(-1))):
                rows[self.position + i] = sorted({int(j) // block_size for j in row.nonzero().flatten().tolist()})
            return mask
        self.model_module.select_candidate_blocks = select

    def sinkhorn(self, *args, **kwargs):
        pre, post, comb = self.port.hc_split_sinkhorn(*args, **kwargs)
        self.add(("sinkhorn", self.block), (pre[0].tolist(), post[0].tolist(), comb[0].tolist()))
        return pre, post, comb

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.model_module.select_candidate_blocks = self.original_select


# The trace of the run in progress, which the kernel module's wrappers report to.
_CURRENT = {"trace": None}


def _kernel_module(port):
    """The `kernel` module DeepSeek's code imports: the F32 port, with the attention output's FP8 activation quantization (model.py:787) left out, as docs/DEEPSEEK4.md section 4.5 has it."""
    module = types.ModuleType("kernel")
    module.fp4_act_quant = port.fp4_act_quant
    module.fp4_gemm = port.fp4_gemm
    module.fp8_gemm = port.fp8_gemm
    module.sparse_attn = port.sparse_attn

    def act_quant(x, *args, **kwargs):
        caller = sys._getframe(1).f_code.co_qualname
        if caller == "Attention._window_kv":
            return port.act_quant(x, *args, **kwargs)
        if caller == "Attention.forward":
            _CURRENT["trace"].skipped_activation_qdq += 1
            return x
        raise RuntimeError("deepseek4 act_quant reached from %s" % caller)

    module.act_quant = act_quant
    module.hc_split_sinkhorn = lambda *args, **kwargs: _CURRENT["trace"].sinkhorn(*args, **kwargs)
    if sorted(vars(module)) != sorted(set(port.NAMES) | {"__name__", "__doc__", "__package__", "__loader__", "__spec__"}):
        raise RuntimeError("the kernel module is not the port's names")
    return module


def pinned_environment():
    """torch and the pinned distributions, refused unless they are the pinned ones."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    check_environment(platform.python_version(), installed_distributions(), read_requirements())
    import torch
    return torch


def _load_deepseek(stage_source, torch):
    """Import DeepSeek's model.py and engram.py from the checked copy, the port standing in for kernel.py."""
    import deepseek4_kernel as port
    inference = Path(stage_source) / "inference"
    wanted = set()
    for name in ("model.py", "engram.py", "vision.py", "image_processor.py"):
        wanted |= kernel_imports((inference / name).read_text(encoding="utf-8"))
    missing = sorted(wanted - set(port.NAMES))
    if missing:
        raise RuntimeError("the kernel port lacks %s" % ", ".join(missing))
    sys.modules["kernel"] = _kernel_module(port)
    sys.path.insert(0, str(inference))
    import model
    import engram
    import torch.distributed as dist
    check_imports(sys.modules, dist.is_available() and dist.is_initialized())
    return model, engram, port


def _model_args(model_module):
    args = dict(fixture.TINY_ARGS)
    args["compress_ratios"] = tuple(args["compress_ratios"])
    for key in ("kv_source_layers", "index_source_layers", "engram_layer_ids"):
        args[key] = tuple(args[key])
    return model_module.ModelArgs(**args)


def _engram_constants(engram_module, model_args, tokenizer, torch):
    layout = engram_module.EngramLayout.from_args(model_args)
    state = engram_module.NgramHashState(model_args, layout, tokenizer)
    token_map = state.token_map.tolist()
    if len(token_map) != TOKENIZER_VOCAB or max(token_map) + 1 != COMPRESSED_VOCAB:
        raise RuntimeError("the token map is not the tokenizer's 129280 ids over 99092 compressed ids")
    digest = hashlib.sha256(struct.pack("<%di" % len(token_map), *token_map)).hexdigest()
    if digest != TOKEN_MAP_SHA256:
        raise RuntimeError("the token map's SHA-256 is %s, not the pinned one" % digest)
    vocab = fixture.TINY_ARGS["vocab_size"]
    backend = tokenizer.backend_tokenizer
    added = {token.content for token in backend.get_added_tokens_decoder().values()}
    tokens = [backend.id_to_token(i) for i in range(vocab)]
    return layout, state, {
        "layer_ids": list(layout.layer_ids),
        "primes": [[p for order in layer for p in order] for layer in layout.primes],
        "offsets": state.offsets.tolist(),
        "multipliers": state.multipliers.tolist(),
        "num_embeddings": [sum(p for order in layer for p in order) for layer in layout.primes],
        "pad_id": model_args.engram_pad_id,
        "mapped_pad_id": int(state.pad_id),
        "token_map": token_map[:vocab],
        "token_map_sha256": digest,
        "token_map_prefix_sha256": hashlib.sha256(struct.pack("<%di" % vocab, *token_map[:vocab])).hexdigest(),
        "tokens": tokens,
        "special_tokens": [i for i, token in enumerate(tokens) if token in added],
    }


def _no_convolution(model):
    """Engram ends at its gate: no parameter, module or state key of a convolution (docs/DEEPSEEK4.md, section 4.4)."""
    for layer, block in enumerate(model.layers):
        if block.engram is None:
            continue
        names = sorted(name for name, _ in block.engram.named_parameters())
        if names != ["embed.scale", "embed.weight", "k_weight", "q_weight", "wkv.weight"]:
            raise RuntimeError("Engram on layer %d has parameters %s" % (layer, names))
        if any("conv" in type(module).__name__.lower() for module in block.engram.modules()):
            raise RuntimeError("Engram on layer %d holds a convolution" % layer)
    if any("conv" in key for key in model.state_dict()):
        raise RuntimeError("the tiny state holds a convolution")
    return {"engram_parameters": ["embed.scale", "embed.weight", "k_weight", "q_weight", "wkv.weight"], "convolution": False}


def _build(model_module, model_args, tokenizer, state, torch):
    """DeepSeek's Transformer with the state loaded, every parameter F32 but the Engram tables' FP8 rows and E8M0 scales."""
    torch.manual_seed(0)
    model = model_module.Transformer(model_args, tokenizer).eval()
    for name, parameter in model.named_parameters():
        if not name.endswith(("engram.embed.weight", "engram.embed.scale")):
            parameter.data = parameter.data.float()
    tensors = {}
    for name, (shape, values) in state.items():
        value = torch.tensor(values, dtype=torch.float32).reshape(shape)
        if name.endswith("engram.embed.weight"):
            value = value.to(torch.float8_e4m3fn)
        elif name.endswith("engram.embed.scale"):
            value = value.to(torch.float8_e8m0fnu)
        tensors[name] = value
    model.load_state_dict(tensors, strict=True)
    for name, tensor in model.state_dict().items():
        if tensor.float().reshape(-1).tolist() != [_f32_value(v) for v in state[name][1]]:
            raise RuntimeError("%s did not load as the fixture's values" % name)
    full = model.head.forward
    model.head.forward = lambda x, full_logits=False: full(x, True)
    return model


def _run(model, ids, prefill, torch, trace):
    """Logits of every position: the first `prefill` ids as one prompt from position 0, then each later id as a decode step."""
    rows = []
    with trace.mode:
        _, logits, _ = model(torch.tensor([ids[:prefill]]), 0)
        rows.extend(logits[0].tolist())
        for position in range(prefill, len(ids)):
            _, logits, _ = model(torch.tensor([ids[position:position + 1]]), position)
            rows.extend(logits[0].tolist())
    return rows


def _rows(records, key):
    return [row for call in records.get(key, []) for row in call]


def _case(model_module, model_args, tokenizer, state, torch, port, ids, prefill):
    model = _build(model_module, model_args, tokenizer, state, torch)
    trace = _Trace(torch, model_module, port)
    _CURRENT["trace"] = trace
    trace.attach(model)
    try:
        logits = _run(model, ids, prefill, torch, trace)
    finally:
        trace.close()
    return logits, trace


def _selections(trace, args, length):
    """Each layer's selections by position: index and candidate ids, empty where nothing is visible yet, and each row's routed experts."""
    out = {"index": {}, "candidate": {}, "router": {}}
    for layer in range(args["n_layers"]):
        for kind in ("index", "candidate"):
            if (kind, layer) in trace.records:
                by_position = trace.records[(kind, layer)]
                out[kind][str(layer)] = [by_position.get(p, []) for p in range(length)]
        out["router"][str(layer)] = _rows(trace.records, ("router", layer))
    return out


def generate(output_dir, checkpoint):
    """Check the environment and the source, run the tiny model on every path, and publish the golden and both tiny files whole."""
    torch = pinned_environment()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="deepseek4_reference_", dir=out) as temporary:
        stage = Path(temporary)
        source = stage / "source"
        for relative in SOURCE:
            (source / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(checkpoint) / relative, source / relative)
        check_sources(source)
        model_module, engram_module, port = _load_deepseek(source, torch)
        model_args = _model_args(model_module)
        tokenizer = _Adapter(source / "tokenizer.json")
        layout, hash_state, engram = _engram_constants(engram_module, model_args, tokenizer, torch)
        model_args.engram_num_embeddings = tuple(engram["num_embeddings"])

        ids = sequence()
        hashes = hash_state(torch.tensor([ids]), 0)[0].tolist()
        layer, position = ZERO_KEY
        index = list(layout.layer_ids).index(layer)
        engram["zero_rows"] = {str(layer): sorted(hashes[position][index])}
        args = fixture.TINY_ARGS
        zero_rows = {layer: engram["zero_rows"][str(layer)]}
        state = fixture.make_state(args, engram["num_embeddings"], SEED, zero_rows)
        shapes = fixture.parameter_shapes(args, engram["num_embeddings"])

        whole, trace = _case(model_module, model_args, tokenizer, state, torch, port, ids, len(ids))
        model = _build(model_module, model_args, tokenizer, state, torch)
        no_conv = _no_convolution(model)
        if sorted(model.state_dict()) != sorted(state) or set(shapes) | {n for n in state if n.endswith("embed.scale")} != set(state):
            raise RuntimeError("the fixture's parameters are not the model's")
        if _rows(trace.records, ("hash",)) != hashes:
            raise RuntimeError("the model's n-gram ids are not NgramHashState's")
        check_margins(trace.margins)
        if trace.clamped == 0:
            raise RuntimeError("the SwiGLU clamp never engaged")
        selections = _selections(trace, args, len(ids))

        runs = []
        for length in WHOLE_LENGTHS:
            logits, other = _case(model_module, model_args, tokenizer, state, torch, port, ids[:length], length)
            runs.append(_compare("whole", length, logits, whole, _selections(other, args, length), selections))
        for prefill in DECODE_FROM:
            logits, other = _case(model_module, model_args, tokenizer, state, torch, port, ids, prefill)
            runs.append(_compare("prefill_then_decode", prefill, logits, whole, _selections(other, args, len(ids)), selections))
            check_margins(other.margins)

        gates = {}
        for engram_layer in layout.layer_ids:
            gate_rows = _rows(trace.records, ("engram_gate", engram_layer))
            dot_rows = _rows(trace.records, ("engram_dot", engram_layer))
            gates[str(engram_layer)] = {str(p): gate_rows[p] for p in (0, 1, 2, 3, position)}
            if engram_layer == layer:
                zero_dot = dot_rows[position]
        expected = torch.sigmoid(torch.tensor(1e-6, dtype=torch.float32).sqrt()).item()
        if any(struct.pack("<f", d) != struct.pack("<f", 0.0) for d in zero_dot):
            raise RuntimeError("the zero key row's dots are not all +0: %s" % zero_dot)
        if gates[str(layer)][str(position)] != [expected] * args["hc_mult"]:
            raise RuntimeError("the zero key row's gate is not sigmoid(+1e-3)")
        mapped = next(i for i in ids if engram["token_map"][i] != i)

        sinkhorn = {}
        for block in range(args["n_layers"]):
            attn, ffn = trace.records[("sinkhorn", block)]
            sinkhorn[str(block)] = {part: {str(p): {"pre": record[0][p], "post": record[1][p], "comb": record[2][p]} for p in SINKHORN_POSITIONS}
                                    for part, record in (("attn", attn), ("ffn", ffn))}

        nll = []
        for row, target in zip(whole[:-1], ids[1:]):
            peak = max(row)
            nll.append(peak + math.log(sum(math.exp(v - peak) for v in row)) - row[target])

        files = {}
        for layout_name in fixture.LAYOUTS:
            path = stage / GGUF_NAMES[layout_name]
            fixture.write_deepseek41(path, args, state, engram, layout_name)
            files[layout_name] = {"file": GGUF_NAMES[layout_name], "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

        doc = {
            "_comment": "Generated by tools/deepseek4_reference.py: DeepSeek's inference/model.py at the pinned revision on the F32 kernel port, top-k in llmx's total order, the model formats' quantization on and the activations' off.",
            "fixture_kind": "synthetic",
            "repo": REPO, "revision": REVISION,
            "source": {relative: {"oid": oid, "sha256": sha256} for relative, (oid, sha256) in sorted(SOURCE.items())},
            "python_version": platform.python_version(),
            "requirements": read_requirements(),
            "requirements_sha256": text_sha256(REQUIREMENTS),
            "port_sha256": text_sha256(TOOLS / "deepseek4_kernel.py"),
            "kernel_imports": sorted(kernel_imports((source / "inference" / "model.py").read_text(encoding="utf-8"))),
            "dtype": "float32", "device": "cpu", "threads": 1,
            "seed": SEED, "sequence_seed": SEQUENCE_SEED, "args": args, "state_sha256": fixture.state_sha256(state),
            "files": files,
            "engram": dict(engram, no_convolution=no_conv,
                           mapped_differs={"position": ids.index(mapped), "raw": mapped, "mapped": engram["token_map"][mapped]},
                           pad_in_text={"position": PAD_POSITION, "raw": 2, "mapped": engram["token_map"][2]},
                           hash_ids=hashes,
                           zero_key={"layer": layer, "position": position, "dot": [0.0] * args["hc_mult"], "gate": expected,
                                     "rule": "sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot)) at a dot of +0"},
                           gates=gates),
            "sequence": ids,
            "logits": [[_short(v) for v in row] for row in whole],
            "nll": [_short(v) for v in nll], "mean_nll": sum(nll) / len(nll),
            "selections": selections,
            "margins": {kind: {"minimum": min(values), "count": len(values)} for kind, values in trace.margins.items()},
            "margin_floor": MARGIN_FLOOR,
            "sinkhorn": sinkhorn,
            "runs": runs, "equal_bound": EQUAL_BOUND,
            "activation_qdq_skipped": trace.skipped_activation_qdq, "bf16_casts_kept_f32": trace.bf16_casts,
            "swiglu_clamped": trace.clamped,
        }
        (stage / OUTPUT_NAME).write_text(json.dumps(doc, separators=(",", ":")) + "\n", encoding="utf-8")
        publish_staged(stage, out)
    print("wrote %s: margins %s, largest cached difference %.3g" % (
        out / OUTPUT_NAME, {k: v["minimum"] for k, v in doc["margins"].items()}, max(r["max_abs_diff"] for r in runs)))
    return out / OUTPUT_NAME


def _compare(kind, length, logits, whole, selections, reference):
    """A path's rows held to the whole prompt's: logits within EQUAL_BOUND and every selection the same."""
    largest = max(abs(a - b) for row, ref in zip(logits, whole) for a, b in zip(row, ref))
    if largest > EQUAL_BOUND:
        raise RuntimeError("%s %d differs from the whole prompt by %.3g" % (kind, length, largest))
    rows = len(logits)
    for part in reference:
        for layer, ref_rows in reference[part].items():
            if selections[part].get(layer, [[]] * rows) != ref_rows[:rows]:
                raise RuntimeError("%s %d selects otherwise than the whole prompt at %s %s" % (kind, length, part, layer))
    return {"kind": kind, "length": length, "rows": rows, "max_abs_diff": largest}


def self_test():
    """By-hand codec record: the port's E4M3, E8M0 and E2M1 against spec_decode's pure decoders on every code, and its Sinkhorn against float64."""
    torch = pinned_environment()
    import deepseek4_kernel as port
    import spec_decode
    raw = torch.arange(256, dtype=torch.uint8)
    e4m3 = raw.view(torch.float8_e4m3fn).float().tolist()
    e8m0 = raw.view(torch.float8_e8m0fnu).float().tolist()
    for byte in range(256):
        for got, want in ((e4m3[byte], spec_decode.e4m3(byte)), (e8m0[byte], spec_decode.e8m0(byte))):
            if not (got == want or (math.isnan(got) and math.isnan(want))):
                raise SystemExit("byte %d decodes as %r against %r" % (byte, got, want))
    codes = [spec_decode.e2m1(c) for c in range(16)]
    if port.to_e2m1(torch.tensor(codes, dtype=torch.float32)).tolist() != codes:
        raise SystemExit("the port's E2M1 rounding moves a representable value")
    x = torch.linspace(-6, 6, 4801)
    nearest = []
    for v in x.tolist():
        code = min(range(8), key=lambda c: (abs(abs(v) - codes[c]), c % 2))
        nearest.append(math.copysign(codes[code], v))
    if port.to_e2m1(x).tolist() != nearest:
        raise SystemExit("the port's E2M1 rounding is not to nearest, ties to even")
    mixes = torch.randn(64, 24, generator=torch.Generator().manual_seed(1))
    scale, base = torch.tensor([0.9, 1.1, 1.0]), torch.randn(24, generator=torch.Generator().manual_seed(2))
    pre, post, comb = port.hc_split_sinkhorn(mixes, scale, base)
    pre64, post64, comb64 = port.hc_split_sinkhorn(mixes.double(), scale.double(), base.double())
    for name, a, b in (("pre", pre, pre64), ("post", post, post64), ("comb", comb, comb64)):
        error = (a.double() - b).abs().max().item()
        print("sinkhorn %s: largest difference from float64 %.3g" % (name, error))
        if error > 1e-5:
            raise SystemExit("the port's Sinkhorn is more than 1e-5 from float64")
    print("deepseek4 codec self-test: 256 E4M3 and E8M0 bytes, 16 E2M1 codes and ties to even agree")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate the tiny deepseek41 reference from DeepSeek's own code.")
    parser.add_argument("--output-dir", required=False, help="where the golden and both tiny files go")
    parser.add_argument("--checkpoint", help="the directory holding DeepSeek's files at the pinned revision")
    parser.add_argument("--self-test", action="store_true", help="check the port's codecs against the pure decoders and its Sinkhorn against float64")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if not args.output_dir or not args.checkpoint:
        parser.error("needs --output-dir and --checkpoint")
    generate(args.output_dir, args.checkpoint)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(TOOLS))
    sys.exit(main())
