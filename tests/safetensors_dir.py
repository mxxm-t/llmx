import json
import os
import random
import struct
import tempfile

import common
import f32
import qwen35
import spec_decode as sd
from common import run_f32_cache as cli
from tokenizer import build_byte_vocab

# The Split pattern of Qwen2 and Qwen3 tokenizer.json files, the one the reader takes (bpe::hf::qwen2_pattern).
QWEN2_PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
# The Split pattern of Qwen 3.5 and 3.6 files (bpe::hf::qwen35_pattern).
QWEN35_PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""


def write_entries(path, entries):
    """One safetensors file of `entries`, (hf name, dtype, gguf shape, bytes) each, with the slowest dimension first."""
    header, offset = {}, 0
    for name, dtype, shape, data in entries:
        header[name] = {"dtype": dtype, "shape": list(reversed(shape)), "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    text = json.dumps(header).encode()
    # Spaces pad the header so the tensors start at a multiple of 8, as writers pad it.
    text += b" " * (-(8 + len(text)) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(text)) + text)
        for _, _, _, data in entries:
            f.write(data)


def write_safetensors(path, weights):
    """One safetensors file of F32 `weights`, (gguf name, hf name, gguf shape, values) each, under their HF names."""
    write_entries(path, [(name, "F32", shape, struct.pack("<%df" % len(values), *values)) for _, name, shape, values in weights])


def write_directory(path, weights, tied, files=1, c=f32.CONFIG, extra=None):
    """The tiny Qwen3 model of config `c` as a Hugging Face directory: config.json with `extra`'s keys, its weights in `files` safetensors files (an index above one), and a byte-level tokenizer whose ids are the bytes, then <|endoftext|>.
    `weights` are F32 weights, or with `extra` the entries of one file (write_entries)."""
    os.makedirs(path)
    config = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "vocab_size": f32.VOCAB,
              "hidden_size": c["embedding_length"], "intermediate_size": c["feed_forward_length"],
              "num_hidden_layers": c["block_count"], "num_attention_heads": c["attention.head_count"],
              "num_key_value_heads": c["attention.head_count_kv"], "head_dim": c["attention.key_length"],
              "max_position_embeddings": c["context_length"], "rope_theta": 10000.0, "rms_norm_eps": 1e-6,
              "tie_word_embeddings": tied, "attention_bias": False, "hidden_act": "silu", "rope_scaling": None,
              "eos_token_id": f32.VOCAB - 1, "torch_dtype": "float32"}
    config.update(extra or {})
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config, f)
    if extra:
        write_entries(os.path.join(path, "model.safetensors"), weights)
    elif files == 1:
        write_safetensors(os.path.join(path, "model.safetensors"), weights)
    else:
        weight_map = {}
        for index in range(files):
            name = "model-%05d-of-%05d.safetensors" % (index + 1, files)
            part = weights[index * len(weights) // files:(index + 1) * len(weights) // files]
            write_safetensors(os.path.join(path, name), part)
            weight_map.update((hf, name) for _, hf, _, _ in part)
        with open(os.path.join(path, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {}, "weight_map": weight_map}, f)
    write_tokenizer(path, QWEN2_PATTERN)
    return path


def write_tokenizer(path, pattern):
    """A byte-level tokenizer whose ids are the bytes, then <|endoftext|>, splitting by `pattern`."""
    vocab = {token: i for i, token in enumerate(build_byte_vocab())}
    end = {"id": f32.VOCAB - 1, "content": "<|endoftext|>", "single_word": False, "lstrip": False, "rstrip": False,
           "normalized": False, "special": True}
    tokenizer = {"added_tokens": [end], "normalizer": {"type": "NFC"},
                 "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                     {"type": "Split", "pattern": {"Regex": pattern}, "behavior": "Isolated", "invert": False},
                     {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False}]},
                 "post_processor": None, "decoder": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False},
                 "model": {"type": "BPE", "vocab": vocab, "merges": []}}
    with open(os.path.join(path, "tokenizer.json"), "w") as f:
        json.dump(tokenizer, f)
    with open(os.path.join(path, "tokenizer_config.json"), "w") as f:
        json.dump({"eos_token": "<|endoftext|>"}, f)


# A 64-wide model whose every linear has whole AWQ groups of 32 inputs and outputs in whole words of eight.
AWQ_CONFIG = {"block_count": 2, "embedding_length": 64, "feed_forward_length": 96, "attention.head_count": 2,
              "attention.head_count_kv": 1, "attention.key_length": 32, "context_length": 16}
AWQ_QUANTIZATION = {"quant_method": "awq", "bits": 4, "group_size": 32, "zero_point": True, "version": "gemm", "modules_to_not_convert": None}
LINEARS = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")


def awq_fixture(seed=7):
    """The AWQ directory's safetensors entries, and its GGUF twin's F32 weights and Q5_0 tensors (spec_decode.awq_q5_0).
    Codes and zero points are random; scales are F16 on layer 0 and BF16 on layer 1, of up to eight significant bits from F16's subnormals up, so F16 holds both exactly."""
    rng = random.Random(seed)
    entries, weights, quantized = [], [], []
    for name, hf, shape, values in f32.tensors(True, config=AWQ_CONFIG):
        layer, _, rest = name[4:].partition(".") if name.startswith("blk.") else ("", "", "")
        if rest.split(".")[0] not in LINEARS:
            entries.append((hf, "F32", shape, struct.pack("<%df" % len(values), *values)))
            weights.append((name, hf, shape, values))
            continue
        cols, rows = shape
        bf16 = layer == "1"
        scales = [[rng.choice((-1, 1)) * rng.randint(1, 255) * 2.0 ** -rng.randint(14, 22) for _ in range(rows)] for _ in range(cols // 32)]
        qweight = sd.awq_pack([[rng.randrange(16) for _ in range(rows)] for _ in range(cols)])
        qzeros = sd.awq_pack([[rng.randrange(16) for _ in range(rows)] for _ in range(cols // 32)])
        flat = [x for line in scales for x in line]
        scale_bytes = b"".join(struct.pack("<f", x)[2:] for x in flat) if bf16 else struct.pack("<%de" % len(flat), *flat)
        base = hf[:-len(".weight")]
        entries += [(base + ".qweight", "I32", [rows // 8, cols], words_of(qweight)),
                    (base + ".qzeros", "I32", [rows // 8, cols // 32], words_of(qzeros)),
                    (base + ".scales", "BF16" if bf16 else "F16", [rows, cols // 32], scale_bytes)]
        quantized.append((name, shape, sd.Q5_0, sd.awq_q5_0(qweight, qzeros, scales, 32)))
    return entries, weights, quantized


def words_of(lines):
    """Lines of int32 words as little-endian bytes."""
    flat = [w for line in lines for w in line]
    return struct.pack("<%di" % len(flat), *flat)


GPTQ_FORMS = ((4, True, True), (4, False, False), (8, True, True))  # bits, symmetric, v1


def gptq_fixture(bits, sym, v1, seed=11):
    """A GPTQ directory's safetensors entries, and its GGUF twin's F32 weights and Q4_0, Q5_0 or Q8_0 tensors (spec_decode.gptq_repack).
    Codes and asymmetric zero points are random and F16 scales as AWQ's; g_idx is each input's group of 32."""
    rng = random.Random(seed + bits + 2 * sym)
    per = 32 // bits
    entries, weights, quantized = [], [], []
    for name, hf, shape, values in f32.tensors(True, config=AWQ_CONFIG):
        layer, _, rest = name[4:].partition(".") if name.startswith("blk.") else ("", "", "")
        if rest.split(".")[0] not in LINEARS:
            entries.append((hf, "F32", shape, struct.pack("<%df" % len(values), *values)))
            weights.append((name, hf, shape, values))
            continue
        cols, rows = shape
        scales = [[rng.choice((-1, 1)) * rng.randint(1, 255) * 2.0 ** -rng.randint(14, 22) for _ in range(rows)] for _ in range(cols // 32)]
        codes = [[rng.randrange(1 << bits) for _ in range(rows)] for _ in range(cols)]
        stored = [[(1 << (bits - 1)) - (1 if v1 else 0) if sym else rng.randrange(16) for _ in range(rows)] for _ in range(cols // 32)]
        qweight, qzeros, g_idx = sd.gptq_pack_inputs(codes, bits), sd.gptq_pack_outputs(stored, bits), [k // 32 for k in range(cols)]
        flat = [x for line in scales for x in line]
        base = hf[:-len(".weight")]
        entries += [(base + ".qweight", "I32", [rows, cols // per], words_of(qweight)),
                    (base + ".qzeros", "I32", [rows // per, cols // 32], words_of(qzeros)),
                    (base + ".scales", "F16", [rows, cols // 32], struct.pack("<%de" % len(flat), *flat)),
                    (base + ".g_idx", "I32", [cols], struct.pack("<%di" % cols, *g_idx))]
        kind, data = sd.gptq_repack(qweight, qzeros, scales, g_idx, 32, bits, sym, v1)
        quantized.append((name, shape, kind, data))
    return entries, weights, quantized


CT_FORMS = ((4, True, True, 32), (4, False, True, 32), (8, True, False, 0), (8, True, True, 0))  # bits, symmetric, pack-quantized, group (0: channel)


def ct_fixture(bits, sym, packed, group, seed=13):
    """A compressed-tensors directory's safetensors entries, and its GGUF twin's F32 weights and Q4_0, Q5_0 or Q8_0 tensors (spec_decode.ct_repack).
    Codes and asymmetric zero points are random and BF16 scales F16-exact, one a group of inputs or one a row under the channel strategy."""
    rng = random.Random(seed + bits + 2 * sym + 4 * packed)
    entries, weights, quantized = [], [], []
    for name, hf, shape, values in f32.tensors(True, config=AWQ_CONFIG):
        layer, _, rest = name[4:].partition(".") if name.startswith("blk.") else ("", "", "")
        if rest.split(".")[0] not in LINEARS:
            entries.append((hf, "F32", shape, struct.pack("<%df" % len(values), *values)))
            weights.append((name, hf, shape, values))
            continue
        cols, rows = shape
        groups = cols // group if group else 1
        half = 1 << (bits - 1)
        scales = [[rng.choice((-1, 1)) * rng.randint(1, 255) * 2.0 ** -rng.randint(14, 22) for _ in range(groups)] for _ in range(rows)]
        codes = [[rng.randrange(-half, half) for _ in range(cols)] for _ in range(rows)]
        zeros = None if sym else [[rng.randrange(-8, 8) for _ in range(groups)] for _ in range(rows)]
        flat = [x for line in scales for x in line]
        base = hf[:-len(".weight")]
        scale = (base + ".weight_scale", "BF16", [groups, rows], b"".join(struct.pack("<f", x)[2:] for x in flat))
        if packed:
            entries += [(base + ".weight_packed", "I32", [cols * bits // 32, rows], words_of(sd.ct_pack(codes, bits))), scale,
                        (base + ".weight_shape", "I64", [2], struct.pack("<2q", rows, cols))]
            if zeros:
                entries.append((base + ".weight_zero_point", "I32", [groups, rows * bits // 32], words_of(sd.ct_pack_zeros(zeros, bits))))
        else:
            entries += [(base + ".weight", "I8", [cols, rows], struct.pack("<%db" % (rows * cols), *(q for line in codes for q in line))), scale]
        kind, data = sd.ct_repack(codes, zeros, scales, None, bits)
        quantized.append((name, shape, kind, data))
    return entries, weights, quantized


def ct_quantization(bits, sym, packed, group):
    """The quantization_config llm-compressor writes for one weight scheme of ct_fixture's."""
    weights = {"num_bits": bits, "type": "int", "symmetric": sym, "strategy": "group" if group else "channel", "group_size": group or None,
               "dynamic": False, "actorder": "weight" if group else None, "block_structure": None, "observer": "minmax", "observer_kwargs": {}}
    activations = None if packed else {"num_bits": 8, "type": "int", "symmetric": True, "strategy": "token", "dynamic": True, "group_size": None,
                                       "actorder": None, "block_structure": None, "observer": None, "observer_kwargs": {}}
    return {"quant_method": "compressed-tensors", "format": "pack-quantized" if packed else "int-quantized", "quantization_status": "compressed",
            "config_groups": {"group_0": {"targets": ["Linear"], "weights": weights, "input_activations": activations, "output_activations": None}},
            "ignore": ["lm_head"], "kv_cache_scheme": None, "global_compression_ratio": None, "sparsity_config": {}}


# A 128-wide model for FP8 linears, whose scales cover 128 inputs: q and the feed-forward rows fill whole 128-row blocks, and k and v fill half of one.
FP8_CONFIG = {"block_count": 2, "embedding_length": 128, "feed_forward_length": 256, "attention.head_count": 2,
              "attention.head_count_kv": 1, "attention.key_length": 64, "context_length": 16}
# Dialect (fp8: quant_method fp8 with weight_scale_inv; ct: compressed-tensors float-quantized with weight_scale), scale form and scale dtype.
FP8_FORMS = (("fp8", "tensor", "F32"), ("fp8", "block", "BF16"), ("ct", "tensor", "F32"), ("ct", "channel", "F16"), ("ct", "block", "BF16"))


def fp8_fixture(dialect, form, dtype, seed=17):
    """An FP8 directory's safetensors entries, its F32 twin's weights (code times scale, exact in F32), and, for one scale a tensor, the GGUF twin's F8_E4M3 tensors with their F32 scalar scales.
    Codes are every non-NaN byte at random, scales of up to eight significant bits in [2^-13, 2^-6], exact in F16 and BF16; static input scales and, in ct's tensor form, KV cache scales ride along unapplied."""
    rng = random.Random(seed + 3 * FP8_FORMS.index((dialect, form, dtype)))
    pack = {"F32": lambda xs: struct.pack("<%df" % len(xs), *xs), "F16": lambda xs: struct.pack("<%de" % len(xs), *xs),
            "BF16": lambda xs: b"".join(struct.pack("<f", x)[2:] for x in xs)}[dtype]
    scale_name = "weight_scale_inv" if dialect == "fp8" else "weight_scale"
    static = form == "tensor"
    entries, weights, quantized = [], [], []
    for name, hf, shape, values in f32.tensors(True, config=FP8_CONFIG):
        layer, _, rest = name[4:].partition(".") if name.startswith("blk.") else ("", "", "")
        if rest.split(".")[0] not in LINEARS:
            entries.append((hf, "F32", shape, struct.pack("<%df" % len(values), *values)))
            weights.append((name, hf, shape, values))
            if dialect == "ct" and static and rest == "attn_k_norm.weight":
                for kv in ("k_scale", "v_scale"):
                    entries.append(("model.layers.%s.self_attn.%s" % (layer, kv), "F32", [1], struct.pack("<f", 0.25)))
            continue
        cols, rows = shape
        codes = [[rng.choice([c for c in range(256) if c & 0x7F != 0x7F]) for _ in range(cols)] for _ in range(rows)]
        scale = lambda: rng.choice((-1, 1)) * rng.randint(1, 255) * 2.0 ** -rng.randint(13, 20)
        if form == "tensor":
            scales, flat, scale_shape = scale(), None, [] if dialect == "fp8" else [1]
            flat = [scales]
        elif form == "channel":
            scales = [scale() for _ in range(rows)]
            flat, scale_shape = scales, [1, rows]
        else:
            scales = [[scale() for _ in range(cols // 128)] for _ in range((rows + 127) // 128)]
            flat, scale_shape = [x for line in scales for x in line], [cols // 128, (rows + 127) // 128]
        base = hf[:-len(".weight")]
        entries += [(hf, "F8_E4M3", shape, bytes(c for line in codes for c in line)), (base + "." + scale_name, dtype, scale_shape, pack(flat))]
        if static:
            entries.append((base + ".input_scale", "F32", [1] if dialect == "ct" else [], struct.pack("<f", 0.5)))
        weights.append((name, hf, shape, [x for line in sd.fp8_dequantize(codes, scales) for x in line]))
        if form == "tensor":
            quantized += [(name, shape, sd.F8_E4M3, bytes(c for line in codes for c in line)), (name[:-len(".weight")] + ".scale", [1], sd.F32, struct.pack("<f", scales))]
    return entries, weights, quantized


def fp8_quantization(dialect, form):
    """The quantization_config of fp8_fixture's dialect and form."""
    if dialect == "fp8":
        return {"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "static" if form == "tensor" else "dynamic",
                "weight_block_size": None if form == "tensor" else [128, 128], "modules_to_not_convert": ["lm_head"]}
    weights = {"num_bits": 8, "type": "float", "symmetric": True, "strategy": form, "group_size": None, "dynamic": False, "actorder": None,
               "block_structure": [128, 128] if form == "block" else None, "observer": "minmax", "observer_kwargs": {}}
    activations = {"num_bits": 8, "type": "float", "symmetric": True, "strategy": "tensor" if form == "tensor" else "token", "dynamic": form != "tensor",
                   "group_size": None, "actorder": None, "block_structure": None, "observer": None, "observer_kwargs": {}}
    kv = {"num_bits": 8, "type": "float", "symmetric": True, "strategy": "tensor", "dynamic": False, "group_size": None, "actorder": None,
          "block_structure": None, "observer": "minmax", "observer_kwargs": {}} if form == "tensor" else None
    return {"quant_method": "compressed-tensors", "format": "float-quantized", "quantization_status": "compressed",
            "config_groups": {"group_0": {"targets": ["Linear"], "weights": weights, "input_activations": activations, "output_activations": None}},
            "ignore": ["lm_head", "re:.*mlp.gate$"], "kv_cache_scheme": kv, "global_compression_ratio": None, "sparsity_config": {}}


def logits_of(out):
    """`llmx logits --top N` output as {id: logit}."""
    return {int(line.split()[0]): float(line.split()[1]) for line in out.strip().splitlines()[1:]}


def fp8_checks(directory, dialect, form, dtype):
    """An FP8 directory: info lists each linear as F8_E4M3 beside its F32 grid and the schemes as not applied, and one scale a tensor reads as its GGUF twin byte for byte.
    Its logits stay within the F32 bound of its F32 twin at f32 and within the frozen F16 budget at f16, or None when the configured device has no F8_E4M3 kernel."""
    entries, weights, quantized = fp8_fixture(dialect, form, dtype)
    label = "fp8-%s-%s" % (dialect, form)
    model = write_directory(os.path.join(directory, label), entries, True, c=FP8_CONFIG, extra={"torch_dtype": "bfloat16", "quantization_config": fp8_quantization(dialect, form)})
    twin = f32.write_model(os.path.join(directory, label + "-f32.gguf"), weights, config=FP8_CONFIG)
    rc, out = cli(["info", model])
    listed = [line for line in out.splitlines() if any(".%s.weight " % name in line for name in LINEARS)]
    grids = [line for line in out.splitlines() if any(".%s.scale " % name in line for name in LINEARS)]
    assert rc == 0 and len(listed) == 14 and all("F8_E4M3" in line for line in listed) and len(grids) == 14 and all("F32" in line for line in grids), \
        "info on the %s directory: %s" % (label, out)
    shown = [line.strip().split(" = ")[0] for line in out.splitlines() if "not applied" in line]
    want = ["quantization.input_activations"] + (["quantization.kv_cache_scheme"] if dialect == "ct" and form == "tensor" else [])
    assert shown == want, "info on the %s directory lists the schemes %r" % (label, shown)
    text = f32.TEXTS[-1]
    reference = common.run_process(["logits", twin, text, "--top", str(f32.VOCAB), "--dtype", "f32"], cache="f32")
    assert reference.returncode == 0, reference.stderr
    worst = {}
    for dt, bound in (("f32", common.F32_HF_LOGIT_BOUND), ("f16", common.hf_bounds("f16", [["f16"]])["logit"])):
        got = common.run_process(["logits", model, text, "--top", str(f32.VOCAB), "--dtype", dt], cache="f32")
        if common.device_lacks_kernel(got.returncode, got.stderr.decode(errors="replace")):
            return None
        assert got.returncode == 0, "logits on the %s directory: %s" % (label, got.stderr)
        a, b = logits_of(got.stdout.decode()), logits_of(reference.stdout.decode())
        assert set(a) == set(b) == set(range(f32.VOCAB)), "missing %s logits" % label
        worst[dt] = max(abs(a[i] - b[i]) for i in a)
        assert worst[dt] < bound, "%s at %s against its F32 twin: %.8f, bound %.8f" % (label, dt, worst[dt], bound)
    if quantized:
        gguf = f32.write_model(os.path.join(directory, label + "-twin.gguf"), [w for w in weights if w[0].split(".")[2 if w[0].startswith("blk.") else 0] not in LINEARS],
                               config=FP8_CONFIG, quantized=quantized)
        for args in (["tokenize", "{}", text], ["logits", "{}", text, "--top", str(f32.VOCAB)], ["generate", "{}", text[:4], "-n", "8", "--temp", "0", "--ignore-eos"]):
            outs = [common.run_process([a.replace("{}", m) for a in args], cache="f32") for m in (model, gguf)]
            text_of = common.generate_text if args[0] == "generate" else common.cli_stdout
            assert outs[0].returncode == 0 == outs[1].returncode and text_of(outs[0].stdout) == text_of(outs[1].stdout), \
                "%s on the %s directory differs from its F8_E4M3 GGUF twin: %r" % (args[0], label, [(p.stdout[-400:], p.stderr[-400:]) for p in outs])
    return worst


def fp8_refusals(directory):
    """What would change an FP8 file's values is refused by name: E5M2 and FNUZ codes, another fmt, blocks other than 128 x 128, another scale dtype and float weights other than 8-bit."""
    entries, _, _ = fp8_fixture("fp8", "block", "BF16")
    cases = [("e5m2", [(n, "F8_E5M2" if d == "F8_E4M3" else d, sh, b) for n, d, sh, b in entries], {}, "F8_E5M2; llmx reads the 8-bit float F8_E4M3 only"),
             ("fnuz", [(n, "F8_E4M3FNUZ" if d == "F8_E4M3" else d, sh, b) for n, d, sh, b in entries], {}, "F8_E4M3FNUZ; llmx reads the 8-bit float F8_E4M3 only"),
             ("fmt", entries, {"fmt": "e5m2"}, "llmx reads E4M3 FP8 only"),
             ("blocks", entries, {"weight_block_size": [64, 64]}, "llmx reads FP8 blocks of 128 x 128 only"),
             ("i32", [(n, "I32", sh, b * 2) if n.endswith("_scale_inv") else (n, d, sh, b) for n, d, sh, b in entries], {}, "llmx reads BF16, F16 and F32 FP8 scales")]
    for label, files, change, message in cases:
        quantization = dict(fp8_quantization("fp8", "block"), **change)
        model = write_directory(os.path.join(directory, "fp8-refused-" + label), files, True, c=FP8_CONFIG, extra={"quantization_config": quantization})
        rc, out = cli(["info", model])
        assert rc != 0 and message in out, "the %s FP8 directory was not refused by name: %s" % (label, out)
    ct = fp8_quantization("ct", "block")
    ct["config_groups"]["group_0"]["weights"]["num_bits"] = 4
    model = write_directory(os.path.join(directory, "fp8-refused-bits"), fp8_fixture("ct", "block", "BF16")[0], True, c=FP8_CONFIG, extra={"quantization_config": ct})
    rc, out = cli(["info", model])
    assert rc != 0 and "llmx reads 8-bit symmetric float weights only" in out, "4-bit float weights were not refused: " + out


def twin_checks(directory, label, fixture, quantization, type_name):
    """The quantized directory reads as its twin: info lists each linear as `type_name`, and ids, text and logits match the twin's byte for byte."""
    entries, weights, quantized = fixture
    extra = {"torch_dtype": "float16", "quantization_config": quantization}
    model = write_directory(os.path.join(directory, label), entries, True, c=AWQ_CONFIG, extra=extra)
    twin = f32.write_model(os.path.join(directory, label + "-twin.gguf"), weights, config=AWQ_CONFIG, quantized=quantized)
    rc, out = cli(["info", model])
    listed = [line for line in out.splitlines() if any(".%s.weight " % name in line for name in LINEARS)]
    assert rc == 0 and len(listed) == 14 and all(type_name in line for line in listed), "info on the %s directory: %s" % (label, out)
    # An activation scheme is listed and not applied: activations follow --dtype.
    scheme = (quantization.get("config_groups") or {}).get("group_0", {}).get("input_activations")
    shown = [line.strip() for line in out.splitlines() if "quantization.input_activations" in line]
    want = ['quantization.input_activations = "8-bit int by token, dynamic, symmetric; not applied, activations follow --dtype"'] if scheme else []
    assert shown == want, "info on the %s directory lists the activation scheme as %r" % (label, shown)
    text = f32.TEXTS[-1]
    for args in (["tokenize", "{}", text], ["logits", "{}", text, "--top", str(f32.VOCAB)],
                 ["generate", "{}", text[:4], "-n", "8", "--temp", "0", "--ignore-eos"]):
        outs = [common.run_process([a.replace("{}", m) for a in args], cache="f32") for m in (model, twin)]
        text_of = common.generate_text if args[0] == "generate" else common.cli_stdout
        same = outs[0].returncode == 0 == outs[1].returncode and text_of(outs[0].stdout) == text_of(outs[1].stdout)
        assert same, "%s on the %s directory differs from its twin: %r" % (args[0], label, [(p.stdout[-400:], p.stderr[-400:]) for p in outs])


def bf16_raw(raw):
    """The HF weights `raw` rounded to BF16, ties to even, as (name, shape, values, BF16 bytes)."""
    out = []
    for name, shape, values in raw:
        bits = [struct.unpack("<I", struct.pack("<f", v))[0] for v in values]
        halves = [(b + 0x7fff + ((b >> 16) & 1)) >> 16 for b in bits]
        out.append((name, shape, [struct.unpack("<f", struct.pack("<I", h << 16))[0] for h in halves], struct.pack("<%dH" % len(halves), *halves)))
    return out


def qwen35_directory(path, spec, text_change=None, rope_change=None, raw=None):
    """The tiny qwen35 fixture `spec` of tests/qwen35.py as a Qwen 3.5 directory: its HF weights, or `raw` from bf16_raw, under the multimodal model's names beside one vision tower tensor.
    The text model's keys sit in text_config, changed by `text_change` and `rope_change`."""
    os.makedirs(path)
    config, v_head, _ = qwen35.fixture_config(spec)
    head = config["attention.key_length"]
    text = {"model_type": "qwen3_5_text", "dtype": "float32", "vocab_size": f32.VOCAB, "hidden_size": config["embedding_length"],
            "intermediate_size": config["feed_forward_length"], "num_hidden_layers": qwen35.LAYERS, "num_attention_heads": config["attention.head_count"],
            "num_key_value_heads": config["attention.head_count_kv"], "head_dim": head, "hidden_act": "silu",
            "max_position_embeddings": config["context_length"], "rms_norm_eps": config["attention.layer_norm_rms_epsilon"],
            "tie_word_embeddings": spec["tied"], "attention_bias": False, "attention_dropout": 0.0, "attn_output_gate": True,
            "linear_conv_kernel_dim": config["ssm.conv_kernel"], "linear_key_head_dim": config["ssm.state_size"], "linear_value_head_dim": v_head,
            "linear_num_key_heads": config["ssm.group_count"], "linear_num_value_heads": spec["v_heads"], "full_attention_interval": config["full_attention_interval"],
            "layer_types": ["full_attention" if qwen35.full_attention(layer) else "linear_attention" for layer in range(qwen35.LAYERS)],
            "mtp_num_hidden_layers": int(spec["mtp"]), "mtp_use_dedicated_embeddings": False, "mamba_ssm_dtype": "float32", "eos_token_id": qwen35.EOS,
            "rope_parameters": dict({"rope_type": "default", "rope_theta": config["rope.freq_base"], "partial_rotary_factor": config["rope.dimension_count"] / head,
                                     "mrope_section": config["rope.dimension_sections"][:3], "mrope_interleaved": True}, **(rope_change or {}))}
    text.update(text_change or {})
    top = {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5", "text_config": text, "tie_word_embeddings": text["tie_word_embeddings"],
           "vision_config": {"depth": 1}, "image_token_id": f32.VOCAB, "transformers_version": "5.17.0"}
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(top, f)
    entries = [("model.visual.patch_embed.proj.weight", "F32", [2, 2, 2, 3, 4], b"\0" * 4 * 96)]
    for name, shape, values, *half in raw or qwen35.raw_weights(spec):
        hf = "model.language_model." + name[len("model."):] if name.startswith("model.") else name
        entries.append((hf, "BF16" if half else "F32", list(reversed(shape)), half[0] if half else struct.pack("<%df" % len(values), *values)))
    write_entries(os.path.join(path, "model.safetensors"), entries)
    write_tokenizer(path, QWEN35_PATTERN)
    return path


def qwen35_checks(directory):
    """Tiny qwen35 directories, Hv = Hk tied and Hv = 3 Hk alone, in BF16 and with an MTP layer, give the outputs of the GGUF the converter writes (tests/qwen35.py) byte for byte, the MTP layer drafting as the file's does.
    The Hv = 3 Hk directory against HF's goldens of its weights, and what the table does not read refused by name. Returns the largest logit error against HF."""
    goldens = {fixture["name"]: fixture for fixture in qwen35.golden()["fixtures"]}
    text = f32.TEXTS[-1]
    worst = 0.0
    hv3 = next(spec for spec in qwen35.FIXTURES if spec["name"] == "hv3")
    for spec, half in [(spec, False) for spec in qwen35.FIXTURES if not spec.get("moe")] + [(hv3, True)]:
        label = spec["name"] + ("-bf16" if half else "")
        if half:
            # BF16 weights reach the 16-bit forms of every transform; the twin holds their exact values.
            raw = bf16_raw(qwen35.raw_weights(spec))
            model = qwen35_directory(os.path.join(directory, "qwen35-" + label), spec, {"dtype": "bfloat16"}, raw=raw)
            twin = f32.write_model(os.path.join(directory, "qwen35-%s.gguf" % label), qwen35.gguf_tensors(spec, [r[:3] for r in raw]), eos_id=qwen35.EOS,
                                   config=qwen35.gguf_config(spec), arch="qwen35")
        else:
            model = qwen35_directory(os.path.join(directory, "qwen35-" + label), spec)
            twin = qwen35.write_fixture(directory, spec)
        runs = [["tokenize", "{}", text], ["logits", "{}", text, "--top", str(f32.VOCAB)], ["generate", "{}", text[:4], "-n", "8", "--temp", "0", "--ignore-eos"]]
        if spec["mtp"]:
            runs.append(["generate", "{}", text[:4], "-n", "8", "--temp", "0", "--ignore-eos", "--drafter", "embedded"])
        for args in runs:
            outs = [common.run_process([a.replace("{}", m) for a in args], cache="f32") for m in (model, twin)]
            text_of = common.generate_text if args[0] == "generate" else common.cli_stdout
            assert outs[0].returncode == 0 == outs[1].returncode and text_of(outs[0].stdout) == text_of(outs[1].stdout), \
                "%s on the qwen35 %s directory differs from its GGUF twin: %r" % (args[0], label, [(p.stdout[-400:], p.stderr[-400:]) for p in outs])
        if label == "hv3":
            error, _ = common.check_hf_fixture("safetensors qwen35 hv3", model, goldens["hv3"]["cases"], goldens["hv3"]["perplexity"], text, (1, 16))
            worst = max(worst, error)
    mtp = next(spec for spec in qwen35.FIXTURES if spec["mtp"])
    cases = [("no-mtp", {"mtp_num_hidden_layers": 0}, None, "is an MTP tensor, and config.json gives no MTP layer"),
             ("layers", {"full_attention_interval": 3}, None, "config.json layer_types does not follow full_attention_interval"),
             ("rope", None, {"rope_type": "yarn"}, 'config.json rope_parameters rope_type = "yarn" is not supported'),
             ("key", {"sliding_window": 4}, None, "config.json key sliding_window is not one llmx reads for qwen3_5")]
    for label, text_change, rope_change, message in cases:
        model = qwen35_directory(os.path.join(directory, "qwen35-refused-" + label), mtp, text_change, rope_change)
        rc, out = cli(["info", model])
        assert rc != 0 and message in out, "the %s qwen35 directory was not refused by name: %s" % (label, out)
    # tie_word_embeddings at the top level alone ties the head as text_config's would, and the two levels disagreeing is refused.
    tied = next(spec for spec in qwen35.FIXTURES if spec["tied"] and not spec.get("moe"))
    outs = []
    for label, inner in [("tie-both", True), ("tie-outer", None), ("tie-differs", False)]:
        model = qwen35_directory(os.path.join(directory, "qwen35-" + label), tied)
        with open(os.path.join(model, "config.json")) as f:
            top = json.load(f)
        if inner is None:
            del top["text_config"]["tie_word_embeddings"]
        else:
            top["text_config"]["tie_word_embeddings"] = inner
        with open(os.path.join(model, "config.json"), "w") as f:
            json.dump(top, f)
        outs.append(cli(["logits", model, text, "--top", "4"]))
    assert outs[0][0] == 0 == outs[1][0] and outs[0][1] == outs[1][1], "a tie at the top level alone did not tie the head: %r" % (outs[:2],)
    assert outs[2][0] != 0 and "tie_word_embeddings differs" in outs[2][1], "disagreeing ties were not refused: " + outs[2][1]
    return worst


def run():
    if common.f32_cache_skip("safetensors") or common.tensor_width_skip("safetensors"):
        return common.SKIPPED
    worst = 0.0
    with tempfile.TemporaryDirectory(prefix="llmx_safetensors_") as directory:
        for fixture in f32.golden("baseline_f32.json")["fixtures"]:
            tied = fixture.get("tied", False)
            # The untied fixture spreads its tensors over two files under an index.
            model = write_directory(os.path.join(directory, "tied" if tied else "untied"), fixture["weights"], tied, 1 if tied else 2)
            twin = f32.write_model(os.path.join(directory, "twin.gguf"), fixture["weights"])
            error, _ = common.check_hf_fixture("safetensors F32", model, fixture["cases"], fixture["perplexity"], f32.TEXTS[-1], (1, 5, 16))
            worst = max(worst, error, f32.check_logits_input(directory, model, fixture["cases"]))
            # The directory reads as its GGUF twin: the same ids, text and logits, byte for byte.
            text = f32.TEXTS[-1]
            for args in (["tokenize", "{}", text], ["logits", "{}", text, "--top", str(f32.VOCAB)],
                         ["generate", "{}", text[:4], "-n", "8", "--temp", "0", "--ignore-eos"]):
                # Bytes, since generated tokens may end inside a character.
                outs = [common.run_process([a.replace("{}", m) for a in args], cache="f32") for m in (model, twin)]
                text_of = common.generate_text if args[0] == "generate" else common.cli_stdout
                assert outs[0].returncode == 0 == outs[1].returncode and text_of(outs[0].stdout) == text_of(outs[1].stdout), \
                    "%s on the directory differs from its GGUF twin: %r" % (args[0], [(p.stdout, p.stderr) for p in outs])
            rc, out = cli(["detokenize", model, ",".join(str(b) for b in text.encode())])
            assert rc == 0 and out.strip() == text, "detokenize on the directory: " + out
            rc, out = cli(["info", model])
            assert rc == 0 and "qwen3.rope.freq_base" in out and (" output.weight " in out) != tied, "info on the directory: " + out
        twin_checks(directory, "awq", awq_fixture(), AWQ_QUANTIZATION, "Q5_0")
        for bits, sym, v1 in GPTQ_FORMS:
            quantization = {"quant_method": "gptq", "bits": bits, "group_size": 32, "desc_act": False, "sym": sym,
                            "checkpoint_format": "gptq" if v1 else "gptq_v2", "lm_head": False, "pack_dtype": "int32"}
            twin_checks(directory, "gptq%d%s" % (bits, "" if sym else "a"), gptq_fixture(bits, sym, v1), quantization,
                        "Q8_0" if bits == 8 else "Q4_0" if sym else "Q5_0")
        for bits, sym, packed, group in CT_FORMS:
            twin_checks(directory, "ct%d%s%s" % (bits, "" if sym else "a", "p" if packed else "i"), ct_fixture(bits, sym, packed, group),
                        ct_quantization(bits, sym, packed, group), "Q8_0" if bits == 8 else "Q4_0" if sym else "Q5_0")
        fp8_worst = {"f32": 0.0, "f16": 0.0}
        for dialect, form, dtype in FP8_FORMS:
            errors = fp8_checks(directory, dialect, form, dtype)
            if errors is None:
                if fp8_worst:
                    print("safetensors: FP8 directories' logits SKIP - %s has no F8_E4M3 kernel" % os.environ["LLMX_DEVICE"])
                fp8_worst = None
                continue
            for dt, error in errors.items():
                fp8_worst[dt] = max(fp8_worst[dt], error)
        fp8_refusals(directory)
        worst = max(worst, qwen35_checks(directory))
    fp8 = ("against their F32 twins, worst %.8f at f32 and %.8f at f16, by tensor as their F8_E4M3 GGUF twins byte for byte"
           % (fp8_worst["f32"], fp8_worst["f16"]) if fp8_worst else "listed by info, their logits skipped")
    print("safetensors: tiny F32 directories, tied in one file and untied in two under an index, all 257 logits vs HF, PPL, --last/--then-ids/--per-token rows, and the GGUF twin's ids, text and logits; an AWQ directory with F16 and BF16 scales and GPTQ 4-bit symmetric, 4-bit v2 and 8-bit directories and compressed-tensors 4-bit symmetric and asymmetric packed and 8-bit packed and int8 directories as their Q5_0, Q4_0 and Q8_0 twins' ids, text and logits; max error %.8f; FP8 directories in both dialects by tensor, channel and block %s, and the refusals; tiny qwen35 directories, F32 and BF16, with their GGUF twins' outputs byte for byte, Hv = 3 Hk against HF, and their refusals  [ok]" % (worst, fp8))
    return True


if __name__ == "__main__":
    run()
