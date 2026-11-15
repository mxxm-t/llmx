import json
import os
import random
import struct
import tempfile

import common
import f32
import spec_decode as sd
from common import run_f32_cache as cli
from tokenizer import build_byte_vocab

# The Split pattern of Qwen2 and Qwen3 tokenizer.json files, the one the reader takes (bpe::hf::qwen2_pattern).
QWEN2_PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""


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
    vocab = {token: i for i, token in enumerate(build_byte_vocab())}
    end = {"id": f32.VOCAB - 1, "content": "<|endoftext|>", "single_word": False, "lstrip": False, "rstrip": False,
           "normalized": False, "special": True}
    tokenizer = {"added_tokens": [end], "normalizer": {"type": "NFC"},
                 "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                     {"type": "Split", "pattern": {"Regex": QWEN2_PATTERN}, "behavior": "Isolated", "invert": False},
                     {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False}]},
                 "post_processor": None, "decoder": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False},
                 "model": {"type": "BPE", "vocab": vocab, "merges": []}}
    with open(os.path.join(path, "tokenizer.json"), "w") as f:
        json.dump(tokenizer, f)
    with open(os.path.join(path, "tokenizer_config.json"), "w") as f:
        json.dump({"eos_token": "<|endoftext|>"}, f)
    return path


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


def awq_checks(directory):
    """The AWQ directory reads as its Q5_0 twin: info lists each linear as Q5_0, and ids, text and logits match the twin's byte for byte."""
    entries, weights, quantized = awq_fixture()
    extra = {"torch_dtype": "float16", "quantization_config": AWQ_QUANTIZATION}
    model = write_directory(os.path.join(directory, "awq"), entries, True, c=AWQ_CONFIG, extra=extra)
    twin = f32.write_model(os.path.join(directory, "awq-twin.gguf"), weights, config=AWQ_CONFIG, quantized=quantized)
    rc, out = cli(["info", model])
    listed = [line for line in out.splitlines() if any(".%s.weight " % name in line for name in LINEARS)]
    assert rc == 0 and len(listed) == 14 and all("Q5_0" in line for line in listed), "info on the AWQ directory: " + out
    text = f32.TEXTS[-1]
    for args in (["tokenize", "{}", text], ["logits", "{}", text, "--top", str(f32.VOCAB)],
                 ["generate", "{}", text[:4], "-n", "8", "--temp", "0", "--ignore-eos"]):
        outs = [common.run_process([a.replace("{}", m) for a in args], cache="f32") for m in (model, twin)]
        text_of = common.generate_text if args[0] == "generate" else common.cli_stdout
        same = outs[0].returncode == 0 == outs[1].returncode and text_of(outs[0].stdout) == text_of(outs[1].stdout)
        assert same, "%s on the AWQ directory differs from its Q5_0 twin: %r" % (args[0], [(p.stdout[-400:], p.stderr[-400:]) for p in outs])


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
        awq_checks(directory)
    print("safetensors: tiny F32 directories, tied in one file and untied in two under an index, all 257 logits vs HF, PPL, --last/--then-ids/--per-token rows, and the GGUF twin's ids, text and logits; an AWQ directory with F16 and BF16 scales as its Q5_0 twin's ids, text and logits; max error %.8f  [ok]" % worst)
    return True


if __name__ == "__main__":
    run()
