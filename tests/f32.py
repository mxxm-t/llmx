import hashlib
import json
import math
import os
import struct
import tempfile

from common import run_f32_cache as cli
from tokenizer import build_byte_vocab, w_str


CONFIG = {"block_count": 2, "embedding_length": 37, "feed_forward_length": 19,
          "attention.head_count": 2, "attention.head_count_kv": 1,
          "attention.key_length": 42, "context_length": 16}
TEXTS = ["a", "ab", "abc", "abcdefg", "abcdefghijklm"]


def tensors(tied):
    state = 12345
    result = []
    hd = CONFIG["attention.key_length"]

    def add(name, hf_name, shape, norm=False):
        nonlocal state
        values = []
        for _ in range(math.prod(shape)):
            state = (1664525 * state + 1013904223) & 0xffffffff
            value = (((state >> 16) & 1023) - 512) / 8192
            values.append(1.0 + value if norm else value)
        result.append((name, hf_name, shape, values))

    add("token_embd.weight", "model.embed_tokens.weight", [37, 257])
    add("output_norm.weight", "model.norm.weight", [37], True)
    for layer in range(2):
        name, hf = "blk.%d." % layer, "model.layers.%d." % layer
        for norm, mapped, width in (("attn_norm", "input_layernorm", 37),
                                    ("ffn_norm", "post_attention_layernorm", 37),
                                    ("attn_q_norm", "self_attn.q_norm", hd),
                                    ("attn_k_norm", "self_attn.k_norm", hd)):
            add(name + norm + ".weight", hf + mapped + ".weight", [width], True)
        for tensor, mapped, shape in (("attn_q", "self_attn.q_proj", [37, 2 * hd]),
                                      ("attn_k", "self_attn.k_proj", [37, hd]),
                                      ("attn_v", "self_attn.v_proj", [37, hd]),
                                      ("attn_output", "self_attn.o_proj", [2 * hd, 37]),
                                      ("ffn_gate", "mlp.gate_proj", [37, 19]),
                                      ("ffn_up", "mlp.up_proj", [37, 19]),
                                      ("ffn_down", "mlp.down_proj", [19, 37])):
            add(name + tensor + ".weight", hf + mapped + ".weight", shape)
    if not tied:
        add("output.weight", "lm_head.weight", [37, 257])
    return result


def weight_hash(weights):
    return hashlib.sha256(b"".join(struct.pack("<%df" % len(v), *v)
                                   for _, _, _, v in weights)).hexdigest()


def write_model(path, weights, chat_template=None, eos_id=None, shards=1, config=CONFIG, arch="qwen3"):
    # A 34-byte Q8 tensor exposes unaligned F32 rows if the loader discards file padding without preserving float alignment in its in-memory blob.
    entries = [("unused.weight", [32], 8, b"\0" * 34)]
    entries += [(name, shape, 0, struct.pack("<%df" % len(v), *v))
                for name, _, shape, v in weights]
    def write_part(filename, subset, index):
        with open(filename, "wb") as f:
            named = arch != "qwen3"
            nmeta = (len(config) + 1 + named + (chat_template is not None) + (eos_id is not None)) if index == 0 else 0
            f.write(struct.pack("<IIQQ", 0x46554747, 3, len(subset), nmeta + (3 if shards > 1 else 0)))
            if shards > 1:
                for key, value in (("split.no", index), ("split.count", shards)):
                    w_str(f, key)
                    f.write(struct.pack("<IH", 2, value))
                w_str(f, "split.tensors.count")
                f.write(struct.pack("<Ii", 5, len(entries)))
            if index == 0 and named:
                w_str(f, "general.architecture")
                f.write(struct.pack("<I", 8))
                w_str(f, arch)
            for name, value in (config.items() if index == 0 else []):
                w_str(f, arch + "." + name)
                f.write(struct.pack("<II", 4, value))
            if index == 0 and chat_template is not None:
                w_str(f, "tokenizer.chat_template")
                f.write(struct.pack("<I", 8))
                w_str(f, chat_template)
            if index == 0 and eos_id is not None:
                w_str(f, "tokenizer.ggml.eos_token_id")
                f.write(struct.pack("<II", 4, eos_id))
            if index == 0:
                w_str(f, "tokenizer.ggml.tokens")
                f.write(struct.pack("<IIQ", 9, 8, 257))
                for token in build_byte_vocab() + ["<|endoftext|>"]:
                    w_str(f, token)
            offset = 0
            for name, shape, kind, data in subset:
                w_str(f, name)
                f.write(struct.pack("<I", len(shape)))
                f.write(struct.pack("<%dQ" % len(shape), *shape))
                f.write(struct.pack("<IQ", kind, offset))
                offset += (len(data) + 31) // 32 * 32
            f.write(b"\0" * (-f.tell() % 32))
            for _, _, _, data in subset:
                f.write(data)
                f.write(b"\0" * (-len(data) % 32))
    if shards == 1:
        write_part(path, entries, 0)
        return path
    assert shards >= 2
    first = None
    for index in range(shards):
        filename = os.path.splitext(path)[0] + "-%05d-of-%05d.gguf" % (index + 1, shards)
        subset = [] if index == 0 else entries[(index - 1) * len(entries) // (shards - 1):index * len(entries) // (shards - 1)]
        write_part(filename, subset, index)
        first = first or filename
    return first


def run():
    # This gate compares exact f32 arithmetic against the HF fixture; an f16 cache rounds keys and values and is checked by the real-model gate.
    if os.environ.get("LLMX_CACHE_TYPE", "f32") != "f32":
        print("f32: SKIP - the exact F32 gate needs f32 caches (LLMX_CACHE_TYPE=%s)" % os.environ["LLMX_CACHE_TYPE"])
        return True
    with open(os.path.join(os.path.dirname(__file__), "data", "baseline_f32.json"), encoding="utf-8") as f:
        golden = json.load(f)
    assert golden["config"] == CONFIG, "F32 fixture config changed"
    worst = 0.0
    with tempfile.TemporaryDirectory(prefix="llmx_f32_") as directory:
        for fixture in golden["fixtures"]:
            weights = tensors(fixture["tied"])
            assert weight_hash(weights) == fixture["weights_sha256"], "F32 fixture weights changed"
            model = os.path.join(directory, "tiny-f32.gguf")
            write_model(model, weights)
            for threads in (1, 4):
                for ubatch in (1, 2, 3, 5, 16):
                    for case in fixture["cases"]:
                        rc, out = cli(["logits", model, case["text"], "--top", "257",
                                       "--threads", str(threads), "--ubatch", str(ubatch)])
                        assert rc == 0, "F32 logits failed: " + out
                        got = {int(p[0]): float(p[1]) for line in out.splitlines()
                               if len(p := line.split()) == 2 and p[0].isdigit()}
                        assert set(got) == set(range(257)), "missing F32 logits"
                        assert all(math.isfinite(v) for v in got.values()), "non-finite F32 logits"
                        error = max(abs(got[i] - expected) for i, expected in enumerate(case["logits"]))
                        assert math.isfinite(error) and error < 2e-5, "F32/HF logit error: %.8f" % error
                        worst = max(worst, error)
                for case in fixture["perplexity"]:
                    rc, out = cli(["perplexity", model, TEXTS[-1], "--threads", str(threads),
                                   "-c", str(case["context"])])
                    assert rc == 0, "F32 PPL failed: " + out
                    fields = dict(line.split(":", 1) for line in out.splitlines())
                    error = abs(float(fields["mean NLL"]) - case["mean_nll"])
                    assert math.isfinite(error) and error < 1e-5, "F32/HF NLL error: %.8f" % error
        # The bench measures on top of a history when asked for a depth, and refuses one with batched decode.
        rc, out = cli(["bench", "--model", model, "--p", "4", "--n", "2", "--r", "1", "--depth", "6"])
        assert rc == 0 and "pp4 @ d6" in out and "tg2 @ d6" in out, "bench --depth failed: " + out
        rc, out = cli(["bench", "--model", model, "--depth", "6", "--seqs", "2"])
        assert rc != 0, "bench --depth accepted batched decode"
    print("f32: all 257 logits vs HF, tied/untied, batch/row/column tails, threads and PPL; max error %.8f  [ok]" % worst)
    return True


if __name__ == "__main__":
    run()
