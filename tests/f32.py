import hashlib
import json
import math
import os
import struct
import tempfile

from common import run as cli
from tokenizer import build_byte_vocab, w_str


CONFIG = {"block_count": 2, "embedding_length": 37, "feed_forward_length": 19,
          "attention.head_count": 2, "attention.head_count_kv": 1,
          "attention.key_length": 10, "context_length": 16}
TEXTS = ["a", "ab", "abc", "abcdefg", "abcdefghijklm"]


def tensors(tied):
    state = 12345
    result = []

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
                                    ("attn_q_norm", "self_attn.q_norm", 10),
                                    ("attn_k_norm", "self_attn.k_norm", 10)):
            add(name + norm + ".weight", hf + mapped + ".weight", [width], True)
        for tensor, mapped, shape in (("attn_q", "self_attn.q_proj", [37, 20]),
                                      ("attn_k", "self_attn.k_proj", [37, 10]),
                                      ("attn_v", "self_attn.v_proj", [37, 10]),
                                      ("attn_output", "self_attn.o_proj", [20, 37]),
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


def write_model(path, weights):
    # A 34-byte Q8 tensor exposes unaligned F32 rows if the loader discards
    # file padding without preserving float alignment in its in-memory blob.
    entries = [("unused.weight", [32], 8, b"\0" * 34)]
    entries += [(name, shape, 0, struct.pack("<%df" % len(v), *v))
                for name, _, shape, v in weights]
    with open(path, "wb") as f:
        f.write(struct.pack("<IIQQ", 0x46554747, 3, len(entries), len(CONFIG) + 1))
        for name, value in CONFIG.items():
            w_str(f, "qwen3." + name)
            f.write(struct.pack("<II", 4, value))
        w_str(f, "tokenizer.ggml.tokens")
        f.write(struct.pack("<IIQ", 9, 8, 257))
        for token in build_byte_vocab() + ["<|endoftext|>"]:
            w_str(f, token)
        offset = 0
        for name, shape, kind, data in entries:
            w_str(f, name)
            f.write(struct.pack("<I", len(shape)))
            f.write(struct.pack("<%dQ" % len(shape), *shape))
            f.write(struct.pack("<IQ", kind, offset))
            offset += (len(data) + 31) // 32 * 32
        f.write(b"\0" * (-f.tell() % 32))
        for _, _, _, data in entries:
            f.write(data)
            f.write(b"\0" * (-len(data) % 32))


def run():
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
    print("f32: all 257 logits vs HF, tied/untied, batch/row/column tails, threads and PPL; max error %.8f  [ok]" % worst)
    return True


if __name__ == "__main__":
    run()
