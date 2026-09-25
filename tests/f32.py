import hashlib
import json
import math
import os
import re
import struct
import tempfile

import common
from common import run_f32_cache as cli
from tokenizer import build_byte_vocab, w_str


CONFIG = {"block_count": 2, "embedding_length": 37, "feed_forward_length": 19,
          "attention.head_count": 2, "attention.head_count_kv": 1,
          "attention.key_length": 42, "context_length": 16}
TEXTS = ["a", "ab", "abc", "abcdefg", "abcdefghijklm"]
# One token per byte, then <|endoftext|>.
VOCAB = 257

# The HF Qwen3 parameter each qwen3 GGUF tensor holds: the tensors outside the blocks, then those of block N by the name after "blk.N.", which take model.layers.N.<name>.weight.
# The tiny models here and in tests/moe.py, and tools/gen_baseline.py file-exact, all name their parameters from these.
HF_NAMES = {"token_embd.weight": "model.embed_tokens.weight", "output_norm.weight": "model.norm.weight", "output.weight": "lm_head.weight"}
HF_BLOCK_NAMES = {"attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm",
                  "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm",
                  "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
                  "attn_output": "self_attn.o_proj", "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj",
                  "ffn_down": "mlp.down_proj"}


def hf_name(name):
    """The HF Qwen3 parameter the qwen3 GGUF tensor `name` holds; a tensor with none raises ValueError."""
    match = re.fullmatch(r"blk\.(\d+)\.(\w+)\.weight", name)
    if match and match[2] in HF_BLOCK_NAMES:
        return "model.layers.%s.%s.weight" % (match[1], HF_BLOCK_NAMES[match[2]])
    if name in HF_NAMES:
        return HF_NAMES[name]
    raise ValueError("GGUF tensor %s has no HF Qwen3 parameter" % name)


def tensors(tied):
    state = 12345
    result = []
    width, ff, hd = CONFIG["embedding_length"], CONFIG["feed_forward_length"], CONFIG["attention.key_length"]
    q, kv = CONFIG["attention.head_count"] * hd, CONFIG["attention.head_count_kv"] * hd

    def add(name, shape, norm=False):
        nonlocal state
        values = []
        for _ in range(math.prod(shape)):
            state = (1664525 * state + 1013904223) & 0xffffffff
            value = (((state >> 16) & 1023) - 512) / 8192
            values.append(1.0 + value if norm else value)
        result.append((name, hf_name(name), shape, values))

    add("token_embd.weight", [width, VOCAB])
    add("output_norm.weight", [width], True)
    for layer in range(CONFIG["block_count"]):
        name = "blk.%d." % layer
        for norm, size in (("attn_norm", width), ("ffn_norm", width), ("attn_q_norm", hd), ("attn_k_norm", hd)):
            add(name + norm + ".weight", [size], True)
        for tensor, shape in (("attn_q", [width, q]), ("attn_k", [width, kv]), ("attn_v", [width, kv]),
                              ("attn_output", [q, width]), ("ffn_gate", [width, ff]), ("ffn_up", [width, ff]),
                              ("ffn_down", [ff, width])):
            add(name + tensor + ".weight", shape)
    if not tied:
        add("output.weight", [width, VOCAB])
    return result


def weight_hash(weights):
    return hashlib.sha256(b"".join(struct.pack("<%df" % len(v), *v)
                                   for _, _, _, v in weights)).hexdigest()


def golden(name):
    """The golden `name` in tests/data for this model, checked against CONFIG and against the weights tensors() builds.
    A golden of several fixtures holds each one's tied flag and weight hash, and one of a single model holds its hash at the top level, for the untied weights.
    Each checked fixture, or the golden itself, gets those weights under "weights"."""
    with open(os.path.join(os.path.dirname(__file__), "data", name), encoding="utf-8") as f:
        doc = json.load(f)
    assert doc["config"] == CONFIG, "%s: fixture config changed" % name
    for fixture in doc.get("fixtures", [doc]):
        fixture["weights"] = tensors(fixture.get("tied", False))
        assert weight_hash(fixture["weights"]) == fixture["weights_sha256"], "%s: fixture weights changed" % name
    return doc


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
                f.write(struct.pack("<IIQ", 9, 8, VOCAB))
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


def check_logits_input(directory, model, cases):
    """`logits --file` against the same prompt inline, then the `--last` rows of the prompt, and of its head continued by `--then-ids`, against every fixture case at its position.
    The rows the prompt prints over several passes, and those of its head continued by the same ids, must be the bytes it prints in one pass.
    Returns the largest logit error."""
    text = TEXTS[-1]
    rc, inline = cli(["logits", model, text, "--top", "257"])
    assert rc == 0, "logits failed: " + inline
    path = os.path.join(directory, "prompt.txt")
    with open(path, "wb") as f:
        f.write(text.encode())
    rc, out = cli(["logits", model, "--file", path, "--top", "257"])
    assert rc == 0 and out == inline, "logits --file differs from the inline prompt: " + out
    # The vocabulary is one token per byte, id equal to the byte, so a case's last position is its length less one.
    head = TEXTS[2]
    ids = os.path.join(directory, "tail.ids")
    tail = [str(b) for b in text[len(head):].encode()]
    with open(ids, "w") as f:
        f.write(" ".join(tail[:3]) + "\n" + "\t".join(tail[3:]) + "\n")
    # The first two leave the prompt's first positions out, the second over several passes, and the third prints exactly the appended positions.
    worst = 0.0
    printed = []
    for args, last in (([text], len(text) - 2), ([text, "--ubatch", "5"], len(text) - 2), ([head, "--then-ids", ids], len(tail))):
        rc, out = cli(["logits", model] + args + ["--last", str(last), "--top", "257"])
        assert rc == 0, "logits --last failed: " + out
        lines = out.splitlines()
        assert lines[0] == "tokens: %d" % len(text), out
        # Each position once and in order, so a position printed twice fails.
        positions = [int(line.split()[0]) for line in lines[1:]]
        assert len(positions) == last and positions == list(range(len(text) - last, len(text))), (args, positions)
        rows = dict(zip(positions, lines[1:]))
        printed.append(rows)
        checked = [case for case in cases if len(case["text"]) - 1 in rows]
        assert checked, args
        for case in checked:
            fields = rows[len(case["text"]) - 1].split()
            got = {int(i): float(v) for i, v in zip(fields[1::2], fields[2::2])}
            worst = max(worst, common.hf_logit_error("F32 --last", got, case["logits"]))
    # A position computes the same bytes however its prompt arrives: in one pass, over passes of five tokens, or as a head continued by ids, whose rows are positions 3 to 12.
    one_pass, sliced, continued = printed
    for position in one_pass:
        assert sliced[position] == one_pass[position], ("--ubatch 5 differs from one pass", position, sliced[position], one_pass[position])
    for position in continued:
        assert continued[position] == one_pass[position], ("--then-ids differs from one pass", position, continued[position], one_pass[position])
    return worst


def run():
    if common.f32_cache_skip("f32"):
        return True
    worst = 0.0
    with tempfile.TemporaryDirectory(prefix="llmx_f32_") as directory:
        for fixture in golden("baseline_f32.json")["fixtures"]:
            model = os.path.join(directory, "tiny-f32.gguf")
            write_model(model, fixture["weights"])
            error, _ = common.check_hf_fixture("F32", model, fixture["cases"], fixture["perplexity"], TEXTS[-1],
                                               (1, 2, 3, 5, 16))
            worst = max(worst, error, check_logits_input(directory, model, fixture["cases"]))
        # The bench measures on top of a history when asked for a depth, and refuses one with batched decode.
        rc, out = cli(["bench", "--model", model, "--p", "4", "--n", "2", "--r", "1", "--depth", "6"])
        assert rc == 0 and "pp4 @ d6" in out and "tg2 @ d6" in out, "bench --depth failed: " + out
        rc, out = cli(["bench", "--model", model, "--depth", "6", "--seqs", "2"])
        assert rc != 0, "bench --depth accepted batched decode"
        # Batched decode holds both sequences' prompts and tokens at once, which this model's one context, a single block, cannot.
        rc, out = cli(["bench", "--model", model, "--p", "4", "--n", "2", "--r", "2", "--seqs", "2"])
        reports = re.findall(r"^bench: (.+?)\s+(\S+) \+- (\S+) tok/s  \((\d+) runs\)$", out, re.M)
        assert rc == 0 and [(what, runs) for what, _, _, runs in reports] == [("pp4", "2"), ("x2 tg2", "2")], "bench --seqs 2 failed: " + out
        assert all(math.isfinite(float(mean)) and float(mean) > 0 and math.isfinite(float(sd)) for _, mean, sd, _ in reports), out
    print("f32: all 257 logits vs HF, tied/untied, batch/row/column tails, threads, PPL, --file and --last/--then-ids rows, the same bytes over passes, bench --seqs 2; max error %.8f  [ok]" % worst)
    return True


if __name__ == "__main__":
    run()
