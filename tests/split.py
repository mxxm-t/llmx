import os
import re
import struct
import subprocess
import tempfile

import common
import f32
import moe
import mxfp4
import qwen35
import tensor_split


# The layer split against one device through llmx-split-check (tools/split_check.cpp), on CPU backends: raw logits compared bit for bit over the prompt path, the prefill a split pipelines over its stages, greedy decode steps, the recompute by class a resume runs and a decoding sequence beside a fresh prompt.
# The tool names its own devices and cache type, so the configured device, shares and cache type do not reach it.
# Each split runs with f16 caches, the default, and with f32 caches, since a split is exact at either.
# LLMX_DTYPE reaches both placements through the tool's optional dtype argument; request records and completed paths are checked, including visible fallback.
# Three decode steps after the 13-token text fill the tiny models' 16-token context.
# The tensor-split fixtures, dense, qwen3moe, qwen35 and qwen35moe (tests/tensor_split.py), also run as one tensor group of two CPU backends against two stages of such groups, which must give the one group's bits (docs/TENSOR-SPLIT.md, section 4.4).
# The tiny qwen35 models run over two CPU backends and over four, a layer a stage, where the first and third stages hold only a linear-attention layer, which keeps a state and no KV; a model that keeps a state is not forked, so the tool recomputes it from no fork.
UBATCHES = (1, 3, 16)
CACHE_TYPES = ("f16", "f32")
STEPS = 3

# A Q8_0 model whose decode and prompt rows take different CPU float reductions, with a context past one 128-token CPU block, so the tool's recompute also runs from a fork at a block.
# Its histories pass the block after a 100-token prompt and 40 steps, and inside a 150-token prompt with 8 steps.
Q8_CONFIG = {"block_count": 2, "embedding_length": 64, "feed_forward_length": 128,
             "attention.head_count": 2, "attention.head_count_kv": 1,
             "attention.key_length": 32, "context_length": 256}
Q8_RUNS = ((100, 40), (150, 8))


def q8_0(values):
    """Values as Q8_0 blocks: an f16 scale of the block's largest magnitude over 127, then each value over that scale rounded to a signed byte."""
    out = bytearray()
    for i in range(0, len(values), 32):
        block = values[i:i + 32]
        scale = struct.pack("<e", max(abs(v) for v in block) / 127)
        d = struct.unpack("<e", scale)[0]
        out += scale + struct.pack("<32b", *[max(-127, min(127, round(v / d))) if d else 0 for v in block])
    return bytes(out)


def write_q8_model(path):
    """The tiny Qwen layout at Q8_CONFIG's sizes: norms and the embedding in F32, every matrix a product reads, the untied head included, in Q8_0."""
    state = 424242
    width, ff, hd = Q8_CONFIG["embedding_length"], Q8_CONFIG["feed_forward_length"], Q8_CONFIG["attention.key_length"]
    q, kv = Q8_CONFIG["attention.head_count"] * hd, Q8_CONFIG["attention.head_count_kv"] * hd
    plain, quantized = [], []

    def values(count, norm=False):
        nonlocal state
        result = []
        for _ in range(count):
            state = (1664525 * state + 1013904223) & 0xffffffff
            value = (((state >> 16) & 1023) - 512) / 8192
            result.append(1.0 + value if norm else value)
        return result

    plain.append(("token_embd.weight", None, [width, f32.VOCAB], values(width * f32.VOCAB)))
    plain.append(("output_norm.weight", None, [width], values(width, True)))
    for layer in range(Q8_CONFIG["block_count"]):
        name = "blk.%d." % layer
        for norm, size in (("attn_norm", width), ("ffn_norm", width), ("attn_q_norm", hd), ("attn_k_norm", hd)):
            plain.append((name + norm + ".weight", None, [size], values(size, True)))
        for tensor, shape in (("attn_q", [width, q]), ("attn_k", [width, kv]), ("attn_v", [width, kv]),
                              ("attn_output", [q, width]), ("ffn_gate", [width, ff]), ("ffn_up", [width, ff]),
                              ("ffn_down", [ff, width])):
            quantized.append((name + tensor + ".weight", shape, 8, q8_0(values(shape[0] * shape[1]))))
    quantized.append(("output.weight", [width, f32.VOCAB], 8, q8_0(values(width * f32.VOCAB))))
    return f32.write_model(path, plain, config=Q8_CONFIG, quantized=quantized)


def tool_path():
    return os.path.join(os.path.dirname(common.exe_path()), "llmx-split-check" + (".exe" if os.name == "nt" else ""))


def run_tool(tool, model, text, split, steps, ubatch, cache, tokens, single="cpu", width=1):
    """One split against one device, or at a tensor width against one group; the number of recomputes the tool ran from a fork, and of the rounds of verifies it ran."""
    dtype = os.environ.get("LLMX_DTYPE", "auto")
    args = [tool, model, text, single, split, str(steps), str(ubatch), cache, dtype, str(width)]
    p = subprocess.run(args, capture_output=True, encoding="utf-8", errors="replace", timeout=120)
    forked = re.search(r"whole and (\d+) from a fork", p.stdout)
    verifies = re.search(r"verifies: (\d+) rounds", p.stdout)
    records = [line for line in p.stderr.splitlines() if line.startswith("dtype:")]
    assert len(records) == 2 and all(line.startswith("dtype: %s -> " % dtype) for line in records), \
        "split: missing or mismatched dtype request records: " + p.stderr
    for (label, devices), record in zip((("single", len(single.split(","))), ("split", len(split.split(",")))), records):
        paths = re.findall(r"^%s device (\d+) matrix paths: (.+)$" % label, p.stdout, re.MULTILINE)
        assert [int(index) for index, _ in paths] == list(range(devices)), "split: missing execution paths: " + p.stdout
        effective = record.split(" -> ", 1)[1].split()[0]
        common.hf_bounds(effective, [values.split() for _, values in paths])
    assert p.returncode == 0 and ": %d tokens, %s caches;" % (tokens, cache) in p.stdout and "bit-identical" in p.stdout and forked and verifies, \
        "split differs from one device: %s\n%s%s" % (" ".join(args[1:]), p.stdout, p.stderr)
    return int(forked.group(1)), int(verifies.group(1))


def run(require=False):
    tool = tool_path()
    if not os.path.exists(tool):
        assert not require, "split: %s not found beside the executable" % tool
        print("split: SKIP - %s not found beside the executable" % tool)
        return common.SKIPPED
    for dtype in ("half", "F16", ""):
        p = subprocess.run([tool, "missing.gguf", "missing.txt", "cpu", "cpu,cpu", "1", "1", "f16", dtype],
                           capture_output=True, encoding="utf-8", errors="replace", timeout=10)
        assert p.returncode == 2 and "dtype must be auto, f16, bf16, f32 or int8" in p.stderr, \
            "split: invalid dtype was not refused before loading: " + p.stderr
    runs = 0
    with tempfile.TemporaryDirectory(prefix="llmx_split_") as directory:
        text = os.path.join(directory, "text.txt")
        with open(text, "w", encoding="utf-8", newline="") as f:
            f.write(f32.TEXTS[-1])
        models = [(f32.write_model(os.path.join(directory, "tiny-f32-%s.gguf" % name), f32.tensors(tied)), ["cpu,cpu"])
                  for name, tied in (("tied", True), ("untied", False))]
        models.append((f32.write_model(os.path.join(directory, "tiny-moe.gguf"), moe.tensors(), config=moe.CONFIG, arch="qwen3moe"),
                       ["cpu,cpu", "cpu,cpu,cpu"]))
        for name, tied, routed in mxfp4.VARIANTS:
            config, weights, packed = mxfp4.fixture(tied, routed)
            model = mxfp4.write_fixture(os.path.join(directory, "tiny-mxfp4-" + name + ".gguf"), config, weights, packed, routed)
            models.append((model, ["cpu,cpu"]))
        models += [(qwen35.write_fixture(directory, spec), ["cpu,cpu", "cpu,cpu,cpu,cpu"]) for spec in qwen35.FIXTURES if not spec["mtp"]]
        for model, splits in models:
            for split in splits:
                for ubatch in UBATCHES:
                    for cache in CACHE_TYPES:
                        run_tool(tool, model, text, split, STEPS, ubatch, cache, len(f32.TEXTS[-1]))
                        runs += 1
        grouped = [f32.write_model(os.path.join(directory, "even-f32.gguf"), f32.tensors(True, config=tensor_split.CONFIG, vocab=tensor_split.VOCAB),
                                   config=tensor_split.CONFIG, tokens=tensor_split.TOKENS),
                   f32.write_model(os.path.join(directory, "even-moe.gguf"), moe.tensors(tensor_split.MOE_SEED, tensor_split.MOE_CONFIG, tensor_split.VOCAB),
                                   config=tensor_split.MOE_CONFIG, arch="qwen3moe", tokens=tensor_split.TOKENS),
                   qwen35.write_fixture(directory, tensor_split.QWEN35, tokens=tensor_split.TOKENS),
                   qwen35.write_fixture(directory, tensor_split.QWEN35MOE, tokens=tensor_split.TOKENS)]
        for model in grouped:
            for ubatch in UBATCHES:
                for cache in CACHE_TYPES:
                    run_tool(tool, model, text, "cpu,cpu,cpu,cpu", STEPS, ubatch, cache, len(f32.TEXTS[-1]), single="cpu,cpu", width=2)
                    runs += 1
        q8 = write_q8_model(os.path.join(directory, "tiny-q8_0.gguf"))
        for length, steps in Q8_RUNS:
            long_text = os.path.join(directory, "text-%d.txt" % length)
            with open(long_text, "w", encoding="utf-8", newline="") as f:
                f.write(("The layer split recomputes a paused request by class. " * 4)[:length])
            for ubatch in UBATCHES:
                for cache in CACHE_TYPES:
                    # One device and the split each recompute from a fork at the block, so every run has two; the decode's 40 steps hold rounds of verifies.
                    forked, verifies = run_tool(tool, q8, long_text, "cpu,cpu", steps, ubatch, cache, length)
                    assert forked == 2, "split: %d recomputes from a fork on the Q8_0 model's %d-token history, against 2" % (forked, length + steps)
                    assert verifies or steps < 17, "split: no rounds of verifies after %d decode steps" % steps
                    runs += 1
    print("split: %d runs of the tiny F32 (tied, untied), MoE, MXFP4 (dense tied/untied and MoE), qwen35 and Q8_0 models over 2, 3 and 4 CPU backends at ubatch %s with %s caches, "
          "bit-identical to one, the tensor-split fixtures (dense, qwen3moe, qwen35 and qwen35moe) as two stages of groups of two against one group, the Q8_0 model's recompute also from a fork at a block and its verifies of drafts  [ok]" % (runs, "/".join(map(str, UBATCHES)), " and ".join(CACHE_TYPES)))
    return True


if __name__ == "__main__":
    run()
