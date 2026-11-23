import json
import os
import struct
import tempfile

import common
import f32
import half_weights


# A dense qwen3 whose every matrix product has nin a multiple of 32 (64 and 96), so CUDA's TF32 BF16 prompt tile takes each of them (docs/CUDA.md, BF16 prompt tile).
CONFIG = {"block_count": 2, "embedding_length": 64, "feed_forward_length": 96,
          "attention.head_count": 2, "attention.head_count_kv": 1,
          "attention.key_length": 32, "context_length": 128}
# The long text fills more than one 64-row tile, and the 70-token window scores it in one pass.
LONG = "the quick brown fox jumps over the lazy dog while seven zebras quietly v"
TEXTS = f32.TEXTS + [LONG]
CONTEXTS = (4, 16, 128)
GOLDEN = "baseline_bf16_tile.json"


def bf16(x):
    """`x` rounded to the nearest BF16, ties to even, as torch's bfloat16 conversion rounds the calibration's weights."""
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def tensors(tied, seed=12345):
    """f32.tensors of CONFIG with every matrix rounded to BF16, as a BF16 file holds it; norms stay F32."""
    return [(name, hf, shape, [bf16(v) for v in values] if len(shape) > 1 else values)
            for name, hf, shape, values in f32.tensors(tied, seed, CONFIG)]


def golden():
    with open(os.path.join(os.path.dirname(__file__), "data", GOLDEN), encoding="utf-8") as f:
        doc = json.load(f)
    assert doc["config"] == CONFIG and doc["texts"] == TEXTS, "%s: fixture config changed" % GOLDEN
    for fixture in doc["fixtures"]:
        fixture["weights"] = tensors(fixture["tied"])
        assert f32.weight_hash(fixture["weights"]) == fixture["weights_sha256"], "%s: fixture weights changed" % GOLDEN
    return doc


def witnessed(model, text, dtype, ubatch, expected):
    """The matrix paths of `logits` over `text`'s last two rows, which must be each device's `expected` set, or hold no f16-mma where it is None; the last row is held to the HF golden's case."""
    p = common.run_process(["logits", model, text, "--last", "2", "--top", "257", "--dtype", dtype, "--ubatch", str(ubatch)], cache="f32", text=True)
    assert p.returncode == 0, "bf16-tile logits failed: " + p.stdout + p.stderr
    devices = common.matrix_paths(p.stderr)["devices"]
    assert not any("f16-mma" in d for d in devices) if expected is None else all(set(d) == expected for d in devices), "bf16-tile %s ubatch %d over %d rows: paths %s, expected %s" % (dtype, ubatch, len(text), devices, sorted(expected or ()))
    fields = p.stdout.splitlines()[-1].split()
    assert int(fields[0]) == len(text) - 1, p.stdout
    return {int(i): float(v) for i, v in zip(fields[1::2], fields[2::2])}, common.hf_execution(p.stderr)


def run():
    if common.tensor_width_skip("bf16-tile"):
        return common.SKIPPED
    doc = golden()
    cuda = all(d.startswith("cuda") for d in os.environ.get("LLMX_DEVICE", "cpu").split(","))
    worst = 0.0
    with tempfile.TemporaryDirectory(prefix="llmx_bf16_tile_") as directory:
        for fixture in doc["fixtures"]:
            model = half_weights.write_variant(os.path.join(directory, "bf16-tile.gguf"), fixture["weights"], CONFIG, "qwen3", "bf16")
            name = "BF16 tile " + ("tied" if fixture["tied"] else "untied")
            error, _ = common.check_hf_fixture(name, model, fixture["cases"], fixture["perplexity"], LONG, (1, 2, 5, 16, 80))
            worst = max(worst, error)
            long_case = fixture["cases"][-1]
            # On CUDA every product of a call of two rows or more takes the tile, int8 included since BF16 has no 8-bit build; a call of one row, and f32, keep the exact products.
            for text, dtype, ubatch, expected in ((LONG, "f16", 16, {"f16-mma"}), (LONG, "f16", 80, {"f16-mma"}), (LONG, "int8", 16, {"f16-mma"}),
                                                  ("ab", "f16", 16, {"f16-mma"}), (LONG, "f16", 1, {"f32"}), (LONG, "f32", 16, {"f32"})):
                got, bounds = witnessed(model, text, dtype, ubatch, expected if cuda else None)
                case = next(c for c in fixture["cases"] if c["text"] == text)
                worst = max(worst, common.hf_logit_error("%s %s ubatch %d" % (name, dtype, ubatch), got, case["logits"], bound=bounds["logit"]))
    print("bf16-tile: BF16 weights, widths 64/96, all 257 logits and NLL vs HF at ubatch 1/2/5/16/80, threads 1/4, tied/untied; %s; max error %.8f  [ok]"
          % ("f16-mma witnessed on every product of 2 rows or more at f16 and int8, f32 at one row and at f32" if cuda else "no f16-mma off CUDA", worst))
    return True
