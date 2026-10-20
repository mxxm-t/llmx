import json
import os
import struct
import tempfile

import common
import spec_decode as sd
from block_fixture import VARIANTS, device_skip, packed_hash, write_fixture
from block_fixture import fixture as block_fixture
from f32 import TEXTS, weight_hash


def raw_block(next_value, b):
    bits = next_value()
    sign = (bits >> 16) & 0x8000
    exponent = 7 + ((bits >> 24) % 3)
    scale = sign | (exponent << 10)
    minimum = (sign ^ 0x8000) | ((exponent + 4) << 10)
    codes = [(next_value() >> 16) & 31 for _ in range(32)]
    if b % 257 in (0, 1, 2):
        scale = sign | (0, 1, 0x03ff)[b % 257]
        minimum = sign ^ 0x8000
    high = sum((code >> 4) << i for i, code in enumerate(codes))
    low = bytes((codes[j] & 15) | ((codes[j + 16] & 15) << 4) for j in range(16))
    return struct.pack("<HHI", scale, minimum, high) + low


def fixture(tied, moe, seed=12345, context=16, split=False):
    """Raw Q5_1 matrices, F32 norms/router, and independently decoded HF weights."""
    return block_fixture(tied, moe, sd.Q5_1, raw_block, seed, context, split)


def run():
    if common.f32_cache_skip("q5-1"):
        return common.SKIPPED
    split = common.tensor_width() > 1
    path = os.path.join(os.path.dirname(__file__), "data", "baseline_q5_1_split.json" if split else "baseline_q5_1.json")
    with open(path, encoding="utf-8") as f:
        goldens = json.load(f)
    assert goldens["transformers_version"] == "4.55.2", "Q5_1 reference version changed"
    assert [x["name"] for x in goldens["fixtures"]] == [x[0] for x in VARIANTS], "Q5_1 fixture coverage changed"
    worst, comparisons, models = 0.0, 0, []
    with tempfile.TemporaryDirectory(prefix="llmx_q5_1_hf_") as directory:
        for (name, tied, moe), golden in zip(VARIANTS, goldens["fixtures"]):
            assert [case["text"] for case in golden["cases"]] == TEXTS, "Q5_1 prompt coverage changed"
            assert [case["context"] for case in golden["perplexity"]] == [4, 16], "Q5_1 NLL coverage changed"
            config, weights, packed = fixture(tied, moe, split=split)
            assert config == golden["config"] and weight_hash(weights) == golden["weights_sha256"], "Q5_1 decoded weights changed"
            assert packed_hash(packed) == golden["packed_sha256"], "Q5_1 packed weights changed"
            model = write_fixture(os.path.join(directory, name + ".gguf"), config, weights, packed, moe)
            models.append((name, model, golden))
        if device_skip(models[0][1]):
            print("q5-1: SKIP - selected device has no Q5_1 kernel")
            return common.SKIPPED
        for name, model, golden in models:
            error, count = common.check_hf_fixture("Q5_1 " + name, model, golden["cases"], golden["perplexity"], TEXTS[-1], (1, 2, 3, 5, 16), family="q5-1")
            worst = max(worst, error)
            comparisons += count
    assert comparisons == 150, "Q5_1 full-logit coverage changed"
    print("q5-1: dense tied/untied and MoE vs HF, 150 full-logit cases and 12 NLL checks; max error %.8f  [ok]" % worst)
    return True
