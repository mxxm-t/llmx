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
    # Modest signed scales keep the HF model finite; sparse zero/subnormal scales still reach embedding, dense and routed products.
    bits = next_value()
    scale = ((bits >> 16) & 0x8000) | ((3 + ((bits >> 24) % 3)) << 10)
    codes = [(next_value() >> 16) & 15 for _ in range(32)]
    if b % 257 in (0, 1, 2):
        scale = (scale & 0x8000) | (0, 1, 0x03ff)[b % 257]
    return struct.pack("<H", scale) + bytes(codes[j] | (codes[j + 16] << 4) for j in range(16))


def fixture(tied, moe, seed=12345, context=16):
    """Raw IQ4_NL matrices, F32 norms/router, and independently decoded HF weights."""
    return block_fixture(tied, moe, sd.IQ4_NL, raw_block, seed, context)


def run():
    if common.f32_cache_skip("iq4-nl"):
        return common.SKIPPED
    path = os.path.join(os.path.dirname(__file__), "data", "baseline_iq4_nl.json")
    with open(path, encoding="utf-8") as f:
        goldens = json.load(f)
    assert goldens["transformers_version"] == "4.55.2", "IQ4_NL reference version changed"
    assert [x["name"] for x in goldens["fixtures"]] == [x[0] for x in VARIANTS], "IQ4_NL fixture coverage changed"
    worst, comparisons, models = 0.0, 0, []
    with tempfile.TemporaryDirectory(prefix="llmx_iq4_nl_hf_") as directory:
        for (name, tied, moe), golden in zip(VARIANTS, goldens["fixtures"]):
            assert [case["text"] for case in golden["cases"]] == TEXTS, "IQ4_NL prompt coverage changed"
            assert [case["context"] for case in golden["perplexity"]] == [4, 16], "IQ4_NL NLL coverage changed"
            config, weights, packed = fixture(tied, moe)
            assert config == golden["config"] and weight_hash(weights) == golden["weights_sha256"], "IQ4_NL decoded weights changed"
            assert packed_hash(packed) == golden["packed_sha256"], "IQ4_NL packed weights changed"
            model = write_fixture(os.path.join(directory, name + ".gguf"), config, weights, packed, moe)
            models.append((name, model, golden))
        if device_skip(models[0][1]):
            print("iq4-nl: SKIP - selected device has no IQ4_NL kernel")
            return common.SKIPPED
        for name, model, golden in models:
            error, count = common.check_hf_fixture("IQ4_NL " + name, model, golden["cases"], golden["perplexity"], TEXTS[-1], (1, 2, 3, 5, 16))
            worst = max(worst, error)
            comparisons += count
    assert comparisons == 150, "IQ4_NL full-logit coverage changed"
    print("iq4-nl: dense tied/untied and MoE vs HF, 150 full-logit cases and 12 NLL checks; max error %.8f  [ok]" % worst)
    return True
