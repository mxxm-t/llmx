import json
import os
import tempfile

import common
import spec_decode as sd
from block_fixture import VARIANTS, device_skip, packed_hash, write_fixture
from block_fixture import fixture as block_fixture
from f32 import TEXTS, weight_hash
from raw_blocks import iq3_s_block


def raw_block(next_value, b):
    # Mixed dictionary indices, signs and subscales at modest finite magnitudes.
    bits = next_value()
    d = ((bits >> 16) & 0x8000) | 0x0400 | ((bits >> 8) & 0x03ff)
    if b % 257 in (0, 1, 2):
        d = (d & 0x8000) | (0, 1, 0x03ff)[b % 257]
    indices = [(next_value() >> 16) & 511 for _ in range(64)]
    scales = [(next_value() >> 16) & 15 for _ in range(8)]
    signs = [bool((next_value() >> 16) & 1) for _ in range(256)]
    return iq3_s_block(d, indices, scales, signs)


def fixture(tied, moe, seed=12345, context=16, split=False):
    """Raw IQ3_S matrices at whole-block widths, F32 norms/router, and independently decoded HF weights."""
    return block_fixture(tied, moe, sd.IQ3_S, raw_block, seed, context, split)


def run():
    if common.f32_cache_skip("iq3-s"):
        return common.SKIPPED
    split = common.tensor_width() > 1
    path = os.path.join(os.path.dirname(__file__), "data", "baseline_iq3_s_split.json" if split else "baseline_iq3_s.json")
    with open(path, encoding="utf-8") as f:
        goldens = json.load(f)
    assert goldens["transformers_version"] == "4.55.2", "IQ3_S reference version changed"
    assert [x["name"] for x in goldens["fixtures"]] == [x[0] for x in VARIANTS], "IQ3_S fixture coverage changed"
    worst, comparisons, models = 0.0, 0, []
    with tempfile.TemporaryDirectory(prefix="llmx_iq3_s_hf_") as directory:
        for (name, tied, moe), golden in zip(VARIANTS, goldens["fixtures"]):
            assert [case["text"] for case in golden["cases"]] == TEXTS, "IQ3_S prompt coverage changed"
            assert [case["context"] for case in golden["perplexity"]] == [4, 16], "IQ3_S NLL coverage changed"
            config, weights, packed = fixture(tied, moe, split=split)
            assert config == golden["config"] and weight_hash(weights) == golden["weights_sha256"], "IQ3_S decoded weights changed"
            assert packed_hash(packed) == golden["packed_sha256"], "IQ3_S packed weights changed"
            model = write_fixture(os.path.join(directory, name + ".gguf"), config, weights, packed, moe)
            models.append((name, model, golden))
        if device_skip(models[0][1]):
            print("iq3-s: SKIP - selected device has no IQ3_S kernel")
            return common.SKIPPED
        for name, model, golden in models:
            error, count = common.check_hf_fixture("IQ3_S " + name, model, golden["cases"], golden["perplexity"], TEXTS[-1], (1, 2, 3, 5, 16), family="iq3-s")
            worst = max(worst, error)
            comparisons += count
    assert comparisons == 150, "IQ3_S full-logit coverage changed"
    print("iq3-s: dense tied/untied and MoE vs HF, 150 full-logit cases and 12 NLL checks; max error %.8f  [ok]" % worst)
    return True
