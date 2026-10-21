import json
import os
import tempfile

import common
import spec_decode as sd
from block_fixture import VARIANTS, device_skip, packed_hash, write_fixture
from block_fixture import fixture as block_fixture
from f32 import TEXTS, weight_hash
from raw_blocks import iq4_xs_block


def raw_block(next_value, b):
    # Signed modest half scales and all six subscale bits exercise the format without deliberately overflowing the HF model.
    bits = next_value()
    d = ((bits >> 16) & 0x8000) | 0x0400 | ((bits >> 8) & 0x03ff)
    if b % 257 in (0, 1, 2):
        d = (d & 0x8000) | (0, 1, 0x03ff)[b % 257]
    scales = [(next_value() >> 16) & 63 for _ in range(8)]
    codes = [(next_value() >> 16) & 15 for _ in range(256)]
    return iq4_xs_block(d, scales, codes)


def fixture(tied, moe, seed=12345, context=16, split=False):
    """Raw IQ4_XS matrices at whole-block widths, F32 norms/router, and independently decoded HF weights."""
    return block_fixture(tied, moe, sd.IQ4_XS, raw_block, seed, context, split)


def run():
    if common.f32_cache_skip("iq4-xs"):
        return common.SKIPPED
    split = common.tensor_width() > 1
    path = os.path.join(os.path.dirname(__file__), "data", "baseline_iq4_xs_split.json" if split else "baseline_iq4_xs.json")
    with open(path, encoding="utf-8") as f:
        goldens = json.load(f)
    assert goldens["transformers_version"] == "4.55.2", "IQ4_XS reference version changed"
    assert [x["name"] for x in goldens["fixtures"]] == [x[0] for x in VARIANTS], "IQ4_XS fixture coverage changed"
    worst, comparisons, models = 0.0, 0, []
    with tempfile.TemporaryDirectory(prefix="llmx_iq4_xs_hf_") as directory:
        for (name, tied, moe), golden in zip(VARIANTS, goldens["fixtures"]):
            assert [case["text"] for case in golden["cases"]] == TEXTS, "IQ4_XS prompt coverage changed"
            assert [case["context"] for case in golden["perplexity"]] == [4, 16], "IQ4_XS NLL coverage changed"
            config, weights, packed = fixture(tied, moe, split=split)
            assert config == golden["config"] and weight_hash(weights) == golden["weights_sha256"], "IQ4_XS decoded weights changed"
            assert packed_hash(packed) == golden["packed_sha256"], "IQ4_XS packed weights changed"
            model = write_fixture(os.path.join(directory, name + ".gguf"), config, weights, packed, moe)
            models.append((name, model, golden))
        if device_skip(models[0][1]):
            print("iq4-xs: SKIP - selected device has no IQ4_XS kernel")
            return common.SKIPPED
        for name, model, golden in models:
            error, count = common.check_hf_fixture("IQ4_XS " + name, model, golden["cases"], golden["perplexity"], TEXTS[-1], (1, 2, 3, 5, 16), family="iq4-xs", fixture=golden)
            worst = max(worst, error)
            comparisons += count
    assert comparisons == 150, "IQ4_XS full-logit coverage changed"
    print("iq4-xs: dense tied/untied and MoE vs HF, 150 full-logit cases and 12 NLL checks; max error %.8f  [ok]" % worst)
    return True
