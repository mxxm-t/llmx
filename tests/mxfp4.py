import json
import os
import subprocess
import tempfile

import common
import spec_decode as sd
from block_fixture import VARIANTS, device_skip, packed_hash, write_fixture
from block_fixture import fixture as block_fixture
from f32 import TEXTS, weight_hash


# Below, at and above the two measured narrow MXFP4 prompt crossovers.
PROMPT_TEXTS = [("abcdefghijklmnopqrstuvwxyz" * 3)[:n] for n in (39, 40, 41, 63, 64, 65)]


def raw_block(next_value, b):
    e = 118 + ((next_value() >> 24) % 5)
    codes = [(next_value() >> 16) & 15 for _ in range(32)]
    if b % 257 in (0, 1):
        e = b % 257
    elif b % 257 == 2:
        e, codes = 0, [0] * 32
    return bytes([e]) + bytes(codes[j] | (codes[j + 16] << 4) for j in range(16))


def fixture(tied, moe, seed=12345, context=16):
    """Raw MXFP4 matrices plus F32 norms/router, and the independent spec-decoded weights for HF."""
    return block_fixture(tied, moe, sd.MXFP4, raw_block, seed, context)


def run(require=False):
    if common.f32_cache_skip("mxfp4"):
        return True
    path = os.path.join(os.path.dirname(__file__), "data", "baseline_mxfp4.json")
    with open(path, encoding="utf-8") as f:
        goldens = json.load(f)
    assert goldens["transformers_version"] == "4.55.2", "MXFP4 reference version changed"
    assert [x["name"] for x in goldens["fixtures"]] == [x[0] for x in VARIANTS], "MXFP4 fixture coverage changed"
    tool = os.path.join(os.path.dirname(common.exe_path()), "llmx-cpu-f32-check" + (".exe" if os.name == "nt" else ""))
    if not os.path.isfile(tool):
        assert not require, "mxfp4: llmx-cpu-f32-check is required beside the executable"
        print("mxfp4: SKIP F32 activation control - llmx-cpu-f32-check is not built")
    worst, control_worst = 0.0, 0.0
    models = []
    with tempfile.TemporaryDirectory(prefix="llmx_mxfp4_hf_") as directory:
        for (name, tied, moe), golden in zip(VARIANTS, goldens["fixtures"]):
            config, weights, packed = fixture(tied, moe)
            assert config == golden["config"] and weight_hash(weights) == golden["weights_sha256"], "MXFP4 decoded weights changed"
            assert packed_hash(packed) == golden["packed_sha256"], "MXFP4 packed weights changed"
            model = write_fixture(os.path.join(directory, name + ".gguf"), config, weights, packed, moe)
            models.append((name, model, golden))
        if device_skip(models[0][1]):
            print("mxfp4: SKIP - selected device has no MXFP4 kernel")
            return common.SKIPPED
        if os.path.isfile(tool):
            for name, model, golden in models:
                for threads in (1, 4):
                    for case in golden["cases"]:
                        for ubatch in (1, 2, 3, 5, 16):
                            result = subprocess.run([tool, model, case["text"], str(threads), str(ubatch)], capture_output=True, text=True, timeout=30)
                            assert result.returncode == 0, "MXFP4 F32 activation control failed: " + result.stderr
                            control_worst = max(control_worst, common.hf_logit_error("MXFP4 F32 activations " + name,
                                                dict(zip(*common.parse_logits(result.stdout))), case["logits"]))
            print("mxfp4: original F32 activations vs HF, 150 cases; max error %.8f  [ok]" % control_worst)
        for name, model, golden in models:
            error, _ = common.check_hf_fixture("MXFP4 " + name, model, golden["cases"], golden["perplexity"], TEXTS[-1], (1, 2, 3, 5, 16))
            worst = max(worst, error)
    with open(os.path.join(os.path.dirname(__file__), "data", "baseline_mxfp4_prompt.json"), encoding="utf-8") as f:
        prompts = json.load(f)
    assert prompts["transformers_version"] == "4.55.2", "MXFP4 prompt reference version changed"
    assert [x["name"] for x in prompts["fixtures"]] == [x[0] for x in VARIANTS], "MXFP4 prompt coverage changed"
    with tempfile.TemporaryDirectory(prefix="llmx_mxfp4_prompt_hf_") as directory:
        for (name, tied, moe), golden in zip(VARIANTS, prompts["fixtures"]):
            config, weights, packed = fixture(tied, moe, context=128)
            assert config == golden["config"] and weight_hash(weights) == golden["weights_sha256"], "MXFP4 prompt weights changed"
            assert packed_hash(packed) == golden["packed_sha256"], "MXFP4 prompt blocks changed"
            assert [case["text"] for case in golden["cases"]] == PROMPT_TEXTS, "MXFP4 prompt threshold texts changed"
            model = write_fixture(os.path.join(directory, name + ".gguf"), config, weights, packed, moe)
            error, _ = common.check_hf_fixture("MXFP4 prompt " + name, model, golden["cases"], golden["perplexity"],
                                                PROMPT_TEXTS[-1], (1, 3, 65))
            worst = max(worst, error)
    print("mxfp4: spec-decoded HF logits and NLL, dense tied/untied and MoE; max error %.8f  [ok]" % worst)
    return True
