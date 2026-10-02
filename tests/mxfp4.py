import hashlib
import json
import math
import os
import subprocess
import tempfile

import common
import spec_decode as sd
from f32 import TEXTS, VOCAB, hf_name, weight_hash, write_model


VARIANTS = (("dense-untied", False, False), ("dense-tied", True, False), ("moe", False, True))
# Below, at and above the two measured narrow MXFP4 prompt crossovers.
PROMPT_TEXTS = [("abcdefghijklmnopqrstuvwxyz" * 3)[:n] for n in (39, 40, 41, 63, 64, 65)]


def fixture(tied, moe, seed=12345, context=16):
    """Raw MXFP4 matrices plus F32 norms/router, and the independent spec-decoded weights for HF."""
    config = {"block_count": 2, "embedding_length": 96, "feed_forward_length": 160,
              "attention.head_count": 2, "attention.head_count_kv": 1,
              "attention.key_length": 32, "context_length": context}
    if moe:
        config.update(expert_count=4, expert_used_count=2, expert_feed_forward_length=160)
    state = seed
    weights, packed = [], []

    def next_value():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xffffffff
        return state

    def add(name, shape, norm=False, router=False, hf=None):
        n = math.prod(shape)
        if norm or router:
            values = [(((next_value() >> 16) & 1023) - 512) / 8192 for _ in range(n)]
            values = [1.0 + x if norm else x * 4 for x in values]
        else:
            assert shape[0] % 32 == 0
            payload = bytearray()
            for b in range(n // 32):
                e = 118 + ((next_value() >> 24) % 5)
                codes = [(next_value() >> 16) & 15 for _ in range(32)]
                if b % 257 in (0, 1):
                    e = b % 257
                elif b % 257 == 2:
                    e, codes = 0, [0] * 32
                payload.append(e)
                payload.extend(codes[j] | (codes[j + 16] << 4) for j in range(16))
            payload = bytes(payload)
            values = sd.decode(sd.MXFP4, payload, n)
            packed.append((name, shape, sd.MXFP4, payload))
        weights.append((name, hf or hf_name(name), shape, values))

    add("token_embd.weight", [96, VOCAB])
    add("output_norm.weight", [96], norm=True)
    for layer in range(2):
        prefix, hf = "blk.%d." % layer, "model.layers.%d." % layer
        for name, width in (("attn_norm", 96), ("ffn_norm", 96), ("attn_q_norm", 32), ("attn_k_norm", 32)):
            add(prefix + name + ".weight", [width], norm=True)
        for name, shape in (("attn_q", [96, 64]), ("attn_k", [96, 32]), ("attn_v", [96, 32]), ("attn_output", [64, 96])):
            add(prefix + name + ".weight", shape)
        if moe:
            add(prefix + "ffn_gate_inp.weight", [96, 4], router=True, hf=hf + "mlp.gate.weight")
            for name, mapped, shape in (("ffn_gate_exps", "gate_proj", [96, 160, 4]),
                                        ("ffn_up_exps", "up_proj", [96, 160, 4]),
                                        ("ffn_down_exps", "down_proj", [160, 96, 4])):
                add(prefix + name + ".weight", shape, hf=[hf + "mlp.experts.%d.%s.weight" % (e, mapped) for e in range(4)])
        else:
            for name, shape in (("ffn_gate", [96, 160]), ("ffn_up", [96, 160]), ("ffn_down", [160, 96])):
                add(prefix + name + ".weight", shape)
    if not tied:
        add("output.weight", [96, VOCAB])
    return config, weights, packed


def packed_hash(packed):
    return hashlib.sha256(b"".join(data for _, _, _, data in packed)).hexdigest()


def write_fixture(path, config, weights, packed, moe):
    """Write the raw-block model, keeping norms and the router in F32."""
    names = {name for name, _, _, _ in packed}
    floats = [t for t in weights if t[0] not in names]
    return write_model(path, floats, config=config, arch="qwen3moe" if moe else "qwen3", quantized=packed)


def device_skip(model):
    """Use the shared refusal policy for this CPU-first type on a selected device."""
    if not os.environ.get("LLMX_DEVICE"):
        return False
    rc, out = common.run_f32_cache(["logits", model, TEXTS[0], "--top", "1"])
    return common.device_lacks_kernel(rc, out)


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
