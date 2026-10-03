import hashlib
import math
import os

import common
import spec_decode as sd
from f32 import TEXTS, VOCAB, hf_name, write_model


VARIANTS = (("dense-untied", False, False), ("dense-tied", True, False), ("moe", False, True))


def fixture(tied, moe, type_id, make_block, seed=12345, context=16):
    """Raw-block matrices in the shared tiny dense/MoE layout, independently decoded for HF; norms and routers stay F32."""
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
            payload = b"".join(make_block(next_value, b) for b in range(n // 32))
            values = sd.decode(type_id, payload, n)
            packed.append((name, shape, type_id, payload))
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
    """Probe a raw-block fixture through the common device refusal policy."""
    if not os.environ.get("LLMX_DEVICE"):
        return False
    rc, out = common.run_f32_cache(["logits", model, TEXTS[0], "--top", "1"])
    return common.device_lacks_kernel(rc, out)
