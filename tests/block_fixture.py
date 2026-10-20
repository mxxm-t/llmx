import hashlib
import json
import tempfile
import math
import os

import common
import spec_decode as sd
from f32 import TEXTS, VOCAB, hf_name, weight_hash, write_model
from tokenizer import build_byte_vocab


VARIANTS = (("dense-untied", False, False), ("dense-tied", True, False), ("moe", False, True))


def fixture(tied, moe, type_id, make_block, seed=12345, context=16, split=False):
    """Raw-block matrices in the shared tiny dense/MoE layout, independently decoded for HF; norms and routers stay F32."""
    config = {"block_count": 2, "embedding_length": 96, "feed_forward_length": 160,
              "attention.head_count": 2, "attention.head_count_kv": 1,
              "attention.key_length": 32, "context_length": context}
    if moe:
        config.update(expert_count=4, expert_used_count=2, expert_feed_forward_length=160)
    vocab = VOCAB
    if split:
        config.update(embedding_length=128, feed_forward_length=256,
                      **{"attention.head_count": 4, "attention.head_count_kv": 2})
        if moe:
            config["expert_feed_forward_length"] = 256
            config["expert_used_count"] = config["expert_count"]
        vocab = 260
    width, hidden = config["embedding_length"], config["feed_forward_length"]
    head = config["attention.key_length"]
    query, kv = head * config["attention.head_count"], head * config["attention.head_count_kv"]
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

    add("token_embd.weight", [width, vocab])
    add("output_norm.weight", [width], norm=True)
    for layer in range(2):
        prefix, hf = "blk.%d." % layer, "model.layers.%d." % layer
        for name, size in (("attn_norm", width), ("ffn_norm", width), ("attn_q_norm", head), ("attn_k_norm", head)):
            add(prefix + name + ".weight", [size], norm=True)
        for name, shape in (("attn_q", [width, query]), ("attn_k", [width, kv]), ("attn_v", [width, kv]), ("attn_output", [query, width])):
            add(prefix + name + ".weight", shape)
        if moe:
            add(prefix + "ffn_gate_inp.weight", [width, 4], router=True, hf=hf + "mlp.gate.weight")
            for name, mapped, shape in (("ffn_gate_exps", "gate_proj", [width, hidden, 4]),
                                        ("ffn_up_exps", "up_proj", [width, hidden, 4]),
                                        ("ffn_down_exps", "down_proj", [hidden, width, 4])):
                add(prefix + name + ".weight", shape, hf=[hf + "mlp.experts.%d.%s.weight" % (e, mapped) for e in range(4)])
        else:
            for name, shape in (("ffn_gate", [width, hidden]), ("ffn_up", [width, hidden]), ("ffn_down", [hidden, width])):
                add(prefix + name + ".weight", shape)
    if not tied:
        add("output.weight", [width, vocab])
    return config, weights, packed


def packed_hash(packed):
    return hashlib.sha256(b"".join(data for _, _, _, data in packed)).hexdigest()


def write_fixture(path, config, weights, packed, moe):
    """Write the raw-block model, keeping norms and the router in F32."""
    names = {name for name, _, _, _ in packed}
    floats = [t for t in weights if t[0] not in names]
    vocab = next(shape[1] for name, _, shape, _ in weights if name == "token_embd.weight")
    tokens = build_byte_vocab() + ["<|endoftext|>"] + ["<|reserved_%d|>" % i for i in range(vocab - VOCAB)]
    return write_model(path, floats, config=config, arch="qwen3moe" if moe else "qwen3", quantized=packed, tokens=tokens)


def device_skip(model):
    """Probe a raw-block fixture through the common device refusal policy."""
    if not os.environ.get("LLMX_DEVICE"):
        return False
    rc, out = common.run_f32_cache(["logits", model, TEXTS[0], "--top", "1"])
    return common.device_lacks_kernel(rc, out)


def run_family(kind, label, make_fixture, family=None):
    if common.f32_cache_skip(kind):
        return common.SKIPPED
    split = common.tensor_width() > 1
    path = os.path.join(os.path.dirname(__file__), "data", "baseline_%s%s.json" % (label.lower(), "_split" if split else ""))
    with open(path, encoding="utf-8") as f:
        goldens = json.load(f)
    assert goldens["transformers_version"] == "4.55.2", label + " reference version changed"
    assert [x["name"] for x in goldens["fixtures"]] == [x[0] for x in VARIANTS], label + " fixture coverage changed"
    worst, comparisons, models = 0.0, 0, []
    with tempfile.TemporaryDirectory(prefix="llmx_%s_hf_" % label.lower()) as directory:
        for (name, tied, moe), golden in zip(VARIANTS, goldens["fixtures"]):
            assert [case["text"] for case in golden["cases"]] == TEXTS, label + " prompt coverage changed"
            assert [case["context"] for case in golden["perplexity"]] == [4, 16], label + " NLL coverage changed"
            config, weights, packed = make_fixture(tied, moe, split=split)
            assert config == golden["config"] and weight_hash(weights) == golden["weights_sha256"], label + " decoded weights changed"
            assert packed_hash(packed) == golden["packed_sha256"], label + " packed weights changed"
            model = write_fixture(os.path.join(directory, name + ".gguf"), config, weights, packed, moe)
            models.append((name, model, golden))
        if device_skip(models[0][1]):
            print("%s: SKIP - selected device has no %s kernel" % (kind, label))
            return common.SKIPPED
        for name, model, golden in models:
            error, count = common.check_hf_fixture(label + " " + name, model, golden["cases"], golden["perplexity"], TEXTS[-1], (1, 2, 3, 5, 16), family=family)
            worst = max(worst, error)
            comparisons += count
    assert comparisons == 150, label + " full-logit coverage changed"
    print("%s: dense tied/untied and MoE vs HF, 150 full-logit cases and 12 NLL checks; max error %.8f  [ok]" % (kind, worst))
    return True
