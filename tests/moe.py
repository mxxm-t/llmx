import hashlib
import json
import math
import os
import struct
import tempfile

import common
from f32 import TEXTS, hf_name, weight_hash, write_model
from spec_decode import Q8_0


# A tiny qwen3moe model with deterministic weights against HF Qwen3MoeForCausalLM (tools/gen_baseline.py moe): all 257 logits and windowed NLL, through batch widths that route one row and many rows to an expert.
# Layers 0 and 2 are routed and layer 1 is dense, so a file mixing the two loads and computes.
CONFIG = {"block_count": 3, "embedding_length": 37, "feed_forward_length": 19,
          "attention.head_count": 2, "attention.head_count_kv": 1,
          "attention.key_length": 42, "context_length": 16,
          "expert_count": 8, "expert_used_count": 3, "expert_feed_forward_length": 11}
DENSE_LAYERS = (1,)
# The router's weights are scaled up so the probabilities spread and no token sits near a tie between its k-th and next expert.
ROUTER_SCALE = 16.0


def tensors(seed=67890, config=None, vocab=257):
    """The model's tensors as (GGUF name, HF name or names, shape, values), of CONFIG's shapes or `config`'s over `vocab` tokens."""
    c = config or CONFIG
    state = seed
    result = []
    width, dense_ff, hd = c["embedding_length"], c["feed_forward_length"], c["attention.key_length"]
    q, kv = c["attention.head_count"] * hd, c["attention.head_count_kv"] * hd
    n_expert, ff = c["expert_count"], c["expert_feed_forward_length"]

    # A tensor takes its HF name from tests/f32.py's map, except the router and the experts, whose names only a MoE model has.
    def add(name, shape, norm=False, scale=1.0, hf=None):
        nonlocal state
        values = []
        for _ in range(math.prod(shape)):
            state = (1664525 * state + 1013904223) & 0xffffffff
            value = (((state >> 16) & 1023) - 512) / 8192
            values.append(1.0 + value if norm else value * scale)
        result.append((name, hf or hf_name(name), shape, values))

    add("token_embd.weight", [width, vocab])
    add("output_norm.weight", [width], True)
    for layer in range(c["block_count"]):
        name, hf = "blk.%d." % layer, "model.layers.%d." % layer
        for norm, size in (("attn_norm", width), ("ffn_norm", width), ("attn_q_norm", hd), ("attn_k_norm", hd)):
            add(name + norm + ".weight", [size], True)
        for tensor, shape in (("attn_q", [width, q]), ("attn_k", [width, kv]), ("attn_v", [width, kv]), ("attn_output", [q, width])):
            add(name + tensor + ".weight", shape)
        if layer in DENSE_LAYERS:
            for tensor, shape in (("ffn_gate", [width, dense_ff]), ("ffn_up", [width, dense_ff]), ("ffn_down", [dense_ff, width])):
                add(name + tensor + ".weight", shape)
            continue
        add(name + "ffn_gate_inp.weight", [width, n_expert], scale=ROUTER_SCALE, hf=hf + "mlp.gate.weight")
        # A stacked tensor is expert-major, so expert e's matrix is the e-th of n_expert equal parts.
        for tensor, mapped, shape in (("ffn_gate_exps", "gate_proj", [width, ff, n_expert]),
                                      ("ffn_up_exps", "up_proj", [width, ff, n_expert]),
                                      ("ffn_down_exps", "down_proj", [ff, width, n_expert])):
            add(name + tensor + ".weight", shape, hf=[hf + "mlp.experts.%d.%s.weight" % (e, mapped) for e in range(n_expert)])
    add("output.weight", [width, vocab])
    return result


# The tiny Q8_0 model: its matrices Q8_0 and every row width, the experts' included, a multiple of 64, so each row holds an even count of 32-value blocks, which a device's Q8_0 decode kernel reads in the order it keeps for such rows (docs/VULKAN.md).
# Its goldens are HF holding the file's own weights as tests/spec_decode.py decodes them (tools/gen_baseline.py moe-q8): the rows from each prompt's last position through Q8_STEPS forced ids, HF's own greedy continuation.
Q8_CONFIG = {"block_count": 3, "embedding_length": 128, "feed_forward_length": 128,
             "attention.head_count": 2, "attention.head_count_kv": 1,
             "attention.key_length": 64, "context_length": 64,
             "expert_count": 8, "expert_used_count": 3, "expert_feed_forward_length": 64}
# The gated variant's router is scaled up as the F32 model's is, so no routing lies within Q8_MIN_ROUTING_GAP router logits of a tie between a token's k-th and next expert; the near-tie variant's is scaled down so every routing lies near one, and it is reported for sensitivity and never held to the bound.
Q8_VARIANTS = (("gated", 16.0), ("near-tie", 1 / 16))
Q8_MIN_ROUTING_GAP = 0.1
Q8_SEED = 24792
# Each prompt and its forced ids stay under the MI50's 32-row tile crossover in one pass, so a batched pass reads them through the decode kernel's wide builds.
Q8_PROMPTS = ("a", "abcdefg", "abcdefghijklm")
Q8_STEPS = 16
# How each case is read: batched in one pass and three rows a pass, the prompt path, and every token through a decode step.
Q8_MODES = (("batched", []), ("batched by 3", ["--ubatch", "3"]), ("decode", ["--per-token"]))
Q8_FILES = {"gated": "d547bb5bf6f06f156998e459eb93ceb688390584f33b0ae24c0aabe1874fc751",
            "near-tie": "93491ced5b64b1948540f8b3e1ce9d71f48a378213353e075e22fdc955589285"}


def q8_tensors(router_scale, seed=Q8_SEED, config=Q8_CONFIG, dense=DENSE_LAYERS, tied=False, vocab=257):
    """The Q8_0 model's tensors as (name, shape, GGUF type, bytes), in the F32 model's order: each matrix's blocks a scale of 2^-11 times 1 to 1.875 and 32 codes from -127 to 127, and the norms and router F32 as the F32 model's values.
    `config`, `dense` (the layers without experts), `tied` and `vocab` give another model of the same draws, as tests/int8.py's fixtures are; a model whose every layer is dense has no router or experts."""
    state = seed
    result = []
    hd = config["attention.key_length"]
    width, ff = config["embedding_length"], config["feed_forward_length"]
    n_expert, eff = config.get("expert_count", 0), config.get("expert_feed_forward_length", 0)

    def draw():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xffffffff
        return (state >> 16) & 1023

    def f32(name, shape, norm=False, scale=1.0):
        values = [1.0 + (draw() - 512) / 8192 if norm else (draw() - 512) / 8192 * scale for _ in range(math.prod(shape))]
        result.append((name, shape, 0, struct.pack("<%df" % len(values), *values)))

    def q8(name, shape):
        data = bytearray()
        for _ in range(math.prod(shape) // 32):
            data += struct.pack("<e", 2.0 ** -11 * (1 + draw() % 8 / 8))
            data += struct.pack("<32b", *(draw() % 255 - 127 for _ in range(32)))
        result.append((name, shape, Q8_0, bytes(data)))

    q8("token_embd.weight", [width, vocab])
    f32("output_norm.weight", [width], True)
    q, kv = config["attention.head_count"] * hd, config["attention.head_count_kv"] * hd
    for layer in range(config["block_count"]):
        name = "blk.%d." % layer
        for norm, size in (("attn_norm", width), ("ffn_norm", width), ("attn_q_norm", hd), ("attn_k_norm", hd)):
            f32(name + norm + ".weight", [size], True)
        for tensor, shape in (("attn_q", [width, q]), ("attn_k", [width, kv]), ("attn_v", [width, kv]), ("attn_output", [q, width])):
            q8(name + tensor + ".weight", shape)
        if layer in dense:
            for tensor, shape in (("ffn_gate", [width, ff]), ("ffn_up", [width, ff]), ("ffn_down", [ff, width])):
                q8(name + tensor + ".weight", shape)
            continue
        f32(name + "ffn_gate_inp.weight", [width, n_expert], scale=router_scale)
        for tensor, shape in (("ffn_gate_exps", [width, eff, n_expert]), ("ffn_up_exps", [width, eff, n_expert]),
                              ("ffn_down_exps", [eff, width, n_expert])):
            q8(name + tensor + ".weight", shape)
    if not tied:
        q8("output.weight", [width, vocab])
    return result


def write_q8_model(path, router_scale):
    """Writes the Q8_0 model of one variant to `path` and returns the file's SHA-256."""
    write_model(path, [], config=Q8_CONFIG, arch="qwen3moe", quantized=q8_tensors(router_scale))
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def q8_errors(directory, golden, variant):
    """One variant of the Q8_0 model against its goldens, every case read in every one of Q8_MODES.
    Returns, per mode, one entry per prompt: its largest logit error and the row it lies in, each row's (HF top-two gap, whether llmx's greedy id is HF's), and the mean NLL of its forced ids, llmx's and HF's."""
    doc = next(v for v in golden["variants"] if v["name"] == variant)
    model = os.path.join(directory, "tiny-moe-q8-%s.gguf" % variant)
    assert write_q8_model(model, doc["router_scale"]) == doc["file_sha256"] == Q8_FILES[variant], "Q8_0 MoE fixture %s changed" % variant
    report = {}
    for mode, args in Q8_MODES:
        report[mode] = []
        for case in doc["cases"]:
            path = os.path.join(directory, "forced.ids")
            with open(path, "w") as f:
                f.write(" ".join(map(str, case["forced_ids"])))
            out, bounds = common.run_hf(["logits", model, case["prompt"], "--then-ids", path, "--last", str(Q8_STEPS + 1),
                                         "--top", "257", "--threads", "4"] + args)
            lines = out.splitlines()
            first = len(case["prompt"]) - 1
            assert lines[0] == "tokens: %d" % (first + 1 + Q8_STEPS) and len(lines) == Q8_STEPS + 2, out
            worst, row, ids, nll, hf_nll = 0.0, 0, [], [], []
            for r, (line, expected) in enumerate(zip(lines[1:], case["rows"])):
                fields = line.split()
                assert int(fields[0]) == first + r, line
                got = {int(i): float(v) for i, v in zip(fields[1::2], fields[2::2])}
                assert set(got) == set(range(257)) and all(math.isfinite(v) for v in got.values()), line
                error = max(abs(got[i] - v) for i, v in enumerate(expected))
                if error > worst:
                    worst, row = error, r
                ids.append((case["gaps"][r], int(fields[1]) == case["top_ids"][r]))
                if r < Q8_STEPS:
                    target = case["forced_ids"][r]
                    nll.append(math.log(sum(math.exp(v) for v in got.values())) - got[target])
                    hf_nll.append(math.log(sum(math.exp(v) for v in expected)) - expected[target])
            report[mode].append({"prompt": case["prompt"], "worst": worst, "row": row, "ids": ids,
                                 "nll": sum(nll) / len(nll), "hf_nll": sum(hf_nll) / len(hf_nll), "bounds": bounds})
    return report


def check_q8(directory, golden):
    """The gated variant on every path: logits and each prompt's forced NLL within the actual execution dtype's frozen bounds, and HF's greedy id wherever its top two part by more than twice that logit bound.
    The near-tie variant is printed and held to nothing."""
    assert golden["config"] == Q8_CONFIG and golden["dense_layers"] == list(DENSE_LAYERS), "Q8_0 MoE fixture config changed"
    assert golden["seed"] == Q8_SEED and golden["steps"] == Q8_STEPS, "Q8_0 MoE fixture seed or steps changed"
    assert [(v["name"], v["router_scale"], [c["prompt"] for c in v["cases"]]) for v in golden["variants"]] == [(n, s, list(Q8_PROMPTS)) for n, s in Q8_VARIANTS]
    for variant, _ in Q8_VARIANTS:
        for mode, prompts in q8_errors(directory, golden, variant).items():
            gated = variant == "gated"
            parts = []
            for p in prompts:
                bound = p["bounds"]
                held = [same for gap, same in p["ids"] if gap > 2 * bound["logit"]]
                parts.append("%r max %.6f (row %d), NLL error %+.6f, HF's id %d of %d rows, %d of %d beyond 2E" % (
                    p["prompt"], p["worst"], p["row"], p["nll"] - p["hf_nll"], sum(s for _, s in p["ids"]), len(p["ids"]), sum(held), len(held)))
                if gated:
                    assert p["worst"] <= bound["logit"] and abs(p["nll"] - p["hf_nll"]) <= bound["nll"] and held and all(held), \
                        "Q8_0 %s, %s: %s (bounds %s)" % (variant, mode, parts[-1], bound)
            aggregate = sum(p["nll"] - p["hf_nll"] for p in prompts) / len(prompts)
            print("moe: Q8_0 %s, %s: %s; aggregate NLL error %+.6f%s" % (
                variant, mode, "; ".join(parts), aggregate, "  [ok]" if gated else "  [sensitivity, no bound]"))


def run():
    if common.f32_cache_skip("moe") or common.tensor_width_skip("moe"):
        return common.SKIPPED
    with open(os.path.join(os.path.dirname(__file__), "data", "baseline_moe.json"), encoding="utf-8") as f:
        golden = json.load(f)
    assert golden["config"] == CONFIG and golden["dense_layers"] == list(DENSE_LAYERS), "MoE fixture config changed"
    weights = tensors()
    assert weight_hash(weights) == golden["weights_sha256"], "MoE fixture weights changed"
    # On a device the experts also run on the CPU beside it: the first routed layer's alone, and all of them.
    # Experts on the CPU are a placement of one device; with several listed the split places whole layers instead.
    # Streamed, the host's layers run on the device with their experts copied there: every prompt of two tokens or more (from 1, which streams what 2 does, since neither a generated token nor a one-token prompt streams), and from 4 only the longer ones.
    device = os.environ.get("LLMX_DEVICE", "cpu")
    placements = [[]] + ([["--n-cpu-moe", "1", "--moe-stream-from", "0"], ["--cpu-moe", "--moe-stream-from", "0"],
                          ["--cpu-moe", "--moe-stream-from", "1"], ["--n-cpu-moe", "1", "--moe-stream-from", "4"]]
                         if device != "cpu" and "," not in device else [])
    with tempfile.TemporaryDirectory(prefix="llmx_moe_") as directory:
        model = write_model(os.path.join(directory, "tiny-moe.gguf"), weights, config=CONFIG, arch="qwen3moe")
        worst, _ = common.check_hf_fixture("MoE", model, golden["cases"], golden["perplexity"], TEXTS[-1],
                                           (1, 2, 3, 5, 16), placements)
        common.check_drafts("MoE", model)
    print("moe: all 257 logits vs HF over routed and dense layers, batch widths, threads, expert placement and PPL; max error %.8f  [ok]" % worst)
    with open(os.path.join(os.path.dirname(__file__), "data", "baseline_moe_q8.json"), encoding="utf-8") as f:
        golden = json.load(f)
    with tempfile.TemporaryDirectory(prefix="llmx_moe_q8_") as directory:
        check_q8(directory, golden)
    return True


if __name__ == "__main__":
    run()
