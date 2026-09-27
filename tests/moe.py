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


def tensors():
    state = 67890
    result = []
    hd = CONFIG["attention.key_length"]
    n_expert, ff = CONFIG["expert_count"], CONFIG["expert_feed_forward_length"]

    # A tensor takes its HF name from tests/f32.py's map, except the router and the experts, whose names only a MoE model has.
    def add(name, shape, norm=False, scale=1.0, hf=None):
        nonlocal state
        values = []
        for _ in range(math.prod(shape)):
            state = (1664525 * state + 1013904223) & 0xffffffff
            value = (((state >> 16) & 1023) - 512) / 8192
            values.append(1.0 + value if norm else value * scale)
        result.append((name, hf or hf_name(name), shape, values))

    add("token_embd.weight", [37, 257])
    add("output_norm.weight", [37], True)
    for layer in range(CONFIG["block_count"]):
        name, hf = "blk.%d." % layer, "model.layers.%d." % layer
        for norm, width in (("attn_norm", 37), ("ffn_norm", 37), ("attn_q_norm", hd), ("attn_k_norm", hd)):
            add(name + norm + ".weight", [width], True)
        for tensor, shape in (("attn_q", [37, 2 * hd]), ("attn_k", [37, hd]), ("attn_v", [37, hd]), ("attn_output", [2 * hd, 37])):
            add(name + tensor + ".weight", shape)
        if layer in DENSE_LAYERS:
            for tensor, shape in (("ffn_gate", [37, 19]), ("ffn_up", [37, 19]), ("ffn_down", [19, 37])):
                add(name + tensor + ".weight", shape)
            continue
        add(name + "ffn_gate_inp.weight", [37, n_expert], scale=ROUTER_SCALE, hf=hf + "mlp.gate.weight")
        # A stacked tensor is expert-major, so expert e's matrix is the e-th of n_expert equal parts.
        for tensor, mapped, shape in (("ffn_gate_exps", "gate_proj", [37, ff, n_expert]),
                                      ("ffn_up_exps", "up_proj", [37, ff, n_expert]),
                                      ("ffn_down_exps", "down_proj", [ff, 37, n_expert])):
            add(name + tensor + ".weight", shape, hf=[hf + "mlp.experts.%d.%s.weight" % (e, mapped) for e in range(n_expert)])
    add("output.weight", [37, 257])
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
# The CPU calibration the bound is fixed from, once (docs/STATUS.md, the half-block order): the gated variant's largest logit error over every case and mode, read by llmx built without Vulkan from 6da7bd57, main's arithmetic with this branch's perplexity and logits output, on the Linux machine's CPU.
Q8_CALIBRATION = {"commit": "6da7bd57", "llmx_sha256": "0a70f69d76690f34aca106b02a4daae871535b972c2701085d71c6903a31b10a", "max_error": 0.063473}
# The agreed bound is max(2e-5, 10 x that error), at most 2e-4; the calibration asks for 0.63473, above the cap, so no bound is set until the developers fix one, and until then the gated variant's errors are printed as the near-tie variant's are.
Q8_BOUND = None


def q8_tensors(router_scale, seed=Q8_SEED):
    """The Q8_0 model's tensors as (name, shape, GGUF type, bytes), in the F32 model's order: each matrix's blocks a scale of 2^-11 times 1 to 1.875 and 32 codes from -127 to 127, and the norms and router F32 as the F32 model's values."""
    state = seed
    result = []
    hd = Q8_CONFIG["attention.key_length"]
    width, ff, n_expert, eff = (Q8_CONFIG[k] for k in ("embedding_length", "feed_forward_length", "expert_count", "expert_feed_forward_length"))

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

    q8("token_embd.weight", [width, 257])
    f32("output_norm.weight", [width], True)
    for layer in range(Q8_CONFIG["block_count"]):
        name = "blk.%d." % layer
        for norm, size in (("attn_norm", width), ("ffn_norm", width), ("attn_q_norm", hd), ("attn_k_norm", hd)):
            f32(name + norm + ".weight", [size], True)
        for tensor, shape in (("attn_q", [width, 2 * hd]), ("attn_k", [width, hd]), ("attn_v", [width, hd]), ("attn_output", [2 * hd, width])):
            q8(name + tensor + ".weight", shape)
        if layer in DENSE_LAYERS:
            for tensor, shape in (("ffn_gate", [width, ff]), ("ffn_up", [width, ff]), ("ffn_down", [ff, width])):
                q8(name + tensor + ".weight", shape)
            continue
        f32(name + "ffn_gate_inp.weight", [width, n_expert], scale=router_scale)
        for tensor, shape in (("ffn_gate_exps", [width, eff, n_expert]), ("ffn_up_exps", [width, eff, n_expert]),
                              ("ffn_down_exps", [eff, width, n_expert])):
            q8(name + tensor + ".weight", shape)
    q8("output.weight", [width, 257])
    return result


def write_q8_model(path, router_scale):
    """Writes the Q8_0 model of one variant to `path` and returns the file's SHA-256."""
    write_model(path, [], config=Q8_CONFIG, arch="qwen3moe", quantized=q8_tensors(router_scale))
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def q8_errors(directory, golden, variant):
    """One variant of the Q8_0 model against its goldens, every case read in every one of Q8_MODES.
    Returns, per mode, the largest logit error, where it lies, and each row's (HF top-two gap, whether llmx's greedy id is HF's), then the mean NLL of the forced ids, llmx's and HF's."""
    doc = next(v for v in golden["variants"] if v["name"] == variant)
    model = os.path.join(directory, "tiny-moe-q8-%s.gguf" % variant)
    assert write_q8_model(model, doc["router_scale"]) == doc["file_sha256"], "Q8_0 MoE fixture %s changed" % variant
    report = {}
    for mode, args in Q8_MODES:
        worst, where, ids, nll, hf_nll = 0.0, None, [], [], []
        for case in doc["cases"]:
            path = os.path.join(directory, "forced.ids")
            with open(path, "w") as f:
                f.write(" ".join(map(str, case["forced_ids"])))
            rc, out = common.run_f32_cache(["logits", model, case["prompt"], "--then-ids", path, "--last", str(Q8_STEPS + 1),
                                            "--top", "257", "--threads", "4"] + args)
            assert rc == 0, "Q8_0 MoE logits failed: %s" % out
            lines = out.splitlines()
            first = len(case["prompt"]) - 1
            assert lines[0] == "tokens: %d" % (first + 1 + Q8_STEPS) and len(lines) == Q8_STEPS + 2, out
            for r, (line, expected) in enumerate(zip(lines[1:], case["rows"])):
                fields = line.split()
                assert int(fields[0]) == first + r, line
                got = {int(i): float(v) for i, v in zip(fields[1::2], fields[2::2])}
                assert set(got) == set(range(257)) and all(math.isfinite(v) for v in got.values()), line
                error = max(abs(got[i] - v) for i, v in enumerate(expected))
                if error > worst:
                    worst, where = error, (case["prompt"], r)
                ids.append((case["gaps"][r], int(fields[1]) == case["top_ids"][r]))
                if r < Q8_STEPS:
                    target = case["forced_ids"][r]
                    nll.append(math.log(sum(math.exp(v) for v in got.values())) - got[target])
                    hf_nll.append(math.log(sum(math.exp(v) for v in expected)) - expected[target])
        report[mode] = (worst, where, ids, sum(nll) / len(nll), sum(hf_nll) / len(hf_nll))
    return report


def check_q8(directory, golden):
    """The gated variant: every logit within Q8_BOUND of HF, and HF's greedy id wherever HF's top two lie more than twice that apart; the near-tie variant is printed and not held to anything."""
    assert golden["config"] == Q8_CONFIG and golden["dense_layers"] == list(DENSE_LAYERS), "Q8_0 MoE fixture config changed"
    assert golden["seed"] == Q8_SEED and golden["steps"] == Q8_STEPS, "Q8_0 MoE fixture seed or steps changed"
    assert [(v["name"], v["router_scale"], [c["prompt"] for c in v["cases"]]) for v in golden["variants"]] == [(n, s, list(Q8_PROMPTS)) for n, s in Q8_VARIANTS]
    # The bound is the calibration's rule or unset, never another number.
    wanted = max(2e-5, 10 * Q8_CALIBRATION["max_error"])
    assert Q8_BOUND is None if wanted > 2e-4 else Q8_BOUND == wanted, "Q8_BOUND is not the calibration's"
    if Q8_BOUND is None:
        print("moe: Q8_0 bound unset: the CPU calibration at %s gives %.6f, which asks for %.6f, above the cap of 2e-4" % (
            Q8_CALIBRATION["commit"], Q8_CALIBRATION["max_error"], wanted))
    for variant, _ in Q8_VARIANTS:
        for mode, (worst, where, ids, nll, hf_nll) in q8_errors(directory, golden, variant).items():
            line = ("moe: Q8_0 %s, %s: max error %.8f (%r, row %d), HF's greedy id at %d of %d rows, forced NLL %.6f against HF's %.6f"
                    % (variant, mode, worst, where[0], where[1], sum(same for _, same in ids), len(ids), nll, hf_nll))
            if variant == "gated" and Q8_BOUND is not None:
                held = [same for gap, same in ids if gap > 2 * Q8_BOUND]
                assert worst <= Q8_BOUND and all(held), line
                line += ", within %.0e with HF's id at all %d rows HF parts by more than twice that" % (Q8_BOUND, len(held))
            print(line + ("  [ok]" if variant == "gated" and Q8_BOUND is not None else "  [reported, no bound]"))


def run():
    if common.f32_cache_skip("moe"):
        return True
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
    print("moe: all 257 logits vs HF over routed and dense layers, batch widths, threads, expert placement and PPL; max error %.8f  [ok]" % worst)
    with open(os.path.join(os.path.dirname(__file__), "data", "baseline_moe_q8.json"), encoding="utf-8") as f:
        golden = json.load(f)
    with tempfile.TemporaryDirectory(prefix="llmx_moe_q8_") as directory:
        check_q8(directory, golden)
    return True


if __name__ == "__main__":
    run()
