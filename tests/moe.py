import json
import math
import os
import tempfile

from common import run_f32_cache as cli
from f32 import TEXTS, weight_hash, write_model


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

    def add(name, hf_name, shape, norm=False, scale=1.0):
        nonlocal state
        values = []
        for _ in range(math.prod(shape)):
            state = (1664525 * state + 1013904223) & 0xffffffff
            value = (((state >> 16) & 1023) - 512) / 8192
            values.append(1.0 + value if norm else value * scale)
        result.append((name, hf_name, shape, values))

    add("token_embd.weight", "model.embed_tokens.weight", [37, 257])
    add("output_norm.weight", "model.norm.weight", [37], True)
    for layer in range(CONFIG["block_count"]):
        name, hf = "blk.%d." % layer, "model.layers.%d." % layer
        for norm, mapped, width in (("attn_norm", "input_layernorm", 37),
                                    ("ffn_norm", "post_attention_layernorm", 37),
                                    ("attn_q_norm", "self_attn.q_norm", hd),
                                    ("attn_k_norm", "self_attn.k_norm", hd)):
            add(name + norm + ".weight", hf + mapped + ".weight", [width], True)
        for tensor, mapped, shape in (("attn_q", "self_attn.q_proj", [37, 2 * hd]),
                                      ("attn_k", "self_attn.k_proj", [37, hd]),
                                      ("attn_v", "self_attn.v_proj", [37, hd]),
                                      ("attn_output", "self_attn.o_proj", [2 * hd, 37])):
            add(name + tensor + ".weight", hf + mapped + ".weight", shape)
        if layer in DENSE_LAYERS:
            for tensor, mapped, shape in (("ffn_gate", "mlp.gate_proj", [37, 19]),
                                          ("ffn_up", "mlp.up_proj", [37, 19]),
                                          ("ffn_down", "mlp.down_proj", [19, 37])):
                add(name + tensor + ".weight", hf + mapped + ".weight", shape)
            continue
        add(name + "ffn_gate_inp.weight", hf + "mlp.gate.weight", [37, n_expert], scale=ROUTER_SCALE)
        # A stacked tensor is expert-major, so expert e's matrix is the e-th of n_expert equal parts.
        for tensor, mapped, shape in (("ffn_gate_exps", "gate_proj", [37, ff, n_expert]),
                                      ("ffn_up_exps", "up_proj", [37, ff, n_expert]),
                                      ("ffn_down_exps", "down_proj", [ff, 37, n_expert])):
            add(name + tensor + ".weight",
                [hf + "mlp.experts.%d.%s.weight" % (e, mapped) for e in range(n_expert)], shape)
    add("output.weight", "lm_head.weight", [37, 257])
    return result


def run():
    # Like f32.py, this compares exact f32 arithmetic against the HF fixture, which an f16 cache would round.
    if os.environ.get("LLMX_CACHE_TYPE", "f32") != "f32":
        print("moe: SKIP - the exact MoE gate needs f32 caches (LLMX_CACHE_TYPE=%s)" % os.environ["LLMX_CACHE_TYPE"])
        return True
    with open(os.path.join(os.path.dirname(__file__), "data", "baseline_moe.json"), encoding="utf-8") as f:
        golden = json.load(f)
    assert golden["config"] == CONFIG and golden["dense_layers"] == list(DENSE_LAYERS), "MoE fixture config changed"
    weights = tensors()
    assert weight_hash(weights) == golden["weights_sha256"], "MoE fixture weights changed"
    worst = 0.0
    # On a device the experts also run on the CPU beside it: the first routed layer's alone, and all of them.
    # Streamed, the host's layers run on the device with their experts copied there: every prompt (from 1, which a generated token never reaches), and from 4 only the longer ones.
    device = os.environ.get("LLMX_DEVICE", "cpu")
    placements = [[]] + ([["--n-cpu-moe", "1", "--moe-stream-from", "0"], ["--cpu-moe", "--moe-stream-from", "0"],
                          ["--cpu-moe", "--moe-stream-from", "1"], ["--n-cpu-moe", "1", "--moe-stream-from", "4"]]
                         if device != "cpu" else [])
    with tempfile.TemporaryDirectory(prefix="llmx_moe_") as directory:
        model = write_model(os.path.join(directory, "tiny-moe.gguf"), weights, config=CONFIG, arch="qwen3moe")
        for threads in (1, 4):
            for ubatch in (1, 2, 3, 5, 16):
                for case, placement in ((c, pl) for c in golden["cases"] for pl in placements if threads == 4 or not pl):
                    rc, out = cli(["logits", model, case["text"], "--top", "257",
                                   "--threads", str(threads), "--ubatch", str(ubatch)] + placement)
                    assert rc == 0, "MoE logits failed: " + out
                    got = {int(p[0]): float(p[1]) for line in out.splitlines()
                           if len(p := line.split()) == 2 and p[0].isdigit()}
                    assert set(got) == set(range(257)), "missing MoE logits"
                    error = max(abs(got[i] - expected) for i, expected in enumerate(case["logits"]))
                    assert math.isfinite(error) and error < 2e-5, "MoE/HF logit error: %.8f" % error
                    worst = max(worst, error)
            for case in golden["perplexity"]:
                rc, out = cli(["perplexity", model, TEXTS[-1], "--threads", str(threads),
                               "-c", str(case["context"])])
                assert rc == 0, "MoE PPL failed: " + out
                fields = dict(line.split(":", 1) for line in out.splitlines())
                error = abs(float(fields["mean NLL"]) - case["mean_nll"])
                assert math.isfinite(error) and error < 1e-5, "MoE/HF NLL error: %.8f" % error
    print("moe: all 257 logits vs HF over routed and dense layers, batch widths, threads, expert placement and PPL; max error %.8f  [ok]" % worst)
    return True


if __name__ == "__main__":
    run()
