import json
import math
import os
import re
import struct
import tempfile

import common
from common import run_f32_cache as cli
from f32 import TEXTS, VOCAB, check_logits_input, weight_hash, write_model


# Tiny qwen35 models with deterministic weights against HF Qwen3_5ForCausalLM (tools/gen_baseline.py qwen35-tiny): all 257 logits and windowed NLL over ubatches, threads, both ways of scoring and greedy decode after a prefill.
# The weights are made as HF holds them, and the writer applies the converter's transforms to write the GGUF, as docs/QWEN35.md, GGUF conventions, gives them.
# Four layers, linear attention then full attention twice, so layer 3 holds the second KV cache and layer 2 the second recurrent state, which a cache indexed by layer number would miss.
# A V head is 10 wide against a K head's 12, so the state is not square, and the rotary width is 8 of 40, with a base of 100 so every rotated pair turns within the context.
CONFIG = {"embedding_length": 37, "feed_forward_length": 19, "context_length": 16,
          "attention.head_count": 4, "attention.head_count_kv": 2,
          "attention.key_length": 40, "attention.value_length": 40,
          "rope.dimension_count": 8, "rope.dimension_sections": [2, 1, 1, 0], "rope.freq_base": 100.0,
          "attention.layer_norm_rms_epsilon": 1e-6,
          "ssm.conv_kernel": 4, "ssm.state_size": 12, "ssm.group_count": 2, "full_attention_interval": 2}
LAYERS = 4
V_HEAD = 10
# qwen35moe's feed-forward block in place of the dense one: 4 experts of 5, 2 a token, and a shared expert of 7.
MOE = {"expert_count": 4, "expert_used_count": 2, "expert_feed_forward_length": 5, "expert_shared_feed_forward_length": 7}
# Hv = Hk with a tied head, and Hv = 3 Hk with its own head, alone and with one MTP block, whose file must give the logits of the file without it, and a qwen35moe model of Hv = 3 Hk.
FIXTURES = [
    {"name": "hv1", "v_heads": 2, "tied": True, "mtp": False},
    {"name": "hv3", "v_heads": 6, "tied": False, "mtp": False},
    {"name": "hv3-mtp", "v_heads": 6, "tied": False, "mtp": True},
    {"name": "moe", "v_heads": 6, "tied": False, "mtp": False, "moe": True},
]
# The end-of-text token, which the greedy goldens and the CLI both leave out of every draw.
EOS = VOCAB - 1
# Physical batches that cut the texts into passes; under one of them the last row of every text of three or more tokens runs alone after a batched pass, as a decode step after a prefill does.
UBATCHES = (1, 2, 3, 5, 16)
# llmx's refusals of a qwen35 file: of the architecture, where it does not run it, and at load on a device whose backend lacks the linear attention's ops.
# Either skips this component, and the real-model checks, rather than failing or passing them.
REFUSALS = ("unsupported metadata general.architecture", "which the backend of its device does not implement")


def refusal(rc, out):
    """The refusal a failed llmx command gave, if it is one of REFUSALS, else None."""
    return next((text for text in REFUSALS if rc != 0 and text in out), None)


def full_attention(layer):
    """Whether decoder layer `layer` is a full-attention layer, which every full_attention_interval-th layer is."""
    return (layer + 1) % CONFIG["full_attention_interval"] == 0


def fixture_config(fixture):
    """CONFIG with the keys a fixture of other shapes overrides (its "config"), and its V head width and vocabulary, which default to V_HEAD and VOCAB."""
    return dict(CONFIG, **fixture.get("config", {})), fixture.get("v_head", V_HEAD), fixture.get("vocab", VOCAB)


def moe_config(fixture):
    """A qwen35moe fixture's experts: MOE, or the keys its "moe" gives in its place."""
    return fixture["moe"] if isinstance(fixture.get("moe"), dict) else MOE


def gguf_config(fixture):
    """The qwen35 or qwen35moe metadata the converter writes for `fixture`, whose MTP block, when it has one, is one more block."""
    config, v_head, _ = fixture_config(fixture)
    if fixture.get("moe"):
        del config["feed_forward_length"]
        config.update(moe_config(fixture))
    config["block_count"] = LAYERS + fixture["mtp"]
    config["ssm.time_step_rank"] = fixture["v_heads"]
    config["ssm.inner_size"] = fixture["v_heads"] * v_head
    if fixture["mtp"]:
        config["nextn_predict_layers"] = 1
    return config


def raw_weights(fixture):
    """The fixture's parameters as HF holds them, [(HF name, HF shape, values)], every value exact in float32.
    One fixed sequence makes them, so the file with an MTP block holds the weights of the one without it, then the block's."""
    state = 24680
    result = []
    config, V_HEAD, VOCAB = fixture_config(fixture)
    width, ff, hd = config["embedding_length"], config["feed_forward_length"], config["attention.key_length"]
    heads, kv_heads = config["attention.head_count"], config["attention.head_count_kv"]
    k_heads, k_width, v_heads = config["ssm.group_count"], config["ssm.state_size"], fixture["v_heads"]
    conv = 2 * k_heads * k_width + v_heads * V_HEAD

    # A value is a multiple of 1/8192 in [-1/16, 1/16), times `scale`, plus `offset`.
    def add(name, shape, scale=1.0, offset=0.0):
        nonlocal state
        values = []
        for _ in range(math.prod(shape)):
            state = (1664525 * state + 1013904223) & 0xffffffff
            values.append(offset + (((state >> 16) & 1023) - 512) / 8192 * scale)
        result.append((name, shape, values))

    # HF's norms multiply by 1 + w, so their weights sit near 0, except the gated norm's, which it multiplies by w itself.
    def layer(prefix, full):
        add(prefix + "input_layernorm.weight", [width])
        if full:
            add(prefix + "self_attn.q_proj.weight", [heads * 2 * hd, width])
            add(prefix + "self_attn.k_proj.weight", [kv_heads * hd, width])
            add(prefix + "self_attn.v_proj.weight", [kv_heads * hd, width])
            add(prefix + "self_attn.q_norm.weight", [hd])
            add(prefix + "self_attn.k_norm.weight", [hd])
            add(prefix + "self_attn.o_proj.weight", [width, heads * hd])
        else:
            # The conv weights are scaled up so the conv's output, and with it v, is not small against the norms' epsilon.
            # beta and alpha are scaled up so the heads' gates spread, and A_log and dt_bias span [-2, 2), which keeps every decay factor far above 2^-126.
            add(prefix + "linear_attn.in_proj_qkv.weight", [conv, width])
            add(prefix + "linear_attn.in_proj_z.weight", [v_heads * V_HEAD, width])
            add(prefix + "linear_attn.in_proj_b.weight", [v_heads, width], scale=16.0)
            add(prefix + "linear_attn.in_proj_a.weight", [v_heads, width], scale=8.0)
            add(prefix + "linear_attn.conv1d.weight", [conv, 1, config["ssm.conv_kernel"]], scale=8.0)
            add(prefix + "linear_attn.dt_bias", [v_heads], scale=32.0)
            add(prefix + "linear_attn.A_log", [v_heads], scale=32.0)
            add(prefix + "linear_attn.norm.weight", [V_HEAD], offset=1.0)
            add(prefix + "linear_attn.out_proj.weight", [width, v_heads * V_HEAD])
        add(prefix + "post_attention_layernorm.weight", [width])
        if fixture.get("moe"):
            # The router is scaled up so every token's top two experts stand clear of the third, which tools/gen_baseline.py checks.
            moe = moe_config(fixture)
            experts, fe, fs = moe["expert_count"], moe["expert_feed_forward_length"], moe["expert_shared_feed_forward_length"]
            add(prefix + "mlp.gate.weight", [experts, width], scale=32.0)
            add(prefix + "mlp.experts.gate_up_proj", [experts, 2 * fe, width])
            add(prefix + "mlp.experts.down_proj", [experts, width, fe])
            add(prefix + "mlp.shared_expert.gate_proj.weight", [fs, width])
            add(prefix + "mlp.shared_expert.up_proj.weight", [fs, width])
            add(prefix + "mlp.shared_expert.down_proj.weight", [width, fs])
            add(prefix + "mlp.shared_expert_gate.weight", [1, width], scale=16.0)
            return
        add(prefix + "mlp.gate_proj.weight", [ff, width])
        add(prefix + "mlp.up_proj.weight", [ff, width])
        add(prefix + "mlp.down_proj.weight", [width, ff])

    add("model.embed_tokens.weight", [VOCAB, width])
    for index in range(LAYERS):
        layer("model.layers.%d." % index, full_attention(index))
    add("model.norm.weight", [width])
    if not fixture["tied"]:
        add("lm_head.weight", [VOCAB, width])
    if fixture["mtp"]:
        add("mtp.pre_fc_norm_embedding.weight", [width])
        add("mtp.pre_fc_norm_hidden.weight", [width])
        add("mtp.fc.weight", [width, 2 * width])
        layer("mtp.layers.0.", True)
        add("mtp.norm.weight", [width])
    return result


def tiled_order(k_heads, v_heads):
    """The HF V head each GGUF V head holds in the converter's tiled order, the one owner of that order for the writer and the references.
    HF groups the V heads by the K head they read, and GGUF V head j = s Hk + h holds HF's V head h r + s, with Hk = `k_heads` and r = `v_heads` / Hk, so it reads K head j mod Hk; with Hv = Hk nothing moves."""
    r = v_heads // k_heads
    return [(j % k_heads) * r + j // k_heads for j in range(v_heads)]


def tiled(values, k_heads, v_heads, block):
    """`values`, `v_heads` blocks of `block` values in HF's order, in the converter's tiled order."""
    return [v for source in tiled_order(k_heads, v_heads) for v in values[source * block:(source + 1) * block]]


def float32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


# The GGUF name and transform the converter gives the parameter of a decoder layer or of the MTP layer, by its name within the layer.
# "norm" stores 1 + w, "a" stores -exp(A_log) in the tiled order, "heads" tiles whole rows or entries by V head, "channels" tiles the v rows after the q and k ones, "conv" does so for the conv's channels and drops its middle axis, and "columns" tiles each row's input columns.
BLOCK_TENSORS = {
    "input_layernorm.weight": ("attn_norm.weight", "norm"),
    "post_attention_layernorm.weight": ("post_attention_norm.weight", "norm"),
    "linear_attn.in_proj_qkv.weight": ("attn_qkv.weight", "channels"),
    "linear_attn.in_proj_z.weight": ("attn_gate.weight", "heads"),
    "linear_attn.in_proj_b.weight": ("ssm_beta.weight", "heads"),
    "linear_attn.in_proj_a.weight": ("ssm_alpha.weight", "heads"),
    "linear_attn.conv1d.weight": ("ssm_conv1d.weight", "conv"),
    "linear_attn.dt_bias": ("ssm_dt.bias", "heads"),
    "linear_attn.A_log": ("ssm_a", "a"),
    "linear_attn.norm.weight": ("ssm_norm.weight", None),
    "linear_attn.out_proj.weight": ("ssm_out.weight", "columns"),
    "self_attn.q_proj.weight": ("attn_q.weight", None),
    "self_attn.k_proj.weight": ("attn_k.weight", None),
    "self_attn.v_proj.weight": ("attn_v.weight", None),
    "self_attn.o_proj.weight": ("attn_output.weight", None),
    "self_attn.q_norm.weight": ("attn_q_norm.weight", "norm"),
    "self_attn.k_norm.weight": ("attn_k_norm.weight", "norm"),
    "mlp.gate_proj.weight": ("ffn_gate.weight", None),
    "mlp.up_proj.weight": ("ffn_up.weight", None),
    "mlp.down_proj.weight": ("ffn_down.weight", None),
    "mlp.gate.weight": ("ffn_gate_inp.weight", None),
    "mlp.experts.gate_up_proj": (("ffn_gate_exps.weight", "ffn_up_exps.weight"), "halves"),
    "mlp.experts.down_proj": ("ffn_down_exps.weight", None),
    "mlp.shared_expert.gate_proj.weight": ("ffn_gate_shexp.weight", None),
    "mlp.shared_expert.up_proj.weight": ("ffn_up_shexp.weight", None),
    "mlp.shared_expert.down_proj.weight": ("ffn_down_shexp.weight", None),
    "mlp.shared_expert_gate.weight": ("ffn_gate_inp_shexp.weight", "vector"),
}
# The parameters outside the layers; the MTP block's own ones are stored under its block, the one after the decoder layers.
OTHER_TENSORS = {
    "model.embed_tokens.weight": ("token_embd.weight", None),
    "model.norm.weight": ("output_norm.weight", "norm"),
    "lm_head.weight": ("output.weight", None),
    "mtp.fc.weight": ("blk.%d.nextn.eh_proj.weight" % LAYERS, None),
    "mtp.pre_fc_norm_embedding.weight": ("blk.%d.nextn.enorm.weight" % LAYERS, "norm"),
    "mtp.pre_fc_norm_hidden.weight": ("blk.%d.nextn.hnorm.weight" % LAYERS, "norm"),
    "mtp.norm.weight": ("blk.%d.nextn.shared_head_norm.weight" % LAYERS, "norm"),
}


def gguf_tensors(fixture, raw):
    """The GGUF tensors the converter writes from `raw`, as tests/f32.py's writer takes them: (GGUF name, HF name, GGUF shape, values).
    A GGUF shape lists HF's dimensions fastest first, and the conv kernel drops HF's middle axis, so tap 3 is the one that multiplies the current token.
    HF's fused experts' gate_up_proj splits into the gate and up stacks, each expert's first half of rows the gate, and the shared expert's gate of [1, E] is written as a vector of E."""
    config = fixture_config(fixture)[0]
    k_heads, v_heads = config["ssm.group_count"], fixture["v_heads"]
    qk = 2 * k_heads * config["ssm.state_size"]
    out = []
    for name, shape, values in raw:
        match = re.fullmatch(r"(?:model\.layers\.(\d+)|mtp\.layers\.0)\.(.+)", name)
        if match:
            target, transform = BLOCK_TENSORS[match[2]]
            block = "blk.%d." % (int(match[1]) if match[1] is not None else LAYERS)
            target = tuple(block + t for t in target) if transform == "halves" else block + target
        else:
            target, transform = OTHER_TENSORS[name]
        if transform == "halves":
            experts, rows, width = shape
            half = rows // 2 * width
            for part, t in enumerate(target):
                stack = [v for e in range(experts) for v in values[e * rows * width + part * half:e * rows * width + (part + 1) * half]]
                out.append((t, name, [width, rows // 2, experts], stack))
            continue
        if transform == "vector":
            out.append((target, name, [shape[1]], values))
            continue
        if transform == "norm":
            values = [1.0 + w for w in values]
        elif transform == "a":
            values = tiled([-float32(math.exp(w)) for w in values], k_heads, v_heads, 1)
        elif transform == "heads":
            values = tiled(values, k_heads, v_heads, len(values) // v_heads)
        elif transform in ("channels", "conv"):
            cut = len(values) // shape[0] * qk
            values = values[:cut] + tiled(values[cut:], k_heads, v_heads, (len(values) - cut) // v_heads)
        elif transform == "columns":
            row = shape[1]
            values = [v for start in range(0, len(values), row) for v in tiled(values[start:start + row], k_heads, v_heads, row // v_heads)]
        out.append((target, name, list(reversed([shape[0], shape[2]] if transform == "conv" else shape)), values))
    return out


def hashed(raw):
    """The raw weights in the form tests/f32.py's weight hash reads."""
    return [(name, name, shape, values) for name, shape, values in raw]


def golden():
    """The committed goldens, checked against CONFIG, FIXTURES and the raw weights each fixture's hash records."""
    with open(os.path.join(os.path.dirname(__file__), "data", "baseline_qwen35.json"), encoding="utf-8") as f:
        doc = json.load(f)
    assert doc["config"] == CONFIG and doc["layers"] == LAYERS and doc["v_head_width"] == V_HEAD, "qwen35 fixture config changed"
    assert len(doc["fixtures"]) == len(FIXTURES) and \
        all({key: fixture.get(key) for key in spec} == spec for fixture, spec in zip(doc["fixtures"], FIXTURES)), "qwen35 fixtures changed"
    for fixture, spec in zip(doc["fixtures"], FIXTURES):
        assert weight_hash(hashed(raw_weights(spec))) == fixture["weights_sha256"], "qwen35 %s weights changed" % spec["name"]
    return doc


def write_fixture(directory, fixture, tokens=None):
    """The fixture's GGUF in `directory`, its vocabulary `tokens` where it holds more than VOCAB."""
    path = os.path.join(directory, "tiny-qwen35-%s.gguf" % fixture["name"])
    return write_model(path, gguf_tensors(fixture, raw_weights(fixture)), eos_id=EOS, config=gguf_config(fixture),
                       arch="qwen35moe" if fixture.get("moe") else "qwen35", tokens=tokens)


def check_serve(directory):
    """The Hv = 3 Hk model served with a context of 1024: greedy ids through /v1/generate alone, four at once and from the CLI are the same; served with drafts, the MTP block's file with its embedded drafter and this model with lookup, each reply alone and at once is its reply without drafts, greedy and seeded; a follow-up turn forks the state its first turn kept at its prompt's last whole block and gives the CLI's text for its whole prompt, on this model and on a qwen35moe one, and on the chat route a regenerate of a message after a reply, read whole, keeps a message boundary where that message starts, which an edit of the message then forks; and uncapped requests on a pool too small for them together are paused and resumed with the text each gives alone, with and without checkpoints, those without recomputing from their start; growth takes the paused requests' donors here, so the take-back of a kept state is `server-resume`'s."""
    # Imported here, since the server test imports the baseline checks, which import this module.
    import server
    fixture = FIXTURES[1]
    config = dict(gguf_config(fixture), context_length=1024)
    model = write_model(os.path.join(directory, "tiny-qwen35-serve.gguf"), gguf_tensors(fixture, raw_weights(fixture)), eos_id=EOS, config=config, arch="qwen35")
    prompts = ["hello world", "abc", "the quick brown fox", "0123456789"]
    bodies = [{"prompt": text, "temperature": 0, "max_tokens": 24, "ignore_eos": True} for text in prompts]
    srv = server.Server(model, "--max-seqs", "4")
    try:
        server.alone_and_together(srv, bodies)
        for body in bodies:
            ids = server.post_ok(srv, "/v1/generate", body)["ids"]
            p = common.run_process(["generate", model, body["prompt"], "-n", "24", "--temp", "0", "--ignore-eos"], cache="f32")
            assert p.returncode == 0, p.stderr.decode("utf-8", "replace")
            # A token below 256 is its byte, so the bytes printed are the ids drawn.
            assert list(common.generate_text(p.stdout)) == ids, (body["prompt"], ids)
    finally:
        srv.close()
    # A follow-up turn repeating a 500-byte prompt and its reply, as a chat client sends the conversation back, on this model and on a qwen35moe one; both pass 449 tokens, so on a device they take one tile split and the fork may take the rows.
    routed = next(f for f in FIXTURES if f.get("moe"))
    routed_model = write_model(os.path.join(directory, "tiny-qwen35moe-serve.gguf"), gguf_tensors(routed, raw_weights(routed)), eos_id=EOS,
                               config=dict(gguf_config(routed), context_length=1024), arch="qwen35moe")
    drafting_spec = next(f for f in FIXTURES if f["mtp"])
    drafting = write_model(os.path.join(directory, "tiny-qwen35-serve-mtp.gguf"), gguf_tensors(drafting_spec, raw_weights(drafting_spec)), eos_id=EOS,
                           config=dict(gguf_config(drafting_spec), context_length=1024), arch="qwen35")
    first = "".join(chr(97 + (i * 7) % 26) for i in range(500))
    for served in (model, routed_model):
        # Drafts verified through a hybrid model's mark and its rerun, over a context that holds a longer reply and a chat; the MTP block's file drafts with it too.
        common.check_drafts(served, served, "abcabcabcabcabcabc xyz abcabcabcabc", 60, chat=True)
        if served == model:
            common.check_drafts(drafting, drafting, "abcabcabcabcabcabc xyz abcabcabcabc", 60, chat=True, drafter="embedded")
        # A host tier, which the CPU takes only when given one, holds the message boundary.
        srv = server.Server(served, "--max-seqs", "4", "--host-cache-bytes", str(1 << 28))
        try:
            reply = server.post_ok(srv, "/v1/generate", {"prompt": first, "temperature": 0, "max_tokens": 24, "ignore_eos": True})
            follow = first + reply["text"] + " and then?"
            again = server.post_ok(srv, "/v1/generate", {"prompt": follow, "temperature": 0, "max_tokens": 24, "ignore_eos": True})
            assert 0 < again["reused_tokens"] < len(first), (served, again["reused_tokens"], srv.get("/v1/health"))
            p = common.run_process(["generate", served, follow, "-n", "24", "--temp", "0", "--ignore-eos"], cache="f32")
            assert p.returncode == 0, p.stderr.decode("utf-8", "replace")
            assert list(common.generate_text(p.stdout)) == again["ids"], (served, again["ids"])
            assert srv.get("/v1/health")["reuse"]["device"]["now"]["state_checkpoints"] > 0
            # A regenerate shares nothing with what came before, and its checkpoint lies at its prompt's last whole block, past where an edit of its message parts from it; the boundary it keeps where that message starts is what the edit forks.
            before = [{"role": "user", "content": first[:300]}, {"role": "assistant", "content": first[300:340]}]
            chat = {"temperature": 0, "max_tokens": 8, "ignore_eos": True}
            regen = server.post_ok(srv, "/v1/chat", dict(chat, messages=before + [{"role": "user", "content": first[340:500]}]))
            edited = server.post_ok(srv, "/v1/chat", dict(chat, messages=before + [{"role": "user", "content": "something else entirely"}]))
            health = srv.get("/v1/health")
            assert regen["reused_tokens"] == 0 and 0 < edited["reused_tokens"] < edited["prompt_tokens"], (served, regen, edited, health)
            assert health["reuse"]["boundaries"]["since_start"]["hits"] >= 1, (served, health)
        finally:
            srv.close()
    # Drafts in the scheduler (docs/SPECULATIVE.md, section 3): with the MTP block's drafter and with lookup, each reply alone and with the others at once is its reply without drafts, greedy and seeded, and drafts were fed.
    texts = ["abcabcabcabcabcabc xyz abcabc", "hello hello hello hello", "the quick brown fox", "0123456789"]
    bodies = [{"prompt": text, "temperature": temp, "seed": 5, "max_tokens": 40, "ignore_eos": True} for text in texts for temp in (0, 0.8)]
    for served, drafter in ((drafting, "embedded"), (model, "lookup")):
        plain = server.Server(served, "--max-seqs", "4")
        try:
            want = [server.post_ok(plain, "/v1/generate", body)["ids"] for body in bodies]
        finally:
            plain.close()
        srv = server.Server(served, "--max-seqs", "4", "--drafter", drafter)
        try:
            server.alone_and_together(srv, bodies)
            for body, ids in zip(bodies, want):
                assert server.post_ok(srv, "/v1/generate", body)["ids"] == ids, (drafter, body)
            health = srv.get("/v1/health")
            drafting = health["drafting"]["since_start"]
            assert drafting["drafted"] > 0 and all(p["kept"] <= p["drafted"] for p in drafting["by_position"]), (drafter, health)
        finally:
            srv.close()
    # Uncapped, each request runs to the context the pool holds, so four together pause and resume with the text each gives alone, with checkpoints and without, where each recomputes its history from its start.
    uncapped = [{"prompt": text, "temperature": 0, "ignore_eos": True} for text in prompts]
    for kept in (["--state-checkpoints", "0"], []):
        srv = server.Server(model, "--max-seqs", "4", "--ctx-size", "1024", *kept)
        try:
            alone = [server.post_ok(srv, "/v1/completions", body)["choices"][0]["text"] for body in uncapped]
            # A request here can run in under 100 ms, and the first admits no other once its first growth step holds the room, so the four must arrive at once to be admitted together and pause.
            for (status, reply), body, text in zip(srv.together("/v1/completions", uncapped), uncapped, alone):
                assert status == 200 and reply["choices"][0]["text"] == text, (body, reply, text)
            health = srv.get("/v1/health")
            assert health["requests"]["now"]["active"] == 0 and health["requests"]["now"]["paused"] == 0 and health["pressure"]["since_start"]["pauses"] > 0, health
            if kept:
                assert health["pressure"]["since_start"]["recomputed_tokens"] > 0 and health["pressure"]["since_start"]["resumes_taking_history_back"] == 0 and health["reuse"]["device"]["now"]["entries"] == 0, health
        finally:
            srv.close()


def check_scores(name, model, perplexity):
    """The windowed NLL of the longest text within its witnessed precision bound of HF's, scored in batched passes of three tokens and one token at a time, the decode path, at 1 and 4 threads."""
    for threads in (1, 4):
        for flags in (["--ubatch", "3"], ["--per-token"]):
            for case in perplexity:
                out, bounds = common.run_hf(["perplexity", model, TEXTS[-1], "--threads", str(threads), "-c", str(case["context"])] + flags)
                error = abs(float(common.perplexity_fields(out)["mean NLL"]) - case["mean_nll"])
                assert math.isfinite(error) and error < bounds["nll"], "%s/HF NLL error %s: %.8f" % (name, flags, error)


def check_greedy(name, model, greedy):
    """Greedy decode after a prefill gives HF's greedy tokens, the end-of-text token left out of both, whatever the prefill's ubatch and the thread count."""
    for threads in (1, 4):
        for ubatch in (1, 3, 16):
            p = common.run_process(["generate", model, greedy["prompt"], "-n", str(len(greedy["ids"])), "--temp", "0", "--ignore-eos",
                                    "--threads", str(threads), "--ubatch", str(ubatch)], cache="f32")
            assert p.returncode == 0, "%s generate failed: %s" % (name, p.stderr.decode("utf-8", "replace"))
            # A token below 256 is its byte, so the bytes printed are the ids drawn.
            got = list(common.generate_text(p.stdout))
            assert got == greedy["ids"], "%s greedy ids %s, HF %s (threads %d, ubatch %d)" % (name, got, greedy["ids"], threads, ubatch)


def check_drafter(model, plain, require):
    """The MTP block as an embedded drafter (docs/SPECULATIVE.md, section 7): its drafts and draft logits at steps 1 and 2 against the assembled HF reference (tools/gen_baseline.py qwen35-mtp) within the F32 bound, from prompts of 1, 2, 5 and 12 tokens, through llmx-decode-probe with f32 caches; the drafter loaded by `bench --model` without drafting; and a file without an MTP block refused by name.
    Returns the largest error, or 0 with a skip line where the tool is not beside the executable and `require` does not ask for it."""
    import decode_probe
    rc, out = cli(["generate", plain, "abc", "-n", "2", "--drafter", "embedded"])
    assert rc != 0 and "carries no MTP block" in out, "a file without an MTP block was not refused as an embedded drafter: " + out
    rc, out = cli(["bench", "--model", model, "--p", "4", "--n", "2", "--r", "1", "--drafter", "embedded"])
    assert rc == 0 and re.search(r"^bench: tg2 ", out, re.M), "bench --drafter embedded failed: " + out
    # The rollback reaches the prompt, the verify's 4 rows and a step: at a 16-token context a prompt of 11 runs it and one of 12 skips it with its reason.
    for p, runs in ((11, True), (12, False), (16, False)):
        rc, out = cli(["bench", "--model", model, "--p", str(p), "--n", "8", "--r", "1", "--drafter", "embedded"])
        assert rc == 0 and re.search(r"^bench: tg8 ", out, re.M), "bench --drafter embedded at a prompt of %d failed: %s" % (p, out)
        ran, skipped = re.search(r"^bench: rollback keeping 3 of 4 rows", out, re.M), re.search(r"^bench: rollback skipped: .* past the context of 16$", out, re.M)
        assert (ran and not skipped) if runs else (skipped and not ran), "bench's rollback at a prompt of %d %s: %s" % (p, "did not run" if runs else "was not skipped", out)
    tool = decode_probe.tool_path()
    if not os.path.exists(tool):
        assert not require, "qwen35: llmx-decode-probe is not beside the executable, and --require-tools asks for it"
        print("qwen35: SKIP the drafter's HF reference - llmx-decode-probe is not beside the executable")
        return 0.0
    with open(os.path.join(os.path.dirname(__file__), "data", "baseline_qwen35_mtp.json"), encoding="utf-8") as f:
        doc = json.load(f)
    spec = next(s for s in FIXTURES if s["mtp"])
    assert doc["fixture"] == spec["name"] and doc["weights_sha256"] == weight_hash(hashed(raw_weights(spec))), "the MTP reference is not of the fixture's weights"
    worst = 0.0
    with tempfile.TemporaryDirectory(prefix="llmx_qwen35_mtp_") as directory:
        for case in doc["cases"]:
            p = decode_probe.probe(tool, model, directory, {"prompt": case["prompt"], "ids": [], "tokens": [0, 1], "draft": doc["steps"], "cache": "f32"})
            assert p.returncode == 0, "llmx-decode-probe failed: " + p.stdout + p.stderr
            pick = re.search(r"^draft pick (\d+), (\d+) drafts$", p.stdout, re.M)
            rows = re.findall(r"^draft (\d+) (\d+):((?: \S+)+)$", p.stdout, re.M)
            assert pick and int(pick.group(1)) == case["pick"] and [int(r[1]) for r in rows] == case["drafts"], \
                "%r: pick and drafts %s %s, the reference's %d %s" % (case["prompt"], pick and pick.group(1), [r[1] for r in rows], case["pick"], case["drafts"])
            for (_, _, values), want in zip(rows, case["logits"]):
                got = [float(v) for v in values.split()]
                assert len(got) == len(want), "a draft row of %d logits, the reference's %d" % (len(got), len(want))
                worst = max(worst, max(abs(a - b) for a, b in zip(got, want)))
    assert worst < common.F32_HF_LOGIT_BOUND, "qwen35 drafter/HF logit error %.8f" % worst
    print("qwen35: the MTP block's drafts and their logits at steps 1 and 2 vs the assembled HF reference, prompts of 1, 2, 5 and 12 tokens; max error %.8f  [ok]" % worst)
    return worst


def run(require=False):
    if common.f32_cache_skip("qwen35") or common.tensor_width_skip("qwen35"):
        return common.SKIPPED
    doc = golden()
    fixtures = {fixture["name"]: fixture for fixture in doc["fixtures"]}
    worst = 0.0
    with tempfile.TemporaryDirectory(prefix="llmx_qwen35_") as directory:
        models = {spec["name"]: write_fixture(directory, spec) for spec in FIXTURES}
        rc, out = cli(["logits", models["hv1"], TEXTS[0], "--top", str(VOCAB)])
        if refusal(rc, out):
            print("qwen35: SKIP - llmx does not run the qwen35 architecture here (%s), so nothing was compared" % refusal(rc, out))
            return common.SKIPPED
        check_serve(directory)
        # The bench holds a state slot for each sequence it decodes at once.
        rc, out = cli(["bench", "--model", models["hv3"], "--p", "4", "--n", "2", "--r", "1", "--seqs", "3"])
        reports = re.findall(r"^bench: (.+?)\s+\S+ \+- \S+ tok/s  \((\d+) runs\)$", out, re.M)
        assert rc == 0 and reports == [("pp4", "1"), ("x3 tg2", "1")], "qwen35 bench --seqs 3 failed: " + out
        for spec in FIXTURES:
            name = "qwen35 " + spec["name"]
            fixture = fixtures[spec["name"]]
            # The MTP block's file is held to the goldens of the file without it.
            goldens = fixtures[fixture["base"]] if "base" in fixture else fixture
            model = models[spec["name"]]
            error, _ = common.check_hf_fixture(name, model, goldens["cases"], goldens["perplexity"], TEXTS[-1], UBATCHES)
            worst = max(worst, error, check_logits_input(directory, model, goldens["cases"]))
            check_scores(name, model, goldens["perplexity"])
            check_greedy(name, model, goldens["greedy"])
            common.check_drafts(name, model)
            if spec["mtp"]:
                common.check_drafts(name, model, drafter="embedded")
                worst = max(worst, check_drafter(model, models[fixture["base"]], require))
            if "base" in fixture:
                for text in TEXTS:
                    printed = [cli(["logits", models[which], text, "--top", str(VOCAB)]) for which in (fixture["base"], spec["name"])]
                    assert printed[0] == printed[1], "%s logits differ from %s's on %r" % (name, fixture["base"], text)
    print("qwen35: all 257 logits vs HF's token-by-token goldens, Hv = Hk tied, Hv = 3 Hk untied and qwen35moe, ubatches, threads, --last rows, "
          "NLL batched and per token, greedy decode after a prefill, an MTP block that leaves the logits as they were and drafts as the reference does with the output of no drafts, serve alone, together, from the CLI and through pauses, and bench --seqs 3; max error %.8f  [ok]" % worst)
    return True


if __name__ == "__main__":
    run()
