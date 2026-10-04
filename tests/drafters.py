"""Drafter files beside a model (docs/SPECULATIVE.md, step 6), on tiny fixtures.

The tiny qwen35 model's MTP block split into a file of its own by llmx-drafter-pack drafts from beside the model as it drafts embedded, bit for bit, and embedded back gives the file it came from; with it, generate, chat, bench and serve give what they give without drafts.
Draft models, a qwen35 and a qwen3 model sharing the target's tokenizer, give generate's and chat's output without drafts too, and the server and bench refuse them.
Every pairing check (infer::spec::pair) refuses its case by name, as does a DFlash drafter that pairs, which nothing runs yet, and a file cut short.
"""
import os
import re
import subprocess
import tempfile

import common
import decode_probe
import qwen35
import spec_decode
from common import run_f32_cache as cli
from f32 import CONFIG as QWEN3_CONFIG, TEXTS, VOCAB, tensors as qwen3_tensors, write_model


def pack_path():
    return os.path.join(os.path.dirname(common.exe_path()), "llmx-drafter-pack" + (".exe" if os.name == "nt" else ""))


def pack(*args):
    p = subprocess.run([pack_path()] + list(args), capture_output=True, encoding="utf-8", errors="replace", timeout=120)
    assert p.returncode == 0, "llmx-drafter-pack %s failed: %s" % (" ".join(args), p.stderr)


def variant(src, dst, metadata=None, rename=None):
    """A copy of the GGUF `src` at `dst` with `metadata` set over its own (a value of None drops the key) and tensors renamed as `rename` maps them."""
    g = spec_decode.GGUF(src)
    md = dict(g.metadata)
    for key, value in (metadata or {}).items():
        if value is None:
            md.pop(key, None)
        else:
            md[key] = value
    tensors = [((rename or {}).get(t.name, t.name), t.shape, t.type, g.raw(t)) for t in g.tensors]
    spec_decode.write_gguf(dst, md, tensors)
    return dst


def refused(model, drafter, text, what):
    rc, out = cli(["generate", model, "abc", "-n", "1", "--drafter", drafter])
    assert rc != 0 and text in out, "%s was not refused by name (%r): %s" % (what, text, out)


def probed(tool, model, directory, fixture):
    p = decode_probe.probe(tool, model, directory, fixture)
    assert p.returncode == 0, "llmx-decode-probe failed: " + p.stdout + p.stderr
    # The first line names the model file.
    return p.stdout.split("\n", 1)[1]


def check_sidecar(directory, tool):
    spec = next(f for f in qwen35.FIXTURES if f["mtp"])
    embedded = qwen35.write_fixture(directory, spec)
    base = qwen35.write_fixture(directory, next(f for f in qwen35.FIXTURES if f["name"] == "hv3"))
    split, side, joined = (os.path.join(directory, name) for name in ("split.gguf", "mtp-blocks.gguf", "joined.gguf"))
    pack("split", embedded, split, side)
    for text in TEXTS:
        assert cli(["logits", split, text, "--top", str(VOCAB)]) == cli(["logits", base, text, "--top", str(VOCAB)]), \
            "the model split from its MTP block gives other logits than the file without one on %r" % text
    # The drafts and every draft row's logits, from beside the model and embedded in it, and embedded back by the tool.
    for prompt in ("a", "abcab", "abcdefghij"):
        fixture = {"prompt": prompt, "ids": [], "tokens": [0, 1], "draft": 3, "cache": "f32"}
        want = probed(tool, embedded, directory, fixture)
        assert re.search(r"^draft pick \d+, \d+ drafts$", want, re.M), want
        assert probed(tool, split, directory, dict(fixture, drafter_file=side)) == want, "the MTP block beside the model drafts otherwise than embedded, after %r" % prompt
        if prompt == "a":
            pack("embed", split, side, joined)
        assert probed(tool, joined, directory, fixture) == want, "the MTP block embedded back drafts otherwise, after %r" % prompt
    common.check_drafts("drafters sidecar", split, drafter=side)
    rc, out = cli(["bench", "--model", split, "--p", "4", "--n", "8", "--r", "1", "--drafter", side])
    assert rc == 0 and re.search(r"^bench: tg8 ", out, re.M) and re.search(r"^bench: rollback keeping 3 of 4 rows", out, re.M), \
        "bench --drafter with the MTP block beside the model failed: " + out
    return embedded, split, side


def check_served(directory):
    """The MTP block beside a model of a 1024-token context, served: each reply alone and at once is its reply without drafts, greedy and seeded, and drafts were fed."""
    import server
    spec = next(f for f in qwen35.FIXTURES if f["mtp"])
    model = qwen35.write_model(os.path.join(directory, "serve-mtp.gguf"), qwen35.gguf_tensors(spec, qwen35.raw_weights(spec)), eos_id=qwen35.EOS,
                               config=dict(qwen35.gguf_config(spec), context_length=1024), arch="qwen35")
    split, side = os.path.join(directory, "serve-split.gguf"), os.path.join(directory, "serve-mtp-blocks.gguf")
    pack("split", model, split, side)
    common.check_drafts("drafters sidecar at 1024", split, "abcabcabcabcabcabc xyz abcabcabcabc", 60, chat=True, drafter=side)
    texts = ["abcabcabcabcabcabc xyz abcabc", "hello hello hello hello", "0123456789"]
    bodies = [{"prompt": text, "temperature": temp, "seed": 5, "max_tokens": 40, "ignore_eos": True} for text in texts for temp in (0, 0.8)]
    plain = server.Server(split, "--max-seqs", "4")
    try:
        want = [server.post_ok(plain, "/v1/generate", body)["ids"] for body in bodies]
    finally:
        plain.close()
    srv = server.Server(split, "--max-seqs", "4", "--drafter", side)
    try:
        server.alone_and_together(srv, bodies)
        for body, ids in zip(bodies, want):
            assert server.post_ok(srv, "/v1/generate", body)["ids"] == ids, body
        health = srv.get("/v1/health")
        assert sum(health["drafted"]) > 0, health
    finally:
        srv.close()


def check_refusals(directory, embedded, split, side):
    d = lambda name: os.path.join(directory, name)
    p = "qwen35."
    g = spec_decode.GGUF(side)
    element, items = g.value("tokenizer.ggml.tokens")
    eos = g.value("tokenizer.ggml.eos_token_id")
    items = list(items)
    items[65] = "changed"
    changed = (9, (element, items))
    refused(split, variant(side, d("v-tokens.gguf"), {"tokenizer.ggml.tokens": changed}), "tokenizer.ggml.tokens is", "a drafter of another token list")
    refused(split, variant(side, d("v-eos.gguf"), {"tokenizer.ggml.eos_token_id": (4, 3)}), "tokenizer.ggml.eos_token_id is 3 where the model has %d" % eos,
            "a drafter of another end of text")
    width = g.value(p + "embedding_length")
    refused(split, variant(side, d("v-width.gguf"), {p + "embedding_length": (4, width + 1)}),
            "%sembedding_length is %d where the model has %d" % (p, width + 1, width), "MTP blocks of another width")
    refused(split, variant(side, d("v-blocks.gguf"), {p + "block_count": (4, 7)}), p + "block_count is 7 where the model's 4 layers", "MTP blocks counted otherwise")
    refused(split, variant(side, d("v-nextn.gguf"), {p + "nextn_predict_layers": (4, 0)}), "nextn_predict_layers is 0", "blocks that are not MTP blocks")
    norm = next(t.name for t in g.tensors if t.name.endswith("nextn.enorm.weight"))
    refused(split, variant(side, d("v-past.gguf"), rename={norm: norm.replace("blk.4.", "blk.9.")}), "is past its 5 blocks", "a tensor past the drafter's blocks")
    refused(embedded, side, "carries an MTP block of its own", "MTP blocks beside a model that embeds its own")
    other = write_model(d("other-arch.gguf"), qwen3_tensors(False), eos_id=eos, arch="mamba")
    refused(split, other, "its architecture 'mamba' is not one llmx runs", "a drafter of an architecture llmx does not run")
    cut = d("cut.gguf")
    with open(side, "rb") as f, open(cut, "wb") as out:
        out.write(f.read(os.path.getsize(side) // 2))
    rc, out = cli(["generate", split, "abc", "-n", "1", "--drafter", cut])
    assert rc != 0 and "GGUF" in out, "a drafter file cut short was not refused: " + out
    # A DFlash drafter is paired on its header: the target's tokenizer, its architecture, its hidden size, its taps and its mask token; one that pairs is refused as nothing runs it yet.
    kinds = [1] * VOCAB
    kinds[eos] = 3
    types = (9, (5, kinds))
    typed = variant(split, d("typed.gguf"), {"tokenizer.ggml.token_type": types})
    def dflash(name, target, **keys):
        t = spec_decode.GGUF(target)
        md = {k: v for k, v in t.metadata.items() if k.startswith("tokenizer.")}
        md["general.architecture"] = (8, "dflash")
        md["dflash.embedding_length"] = (4, keys.get("width", width))
        md["dflash.target_layers"] = (9, (5, keys.get("taps", [1, 3])))
        md["dflash.block_size"] = (4, 8)
        md["tokenizer.ggml.mask_token_id"] = (4, keys.get("mask", eos))
        spec_decode.write_gguf(d(name), md, [])
        return d(name)
    refused(typed, dflash("df-width.gguf", typed, width=width + 1), "dflash.embedding_length is %d where the model has %d" % (width + 1, width), "a DFlash drafter of another width")
    refused(typed, dflash("df-order.gguf", typed, taps=[3, 1]), "dflash.target_layers is not strictly increasing within the model's 4 layers", "DFlash taps out of order")
    refused(typed, dflash("df-past.gguf", typed, taps=[1, 5]), "dflash.target_layers is not strictly increasing within the model's 4 layers", "DFlash taps past the layers")
    refused(typed, dflash("df-mask.gguf", typed, mask=65), "tokenizer.ggml.mask_token_id 65 is not a control or user token", "a DFlash mask that is a plain token")
    refused(typed, dflash("df-ok.gguf", typed), "is a DFlash drafter, which pairs with", "a DFlash drafter, which nothing runs yet")
    qwen3_typed = variant(write_model(d("qwen3-target.gguf"), qwen3_tensors(False), eos_id=eos), d("qwen3-typed.gguf"), {"tokenizer.ggml.token_type": types})
    refused(qwen3_typed, dflash("df-qwen3.gguf", qwen3_typed, width=QWEN3_CONFIG["embedding_length"]), "a DFlash drafter drafts for no model of the architecture 'qwen3'",
            "a DFlash drafter beside a model of an architecture it drafts for none of")


def wide(directory, name):
    """The tiny qwen35 fixture `name` at a context of 1024, which a chat of two turns fits."""
    spec = next(f for f in qwen35.FIXTURES if f["name"] == name)
    return qwen35.write_model(os.path.join(directory, "wide-%s.gguf" % name), qwen35.gguf_tensors(spec, qwen35.raw_weights(spec)), eos_id=qwen35.EOS,
                              config=dict(qwen35.gguf_config(spec), context_length=1024), arch="qwen35")


def check_draft_models(directory):
    target, hybrid = wide(directory, "hv3"), wide(directory, "hv1")
    plain = write_model(os.path.join(directory, "draft-qwen3.gguf"), qwen3_tensors(False), eos_id=qwen35.EOS, config=dict(QWEN3_CONFIG, context_length=1024))
    for name, drafter in (("a qwen35 draft model", hybrid), ("a qwen3 draft model", plain)):
        common.check_drafts("drafters " + name, target, "abcabcabcabcabcabc xyz abcabcabcabc", 60, chat=True, drafter=drafter)
    no_eos = write_model(os.path.join(directory, "draft-no-eos.gguf"), qwen3_tensors(False))
    refused(target, no_eos, "tokenizer.ggml.eos_token_id is none where the model has %d" % qwen35.EOS, "a draft model without the target's end of text")
    rc, out = cli(["bench", "--model", target, "--p", "4", "--n", "2", "--r", "1", "--drafter", plain])
    assert rc != 0 and "is a draft model" in out, "bench took a draft model: " + out
    rc, out = cli(["serve", target, "--port", "0", "--drafter", plain])
    assert rc != 0 and "is a draft model, which drafts in generate and chat" in out, "serve took a draft model: " + out


def run(require=False):
    if common.f32_cache_skip("drafters"):
        return common.SKIPPED
    tool = decode_probe.tool_path()
    if not os.path.exists(pack_path()) or not os.path.exists(tool):
        assert not require, "drafters: llmx-drafter-pack or llmx-decode-probe is not beside the executable, and --require-tools asks for it"
        print("drafters: SKIP - llmx-drafter-pack or llmx-decode-probe is not beside the executable, so nothing was compared")
        return common.SKIPPED
    with tempfile.TemporaryDirectory(prefix="llmx_drafters_") as directory:
        rc, out = cli(["logits", qwen35.write_fixture(directory, qwen35.FIXTURES[0]), TEXTS[0]])
        if qwen35.refusal(rc, out):
            print("drafters: SKIP - llmx does not run the qwen35 architecture here (%s), so nothing was compared" % qwen35.refusal(rc, out))
            return common.SKIPPED
        embedded, split, side = check_sidecar(directory, tool)
        check_served(directory)
        check_refusals(directory, embedded, split, side)
        check_draft_models(directory)
    print("drafters: the MTP block split beside its model drafts as embedded and embedded back, bit for bit, and gives generate, chat, bench and serve "
          "their output without drafts; qwen35 and qwen3 draft models give generate's and chat's; every pairing check, a DFlash drafter and a file cut short refused by name  [ok]")
    return True


if __name__ == "__main__":
    run()
