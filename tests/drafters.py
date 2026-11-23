"""Drafter files beside a model (docs/SPECULATIVE.md, step 6), on tiny fixtures.

The tiny qwen35 model's MTP block split into a file of its own by llmx-drafter-pack drafts from beside the model as it drafts embedded, bit for bit, and embedded back gives the file it came from; with it, generate, chat, bench and serve give what they give without drafts, and generate does at a one-block --kv-tokens.
Draft models, a qwen35 and a qwen3 model sharing the target's tokenizer, give generate's and chat's output without drafts too, and the server and bench refuse them.
Every pairing check (infer::spec::pair) refuses its case by name, as it does a DFlash drafter, whose architecture llmx does not run, and a file cut short.
The drafting set's driver, tools/drafting_set.py, reads generate's lines on the easy prompt, and refuses a changed prompt and differing ids.
"""
import contextlib
import io
import json
import os
import re
import subprocess
import sys
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
    common.check_drafts("drafters sidecar behind lookup", split, drafter="lookup," + side)
    rc, out = cli(["bench", "--model", split, "--p", "4", "--n", "8", "--r", "1", "--drafter", side])
    assert rc == 0 and re.search(r"^bench: tg8 ", out, re.M) and re.search(r"^bench: rollback keeping 3 of 4 rows", out, re.M) and \
        re.search(r"^bench: draft 3  chain ", out, re.M), \
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
    # generate takes --kv-tokens as serve does: a budget of the model's context beside the drafter gives the ids of the default, its prompt and reply in whole blocks, and the cache it backs is larger.
    runs = []
    for budget in ([], ["--kv-tokens", "1k"]):
        p = common.run_process(["generate", split, "abcabc xyz", "-n", "20", "--ignore-eos", "--temp", "0", "--verbose", "--drafter", side] + budget)
        out = p.stdout.decode("utf-8", "replace")
        assert p.returncode == 0, "generate --drafter %s failed: %s" % (budget, p.stderr.decode("utf-8", "replace"))
        runs.append((re.search(r"^ids:.*$", out, re.M)[0], int(re.search(r"^kv: allocated (\d+) bytes", out, re.M)[1])))
    assert runs[1][0] == runs[0][0] and runs[1][1] > runs[0][1], "generate --drafter with --kv-tokens 1k gave %r, without it %r" % (runs[1], runs[0])
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
        assert health["drafting"]["since_start"]["drafted"] > 0, health
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
    # A DFlash drafter of the target's tokenizer is refused by its architecture, which llmx does not run (docs/SPECULATIVE.md, step 7).
    md = {k: v for k, v in spec_decode.GGUF(split).metadata.items() if k.startswith("tokenizer.")}
    md["general.architecture"] = (8, "dflash")
    spec_decode.write_gguf(d("dflash.gguf"), md, [])
    refused(split, d("dflash.gguf"), "its architecture 'dflash' is not one llmx runs", "a DFlash drafter")


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
        common.check_drafts("drafters " + name + " behind lookup", target, "abcabcabcabcabcabc xyz abcabcabcabc", 60, chat=True, drafter="lookup," + drafter)
    no_eos = write_model(os.path.join(directory, "draft-no-eos.gguf"), qwen3_tensors(False))
    refused(target, no_eos, "tokenizer.ggml.eos_token_id is none where the model has %d" % qwen35.EOS, "a draft model without the target's end of text")
    rc, out = cli(["bench", "--model", target, "--p", "4", "--n", "2", "--r", "1", "--drafter", plain])
    assert rc != 0 and "is a draft model" in out, "bench took a draft model: " + out
    rc, out = cli(["serve", target, "--port", "0", "--no-auth", "--drafter", plain] + common.pool_args(target))
    assert rc != 0 and "is a draft model, which drafts in generate and chat" in out, "serve took a draft model: " + out


def check_drafting_set(directory):
    """The driver of docs/SPECULATIVE.md, section 10, on the easy prompt raw, which the fixture's byte tokens fit with its 512 generated tokens."""
    sys.path.insert(0, os.path.join(common.ROOT, "tools"))
    import drafting_set
    spec = next(f for f in qwen35.FIXTURES if f["mtp"])
    model = qwen35.write_model(os.path.join(directory, "set-mtp.gguf"), qwen35.gguf_tensors(spec, qwen35.raw_weights(spec)), eos_id=qwen35.EOS,
                               config=dict(qwen35.gguf_config(spec), context_length=4608), arch="qwen35")
    out = os.path.join(directory, "set.json")
    flags = common.device_args(["generate", "--ignore-eos"], cache="f32")[1:]
    argv = ["--exe", common.exe_path(), "--model", model, "--drafter", "lookup", "--prompts", "easy", "--path", "raw", "--json", out]
    p = subprocess.run([sys.executable, drafting_set.__file__] + argv + ["--"] + flags, capture_output=True, encoding="utf-8", errors="replace")
    assert p.returncode == 0, "the drafting set failed on the tiny model: " + p.stdout[-2000:] + p.stderr[-2000:]
    with open(out, encoding="utf-8") as f:
        easy = json.load(f)["prompts"]["easy"]
    size = os.path.getsize(drafting_set.fixture("easy"))
    off, on = easy["off"], easy["drafter"]
    assert easy["ids_equal"] and "| yes |" in p.stdout, "the drafting set's ids witness: " + p.stdout
    assert off["prompt_tokens"] == size and off["generated"] == drafting_set.MAX_TOKENS and on["fed"] > 0 and on["pp"] > 0 and on["tg"] > 0, "the drafting set read other figures from generate's lines than it ran: %r" % easy
    # Differing ids and a changed prompt each end the driver with its reason, from the runs just read.
    runs = easy["runs"]
    stub = lambda args, prompt, path, drafter: dict(runs["drafter" if drafter else "off"][0], ids="0" if drafter else runs["off"][0]["ids"])
    changed = os.path.join(directory, "drafting")
    os.mkdir(changed)
    with open(os.path.join(changed, "easy.txt"), "wb") as f:
        f.write(b" ")
    for patch, reason in (({"run": stub}, "greedy ids differ"), ({"DATA": changed}, "SHA-256")):
        saved, argv0 = {k: getattr(drafting_set, k) for k in patch}, sys.argv
        sys.argv = [drafting_set.__file__] + argv
        try:
            for k, v in patch.items():
                setattr(drafting_set, k, v)
            with contextlib.redirect_stdout(io.StringIO()):
                drafting_set.main()
            raise AssertionError("the drafting set did not stop on " + reason)
        except SystemExit as e:
            assert reason in str(e.code), "the drafting set stopped with %r, not %r" % (e.code, reason)
        finally:
            sys.argv = argv0
            for k, v in saved.items():
                setattr(drafting_set, k, v)


def check_dspine(directory, tool):
    """The tiny DSpine sidecar (tests/fixture_dspine.py) beside the tiny qwen35 target, its K and V heads not the target's: each golden case's pick, its block's drafts and each draft row's logits through llmx-decode-probe, against tools/dspine_reference.py's (tests/data/qwen35_dspine.json), the ids exactly, since the golden has no near-tie, and the logits within the F32 bound."""
    import fixture_dspine
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "qwen35_dspine.json"), encoding="utf-8") as f:
        doc = json.load(f)
    target = qwen35.write_fixture(directory, fixture_dspine.target_spec())
    sidecar = os.path.join(directory, "dspine.gguf")
    assert fixture_dspine.write_sidecar(sidecar, target, doc["seed"]) == doc["sha256"], "drafters: the DSpine sidecar is not the golden's"
    worst = 0.0
    for case in doc["cases"]:
        drafts = len(case["drafts"])
        p = decode_probe.probe(tool, target, directory, {"prompt": case["prompt"], "ids": [], "tokens": [0, 1], "draft": drafts, "cache": "f32", "drafter_file": sidecar})
        assert p.returncode == 0, "drafters: llmx-decode-probe with the DSpine sidecar failed: " + p.stdout[-2000:] + p.stderr[-2000:]
        pick = re.search(r"^draft pick (\d+), (\d+) drafts$", p.stdout, re.M)
        rows = re.findall(r"^draft (\d+) (\d+):((?: \S+)+)$", p.stdout, re.M)
        assert pick and int(pick.group(1)) == case["anchor"] and [int(r[1]) for r in rows] == case["drafts"], \
            "drafters: %r gave pick and drafts %s %s, the reference %d %s" % (case["prompt"], pick and pick.group(1), [r[1] for r in rows], case["anchor"], case["drafts"])
        for (_, _, values), want in zip(rows, case["logits"]):
            got = [float(v) for v in values.split()]
            assert len(got) == len(want), "drafters: a DSpine draft row of %d logits, the reference's %d" % (len(got), len(want))
            worst = max(worst, max(abs(a - b) for a, b in zip(got, want)))
    assert worst < common.F32_HF_LOGIT_BOUND, "drafters: DSpine draft logits %.8f from the reference" % worst
    return worst


def run(require=False):
    if common.f32_cache_skip("drafters") or common.tensor_width_skip("drafters"):
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
        check_drafting_set(directory)
        dspine = check_dspine(directory, tool)
    print("drafters: the MTP block split beside its model drafts as embedded and embedded back, bit for bit, and gives generate, chat, bench and serve "
          "their output without drafts; qwen35 and qwen3 draft models give generate's and chat's; every pairing check, a DFlash drafter and a file cut short refused by name; the drafting set's driver reads generate's lines; the DSpine sidecar's drafts the reference's, logits within %.8f  [ok]" % dspine)
    return True


if __name__ == "__main__":
    run()
