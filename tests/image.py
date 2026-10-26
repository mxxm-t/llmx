import base64
import hashlib
import json
import math
import os
import re
import tempfile

import common
import qwen35
import qwen4exp
import spec_decode
from f32 import build_byte_vocab, write_model
from server import Server, post_ok

# A prompt with images through `llmx logits` and `generate` with --chat, --mmproj and --image, against HF Qwen3_5ForConditionalGeneration (tools/gen_baseline.py image-tiny, AGENTS.md, Tests).
# Every prompt row and the greedy ids after it are held to HF's token-by-token cached forward at the F32 bound, and on a tiny qwen4exp model to HF Qwen4ExpForConditionalGeneration (image-qwen4exp).
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
# The vision tokens after the byte vocabulary and the end of text, as control tokens.
SPECIAL = ["<|endoftext|>", "<|vision_start|>", "<|image_pad|>", "<|vision_end|>"]
EOS, VISION_START, IMAGE_PAD, VISION_END = 256, 257, 258, 259
FIXTURE = {"name": "image", "v_heads": 2, "tied": True, "mtp": False, "vocab": 256 + len(SPECIAL), "config": {"context_length": 1024}}
# A template of the Qwen shape, an image part as its three tokens, small enough that every byte of it is a token.
TEMPLATE = ("{% for m in messages %}{{ m.role }}: {% if m.content is string %}{{ m.content }}{% else %}{% for item in m.content %}"
            "{% if item.type == 'image' %}<|vision_start|><|image_pad|><|vision_end|>{% else %}{{ item.text }}{% endif %}{% endfor %}{% endif %}"
            "{{ '\\n' }}{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}")
# One image at a small pixel bound, and two at the processor's own, of other grids.
CASES = [{"text": "Describe it.", "images": ["photo-300x200.png"], "max_pixels": 32 * 32 * 12},
         {"text": "Compare them.", "images": ["photo-300x200.jpg", "rgb-23x37.png"], "max_pixels": None}]
MMPROJ = os.path.join(DATA, "vision", "tiny-mmproj-image.gguf")
# The Flash-Next fixture's text model and tower, which tools/gen_baseline.py image-qwen4exp writes beside its goldens.
QWEN4EXP_MODEL = os.path.join(DATA, "vision", "tiny-qwen4exp-image.gguf")
QWEN4EXP_MMPROJ = os.path.join(DATA, "vision", "tiny-mmproj-qwen4exp.gguf")
GREEDY = 8


def tokens():
    return build_byte_vocab() + SPECIAL


def write_fixture(directory):
    """The text model's GGUF, its weights tests/qwen35.py's for this fixture."""
    path = os.path.join(directory, "tiny-qwen35-image.gguf")
    return write_model(path, qwen35.gguf_tensors(FIXTURE, qwen35.raw_weights(FIXTURE)), chat_template=TEMPLATE, eos_id=EOS,
                       config=qwen35.gguf_config(FIXTURE), arch="qwen35", tokens=tokens(), control=set(range(256, 256 + len(SPECIAL))))


def write_qwen4exp_fixture(path, config, state):
    """The Flash-Next fixture's text GGUF: tests/qwen4exp.py's map of an HF state, with this fixture's vocabulary, control tokens and template."""
    metadata, tensors = qwen4exp.map_qwen4exp(config, state)
    vocab = tokens()
    metadata["tokenizer.ggml.tokens"] = (spec_decode.ARRAY, (spec_decode.STRING, vocab))
    metadata["tokenizer.ggml.token_type"] = (spec_decode.ARRAY, (qwen4exp.GGUF_INT32, [3 if i >= 256 else 1 for i in range(len(vocab))]))
    metadata["tokenizer.chat_template"] = (spec_decode.STRING, TEMPLATE)
    spec_decode.write_gguf(path, metadata, tensors)
    return path


def qwen4exp_golden():
    with open(os.path.join(DATA, "baseline_image_qwen4exp.json"), encoding="utf-8") as f:
        doc = json.load(f)
    assert doc["template"] == TEMPLATE, "the Flash-Next image fixture changed"
    assert [{k: c[k] for k in ("text", "images", "max_pixels")} for c in doc["cases"]] == CASES, "the Flash-Next image cases changed"
    for path, key in ((QWEN4EXP_MODEL, "model_sha256"), (QWEN4EXP_MMPROJ, "mmproj_sha256")):
        with open(path, "rb") as f:
            assert hashlib.sha256(f.read()).hexdigest() == doc[key], "%s is not the one its goldens were made from" % path
    return doc


def golden():
    with open(os.path.join(DATA, "baseline_image_tiny.json"), encoding="utf-8") as f:
        doc = json.load(f)
    assert doc["fixture"] == FIXTURE and doc["template"] == TEMPLATE, "the image fixture changed"
    assert [{k: c[k] for k in ("text", "images", "max_pixels")} for c in doc["cases"]] == CASES, "the image cases changed"
    with open(MMPROJ, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == doc["mmproj_sha256"], "the image fixture's mmproj is not the one its goldens were made from"
    return doc


def image_args(case, mmproj=MMPROJ):
    args = ["--chat", "--dtype", "f32", "--mmproj", mmproj, "--image", ",".join(os.path.join(DATA, "images", f) for f in case["images"])]
    return args + (["--image-max-pixels", str(case["max_pixels"])] if case["max_pixels"] else [])


def check_rows(model, case, mmproj=MMPROJ):
    """Every prompt row's logits within the F32 bound of HF's, the prompt's mean NLL within its bound, at 1 and 4 threads and passes of 3 and 16 rows, each the bytes of one pass, and read a token at a time through decode steps.
    Returns the worst logit and NLL errors."""
    worst, worst_nll, first = 0.0, 0.0, None
    n = len(case["ids"])
    for threads, ubatch in ((1, 1024), (4, 1024), (1, 3), (4, 16), (1, 0)):
        how = ["--per-token"] if not ubatch else ["--ubatch", str(ubatch)]
        out, bounds = common.run_hf(["logits", model, case["text"]] + image_args(case, mmproj) +
                                    ["--last", str(n), "--top", str(FIXTURE["vocab"]), "--threads", str(threads)] + how)
        lines = out.splitlines()
        assert lines[0] == "tokens: %d" % n, "image %r: llmx read %s, HF %d tokens" % (case["text"], lines[0], n)
        rows = lines[1:]
        assert [int(r.split()[0]) for r in rows] == list(range(n)), "image %r: positions printed" % case["text"]
        if first is None:
            first = rows
        # Decode steps take another path than a prompt's passes, so only the bound holds them.
        assert not ubatch or rows == first, "image %r: threads %d ubatch %d differ from one pass" % (case["text"], threads, ubatch)
        nll_got = nll_ref = 0.0
        for pos, (row, ref) in enumerate(zip(rows, case["logits"])):
            fields = row.split()
            got = {int(i): float(v) for i, v in zip(fields[1::2], fields[2::2])}
            worst = max(worst, common.hf_logit_error("image %r row %d" % (case["text"], pos), got, ref, bound=bounds["logit"]))
            if pos + 1 < n:
                target = case["ids"][pos + 1]
                nll_got += math.log(math.fsum(math.exp(v - got[target]) for v in got.values()))
                nll_ref += math.log(math.fsum(math.exp(v - ref[target]) for v in ref))
        error = abs(nll_got - nll_ref) / (n - 1)
        assert error < bounds["nll"], "image %r: mean NLL error %.8f" % (case["text"], error)
        worst_nll = max(worst_nll, error)
    return worst, worst_nll


def check_greedy(model, case, mmproj=MMPROJ):
    """Greedy decode after the prompt gives HF's ids, the end of text left out of both, with lookup drafts as without, since they never draft from an image's rows."""
    for threads, ubatch, drafter in ((1, 1024, "off"), (4, 3, "off"), (1, 1024, "lookup")):
        p = common.run_process(["generate", model, case["text"]] + image_args(case, mmproj) +
                               ["-n", str(GREEDY), "--temp", "0", "--ignore-eos", "--verbose", "--threads", str(threads), "--ubatch", str(ubatch),
                                "--drafter", drafter], cache="f32")
        out = p.stdout.decode("utf-8", "replace")
        assert p.returncode == 0, "image generate failed: " + out + p.stderr.decode("utf-8", "replace")
        ids = [int(x) for x in re.search(r"^ids: (.*)$", out, re.M).group(1).split(",")]
        assert ids == case["greedy"], "image %r greedy ids %s, HF %s (threads %d, ubatch %d)" % (case["text"], ids, case["greedy"], threads, ubatch)


def check_refusals(model):
    """The image flags' refusals that need a model: a text whose own pad token makes the prompt mark more images than were given, and an encoder of another width."""
    case = CASES[0]
    p = common.run_process(["logits", model, "Look <|image_pad|> here."] + image_args(case), cache="f32")
    err = p.stderr.decode("utf-8", "replace")
    assert p.returncode == 1 and "the prompt marks 2 images and 1 were given" in err, err
    other = os.path.join(DATA, "vision", "tiny-mmproj.gguf")
    p = common.run_process(["logits", model, case["text"], "--chat", "--mmproj", other, "--image", os.path.join(DATA, "images", case["images"][0])], cache="f32")
    err = p.stderr.decode("utf-8", "replace")
    assert p.returncode == 1 and "the vision encoder's rows are 48 wide and the model's embedding 37" in err, err


def data_url(name):
    with open(os.path.join(DATA, "images", name), "rb") as f:
        raw = f.read()
    return "data:image/%s;base64,%s" % ("png" if name.endswith(".png") else "jpeg", base64.b64encode(raw).decode("ascii"))


def user(text, images):
    """A user message of `images` before `text`, as the CLI's --chat places them."""
    return {"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url(f)}} for f in images] + [{"type": "text", "text": text}]}


def ask(messages, n=GREEDY):
    return {"messages": messages, "max_tokens": n, "temperature": 0, "ignore_eos": True, "logprobs": True}


def same_reply(a, b, what):
    assert a["ids"] == b["ids"] and a["logprobs"] == b["logprobs"], "%s: %s, against %s" % (what, a, b)


def image_server(model, case, *extra):
    flags = ["--mmproj", MMPROJ, "--dtype", "f32"] + (["--image-max-pixels", str(case["max_pixels"])] if case["max_pixels"] else [])
    return Server(model, *(flags + list(extra)))


def check_served(model, doc):
    """Each case through /v1/chat and /v1/chat/completions: HF's token count through /v1/tokenize and the prompt, HF's greedy ids, and the same ids and log-probabilities alone and four at once."""
    for case in doc["cases"]:
        srv = image_server(model, case)
        try:
            messages = [user(case["text"], case["images"])]
            tokenized = post_ok(srv, "/v1/tokenize", {"messages": messages})
            assert tokenized["tokens"] == case["ids"], "image %r: /v1/tokenize %s, HF %s" % (case["text"], tokenized["tokens"], case["ids"])
            alone = post_ok(srv, "/v1/chat", ask(messages))
            assert alone["prompt_tokens"] == len(case["ids"]) and alone["ids"] == case["greedy"], "image %r: served %s, HF %s" % (case["text"], alone, case["greedy"])
            compat = post_ok(srv, "/v1/chat/completions", {k: v for k, v in ask(messages).items() if k != "logprobs"})
            assert compat["usage"]["prompt_tokens"] == len(case["ids"]), compat
            together = srv.together("/v1/chat", [ask(messages)] * 4)
            for status, reply in together:
                assert status == 200, reply
                same_reply(reply, alone, "image %r four at once" % case["text"])
        finally:
            srv.close()


def check_served_reuse(model):
    """A follow-up turn of the same image reuses its first turn's checkpoint past the image, and with another image of the same grid nothing from the image's first row on, each reply that of a fresh server."""
    case = CASES[1]
    text = "Tell me about this picture in detail. " * 8
    first = [user(text, ["photo-300x200.png"])]
    other = [user(text, ["photo-300x200.jpg"])]
    srv = image_server(model, case)
    try:
        prompt = post_ok(srv, "/v1/tokenize", {"messages": first})["tokens"]
        image_at = prompt.index(IMAGE_PAD)
        reply = post_ok(srv, "/v1/chat", ask(first))
        follow = first + [{"role": "assistant", "content": reply["text"]}, {"role": "user", "content": "And the colours?"}]
        again = post_ok(srv, "/v1/chat", ask(follow))
        assert again["reused_tokens"] >= 128 and again["reused_tokens"] > prompt.count(IMAGE_PAD) + image_at, ("the same image reused too little", again, image_at)
        changed = other + follow[1:]
        moved = post_ok(srv, "/v1/chat", ask(changed))
        assert moved["reused_tokens"] <= image_at, ("another image of the same grid reused past its first row", moved, image_at)
    finally:
        srv.close()
    for messages, got in ((follow, again), (changed, moved)):
        fresh = image_server(model, case)
        try:
            same_reply(got, post_ok(fresh, "/v1/chat", ask(messages)), "a follow-up against a fresh server")
        finally:
            fresh.close()


def check_served_cut(model):
    """With --context-overflow shift a cut that drops the image's turn gives the reply of the messages it kept, sent alone."""
    case = CASES[1]
    srv = image_server(model, case, "--context-overflow", "shift", "--ctx-size", "256")
    try:
        messages = [user("What is this?", ["photo-300x200.png"]), {"role": "assistant", "content": "A photo."},
                    {"role": "user", "content": "x" * 150}, {"role": "assistant", "content": "Many x."}, {"role": "user", "content": "y" * 40}]
        cut = post_ok(srv, "/v1/chat", ask(messages))
        dropped = cut["context"].get("dropped_messages", 0)
        assert dropped >= 2, ("the cut kept the image's turn", cut)
        kept = post_ok(srv, "/v1/chat", ask(messages[dropped:]))
        assert cut["prompt_tokens"] == kept["prompt_tokens"] == post_ok(srv, "/v1/tokenize", {"messages": messages[dropped:]})["count"], (cut, kept)
        same_reply(cut, kept, "a cut conversation")
    finally:
        srv.close()


def check_served_refusals(model):
    """Every refusal of an image on the routes, with 400 and its reason, and a server without --mmproj refusing images."""
    case = CASES[1]
    plain = Server(model, "--dtype", "f32")
    try:
        status, err = plain.post("/v1/chat", ask([user("a", ["rgb-2x3.png"])]))
        assert status == 400 and "--mmproj" in err["error"], err
    finally:
        plain.close()
    srv = image_server(model, case)
    try:
        def part(url, **extra):
            return [{"role": "user", "content": [{"type": "image_url", "image_url": dict({"url": url}, **extra)}, {"type": "text", "text": "a"}]}]
        with open(os.path.join(DATA, "images", "refuse.gif"), "rb") as f:
            gif = base64.b64encode(f.read()).decode("ascii")
        with open(os.path.join(DATA, "images", "refuse-cmyk.jpg"), "rb") as f:
            cmyk = base64.b64encode(f.read()).decode("ascii")
        good = data_url("rgb-2x3.png")
        cases = [(part("https://example.com/a.png"), "data: URL"), (part("data:image/png;base64,iVBOR=w"), "strict base64"),
                 (part("data:image/png;base64,ab!d"), "strict base64"), (part("data:image/gif;base64," + gif), "image/png or image/jpeg"),
                 (part("data:image/png;base64," + gif), "neither PNG nor JPEG"), (part("data:image/jpeg;base64," + cmyk), "an image was refused"),
                 (part(good, detail="high"), "detail"), (part("data:image/png;base64," + "A" * (28 << 20)), "20 MiB"),
                 ([{"role": "user", "content": [{"type": "image_url", "image_url": "x"}]}], "image_url.url"),
                 ([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": good}}, {"type": "text", "text": "Look <|image_pad|> here."}]}],
                  "the prompt marks 2 images and 1 were given")]
        for messages, reason in cases:
            for path in ("/v1/chat", "/v1/chat/completions", "/v1/tokenize"):
                body = {"messages": messages} if path == "/v1/tokenize" else ask(messages)
                status, err = srv.post(path, body)
                message = err["error"]["message"] if isinstance(err["error"], dict) else err["error"]
                assert status == 400 and reason in message, (path, reason, status, err)
        status, err = srv.post("/v1/generate", dict(ask(part(good)), prompt="a"))
        assert status == 400 and "chat routes" in err["error"], err
    finally:
        srv.close()


def run():
    if common.f32_cache_skip("image"):
        return common.SKIPPED
    doc = golden()
    with tempfile.TemporaryDirectory(prefix="llmx_image_") as directory:
        model = write_fixture(directory)
        p = common.run_process(["logits", model, "a", "--chat", "--mmproj", MMPROJ, "--image", os.path.join(DATA, "images", CASES[0]["images"][0])], cache="f32")
        refused = qwen35.refusal(p.returncode, p.stdout.decode("utf-8", "replace") + p.stderr.decode("utf-8", "replace"))
        if refused or "the vision encoder needs" in p.stderr.decode("utf-8", "replace"):
            print("image: SKIP - the device refuses the model or its encoder: %s" % (refused or p.stderr.decode("utf-8", "replace").strip().splitlines()[-1]))
            return common.SKIPPED
        worst = worst_nll = 0.0
        for case in doc["cases"]:
            w, n = check_rows(model, case)
            worst, worst_nll = max(worst, w), max(worst_nll, n)
            check_greedy(model, case)
        check_refusals(model)
        check_served(model, doc)
        check_served_reuse(model)
        check_served_cut(model)
        check_served_refusals(model)
    print("image: %d prompts of 1 and 2 images, every row against HF over threads, passes and decode steps, greedy ids with and without lookup drafts, and served alone,"
          " four at once, reused, cut and refused; max logit error %.8f, NLL error %.8f  [ok]" % (len(doc["cases"]), worst, worst_nll))
    # Flash-Next's path: the image rows in every residual stream, the pooled groups at the image rows' positions, the same passes cutting inside an image and at a group's edge.
    doc = qwen4exp_golden()
    worst = worst_nll = 0.0
    for case in doc["cases"]:
        w, n = check_rows(QWEN4EXP_MODEL, case, QWEN4EXP_MMPROJ)
        worst, worst_nll = max(worst, w), max(worst_nll, n)
        check_greedy(QWEN4EXP_MODEL, case, QWEN4EXP_MMPROJ)
    print("image: qwen4exp, %d prompts of 1 and 2 images, every row against HF over threads, passes and decode steps, and greedy ids; max logit error %.8f, NLL error %.8f  [ok]"
          % (len(doc["cases"]), worst, worst_nll))
    return True
