import hashlib
import json
import math
import os
import re
import tempfile

import common
import qwen35
from f32 import build_byte_vocab, write_model

# A prompt with images through `llmx logits` and `generate` with --chat, --mmproj and --image, against HF Qwen3_5ForConditionalGeneration (tools/gen_baseline.py image-tiny, AGENTS.md, Tests).
# Every prompt row and the greedy ids after it are held to HF's token-by-token cached forward at the F32 bound.
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
GREEDY = 8


def tokens():
    return build_byte_vocab() + SPECIAL


def write_fixture(directory):
    """The text model's GGUF, its weights tests/qwen35.py's for this fixture."""
    path = os.path.join(directory, "tiny-qwen35-image.gguf")
    return write_model(path, qwen35.gguf_tensors(FIXTURE, qwen35.raw_weights(FIXTURE)), chat_template=TEMPLATE, eos_id=EOS,
                       config=qwen35.gguf_config(FIXTURE), arch="qwen35", tokens=tokens(), control=set(range(256, 256 + len(SPECIAL))))


def golden():
    with open(os.path.join(DATA, "baseline_image_tiny.json"), encoding="utf-8") as f:
        doc = json.load(f)
    assert doc["fixture"] == FIXTURE and doc["template"] == TEMPLATE, "the image fixture changed"
    assert [{k: c[k] for k in ("text", "images", "max_pixels")} for c in doc["cases"]] == CASES, "the image cases changed"
    with open(MMPROJ, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == doc["mmproj_sha256"], "the image fixture's mmproj is not the one its goldens were made from"
    return doc


def image_args(case):
    args = ["--chat", "--dtype", "f32", "--mmproj", MMPROJ, "--image", ",".join(os.path.join(DATA, "images", f) for f in case["images"])]
    return args + (["--image-max-pixels", str(case["max_pixels"])] if case["max_pixels"] else [])


def check_rows(model, case):
    """Every prompt row's logits within the F32 bound of HF's, the prompt's mean NLL within its bound, at 1 and 4 threads and passes of 3 and 16 rows, each the bytes of one pass.
    Returns the worst logit and NLL errors."""
    worst, worst_nll, first = 0.0, 0.0, None
    n = len(case["ids"])
    for threads, ubatch in ((1, 1024), (4, 1024), (1, 3), (4, 16)):
        out, bounds = common.run_hf(["logits", model, case["text"]] + image_args(case) +
                                    ["--last", str(n), "--top", str(FIXTURE["vocab"]), "--threads", str(threads), "--ubatch", str(ubatch)])
        lines = out.splitlines()
        assert lines[0] == "tokens: %d" % n, "image %r: llmx read %s, HF %d tokens" % (case["text"], lines[0], n)
        rows = lines[1:]
        assert [int(r.split()[0]) for r in rows] == list(range(n)), "image %r: positions printed" % case["text"]
        if first is None:
            first = rows
        assert rows == first, "image %r: threads %d ubatch %d differ from one pass" % (case["text"], threads, ubatch)
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


def check_greedy(model, case):
    """Greedy decode after the prompt gives HF's ids, the end of text left out of both."""
    for threads, ubatch in ((1, 1024), (4, 3)):
        p = common.run_process(["generate", model, case["text"]] + image_args(case) +
                               ["-n", str(GREEDY), "--temp", "0", "--ignore-eos", "--verbose", "--threads", str(threads), "--ubatch", str(ubatch)], cache="f32")
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
    print("image: %d prompts of 1 and 2 images, every row against HF over threads and passes, greedy ids; max logit error %.8f, NLL error %.8f  [ok]"
          % (len(doc["cases"]), worst, worst_nll))
    return True
