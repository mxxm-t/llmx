"""Generate golden fixtures for the correctness baseline.

Run this ONCE on a machine with the HF tooling, then commit the output.
tests/baseline.py only reads the committed JSON, so running the test suite never needs torch, transformers, or network access -- llmx stays dependency-free at runtime and the suite stays self-contained.

    python tools/gen_baseline.py [all|tokenizer|logits|perplexity|f32|moe|moe-q8|mxfp4|tokenizer-qwen35|qwen35-tiny|qwen35-mtp]
    python tools/gen_baseline.py qwen35 --model Qwen3.5-0.8B|Qwen3.5-4B [--output-dir DIR]
    python tools/gen_baseline.py file-exact --weights-gguf FILE.gguf --output-dir DIR

Defaults use the pinned Qwen3-0.6B reference.
Another model requires --repo, --revision (full commit SHA), --output-dir, --gguf-repo and --gguf-file.
The GGUF arguments are labels, not proof of the converted model's provenance.
Real-model logits/PPL use CPU float32 eager attention and --threads (default 6).
qwen35 writes the logit, chat and PPL goldens of a pinned Qwen3.5 checkpoint (QWEN35_MODELS) into tests/data/<its directory>, and is not part of all.
file-exact writes the logit and PPL goldens of the reference model holding a qwen3, qwen3moe or qwen35 GGUF file's own weights, every tensor decoded to f32 by tests/spec_decode.py and, for qwen35, the converter's changes undone, so llmx can be held on that file to Q8_0-class bounds.
The independent synthetic f32, moe and mxfp4 fixtures use one thread; only --output-dir applies to those modes.
moe-q8 writes the goldens of tests/moe.py's Q8_0 model, HF holding each variant's file's own weights as tests/spec_decode.py decodes them, with one thread; it needs numpy, takes only --output-dir, and is not part of all.
all includes f32 and moe regardless of --repo; mxfp4 is generated explicitly.
tokenizer-qwen35 writes the qwen35 tokenizer golden from its own pinned tokenizer.json and tokenizer_config.json, takes only --output-dir, and is not part of all.
qwen35-tiny writes the goldens of the tiny qwen35 fixtures of tests/qwen35.py from HF Qwen3_5ForCausalLM's token-by-token cached forward, takes only --output-dir, and is not part of all.
qwen35-mtp writes the draft goldens of the tiny fixture with an MTP block from an assembled HF reference, HF's own full-attention Qwen3_5DecoderLayer with the block's input norms, fc, final norm and the target's head around it, since HF drops mtp.*; it takes only --output-dir and is not part of all.

Requires: tokenizers, huggingface_hub (tokenizer goldens) and, for the logit/PPL goldens, torch + transformers.
Those two segfault together in some environments (any `from transformers import Auto*` dies); an isolated venv with numpy<2.3, torch 2.5.1+cpu and transformers 4.55.2 is known to work.
file-exact also needs numpy.
The qwen35 goldens come from a second isolated venv, so the first stays as it is: Python 3.12.13 with torch 2.5.1+cpu, transformers 5.17.0, tokenizers 0.23.2, huggingface_hub 1.33.0, safetensors 0.8.0, numpy 2.2.6 and Jinja2 3.1.6.
"""

import argparse
import contextlib
import hashlib
import importlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace

TOKENIZER_REPO = "Qwen/Qwen3-0.6B"
GGUF_REPO = "Qwen/Qwen3-0.6B-GGUF"
GGUF_FILE = "Qwen3-0.6B-Q8_0.gguf"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "tests", "data")
# The synthetic fixtures' configurations, weights and texts come from the test modules that check them.
sys.path.insert(0, os.path.join(ROOT, "tests"))
# The qwen35 references run in the chat renderer's environment, whose transformers version tools/gen_chat_baseline.py pins.
sys.path.insert(0, os.path.join(ROOT, "tools"))
from gen_chat_baseline import REFERENCE as CHAT_REFERENCE
REFERENCE_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"

# Each case targets a class of pretokenizer behaviour.
# The multi-space and indentation cases hold the GPT-2 rule `\s+(?!\S)`: a run of two or more spaces leaves its last space to attach to the following word.
CASES = [
    "hello world",
    "The capital of France is Paris.",
    "  hello",
    "   spaced   out   ",
    "\ttabbed",
    "trailing   ",
    "a\n\nb",
    "don't can't won't",
    "Hello, World! 123 456",
    "3.14159 and 1e-9",
    "cafe naive jalapeno",
    "\u00e9\u00e8\u00ea \u65e5\u672c\u8a9e \u0440\u0443\u0441\u0441\u043a\u0438\u0439",
    "\U0001f600 emoji \U0001f680",
    "def f(x):\n    if x > 0:\n        return x\n    return -x",
    "<|im_start|>user\nhi<|im_end|>",
    "MixedCASE_snake_case-kebab.dot",
    "\u00ad",
    "co\u00adoperate",
    "\u4e2d\u6587",
    "\u036d",
]

# The qwen35 tokenizer golden reads Qwen3.5-0.8B's tokenizer.json, whose vocabulary and merges every Qwen3.5, 3.6 and 3.8 file holds, and its tokenizer_config.json, which adds the control tokens those files hold beyond it.
# Each file must have the digest the committed golden records.
QWEN35_REPO = "Qwen/Qwen3.5-0.8B"
QWEN35_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
QWEN35_SHA256 = {
    "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    "tokenizer_config.json": "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c",
}
# The cases above, then Thai and Devanagari, whose combining marks are where qwen35's pretokenizer differs from qwen2's, CJK punctuation, and every token HF's tokenizer adds, between words, side by side and inside a word.
QWEN35_CASES = CASES + [
    "\u0e17\u0e48\u0e32\u0e19\u0e1c\u0e39\u0e49\u0e2b\u0e0d\u0e34\u0e07",
    "\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35\u0e04\u0e23\u0e31\u0e1a \u0e22\u0e34\u0e19\u0e14\u0e35\u0e17\u0e35\u0e48\u0e44\u0e14\u0e49\u0e23\u0e39\u0e49\u0e08\u0e31\u0e01",
    "\u0e20\u0e32\u0e29\u0e32\u0e44\u0e17\u0e22\u0e40\u0e1b\u0e47\u0e19\u0e20\u0e32\u0e29\u0e32\u0e17\u0e35\u0e48\u0e2a\u0e27\u0e22\u0e07\u0e32\u0e21 "
    "\u0e41\u0e25\u0e30\u0e21\u0e35\u0e27\u0e23\u0e23\u0e13\u0e22\u0e38\u0e01\u0e15\u0e4c",
    "\u0e23\u0e32\u0e04\u0e32 125 \u0e1a\u0e32\u0e17 (\u0e1b\u0e23\u0e30\u0e21\u0e32\u0e13)",
    "\u0928\u092e\u0938\u094d\u0924\u0947 \u0926\u0941\u0928\u093f\u092f\u093e",
    "\u092f\u0939 \u090f\u0915 \u092a\u0930\u0940\u0915\u094d\u0937\u0923 \u0935\u093e\u0915\u094d\u092f \u0939\u0948\u0964",
    "\u0939\u093f\u0928\u094d\u0926\u0940 \u092e\u0947\u0902 \u0915\u094d\u0937\u0924\u094d\u0930\u093f\u092f \u0914\u0930 \u091c\u094d\u091e\u093e\u0928\u0964 "
    "\u0915\u094d\u092f\u093e \u0906\u092a \u0920\u0940\u0915 \u0939\u0948\u0902?",
    "\u4f60\u597d\uff0c\u4e16\u754c\uff01\u8fd9\u662f\u4e00\u4e2a\u6d4b\u8bd5\u3002",
    "\u300c\u3053\u3093\u306b\u3061\u306f\u300d\u3068\u8a00\u3044\u307e\u3057\u305f\u3002",
    "\u3010\u6ce8\u610f\u3011\u8bf7\u9605\u8bfb\u300a\u7528\u6237\u624b\u518c\u300b\uff08\u7b2c\u4e8c\u7248\uff09\uff1a"
    "\u7b2c\u4e09\u7ae0\u3001\u7b2c\u56db\u7ae0\uff1b\u8c22\u8c22\u2026\u2026",
    "\u597d\u7684\u3002\n\n\u4e0b\u4e00\u6b65\uff1f\u300e\u5b8c\u6210\u300f",
    "\u4f60\u597d\uff0c\u3002\u4e16\u754c",
    "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\nHello!<|im_end|><|endoftext|>",
    "<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n</tool_call><tool_response>ok</tool_response>",
    "<|vision_start|><|image_pad|><|video_pad|><|vision_end|><|object_ref_start|>x<|object_ref_end|>"
    "<|box_start|>(1,2)<|box_end|><|quad_start|><|quad_end|><|vision_pad|>",
    "<|fim_prefix|>def f():<|fim_suffix|>\n<|fim_middle|>    return 1<|fim_pad|><|repo_name|>r<|file_sep|>a.py",
    "a<think>b</think>c <|im_start and im_end|>",
]

# Prompts for the logit golden.
# Plain ASCII and short, so the fixture stays small and the comparison is about the forward pass rather than tokenization, which the tokenizer golden already covers.
LOGIT_PROMPTS = [
    "The capital of France is",
    "Machine learning is",
    "def add(a, b):",
    "The three primary colors are red,",
    "In 1969, humans first walked on the",
    "The capital of France is Paris. The capital of Italy is Rome. The capital of",
]
TOPN = 10

# The perplexity golden's text is the first PPL_CHARS characters of wiki.test.raw, scored whole and then in the (context, most windows) cases of PPL_WINDOWS, 0 meaning no limit.
PPL_CHARS = 1024
PPL_WINDOWS = ((64, 0), (64, 2), (123, 0))


def ppl_text(chars=PPL_CHARS):
    """The first `chars` characters of wiki.test.raw, its line endings read as LF, the text every perplexity golden scores."""
    with open(os.path.join(ROOT, "tests", "data", "wiki.test.raw"), encoding="utf-8") as f:
        return f.read(chars)


def ppl_window_bounds(n_tokens, context, limit):
    """The [start, end) token spans a (context, limit) case scores in a text of `n_tokens`: disjoint windows of `context` in order, at most `limit` of them unless it is 0, ending before the first shorter than 2 tokens."""
    bounds = []
    for start in range(0, n_tokens, context):
        end = min(start + context, n_tokens)
        if end - start < 2 or (limit and len(bounds) >= limit):
            break
        bounds.append((start, end))
    return bounds


def gen_tokenizer(args, tokenizer_file=None):
    from tokenizers import Tokenizer
    if tokenizer_file is None:
        from huggingface_hub import hf_hub_download
        tokenizer_file = hf_hub_download(args.repo, "tokenizer.json", revision=args.revision)
    tok = Tokenizer.from_file(tokenizer_file)
    cases = [{"text": t, "ids": tok.encode(t, add_special_tokens=False).ids}
             for t in CASES]
    doc = {
        "_comment": "Generated by tools/gen_baseline.py. Do not hand-edit.",
        "tokenizer_repo": args.repo,
        "tokenizer_revision": args.revision,
        "gguf_repo": args.gguf_repo,
        "gguf_file": args.gguf_file,
        "cases": cases,
    }
    path = os.path.join(args.output_dir, "baseline_tokenizer.json")
    _write(path, doc)
    print("wrote %s (%d cases)" % (path, len(cases)))


def qwen35_tokenizer_file(name):
    """The path and parsed contents of one pinned qwen35 tokenizer file, refused unless it has the pinned digest."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(QWEN35_REPO, name, revision=QWEN35_REVISION)
    with open(path, "rb") as f:
        raw = f.read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != QWEN35_SHA256[name]:
        raise SystemExit("qwen35: %s at %s has SHA-256 %s, not the pinned %s" % (name, QWEN35_REVISION, digest, QWEN35_SHA256[name]))
    return path, json.loads(raw)


def gen_tokenizer_qwen35(output_dir):
    """HF's ids for QWEN35_CASES from the pinned tokenizer.json, with the part of its vocabulary those texts reach, so tests/tokenizer.py checks them without a model file.
    The control tokens only tokenizer_config.json adds are kept apart, with the GGUF type of every added token."""
    import unicodedata
    import tokenizers
    from tokenizers import Tokenizer
    from tokenizers.pre_tokenizers import ByteLevel

    path, spec = qwen35_tokenizer_file("tokenizer.json")
    config = qwen35_tokenizer_file("tokenizer_config.json")[1]
    tok = Tokenizer.from_file(path)
    byte_level = ByteLevel(add_prefix_space=False, use_regex=False)
    # BPE only joins neighbours within one pretokenized piece, and every piece is a substring of its text however the text is cut.
    # So the merges that form a substring of a text, in rank order with the tokens they form, give the ids the whole vocabulary gives, whatever the pretokenizer.
    reach = set()
    for text in QWEN35_CASES:
        for form in {text, unicodedata.normalize("NFC", text)}:
            mapped = byte_level.pre_tokenize_str(form)[0][0]
            reach.update(mapped[i:j] for i in range(len(mapped)) for j in range(i + 2, len(mapped) + 1))
    assert all(isinstance(m, str) and m.count(" ") == 1 for m in spec["model"]["merges"])
    merges = [m for m in spec["model"]["merges"] if m.replace(" ", "") in reach]
    vocab = spec["model"]["vocab"]
    tokens = {vocab[t]: t for t in ByteLevel.alphabet()}
    tokens.update((vocab[m.replace(" ", "")], m.replace(" ", "")) for m in merges)
    tokens.update((a["id"], a["content"]) for a in spec["added_tokens"])
    # tokenizer_config.json adds control tokens that tokenizer.json lacks, and the GGUF files hold them too.
    # transformers' tokenizer encodes each as its one id, so they are kept apart from the tokens whose ids come from tokenizer.json.
    added = {a["id"]: a for a in spec["added_tokens"]}
    config_tokens = {int(i): a for i, a in config["added_tokens_decoder"].items() if int(i) not in added}
    assert all(added[int(i)]["content"] == a["content"] for i, a in config["added_tokens_decoder"].items() if int(i) in added)
    assert not set(config_tokens) & set(vocab.values())
    # The GGUF token types of the added tokens, as the Qwen3.5 files give them: control (3) for HF's special ones and for any written <|name|>, which takes in the fim and repo tokens HF leaves unspecial, and user-defined (4) for the rest.
    types = {i: 3 if a["special"] or (a["content"].startswith("<|") and a["content"].endswith("|>")) else 4
             for i, a in list(added.items()) + list(config_tokens.items())}
    cases = [{"text": t, "ids": tok.encode(t, add_special_tokens=False).ids} for t in QWEN35_CASES]
    assert all(i in tokens for case in cases for i in case["ids"])
    doc = {
        "_comment": "Generated by tools/gen_baseline.py tokenizer-qwen35. Do not hand-edit.",
        "tokenizer_repo": QWEN35_REPO,
        "tokenizer_revision": QWEN35_REVISION,
        "tokenizer_json_sha256": QWEN35_SHA256["tokenizer.json"],
        "tokenizer_config_json_sha256": QWEN35_SHA256["tokenizer_config.json"],
        "tokenizers_version": tokenizers.__version__,
        "tokens": {str(i): tokens[i] for i in sorted(tokens)},
        "config_tokens": {str(i): config_tokens[i]["content"] for i in sorted(config_tokens)},
        "token_types": {str(i): types[i] for i in sorted(types)},
        "merges": merges,
        "cases": cases,
    }
    path = os.path.join(output_dir, "baseline_tokenizer_qwen35.json")
    _write(path, doc)
    print("wrote %s (%d cases, %d tokens, %d merges, %d config tokens)" % (path, len(cases), len(tokens), len(merges), len(config_tokens)))


def load_reference(args):
    if args.family == "qwen35":
        return load_qwen35(args)
    if args.weights_gguf:
        import spec_decode
        if spec_decode.GGUF(args.weights_gguf).value("general.architecture") == "qwen3moe":
            qwen3moe_environment()
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(args.threads)
    tok = AutoTokenizer.from_pretrained(args.repo, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.repo, revision=args.revision, torch_dtype=torch.float32,
        attn_implementation="eager")
    model.cpu().eval()
    if args.weights_gguf:
        load_gguf_weights(args, torch, model)
    return torch, transformers, tok, model


def qwen35_names():
    """The HF Qwen3.5 text parameter each qwen35 GGUF tensor holds, and what the converter did to it, from the one map tests/qwen35.py writes the tiny models with (docs/QWEN35.md, GGUF conventions).
    Returns {GGUF name: (HF name, transform)} for the tensors outside the layers and {GGUF name within a block: (HF name within a layer, transform)}, where each of the halves a routed layer's gate_up_proj is stored as names that parameter."""
    from qwen35 import BLOCK_TENSORS, OTHER_TENSORS
    return ({gguf: (hf, kind) for hf, (gguf, kind) in OTHER_TENSORS.items() if not gguf.startswith("blk.")},
            {part: (hf, kind) for hf, (gguf, kind) in BLOCK_TENSORS.items() for part in (gguf if isinstance(gguf, tuple) else (gguf,))})


def take_rows(values, cols, order, numpy):
    """The rows of a flat row-major matrix `cols` wide, in `order`."""
    if numpy:
        return values.reshape(-1, cols)[order].reshape(-1)
    return [v for r in order for v in values[r * cols:(r + 1) * cols]]


def take_columns(values, cols, order, numpy):
    """The columns of a flat row-major matrix `cols` wide, in `order`."""
    if numpy:
        return values.reshape(-1, cols)[:, order].reshape(-1)
    return [values[r * cols + c] for r in range(len(values) // cols) for c in order]


def qwen35_tensors(model, numpy, tensors=None):
    """The tensors of the qwen35 or qwen35moe GGUF `model` as HF holds them, with the converter's changes undone: (HF name, HF shape, flat values, type name).
    Only `tensors`, a list of the file's tensors, are decoded when it is given; a routed layer's gate and up stacks give one gate_up_proj, where its gate stack is listed.
    The MTP block, which HF has no module for, is left out."""
    import spec_decode
    from qwen35 import BLOCK_TENSORS, tiled_order
    np = __import__("numpy") if numpy else None
    names, block_names = qwen35_names()
    arch = model.value("general.architecture")
    hk, hv, dk = (model.value(arch + ".ssm." + k) for k in ("group_count", "time_step_rank", "state_size"))
    dv = model.value(arch + ".ssm.inner_size") // hv
    layers = model.value(arch + ".block_count") - model.value(arch + ".nextn_predict_layers", 0)
    by_name = {t.name: t for t in model.tensors}
    if hv % hk:
        raise SystemExit("%s has %d V heads over %d K heads" % (model.path, hv, hk))
    # The file's head j holds HF's head tiled_order[j], so HF's head i is the file's head heads[i].
    order = tiled_order(hk, hv)
    heads = sorted(range(hv), key=order.__getitem__)

    def v_rows(block, offset=0):
        return list(range(offset)) + [offset + heads[i] * block + j for i in range(hv) for j in range(block)]

    for t in model.tensors if tensors is None else tensors:
        match = re.fullmatch(r"blk\.(\d+)\.(.+)", t.name)
        if match and int(match[1]) >= layers:
            continue
        hf, kind = block_names.get(match[2], (None, None)) if match else names.get(t.name, (None, None))
        if hf is None:
            raise SystemExit("GGUF tensor %s has no HF Qwen3.5 parameter" % t.name)
        if kind == "halves":
            # HF's gate_up_proj is [E, 2I, H], each expert's gate rows then its up rows, and the file stacks the two halves apart.
            parts = [by_name["blk.%s.%s" % (match[1], part)] for part in BLOCK_TENSORS[hf][0]]
            if t is not parts[0]:
                continue
            experts, rows, width = reversed(parts[0].shape)
            halves = [model.decode(part, numpy=numpy) for part in parts]
            size = rows * width
            values = (np.concatenate([h.reshape(experts, rows, width) for h in halves], axis=1).reshape(-1) if numpy else
                      [v for e in range(experts) for h in halves for v in h[e * size:(e + 1) * size]])
            yield "model.layers.%s.%s" % (match[1], hf), [experts, 2 * rows, width], values, spec_decode.type_name(t.type)
            continue
        hf = "model.layers.%s.%s" % (match[1], hf) if match else hf
        shape = list(reversed(t.shape))
        values = model.decode(t, numpy=numpy)
        cols = shape[1] if len(shape) == 2 else 1
        if kind == "vector":
            shape = [1, shape[0]]
        elif kind == "norm":
            values = values - np.float32(1) if numpy else [v - 1.0 for v in values]
        elif kind == "a":
            values = take_rows(values, 1, heads, numpy)
            values = np.log(-values.astype(np.float64)).astype(np.float32) if numpy else [math.log(-v) for v in values]
        elif kind == "heads":
            values = take_rows(values, cols, v_rows(shape[0] // hv), numpy)
        elif kind in ("channels", "conv"):
            values = take_rows(values, cols, v_rows(dv, 2 * hk * dk), numpy)
            shape = [shape[0], 1, shape[1]] if kind == "conv" else shape
        elif kind == "columns":
            values = take_columns(values, cols, v_rows(dv), numpy)
        yield hf, shape, values, spec_decode.type_name(t.type)


def qwen3moe_tensors(model, numpy, tensors=None):
    """Qwen3-MoE parameters, splitting each expert-major stack into views of its decoded values; routers keep their own HF names."""
    import spec_decode
    from f32 import hf_name
    seen = set()
    for t in model.tensors if tensors is None else tensors:
        if t.name in seen:
            raise SystemExit("duplicate GGUF tensor %s" % t.name)
        seen.add(t.name)
        shape = list(reversed(t.shape))
        values = model.decode(t, numpy=numpy)
        kind = spec_decode.type_name(t.type)
        experts = re.fullmatch(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight", t.name)
        if experts:
            if len(shape) != 3:
                raise SystemExit("%s: an expert stack needs three dimensions" % t.name)
            size = math.prod(shape[1:])
            for e in range(shape[0]):
                yield ("model.layers.%s.mlp.experts.%d.%s_proj.weight" % (experts[1], e, experts[2]),
                       shape[1:], values[e * size:(e + 1) * size], kind)
        else:
            router = re.fullmatch(r"blk\.(\d+)\.ffn_gate_inp\.weight", t.name)
            yield "model.layers.%s.mlp.gate.weight" % router[1] if router else hf_name(t.name), shape, values, kind


def gguf_tensors(path, numpy=True):
    """A supported GGUF file's tensors as the HF parameters they hold, each decoded to f32 by tests/spec_decode.py, in file order: (HF name, HF shape, flat values, type name).
    GGUF lists a matrix's dimensions fastest first, so the HF shape is the reverse; qwen3 files store the projections unpermuted, and qwen35 ones come back with the converter's changes undone."""
    import spec_decode
    model = spec_decode.GGUF(path)
    architecture = model.value("general.architecture")
    if architecture == "qwen3moe":
        yield from qwen3moe_tensors(model, numpy)
        return
    if architecture in ("qwen35", "qwen35moe"):
        yield from qwen35_tensors(model, numpy)
        return
    if architecture != "qwen3":
        raise SystemExit("%s holds a %s model; file-exact references map qwen3, qwen3moe, qwen35 and qwen35moe tensors only" % (path, architecture))
    from f32 import hf_name
    for t in model.tensors:
        yield hf_name(t.name), list(reversed(t.shape)), model.decode(t, numpy=numpy), spec_decode.type_name(t.type)


def gguf_state(path, numpy=True):
    """gguf_tensors as {HF name: (HF shape, flat values)}, and {type name: tensor count}."""
    state, types = {}, {}
    for key, shape, values, type_name in gguf_tensors(path, numpy):
        state[key] = (shape, values)
        types[type_name] = types.get(type_name, 0) + 1
    return state, types


def load_gguf_weights(args, torch, model):
    """Replace every parameter of `model` with the GGUF file's own weights, one tensor at a time, and record in args.weights which file they came from."""
    from baseline import pinned_fixture
    from baseline_8b import file_sha256
    name = os.path.basename(args.weights_gguf)
    sha256 = file_sha256(Path(args.weights_gguf))
    pinned = pinned_fixture(name)
    if pinned and pinned["sha256"] != sha256:
        raise SystemExit("%s is named like a pinned fixture but does not match its SHA-256" % args.weights_gguf)
    # A tied head is the embedding's parameter, which named_parameters gives once, under the embedding's name.
    params = dict(model.named_parameters())
    types, equal, loaded = {}, 0, set()
    with torch.no_grad():
        for key, shape, values, type_name in gguf_tensors(args.weights_gguf):
            if key not in params:
                raise SystemExit("%s holds %s, which the reference model %s" % (name, key, "ties to the embedding" if key == "lm_head.weight" else "does not have"))
            tensor = torch.from_numpy(values).reshape(shape)
            if params[key].shape != tensor.shape:
                raise SystemExit("%s: %s is %s in the file and %s in the reference" % (name, key, list(tensor.shape), list(params[key].shape)))
            # How many tensors are bit for bit the reference checkpoint's, as those of a file converted without rounding are.
            same = torch.equal(params[key], tensor)
            # A qwen35 file whose dt_bias is not the checkpoint's has its V heads in another order than the tiled one (docs/QWEN35.md).
            if key.endswith("linear_attn.dt_bias") and not same:
                raise SystemExit("%s: %s is not the reference's in the tiled V-head order" % (name, key))
            equal += same
            params[key].copy_(tensor)
            loaded.add(key)
            types[type_name] = types.get(type_name, 0) + 1
    if loaded != set(params):
        raise SystemExit("%s lacks %s" % (name, ", ".join(sorted(set(params) - loaded))))
    args.weights = {"file": name, "sha256": sha256, "bytes": os.path.getsize(args.weights_gguf),
                    "repo": pinned["repo"] if pinned else None, "revision": pinned["revision"] if pinned else None,
                    "types": dict(sorted(types.items())), "decoder": "tests/spec_decode.py, numpy form",
                    "tensors": len(loaded), "tensors_equal_to_reference": equal}


def reference_metadata(args, torch, transformers):
    metadata = {"reference_repo": args.repo, "reference_revision": args.revision,
                "reference_dtype": "float32", "attention": "eager", "device": "cpu",
                "threads": args.threads, "torch_version": torch.__version__,
                "transformers_version": transformers.__version__}
    # Only the loader of a pinned qwen35 checkpoint records its files, tokenizer and linear-attention form; the layered reference's writer arguments carry none.
    metadata.update(getattr(args, "qwen35", None) or {})
    if args.weights_gguf:
        metadata["weights"] = args.weights
    return metadata


def gen_logits(args, loaded=None):
    torch, transformers, tok, model = loaded or load_reference(args)
    cases = []
    for p in LOGIT_PROMPTS:
        ids = tok(p, add_special_tokens=False, return_tensors="pt").input_ids
        with torch.inference_mode():
            lg = model(ids, use_cache=False).logits[0, -1].float()
        top = torch.topk(lg, TOPN)
        cases.append({"text": p,
                      "token_ids": ids[0].tolist(),
                      "n_tokens": int(ids.shape[1]),
                      "top_ids": top.indices.tolist(),
                      "top_logits": [round(float(v), 4) for v in top.values]})
        print("  %-46s -> %s" % (repr(p)[:46], top.indices.tolist()[:3]))
    doc = {
        "_comment": ("Generated by tools/gen_baseline.py file-exact from the reference model holding "
                     "the GGUF file's own weights as tests/spec_decode.py decodes them."
                     if args.weights_gguf else
                     "Generated by tools/gen_baseline.py from the FULL-PRECISION "
                     "reference. Quantized GGUF logits can differ; establish "
                     "model-specific rank and numerical bounds independently."),
        **reference_metadata(args, torch, transformers),
        "gguf_repo": args.gguf_repo,
        "gguf_file": args.gguf_file,
        "topn": TOPN,
        "cases": cases,
    }
    path = os.path.join(args.output_dir, "baseline_logits.json")
    _write(path, doc)
    print("wrote %s (%d prompts)" % (path, len(cases)))


def gen_perplexity(args, loaded=None):
    torch, transformers, tok, model = loaded or load_reference(args)
    text = ppl_text()
    ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids
    with torch.inference_mode():
        logits = model(ids, use_cache=False).logits[0, :-1].double()
        targets = ids[0, 1:]
        nll = torch.logsumexp(logits, dim=-1) - logits.gather(1, targets[:, None]).squeeze(1)
        mean_nll = nll.mean().item()
    chunk_cases = []
    for context, limit in PPL_WINDOWS:
        total_nll, used, scored, chunks = 0.0, 0, 0, 0
        for start, end in ppl_window_bounds(ids.shape[1], context, limit):
            window = ids[:, start:end]
            count = window.shape[1]
            with torch.inference_mode():
                logits = model(window, use_cache=False).logits[0, :-1].double()
                targets = window[0, 1:]
                loss = torch.logsumexp(logits, dim=-1) - logits.gather(1, targets[:, None]).squeeze(1)
            total_nll += loss.sum().item()
            used += count
            scored += count - 1
            chunks += 1
        chunk_cases.append({"context_size": context, "max_chunks": limit,
                            "used_tokens": used, "n_scored": scored, "chunks": chunks,
                            "mean_nll": total_nll / scored,
                            "perplexity": math.exp(total_nll / scored)})
    doc = {
        "_comment": "Generated by tools/gen_baseline.py %s. Do not hand-edit." % ("file-exact" if args.weights_gguf else "perplexity"),
        **reference_metadata(args, torch, transformers),
        "source": "wiki.test.raw, first 1024 Unicode characters after CRLF/CR normalization to LF",
        "scoring": "One continuous sequence, no BOS/EOS added; score tokens 1..N-1 from preceding tokens; float64 log-softmax/reduction of fp32 logits.",
        "text": text,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "token_ids": ids[0].tolist(),
        "n_tokens": ids.shape[1],
        "n_scored": ids.shape[1] - 1,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll),
        "chunk_scoring": "Tokenize once; disjoint windows with reset positions/KV; score tokens 1..L-1 per window; include partial windows of >=2 tokens; omit singleton tail; aggregate total NLL / total targets.",
        "chunk_cases": chunk_cases,
    }
    path = os.path.join(args.output_dir, "baseline_perplexity.json")
    _write(path, doc)
    print("wrote %s (%d tokens, mean NLL %.9f, PPL %.9f)"
          % (path, doc["n_tokens"], mean_nll, doc["perplexity"]))


# The versions every qwen35 reference is made with, the second reference environment of docs/ASSETS.md, which the chat renderer shares; another is refused, since only the recorded fields would show the change.
QWEN35_ENV = {"torch": "2.5.1+cpu", "transformers": CHAT_REFERENCE["transformers"], "tokenizers": "0.23.2"}
# HF's qwen3_5 code runs the linear-attention or conv package in place of its own torch functions whenever one is installed, and a hub kernel whenever a load asks the kernels package for one, so none may be installed.
QWEN35_REPLACEMENTS = ("kernels", "fla", "causal_conv1d")
# The checkpoint keys Qwen3_5ForCausalLM may leave unused: the MTP block and the vision tower, which it drops at load.
QWEN35_UNUSED = re.compile(r"(?:mtp|model\.visual)\.")


def qwen35_runtime():
    """torch, transformers and HF's qwen3_5 modeling module, refused unless none of QWEN35_REPLACEMENTS is installed and the versions are QWEN35_ENV's."""
    for name in QWEN35_REPLACEMENTS:
        if importlib.util.find_spec(name) is not None:
            raise SystemExit("qwen35: the %s package is installed, and HF would run it in place of its torch functions" % name)
    import torch
    import tokenizers
    import transformers
    found = {"torch": torch.__version__, "transformers": transformers.__version__, "tokenizers": tokenizers.__version__}
    if found != QWEN35_ENV:
        raise SystemExit("qwen35: the references are made with %s; this environment has %s" % (QWEN35_ENV, found))
    return torch, transformers, importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")


def qwen35_environment():
    """qwen35_runtime with the Hub and transformers kept offline, for the references that read only local files."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return qwen35_runtime()


def qwen3moe_environment():
    """The existing tiny-MoE reference environment, kept offline for file-exact generation."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    import transformers
    import numpy
    for name, module, expected in (("torch", torch, "2.5.1+cpu"), ("transformers", transformers, "4.55.2"),
                                   ("numpy", numpy, "2.2.6")):
        if module.__version__ != expected:
            raise SystemExit("qwen3moe reference requires %s %s, found %s" % (name, expected, module.__version__))
    return torch, transformers


# The qwen35 checkpoints whose real-model goldens `qwen35` writes from HF's full forward in float32.
# Each gives its repository and commit with the SHA-256 of every file the reference reads, the pinned GGUF files of tests/data/fixtures.json the goldens are for, the SHA-256 of the chat template those files carry, whose text tests/data/baseline_chat_template.json holds, and the directory under tests/data the goldens go to.
QWEN35_MODELS = {
    "Qwen3.5-0.8B": {
        "repo": QWEN35_REPO, "revision": QWEN35_REVISION,
        "sha256": {"config.json": "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
                   "model.safetensors.index.json": "d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e",
                   "model.safetensors-00001-of-00001.safetensors": "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"},
        "gguf_files": ["Qwen3.5-0.8B-Q8_0.gguf", "Qwen3.5-0.8B-Q4_K_M.gguf"],
        "template_sha256": "7f0e529032c25183bcd66c7f238da2d377f43be754a94e2725a58c4e16d2ed67", "directory": "qwen35-0.8b"},
    "Qwen3.5-4B": {
        "repo": "Qwen/Qwen3.5-4B", "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "sha256": {"config.json": "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670",
                   "model.safetensors.index.json": "cf3f798ee02ba45f9622aa8892a47369ab667d0afbf154ee7c2212de42e6302d",
                   "model.safetensors-00001-of-00002.safetensors": "26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61",
                   "model.safetensors-00002-of-00002.safetensors": "cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188",
                   "tokenizer.json": QWEN35_SHA256["tokenizer.json"]},
        "gguf_files": ["Qwen3.5-4B-Q4_K_M.gguf"],
        "template_sha256": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715", "directory": "qwen35-4b"},
}
# The characters of wiki.test.raw each perplexity golden scores, by its window: past two windows of 512 tokens, which the hosted HF job scores for a file on disk, and past one of 4096, scored by hand.
QWEN35_PPL = {512: 5000, 4096: 20000}
# The chat golden's conversations, each rendered with a generation prompt under the model file's own template.
QWEN35_CHATS = [
    ("one-turn", [{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": "What is the capital of France?"}]),
    ("two-turn", [{"role": "user", "content": "Remember the word violet."},
                  {"role": "assistant", "content": "<think>\nThe user wants a word kept.\n</think>\n\nI will remember violet."},
                  {"role": "user", "content": "Which word did I ask you to remember?"}]),
]


class Qwen35Tokenizer:
    """The tokenizer the qwen35 tokenizer golden holds, called as gen_logits calls an HF one: the pinned tokenizer.json read by the tokenizers library, with the control tokens only tokenizer_config.json adds encoded as one id each."""

    def __init__(self, torch):
        from tokenizers import AddedToken, Tokenizer
        path, spec = qwen35_tokenizer_file("tokenizer.json")
        config = qwen35_tokenizer_file("tokenizer_config.json")[1]
        self.tokenizer, self.torch = Tokenizer.from_file(path), torch
        added = {a["id"] for a in spec["added_tokens"]}
        extra = sorted((int(i), a["content"]) for i, a in config["added_tokens_decoder"].items() if int(i) not in added)
        self.tokenizer.add_special_tokens([AddedToken(content, special=True, normalized=False) for _, content in extra])
        if any(self.tokenizer.token_to_id(content) != i for i, content in extra):
            raise SystemExit("qwen35: the control tokens only tokenizer_config.json adds do not take its ids")

    def __call__(self, text, add_special_tokens=False, return_tensors="pt"):
        return SimpleNamespace(input_ids=self.torch.tensor([self.tokenizer.encode(text, add_special_tokens=add_special_tokens).ids]))


def qwen35_checkpoint(spec):
    """The local directory of a pinned qwen35 checkpoint, each file the reference reads refused unless it has its pinned SHA-256."""
    from huggingface_hub import hf_hub_download
    from baseline_8b import file_sha256
    paths = [hf_hub_download(spec["repo"], name, revision=spec["revision"]) for name in spec["sha256"]]
    for name, path in zip(spec["sha256"], paths):
        digest = file_sha256(Path(path))
        if digest != spec["sha256"][name]:
            raise SystemExit("qwen35: %s at %s has SHA-256 %s, not the pinned %s" % (name, spec["revision"], digest, spec["sha256"][name]))
    return os.path.dirname(paths[0])


def load_qwen35(args):
    """HF's Qwen3_5ForCausalLM on the pinned checkpoint of args.qwen35_model, in float32 with eager attention and transformers' torch forms of the linear attention and conv, and the tokenizer of the qwen35 tokenizer golden.
    The model must take every parameter from the checkpoint, which may hold beyond them only the MTP and vision weights."""
    torch, transformers, modeling_qwen3_5 = qwen35_runtime()
    import tokenizers
    from transformers import Qwen3_5ForCausalLM
    spec = QWEN35_MODELS[args.qwen35_model]
    torch.set_num_threads(args.threads)
    model, info = Qwen3_5ForCausalLM.from_pretrained(qwen35_checkpoint(spec), dtype=torch.float32, attn_implementation="eager",
                                                     output_loading_info=True)
    unexpected = [key for key in info["unexpected_keys"] if not QWEN35_UNUSED.match(key)]
    if info["missing_keys"] or info["mismatched_keys"] or unexpected:
        raise SystemExit("qwen35: %s loads with missing %s, mismatched %s and unexpected %s"
                         % (args.qwen35_model, sorted(info["missing_keys"]), sorted(info["mismatched_keys"]), sorted(unexpected)))
    model.cpu().eval()
    guard_linear_attention(modeling_qwen3_5, args)
    args.qwen35 = {"gguf_files": spec["gguf_files"],
                   "tokenizer": {"repo": QWEN35_REPO, "revision": QWEN35_REVISION, "library": "tokenizers " + tokenizers.__version__,
                                 "tokenizer_json_sha256": QWEN35_SHA256["tokenizer.json"],
                                 "tokenizer_config_json_sha256": QWEN35_SHA256["tokenizer_config.json"]},
                   "linear_attention": "HF's full forward: the chunked form, 64 rows a chunk, in float32"}
    tok = Qwen35Tokenizer(torch)
    if args.weights_gguf:
        load_gguf_weights(args, torch, model)
    return torch, transformers, tok, model


def guard_linear_attention(modeling, args):
    """Count the calls of HF's chunked form into args.chunked_calls and refuse its token-by-token recurrence, since the real-model goldens are HF's full forward."""
    chunked = modeling.torch_chunk_gated_delta_rule
    args.chunked_calls = 0

    def counted(*a, **k):
        args.chunked_calls += 1
        return chunked(*a, **k)

    def refused(*a, **k):
        raise SystemExit("a qwen35 golden reached HF's token-by-token recurrence, not its full forward")

    modeling.torch_chunk_gated_delta_rule, modeling.torch_recurrent_gated_delta_rule = counted, refused


def sequence_nll(model, torch, ids, rows=512):
    """The float64 NLL of tokens 1..n-1 of `ids` (1 x n) from HF's full forward over them, the head applied to `rows` positions at a time so the n x V logits are never held at once."""
    with torch.inference_mode():
        hidden = model.model(ids, use_cache=False).last_hidden_state[0, :-1]
        targets = ids[0, 1:]
        parts = []
        for start in range(0, hidden.shape[0], rows):
            logits = model.lm_head(hidden[start:start + rows]).double()
            parts.append(torch.logsumexp(logits, dim=-1) - logits.gather(1, targets[start:start + rows, None]).squeeze(1))
    return torch.cat(parts)


def gen_perplexity_qwen35(args, loaded, context):
    """The qwen35 perplexity golden for windows of `context` tokens: the whole excerpt as one sequence, then its disjoint windows, scored as gen_perplexity scores them.
    The golden keeps the excerpt's length and the digests of its text and ids, from which tests/baseline_qwen35.py reads it back."""
    from baseline_qwen35 import PPL_GOLDENS, ids_sha256
    torch, transformers, tok, model = loaded
    chars = QWEN35_PPL[context]
    text = ppl_text(chars)
    ids = tok(text).input_ids
    mean_nll = sequence_nll(model, torch, ids).mean().item()
    total_nll, used, scored, chunks = 0.0, 0, 0, 0
    for start, end in ppl_window_bounds(ids.shape[1], context, 0):
        window = ids[:, start:end]
        total_nll += sequence_nll(model, torch, window).sum().item()
        used += window.shape[1]
        scored += window.shape[1] - 1
        chunks += 1
    doc = {
        "_comment": "Generated by tools/gen_baseline.py %s. Do not hand-edit." % args.kind,
        **reference_metadata(args, torch, transformers),
        "source": "wiki.test.raw, first %d Unicode characters after CRLF/CR normalization to LF" % chars,
        "chars": chars,
        "scoring": "One continuous sequence, no BOS/EOS added; score tokens 1..N-1 from preceding tokens; float64 log-softmax/reduction of fp32 logits, the head applied to 512 positions at a time.",
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "n_tokens": ids.shape[1],
        "token_ids_sha256": ids_sha256(ids[0].tolist()),
        "n_scored": ids.shape[1] - 1,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll),
        "chunk_scoring": "Tokenize once; disjoint windows with reset positions/KV; score tokens 1..L-1 per window; include partial windows of >=2 tokens; omit singleton tail; aggregate total NLL / total targets.",
        "chunk_cases": [{"context_size": context, "max_chunks": 0, "used_tokens": used, "n_scored": scored, "chunks": chunks,
                         "mean_nll": total_nll / scored, "perplexity": math.exp(total_nll / scored)}],
    }
    path = os.path.join(args.output_dir, PPL_GOLDENS[context])
    _write(path, doc)
    print("wrote %s (%d tokens, mean NLL %.9f; %d windows of %d, mean NLL %.9f)" % (path, doc["n_tokens"], mean_nll, chunks, context, total_nll / scored))


def gen_chat_qwen35(args, loaded):
    """The chat golden: each conversation of QWEN35_CHATS rendered with a generation prompt by transformers' renderer under the template of the model's files, its ids, and HF's top logits after them."""
    from gen_chat_baseline import BOS, EOS, TEMPLATES, reference
    torch, transformers, tok, model = loaded
    digest = QWEN35_MODELS[args.qwen35_model]["template_sha256"]
    ref = reference()
    template = next((t for t in json.loads(TEMPLATES.read_text(encoding="utf-8"))["templates"] if t["sha256"] == digest), None)
    if template is None or hashlib.sha256(template["template"].encode("utf-8", "surrogateescape")).hexdigest() != digest:
        raise SystemExit("qwen35: %s does not hold the template with SHA-256 %s" % (TEMPLATES, digest))
    cases = []
    for name, messages in QWEN35_CHATS:
        text = ref.render(template["template"], messages, True)
        ids = tok(text).input_ids
        with torch.inference_mode():
            top = torch.topk(model(ids, use_cache=False).logits[0, -1].float(), TOPN)
        cases.append({"name": name, "messages": messages, "generate": True, "text": text, "token_ids": ids[0].tolist(),
                      "n_tokens": int(ids.shape[1]), "top_ids": top.indices.tolist(), "top_logits": [round(float(v), 4) for v in top.values]})
    doc = {
        "_comment": "Generated by tools/gen_baseline.py %s. Do not hand-edit." % args.kind,
        **reference_metadata(args, torch, transformers),
        **ref.versions,
        "renderer": "transformers render_jinja_template, the renderer of apply_chat_template",
        "template_name": template["name"], "template_sha256": digest, "bos_token": BOS, "eos_token": EOS,
        "topn": TOPN,
        "cases": cases,
    }
    path = os.path.join(args.output_dir, "baseline_chat.json")
    _write(path, doc)
    print("wrote %s (%s)" % (path, ", ".join("%s %d tokens" % (case["name"], case["n_tokens"]) for case in cases)))


def _write(path, doc, indent=1):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, ensure_ascii=False, indent=indent)
        f.write("\n")


def load_tiny_weights(model, weights, torch, tied=False):
    """Bind named F32 tensors and expert-major stacks to an HF tiny model, with no missing or extra parameters."""
    state = {}
    for _, name, shape, values in weights:
        if isinstance(name, list):
            per = len(values) // len(name)
            for e, expert in enumerate(name):
                state[expert] = torch.tensor(values[e * per:(e + 1) * per], dtype=torch.float32).reshape(shape[1], shape[0])
        else:
            state[name] = torch.tensor(values, dtype=torch.float32).reshape(list(reversed(shape)))
    if tied:
        state["lm_head.weight"] = state["model.embed_tokens.weight"]
    model.load_state_dict(state, strict=True)


def tiny_qwen3(tied, seed=12345):
    """The tiny dense model of tests/f32.py as an HF Qwen3ForCausalLM with eager attention, holding the weights f32.tensors builds from `seed`; returns the model and those weights."""
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from f32 import CONFIG, VOCAB, tensors

    config = Qwen3Config(vocab_size=VOCAB, hidden_size=CONFIG["embedding_length"],
                         intermediate_size=CONFIG["feed_forward_length"], num_hidden_layers=CONFIG["block_count"],
                         num_attention_heads=CONFIG["attention.head_count"],
                         num_key_value_heads=CONFIG["attention.head_count_kv"], head_dim=CONFIG["attention.key_length"],
                         max_position_embeddings=CONFIG["context_length"], rope_theta=10000.0,
                         rms_norm_eps=1e-6, tie_word_embeddings=tied, attention_dropout=0.0)
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).float().eval()
    weights = tensors(tied, seed)
    load_tiny_weights(model, weights, torch, tied)
    return model, weights


def reference_outputs(model, torch, texts=None, contexts=(4, 16)):
    """A tiny model's full final logits and windowed NLL, using the ordinary short texts and windows unless a fixture supplies longer ones."""
    from f32 import TEXTS
    texts = TEXTS if texts is None else texts

    cases, perplexity = [], []
    with torch.inference_mode():
        for text in texts:
            ids = torch.tensor([list(text.encode("ascii"))])
            logits = model(ids, use_cache=False).logits[0, -1]
            cases.append({"text": text, "logits": logits.tolist()})
        ids = torch.tensor([list(texts[-1].encode("ascii"))])
        for context in contexts:
            total, targets = 0.0, 0
            for window in ids.split(context, dim=1):
                if window.shape[1] < 2:
                    continue
                logits = model(window, use_cache=False).logits[0, :-1].double()
                loss = torch.logsumexp(logits, -1) - logits.gather(1, window[0, 1:, None]).squeeze(1)
                total += loss.sum().item()
                targets += loss.numel()
            perplexity.append({"context": context, "mean_nll": total / targets})
    return cases, perplexity


def gen_f32(output_dir=OUT_DIR):
    import torch
    import transformers
    from f32 import CONFIG, weight_hash

    torch.set_num_threads(1)
    fixtures = []
    for tied in (False, True):
        model, weights = tiny_qwen3(tied)
        cases, perplexity = reference_outputs(model, torch)
        fixtures.append({"tied": tied, "weights_sha256": weight_hash(weights),
                         "cases": cases, "perplexity": perplexity})
    path = os.path.join(output_dir, "baseline_f32.json")
    _write(path, {
        "_comment": "Generated by tools/gen_baseline.py f32 using HF Qwen3ForCausalLM with deterministic synthetic weights.",
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "dtype": "float32", "attention": "eager", "config": CONFIG, "fixtures": fixtures})
    print("wrote %s (tied/untied, full logits and windowed NLL)" % path)


def tiny_mxfp4(tied, moe, seed=12345, context=16):
    """A tiny MXFP4 fixture of tests/mxfp4.py as an HF model holding its spec-decoded weights; returns the model, its configuration, weights and raw blocks."""
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM
    from mxfp4 import fixture

    config, weights, packed = fixture(tied, moe, seed, context)
    params = dict(vocab_size=257, hidden_size=config["embedding_length"],
                  intermediate_size=config["feed_forward_length"], num_hidden_layers=config["block_count"],
                  num_attention_heads=config["attention.head_count"], num_key_value_heads=config["attention.head_count_kv"],
                  head_dim=config["attention.key_length"], max_position_embeddings=config["context_length"],
                  rope_theta=10000.0, rms_norm_eps=1e-6, tie_word_embeddings=tied, attention_dropout=0.0)
    if moe:
        params.update(moe_intermediate_size=config["expert_feed_forward_length"], num_experts=config["expert_count"],
                      num_experts_per_tok=config["expert_used_count"], norm_topk_prob=True, decoder_sparse_step=1, mlp_only_layers=[])
    hf_config = (Qwen3MoeConfig if moe else Qwen3Config)(**params)
    hf_config._attn_implementation = "eager"
    model = (Qwen3MoeForCausalLM if moe else Qwen3ForCausalLM)(hf_config).float().eval()
    load_tiny_weights(model, weights, torch, tied)
    return model, config, weights, packed


def gen_mxfp4(output_dir=OUT_DIR):
    import torch
    import transformers
    from mxfp4 import VARIANTS, PROMPT_TEXTS, packed_hash, weight_hash

    if transformers.__version__ != "4.55.2":
        raise SystemExit("mxfp4 requires pinned transformers 4.55.2")
    torch.set_num_threads(1)
    for suffix, context, texts, windows in (("", 16, None, (4, 16)), ("_prompt", 128, PROMPT_TEXTS, (40, 64, 128))):
        fixtures = []
        for name, tied, moe in VARIANTS:
            model, config, weights, packed = tiny_mxfp4(tied, moe, context=context)
            gaps = []
            if moe:
                def watch(_, __, out):
                    probabilities = torch.softmax(out.double(), dim=-1).sort(dim=-1, descending=True).values
                    k = config["expert_used_count"]
                    gaps.append((probabilities[:, k - 1] - probabilities[:, k]).min().item())
                for layer in model.model.layers:
                    layer.mlp.gate.register_forward_hook(watch)
            cases, perplexity = reference_outputs(model, torch, texts, windows)
            if gaps and min(gaps) < 1e-4:
                raise SystemExit("MXFP4 MoE routing gap %.2e is too near a tie; change the fixture weights" % min(gaps))
            fixtures.append(dict(name=name, config=config, weights_sha256=weight_hash(weights), packed_sha256=packed_hash(packed),
                                 min_routing_gap=min(gaps) if gaps else None, cases=cases, perplexity=perplexity))
        path = os.path.join(output_dir, "baseline_mxfp4%s.json" % suffix)
        _write(path, {"_comment": "Generated by tools/gen_baseline.py mxfp4 from independently spec-decoded raw blocks.",
                      "torch_version": torch.__version__, "transformers_version": transformers.__version__,
                      "dtype": "float32", "attention": "eager", "fixtures": fixtures})
        print("wrote %s (MXFP4 dense tied/untied and MoE, logits and windowed NLL)" % path)


def tiny_moe(seed=67890):
    """The tiny qwen3moe model of tests/moe.py as an HF Qwen3MoeForCausalLM holding the weights moe.tensors builds from `seed`; returns the model and those weights."""
    import torch
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    from moe import CONFIG, DENSE_LAYERS, tensors

    config = Qwen3MoeConfig(vocab_size=257, hidden_size=CONFIG["embedding_length"],
                            intermediate_size=CONFIG["feed_forward_length"],
                            moe_intermediate_size=CONFIG["expert_feed_forward_length"],
                            num_hidden_layers=CONFIG["block_count"], num_attention_heads=2, num_key_value_heads=1,
                            head_dim=CONFIG["attention.key_length"], max_position_embeddings=16, rope_theta=10000.0,
                            rms_norm_eps=1e-6, tie_word_embeddings=False, attention_dropout=0.0,
                            num_experts=CONFIG["expert_count"], num_experts_per_tok=CONFIG["expert_used_count"],
                            norm_topk_prob=True, decoder_sparse_step=1, mlp_only_layers=list(DENSE_LAYERS))
    config._attn_implementation = "eager"
    model = Qwen3MoeForCausalLM(config).float().eval()
    weights = tensors(seed)
    load_tiny_weights(model, weights, torch)
    return model, weights


def gen_moe(output_dir=OUT_DIR):
    import torch
    import transformers
    from moe import CONFIG, DENSE_LAYERS, weight_hash

    torch.set_num_threads(1)
    model, weights = tiny_moe()
    # The smallest gap between a token's k-th and next expert probability over every forward reference_outputs runs; a near tie could route differently under other rounding.
    k, gaps = CONFIG["expert_used_count"], []
    def watch(_, __, out):
        p = torch.softmax(out.double(), dim=-1).sort(dim=-1, descending=True).values
        gaps.append((p[:, k - 1] - p[:, k]).min().item())
    for layer in model.model.layers:
        if hasattr(layer.mlp, "gate"):
            layer.mlp.gate.register_forward_hook(watch)
    cases, perplexity = reference_outputs(model, torch)
    if min(gaps) < 1e-4:
        raise SystemExit("moe: a token's routing is within %.2e of a tie; change the weights" % min(gaps))
    path = os.path.join(output_dir, "baseline_moe.json")
    _write(path, {
        "_comment": "Generated by tools/gen_baseline.py moe using HF Qwen3MoeForCausalLM with deterministic synthetic weights.",
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "dtype": "float32", "attention": "eager", "config": CONFIG, "dense_layers": list(DENSE_LAYERS),
        "weights_sha256": weight_hash(weights), "min_routing_gap": min(gaps),
        "cases": cases, "perplexity": perplexity})
    print("wrote %s (full logits and windowed NLL; smallest routing gap %.2e)" % (path, min(gaps)))


def moe_q8_state(path, torch):
    """A qwen3moe GGUF file's tensors as HF Qwen3MoeForCausalLM parameters, each decoded to f32 by tests/spec_decode.py's numpy form: a stacked expert tensor split into its experts' projections and the router named as HF names it."""
    import spec_decode
    model = spec_decode.GGUF(path)
    if model.value("general.architecture") != "qwen3moe":
        raise SystemExit("%s does not hold a qwen3moe model" % path)
    tensors = [t for t in model.tensors if t.name != "unused.weight"]
    return {key: torch.from_numpy(values).reshape(shape)
            for key, shape, values, _ in qwen3moe_tensors(model, True, tensors)}


def gen_moe_q8(output_dir=OUT_DIR):
    """The goldens of tests/moe.py's Q8_0 model: each variant written by the test's own writer and read back through tests/spec_decode.py into HF Qwen3MoeForCausalLM.
    For each prompt, HF's greedy continuation of Q8_STEPS ids, then every logit from the prompt's last position through them with each row's top id and top-two gap, and the smallest margin between a token's k-th and next router logit over every position and routed layer."""
    import numpy
    import torch
    import transformers
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    from moe import DENSE_LAYERS, Q8_CONFIG, Q8_MIN_ROUTING_GAP, Q8_PROMPTS, Q8_SEED, Q8_STEPS, Q8_VARIANTS, write_q8_model

    torch.set_num_threads(1)
    c = Q8_CONFIG
    config = Qwen3MoeConfig(vocab_size=257, hidden_size=c["embedding_length"], intermediate_size=c["feed_forward_length"],
                            moe_intermediate_size=c["expert_feed_forward_length"], num_hidden_layers=c["block_count"],
                            num_attention_heads=c["attention.head_count"], num_key_value_heads=c["attention.head_count_kv"],
                            head_dim=c["attention.key_length"], max_position_embeddings=c["context_length"], rope_theta=10000.0,
                            rms_norm_eps=1e-6, tie_word_embeddings=False, attention_dropout=0.0,
                            num_experts=c["expert_count"], num_experts_per_tok=c["expert_used_count"],
                            norm_topk_prob=True, decoder_sparse_step=1, mlp_only_layers=list(DENSE_LAYERS))
    config._attn_implementation = "eager"
    k = c["expert_used_count"]
    variants = []
    with tempfile.TemporaryDirectory(prefix="llmx_moe_q8_") as directory:
        for name, scale in Q8_VARIANTS:
            path = os.path.join(directory, name + ".gguf")
            sha256 = write_q8_model(path, scale)
            model = Qwen3MoeForCausalLM(config).float().eval()
            model.load_state_dict(moe_q8_state(path, torch), strict=True)
            margins, cases = [], []

            def watch(_, __, out):
                top = out.double().sort(dim=-1, descending=True).values
                margins.append((top[:, k - 1] - top[:, k]).min().item())

            with torch.inference_mode():
                for prompt in Q8_PROMPTS:
                    ids = list(prompt.encode("ascii"))
                    # Greedy takes the lowest id among equal logits, as argmax does.
                    for _ in range(Q8_STEPS):
                        ids.append(int(model(torch.tensor([ids]), use_cache=False).logits[0, -1].argmax()))
                    hooks = [layer.mlp.gate.register_forward_hook(watch) for layer in model.model.layers if hasattr(layer.mlp, "gate")]
                    rows = model(torch.tensor([ids]), use_cache=False).logits[0, len(prompt) - 1:]
                    for hook in hooks:
                        hook.remove()
                    top = rows.double().sort(dim=-1, descending=True)
                    forced = ids[len(prompt):]
                    if top.indices[:Q8_STEPS, 0].tolist() != forced:
                        raise SystemExit("moe-q8 %s: the whole forward's greedy ids are not the step-by-step ones for %r" % (name, prompt))
                    cases.append({"prompt": prompt, "forced_ids": forced, "top_ids": top.indices[:, 0].tolist(),
                                  "gaps": (top.values[:, 0] - top.values[:, 1]).tolist(),
                                  "rows": [[round(v, 7) for v in row] for row in rows.tolist()]})
            gated = name == "gated"
            if gated != (min(margins) >= Q8_MIN_ROUTING_GAP):
                raise SystemExit("moe-q8 %s: the smallest router margin is %.2e; change Q8_SEED or the variant's router scale" % (name, min(margins)))
            variants.append({"name": name, "router_scale": scale, "file_sha256": sha256, "min_routing_gap": min(margins), "cases": cases})
            print("moe-q8 %s: smallest router margin %.2e, greedy %s" % (name, min(margins), [case["forced_ids"] for case in cases]))
    path = os.path.join(output_dir, "baseline_moe_q8.json")
    _write(path, {
        "_comment": "Generated by tools/gen_baseline.py moe-q8 using HF Qwen3MoeForCausalLM holding each variant's Q8_0 file's own weights, decoded by tests/spec_decode.py's numpy form.",
        "torch_version": torch.__version__, "transformers_version": transformers.__version__, "numpy_version": numpy.__version__,
        "dtype": "float32", "attention": "eager", "config": Q8_CONFIG, "dense_layers": list(DENSE_LAYERS),
        "seed": Q8_SEED, "steps": Q8_STEPS, "variants": variants}, indent=None)
    print("wrote %s (%d variants, %d prompts each)" % (path, len(variants), len(Q8_PROMPTS)))


# The smallest gap allowed between a greedy step's top two logits, since a nearer tie could turn over under other rounding.
QWEN35_GREEDY_GAP = 1e-4


def qwen35_hf_config(fixture):
    """The HF text config of a tiny qwen35 or qwen35moe fixture, from the metadata tests/qwen35.py writes to its GGUF."""
    from f32 import VOCAB
    from qwen35 import CONFIG, LAYERS, MOE, V_HEAD, full_attention
    head = CONFIG["attention.key_length"]
    moe = {"architectures": ["Qwen3_5MoeForCausalLM"], "model_type": "qwen3_5_moe_text", "num_experts": MOE["expert_count"],
           "num_experts_per_tok": MOE["expert_used_count"], "moe_intermediate_size": MOE["expert_feed_forward_length"],
           "shared_expert_intermediate_size": MOE["expert_shared_feed_forward_length"]} if fixture.get("moe") else {}
    return {"architectures": ["Qwen3_5ForCausalLM"], "model_type": "qwen3_5_text", "dtype": "float32",
            "vocab_size": VOCAB, "hidden_size": CONFIG["embedding_length"], "intermediate_size": CONFIG["feed_forward_length"],
            "num_hidden_layers": LAYERS, "num_attention_heads": CONFIG["attention.head_count"],
            "num_key_value_heads": CONFIG["attention.head_count_kv"], "head_dim": head, "hidden_act": "silu",
            "max_position_embeddings": CONFIG["context_length"], "rms_norm_eps": CONFIG["attention.layer_norm_rms_epsilon"],
            "tie_word_embeddings": fixture["tied"], "attention_bias": False, "attention_dropout": 0.0,
            "linear_conv_kernel_dim": CONFIG["ssm.conv_kernel"], "linear_key_head_dim": CONFIG["ssm.state_size"],
            "linear_value_head_dim": V_HEAD, "linear_num_key_heads": CONFIG["ssm.group_count"],
            "linear_num_value_heads": fixture["v_heads"],
            "layer_types": ["full_attention" if full_attention(layer) else "linear_attention" for layer in range(LAYERS)],
            "rope_parameters": {"rope_type": "default", "rope_theta": CONFIG["rope.freq_base"],
                                "partial_rotary_factor": CONFIG["rope.dimension_count"] / head,
                                "mrope_section": CONFIG["rope.dimension_sections"][:3], "mrope_interleaved": True}, **moe}


def load_qwen35_tiny(directory, keys, torch, transformers, moe=False):
    """The Qwen3_5ForCausalLM checkpoint in `directory`, or with `moe` the Qwen3_5MoeForCausalLM one, whose parameters are named `keys`, loaded from local files in float32 with eager attention.
    It may leave only mtp.* and model.visual.* keys unused and may miss no key; returns the model and the keys it left unused."""
    causal = transformers.Qwen3_5MoeForCausalLM if moe else transformers.Qwen3_5ForCausalLM
    model, info = causal.from_pretrained(directory, dtype=torch.float32, attn_implementation="eager",
                                         local_files_only=True, output_loading_info=True)
    # HF leaves out of its report the keys it is told to drop, so the unused keys are the checkpoint's less the model's own.
    unused = sorted(set(keys) - set(model.state_dict()))
    stray = sorted(set(info["unexpected_keys"]) | {key for key in unused if not QWEN35_UNUSED.match(key)})
    if stray:
        raise SystemExit("qwen35: the model leaves checkpoint keys other than mtp.* and model.visual.* unused: " + ", ".join(stray))
    missing = sorted(set(info["missing_keys"]) | {mismatch[0] for mismatch in info["mismatched_keys"]})
    if missing or info["error_msgs"]:
        raise SystemExit("qwen35: the checkpoint misses or mismatches %s %s" % (", ".join(missing), info["error_msgs"]))
    if model.config._attn_implementation != "eager" or any(p.dtype != torch.float32 for p in model.parameters()):
        raise SystemExit("qwen35: the model did not load in float32 with eager attention")
    model.eval()
    return model, unused


@contextlib.contextmanager
def counted_delta_rules(*modelings):
    """HF's two delta-rule functions in each modeling module, the per-token recurrence and the chunked form, wrapped to count their calls in one count, with the lowest log-decay the recurrence reads."""
    calls = {"recurrent": 0, "chunk": 0, "log_decay": 0.0}
    originals = [(m, m.torch_recurrent_gated_delta_rule, m.torch_chunk_gated_delta_rule) for m in modelings]

    def counted(recurrent, chunk):
        def count_recurrent(*args, **kwargs):
            calls["recurrent"] += 1
            calls["log_decay"] = min(calls["log_decay"], kwargs["g"].min().item())
            return recurrent(*args, **kwargs)

        def count_chunk(*args, **kwargs):
            calls["chunk"] += 1
            return chunk(*args, **kwargs)
        return count_recurrent, count_chunk

    for m, recurrent, chunk in originals:
        m.torch_recurrent_gated_delta_rule, m.torch_chunk_gated_delta_rule = counted(recurrent, chunk)
    try:
        yield calls
    finally:
        for m, recurrent, chunk in originals:
            m.torch_recurrent_gated_delta_rule, m.torch_chunk_gated_delta_rule = recurrent, chunk


# The smallest gap allowed between a token's k-th and (k+1)-th router logit in the tiny qwen35moe fixture, so that no expert choice can turn over under other rounding.
QWEN35_ROUTER_GAP = 1e-3


@contextlib.contextmanager
def router_gaps(model, torch):
    """Every router of `model` hooked to record, over the tokens it routes, the smallest gap between the k-th and (k+1)-th logit, which order the probabilities."""
    gaps = []

    def hook(module, inputs, output):
        top = torch.topk(output[0].double(), module.top_k + 1, dim=-1).values
        gaps.append((top[:, -2] - top[:, -1]).min().item())

    handles = [m.register_forward_hook(hook) for m in model.modules() if type(m).__name__ == "Qwen3_5MoeTopKRouter"]
    try:
        yield gaps
    finally:
        for h in handles:
            h.remove()


def qwen35_cache(model, torch, transformers):
    """An HF cache for `model` holding the zero conv rows and state a sequence starts from, so its first token is a cached step too."""
    config = model.config
    cache = transformers.DynamicCache(config=config)
    conv = 2 * config.linear_num_key_heads * config.linear_key_head_dim + config.linear_num_value_heads * config.linear_value_head_dim
    for layer, kind in enumerate(config.layer_types):
        if kind == "linear_attention":
            cache.update_conv_state(torch.zeros(1, conv, config.linear_conv_kernel_dim), layer)
            cache.update_recurrent_state(torch.zeros(1, config.linear_num_value_heads, config.linear_key_head_dim,
                                                     config.linear_value_head_dim), layer)
    return cache


def qwen35_steps(model, torch, cache, tokens, calls):
    """HF's logits after each of `tokens`, fed to `cache` one cached step at a time; every step must run HF's recurrence in every linear layer, and none its chunked form."""
    linear = model.config.layer_types.count("linear_attention")
    before = calls["recurrent"], calls["chunk"]
    rows = [model(input_ids=torch.tensor([[token]]), past_key_values=cache, use_cache=True).logits[0, -1] for token in tokens]
    if (calls["recurrent"] - before[0], calls["chunk"] - before[1]) != (len(tokens) * linear, 0):
        raise SystemExit("qwen35: %d cached steps did not each run HF's recurrence in all %d linear layers" % (len(tokens), linear))
    return torch.stack(rows)


def qwen35_full(model, torch, tokens, calls):
    """HF's logits at every position of `tokens` from one full forward, which runs the chunked form once in every linear layer."""
    linear = model.config.layer_types.count("linear_attention")
    before = calls["recurrent"], calls["chunk"]
    logits = model(input_ids=torch.tensor([tokens]), use_cache=False).logits[0]
    if (calls["recurrent"] - before[0], calls["chunk"] - before[1]) != (0, linear):
        raise SystemExit("qwen35: the full forward did not run HF's chunked form once in each linear layer")
    return logits


def qwen35_goldens(model, torch, transformers, calls):
    """A fixture's goldens from HF's token-by-token cached forward, each text and window from a fresh zero state: all logits at each text's last position, the mean NLL of the longest text in windows of 4 and 16 tokens, and six greedy tokens after the fourth text, the end-of-text token left out.
    Returns them with the stepwise and full-forward logits over the longest text."""
    from f32 import TEXTS
    from qwen35 import EOS

    def steps(tokens, cache=None):
        return qwen35_steps(model, torch, cache or qwen35_cache(model, torch, transformers), tokens, calls)

    ids = [list(text.encode("ascii")) for text in TEXTS]
    longest = steps(ids[-1])
    cases = []
    for text, tokens in zip(TEXTS, ids):
        rows = steps(tokens)
        # Every text is a head of the longest one, so its steps are the same steps.
        if not torch.equal(rows, longest[:len(tokens)]):
            raise SystemExit("qwen35: the steps of %r differ from the same steps of the longest text" % text)
        cases.append({"text": text, "logits": rows[-1].tolist()})
    perplexity = []
    for context in (4, 16):
        total, targets = 0.0, 0
        for start in range(0, len(ids[-1]), context):
            window = ids[-1][start:start + context]
            if len(window) < 2:
                continue
            logits = steps(window)[:-1].double()
            loss = torch.logsumexp(logits, -1) - logits.gather(1, torch.tensor(window[1:])[:, None]).squeeze(1)
            total += loss.sum().item()
            targets += loss.numel()
        perplexity.append({"context": context, "mean_nll": total / targets})
    cache = qwen35_cache(model, torch, transformers)
    last = steps(ids[3], cache)[-1]
    greedy, gap = [], math.inf
    for step in range(6):
        scores = last.clone()
        scores[EOS] = -math.inf
        top = torch.topk(scores, 2)
        gap = min(gap, (top.values[0] - top.values[1]).item())
        greedy.append(int(top.indices[0]))
        if step < 5:
            last = steps(greedy[-1:], cache)[-1]
    if gap < QWEN35_GREEDY_GAP:
        raise SystemExit("qwen35: a greedy step's top two logits are within %.2e; change the weights" % gap)
    goldens = {"cases": cases, "perplexity": perplexity, "greedy": {"prompt": TEXTS[3], "ids": greedy, "min_gap": gap}}
    return goldens, longest, qwen35_full(model, torch, ids[-1], calls)


def gen_qwen35_tiny(output_dir=OUT_DIR):
    torch, transformers, modeling = qwen35_environment()
    from safetensors.torch import save_file
    import qwen35
    from f32 import TEXTS, weight_hash

    torch.set_num_threads(1)
    fixtures, forwards = [], {}
    moe_modeling = importlib.import_module("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe")
    with counted_delta_rules(modeling, moe_modeling) as calls, torch.no_grad():
        for spec in qwen35.FIXTURES:
            raw = qwen35.raw_weights(spec)
            record = dict(spec, weights_sha256=weight_hash(qwen35.hashed(raw)))
            with tempfile.TemporaryDirectory(prefix="llmx_qwen35_hf_") as directory:
                with open(os.path.join(directory, "config.json"), "w", encoding="utf-8") as f:
                    json.dump(qwen35_hf_config(spec), f)
                tensors = {name: torch.tensor(values, dtype=torch.float32).reshape(shape) for name, shape, values in raw}
                save_file(tensors, os.path.join(directory, "model.safetensors"), metadata={"format": "pt"})
                model, record["unused_keys"] = load_qwen35_tiny(directory, list(tensors), torch, transformers, bool(spec.get("moe")))
                state = model.state_dict()
                if any(not torch.equal(state[name], tensor) for name, tensor in tensors.items() if name in state) or \
                        (spec["tied"] and not torch.equal(state["lm_head.weight"], state["model.embed_tokens.weight"])):
                    raise SystemExit("qwen35: %s holds other values than its checkpoint" % spec["name"])
                if model.model.rotary_emb.inv_freq.numel() * 2 != qwen35.CONFIG["rope.dimension_count"]:
                    raise SystemExit("qwen35: %s rotates another width than the fixture's" % spec["name"])
                calls.update(recurrent=0, chunk=0, log_decay=0.0)
                if spec["mtp"]:
                    # HF drops the MTP block, so the file with it must give the logits of the file without it, bit for bit.
                    base = next(r for r in fixtures if not r["mtp"] and (r["v_heads"], r["tied"]) == (spec["v_heads"], spec["tied"]))
                    tokens = list(TEXTS[-1].encode("ascii"))
                    longest = qwen35_steps(model, torch, qwen35_cache(model, torch, transformers), tokens, calls)
                    full = qwen35_full(model, torch, tokens, calls)
                    if not (torch.equal(longest, forwards[base["name"]][0]) and torch.equal(full, forwards[base["name"]][1])):
                        raise SystemExit("qwen35: %s gives other logits than %s" % (spec["name"], base["name"]))
                    record["base"] = base["name"]
                else:
                    with router_gaps(model, torch) as gaps:
                        goldens, longest, full = qwen35_goldens(model, torch, transformers, calls)
                    if spec.get("moe"):
                        record["min_router_gap"] = min(gaps)
                        if record["min_router_gap"] < QWEN35_ROUTER_GAP:
                            raise SystemExit("qwen35: %s routes a token whose k-th and next expert are within %.2e; change the weights" % (spec["name"], min(gaps)))
                    forwards[spec["name"]] = longest, full
                    # The goldens hold the recurrence's arithmetic only if the chunked form, which the full forward runs, gives other bits.
                    record["full_forward_distance"] = (full - longest).abs().max().item()
                    if record["full_forward_distance"] == 0:
                        raise SystemExit("qwen35: %s's full forward equals its steps bit for bit, so the goldens do not show which form made them" % spec["name"])
                    record["min_log_decay"] = calls["log_decay"]
                    record.update(goldens)
                record["recurrent_steps"] = calls["recurrent"]
            fixtures.append(record)
            print("  %s: %d recurrent steps%s" % (spec["name"], record["recurrent_steps"],
                  ", full forward within %.3g" % record["full_forward_distance"] if "full_forward_distance" in record else ""))
    path = os.path.join(output_dir, "baseline_qwen35.json")
    _write(path, {
        "_comment": "Generated by tools/gen_baseline.py qwen35-tiny using HF Qwen3_5ForCausalLM, and Qwen3_5MoeForCausalLM for the qwen35moe fixture, with deterministic synthetic weights.",
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "dtype": "float32", "attention": "eager",
        "goldens": "HF's token-by-token cached forward from a cache holding a zero state, every step through its recurrent delta rule",
        "config": qwen35.CONFIG, "layers": qwen35.LAYERS, "v_head_width": qwen35.V_HEAD, "fixtures": fixtures})
    print("wrote %s (%d fixtures)" % (path, len(fixtures)))


# The prompts of the assembled MTP reference: one and two tokens, so row 0's zero row and the first carried row are both reached, and longer ones; each leaves the context room for the draft steps.
QWEN35_MTP_PROMPTS = ["a", "ab", "hello", "the quick br"]
QWEN35_MTP_STEPS = 2


def mtp_block(model, raw, torch, modeling):
    """The tiny fixture's MTP block assembled around HF's own full-attention Qwen3_5DecoderLayer: mtp.fc, the input norms of the embedding and of the target's row, the layer and mtp.norm, each holding the fixture's mtp.* weights and every one of them taken."""
    config = model.config
    width = config.hidden_size
    layer = modeling.Qwen3_5DecoderLayer(config, config.layer_types.index("full_attention"))
    fc = torch.nn.Linear(2 * width, width, bias=False)
    norms = {name: modeling.Qwen3_5RMSNorm(width, eps=config.rms_norm_eps) for name in ("pre_fc_norm_embedding", "pre_fc_norm_hidden", "norm")}
    weights = {name: torch.tensor(values, dtype=torch.float32).reshape(shape) for name, shape, values in raw if name.startswith("mtp.")}
    prefix = "mtp.layers.0."
    layer.load_state_dict({name[len(prefix):]: w for name, w in weights.items() if name.startswith(prefix)}, strict=True)
    fc.weight.copy_(weights["mtp.fc.weight"])
    for name, norm in norms.items():
        norm.weight.copy_(weights["mtp.%s.weight" % name])
    taken = {"mtp.fc.weight"} | {"mtp.%s.weight" % name for name in norms} | {prefix + name for name in layer.state_dict()}
    if taken != set(weights):
        raise SystemExit("qwen35-mtp: the assembled block takes %s and the fixture holds %s" % (sorted(taken), sorted(weights)))
    layer.eval()
    return layer, fc, norms


def mtp_rows(model, block, torch, embeddings, hidden):
    """The assembled block over rows of (token embedding, the target's row before it) at rotary positions 0 to T - 1, causal and eager, the embedding first in the concatenation: each row after mtp.norm."""
    layer, fc, norms = block
    rows = embeddings.shape[0]
    x = fc(torch.cat([norms["pre_fc_norm_embedding"](embeddings), norms["pre_fc_norm_hidden"](hidden)], -1))[None]
    positions = torch.arange(rows).view(1, 1, rows).expand(3, 1, rows)
    mask = torch.full((rows, rows), float("-inf")).triu(1)[None, None]
    return norms["norm"](layer(x, position_embeddings=model.model.rotary_emb(x, positions), attention_mask=mask)[0])


def gen_qwen35_mtp(output_dir=OUT_DIR):
    torch, transformers, modeling = qwen35_environment()
    from safetensors.torch import save_file
    import qwen35
    from f32 import weight_hash

    torch.set_num_threads(1)
    spec = next(f for f in qwen35.FIXTURES if f["mtp"])
    raw = qwen35.raw_weights(spec)
    cases = []
    with counted_delta_rules(modeling) as calls, torch.no_grad(), tempfile.TemporaryDirectory(prefix="llmx_qwen35_mtp_") as directory:
        with open(os.path.join(directory, "config.json"), "w", encoding="utf-8") as f:
            json.dump(qwen35_hf_config(spec), f)
        tensors = {name: torch.tensor(values, dtype=torch.float32).reshape(shape) for name, shape, values in raw}
        save_file(tensors, os.path.join(directory, "model.safetensors"), metadata={"format": "pt"})
        model, unused = load_qwen35_tiny(directory, list(tensors), torch, transformers)
        block = mtp_block(model, raw, torch, modeling)
        if sorted(unused) != sorted(name for name in tensors if name.startswith("mtp.")):
            raise SystemExit("qwen35-mtp: the target leaves keys unused other than the MTP block's: %s" % unused)
        embed, head = model.model.embed_tokens, model.lm_head
        linear = model.config.layer_types.count("linear_attention")
        for text in QWEN35_MTP_PROMPTS:
            tokens = list(text.encode("ascii"))
            if len(tokens) + QWEN35_MTP_STEPS > qwen35.CONFIG["context_length"]:
                raise SystemExit("qwen35-mtp: %r leaves no room for the draft steps" % text)
            # The target's rows after its final norm, one cached step a token from a zero state, each through HF's recurrence.
            cache = qwen35_cache(model, torch, transformers)
            before = calls["recurrent"]
            target = torch.stack([model.model(input_ids=torch.tensor([[token]]), past_key_values=cache, use_cache=True).last_hidden_state[0, -1]
                                  for token in tokens])
            if calls["recurrent"] - before != len(tokens) * linear or calls["chunk"]:
                raise SystemExit("qwen35-mtp: the target's steps did not each run HF's recurrence")
            pick = int(torch.argmax(head(target[-1])))
            # Row j reads token j and the target's row j - 1, row 0 a zero row; the draft rows follow the prompt's, the first reading the pick and the target's last row, each later one the draft before it and the block's row before it.
            ids = tokens + [pick]
            hidden = torch.cat([torch.zeros(1, target.shape[1]), target])
            drafts, logits, gap = [], [], math.inf
            for step in range(QWEN35_MTP_STEPS):
                out = mtp_rows(model, block, torch, embed(torch.tensor(ids)), hidden)
                row = head(out[-1])
                top = torch.topk(row, 2)
                gap = min(gap, (top.values[0] - top.values[1]).item())
                drafts.append(int(top.indices[0]))
                logits.append(row.tolist())
                ids.append(drafts[-1])
                hidden = torch.cat([hidden, out[-1:]])
            if gap < QWEN35_GREEDY_GAP:
                raise SystemExit("qwen35-mtp: a draft row's top two logits are within %.2e; change the weights" % gap)
            cases.append({"prompt": text, "pick": pick, "drafts": drafts, "logits": logits, "min_gap": gap})
            print("  %r: pick %d, drafts %s, gap %.3g" % (text, pick, drafts, gap))
    path = os.path.join(output_dir, "baseline_qwen35_mtp.json")
    _write(path, {
        "_comment": "Generated by tools/gen_baseline.py qwen35-mtp: an assembled HF reference, not HF's own MTP, which drops mtp.*: HF's full-attention Qwen3_5DecoderLayer with mtp.fc, the two input norms, mtp.norm and the target's head around it.",
        "torch_version": torch.__version__, "transformers_version": transformers.__version__, "dtype": "float32", "attention": "eager",
        "conventions": "token embedding first in the concatenation; the target's row after its final norm; a draft step's row after mtp.norm for the next step; row j at rotary position j; row 0 reading a zero row",
        "fixture": spec["name"], "weights_sha256": weight_hash(qwen35.hashed(raw)), "steps": QWEN35_MTP_STEPS, "cases": cases})
    print("wrote %s (%d cases)" % (path, len(cases)))


# The kinds whose inputs are fixed, so only --output-dir applies to them, each with the reason a refusal gives.
FIXED_KINDS = {
    "f32": "uses fixed synthetic weights and one thread",
    "moe": "uses fixed synthetic weights and one thread",
    "moe-q8": "uses fixed synthetic weights and one thread",
    "mxfp4": "uses fixed synthetic raw blocks and one thread",
    "tokenizer-qwen35": "reads its own pinned tokenizer files",
    "qwen35-tiny": "uses fixed synthetic weights and one thread",
    "qwen35-mtp": "uses fixed synthetic weights and one thread",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate independent pinned HF reference fixtures on CPU.")
    parser.add_argument("kind", nargs="?", default="all", choices=("all", "tokenizer", "logits", "perplexity", "f32", "moe", "moe-q8", "mxfp4", "tokenizer-qwen35", "qwen35-tiny", "qwen35-mtp", "qwen35", "file-exact"))
    parser.add_argument("--repo", default=TOKENIZER_REPO, help="HF model/tokenizer repository")
    parser.add_argument("--revision", help="full 40-character HF commit SHA (required for another repository)")
    parser.add_argument("--output-dir", help="fixture directory (required for another model/revision)")
    parser.add_argument("--gguf-repo", help="associated GGUF repository label; required with --gguf-file")
    parser.add_argument("--gguf-file", help="associated GGUF filename label; required for another model")
    parser.add_argument("--threads", type=int, help="HF CPU threads for real-model logits/PPL (default: 6)")
    parser.add_argument("--weights-gguf", help="file-exact: the qwen3, qwen3moe or pinned qwen35 GGUF whose weights the reference model takes, which labels the goldens")
    parser.add_argument("--model", choices=sorted(QWEN35_MODELS), help="qwen35: the pinned checkpoint whose goldens to write")
    args = parser.parse_args(argv)
    args.family = "qwen3"
    if os.path.isdir(args.repo):
        parser.error("--repo must identify a Hub repository, not a local directory that bypasses revision pinning")
    if (args.kind == "qwen35") != bool(args.model):
        parser.error("qwen35 needs --model, which no other mode takes")
    if args.kind == "qwen35":
        if args.repo != TOKENIZER_REPO or args.revision or args.gguf_repo or args.gguf_file or args.weights_gguf:
            parser.error("qwen35 reads its model's pinned checkpoint; only --model, --output-dir and --threads apply")
        spec = QWEN35_MODELS[args.model]
        return qwen35_args(parser, args, args.model, args.output_dir or os.path.join(OUT_DIR, spec["directory"]))
    if args.kind in FIXED_KINDS and (args.repo != TOKENIZER_REPO or args.revision or args.gguf_repo or args.gguf_file or args.threads is not None):
        parser.error("%s %s; only --output-dir applies" % (args.kind, FIXED_KINDS[args.kind]))
    if args.kind == "tokenizer" and args.threads is not None:
        parser.error("--threads applies to real-model logits/PPL, not tokenizer generation")
    if (args.kind == "file-exact") != bool(args.weights_gguf):
        parser.error("file-exact needs --weights-gguf, which no other mode takes")
    if args.weights_gguf and (args.gguf_repo or args.gguf_file):
        parser.error("file-exact labels its goldens with the GGUF it reads, not with --gguf-repo or --gguf-file")
    if args.weights_gguf and not os.path.isfile(args.weights_gguf):
        parser.error("--weights-gguf %s is not a file" % args.weights_gguf)
    if args.repo != TOKENIZER_REPO and not args.revision:
        parser.error("another repository requires --revision")
    args.revision = args.revision or REFERENCE_REVISION
    if not re.fullmatch(r"[0-9a-fA-F]{40}", args.revision):
        parser.error("--revision must be a full 40-character commit SHA")
    args.revision = args.revision.lower()
    alternate = args.repo != TOKENIZER_REPO or args.revision != REFERENCE_REVISION or args.kind == "file-exact"
    if alternate and not args.output_dir:
        parser.error("another model/revision and file-exact require --output-dir")
    args.output_dir = os.path.abspath(args.output_dir or OUT_DIR)
    if alternate and os.path.normcase(os.path.realpath(args.output_dir)) == os.path.normcase(os.path.realpath(OUT_DIR)):
        parser.error("another model/revision or file-exact cannot overwrite the default fixture directory")
    if bool(args.gguf_repo) != bool(args.gguf_file):
        parser.error("--gguf-repo and --gguf-file must be supplied together")
    if args.weights_gguf:
        # The goldens name the file they were made from, with its pinned repository when it is a pinned fixture.
        from baseline import pinned_fixture
        args.weights_gguf = os.path.abspath(args.weights_gguf)
        args.gguf_file = os.path.basename(args.weights_gguf)
        pinned = pinned_fixture(args.gguf_file)
        args.gguf_repo = pinned["repo"] if pinned else None
        if pinned and pinned["family"] == "qwen35":
            # A qwen35 file's reference is the pinned checkpoint whose goldens are for it.
            model = next((name for name, spec in QWEN35_MODELS.items() if args.gguf_file in spec["gguf_files"]), None)
            if model is None or args.repo != TOKENIZER_REPO or args.revision != REFERENCE_REVISION:
                parser.error("a qwen35 file takes the pinned checkpoint of QWEN35_MODELS that lists it, not --repo or --revision")
            return qwen35_args(parser, args, model, args.output_dir)
    else:
        if args.repo != TOKENIZER_REPO and not args.gguf_repo:
            parser.error("another model requires explicit --gguf-repo and --gguf-file labels")
        args.gguf_repo = args.gguf_repo or GGUF_REPO
        args.gguf_file = args.gguf_file or GGUF_FILE
    args.threads = 6 if args.threads is None else args.threads
    if args.threads < 1:
        parser.error("--threads must be positive")
    return args


def qwen35_args(parser, args, model, output_dir):
    """`args` for the pinned qwen35 checkpoint `model`: its repository and commit, the GGUF labels of its first file unless a file is given, and `output_dir`."""
    spec = QWEN35_MODELS[model]
    args.family, args.qwen35_model = "qwen35", model
    args.repo, args.revision = spec["repo"], spec["revision"]
    args.output_dir = os.path.abspath(output_dir)
    # Each directory of committed goldens under tests/data is written only by the mode and model it belongs to.
    own = os.path.join(OUT_DIR, spec["directory"]) if args.kind == "qwen35" else None
    for directory in [OUT_DIR] + [os.path.join(OUT_DIR, s["directory"]) for s in QWEN35_MODELS.values()]:
        if directory != own and os.path.normcase(os.path.realpath(args.output_dir)) == os.path.normcase(os.path.realpath(directory)):
            parser.error("%s holds committed goldens of another mode or model" % directory)
    if not args.weights_gguf:
        from baseline import pinned_fixture
        args.gguf_file = spec["gguf_files"][0]
        args.gguf_repo = pinned_fixture(args.gguf_file)["repo"]
    args.threads = 6 if args.threads is None else args.threads
    if args.threads < 1:
        parser.error("--threads must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.kind in ("all", "tokenizer"):
        gen_tokenizer(args)
    if args.kind in ("all", "logits"):
        gen_logits(args)
    if args.kind in ("all", "perplexity"):
        gen_perplexity(args)
    if args.kind in ("all", "f32"):
        gen_f32(args.output_dir)
    if args.kind in ("all", "moe"):
        gen_moe(args.output_dir)
    if args.kind == "moe-q8":
        gen_moe_q8(args.output_dir)
    if args.kind == "mxfp4":
        gen_mxfp4(args.output_dir)
    if args.kind == "tokenizer-qwen35":
        gen_tokenizer_qwen35(args.output_dir)
    if args.kind == "qwen35-tiny":
        gen_qwen35_tiny(args.output_dir)
    if args.kind == "qwen35-mtp":
        gen_qwen35_mtp(args.output_dir)
    if args.kind == "qwen35":
        loaded = load_reference(args)
        gen_logits(args, loaded)
        gen_chat_qwen35(args, loaded)
        for context in sorted(QWEN35_PPL):
            gen_perplexity_qwen35(args, loaded, context)
    if args.kind == "file-exact":
        loaded = load_reference(args)
        gen_logits(args, loaded)
        if args.family == "qwen35":
            gen_perplexity_qwen35(args, loaded, 512)
        else:
            gen_perplexity(args, loaded)
    if args.family == "qwen35" and not args.chunked_calls:
        raise SystemExit("qwen35: HF's chunked form never ran, so the goldens are not its full forward")


if __name__ == "__main__":
    sys.exit(main())
