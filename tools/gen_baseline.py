"""Generate golden fixtures for the correctness baseline.

Run this ONCE on a machine with the HF tooling, then commit the output.
tests/baseline.py only reads the committed JSON, so running the test suite never needs torch, transformers, or network access -- llmx stays dependency-free at runtime and the suite stays self-contained.

    python tools/gen_baseline.py [all|tokenizer|logits|perplexity|f32|moe|tokenizer-qwen35]

Defaults use the pinned Qwen3-0.6B reference.
Another model requires --repo, --revision (full commit SHA), --output-dir, --gguf-repo and --gguf-file.
The GGUF arguments are labels, not proof of the converted model's provenance.
Real-model logits/PPL use CPU float32 eager attention and --threads (default 6).
The independent synthetic f32 and moe fixtures use one thread; only --output-dir applies to those modes.
all includes both regardless of --repo.
tokenizer-qwen35 writes the qwen35 tokenizer golden from its own pinned tokenizer.json and tokenizer_config.json, takes only --output-dir, and is not part of all.

Requires: tokenizers, huggingface_hub (tokenizer goldens) and, for the logit/PPL goldens, torch + transformers.
Those two segfault together in some environments (any `from transformers import Auto*` dies); an isolated venv with numpy<2.3, torch 2.5.1+cpu and transformers 4.55.2 is known to work.
The qwen35 goldens come from a second isolated venv, so the first stays as it is: Python 3.12.13 with torch 2.5.1+cpu, transformers 5.17.0, tokenizers 0.23.2, huggingface_hub 1.33.0, safetensors 0.8.0, numpy 2.2.6 and Jinja2 3.1.6.
"""

import argparse
import hashlib
import io
import json
import math
import os
import re
import sys

TOKENIZER_REPO = "Qwen/Qwen3-0.6B"
GGUF_REPO = "Qwen/Qwen3-0.6B-GGUF"
GGUF_FILE = "Qwen3-0.6B-Q8_0.gguf"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "tests", "data")
# The synthetic fixtures' configurations, weights and texts come from the test modules that check them.
sys.path.insert(0, os.path.join(ROOT, "tests"))
REFERENCE_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"

# Each case targets a class of pretokenizer behaviour.
# The multi-space and indentation cases are the ones that caught the GPT-2 `\s+(?!\S)` bug, where runs of 2+ spaces were emitted whole instead of leaving the last space to attach to the following word.
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


def gen_tokenizer(args):
    from tokenizers import Tokenizer
    from huggingface_hub import hf_hub_download
    tok = Tokenizer.from_file(hf_hub_download(args.repo, "tokenizer.json", revision=args.revision))
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


def gen_tokenizer_qwen35(output_dir):
    """HF's ids for QWEN35_CASES from the pinned tokenizer.json, with the part of its vocabulary those texts reach, so tests/tokenizer.py checks them without a model file.
    The control tokens only tokenizer_config.json adds are kept apart, with the GGUF type of every added token."""
    import unicodedata
    import tokenizers
    from tokenizers import Tokenizer
    from tokenizers.pre_tokenizers import ByteLevel
    from huggingface_hub import hf_hub_download

    def pinned(name):
        """The path and parsed contents of one pinned tokenizer file, refused unless it has the pinned digest."""
        path = hf_hub_download(QWEN35_REPO, name, revision=QWEN35_REVISION)
        with open(path, "rb") as f:
            raw = f.read()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != QWEN35_SHA256[name]:
            raise SystemExit("tokenizer-qwen35: %s at %s has SHA-256 %s, not the pinned %s" % (name, QWEN35_REVISION, digest, QWEN35_SHA256[name]))
        return path, json.loads(raw)

    path, spec = pinned("tokenizer.json")
    config = pinned("tokenizer_config.json")[1]
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
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(args.threads)
    tok = AutoTokenizer.from_pretrained(args.repo, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.repo, revision=args.revision, torch_dtype=torch.float32,
        attn_implementation="eager")
    model.cpu().eval()
    return torch, transformers, tok, model


def reference_metadata(args, torch, transformers):
    return {"reference_repo": args.repo, "reference_revision": args.revision,
            "reference_dtype": "float32", "attention": "eager", "device": "cpu",
            "threads": args.threads, "torch_version": torch.__version__,
            "transformers_version": transformers.__version__}


def gen_logits(args):
    torch, transformers, tok, model = load_reference(args)
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
        "_comment": ("Generated by tools/gen_baseline.py from the FULL-PRECISION "
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


def gen_perplexity(args):
    torch, transformers, tok, model = load_reference(args)
    with open(os.path.join(ROOT, "tests", "data", "wiki.test.raw"), encoding="utf-8") as f:
        text = f.read(1024)
    ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids
    with torch.inference_mode():
        logits = model(ids, use_cache=False).logits[0, :-1].double()
        targets = ids[0, 1:]
        nll = torch.logsumexp(logits, dim=-1) - logits.gather(1, targets[:, None]).squeeze(1)
        mean_nll = nll.mean().item()
    chunk_cases = []
    for context, limit in ((64, 0), (64, 2), (123, 0)):
        total_nll, used, scored, chunks = 0.0, 0, 0, 0
        for window in ids.split(context, dim=1):
            count = window.shape[1]
            if count < 2 or (limit and chunks >= limit):
                break
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
        "_comment": "Generated by tools/gen_baseline.py perplexity. Do not hand-edit.",
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


def _write(path, doc):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
        f.write("\n")


def tiny_qwen3(tied):
    """The tiny dense model of tests/f32.py as an HF Qwen3ForCausalLM with eager attention, holding the weights f32.tensors builds; returns the model and those weights."""
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
    weights = tensors(tied)
    state = {name: torch.tensor(values, dtype=torch.float32).reshape(list(reversed(shape)))
             for _, name, shape, values in weights}
    if tied:
        state["lm_head.weight"] = state["model.embed_tokens.weight"]
    model.load_state_dict(state, strict=True)
    return model, weights


def reference_outputs(model, torch):
    """A tiny model's goldens over the texts of tests/f32.py: all logits at each text's last position, then the mean NLL of the longest text in windows of 4 and of 16 tokens."""
    from f32 import TEXTS

    cases, perplexity = [], []
    with torch.inference_mode():
        for text in TEXTS:
            ids = torch.tensor([list(text.encode("ascii"))])
            logits = model(ids, use_cache=False).logits[0, -1]
            cases.append({"text": text, "logits": logits.tolist()})
        ids = torch.tensor([list(TEXTS[-1].encode("ascii"))])
        for context in (4, 16):
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


def gen_moe(output_dir=OUT_DIR):
    import torch
    import transformers
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    from moe import CONFIG, DENSE_LAYERS, tensors, weight_hash

    torch.set_num_threads(1)
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
    weights = tensors()
    state = {}
    for _, name, shape, values in weights:
        if isinstance(name, list):
            per = len(values) // len(name)
            for e, expert in enumerate(name):
                state[expert] = torch.tensor(values[e * per:(e + 1) * per], dtype=torch.float32).reshape(shape[1], shape[0])
        else:
            state[name] = torch.tensor(values, dtype=torch.float32).reshape(list(reversed(shape)))
    model.load_state_dict(state, strict=True)
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


# The kinds whose inputs are fixed, so only --output-dir applies to them, each with the reason a refusal gives.
FIXED_KINDS = {
    "f32": "uses fixed synthetic weights and one thread",
    "moe": "uses fixed synthetic weights and one thread",
    "tokenizer-qwen35": "reads its own pinned tokenizer files",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate independent pinned HF reference fixtures on CPU.")
    parser.add_argument("kind", nargs="?", default="all", choices=("all", "tokenizer", "logits", "perplexity", "f32", "moe", "tokenizer-qwen35"))
    parser.add_argument("--repo", default=TOKENIZER_REPO, help="HF model/tokenizer repository")
    parser.add_argument("--revision", help="full 40-character HF commit SHA (required for another repository)")
    parser.add_argument("--output-dir", help="fixture directory (required for another model/revision)")
    parser.add_argument("--gguf-repo", help="associated GGUF repository label; required with --gguf-file")
    parser.add_argument("--gguf-file", help="associated GGUF filename label; required for another model")
    parser.add_argument("--threads", type=int, help="HF CPU threads for real-model logits/PPL (default: 6)")
    args = parser.parse_args(argv)
    if os.path.isdir(args.repo):
        parser.error("--repo must identify a Hub repository, not a local directory that bypasses revision pinning")
    if args.kind in FIXED_KINDS and (args.repo != TOKENIZER_REPO or args.revision or args.gguf_repo or args.gguf_file or args.threads is not None):
        parser.error("%s %s; only --output-dir applies" % (args.kind, FIXED_KINDS[args.kind]))
    if args.kind == "tokenizer" and args.threads is not None:
        parser.error("--threads applies to real-model logits/PPL, not tokenizer generation")
    if args.repo != TOKENIZER_REPO and not args.revision:
        parser.error("another repository requires --revision")
    args.revision = args.revision or REFERENCE_REVISION
    if not re.fullmatch(r"[0-9a-fA-F]{40}", args.revision):
        parser.error("--revision must be a full 40-character commit SHA")
    args.revision = args.revision.lower()
    alternate = args.repo != TOKENIZER_REPO or args.revision != REFERENCE_REVISION
    if alternate and not args.output_dir:
        parser.error("another model/revision requires --output-dir")
    args.output_dir = os.path.abspath(args.output_dir or OUT_DIR)
    if alternate and os.path.normcase(os.path.realpath(args.output_dir)) == os.path.normcase(os.path.realpath(OUT_DIR)):
        parser.error("another model/revision cannot overwrite the default fixture directory")
    if bool(args.gguf_repo) != bool(args.gguf_file):
        parser.error("--gguf-repo and --gguf-file must be supplied together")
    if args.repo != TOKENIZER_REPO and not args.gguf_repo:
        parser.error("another model requires explicit --gguf-repo and --gguf-file labels")
    args.gguf_repo = args.gguf_repo or GGUF_REPO
    args.gguf_file = args.gguf_file or GGUF_FILE
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
    if args.kind == "tokenizer-qwen35":
        gen_tokenizer_qwen35(args.output_dir)


if __name__ == "__main__":
    sys.exit(main())
