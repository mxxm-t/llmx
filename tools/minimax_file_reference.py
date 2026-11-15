"""File-exact MiniMax-M3.1 HF reference: transformers' own MiniMaxM3VLForCausalLM on a GGUF's stored weights, each module's weights resident only while it runs.

    python -X utf8 tools/minimax_file_reference.py rows --hf HF_DIR --output-dir OUT
    python -X utf8 tools/minimax_file_reference.py run --hf HF_DIR --gguf SHARD... --output-dir OUT [--threads N]
    python -X utf8 tools/minimax_file_reference.py compare --output-dir OUT --name NAME --llmx LOGITS.txt

rows tokenizes the test texts with the checkpoint's own tokenizer offline (docs/MINIMAX.md, Classes and checks), run writes HF's logits of each row, and compare holds `llmx logits` output to them.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re
import sys
import time

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS.parent / "tests"))
sys.path.insert(0, str(TOOLS))
import spec_decode  # noqa: E402
import fixture_minimax as names  # noqa: E402
import minimax_reference as pins  # noqa: E402
from gen_layered_reference import loaded_modules, peak_gib  # noqa: E402
from qwen4exp_file_reference import Shards, score_nll  # noqa: E402

ARCH = "minimax-m3"
TEXT = TOOLS.parent / "tests" / "data" / "wiki.test.raw"
# The rows: 512 raw tokens, the 512-token chat rendering of an instruction and a later extract, and 2304 raw tokens after the first window, past the first sparse drop at 2048.
WINDOW, LONG = 512, 2304
INSTRUCTION = "Summarize the following text.\n\n"
CHAT_OFFSET = 200000
# Experts decoded ahead of HF's loop, in the router's hit order.
AHEAD = 4


def rows(hf, output_dir):
    """Write rows.json and NAME.ids, each row's ids as HF's tokenizer gives them."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf, local_files_only=True)
    text = TEXT.read_text(encoding="utf-8")
    raw = tok(text[:40000], add_special_tokens=False).input_ids
    if len(raw) < WINDOW + LONG:
        raise SystemExit("the text gives %d tokens, fewer than %d" % (len(raw), WINDOW + LONG))

    def chat(chars):
        ids = tok.apply_chat_template([{"role": "user", "content": INSTRUCTION + text[CHAT_OFFSET:CHAT_OFFSET + chars]}], add_generation_prompt=True, tokenize=True)
        return list(ids["input_ids"] if hasattr(ids, "keys") else ids)

    low, high = 0, 8000
    while low < high:
        middle = (low + high + 1) // 2
        low, high = (middle, high) if len(chat(middle)) <= WINDOW else (low, middle - 1)
    out = {"raw": {"ids": raw[:WINDOW], "source": "wiki.test.raw tokens [0, %d)" % WINDOW},
           "chat": {"ids": chat(low), "source": "chat template, one user message: the instruction and wiki.test.raw characters [%d, %d)" % (CHAT_OFFSET, CHAT_OFFSET + low)},
           "long": {"ids": raw[WINDOW:WINDOW + LONG], "source": "wiki.test.raw tokens [%d, %d)" % (WINDOW, WINDOW + LONG)}}
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for name, row in out.items():
        (directory / (name + ".ids")).write_text(" ".join(map(str, row["ids"])) + "\n", encoding="utf-8")
    (directory / "rows.json").write_text(json.dumps(out) + "\n", encoding="utf-8")
    print({name: len(row["ids"]) for name, row in out.items()})


class Experts:
    """One routed layer's experts, each decoded from the file by a pool thread ahead of HF's loop over the experts the router hit."""

    def __init__(self, owner, layer):
        self.owner, self.layer, self.queue, self.pending = owner, layer, [], {}

    def plan(self, hit):
        self.clear()
        self.queue = list(hit)
        self.fill()

    def fill(self):
        while self.queue and len(self.pending) < 2 * AHEAD:
            expert = self.queue.pop(0)
            for kind in ("gate_up", "down"):
                self.pending[(expert, kind)] = self.owner.pool.submit(self.owner.expert, self.layer, expert, kind)

    def take(self, expert, kind):
        future = self.pending.pop((expert, kind), None)
        values = future.result() if future else self.owner.expert(self.layer, expert, kind)
        self.fill()
        return self.owner.torch.from_numpy(values)

    def clear(self):
        for future in self.pending.values():
            future.cancel()
        self.queue, self.pending = [], {}


class ExpertView:
    """What HF's expert loop indexes in place of a stacked parameter: one expert's decoded matrix."""

    def __init__(self, experts, kind):
        self.experts, self.kind = experts, kind

    def __getitem__(self, expert):
        return self.experts.take(int(expert), self.kind)


class FileReference:
    """HF's model of the checkpoint's text config on the file's weights: ordinary modules loaded around their calls, embedding rows read as selected, experts one at a time."""

    def __init__(self, hf, paths, max_decoded_bytes, decode_threads):
        self.torch, modeling, configuration = pins.pinned_environment()
        raw = json.loads((Path(hf) / "config.json").read_text(encoding="utf-8"))
        c = self.config = configuration.MiniMaxM3VLTextConfig(**raw.get("text_config", raw))
        c._attn_implementation = "eager"
        c._experts_implementation = "eager"
        self.source = Shards(paths, max_decoded_bytes, ARCH)
        dense = c.mlp_layer_types.count("dense")
        if c.mlp_layer_types != ["dense"] * dense + ["sparse"] * (c.num_hidden_layers - dense) or set(c.layer_types) != {"minimax_m3_sparse"}:
            raise ValueError("the reference runs a dense lead, then routed layers, every layer sparse")
        expected = {"embedding_length": c.hidden_size, "block_count": c.num_hidden_layers, "leading_dense_block_count": dense,
                    "attention.head_count": c.num_attention_heads, "attention.head_count_kv": c.num_key_value_heads,
                    "attention.key_length": c.head_dim, "attention.value_length": c.head_dim, "attention.layer_norm_rms_epsilon": c.rms_norm_eps,
                    "rope.dimension_count": int(c.head_dim * c.rope_parameters["partial_rotary_factor"]), "rope.freq_base": float(c.rope_parameters["rope_theta"]),
                    "attention.indexer.head_count": c.index_n_heads, "attention.indexer.key_length": c.index_head_dim,
                    "attention.indexer.block_size": c.index_block_size, "attention.indexer.top_k": c.index_topk_blocks,
                    "attention.indexer.local_blocks": c.index_local_blocks, "feed_forward_length": c.dense_intermediate_size,
                    "expert_count": c.num_local_experts, "expert_used_count": c.num_experts_per_tok, "expert_feed_forward_length": c.intermediate_size,
                    "expert_shared_count": 1, "expert_weights_scale": float(c.routed_scaling_factor), "expert_gating_func": 2, "expert_weights_norm": True}
        for key, value in expected.items():
            if self.source.value(ARCH + "." + key) != (spec_decode.to_f32(value) if isinstance(value, float) else value):
                raise ValueError("HF configuration disagrees with GGUF metadata: " + key)
        if c.shared_intermediate_size != c.intermediate_size:
            raise ValueError("the file has one expert width for the shared and routed experts")
        with self.torch.device("meta"):
            self.model = modeling.MiniMaxM3VLForCausalLM(c)
        self.model.eval().requires_grad_(False)
        text = self.model.model
        text.rotary_emb = type(text.rotary_emb)(config=c)
        self.pool = ThreadPoolExecutor(max_workers=decode_threads)
        self.experts = {}
        for i, layer in enumerate(text.layers):
            if hasattr(layer.mlp, "experts"):
                experts = self.experts[i] = Experts(self, i)
                for kind in ("gate_up", "down"):
                    delattr(layer.mlp.experts, kind + "_proj")
                    setattr(layer.mlp.experts, kind + "_proj", ExpertView(experts, kind))
        delattr(text.embed_tokens, "weight")
        text.embed_tokens.weight = self.torch.empty(0, c.hidden_size)
        self.modules = [("model.layers.%d" % i, layer) for i, layer in enumerate(text.layers)]
        self.modules += [("model.norm", text.norm), ("lm_head", self.model.lm_head)]
        self.norm_roundtrip = 0.0
        for prefix, module in self.modules:
            for name, tensor in module.state_dict().items():
                self.parts(prefix + "." + name, tuple(tensor.shape), validate=True)
        e, f, x = c.hidden_size, c.intermediate_size, c.num_local_experts
        for i in self.experts:
            for name, shape in (("ffn_gate_exps", (e, f, x)), ("ffn_up_exps", (e, f, x)), ("ffn_down_exps", (f, e, x))):
                self.check("blk.%d.%s.weight" % (i, name), shape)
        self.check("token_embd.weight", (c.hidden_size, c.vocab_size))

    def check(self, name, shape):
        if name not in self.source.tensors:
            raise ValueError("missing GGUF tensor " + name)
        tensor = self.source.tensors[name][1]
        if tuple(tensor.shape) != tuple(shape) or tensor.type not in spec_decode.NUMPY:
            raise ValueError("unexpected GGUF tensor shape or type: " + name)

    def parts(self, name, shape, validate=False):
        """The HF parameter `name` of `shape` from the file, through the fixture writer's names (tests/fixture_minimax.py)."""
        import numpy as np
        match = re.fullmatch(r"model\.layers\.(\d+)\.(.+)", name)
        kind = "plain"
        if name == "model.norm.weight":
            sources, kind = ["output_norm.weight"], "norm"
        elif name == "lm_head.weight":
            sources = ["output.weight"]
        elif match:
            prefix, suffix = "blk.%s." % match[1], match[2]
            if suffix in names._MATRICES:
                sources = [prefix + names._MATRICES[suffix] + ".weight"]
            elif suffix in names._NORMS:
                sources, kind = [prefix + names._NORMS[suffix] + ".weight"], "norm"
            elif suffix in names._FUSED:
                sources = [prefix + part + ".weight" for part in names._FUSED[suffix]]
            elif suffix == "mlp.gate.e_score_correction_bias":
                sources = [prefix + "exp_probs_b.bias"]
            else:
                raise ValueError("unmapped HF parameter " + name)
        else:
            raise ValueError("unmapped HF parameter " + name)
        part = (shape[0] // len(sources),) + tuple(shape[1:])
        for source in sources:
            self.check(source, tuple(reversed(part)))
        if validate:
            return None
        values = np.concatenate([self.source.read(source).reshape(part) for source in sources]) if len(sources) > 1 else self.source.read(sources[0]).reshape(shape)
        if kind == "norm":
            # The converter stored 1 + w; HF adds the 1 back in F32, so the stored value must come back exactly.
            stored = values
            values = stored - np.float32(1)
            if not np.array_equal(values + np.float32(1), stored):
                raise ValueError("norm offset does not invert exactly: " + name)
        return np.ascontiguousarray(values)

    def expert(self, layer, expert, kind):
        c = self.config
        count = c.hidden_size * c.intermediate_size
        prefix = "blk.%d." % layer
        if kind == "gate_up":
            import numpy as np
            gate = self.source.read(prefix + "ffn_gate_exps.weight", expert * count, count)
            up = self.source.read(prefix + "ffn_up_exps.weight", expert * count, count)
            return np.concatenate((gate.reshape(-1, c.hidden_size), up.reshape(-1, c.hidden_size)))
        return self.source.read(prefix + "ffn_down_exps.weight", expert * count, count).reshape(c.hidden_size, -1)

    def load(self, module, prefix):
        state = {name: self.torch.from_numpy(self.parts(prefix + name, tuple(value.shape))) for name, value in module.state_dict().items()}
        module.load_state_dict(state, strict=True, assign=True)
        if any(t.is_meta for t in list(module.parameters()) + list(module.buffers())):
            raise ValueError("reference module still has meta weights: " + prefix)

    def release(self, module):
        module.to("meta")
        for i, experts in self.experts.items():
            if module is self.model.model.layers[i]:
                experts.clear()

    @contextmanager
    def scope(self, on_load):
        torch, text = self.torch, self.model.model
        hooks = []

        def embed(module, args):
            values, ids = self.source.embedding("token_embd.weight", args[0])
            module.weight = torch.from_numpy(values)
            return (torch.from_numpy(ids),)

        def routed(layer):
            return lambda module, args, output: self.experts[layer].plan(sorted(set(output[2].flatten().tolist())))

        try:
            hooks.append(text.embed_tokens.register_forward_pre_hook(embed))
            hooks.append(text.embed_tokens.register_forward_hook(lambda module, args, output: setattr(module, "weight", torch.empty(0, module.embedding_dim))))
            for i in self.experts:
                hooks.append(text.layers[i].mlp.gate.register_forward_hook(routed(i)))
            with loaded_modules(self.modules, self.load, self.release, on_load), torch.inference_mode():
                yield
        finally:
            for hook in hooks:
                hook.remove()

    def forward(self, ids, last, on_load=None):
        """The logits of the last `last` positions of `ids`, with each layer's least router and index cut margin and the indexers' kept blocks."""
        trace = pins._Trace(self.model, self.torch)
        layers, marks = [], []
        for i, layer in enumerate(self.model.model.layers):
            marks.append(layer.register_forward_hook(lambda module, args, output, i=i: layers.append((i, len(trace.margins["router"]), len(trace.margins["index"])))))
        try:
            with self.scope(on_load), trace.mode:
                logits = self.model(input_ids=ids, use_cache=False, logits_to_keep=last).logits[0]
        finally:
            trace.close()
            for mark in marks:
                mark.remove()
        per_layer, start = [], (0, 0)
        for i, router, index in layers:
            r, x = trace.margins["router"][start[0]:router], trace.margins["index"][start[1]:index]
            per_layer.append({"layer": i, "router_least": min(r) if r else None, "index_cuts": len(x), "index_least": min(x) if x else None,
                              "index_within_1e-3": sum(1 for v in x if v < 1e-3)})
            start = (router, index)
        first = pins._first_sparse(trace.kept, self.config.index_block_size)
        return logits, {"layers": per_layer, "first_sparse_position": first}


def run(args):
    import numpy as np
    out = Path(args.output_dir)
    table = json.loads((out / "rows.json").read_text(encoding="utf-8"))
    runner = FileReference(args.hf, args.gguf, args.max_decoded_mib << 20, args.decode_threads)
    torch = runner.torch
    torch.set_num_threads(args.threads)
    document = {"_comment": "Generated by tools/minimax_file_reference.py run: transformers' MiniMaxM3VLForCausalLM, float32 eager, on the GGUF's own weights as tests/spec_decode.py decodes them.",
                "python_version": pins.PYTHON_VERSION, "torch_version": torch.__version__, "transformers_version": pins.TRANSFORMERS_VERSION,
                "source_sha256": pins.SOURCE_SHA256, "threads": args.threads, "rows": {}}
    for name in args.rows:
        ids = table[name]["ids"]
        last = min(len(ids), WINDOW)
        start = time.time()
        logits, trace = runner.forward(torch.tensor([ids], dtype=torch.long), last,
                                       lambda layer, module: print("HF %s %s %.0fs" % (name, layer, time.time() - start), file=sys.stderr, flush=True))
        if not torch.isfinite(logits).all():
            raise ValueError("nonfinite HF logits in row " + name)
        nll = score_nll(torch, logits, torch.tensor(ids[len(ids) - last:]))
        np.save(out / ("hf-%s.npy" % name), logits.numpy())
        document["rows"][name] = {"tokens": len(ids), "positions": [len(ids) - last, len(ids)], "nll": nll.tolist(), "mean_nll": float(nll.mean()),
                                  "seconds": time.time() - start, **trace}
        print("HF %s done, mean NLL %.6f, %.0fs" % (name, float(nll.mean()), time.time() - start), file=sys.stderr, flush=True)
    document["weights"] = runner.source.labels()
    document["peak_rss_gib"] = peak_gib()
    (out / "hf.json").write_text(json.dumps(document) + "\n", encoding="utf-8")


def llmx_rows(path, vocab):
    """The rows `llmx logits --last N --top V` prints, as {position: float32 row}."""
    import numpy as np
    got = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("tokens:"):
                continue
            fields = line.split()
            if len(fields) != 1 + 2 * vocab:
                raise ValueError("a row of %d fields, not the whole vocabulary" % len(fields))
            row = np.empty(vocab, dtype=np.float32)
            row[np.array(fields[1::2], dtype=np.int64)] = np.array(fields[2::2], dtype=np.float32)
            got[int(fields[0])] = row
    return got


def compare(args):
    """Hold llmx's rows of one row set to HF's at the file-exact class's bounds, frozen before the run (tests/baseline_qwen35.py, FILE_EXACT_BOUNDS)."""
    import numpy as np
    from baseline_qwen35 import BOUNDS, FILE_EXACT_BOUNDS
    bounds = BOUNDS[FILE_EXACT_BOUNDS]
    out = Path(args.output_dir)
    doc = json.loads((out / "hf.json").read_text(encoding="utf-8"))["rows"][args.name]
    ids = json.loads((out / "rows.json").read_text(encoding="utf-8"))[args.name]["ids"]
    hf = np.load(out / ("hf-%s.npy" % args.name))
    first, end = doc["positions"]
    got = llmx_rows(args.llmx, hf.shape[1])
    if sorted(got) != list(range(first, end)):
        raise SystemExit("llmx printed positions %s, HF kept [%d, %d)" % (sorted(got)[:3], first, end))
    llmx = np.stack([got[p] for p in range(first, end)]).astype(np.float64)
    ref = hf.astype(np.float64)
    targets = np.array(ids[first + 1:end])
    lse = np.log(np.exp(llmx[:-1] - llmx[:-1].max(1, keepdims=True)).sum(1)) + llmx[:-1].max(1)
    nll = lse - llmx[np.arange(len(targets)), targets]
    nll_hf = np.array(doc["nll"])
    top_hf, top_llmx = np.argsort(-ref, 1, kind="stable")[:, :5], np.argsort(-llmx, 1, kind="stable")[:, :5]
    overlap = [len(set(a) & set(b)) for a, b in zip(top_hf, top_llmx)]
    flips = [{"position": first + i, "hf": int(top_hf[i, 0]), "llmx": int(top_llmx[i, 0]), "hf_margin": float(ref[i, top_hf[i, 0]] - ref[i, top_hf[i, 1]])}
             for i in range(len(ref)) if top_hf[i, 0] != top_llmx[i, 0]]
    window = abs(float(nll.mean()) - float(nll_hf.mean()))
    report = {"row": args.name, "positions": [first, end], "max_abs_logit": float(np.abs(llmx - ref).max()),
              "max_target_nll": float(np.abs(nll - nll_hf).max()), "mean_nll_hf": float(nll_hf.mean()), "mean_nll_llmx": float(nll.mean()),
              "window_nll": window, "top1_agree": len(ref) - len(flips), "top1_flips": flips, "top5_overlap_min": min(overlap),
              "top5_overlap_last": overlap[-1], "bounds": bounds,
              "pass": {"window_nll": window <= bounds["window_nll"], "top5_overlap_last": overlap[-1] >= bounds["top5_overlap"]},
              "headroom_window_nll": bounds["window_nll"] / window if window else math.inf}
    (out / ("compare-%s.json" % args.name)).write_text(json.dumps(report) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "top1_flips"}))
    return 0 if all(report["pass"].values()) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("rows")
    make.add_argument("--hf", required=True)
    make.add_argument("--output-dir", required=True)
    reference = sub.add_parser("run")
    reference.add_argument("--hf", required=True)
    reference.add_argument("--gguf", required=True, nargs="+")
    reference.add_argument("--output-dir", required=True)
    reference.add_argument("--rows", nargs="+", default=["raw", "chat", "long"])
    reference.add_argument("--threads", type=int, default=8)
    reference.add_argument("--decode-threads", type=int, default=4)
    reference.add_argument("--max-decoded-mib", type=int, default=5120)
    check = sub.add_parser("compare")
    check.add_argument("--output-dir", required=True)
    check.add_argument("--name", required=True)
    check.add_argument("--llmx", required=True)
    args = parser.parse_args(argv)
    if args.command == "rows":
        return rows(args.hf, args.output_dir)
    if args.command == "run":
        return run(args)
    return compare(args)


if __name__ == "__main__":
    sys.exit(main())
