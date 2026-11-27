"""File-exact Inkling-2-Mini HF reference: transformers' own Inkling2LoopForCausalLM in float32 on a GGUF's stored weights, the experts decoded one at a time.

    python -X utf8 tools/inkling2_file_reference.py rows --hf HF_DIR --output-dir OUT
    python -X utf8 tools/inkling2_file_reference.py names --hf HF_DIR --gguf FILE
    python -X utf8 tools/inkling2_file_reference.py run --hf HF_DIR --gguf FILE --output-dir OUT [--rows NAME...] [--threads N] [--kv f16]
    python -X utf8 tools/inkling2_file_reference.py image-rows --hf HF_DIR --mmproj MMPROJ --output-dir OUT --row NAME=IMAGE[,IMAGE...] [--text TEXT]
    python -X utf8 tools/inkling2_file_reference.py greedy --hf HF_DIR --gguf FILE --output-dir OUT --rows NAME... [--steps N]
    python -X utf8 tools/inkling2_file_reference.py compare --output-dir OUT --name NAME --llmx LOGITS.txt
    python -X utf8 tools/inkling2_file_reference.py fixture

rows tokenizes the test texts with the checkpoint's own tokenizer offline, image-rows adds chat rows holding images with HF's vision tower rows on the mmproj's weights (docs/VISION.md, section 12), greedy writes HF's greedy tokens after a row, names checks every tensor name, shape and type the reference reads without reading data, run writes HF's logits of each row, compare holds `llmx logits` output to them, and fixture runs the same reference on the synthetic fixture's file against its golden (docs/INKLING2.md, 6. Correctness).
Every weight but the experts stays resident; rows longer than WINDOW run in CHUNK-token chunks through HF's own cache, keeping the last WINDOW rows whole and every position's top logits, log-sum-exp and target NLL.
--kv f16 rounds every new key after its convolution and clip and every new value after its norm to binary16 before the cache takes them, as llmx's f16 KV cache stores them (docs/PRECISION.md, 4. The test side).
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import sys
import tempfile
import time

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS.parent / "tests"))
sys.path.insert(0, str(TOOLS))
import spec_decode  # noqa: E402
import fixture_inkling2 as names  # noqa: E402
import inkling2_reference as pins  # noqa: E402
from gen_layered_reference import peak_gib  # noqa: E402
from minimax_file_reference import Experts, ExpertView, RouteFollower, llmx_rows, read_routes, compare as compare_rows  # noqa: E402
from qwen4exp_file_reference import Shards  # noqa: E402

ARCH = "inkling2_loop"
TEXT = TOOLS.parent / "tests" / "data" / "wiki.test.raw"
# The rows: 512 raw tokens and a 512-token chat rendering, then 16k raw tokens from the same start and a 16k chat rendering, past the 8192-position log-scale floor.
WINDOW, LONG = 512, 16384
CHUNK = 1024
TOP = 20
# A pick is llmx's where every expert the two picks differ in scores within this many ulps of HF's k-th score, frozen at twice the CPU F32 router's largest deviation over raw[:64], 1536 ulps (docs/INKLING2.md, 6. Correctness).
MARGIN_ULPS = 3072
# The router's normalisation, as llmx's RouteRule gives it: the picks' sigmoid scores over their sum plus this, times route_scale.
ROUTE_EPS = 1e-20
INSTRUCTION = "Summarize the following text.\n\n"
CHAT_OFFSET = 200000
IMAGE_TEXT = "Describe this image in one sentence."


def rows(hf, output_dir):
    """Write rows.json and NAME.ids, each row's ids as HF's tokenizer gives them."""
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf, local_files_only=True)
    text = TEXT.read_text(encoding="utf-8")
    raw = tok(text[:120000], add_special_tokens=False).input_ids
    if len(raw) < LONG:
        raise SystemExit("the text gives %d tokens, fewer than %d" % (len(raw), LONG))

    def chat(chars):
        ids = tok.apply_chat_template([{"role": "user", "content": INSTRUCTION + text[CHAT_OFFSET:CHAT_OFFSET + chars]}], add_generation_prompt=True, tokenize=True)
        return list(ids["input_ids"] if hasattr(ids, "keys") else ids)

    def longest(tokens):
        low, high = 0, 8 * tokens
        while low < high:
            middle = (low + high + 1) // 2
            low, high = (middle, high) if len(chat(middle)) <= tokens else (low, middle - 1)
        return low

    out = {"raw": {"ids": raw[:WINDOW], "source": "wiki.test.raw tokens [0, %d)" % WINDOW},
           "raw16k": {"ids": raw[:LONG], "source": "wiki.test.raw tokens [0, %d)" % LONG}}
    for name, tokens in (("chat", WINDOW), ("chat16k", LONG)):
        chars = longest(tokens)
        out[name] = {"ids": chat(chars), "source": "chat template, one user message: the instruction and wiki.test.raw characters [%d, %d)" % (CHAT_OFFSET, CHAT_OFFSET + chars)}
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for name, row in out.items():
        (directory / (name + ".ids")).write_text(" ".join(map(str, row["ids"])) + "\n", encoding="utf-8")
    (directory / "rows.json").write_text(json.dumps(out) + "\n", encoding="utf-8")
    print({name: len(row["ids"]) for name, row in out.items()})


def image_rows(args):
    """Add chat rows holding images to rows.json: HF's processor renders one user message, the images then the text, and expands each image token; HF's vision tower on the mmproj's own weights in float32 gives their rows, kept in NAME-images.npy."""
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    from transformers import AutoProcessor
    import inkling2_vision_reference as vision
    processor = AutoProcessor.from_pretrained(args.hf, local_files_only=True)
    model, _ = vision.vision_model(args.mmproj, args.hf)
    torch.set_num_threads(args.threads)
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "rows.json"
    table = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for row in args.row:
        name, files = row.split("=", 1)
        files = files.split(",")
        message = [{"role": "user", "content": [{"type": "image"} for _ in files] + [{"type": "text", "text": args.text}]}]
        text = processor.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
        images = [ImageOps.exif_transpose(Image.open(f)) for f in files]
        inputs = processor(text=[text], images=images, return_tensors="pt")
        with torch.no_grad():
            rows = model(inputs["pixel_values"].to(torch.float32), image_grid_thw=inputs["image_grid_thw"]).pooler_output.numpy().astype(np.float32)
        ids = inputs["input_ids"][0].tolist()
        if sum(1 for i in ids if i == processor.image_token_id) != rows.shape[0]:
            raise SystemExit("%s: %d image ids for %d image rows" % (name, sum(1 for i in ids if i == processor.image_token_id), rows.shape[0]))
        np.save(directory / ("%s-images.npy" % name), rows)
        table[name] = {"ids": ids, "image_id": processor.image_token_id, "images": [os.path.basename(f) for f in files], "text": args.text,
                       "source": "chat template, one user message: %d images, then the text" % len(files)}
        (directory / (name + ".ids")).write_text(" ".join(map(str, ids)) + "\n", encoding="utf-8")
        print("%s: %d tokens, %d image rows" % (name, len(ids), rows.shape[0]))
    path.write_text(json.dumps(table) + "\n", encoding="utf-8")


def row_images(directory, table, name):
    """A row's image rows for FileReference.hooks, None for a row without images."""
    import numpy as np
    row = table[name]
    if "image_id" not in row:
        return None
    return {"id": row["image_id"], "rows": np.load(Path(directory) / ("%s-images.npy" % name))}


class FileReference:
    """HF's text model on the file's weights: everything but the embedding and the experts resident, embedding rows read as selected, experts one at a time."""

    def __init__(self, config, paths, max_decoded_bytes, decode_threads, dry=False):
        self.torch, _, modeling, _ = pins.pinned_environment()
        c = self.config = config
        c._attn_implementation = "eager"
        c._experts_implementation = "eager"
        self.source = Shards(paths, max_decoded_bytes, ARCH)
        self.dry, self.seen, problems = dry, set(), []
        a = ARCH + "."
        expected = {"block_count": c.num_hidden_layers, "embedding_length": c.hidden_size, "expert_count": c.n_routed_experts,
                    "expert_used_count": c.num_experts_per_tok, "expert_feed_forward_length": c.moe_intermediate_size,
                    "attention.key_length": c.head_dim, "attention.key_length_swa": c.swa_head_dim, "attention.sliding_window": c.sliding_window_size,
                    "attention.relative_length": c.d_rel, "attention.relative_extent": c.rel_extent, "attention.log_scale_floor": c.log_scaling_n_floor,
                    "loop.start": c.loop_layer_index_start, "loop.end": c.loop_layer_index_end, "loop.count": c.loop_num_passes, "loop.gate_rank": c.loop_gate_rank,
                    "attn_res.block_size": c.n_layers_per_attn_res_block, "vocab_size_unpadded": c.unpadded_vocab_size,
                    "expert_weights_scale": c.route_scale, "attention.qk_clip": c.qk_l2_clip_threshold, "attention.relative_clip": c.rel_l2_clip_threshold,
                    "attention.log_scale_alpha": c.log_scaling_alpha, "final_logit_softcapping": c.final_logit_softcapping,
                    "activation.situ_beta": c.mlp_gate_softcap, "activation.situ_linear_beta": c.mlp_up_softcap}
        for key, value in expected.items():
            if self.source.value(a + key) != (spec_decode.to_f32(value) if isinstance(value, float) else value):
                raise ValueError("HF configuration disagrees with GGUF metadata: " + key)
        if [bool(v) for v in self.source.value(a + "attention.sliding_window_pattern")[1]] != [t == names.SLIDING for t in c.layer_types]:
            raise ValueError("HF layer types disagree with the GGUF's sliding window pattern")
        torch = self.torch
        with torch.device("meta"):
            self.model = modeling.Inkling2LoopForCausalLM(c)
        self.model.eval().requires_grad_(False)
        text = self.model.model
        self.pool = ThreadPoolExecutor(max_workers=decode_threads)
        self.experts, self.follow, self.calls = {}, None, 0
        for i, layer in enumerate(text.layers):
            # Registered before the expert plan, so the plan reads the followed picks; llmx keys its routes by the router call's place in the pass, loop passes included.
            layer.mlp.gate.register_forward_hook(lambda module, args, output: None if self.follow is None else self.follow.follow(self.call(), c.num_experts_per_tok, module.e_score_correction_bias, output))
            experts = self.experts[i] = Experts(self, i)
            for kind in ("gate_up", "down"):
                delattr(layer.mlp.experts, kind + "_proj")
                setattr(layer.mlp.experts, kind + "_proj", ExpertView(experts, kind))
            layer.mlp.gate.register_forward_hook(lambda module, args, output, i=i: self.experts[i].plan(sorted(set(output[2].flatten().tolist()))))
        delattr(text.embed_tokens, "weight")
        text.embed_tokens.weight = torch.empty(0, c.hidden_size)
        text.embed_tokens.padding_idx = None
        state = {}
        for name, value in self.model.state_dict().items():
            if name != "model.embed_tokens.weight":
                try:
                    state[name] = torch.from_numpy(self.parameter(name, tuple(value.shape)))
                except ValueError as error:
                    problems.append("%s: %s" % (name, error))
        e, f, x = c.hidden_size, c.moe_intermediate_size, c.n_routed_experts
        wanted = [("blk.%d.ffn_gate_up_exps.weight" % i, (e, 2 * f, x)) for i in self.experts] + \
            [("blk.%d.ffn_down_exps.weight" % i, (f, e, x)) for i in self.experts] + [("token_embd.weight", (e, c.vocab_size))]
        for name, shape in wanted:
            try:
                self.check(name, shape)
            except ValueError as error:
                problems.append(str(error))
        problems += ["GGUF tensor the reference never reads: " + name for name in sorted(set(self.source.tensors) - self.seen)]
        if problems:
            raise ValueError("the GGUF does not match the HF model:\n" + "\n".join(problems))
        if dry:
            return
        self.model.load_state_dict(state, strict=False, assign=True)
        left = [name for name, t in list(self.model.named_parameters()) + list(self.model.named_buffers()) if t.is_meta]
        if left:
            raise ValueError("reference parameters still on the meta device: %s" % left)

    def check(self, name, shape):
        # A tensor's scale companions (tests/spec_decode.py, weight_tensors) are read with it.
        self.seen.update((name, name[:-len(".weight")] + ".scale", name[:-len(".weight")] + ".input_scale"))
        if name not in self.source.tensors:
            raise ValueError("missing GGUF tensor " + name)
        tensor = self.source.tensors[name][1]
        if tuple(tensor.shape) != tuple(shape) or tensor.type not in spec_decode.NUMPY:
            raise ValueError("unexpected GGUF tensor shape or type: %s %s" % (name, tuple(tensor.shape)))

    def read(self, name, shape):
        self.check(name, tuple(reversed(shape)))
        if self.dry:
            import numpy as np
            return np.zeros(shape, dtype=np.float32)
        return self.source.read(name).reshape(shape)

    def parameter(self, name, shape):
        """The HF parameter `name` of `shape` from the file, through the fixture writer's map (tests/fixture_inkling2.py, map_inkling2)."""
        import numpy as np
        c = self.config
        top = {"model.embed_tokens.embed_norm.weight": "token_embd_norm", "model.norm.weight": "output_norm", "model.final_input.query": "output_res_query"}
        if name in top:
            return self.read(top[name] + ".weight", shape)
        if name == "lm_head.weight":
            # The file's head is padded to the full vocabulary with rows no token has, nonzero in the real files; HF drops them on load.
            values = self.read("output.weight", (c.vocab_size, c.hidden_size))
            return np.ascontiguousarray(values[:c.unpadded_vocab_size])
        if name == "model.message_input.query":
            return np.stack([self.read("blk.%d.loop_msg_query.weight" % i, shape[1:]) for i in range(c.loop_layer_index_start, c.loop_layer_index_end)])
        parts = name.split(".", 3)
        if parts[:2] != ["model", "layers"]:
            raise ValueError("unmapped HF parameter " + name)
        prefix, suffix = "blk.%s." % parts[2], parts[3]
        if suffix in names._STATIC:
            # HF's [K, C] static taps with tap 0 on the current token; the file's [C, K] has the current token last.
            k, channels = shape
            return np.ascontiguousarray(self.read(prefix + names._STATIC[suffix] + ".weight", (channels, k))[:, ::-1].T)
        if suffix in names._TRANSPOSED:
            rows, columns = shape
            return np.ascontiguousarray(self.read(prefix + names._TRANSPOSED[suffix] + ".weight", (columns, rows)).T)
        if suffix == "mlp.gate.e_score_correction_bias":
            return self.read(prefix + "exp_probs_b.bias", shape)
        if suffix not in names._PLAIN:
            raise ValueError("unmapped HF parameter " + name)
        stored = (shape[0], shape[2]) if suffix.endswith("conv1d.weight") else shape
        return self.read(prefix + names._PLAIN[suffix] + ".weight", stored).reshape(shape)

    def expert(self, layer, expert, kind):
        c = self.config
        e, f = c.hidden_size, c.moe_intermediate_size
        if kind == "gate_up":
            # The file holds HF's de-interleaved order, the gate rows first.
            return self.source.read("blk.%d.ffn_gate_up_exps.weight" % layer, expert * 2 * f * e, 2 * f * e).reshape(2 * f, e)
        return self.source.read("blk.%d.ffn_down_exps.weight" % layer, expert * e * f, e * f).reshape(e, f)

    def call(self):
        self.calls += 1
        return self.calls - 1

    def hooks(self, images, kv="f32"):
        """The embedding's hooks: its rows read as selected, and with `images` ({"id", "rows"}) each image id's row of the normed embedding replaced by the next image row, as Inkling2LoopModel's masked_scatter does; the caller sets state["ids"] to each call's ids."""
        torch, text = self.torch, self.model.model
        state = {"ids": None, "next": 0}

        def embed(module, args):
            values, remapped = self.source.embedding("token_embd.weight", args[0])
            module.weight = torch.from_numpy(values)
            return (torch.from_numpy(remapped),)

        def scatter(module, args, output):
            module.weight = torch.empty(0, module.embedding_dim)
            mask = state["ids"] == images["id"] if images else None
            if mask is None or not bool(mask.any()):
                return None
            count = int(mask.sum())
            if state["next"] + count > len(images["rows"]):
                raise ValueError("more image ids than image rows")
            output = output.clone()
            output[mask] = torch.from_numpy(images["rows"][state["next"]:state["next"] + count]).to(output.dtype)
            state["next"] += count
            return output

        hooks = [text.embed_tokens.register_forward_pre_hook(embed), text.embed_tokens.register_forward_hook(scatter)]
        if kv == "f16":
            hooks += [module.register_forward_hook(lambda module, args, output: output.to(torch.float16).to(output.dtype))
                      for module in self.model.modules() if type(module).__name__ == "Inkling2LoopKeyValueProjection"]
            if len(hooks) != 2 + 2 * self.config.num_hidden_layers:
                raise RuntimeError("inkling2 reference: --kv f16 found %d key and value projections, not %d" % (len(hooks) - 2, 2 * self.config.num_hidden_layers))
        return hooks, state

    def greedy(self, ids, steps, images=None):
        """`steps` greedy tokens after `ids`, the prompt whole and then one token a call through HF's DynamicCache."""
        torch = self.torch
        from transformers.cache_utils import DynamicCache
        hooks, state = self.hooks(images)
        cache, out, at, piece = DynamicCache(config=self.config), [], 0, list(ids)
        try:
            with torch.inference_mode():
                for _ in range(steps):
                    state["ids"] = torch.tensor([piece], dtype=torch.long)
                    positions = torch.arange(at, at + len(piece), dtype=torch.long)[None]
                    logits = self.model(input_ids=state["ids"], position_ids=positions, past_key_values=cache, use_cache=True).logits[0, -1].float()
                    at += len(piece)
                    piece = [int(logits.argmax())]
                    out.append(piece[0])
                    print("HF greedy %d: %d" % (len(out), piece[0]), file=sys.stderr, flush=True)
        finally:
            for hook in hooks:
                hook.remove()
            for experts in self.experts.values():
                experts.clear()
        return out

    def forward(self, ids, keep, chunk, on_chunk=None, routes=None, kv="f32", images=None):
        """Run `ids` whole, or in `chunk`-token chunks through HF's DynamicCache; returns the last `keep` logit rows, every position's top logits, log-sum-exp and target NLL, and the router and clip trace.
        Given llmx's `routes` (read_routes), its picks are taken where HF cannot order them, as tools/minimax_file_reference.py takes them, within this file's MARGIN_ULPS."""
        import numpy as np
        torch, c = self.torch, self.config
        from transformers.cache_utils import DynamicCache

        hooks, state = self.hooks(images, kv)
        trace = pins._Trace(self.model, torch)
        n = len(ids)
        tops, top_ids, lse, nll, kept = [], [], [], [], []
        cache = DynamicCache(config=c) if chunk < n else None
        follower = self.follow = RouteFollower(torch, routes, MARGIN_ULPS, ROUTE_EPS, c.route_scale) if routes is not None else None
        try:
            with torch.inference_mode(), trace.mode:
                for at in range(0, n, chunk if cache is not None else n):
                    end = min(n, at + (chunk if cache is not None else n))
                    piece = state["ids"] = torch.tensor([ids[at:end]], dtype=torch.long)
                    positions = torch.arange(at, end, dtype=torch.long)[None]
                    self.calls = 0
                    logits = self.model(input_ids=piece, position_ids=positions, past_key_values=cache, use_cache=cache is not None).logits[0].float()
                    if not torch.isfinite(logits).all():
                        raise ValueError("nonfinite HF logits at positions [%d, %d)" % (at, end))
                    values = logits.double()
                    row_lse = torch.logsumexp(values, dim=-1)
                    lse.append(row_lse.numpy())
                    targets = torch.tensor(ids[at + 1:end + 1], dtype=torch.long)
                    nll.append((row_lse[:len(targets)] - values[:len(targets)].gather(1, targets[:, None]).squeeze(1)).numpy())
                    best = logits.topk(TOP, dim=-1)
                    tops.append(best.values.numpy())
                    top_ids.append(best.indices.numpy().astype(np.int32))
                    if end > n - keep:
                        kept.append(logits[max(0, n - keep - at):].numpy().copy())
                    if on_chunk:
                        on_chunk(at, end)
        finally:
            self.follow = None
            trace.close()
            for hook in hooks:
                hook.remove()
            for experts in self.experts.values():
                experts.clear()
        margins = sorted(trace.margins)
        return np.concatenate(kept), {"top_values": np.concatenate(tops), "top_ids": np.concatenate(top_ids), "lse": np.concatenate(lse), "nll": np.concatenate(nll),
                                      "router_margin": {"minimum": margins[0], "below_1e-5": sum(1 for m in margins if m < 1e-5), "count": len(margins)},
                                      "clipped_rows": [trace.clipped, trace.rows], **({"routes": follower.summary()} if follower else {})}


def text_config(hf):
    _, _, _, configuration = pins.pinned_environment()
    raw = json.loads((Path(hf) / "config.json").read_text(encoding="utf-8"))
    return configuration.Inkling2LoopTextConfig(**raw.get("text_config", raw))


def check_names(args):
    """Every tensor the reference reads, by name, shape and type, and no file tensor left unread, without reading tensor data."""
    FileReference(text_config(args.hf), args.gguf, 1 << 30, 1, dry=True)
    print("PASS")
    return 0


def run(args):
    import numpy as np
    out = Path(args.output_dir)
    table = json.loads((out / "rows.json").read_text(encoding="utf-8"))
    runner = FileReference(text_config(args.hf), args.gguf, args.max_decoded_mib << 20, args.decode_threads)
    torch = runner.torch
    torch.set_num_threads(args.threads)
    for name in args.rows:
        ids = table[name]["ids"]
        keep = min(len(ids), WINDOW)
        start = time.time()
        routes = read_routes(Path(args.routes) / (name + ".routes")) if args.routes else None
        logits, trace = runner.forward(ids, keep, args.chunk, images=row_images(out, table, name), routes=routes, kv=args.kv, on_chunk=lambda at, end: print("HF %s [%d, %d) %.0fs, peak %.1f GiB" % (name, at, end, time.time() - start, peak_gib()), file=sys.stderr, flush=True))
        np.save(out / ("hf-%s.npy" % name), logits)
        np.savez(out / ("hf-%s-positions.npz" % name), **{k: trace.pop(k) for k in ("top_values", "top_ids", "lse", "nll")})
        nll = np.load(out / ("hf-%s-positions.npz" % name))["nll"]
        doc = {"_comment": "Generated by tools/inkling2_file_reference.py run: transformers' Inkling2LoopForCausalLM, float32 eager, on the GGUF's own weights as tests/spec_decode.py decodes them.",
               "python_version": pins.PYTHON_VERSION, "torch_version": torch.__version__, "transformers_version": pins.TRANSFORMERS_VERSION,
               "threads": args.threads, "kv": args.kv, "chunk": args.chunk if args.chunk < len(ids) else None, "tokens": len(ids), "positions": [len(ids) - keep, len(ids)],
               "nll": nll[len(ids) - keep:].tolist(), "mean_nll": float(nll[len(ids) - keep:].mean()), "mean_nll_all": float(nll.mean()),
               "seconds": time.time() - start, "weights": runner.source.labels(), "peak_rss_gib": peak_gib(), **trace}
        (out / ("hf-%s.json" % name)).write_text(json.dumps(doc) + "\n", encoding="utf-8")
        print("HF %s done, mean NLL %.6f, %.0fs" % (name, doc["mean_nll"], doc["seconds"]), file=sys.stderr, flush=True)


def greedy(args):
    """HF's greedy tokens after each row, written to hf-NAME-greedy.json."""
    out = Path(args.output_dir)
    table = json.loads((out / "rows.json").read_text(encoding="utf-8"))
    runner = FileReference(text_config(args.hf), args.gguf, args.max_decoded_mib << 20, args.decode_threads)
    runner.torch.set_num_threads(args.threads)
    for name in args.rows:
        start = time.time()
        tokens = runner.greedy(table[name]["ids"], args.steps, row_images(out, table, name))
        doc = {"_comment": "Generated by tools/inkling2_file_reference.py greedy: transformers' Inkling2LoopForCausalLM, float32 eager, greedy after the row's ids.",
               "tokens": tokens, "seconds": time.time() - start}
        (out / ("hf-%s-greedy.json" % name)).write_text(json.dumps(doc) + "\n", encoding="utf-8")
        print("HF %s greedy: %s" % (name, tokens), file=sys.stderr, flush=True)


def compare(args):
    """Hold llmx's last rows of one row set to HF's at the file-exact class's bounds (tools/minimax_file_reference.py, compare)."""
    out = Path(args.output_dir)
    doc = json.loads((out / ("hf-%s.json" % args.name)).read_text(encoding="utf-8"))
    (out / "hf.json").write_text(json.dumps({"rows": {args.name: doc}}) + "\n", encoding="utf-8")
    return compare_rows(args)


def fixture(args):
    """The reference on the synthetic fixture's file, whole and in chunks, against its HF golden (tests/data/inkling2_reference.json)."""
    import numpy as np
    golden = json.loads((TOOLS.parent / "tests" / "data" / pins.OUTPUT_NAME).read_text(encoding="utf-8"))["fixture"]
    _, _, _, configuration = pins.pinned_environment()
    config = configuration.Inkling2LoopTextConfig(**names.CONFIG)
    with tempfile.TemporaryDirectory(prefix="inkling2_file_reference_") as stage:
        path = names.write_inkling2(Path(stage) / "inkling2.gguf", names.CONFIG, names.make_state(names.CONFIG, golden["seed"]))
        runner = FileReference(config, [path], 1 << 30, 2)
        runner.torch.set_num_threads(1)
        want = np.array(golden["logits"], dtype=np.float64)
        ids = golden["sequence"]
        report = {}
        for label, chunk in (("whole", len(ids)), ("chunked", pins.CHUNK)):
            logits, trace = runner.forward(ids, len(ids), chunk)
            report[label] = float(np.abs(logits.astype(np.float64) - want).max())
        report["nll"] = float(np.abs(trace["nll"] - np.array(golden["nll"])).max())
    print(json.dumps(report))
    passed = report["whole"] <= pins.EQUAL_BOUND and report["chunked"] <= pins.EQUAL_BOUND and report["nll"] <= pins.EQUAL_BOUND
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("rows")
    make.add_argument("--hf", required=True)
    make.add_argument("--output-dir", required=True)
    dry = sub.add_parser("names")
    dry.add_argument("--hf", required=True)
    dry.add_argument("--gguf", required=True, nargs="+")
    reference = sub.add_parser("run")
    reference.add_argument("--hf", required=True)
    reference.add_argument("--gguf", required=True, nargs="+")
    reference.add_argument("--output-dir", required=True)
    reference.add_argument("--rows", nargs="+", default=["raw", "chat", "raw16k", "chat16k"])
    reference.add_argument("--chunk", type=int, default=CHUNK)
    reference.add_argument("--routes", help="the directory of llmx-route-probe's NAME.routes files, whose picks HF takes where it cannot order the experts")
    reference.add_argument("--threads", type=int, default=12)
    reference.add_argument("--decode-threads", type=int, default=4)
    reference.add_argument("--max-decoded-mib", type=int, default=5120)
    reference.add_argument("--kv", choices=("f32", "f16"), default="f32", help="the KV cache type whose stored keys and values HF reads back")
    pictures = sub.add_parser("image-rows")
    pictures.add_argument("--hf", required=True)
    pictures.add_argument("--mmproj", required=True)
    pictures.add_argument("--output-dir", required=True)
    pictures.add_argument("--row", required=True, nargs="+", help="NAME=IMAGE[,IMAGE...]")
    pictures.add_argument("--text", default=IMAGE_TEXT)
    pictures.add_argument("--threads", type=int, default=12)
    decode = sub.add_parser("greedy")
    decode.add_argument("--hf", required=True)
    decode.add_argument("--gguf", required=True, nargs="+")
    decode.add_argument("--output-dir", required=True)
    decode.add_argument("--rows", required=True, nargs="+")
    decode.add_argument("--steps", type=int, default=16)
    decode.add_argument("--threads", type=int, default=12)
    decode.add_argument("--decode-threads", type=int, default=4)
    decode.add_argument("--max-decoded-mib", type=int, default=5120)
    check = sub.add_parser("compare")
    check.add_argument("--output-dir", required=True)
    check.add_argument("--name", required=True)
    check.add_argument("--llmx", required=True)
    sub.add_parser("fixture")
    args = parser.parse_args(argv)
    if args.command == "rows":
        return rows(args.hf, args.output_dir)
    return {"names": check_names, "run": run, "image-rows": image_rows, "greedy": greedy, "compare": compare, "fixture": fixture}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
