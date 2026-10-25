"""The vision encoder's references of tests/vision.py (docs/VISION.md, section 7), written by `tools/gen_baseline.py vision-tiny` and `vision-file-exact`.

vision-tiny builds a random-weight Qwen3_5VisionModel with HF's own code in the qwen35 reference environment, writes it as an mmproj GGUF the way the converter names and lays out its tensors, and keeps HF's rows for fixture images at three grids.
vision-file-exact holds a released mmproj's every tensor to the checkpoint's model.visual.* bit for bit, then keeps HF's rows in float64 for fixture images with the mmproj's own weights, and the bound: twice HF float32's own worst per-row error against them over every image.
Both run with eager attention, offline; the tiny reference in float32.
"""

import hashlib
import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "tests"))

# The tiny tower: two blocks 32 wide of two 16-wide heads, an MLP of 64, a 4 by 4 learned grid and 48-wide rows.
TINY = {"depth": 2, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2, "out_hidden_size": 48, "patch_size": 16, "spatial_merge_size": 2,
        "temporal_patch_size": 2, "num_position_embeddings": 16, "hidden_act": "gelu_pytorch_tanh", "in_channels": 3, "deepstack_visual_indexes": []}
TINY_SEED = 20261025
# The images each reference encodes, with the pixel bound each runs at (None the processor's own), so three grids are reached.
IMAGES = [("photo-300x200.jpg", None), ("photo-300x200.png", 32 * 32 * 12), ("rgb-23x37.png", None)]
MIN_PIXELS, MAX_PIXELS = 65536, 16777216
IMAGE_MEAN = IMAGE_STD = [0.5, 0.5, 0.5]


def environment():
    import gen_baseline
    torch, transformers, modeling = gen_baseline.qwen35_environment()
    torch.set_num_threads(1)
    return torch, transformers, modeling


def hf_name(gguf_name):
    """The HF parameter of model.visual a GGUF tensor holds, and which temporal frame of the patch kernel, or None."""
    if gguf_name in ("v.patch_embd.weight", "v.patch_embd.weight.1"):
        return "patch_embed.proj.weight", 0 if gguf_name.endswith("weight") else 1
    fixed = {"v.patch_embd.bias": "patch_embed.proj.bias", "v.position_embd.weight": "pos_embed.weight", "v.post_ln.weight": "merger.norm.weight",
             "v.post_ln.bias": "merger.norm.bias", "mm.0.weight": "merger.linear_fc1.weight", "mm.0.bias": "merger.linear_fc1.bias",
             "mm.2.weight": "merger.linear_fc2.weight", "mm.2.bias": "merger.linear_fc2.bias"}
    if gguf_name in fixed:
        return fixed[gguf_name], None
    _, _, layer, rest = gguf_name.split(".", 3)
    part = {"ln1": "norm1", "ln2": "norm2", "attn_qkv": "attn.qkv", "attn_out": "attn.proj", "ffn_up": "mlp.linear_fc1", "ffn_down": "mlp.linear_fc2"}
    name, kind = rest.rsplit(".", 1)
    return "blocks.%s.%s.%s" % (layer, part[name], kind), None


def gguf_tensors(state, depth):
    """The mmproj's tensor names in the converter's order, each with its HF parameter's values as GGUF lays them out (numpy, fastest axis last)."""
    names = ["v.patch_embd.weight", "v.patch_embd.weight.1", "v.patch_embd.bias", "v.position_embd.weight"]
    for l in range(depth):
        for t in ("attn_out", "attn_qkv", "ffn_up", "ffn_down"):
            names += ["v.blk.%d.%s.weight" % (l, t), "v.blk.%d.%s.bias" % (l, t)]
        names += ["v.blk.%d.ln1.weight" % l, "v.blk.%d.ln1.bias" % l, "v.blk.%d.ln2.weight" % l, "v.blk.%d.ln2.bias" % l]
    names += ["v.post_ln.weight", "v.post_ln.bias", "mm.0.weight", "mm.0.bias", "mm.2.weight", "mm.2.bias"]
    out = []
    for n in names:
        hf, frame = hf_name(n)
        v = state[hf]
        if frame is not None:
            v = v[:, :, frame]
        out.append((n, v))
    return out


def write_mmproj(path, config, state):
    """An mmproj GGUF of a Qwen3_5VisionModel's parameters as the converter writes one, every tensor F32."""
    import numpy as np
    import spec_decode
    md = {
        "general.architecture": (spec_decode.STRING, "clip"), "general.type": (spec_decode.STRING, "mmproj"),
        "clip.has_vision_encoder": (7, True), "clip.projector_type": (spec_decode.STRING, "qwen3vl_merger"), "clip.use_gelu": (7, True),
        "clip.vision.projection_dim": (4, config["out_hidden_size"]), "clip.vision.image_size": (4, 768),
        "clip.vision.patch_size": (4, config["patch_size"]), "clip.vision.embedding_length": (4, config["hidden_size"]),
        "clip.vision.feed_forward_length": (4, config["intermediate_size"]), "clip.vision.block_count": (4, config["depth"]),
        "clip.vision.attention.head_count": (4, config["num_heads"]), "clip.vision.attention.layer_norm_epsilon": (6, 1e-6),
        "clip.vision.image_mean": (spec_decode.ARRAY, (6, IMAGE_MEAN)), "clip.vision.image_std": (spec_decode.ARRAY, (6, IMAGE_STD)),
        "clip.vision.spatial_merge_size": (4, config["spatial_merge_size"]),
        "clip.vision.is_deepstack_layers": (spec_decode.ARRAY, (7, [False] * config["depth"])),
    }
    tensors = []
    for name, v in gguf_tensors(state, config["depth"]):
        a = np.ascontiguousarray(v, dtype=np.float32)
        tensors.append((name, list(reversed(a.shape)), 0, a.tobytes()))
    spec_decode.write_gguf(path, md, tensors)


def processor(min_pixels, max_pixels):
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    return Qwen2VLImageProcessor(size={"shortest_edge": min_pixels, "longest_edge": max_pixels}, patch_size=16, temporal_patch_size=2, merge_size=2,
                                 image_mean=IMAGE_MEAN, image_std=IMAGE_STD)


def rows(torch, model, image_path, max_pixels, dtype=None):
    """HF's rows for an image: load_image, the processor at the pixel bounds, then the tower's merged output, in the model's dtype."""
    from transformers.image_utils import load_image
    feat = processor(MIN_PIXELS, max_pixels or MAX_PIXELS)(images=[load_image(image_path)], return_tensors="pt")
    with torch.no_grad():
        out = model(feat["pixel_values"].to(dtype or torch.float32), grid_thw=feat["image_grid_thw"])
    merged = out.pooler_output if hasattr(out, "pooler_output") else out
    grid = [int(v) for v in feat["image_grid_thw"][0]]
    return grid[1:], merged.numpy()


def worst_row_error(got, ref):
    """The largest per-row error of `got` against `ref`, each row's relative to max(1, its largest reference magnitude)."""
    import numpy as np
    w = 0.0
    for t in range(ref.shape[0]):
        w = max(w, float(np.abs(got[t] - ref[t]).max() / max(1.0, float(np.abs(ref[t]).max()))))
    return w


def tiny(output_dir):
    torch, transformers, modeling = environment()
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
    config = Qwen3_5VisionConfig(**TINY)
    config._attn_implementation = "eager"
    torch.manual_seed(TINY_SEED)
    model = modeling.Qwen3_5VisionModel(config).to(torch.float32).eval()
    gen = torch.Generator().manual_seed(TINY_SEED)
    with torch.no_grad():
        for name, p in model.named_parameters():
            scale = 0.1 if "norm" in name and name.endswith("weight") else 0.3
            base = 1.0 if "norm" in name and name.endswith("weight") else 0.0
            p.copy_(base + scale * torch.randn(p.shape, generator=gen, dtype=torch.float32))
    state = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    root = os.path.join(output_dir, "vision")
    os.makedirs(root, exist_ok=True)
    mmproj = os.path.join(root, "tiny-mmproj.gguf")
    write_mmproj(mmproj, TINY, state)
    images = os.path.join(output_dir, "images")
    cases = []
    for name, max_pixels in IMAGES:
        grid, r = rows(torch, model, os.path.join(images, name), max_pixels)
        cases.append({"file": name, "max_pixels": max_pixels, "grid": grid, "tokens": int(r.shape[0]), "width": int(r.shape[1]),
                      "rows": [float(x) for x in r.reshape(-1)]})
    doc = {"generator": "tools/gen_baseline.py vision-tiny", "transformers": transformers.__version__, "torch": torch.__version__, "config": TINY,
           "seed": TINY_SEED, "mmproj_sha256": hashlib.sha256(open(mmproj, "rb").read()).hexdigest(), "dtype": "float32", "attention": "eager", "cases": cases}
    path = os.path.join(output_dir, "baseline_vision_tiny.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f)
        f.write("\n")
    print("wrote %s and %s (%d images)" % (path, mmproj, len(cases)))


def file_exact(output_dir, mmproj, checkpoint):
    """HF's tower from the checkpoint's vision config holding the mmproj's own weights, after every mmproj tensor is held equal to the checkpoint's."""
    import numpy as np
    import spec_decode
    from safetensors import safe_open
    torch, transformers, modeling = environment()
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
    vision = json.load(open(os.path.join(checkpoint, "config.json")))["vision_config"]
    config = Qwen3_5VisionConfig(**vision)
    config._attn_implementation = "eager"
    model = modeling.Qwen3_5VisionModel(config).to(torch.float32).eval()
    g = spec_decode.GGUF(mmproj)
    params = dict(model.named_parameters())
    index = json.load(open(os.path.join(checkpoint, "model.safetensors.index.json")))["weight_map"]
    loaded = {}
    handles = {}
    seen = set()
    for t in g.tensors:
        hf, frame = hf_name(t.name)
        values = np.asarray(g.decode(t), dtype=np.float32).reshape(list(reversed(t.shape)))
        key = "model.visual." + hf
        shard = index[key]
        if shard not in handles:
            handles[shard] = safe_open(os.path.join(checkpoint, shard), framework="pt")
        ref = handles[shard].get_tensor(key).to(torch.float32).numpy()
        if frame is not None:
            ref = ref[:, :, frame]
        if ref.shape != values.shape or not np.array_equal(ref, values):
            raise SystemExit("vision file-exact: %s is not the checkpoint's %s" % (t.name, key))
        if frame is None:
            loaded[hf] = values
        else:
            loaded.setdefault(hf, np.zeros(params[hf].shape, dtype=np.float32))[:, :, frame] = values
        seen.add(hf)
    if seen != set(params):
        raise SystemExit("vision file-exact: the mmproj lacks %s" % sorted(set(params) - seen))
    with torch.no_grad():
        for name, p in params.items():
            p.copy_(torch.from_numpy(loaded[name]))
    exact = modeling.Qwen3_5VisionModel(config).to(torch.float64).eval()
    exact.load_state_dict({k: v.to(torch.float64) for k, v in model.state_dict().items()})
    root = os.path.join(output_dir, "vision")
    os.makedirs(root, exist_ok=True)
    images = os.path.join(output_dir, "images")
    digest = hashlib.sha256(open(mmproj, "rb").read()).hexdigest()
    cases = []
    for name, max_pixels in IMAGES:
        grid, r64 = rows(torch, exact, os.path.join(images, name), max_pixels, torch.float64)
        _, r32 = rows(torch, model, os.path.join(images, name), max_pixels)
        out = "%s-%s.f32" % (digest[:12], os.path.splitext(name)[0] + ("-%d" % max_pixels if max_pixels else ""))
        with open(os.path.join(root, out), "wb") as f:
            f.write(np.ascontiguousarray(r64, dtype=np.float32).tobytes())
        cases.append({"file": name, "max_pixels": max_pixels, "grid": grid, "tokens": int(r64.shape[0]), "width": int(r64.shape[1]), "rows": out,
                      "hf_float32_error": worst_row_error(r32.astype(np.float64), r64)})
    # The bound, frozen from the reference alone before any candidate's rows are compared: twice HF float32's worst per-row error against float64 over every image.
    bound = 2.0 * max(c["hf_float32_error"] for c in cases)
    doc = {"generator": "tools/gen_baseline.py vision-file-exact", "transformers": transformers.__version__, "torch": torch.__version__,
           "mmproj_sha256": digest, "checkpoint_config": vision, "dtype": "float64", "attention": "eager", "bound": bound, "cases": cases}
    path = os.path.join(root, "file_exact_%s.json" % digest[:12])
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    print("wrote %s (%d images, every tensor equal to the checkpoint's)" % (path, len(cases)))


# The tower of the image prompt fixture: the tiny tower with rows the width of tests/image.py's text model.
IMAGE_SEED = 20261026


def hf_processor(transformers, max_pixels):
    """HF's Qwen 3.5 processor over tests/image.py's vocabulary: a byte-level tokenizer without merges, each byte its own id, the vision tokens and the end of text added as special tokens, and the fixture's chat template."""
    import image
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    vocab = {token: i for i, token in enumerate(image.tokens())}
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tok.decoder = decoders.ByteLevel()
    fast = transformers.PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|endoftext|>", additional_special_tokens=image.SPECIAL[1:])
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
    return Qwen3VLProcessor(image_processor=processor(MIN_PIXELS, max_pixels), tokenizer=fast, video_processor=Qwen3VLVideoProcessor(), chat_template=image.TEMPLATE)


def image_tiny(output_dir):
    """tests/image.py's goldens: HF Qwen3_5ForConditionalGeneration holding the qwen35 fixture's text weights and a seeded tiny tower, each case's every prompt row and greedy ids from HF's token-by-token cached forward.
    The prompt is HF's processor's (its template render, tokenizer and pad expansion), and the image rows and positions are HF's own (get_image_features, get_rope_index, and the rope delta its generation adds), stepped through the text model so its linear layers run the recurrence llmx runs; the full forward over the prompt must agree within 1e-4."""
    import math
    import gen_baseline
    import image
    import qwen35
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    torch, transformers, modeling = environment()
    text = gen_baseline.qwen35_hf_config(image.FIXTURE)
    text.pop("architectures")
    vision = dict(TINY, out_hidden_size=text["hidden_size"])
    config = Qwen3_5Config(text_config=text, vision_config=vision, image_token_id=image.IMAGE_PAD, vision_start_token_id=image.VISION_START,
                           vision_end_token_id=image.VISION_END, tie_word_embeddings=image.FIXTURE["tied"])
    config._attn_implementation = config.text_config._attn_implementation = config.vision_config._attn_implementation = "eager"
    model = modeling.Qwen3_5ForConditionalGeneration(config).to(torch.float32).eval()
    raw = qwen35.raw_weights(image.FIXTURE)
    state = {("model.language_model." + name[len("model."):] if name.startswith("model.") else name): torch.tensor(values, dtype=torch.float32).reshape(shape)
             for name, shape, values in raw}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not k.startswith("model.visual.") and k != "lm_head.weight" for k in missing):
        raise SystemExit("image-tiny: the text weights do not load: missing %s, unexpected %s" % (missing, unexpected))
    gen = torch.Generator().manual_seed(IMAGE_SEED)
    with torch.no_grad():
        for name, p in model.model.visual.named_parameters():
            scale = 0.1 if "norm" in name and name.endswith("weight") else 0.3
            base = 1.0 if "norm" in name and name.endswith("weight") else 0.0
            p.copy_(base + scale * torch.randn(p.shape, generator=gen, dtype=torch.float32))
        model.lm_head.weight.copy_(model.model.language_model.embed_tokens.weight)
    if not torch.equal(model.lm_head.weight, model.model.language_model.embed_tokens.weight):
        raise SystemExit("image-tiny: the head is not the tied embedding")
    visual = {k: v.detach().numpy() for k, v in model.model.visual.state_dict().items()}
    root = os.path.join(output_dir, "vision")
    os.makedirs(root, exist_ok=True)
    mmproj = os.path.join(root, "tiny-mmproj-image.gguf")
    write_mmproj(mmproj, vision, visual)
    lm = model.model.language_model
    cases = []
    with gen_baseline.counted_delta_rules(modeling) as calls, torch.no_grad():
        for case in image.CASES:
            # HF's processor end to end: its chat template render, its tokenizer and its pad expansion, beside the image processor at the case's bound.
            hf = hf_processor(transformers, case["max_pixels"] or MAX_PIXELS)
            content = [{"type": "image", "path": os.path.join(output_dir, "images", f)} for f in case["images"]] + [{"type": "text", "text": case["text"]}]
            feat = hf.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
            grid, input_ids, types = feat["image_grid_thw"], feat["input_ids"], feat["mm_token_type_ids"]
            ids = input_ids[0].tolist()
            embeds = lm.embed_tokens(input_ids)
            features = torch.cat(model.model.get_image_features(feat["pixel_values"], grid).pooler_output, dim=0)
            mask, _ = model.model.get_placeholder_mask(input_ids, inputs_embeds=embeds, image_features=features)
            embeds = embeds.masked_scatter(mask, features)
            positions, delta = model.model.get_rope_index(input_ids, types, image_grid_thw=grid)
            full = model(input_ids=input_ids, pixel_values=feat["pixel_values"], image_grid_thw=grid, mm_token_type_ids=types, use_cache=False).logits[0]
            model.model.rope_deltas = None
            cache = gen_baseline.qwen35_cache(lm, torch, transformers)

            def step(e, pos):
                return model.lm_head(lm(inputs_embeds=e, position_ids=pos, past_key_values=cache, use_cache=True).last_hidden_state)[0, -1]

            calls.update(recurrent=0, chunk=0, log_decay=0.0)
            rows = torch.stack([step(embeds[:, i:i + 1], positions[:, :, i:i + 1]) for i in range(len(ids))])
            distance = (full - rows).abs().max().item()
            if not 0 < distance < 1e-4:
                raise SystemExit("image-tiny: the full forward is %.3g from the steps" % distance)
            greedy, gap, last = [], math.inf, rows[-1]
            for k in range(image.GREEDY):
                scores = last.clone()
                scores[image.EOS] = -math.inf
                top = torch.topk(scores, 2)
                gap = min(gap, (top.values[0] - top.values[1]).item())
                greedy.append(int(top.indices[0]))
                if k + 1 < image.GREEDY:
                    # HF's position for a generated token: its index plus the rope delta get_rope_index gave the prompt.
                    pos = (torch.tensor([[[len(ids) + k]]]) + delta).expand(3, 1, 1)
                    last = step(lm.embed_tokens(torch.tensor([[greedy[-1]]])), pos)
            if gap < gen_baseline.QWEN35_GREEDY_GAP:
                raise SystemExit("image-tiny: a greedy step's top two logits are within %.2e" % gap)
            cases.append(dict(case, ids=ids, grids=[[int(v) for v in g[1:]] for g in grid], logits=[[float(v) for v in r] for r in rows.tolist()],
                              greedy=greedy, min_gap=gap, full_forward_distance=distance, recurrent_steps=calls["recurrent"]))
            print("  %r: %d tokens, grids %s, full forward within %.3g, greedy %s" % (case["text"], len(ids), cases[-1]["grids"], distance, greedy))
    doc = {"generator": "tools/gen_baseline.py image-tiny", "transformers": transformers.__version__, "torch": torch.__version__, "dtype": "float32",
           "attention": "eager", "fixture": image.FIXTURE, "template": image.TEMPLATE, "vision": vision, "seed": IMAGE_SEED,
           "mmproj_sha256": hashlib.sha256(open(mmproj, "rb").read()).hexdigest(), "cases": cases}
    path = os.path.join(output_dir, "baseline_image_tiny.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f)
        f.write("\n")
    print("wrote %s and %s (%d cases)" % (path, mmproj, len(cases)))
