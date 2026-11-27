"""The Inkling-2-Mini image references of tests/image_decode.cpp (docs/VISION.md, section 2), written by `tools/inkling2_vision_reference.py --fit OUTPUT_DIR`.

transformers' Inkling2ImageProcessor never resizes: it pads each image's bottom and right with -1 on the 0..255 scale to whole 40 px tokens, then rescales and normalizes.
An image with transparency as Pillow sees it is composited onto a grey checkerboard first; the reference is the processor given `ImageOps.exif_transpose(Image.open(path))`.
HF's path and URL route decodes to RGB and drops alpha; for every opaque image both routes are checked here to give the same pixel_values.
The fixture holds, per image, its size, whether it is transparent, the token grid and the SHA-256 of pixel_values in llmx's patch layout (tokens row by row, each token's four 20 px patches row by row, each patch (c, y, x), F32).
Images are the decode fixture's own PNGs (tools/image_reference.py) and the ones made here at the pad's edges and for each kind of PNG transparency.
"""

import hashlib
import json
import os
import struct
import sys

TOKEN = 40
PATCH = 20
# Decode fixture PNGs (tests/data/images) the fit cases read: odd sizes, gray and palette conversions, an interlaced file, an EXIF rotation and alpha of each color type.
EXISTING = ["rgb-1x1.png", "rgb-2x3-i.png", "rgb-17x9.png", "rgb-23x37.png", "gray8-23x37.png", "palette8-17x9.png", "photo-300x200.png", "exif-6.png",
            "rgba-1x1.png", "rgba-23x37.png", "rgba-17x9-i.png", "graya-23x37.png", "gray-trns.png", "palette-trns.png"]
# (width, height) of the opaque images made here: exact tokens, one pixel either side of a token, and 4096 px strips.
MADE = [(40, 40), (80, 120), (41, 39), (39, 41), (4096, 1), (1, 4096)]


def _ref():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import image_reference
    return image_reference


def environment():
    import PIL
    import torch
    import torchvision
    import transformers
    return {"python": "%d.%d.%d" % sys.version_info[:3], "torch": torch.__version__, "torchvision": torchvision.__version__,
            "transformers": transformers.__version__, "pillow": PIL.__version__}


def made_images():
    """(name, bytes) of the images made here."""
    import numpy as np
    ref = _ref()
    out = []
    for w, h in MADE:
        img = ref.photo(h, w, w * 7 + h)
        out.append(("inkling-%dx%d.png" % (w, h), ref.png_bytes([[tuple(int(v) for v in px) for px in row] for row in img], 2, 8)))
    rng = np.random.default_rng(23)
    # RGBA at 21x13 under EXIF orientations 5 to 8, so the checkerboard's origin is the turned image's, alpha at the blend's edges among random values.
    img = ref.photo(13, 21, 5)
    edges = [0, 1, 127, 128, 254, 255]
    a = rng.integers(0, 256, (13, 21))
    a.flat[:len(edges)] = edges
    rgba = [[tuple(int(v) for v in img[y][x]) + (int(a[y][x]),) for x in range(21)] for y in range(13)]
    for o in (5, 6, 7, 8):
        out.append(("inkling-rgba-21x13-e%d.png" % o, ref.png_bytes(rgba, 6, 8, chunks_before=(ref.png_chunk(b"eXIf", ref.tiff_orientation(o)),))))
    opaque = [[px[:3] + (255,) for px in row] for row in rgba]
    out.append(("inkling-rgba-opaque.png", ref.png_bytes(opaque, 6, 8)))
    # Palettes: one transparent entry (Pillow's index form), a tRNS shorter than the palette, an empty one, and one after the image data.
    pal = [tuple(int(c) for c in rng.integers(0, 256, 3)) for _ in range(16)]
    idx = [[(int(v),) for v in row] for row in rng.integers(0, 16, (13, 21))]
    out.append(("inkling-palette-index.png", ref.png_bytes(idx, 3, 8, palette=pal, plte_trns=b"\xff\xff\xff\x00\xff")))
    out.append(("inkling-palette-short.png", ref.png_bytes(idx, 3, 8, palette=pal, plte_trns=bytes([0, 128, 254, 1]))))
    out.append(("inkling-palette-empty.png", ref.png_bytes(idx, 3, 8, palette=pal, plte_trns=b"")))
    out.append(("inkling-palette-after.png", ref.png_bytes(idx, 3, 8, palette=pal, chunks_after=(ref.png_chunk(b"tRNS", bytes([0, 128, 254, 1])),))))
    idx4 = [[(int(v),) for v in row] for row in rng.integers(0, 15, (13, 21))]
    out.append(("inkling-palette4-trns.png", ref.png_bytes(idx4, 3, 4, palette=pal[:15], plte_trns=bytes(range(0, 150, 10)))))
    # Gray keys as Pillow stores them: 1-bit as 0 or 255, deeper ones raw against scaled samples, and a key no 8-bit sample reaches.
    for depth, key in ((1, 1), (1, 0), (2, 0), (2, 3), (4, 1), (4, 17), (8, 300)):
        s = [[(int(v),) for v in row] for row in rng.integers(0, 1 << depth, (13, 21))]
        out.append(("inkling-gray%d-key%d.png" % (depth, key), ref.png_bytes(s, 0, depth, plte_trns=struct.pack(">H", key))))
    # RGB keys: one some pixels take, and one whose high bytes are not zero.
    rgb = [[tuple(int(v) for v in rng.integers(0, 2, 3) * 200) for _ in range(21)] for _ in range(13)]
    out.append(("inkling-rgb-key.png", ref.png_bytes(rgb, 2, 8, plte_trns=struct.pack(">HHH", 200, 0, 200))))
    out.append(("inkling-rgb-key-high.png", ref.png_bytes(rgb, 2, 8, plte_trns=struct.pack(">HHH", 0x100 | 200, 0, 200))))
    return out


def truncated_key():
    """A gray PNG whose tRNS holds one byte, which Pillow refuses to open."""
    ref = _ref()
    return ref.png_bytes([[(7,)] * 3] * 2, 0, 8, plte_trns=b"\x07")


def llmx_layout(pixel_values, rows, columns):
    """pixel_values [tokens, 40, 40, 3] (y, x, c) as llmx's patches: token by token, its 2x2 patches row by row, each (c, y, x)."""
    t = pixel_values.reshape(rows * columns, 2, PATCH, 2, PATCH, 3)
    return t.permute(0, 1, 3, 5, 2, 4).contiguous().float()


def fit(output_dir):
    from PIL import Image, ImageOps
    from transformers.models.inkling2.image_processing_inkling2 import Inkling2ImageProcessor
    image_dir = os.path.join(output_dir, "images")
    names = list(EXISTING)
    for name, data in made_images():
        with open(os.path.join(image_dir, name), "wb") as f:
            f.write(data)
        names.append(name)
    proc = Inkling2ImageProcessor()
    cases, refuse = [], []
    for name in names:
        path = os.path.join(image_dir, name)
        with open(path, "rb") as f:
            data = f.read()
        img = ImageOps.exif_transpose(Image.open(path))
        transparent = img.has_transparency_data
        out = proc(images=img, return_tensors="pt")
        _, rows, columns = (int(v) for v in out["image_grid_thw"][0])
        pv = out["pixel_values"]
        assert tuple(pv.shape) == (rows * columns, TOKEN, TOKEN, 3), (name, tuple(pv.shape))
        if not transparent:
            routed = proc(images=path, return_tensors="pt")["pixel_values"]
            assert routed.equal(pv), name + ": the path route gives other pixel_values"
        patches = llmx_layout(pv, rows, columns).numpy().tobytes()
        cases.append({"file": name, "sha256": hashlib.sha256(data).hexdigest(), "size": [img.height, img.width], "transparent": transparent,
                      "max_rows": rows * columns, "grid": [rows * 2, columns * 2], "tokens": rows * columns, "patches_sha256": hashlib.sha256(patches).hexdigest()})
        refuse.append({"file": name, "max_rows": rows * columns - 1, "error": "takes %d tokens, past the bound of %d" % (rows * columns, rows * columns - 1)})
    name = "inkling-gray-key-truncated.png"
    with open(os.path.join(image_dir, name), "wb") as f:
        f.write(truncated_key())
    try:
        Image.open(os.path.join(image_dir, name)).load()
        raise SystemExit(name + ": Pillow opened it")
    except ValueError as e:
        assert "Truncated tRNS" in str(e), e
    doc = {"_comment": "Generated by tools/inkling2_vision_reference.py --fit: transformers' Inkling2ImageProcessor on ImageOps.exif_transpose(Image.open(path)), pixel_values in llmx's patch layout.",
           "environment": environment(), "processor": {"patch": PATCH, "merge": TOKEN // PATCH, "image_mean": proc.image_mean, "image_std": proc.image_std, "pad_value": -1.0},
           "cases": cases, "refuse": refuse, "refuse_decode": [{"file": name, "error": "a truncated tRNS"}]}
    with open(os.path.join(output_dir, "inkling2_image_fit.json"), "w", encoding="ascii") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    print("wrote %d cases, %d transparent" % (len(cases), sum(c["transparent"] for c in cases)))


# The images the encoder reference encodes: one token, one transparent token, a 2 by 1 and a 3 by 2 token grid and a photo of 8 by 5 tokens.
ENCODE_IMAGES = ["rgb-17x9.png", "rgba-23x37.png", "inkling-41x39.png", "inkling-80x120.png", "photo-300x200.png"]


def hf_name(gguf_name):
    """The Inkling2VisionModel parameter a GGUF tensor of the mmproj holds."""
    fixed = {"v.patch_embd.weight": "embeddings.patch_embedding.weight", "v.position_embd.weight": "embeddings.position_embedding.weight",
             "v.class_embd": "embeddings.class_embedding", "v.class_position_embd.weight": "embeddings.cls_position_embedding",
             "v.pre_ln.weight": "pre_layernorm.weight", "v.pre_ln.bias": "pre_layernorm.bias", "mm.1.weight": "merger.linear_1.weight",
             "mm.2.weight": "merger.linear_2.weight"}
    if gguf_name in fixed:
        return fixed[gguf_name]
    _, _, layer, rest = gguf_name.split(".", 3)
    part = {"ln1": "layernorm_before", "ln2": "layernorm_after", "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
            "attn_out": "self_attn.out_proj", "ffn_up": "mlp.fc1", "ffn_down": "mlp.fc2"}
    name, kind = rest.rsplit(".", 1)
    if name in ("ls1", "ls2"):
        return "layers.%s.lambda_%s" % (layer, name[2])
    return "layers.%s.%s.%s" % (layer, part[name], kind)


def vision_model(mmproj, config_dir):
    """HF's Inkling2VisionModel in float32, eager attention, holding the mmproj's own weights, and its vision_config."""
    import numpy as np
    import torch
    from transformers.models.inkling2.configuration_inkling2 import Inkling2VisionConfig
    from transformers.models.inkling2.modeling_inkling2 import Inkling2VisionModel
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
    import spec_decode
    vision = json.load(open(os.path.join(config_dir, "config.json")))["vision_config"]
    config = Inkling2VisionConfig(**vision)
    config._attn_implementation = "eager"
    model = Inkling2VisionModel(config).to(torch.float32).eval()
    params = dict(model.named_parameters())
    g = spec_decode.GGUF(mmproj)
    loaded = {}
    for t in g.tensors:
        values = np.asarray(g.decode(t), dtype=np.float32).reshape(list(reversed(t.shape)))
        if t.name == "v.patch_embd.weight":
            # GGUF [out, c, y, x]; HF's input is a patch (y, x, c).
            values = values.transpose(0, 2, 3, 1).reshape(values.shape[0], -1)
        name = hf_name(t.name)
        if tuple(params[name].shape) != values.shape:
            raise SystemExit("inkling2 encode: %s %s is not %s %s" % (t.name, values.shape, name, tuple(params[name].shape)))
        loaded[name] = values
    if set(loaded) != set(params):
        raise SystemExit("inkling2 encode: unmatched %s / %s" % (sorted(set(params) - set(loaded)), sorted(set(loaded) - set(params))))
    with torch.no_grad():
        for name, p in params.items():
            p.copy_(torch.from_numpy(loaded[name]))
    return model, vision


def encode(output_dir, mmproj, config_dir):
    """HF's Inkling2VisionModel holding the mmproj's own weights, eager attention, in float64 and float32: pooler_output per image and the bound."""
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    from transformers.models.inkling2.configuration_inkling2 import Inkling2VisionConfig
    from transformers.models.inkling2.image_processing_inkling2 import Inkling2ImageProcessor
    from transformers.models.inkling2.modeling_inkling2 import Inkling2VisionModel
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    model, vision = vision_model(mmproj, config_dir)
    config = Inkling2VisionConfig(**vision)
    config._attn_implementation = "eager"
    exact = Inkling2VisionModel(config).to(torch.float64).eval()
    exact.load_state_dict({k: v.to(torch.float64) for k, v in model.state_dict().items()})
    proc = Inkling2ImageProcessor()
    root = os.path.join(output_dir, "vision")
    os.makedirs(root, exist_ok=True)
    with open(mmproj, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    cases = []
    for name in ENCODE_IMAGES:
        feat = proc(images=ImageOps.exif_transpose(Image.open(os.path.join(output_dir, "images", name))), return_tensors="pt")
        with torch.no_grad():
            r64 = exact(feat["pixel_values"].to(torch.float64), image_grid_thw=feat["image_grid_thw"]).pooler_output.numpy()
            r32 = model(feat["pixel_values"].to(torch.float32), image_grid_thw=feat["image_grid_thw"]).pooler_output.numpy().astype(np.float64)
        err = 0.0
        for t in range(r64.shape[0]):
            err = max(err, float(np.abs(r32[t] - r64[t]).max() / max(1.0, float(np.abs(r64[t]).max()))))
        out = "%s-%s.f32" % (digest[:12], os.path.splitext(name)[0])
        with open(os.path.join(root, out), "wb") as f:
            f.write(np.ascontiguousarray(r64, dtype=np.float32).tobytes())
        _, rows, columns = (int(v) for v in feat["image_grid_thw"][0])
        cases.append({"file": name, "grid": [rows * 2, columns * 2], "tokens": int(r64.shape[0]), "width": int(r64.shape[1]), "rows": out, "hf_float32_error": err})
        print("%s: %d tokens, HF float32 error %.3g" % (name, r64.shape[0], err))
    # The bound, frozen from the reference alone before any candidate is compared: twice HF float32's worst per-row error against float64 over every image.
    bound = 2.0 * max(c["hf_float32_error"] for c in cases)
    doc = {"generator": "tools/inkling2_vision_reference.py --encode", "environment": environment(), "mmproj_sha256": digest, "vision_config": vision,
           "dtype": "float64", "attention": "eager", "bound": bound, "cases": cases}
    path = os.path.join(root, "inkling2_file_exact_%s.json" % digest[:12])
    with open(path, "w", encoding="ascii", newline="\n") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    print("wrote %s, bound %.3g" % (path, bound))


USAGE = "usage: inkling2_vision_reference.py --fit tests/data | --encode tests/data MMPROJ HF_CONFIG_DIR"

if __name__ == "__main__":
    if sys.argv[1:] in (["--help"], ["-h"]):
        print(USAGE)
    elif len(sys.argv) == 3 and sys.argv[1] == "--fit":
        fit(sys.argv[2])
    elif len(sys.argv) == 5 and sys.argv[1] == "--encode":
        encode(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        raise SystemExit(USAGE)
