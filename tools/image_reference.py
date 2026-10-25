"""The image decoding references of tests/image_decode.cpp (docs/VISION.md, section 2), written by `tools/gen_baseline.py image-decode`.

Each image file is made here: PNG by a writer of this module over zlib, so every color type, bit depth, filter and interlace is reached, and JPEG, GIF and WebP by Pillow.
The reference for each is what the model processor receives: transformers' load_image (decoding, the EXIF orientation and Pillow's conversion to RGB), then Qwen2VLImageProcessor's smart_resize and torchvision resize.
The committed fixture holds the files, the digests of those bytes and the bytes themselves for the small images, so the suite needs neither Pillow nor torchvision.
"""

import hashlib
import io
import json
import os
import struct
import sys
import zlib

# The processor of the Qwen 3.5 and 3.8 checkpoints (their preprocessor_config.json) that the resize cases follow.
PROCESSOR = {"patch_size": 16, "temporal_patch_size": 2, "merge_size": 2, "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
             "min_pixels": 65536, "max_pixels": 16777216}
FACTOR = 32
# The images whose decoded bytes are committed beside their digest, so a mismatch names its first pixel.
SMALL = 64 * 64
# The pixel bounds each resize case runs at: the processor's own, and a small pair that makes the larger images shrink.
RESIZE_BOUNDS = [(PROCESSOR["min_pixels"], PROCESSOR["max_pixels"]), (32 * 32 * 2, 32 * 32 * 12)]


def _env_runtime():
    """torch and transformers from the qwen35 environment, which refuses other versions of them, torchvision and Pillow."""
    torch, transformers, _ = _generator().qwen35_environment()
    return torch, transformers


def _generator():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import gen_baseline
    return gen_baseline


def environment():
    """The pinned reference environment with Pillow's wheel and the codecs it reports; another is refused."""
    gen_baseline = _generator()
    torch, transformers = _env_runtime()
    import importlib.metadata
    import numpy
    import PIL
    from PIL import features
    found = {"pillow_wheel": sorted(line.split(": ", 1)[1] for line in importlib.metadata.distribution("pillow").read_text("WHEEL").splitlines() if line.startswith("Tag: ")),
             "libjpeg_turbo": features.version("libjpeg_turbo"), "zlib": features.version("zlib")}
    if found != gen_baseline.QWEN35_IMAGE_CODECS:
        raise SystemExit("image references are made with %s; this environment has %s" % (gen_baseline.QWEN35_IMAGE_CODECS, found))
    import torchvision
    return {"python": "%d.%d.%d" % sys.version_info[:3], "torch": torch.__version__, "torchvision": torchvision.__version__,
            "transformers": transformers.__version__, "pillow": PIL.__version__, "numpy": numpy.__version__, **found}


def photo(h, w, seed):
    """Photograph-like content: smooth gradients and waves, sharp edges and a little noise."""
    import numpy as np
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    r = 128 + 100 * np.sin(x / max(w, 1) * 6.0 + seed) * np.cos(y / max(h, 1) * 3.0)
    g = 255 * (x + y) / max(h + w - 2, 1)
    b = np.where((x // 7 + y // 5) % 2 == 0, 40.0, 210.0)
    img = np.stack([r, g, b], axis=-1) + rng.normal(0, 6, (h, w, 3))
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def png_chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)


def png_bytes(samples, color, depth, interlace=False, palette=None, chunks_before=(), chunks_after=(), idat_parts=1, plte_trns=None):
    """A PNG of `samples` (rows of per-pixel sample tuples) with every row filtered by the next of the five filters in turn."""
    h, w = len(samples), len(samples[0])
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color]
    bpp = max(1, channels * depth // 8)

    def pack(row):
        if depth == 8:
            return bytes(v for px in row for v in px)
        out, acc, bits = bytearray(), 0, 0
        for px in row:
            acc = (acc << depth) | px[0]
            bits += depth
            if bits == 8:
                out.append(acc)
                acc, bits = 0, 0
        if bits:
            out.append(acc << (8 - bits))
        return bytes(out)

    def filtered(lines):
        out, prev, n = bytearray(), None, 0
        for line in lines:
            f = n % 5
            n += 1
            cur = bytearray()
            for i, v in enumerate(line):
                a = line[i - bpp] if i >= bpp else 0
                b = prev[i] if prev is not None else 0
                c = prev[i - bpp] if prev is not None and i >= bpp else 0
                if f == 0:
                    p = 0
                elif f == 1:
                    p = a
                elif f == 2:
                    p = b
                elif f == 3:
                    p = (a + b) >> 1
                else:
                    pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                    p = a if pa <= pb and pa <= pc else b if pb <= pc else c
                cur.append((v - p) & 255)
            out.append(f)
            out += cur
            prev = line
        return bytes(out)

    if interlace:
        raw = b""
        for x0, y0, dx, dy in ((0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)):
            rows = [[samples[y][x] for x in range(x0, w, dx)] for y in range(y0, h, dy)]
            rows = [r for r in rows if r]
            if rows:
                raw += filtered([pack(r) for r in rows])
    else:
        raw = filtered([pack(r) for r in samples])
    z = zlib.compress(raw, 9)
    parts = [z[i * len(z) // idat_parts:(i + 1) * len(z) // idat_parts] for i in range(idat_parts)]
    out = b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, depth, color, 0, 0, 1 if interlace else 0))
    for c in chunks_before:
        out += c
    if palette is not None:
        out += png_chunk(b"PLTE", bytes(v for px in palette for v in px))
    if plte_trns is not None:
        out += png_chunk(b"tRNS", plte_trns)
    for part in parts:
        out += png_chunk(b"IDAT", part)
    for c in chunks_after:
        out += c
    return out + png_chunk(b"IEND", b"")


def tiff_orientation(value, big_endian=False, kind=3, count=1):
    """A TIFF-structured EXIF block whose IFD0 holds one orientation entry."""
    e = ">" if big_endian else "<"
    head = (b"MM\x00\x2a" if big_endian else b"II\x2a\x00") + struct.pack(e + "I", 8)
    if kind == 3:
        payload = struct.pack(e + "HH", value, 0)
    else:
        payload = struct.pack(e + "I", value)
    return head + struct.pack(e + "H", 1) + struct.pack(e + "HHI", 0x0112, kind, count) + payload + struct.pack(e + "I", 0)


def raw_profile(exif):
    text = exif.hex()
    lines = "\n".join(text[i:i + 72] for i in range(0, len(text), 72))
    return "\nexif\n%8d\n%s\n" % (len(exif), lines)


def xmp(value):
    return '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF><rdf:Description tiff:Orientation="%d"/></rdf:RDF></x:xmpmeta>' % value


def png_cases():
    """(name, bytes) of every PNG the decode cases read."""
    import numpy as np
    cases = []
    rng = np.random.default_rng(7)
    for interlace in (False, True):
        tag = "-i" if interlace else ""
        for h, w in ((37, 23), (1, 1), (3, 2), (9, 17)):
            size = "%dx%d" % (w, h)
            for depth in (1, 2, 4, 8):
                s = [[(int(v),) for v in row] for row in rng.integers(0, 1 << depth, (h, w))]
                cases.append(("gray%d-%s%s.png" % (depth, size, tag), png_bytes(s, 0, depth, interlace)))
                entries = max(1, (1 << depth) - 1)
                pal = [tuple(int(c) for c in rng.integers(0, 256, 3)) for _ in range(entries)]
                s = [[(int(v),) for v in row] for row in rng.integers(0, entries, (h, w))]
                cases.append(("palette%d-%s%s.png" % (depth, size, tag), png_bytes(s, 3, depth, interlace, palette=pal)))
            img = photo(h, w, h * 31 + w)
            a = rng.integers(0, 256, (h, w))
            cases.append(("rgb-%s%s.png" % (size, tag), png_bytes([[tuple(int(v) for v in px) for px in row] for row in img], 2, 8, interlace)))
            cases.append(("rgba-%s%s.png" % (size, tag), png_bytes([[tuple(int(v) for v in px) + (int(a[y][x]),) for x, px in enumerate(row)] for y, row in enumerate(img)], 6, 8, interlace)))
            cases.append(("graya-%s%s.png" % (size, tag), png_bytes([[(int(px[0]), int(a[y][x])) for x, px in enumerate(row)] for y, row in enumerate(img)], 4, 8, interlace)))
    img = photo(48, 40, 3)
    rgb = [[tuple(int(v) for v in px) for px in row] for row in img]
    gray = [[(int(px[1]),) for px in row] for row in img]
    cases.append(("gray-trns.png", png_bytes(gray, 0, 8, chunks_after=(), plte_trns=struct.pack(">H", 40))))
    pal = [tuple(int(c) for c in rng.integers(0, 256, 3)) for _ in range(16)]
    cases.append(("palette-trns.png", png_bytes([[(int(v) % 16,) for v in row[:, 0]] for row in img], 3, 8, palette=pal, plte_trns=bytes(range(0, 160, 10)))))
    cases.append(("rgb-many-idat.png", png_bytes(rgb, 2, 8, idat_parts=5)))
    cases.append(("photo-300x200.png", png_bytes([[tuple(int(v) for v in px) for px in row] for row in photo(200, 300, 11)], 2, 8)))
    cases.append(("exif-6.png", png_bytes(rgb, 2, 8, chunks_before=(png_chunk(b"eXIf", tiff_orientation(6)),))))
    cases.append(("exif-after-idat-5.png", png_bytes(rgb, 2, 8, chunks_after=(png_chunk(b"eXIf", tiff_orientation(5, big_endian=True)),))))
    cases.append(("text-exif-2.png", png_bytes(rgb, 2, 8, chunks_before=(png_chunk(b"tEXt", b"exif\0Exif\0\0" + tiff_orientation(2)),))))
    cases.append(("raw-profile-3.png", png_bytes(rgb, 2, 8, chunks_before=(png_chunk(b"tEXt", b"Raw profile type exif\0" + raw_profile(b"Exif\0\0" + tiff_orientation(3)).encode()),))))
    cases.append(("raw-profile-z-7.png", png_bytes(rgb, 2, 8, chunks_before=(png_chunk(b"zTXt", b"Raw profile type exif\0\0" + zlib.compress(raw_profile(tiff_orientation(7)).encode())),))))
    cases.append(("xmp-8.png", png_bytes(rgb, 2, 8, chunks_before=(png_chunk(b"iTXt", b"XML:com.adobe.xmp\0\0\0\0\0" + xmp(8).encode()),))))
    cases.append(("xmp-z-6.png", png_bytes(rgb, 2, 8, chunks_after=(png_chunk(b"iTXt", b"XML:com.adobe.xmp\0\1\0\0\0" + zlib.compress(xmp(6).encode())),))))
    cases.append(("exif-1-xmp-6.png", png_bytes(rgb, 2, 8, chunks_before=(png_chunk(b"eXIf", tiff_orientation(1)), png_chunk(b"iTXt", b"XML:com.adobe.xmp\0\0\0\0\0" + xmp(6).encode())))))
    return cases


def jpeg_cases():
    """(name, bytes) of every JPEG, GIF and WebP the decode cases read, written by Pillow."""
    import numpy as np
    from PIL import Image
    cases = []

    def save(img, **kw):
        b = io.BytesIO()
        img.save(b, **kw)
        return b.getvalue()

    for h, w in ((1, 1), (2, 2), (2, 3), (3, 5), (5, 3), (4, 9), (15, 17), (16, 16), (17, 33), (48, 64), (97, 129), (200, 300)):
        im = Image.fromarray(photo(h, w, h + 7 * w))
        size = "%dx%d" % (w, h)
        for sub, name in ((0, "444"), (1, "422"), (2, "420")):
            cases.append(("%s-%s.jpg" % (name, size), save(im, format="JPEG", quality=90, subsampling=sub)))
            cases.append(("%s-%s-prog.jpg" % (name, size), save(im, format="JPEG", quality=90, subsampling=sub, progressive=True)))
        cases.append(("gray-%s.jpg" % size, save(im.convert("L"), format="JPEG", quality=90)))
    im = Image.fromarray(photo(48, 64, 5))
    for q in (30, 75, 100):
        cases.append(("420-q%d.jpg" % q, save(im, format="JPEG", quality=q, subsampling=2)))
    cases.append(("420-optimized.jpg", save(im, format="JPEG", quality=85, subsampling=2, optimize=True)))
    cases.append(("420-prog-optimized.jpg", save(im, format="JPEG", quality=85, subsampling=2, progressive=True, optimize=True)))
    cases.append(("420-restart-blocks.jpg", save(im, format="JPEG", quality=85, subsampling=2, restart_marker_blocks=3)))
    cases.append(("422-restart-rows.jpg", save(im, format="JPEG", quality=85, subsampling=1, restart_marker_rows=1)))
    cases.append(("420-prog-restart.jpg", save(im, format="JPEG", quality=85, subsampling=2, progressive=True, restart_marker_blocks=2)))
    cases.append(("rgb-adobe.jpg", save(im, format="JPEG", quality=90, subsampling=0, keep_rgb=True)))
    cases.append(("photo-300x200.jpg", save(Image.fromarray(photo(200, 300, 11)), format="JPEG", quality=92)))
    rot = Image.fromarray(photo(17, 33, 9))
    for o in range(0, 10):
        exif = Image.Exif()
        exif[0x0112] = o
        cases.append(("exif-%d.jpg" % o, save(rot, format="JPEG", quality=90, exif=exif.tobytes())))
    cases.append(("exif-mm-6.jpg", save(rot, format="JPEG", quality=90, exif=b"Exif\0\0" + tiff_orientation(6, big_endian=True))))
    cases.append(("exif-long-8.jpg", save(rot, format="JPEG", quality=90, exif=b"Exif\0\0" + tiff_orientation(8, kind=4))))
    cases.append(("exif-count2.jpg", save(rot, format="JPEG", quality=90, exif=b"Exif\0\0" + tiff_orientation(6, count=2))))
    cases.append(("xmp-5.jpg", save(rot, format="JPEG", quality=90, xmp=xmp(5).encode())))
    cases.append(("xmp-attr-close-7.jpg", save(rot, format="JPEG", quality=90, xmp=b"<x><tiff:Orientation>7</tiff:Orientation></x>")))
    return cases


def refusal_cases():
    """(name, bytes, the text llmx's refusal holds) of the files llmx refuses by name."""
    from PIL import Image
    im = Image.fromarray(photo(16, 24, 2))
    b = io.BytesIO()
    im.save(b, format="JPEG", quality=90)
    base = b.getvalue()
    sof = base.index(b"\xff\xc0")

    def with_sof(marker):
        return base[:sof + 1] + bytes([marker]) + base[sof + 2:]

    cmyk, gif, webp = io.BytesIO(), io.BytesIO(), io.BytesIO()
    im.convert("CMYK").save(cmyk, format="JPEG")
    im.save(gif, format="GIF")
    im.save(webp, format="WEBP")
    rgb16 = png_bytes([[(10, 20, 30)] * 3] * 2, 2, 8)
    rgb16 = rgb16[:8] + png_chunk(b"IHDR", struct.pack(">IIBBBBB", 3, 2, 16, 2, 0, 0, 0)) + rgb16[33:]
    good = png_bytes([[(1, 2, 3)] * 3] * 2, 2, 8)
    bad_crc = good[:-5] + bytes([good[-5] ^ 1]) + good[-4:]
    pal = png_bytes([[(5,)] * 3] * 2, 3, 8, palette=[(0, 0, 0), (9, 9, 9)])
    return [
        ("refuse-cmyk.jpg", cmyk.getvalue(), "CMYK"),
        ("refuse-arith.jpg", with_sof(0xc9), "arithmetic coding"),
        ("refuse-lossless.jpg", with_sof(0xc3), "lossless"),
        ("refuse-12bit.jpg", base[:sof + 4] + b"\x0c" + base[sof + 5:], "8-bit samples"),
        ("refuse.gif", gif.getvalue(), "GIF is not supported"),
        ("refuse.webp", webp.getvalue(), "WebP is not supported"),
        ("refuse-rgb16.png", rgb16, "16-bit samples"),
        ("refuse-crc.png", bad_crc, "CRC"),
        ("refuse-palette-index.png", pal, "palette index past the palette"),
        ("refuse-text.txt", b"not an image\n", "not a PNG or JPEG file"),
    ]


def smart_resize_cases():
    """HF's smart_resize over sizes that reach each branch, its half-to-even ties and its refusal."""
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    sizes = [(1, 1), (16, 16), (48, 80), (80, 48), (112, 144), (1080, 1920), (1920, 1080), (3000, 4000), (4096, 4096), (5000, 5000),
             (17, 3400), (10, 2001), (10, 2000), (1, 200), (1, 201), (8191, 2047), (65535, 65535), (777, 333), (240, 16), (16, 240)]
    bounds = RESIZE_BOUNDS + [(4096, 65536), (32 * 32, 32 * 32)]
    out = []
    for h, w in sizes:
        for mn, mx in bounds:
            case = {"height": h, "width": w, "factor": FACTOR, "min_pixels": mn, "max_pixels": mx}
            try:
                case["result"] = list(smart_resize(h, w, factor=FACTOR, min_pixels=mn, max_pixels=mx))
            except ValueError:
                case["result"] = "error"
            out.append(case)
    return out


def generate(output_dir):
    env = environment()
    import numpy as np
    from transformers.image_utils import SizeDict, load_image
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    proc = Qwen2VLImageProcessor(size={"shortest_edge": PROCESSOR["min_pixels"], "longest_edge": PROCESSOR["max_pixels"]},
                                 patch_size=PROCESSOR["patch_size"], temporal_patch_size=PROCESSOR["temporal_patch_size"],
                                 merge_size=PROCESSOR["merge_size"], image_mean=PROCESSOR["image_mean"], image_std=PROCESSOR["image_std"])
    root = os.path.join(output_dir, "images")
    os.makedirs(root, exist_ok=True)
    doc = {"generator": "tools/gen_baseline.py image-decode", "environment": env, "processor": PROCESSOR,
           "decode": [], "resize": [], "refuse": [], "smart_resize": smart_resize_cases()}
    for name, data in png_cases() + jpeg_cases():
        path = os.path.join(root, name)
        with open(path, "wb") as f:
            f.write(data)
        img = load_image(path)
        rgb = np.asarray(img, dtype=np.uint8)
        entry = {"file": name, "sha256": hashlib.sha256(data).hexdigest(), "width": int(rgb.shape[1]), "height": int(rgb.shape[0]),
                 "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest()}
        if rgb.shape[0] * rgb.shape[1] <= SMALL:
            entry["rgb"] = name + ".rgb"
            with open(os.path.join(root, entry["rgb"]), "wb") as f:
                f.write(rgb.tobytes())
        doc["decode"].append(entry)
        if rgb.shape[0] * rgb.shape[1] >= 15 * 15:
            tensor = proc.process_image(img)
            for mn, mx in RESIZE_BOUNDS:
                out = proc.resize(images=tensor[None], size=SizeDict(shortest_edge=mn, longest_edge=mx), resample=proc.resample, factor=FACTOR)[0]
                hwc = out.permute(1, 2, 0).contiguous().numpy()
                doc["resize"].append({"file": name, "min_pixels": mn, "max_pixels": mx, "height": int(hwc.shape[0]), "width": int(hwc.shape[1]),
                                      "rgb_sha256": hashlib.sha256(hwc.tobytes()).hexdigest()})
    for name, data, error in refusal_cases():
        with open(os.path.join(root, name), "wb") as f:
            f.write(data)
        doc["refuse"].append({"file": name, "sha256": hashlib.sha256(data).hexdigest(), "error": error})
    path = os.path.join(output_dir, "image_decode.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    print("wrote %s: %d decodes, %d resizes, %d refusals, %d smart_resize cases" % (path, len(doc["decode"]), len(doc["resize"]), len(doc["refuse"]), len(doc["smart_resize"])))
