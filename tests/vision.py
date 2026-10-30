import hashlib
import json
import os
import struct
import subprocess
import tempfile

import common

# llmx-encode-image (tools/encode_image.cpp) found beside the executable, held to HF's vision tower at every thread count, the same bits each time (docs/VISION.md, section 7).
# The tiny fixture is held to HF float32 at TINY_BOUND of each row's scale, and a released mmproj on disk to HF float64 at the bound its goldens froze from HF float32's own error.
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
TINY_BOUND = 1e-5
THREADS = (1, 4, 16)
# The released mmproj files whose file-exact goldens tests/data/vision holds, by the SHA-256 that names their goldens.
FILE_EXACT = {"mmproj-BF16.gguf": "file_exact_d312c4d02fd4.json"}


def tool_path():
    return os.path.join(os.path.dirname(common.exe_path()), "llmx-encode-image" + (".exe" if os.name == "nt" else ""))


def device_flag():
    """The configured device where it is one, which the encoder runs on; a device list or none runs it on the CPU."""
    configured = os.environ.get("LLMX_DEVICE")
    return ["--device", configured] if configured and "," not in configured else []


def encode(tool, mmproj, image, out, threads, max_pixels):
    cmd = [tool, mmproj, image, out, "--dtype", "f32", "--threads", str(threads)] + device_flag()
    if max_pixels:
        cmd += ["--image-max-tokens", str(max_pixels // (32 * 32))]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, "vision: %s failed\n%s%s" % (" ".join(cmd), p.stdout, p.stderr)
    info = json.loads(p.stdout)
    with open(out, "rb") as f:
        data = f.read()
    return info, data


def worst(got, ref, tokens, width):
    """The largest per-row error, each row's relative to max(1, its largest reference magnitude)."""
    w = 0.0
    for t in range(tokens):
        row = ref[t * width:(t + 1) * width]
        scale = max(1.0, max(abs(x) for x in row))
        w = max(w, max(abs(got[t * width + i] - row[i]) for i in range(width)) / scale)
    return w


def check(tool, mmproj, cases, rows_of, bound, directory, label):
    """Each case encoded at every thread count: the same bits each time, its grid and rows within `bound`; the worst error of each image."""
    errors = []
    for c in cases:
        image = os.path.join(DATA, "images", c["file"])
        first = None
        for threads in THREADS:
            info, data = encode(tool, mmproj, image, os.path.join(directory, "rows.f32"), threads, c["max_pixels"])
            assert info["grid"] == c["grid"] and info["tokens"] == c["tokens"] and info["width"] == c["width"], \
                "vision %s: %s gave grid %s, %d tokens of %d, not HF's %s, %d of %d" % (label, c["file"], info["grid"], info["tokens"], info["width"],
                                                                                        c["grid"], c["tokens"], c["width"])
            if first is None:
                first = data
                got = struct.unpack("<%df" % (c["tokens"] * c["width"]), data)
                e = worst(got, rows_of(c), c["tokens"], c["width"])
                assert e <= bound, "vision %s: %s at grid %s is %.3g of its row scale from HF, past %.3g" % (label, c["file"], c["grid"], e, bound)
                errors.append(e)
            assert data == first, "vision %s: %s differs at %d threads" % (label, c["file"], threads)
        _, again = encode(tool, mmproj, image, os.path.join(directory, "rows.f32"), THREADS[-1], c["max_pixels"])
        assert again == first, "vision %s: %s encoded again differs" % (label, c["file"])
    return errors


def run(require=False):
    tool = tool_path()
    if not os.path.exists(tool):
        assert not require, "vision: %s not found beside the executable" % tool
        print("vision: SKIP - %s not found beside the executable" % tool)
        return common.SKIPPED
    import baseline
    with open(os.path.join(DATA, "baseline_vision_tiny.json"), encoding="utf-8") as f:
        tiny = json.load(f)
    mmproj = os.path.join(DATA, "vision", "tiny-mmproj.gguf")
    with open(mmproj, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == tiny["mmproj_sha256"], "vision: the tiny mmproj is not the one its goldens were made from"
    with tempfile.TemporaryDirectory(prefix="llmx_vision_") as directory:
        errors = check(tool, mmproj, tiny["cases"], lambda c: c["rows"], TINY_BOUND, directory, "tiny")
        print("vision: tiny HF tower, %d images at grids %s, %s threads, worst %.3g of the row scale against %.3g  [ok]"
              % (len(errors), ", ".join("%dx%d" % tuple(c["grid"]) for c in tiny["cases"]), "/".join(map(str, THREADS)), max(errors), TINY_BOUND))
        for name, goldens in FILE_EXACT.items():
            spec = baseline.pinned_fixture(name)
            path = baseline.snapshot_path(spec["repo"], spec["revision"], spec["file"]) if spec else None
            if not path or not path.exists():
                print("vision: file-exact %s SKIP - not on disk" % name)
                continue
            with open(os.path.join(DATA, "vision", goldens), encoding="utf-8") as f:
                doc = json.load(f)
            assert doc["mmproj_sha256"] == spec["sha256"], "vision: %s's goldens were made from another file" % name

            def reference(c):
                with open(os.path.join(DATA, "vision", c["rows"]), "rb") as f:
                    return struct.unpack("<%df" % (c["tokens"] * c["width"]), f.read())

            errors = check(tool, str(path), doc["cases"], reference, doc["bound"], directory, "file-exact")
            for c, e in zip(doc["cases"], errors):
                print("vision: file-exact %s %s grid %dx%d: llmx %.3g, HF float32 %.3g, bound %.3g" % (name, c["file"], c["grid"][0], c["grid"][1], e,
                                                                                                     c["hf_float32_error"], doc["bound"]))
    return True
