import hashlib
import json
import os
import struct
import subprocess
import tempfile

import common
import fixture_deepseek4

# llmx-encode-image (tools/encode_image.cpp) found beside the executable, held to HF's vision tower at every thread count, the same bits each time (docs/VISION.md, section 7).
# The tiny fixture is held to HF float32, a released mmproj to HF float64 at its goldens' frozen bound, and the tiny deepseek41v tower to DeepSeek's float64 at twice its float32 error in llmx's attention order.
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
TINY_BOUND = 1e-5
THREADS = (1, 4, 16)
# The released mmproj files whose file-exact goldens tests/data/vision holds, by repo and file, since converters name every mmproj alike.
FILE_EXACT = {("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-BF16.gguf"): "file_exact_d312c4d02fd4.json",
              ("unsloth/Inkling-2-Mini-GGUF", "mmproj-BF16.gguf"): "inkling2_file_exact_2c165c78790e.json"}
# The deepseek41v tower's golden (tools/deepseek4_reference.py --vision), its mmproj rebuilt from the golden's seed.
DEEPSEEK = os.path.join(DATA, "deepseek41_vision_reference.json")


class Refused(Exception):
    """The configured device refuses the encoder (common.encoder_refusal)."""


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
    if common.encoder_refusal(p.returncode, p.stderr):
        raise Refused(common.encoder_refusal(p.returncode, p.stderr))
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
            info, data = encode(tool, mmproj, image, os.path.join(directory, "rows.f32"), threads, c.get("max_pixels"))
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
        _, again = encode(tool, mmproj, image, os.path.join(directory, "rows.f32"), THREADS[-1], c.get("max_pixels"))
        assert again == first, "vision %s: %s encoded again differs" % (label, c["file"])
    return errors


def deepseek_tower(tool, directory):
    """The tiny deepseek41v tower, written torch-free from the vision golden's seed, held to DeepSeek's vision.py in float64 at each image's own bound (docs/TESTS.md, vision).
    Every image is encoded at every thread count to the same bits, and a budget below a block of 4 rows is refused."""
    with open(DEEPSEEK, encoding="utf-8") as f:
        doc = json.load(f)
    args, _, _, vstate = fixture_deepseek4.vl_fixture(doc)
    mmproj = str(fixture_deepseek4.write_whole(os.path.join(directory, doc["files"]["mmproj"]["file"]), *fixture_deepseek4.map_mmproj(args, vstate)))
    with open(mmproj, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == doc["files"]["mmproj"]["sha256"], "vision: the deepseek41v mmproj writer's file is not the golden's"
    cases = doc["encoder"]["cases"]
    lines = []
    for c in cases:
        flags = ["--image-max-tokens", str(c["max_tokens"])] if c["max_tokens"] else []
        first = None
        for threads in THREADS:
            cmd = [tool, mmproj, os.path.join(DATA, "images", c["file"]), os.path.join(directory, "rows.f32"), "--dtype", "f32", "--threads", str(threads)] + flags + device_flag()
            p = subprocess.run(cmd, capture_output=True, text=True)
            if common.encoder_refusal(p.returncode, p.stderr):
                raise Refused(common.encoder_refusal(p.returncode, p.stderr))
            assert p.returncode == 0, "vision deepseek41v: %s failed\n%s%s" % (" ".join(cmd), p.stdout, p.stderr)
            info = json.loads(p.stdout)
            assert info["grid"] == c["grid"] and info["tokens"] == c["tokens"] and info["width"] == c["width"], \
                "vision deepseek41v: %s gave grid %s, %d rows of %d, not the official %s, %d of %d" % (c["file"], info["grid"], info["tokens"], info["width"], c["grid"], c["tokens"], c["width"])
            with open(os.path.join(directory, "rows.f32"), "rb") as f:
                data = f.read()
            if first is None:
                first = data
                got = struct.unpack("<%df" % (c["tokens"] * c["width"]), data)
                e = worst(got, [v for row in c["rows"] for v in row], c["tokens"], c["width"])
                assert e <= c["bound"], "vision deepseek41v: %s at grid %s is %.3g of its row scale from float64, past %.3g" % (c["file"], c["grid"], e, c["bound"])
                lines.append("vision: deepseek41v tiny %s grid %dx%d, %d rows: llmx %.3g, official float32 %.3g, in llmx's attention order %.3g, bound %.3g (%gx the latter)"
                             % (c["file"], c["grid"][0], c["grid"][1], c["tokens"], e, c["official_float32_error"], c["llmx_order_float32_error"], c["bound"],
                                doc["encoder"]["multiple"]))
            assert data == first, "vision deepseek41v: %s differs at %d threads" % (c["file"], threads)
    p = subprocess.run([tool, mmproj, os.path.join(DATA, "images", cases[0]["file"]), os.path.join(directory, "rows.f32"), "--image-max-tokens", "3"] + device_flag(),
                       capture_output=True, text=True)
    assert p.returncode == 1 and "vision: an image takes at least 4 tokens, not 3" in p.stderr, (p.returncode, p.stderr)
    print("\n".join(lines))
    print("vision: deepseek41v tiny tower, %d images at %s threads, each within its bound of the official float64 rows; --image-max-tokens 3 refused  [ok]"
          % (len(cases), "/".join(map(str, THREADS))))


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
        try:
            errors = check(tool, mmproj, tiny["cases"], lambda c: c["rows"], TINY_BOUND, directory, "tiny")
        except Refused as refused:
            print("vision: SKIP - the device refuses the encoder: %s" % refused)
            return common.SKIPPED
        print("vision: tiny HF tower, %d images at grids %s, %s threads, worst %.3g of the row scale against %.3g  [ok]"
              % (len(errors), ", ".join("%dx%d" % tuple(c["grid"]) for c in tiny["cases"]), "/".join(map(str, THREADS)), max(errors), TINY_BOUND))
        try:
            deepseek_tower(tool, directory)
        except Refused as refused:
            print("vision: deepseek41v SKIP - the device refuses the encoder: %s" % refused)
        for (repo, name), goldens in FILE_EXACT.items():
            spec = baseline.pinned_fixture(name, repo)
            path = baseline.snapshot_path(spec["repo"], spec["revision"], spec["file"]) if spec else None
            if not path or not path.exists():
                print("vision: file-exact %s %s SKIP - not on disk" % (repo, name))
                continue
            with open(os.path.join(DATA, "vision", goldens), encoding="utf-8") as f:
                doc = json.load(f)
            assert doc["mmproj_sha256"] == spec["sha256"], "vision: %s's goldens were made from another file" % name

            def reference(c):
                with open(os.path.join(DATA, "vision", c["rows"]), "rb") as f:
                    return struct.unpack("<%df" % (c["tokens"] * c["width"]), f.read())

            errors = check(tool, str(path), doc["cases"], reference, doc["bound"], directory, "file-exact")
            for c, e in zip(doc["cases"], errors):
                print("vision: file-exact %s %s grid %dx%d: llmx %.3g, HF float32 %.3g, bound %.3g" % (repo, c["file"], c["grid"][0], c["grid"][1], e,
                                                                                                     c["hf_float32_error"], doc["bound"]))
    return True
