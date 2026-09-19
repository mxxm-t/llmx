import os
import sys
import struct
import tempfile
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import run as cli, write_bin, read_bin_floats, max_err

# Regression gate for quant/ + format/: build a random F32 model, quantize it to
# Q8_0 (and Q4_0) via the CLI, dequantize it back, and check the max error is
# within each type's quantization bound.

rng = random.Random(42)


def tensor(rows, cols):
    return [rng.gauss(0.0, 1.0) for _ in range(rows * cols)]


def make_fixtures(d):
    t1 = tensor(512, 256)   # 131072 = 32*4096
    t2 = [rng.gauss(0, 1) for _ in range(256)]
    t3 = tensor(64, 32)
    # Tiny magnitudes on purpose: the per-block scale is amax/127 (Q8_0) or
    # amax/7 (Q4_0), so values around 1e-4 put the scale in the f16 SUBNORMAL
    # band, below the normal minimum of ~6.1e-5 but above the smallest
    # subnormal of ~5.96e-8. A bug in that decode path made every subnormal
    # 16x too large and went unnoticed for the life of the project, because
    # ordinary weights never produce one.
    # Do not shrink this further: below about 1e-5 the scale itself underflows
    # to zero, which is a real f16 limit rather than a defect.
    t4 = [rng.gauss(0.0, 1e-4) for _ in range(256)]

    json_doc = {
        "name": "MyModel",
        "tensors": [
            {"name": "tok_embeddings.weight", "shape": [256, 512]},
            {"name": "norm.weight",           "shape": [256]},
            {"name": "layer.0.weight",        "shape": [32, 64]},
            {"name": "tiny.weight",           "shape": [256]},
        ],
    }
    with open(os.path.join(d, "model.json"), "w") as f:
        import json
        json.dump(json_doc, f, indent=2)

    allf = list(t1) + list(t2) + list(t3) + list(t4)
    write_bin(os.path.join(d, "model.bin"), allf)
    return allf


def run():
    d = tempfile.mkdtemp(prefix="llmx_rt_")
    try:
        original = make_fixtures(d)
        mj, mb, mg = (os.path.join(d, n) for n in ("model.json", "model.bin", "model.gguf"))
        oo, oj, ob = (os.path.join(d, n) for n in ("out.json", "out.bin", "out.gguf"))

        for qtype, bound in (("q8_0", 0.05), ("q4_0", 1.0)):
            rc, _ = cli(["quantize", mj, mb, mg, qtype])
            assert rc == 0, "quantize (%s) failed" % qtype
            rc, _ = cli(["dequantize", mg, oj, ob])
            assert rc == 0, "dequantize (%s) failed" % qtype

            got = read_bin_floats(ob)
            assert len(got) == len(original), "element count mismatch"

            err = max_err(original, got)
            # The tiny tensor needs a RELATIVE check: its absolute error is
            # ~1e-7 no matter how badly the scale decodes, so an absolute
            # bound would pass a 16x subnormal error silently.
            tiny_o, tiny_g = original[-256:], got[-256:]
            scale = max(abs(x) for x in tiny_o)
            tiny_err = max_err(tiny_o, tiny_g) / scale
            assert tiny_err < bound, (
                "%s subnormal-scale round-trip relative error %.6f exceeds %.3f"
                % (qtype, tiny_err, bound))
            # Q8_0: f16 + 7-bit mantissa scale -> ~0.05 for unit-variance gaussian.
            # Q4_0: 4-bit signed range scaled by amax/7 -> ~0.3 typical; 1.0 is a
            # generous bound that still catches real corruption.
            assert err < bound, "%s round-trip error %.6f exceeds bound %.3f" % (
                qtype, err, bound)
            print("roundtrip: %s, %d elements, max abs err = %.6f, "
                  "subnormal-scale rel err = %.6f  [ok]" % (
                      qtype, len(got), err, tiny_err))
        return True
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
