import os
import sys
import struct
import tempfile
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import run as cli, write_bin, read_bin_floats, max_err

# Regression gate for quant/ + format/: build a random F32 model, quantize it to
# Q8_0 via the CLI, dequantize it back, and check the max error is within the
# Q8_0 quantization bound.

rng = random.Random(42)


def tensor(rows, cols):
    return [rng.gauss(0.0, 1.0) for _ in range(rows * cols)]


def make_fixtures(d):
    t1 = tensor(512, 256)   # 131072 = 32*4096
    t2 = [rng.gauss(0, 1) for _ in range(256)]
    t3 = tensor(64, 32)

    json_doc = {
        "name": "MyModel",
        "tensors": [
            {"name": "tok_embeddings.weight", "shape": [256, 512]},
            {"name": "norm.weight",           "shape": [256]},
            {"name": "layer.0.weight",        "shape": [32, 64]},
        ],
    }
    with open(os.path.join(d, "model.json"), "w") as f:
        import json
        json.dump(json_doc, f, indent=2)

    allf = list(t1) + list(t2) + list(t3)
    write_bin(os.path.join(d, "model.bin"), allf)
    return allf


def run():
    d = tempfile.mkdtemp(prefix="llmx_rt_")
    try:
        original = make_fixtures(d)
        mj, mb, mg = (os.path.join(d, n) for n in ("model.json", "model.bin", "model.gguf"))
        oo, oj, ob = (os.path.join(d, n) for n in ("out.json", "out.bin", "out.gguf"))

        rc, _ = cli(["quantize", mj, mb, mg])
        assert rc == 0, "quantize failed"
        rc, _ = cli(["dequantize", mg, oj, ob])
        assert rc == 0, "dequantize failed"

        got = read_bin_floats(ob)
        assert len(got) == len(original), "element count mismatch"

        err = max_err(original, got)
        # Q8_0 scale is f16 + 7-bit mantissa: a generous bound of ~0.05 holds for
        # unit-variance gaussian data while still catching real corruption.
        assert err < 0.05, "Q8_0 round-trip error %.6f exceeds bound" % err
        print("roundtrip: %d elements, max abs err = %.6f  [ok]" % (len(got), err))
        return True
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
