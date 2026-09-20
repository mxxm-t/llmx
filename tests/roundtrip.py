import os
import sys
import struct
import tempfile
import random
import json
import math
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import run as cli, exe_path, write_bin, read_bin_floats, max_err

# Regression gate for quant/ + format/: build a random F32 model, quantize it to
# Q8_0 (and Q4_0) via the CLI, dequantize it back, and check the max error is
# within each type's quantization bound.

rng = random.Random(42)
UNICODE_NAME = "layer.\u00e9.\u4e2d.\U0001f600.\"\\\n.weight"


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
            {"name": UNICODE_NAME,           "shape": [32, 64]},
            {"name": "tiny.weight",           "shape": [256]},
        ],
    }
    with open(os.path.join(d, "model.json"), "w", encoding="ascii") as f:
        json.dump(json_doc, f, indent=2, ensure_ascii=True)

    allf = list(t1) + list(t2) + list(t3) + list(t4)
    write_bin(os.path.join(d, "model.bin"), allf)
    return allf


def check_tensor_extents(d):
    mj, mb, mg, oj, ob = (os.path.join(d, "extent." + suffix)
                          for suffix in ("json", "bin", "gguf", "out.json", "out.bin"))

    def document(shapes):
        return '{"name":"extents","tensors":[' + ','.join(
            '{"name":"t%d","shape":%s}' % (i, shape)
            for i, shape in enumerate(shapes)) + ']}'

    def inputs(shapes, payload):
        with open(mj, "w", encoding="ascii") as f:
            f.write(document(shapes))
        with open(mb, "wb") as f:
            f.write(payload)

    invalid = []
    for value in ("-1", "-32", "-0", "0", "0.5", "32.5", "1e-3",
                  "9007199254740992", "9007199254740993", "1e100"):
        invalid.append(("dimension " + value, ["[32," + value + "]"], b"", "dimension"))
    for value in ('"32"', "true", "false", "null", "[]", "{}"):
        invalid.append(("nonnumeric " + value, ["[32," + value + "]"], b"", "not a number"))
    invalid.extend([
        ("rank zero", ["[]"], b"", "rank"),
        ("rank five", ["[32,1,1,1,1]"], b"", "rank"),
        ("nonarray shape", ["32"], b"", "shape"),
        ("dimension product overflow", ["[4503599627370496,4096]"], b"", "multiplication overflow"),
        ("F32 byte overflow", ["[4503599627370496,1024]"], b"", "multiplication overflow"),
        ("F32 sum overflow", ["[4503599627370496,512]"] * 2, b"", "addition overflow"),
        ("allocation or stream limit", ["[4503599627370496,512]"], b"", "storage"),
        ("maximum exact dimension", ["[32,9007199254740991]"], b"", "size does not match"),
        ("maximum aligned dimension", ["[9007199254740960]"], b"", "size does not match"),
        ("partial quantized row", ["[16,2]"], bytes(128), "row"),
        ("partial quantized rank one", ["[33]"], bytes(132), "row"),
        ("truncated F32", ["[32]"], bytes(127), "size does not match"),
        ("extra F32 byte", ["[32]"], bytes(129), "size does not match"),
        ("extra F32 element", ["[32]"], bytes(132), "size does not match"),
        ("truncated second tensor", ["[32]", "[32]"], bytes(128), "size does not match"),
        ("empty tensors with payload", [], bytes(4), "size does not match"),
    ])
    sentinel = b"existing output must survive validation\x00\xff"
    for qtype, bound in (("q8_0", 0.05), ("q4_0", 1.0)):
        for label, shapes, payload, diagnostic in invalid:
            inputs(shapes, payload)
            with open(mg, "wb") as f:
                f.write(sentinel)
            result = subprocess.run([exe_path(), "quantize", mj, mb, mg, qtype],
                                    capture_output=True, text=True, encoding="utf-8", timeout=30)
            assert result.returncode != 0, "%s accepted %s" % (qtype, label)
            assert diagnostic in result.stderr, "%s: %s: %s" % (qtype, label, result.stderr)
            with open(mg, "rb") as f:
                assert f.read() == sentinel, "%s changed output for %s" % (qtype, label)

        os.remove(mg)
        inputs(["[32]"], bytes(127))
        rc, _ = cli(["quantize", mj, mb, mg, qtype])
        assert rc != 0 and not os.path.exists(mg), "invalid input created an output"

        shapes = [[32], [32, 2], [32, 2, 3], [32, 2, 3, 2]]
        original = [((i % 29) - 14) / 8 for i in range(sum(math.prod(s) for s in shapes))]
        payload = struct.pack("<%df" % len(original), *original)
        inputs([json.dumps(s) for s in shapes], payload)
        rc, _ = cli(["quantize", mj, mb, mg, qtype])
        assert rc == 0, "%s valid ranks failed" % qtype
        with open(mg, "rb") as f:
            integer_output = f.read()
        rc, _ = cli(["dequantize", mg, oj, ob])
        assert rc == 0, "%s valid ranks dequantize failed" % qtype
        with open(oj, encoding="utf-8") as f:
            restored = json.load(f)
        assert restored["tensors"] == [
            {"name": "t%d" % i, "shape": shape} for i, shape in enumerate(shapes)]
        got = read_bin_floats(ob)
        assert len(got) == len(original) and all(math.isfinite(x) for x in got)
        assert max_err(original, got) < bound, "%s valid ranks changed values" % qtype

        inputs(["[32.0]", "[3.2e1,2e0]", "[32,2.0,3e0]", "[32,2,3,2.0]"], payload)
        rc, _ = cli(["quantize", mj, mb, mg, qtype])
        assert rc == 0, "%s integral decimal/exponent dimensions failed" % qtype
        with open(mg, "rb") as f:
            assert f.read() == integer_output, "number spelling changed conversion bytes"

        inputs([], b"")
        rc, _ = cli(["quantize", mj, mb, mg, qtype])
        assert rc == 0, "%s empty model rejected" % qtype
        rc, _ = cli(["dequantize", mg, oj, ob])
        assert rc == 0, "%s empty model dequantize failed" % qtype
        with open(oj, encoding="utf-8") as f:
            assert json.load(f)["tensors"] == []
        assert os.path.getsize(ob) == 0, "empty model produced tensor bytes"
        print("roundtrip: %s, %d invalid extents preserve output; ranks 1..4, "
              "integral spellings and empty model [ok]" % (qtype, len(invalid)))


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
            with open(oj, encoding="utf-8") as f:
                restored = json.load(f)
            assert restored["name"] == mg, "model path changed during %s round-trip" % qtype
            assert [entry["name"] for entry in restored["tensors"]] == [
                "tok_embeddings.weight", "norm.weight", UNICODE_NAME, "tiny.weight"
            ], "Unicode tensor names changed during %s round-trip" % qtype

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
        check_tensor_extents(d)
        return True
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
