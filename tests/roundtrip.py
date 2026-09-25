import itertools
import os
import sys
import struct
import tempfile
import random
import json
import math
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import run as cli, run_process, write_bin, read_bin_floats, max_err

# Regression gate for quant/ + format/: build a random F32 model, quantize it to Q8_0 (and Q4_0) via the CLI, dequantize it back, and check the max error is within each type's quantization bound.
# The Q8_0, Q4_0, Q4_1 and Q4_K decoders must also match a decode written from the format description, bit for bit.

rng = random.Random(42)
UNICODE_NAME = "layer.\u00e9.\u4e2d.\U0001f600.\"\\\n.weight"


def tensor(rows, cols):
    return [rng.gauss(0.0, 1.0) for _ in range(rows * cols)]


def make_fixtures(d):
    t1 = tensor(512, 256)   # 131072 = 32*4096
    t2 = [rng.gauss(0, 1) for _ in range(256)]
    t3 = tensor(64, 32)
    # Tiny magnitudes on purpose: the per-block scale is amax/127 (Q8_0) or amax/7 (Q4_0), so values around 1e-4 put the scale in the f16 SUBNORMAL band, below the normal minimum of ~6.1e-5 but above the smallest subnormal of ~5.96e-8.
    # A bug in that decode path made every subnormal 16x too large and went unnoticed for the life of the project, because ordinary weights never produce one.
    # Do not shrink this further: below about 1e-5 the scale itself underflows to zero, which is a real f16 limit rather than a defect.
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
            result = run_process(["quantize", mj, mb, mg, qtype], text=True, timeout=30)
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


# Values per block and bytes per block.
BLOCKS = {"q8_0": (32, 34), "q4_0": (32, 18), "q4_1": (32, 20), "q4_k": (256, 144)}


# Decoding straight from the format description, so the check compares llmx against the spec rather than against itself.
# Every layout starts with an f16 scale d.
# Q8_0 then stores 32 signed bytes decoding as d*q.
# Q4_0 packs 32 values in 16 bytes, byte j's low nibble value j and high nibble value j+16, each decoding as d*(nibble-8).
# Q4_1 puts an f16 min m after d and packs its 16 bytes the same way, each nibble decoding as d*nibble + m.
# Q4_K is a 256-value super-block: an f16 dmin after d, 12 bytes of eight 6-bit scales and eight 6-bit mins, then 128 bytes of nibbles.
# Sub-block j < 4 takes scale byte[j] & 63 and min byte[j+4] & 63; j >= 4 takes scale (byte[j+4] & 15) | (byte[j-4] >> 6) << 4 and min (byte[j+4] >> 4) | (byte[j] >> 6) << 4.
# The nibble bytes run in four groups of 32: group g's low nibbles are sub-block 2g and its high nibbles sub-block 2g+1, each decoding as d*scale*nibble - dmin*min.
# The products are exact in f32 and the sums exact in a double, so packing the double as f32 rounds once, just where the format's f32 arithmetic does.
def decode_blocks(payload, qtype, count):
    block, typesize = BLOCKS[qtype]
    out = []
    for b in range(count // block):
        blk = payload[b * typesize:(b + 1) * typesize]
        d = struct.unpack("<e", blk[0:2])[0]
        if qtype == "q8_0":
            out.extend(d * v for v in struct.unpack("<32b", blk[2:34]))
        elif qtype == "q4_0":
            lo = [d * ((blk[2 + j] & 0x0F) - 8) for j in range(16)]
            hi = [d * ((blk[2 + j] >> 4) - 8) for j in range(16)]
            out.extend(lo + hi)
        elif qtype == "q4_1":
            m = struct.unpack("<e", blk[2:4])[0]
            out.extend([d * (q & 0x0F) + m for q in blk[4:20]] + [d * (q >> 4) + m for q in blk[4:20]])
        else:
            dmin = struct.unpack("<e", blk[2:4])[0]
            sm, qs = blk[4:16], blk[16:144]
            for j in range(8):
                if j < 4:
                    scale, low = sm[j] & 63, sm[j + 4] & 63
                else:
                    scale = (sm[j + 4] & 15) | (sm[j - 4] >> 6) << 4
                    low = (sm[j + 4] >> 4) | (sm[j] >> 6) << 4
                shift = 4 * (j % 2)
                group = qs[32 * (j // 2):32 * (j // 2) + 32]
                out.extend(d * scale * ((q >> shift) & 0x0F) - dmin * low for q in group)
    return out


def assert_decode_matches(qtype, reference, got):
    assert len(got) == len(reference), "independent decode: %s element count" % qtype
    differing = [i for i, (x, y) in enumerate(zip(reference, got))
                 if struct.pack("<f", x) != struct.pack("<f", y)]
    assert not differing, (
        "%s decode differs from the format description at %d of %d values, "
        "first at index %d: spec %r, llmx %r"
        % (qtype, len(differing), len(reference), differing[0],
           reference[differing[0]], got[differing[0]]))


# A single-tensor model puts the payload at the end of the file, so this needs no GGUF parser of its own to find it.
# The suite's own parser is not used on purpose: a check that shares a reader with the thing it checks is not independent of it.
def check_independent_decode(d):
    count = 4096
    values = [((i * 37 % 199) - 99) / 23.0 for i in range(count)]
    mj = os.path.join(d, "one.json")
    mb = os.path.join(d, "one.bin")
    with open(mb, "wb") as f:
        f.write(struct.pack("<%df" % count, *values))
    with open(mj, "w", encoding="utf-8") as f:
        json.dump({"name": "one", "tensors": [{"name": "w", "shape": [count]}]}, f)

    for qtype in ("q8_0", "q4_0"):
        block, typesize = BLOCKS[qtype]
        mg = os.path.join(d, "one_%s.gguf" % qtype)
        oj = os.path.join(d, "one_%s.json" % qtype)
        ob = os.path.join(d, "one_%s.bin" % qtype)
        rc, _ = cli(["quantize", mj, mb, mg, qtype])
        assert rc == 0, "independent decode: quantize (%s) failed" % qtype
        rc, _ = cli(["dequantize", mg, oj, ob])
        assert rc == 0, "independent decode: dequantize (%s) failed" % qtype

        with open(mg, "rb") as f:
            raw = f.read()
        payload = raw[len(raw) - count // block * typesize:]
        got = read_bin_floats(ob)
        assert_decode_matches(qtype, decode_blocks(payload, qtype, count), got)
        loss = max(abs(x - y) for x, y in zip(values, got))
        print("roundtrip: %s decode matches the format description on %d values, "
              "format loss %.6f  [ok]" % (qtype, count, loss))


# f16 bit patterns for the scales and mins: each of the 16 bits is set in one and clear in another, with no Inf or NaN.
# 1, the smallest and largest subnormals, the smallest normal, the largest finite, a negative normal, a negative subnormal and 13.33.
HALVES = (0x3C00, 0x0001, 0x03FF, 0x0400, 0x7BFF, 0xB555, 0x8201, 0x4AAA)


# Nibble byte i of block k, with a = i // 16 and c alternating between 0 and 1 every 16 blocks, has low nibble i + k + c*a and high nibble 5i + 3 + k + (c+2)*a, mod 16.
# Each block's low nibbles and its high nibbles take all 16 values, the two nibbles of a byte differ, and each run of 16 blocks sharing c gives every position every value.
# The two values of c tell every pair of positions apart, so a byte read from the wrong one of the 128 changes values.
def nibble_bytes(k, n):
    c = k // 16 % 2
    return bytes(((i + k + c * (i // 16)) & 15) | ((5 * i + 3 + k + (c + 2) * (i // 16)) & 15) << 4
                 for i in range(n))


# Q4_1 blocks pairing every scale with every min.
def q4_1_blocks():
    return b"".join(struct.pack("<2H", d, m) + nibble_bytes(k, 16)
                    for k, (d, m) in enumerate(itertools.product(HALVES, HALVES)))


# Q4_K super-blocks whose 12 scale and min bytes are all ones but for one cleared bit, one per bit, then all ones under every pairing of d and dmin.
# Every 6-bit field reads 63 except the one the cleared bit belongs to, so a bit read from the wrong place or into the wrong sub-block changes values.
# The cleared-bit blocks take d 1 and dmin 13.33, where no scale or min bit is lost to rounding against the other product.
def q4_k_blocks():
    heads = []
    for bit in range(96):
        sm = bytearray(b"\xff" * 12)
        sm[bit // 8] &= ~(1 << (bit % 8))
        heads.append(struct.pack("<2H", 0x3C00, 0x4AAA) + sm)
    heads += [struct.pack("<2H", d, dmin) + b"\xff" * 12 for d, dmin in itertools.product(HALVES, HALVES)]
    return b"".join(head + nibble_bytes(k, 128) for k, head in enumerate(heads))


# The formats `quantize` does not write get their blocks from the test, written into a one-tensor GGUF with no metadata.
# GGUF v3: magic, version, tensor count, metadata count, then the tensor's name, rank, dimensions, type (3 is Q4_1, 12 is Q4_K) and offset, and the data at the next 32-byte boundary.
def check_raw_decode(d):
    for qtype, ggml_type, payload in (("q4_1", 3, q4_1_blocks()), ("q4_k", 12, q4_k_blocks())):
        block, typesize = BLOCKS[qtype]
        count = len(payload) // typesize * block
        mg, oj, ob = (os.path.join(d, "raw_%s.%s" % (qtype, ext)) for ext in ("gguf", "json", "bin"))
        header = struct.pack("<IIQQQ", 0x46554747, 3, 1, 0, 1) + b"w" + struct.pack("<IQIQ", 1, count, ggml_type, 0)
        with open(mg, "wb") as f:
            f.write(header + b"\0" * (-len(header) % 32) + payload)
        rc, out = cli(["dequantize", mg, oj, ob])
        assert rc == 0, "raw decode: dequantize (%s) failed: %s" % (qtype, out)
        assert_decode_matches(qtype, decode_blocks(payload, qtype, count), read_bin_floats(ob))
        print("roundtrip: %s decode matches the format description on %d values "
              "from %d raw blocks  [ok]" % (qtype, count, len(payload) // typesize))


# Paths reach the converter as UTF-8, and Windows reads a narrow path in the system code page, so a directory named outside it is where a narrow open fails.
def check_non_ascii_directory(d):
    sub = os.path.join(d, "é中\U0001f600")
    os.mkdir(sub)
    for name in ("model.json", "model.bin"):
        shutil.copyfile(os.path.join(d, name), os.path.join(sub, name))
    for qtype in ("q8_0", "q4_0"):
        written = []
        for where in (d, sub):
            mj, mb = os.path.join(where, "model.json"), os.path.join(where, "model.bin")
            mg, oj, ob = (os.path.join(where, "dir_%s.%s" % (qtype, ext)) for ext in ("gguf", "json", "bin"))
            rc, out = cli(["quantize", mj, mb, mg, qtype])
            assert rc == 0, "quantize (%s) under %a failed: %a" % (qtype, where, out)
            rc, out = cli(["dequantize", mg, oj, ob])
            assert rc == 0, "dequantize (%s) under %a failed: %a" % (qtype, where, out)
            with open(oj, encoding="utf-8") as f:
                assert json.load(f)["name"] == mg, "model path changed under %a" % where
            with open(mg, "rb") as f, open(ob, "rb") as g:
                written.append((f.read(), g.read()))
        assert written[0] == written[1], "%s files differ under a non-ASCII directory" % qtype
        print("roundtrip: %s under a non-ASCII directory writes the same files  [ok]" % qtype)


# quantize takes only the names of the types it writes: Q4_1 has a quantizer but no GGUF file type here.
def check_type_names(d):
    mj, mb, mg = (os.path.join(d, n) for n in ("model.json", "model.bin", "names.gguf"))
    for name in ("q4_1", "Q8_0"):
        result = run_process(["quantize", mj, mb, mg, name], text=True, timeout=30)
        assert result.returncode == 2 and "unknown quant type" in result.stderr, (
            "quantize did not refuse type %a with status 2: %d %a" % (name, result.returncode, result.stderr))
        assert not os.path.exists(mg), "refused type %a created an output" % name
    print("roundtrip: type names quantize does not write are refused with status 2  [ok]")


def run():
    d = tempfile.mkdtemp(prefix="llmx_rt_")
    try:
        original = make_fixtures(d)
        mj, mb, mg = (os.path.join(d, n) for n in ("model.json", "model.bin", "model.gguf"))
        oj, ob = (os.path.join(d, n) for n in ("out.json", "out.bin"))

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
            # The tiny tensor needs a RELATIVE check: its absolute error is ~1e-7 no matter how badly the scale decodes, so an absolute bound would pass a 16x subnormal error silently.
            tiny_o, tiny_g = original[-256:], got[-256:]
            scale = max(abs(x) for x in tiny_o)
            tiny_err = max_err(tiny_o, tiny_g) / scale
            assert tiny_err < bound, (
                "%s subnormal-scale round-trip relative error %.6f exceeds %.3f"
                % (qtype, tiny_err, bound))
            # Q8_0: f16 + 7-bit mantissa scale -> ~0.05 for unit-variance gaussian.
            # Q4_0: 4-bit signed range scaled by amax/7 -> ~0.3 typical; 1.0 is a generous bound that still catches real corruption.
            assert err < bound, "%s round-trip error %.6f exceeds bound %.3f" % (
                qtype, err, bound)
            print("roundtrip: %s, %d elements, max abs err = %.6f, "
                  "subnormal-scale rel err = %.6f  [ok]" % (
                      qtype, len(got), err, tiny_err))
        check_independent_decode(d)
        check_raw_decode(d)
        check_non_ascii_directory(d)
        check_type_names(d)
        check_tensor_extents(d)
        return True
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
