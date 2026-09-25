from fractions import Fraction
import hashlib
import importlib.util
import itertools
import math
import os
import random
import struct
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roundtrip
import spec_decode as sd

# Raw-block round trips for the decoders of tests/spec_decode.py, which the file-exact HF references and the MXFP4 writer decode with.
# Blocks are built from chosen fields and must decode to the values those fields define, computed here with exact fractions and rounded once to f32; a few blocks decode to values written out by hand.
# A value that is zero must also carry the sign the format's own arithmetic gives it, -0 where a negative scale meets a zero code or a scale of +0 meets a negative one.
# The numpy form must give the pure form's bits, NaN as NaN, on every block here, on tests/roundtrip.py's raw blocks and on random blocks of every type, and the MXFP4 writer's blocks must decode to the values it intended.
# It runs no llmx binary; tests/roundtrip.py holds llmx's own decoders to these where llmx reads the type.

WRITER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "write_mxfp4.py")

# f16 bit patterns for scales and mins: 1, -1, +0, -0, the smallest and largest subnormals, the smallest normal, the largest finite, 13.33, a negative normal and a negative subnormal.
HALVES = (0x3C00, 0xBC00, 0x0000, 0x8000, 0x0001, 0x03FF, 0x0400, 0x7BFF, 0x4AAA, 0xB555, 0x8201)

# The IQ4 code values, written out again here as the expected values; the E2M1 values of MXFP4 come from their bit fields in e2m1().
IQ4 = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)


def half_float(bits):
    return struct.unpack("<e", struct.pack("<H", bits))[0]


def half(bits):
    return Fraction(half_float(bits))


def cell(value):
    """The expected decode of one value, from `value`, a function of a reader of f16 bits.
    It is evaluated exactly with fractions, and in Python floats in the format's order, which gives a zero the sign the format's arithmetic does."""
    return value(half), value(half_float)


def f32(value):
    """An exact value rounded once to the nearest f32, an infinity past the largest finite."""
    return sd.to_f32(float(value)) if isinstance(value, Fraction) else value


def assert_values(name, expected, got):
    """`got`, a decode, holds the f32 of each exact value of the cells in `expected`, and a zero where the cell's float evaluation has one of the same sign."""
    assert len(got) == len(expected), "%s: %d values decoded, %d expected" % (name, len(got), len(expected))
    for i, ((exact, ordered), have) in enumerate(zip(expected, got)):
        want = f32(exact)
        if want != have or want == 0 and math.copysign(1, ordered) != math.copysign(1, have):
            raise AssertionError("%s: value %d is %r, expected %r" % (name, i, have, want if want else ordered))


def assert_same_bits(name, a, b):
    """Two decodes as f32 bytes, equal bit for bit, except that a NaN matches any NaN.
    The pure form's F16 and BF16 NaNs are Python floats, which keep no signalling NaN and, for F16, no payload."""
    assert len(a) == len(b), "%s: %d bytes against %d" % (name, len(a), len(b))
    if a == b:
        return
    for i in range(0, len(a), 4):
        x, y = struct.unpack_from("<f", a, i)[0], struct.unpack_from("<f", b, i)[0]
        if a[i:i + 4] != b[i:i + 4] and not (math.isnan(x) and math.isnan(y)):
            raise AssertionError("%s: value %d is %s in the numpy form and %s in the pure one"
                                 % (name, i // 4, b[i:i + 4].hex(), a[i:i + 4].hex()))


def assert_numpy_matches(name, type_id, payload, count):
    """The numpy form decodes `payload` to the pure form's bits; skipped without numpy, which run() reports once."""
    if sd.np is None:
        return
    assert_same_bits(name, sd.pack_f32(sd.decode(type_id, payload, count)), sd.decode_numpy(type_id, payload, count).tobytes())


def code_runs(rng, n, bits):
    """Codes of `bits` bits for `n` values of each of a series of blocks, one list per block.
    Each run of 2^bits values in a block holds every code once, and each group of 2^bits consecutive blocks takes one shuffle of those runs, rotated by one code a block, so every position takes every code over a group."""
    size = 1 << bits
    for k in itertools.count():
        if k % size == 0:
            base = []
            for _ in range(n // size):
                run = list(range(size))
                rng.shuffle(run)
                base.extend(run)
        yield [(c + k) % size for c in base]


# F16 and BF16: every bit pattern against the value its sign, exponent and mantissa give.
def check_16bit():
    for type_id, mantissa, bias in ((sd.F16, 10, 15), (sd.BF16, 7, 127)):
        top = (1 << (15 - mantissa)) - 1
        payload = struct.pack("<65536H", *range(65536))
        got = sd.decode(type_id, payload, 65536)
        for bits in range(65536):
            sign, exponent, m = bits >> 15, (bits >> mantissa) & top, bits & ((1 << mantissa) - 1)
            if exponent == top:
                ok = math.isinf(got[bits]) and m == 0 or math.isnan(got[bits]) and m != 0
                assert ok and (math.isnan(got[bits]) or (got[bits] < 0) == bool(sign)), (type_id, bits, got[bits])
                continue
            value = Fraction(m if exponent == 0 else (1 << mantissa) + m) * Fraction(2) ** (max(exponent, 1) - bias - mantissa)
            assert got[bits] == (-value if sign else value) and math.copysign(1, got[bits]) == (-1 if sign else 1), (type_id, bits, got[bits])
        assert_numpy_matches(sd.type_name(type_id), type_id, payload, 65536)
    payload = struct.pack("<5I", 0x00000001, 0x80000000, 0x7F7FFFFF, 0x3F800000, 0xC2280000)
    assert sd.pack_f32(sd.decode(sd.F32, payload, 5)) == payload
    assert_numpy_matches("F32", sd.F32, payload, 5)
    print("raw-blocks: F16 and BF16 decode all 65536 bit patterns to their sign, exponent and mantissa, F32 as it is  [ok]")


# IQ4_NL and Q4_0 blocks share a layout: an f16 scale d, then 32 codes as nibbles in Q4_0's order.
def iq4_nl_block(d, cs):
    return struct.pack("<H", d) + bytes(cs[j] | cs[j + 16] << 4 for j in range(16))


def iq4_xs_block(d, scales, cs):
    high = sum((s >> 4) << 2 * b for b, s in enumerate(scales))
    low = bytes(scales[2 * i] & 15 | (scales[2 * i + 1] & 15) << 4 for i in range(4))
    qs = b"".join(bytes(cs[32 * b + j] | cs[32 * b + 16 + j] << 4 for j in range(16)) for b in range(8))
    return struct.pack("<HH", d, high) + low + qs


# IQ4_NL and IQ4_XS: every code in both nibbles, scales d positive, negative, zero, subnormal and the largest finite, and for IQ4_XS every one of the 48 scale bits cleared once, all scales 0 and all 63.
def check_iq4(rng):
    payload, expected, runs = b"", [], code_runs(rng, 32, 4)
    for d in HALVES * 16:
        cs = next(runs)
        payload += iq4_nl_block(d, cs)
        expected += [cell(lambda h: h(d) * IQ4[c]) for c in cs]
    assert_values("IQ4_NL", expected, sd.decode(sd.IQ4_NL, payload, len(expected)))
    assert_numpy_matches("IQ4_NL", sd.IQ4_NL, payload, len(expected))
    # d 0.5, codes 0 to 15 in the low nibbles and 15 to 0 in the high ones.
    hand = [-63.5, -52, -41.5, -32.5, -24.5, -17.5, -11, -5, 0.5, 6.5, 12.5, 19, 26.5, 34.5, 44.5, 56.5]
    assert sd.decode(sd.IQ4_NL, iq4_nl_block(0x3800, list(range(16)) + list(range(15, -1, -1))), 32) == hand + hand[::-1]
    nl = len(expected)

    patterns = [[63] * 8, [0] * 8, [0, 9, 18, 27, 36, 45, 54, 63]]
    patterns += [[63 & ~(1 << bit) if b == sub else 63 for b in range(8)] for sub in range(8) for bit in range(6)]
    payload, expected, runs = b"", [], code_runs(rng, 256, 4)
    for d, scales in [(d, patterns[0]) for d in HALVES] + [(0x3C00, s) for s in patterns] + [(0x0001, patterns[2]), (0xB555, patterns[2])]:
        cs = next(runs)
        payload += iq4_xs_block(d, scales, cs)
        expected += [cell(lambda h: h(d) * (scales[v // 32] - 32) * IQ4[c]) for v, c in enumerate(cs)]
    assert_values("IQ4_XS", expected, sd.decode(sd.IQ4_XS, payload, len(expected)))
    assert_numpy_matches("IQ4_XS", sd.IQ4_XS, payload, len(expected))
    # d 1 with sub-block b's scale 32 + b, so value 32b + j is b times the code of j, codes 0 to 15 then 15 to 0 in every sub-block.
    # Sub-block 0's scale is +0, so its values are zeros with the signs of their codes' values.
    cs = (list(range(16)) + list(range(15, -1, -1))) * 8
    got = sd.decode(sd.IQ4_XS, iq4_xs_block(0x3C00, [32 + b for b in range(8)], cs), 256)
    assert got[32:64] == [float(v) for v in IQ4 + IQ4[::-1]] and got[7 * 32 + 1] == -728 and not any(got[:32])
    assert [math.copysign(1, v) for v in got[:32]] == [-1] * 8 + [1] * 16 + [-1] * 8
    print("raw-blocks: IQ4_NL %d and IQ4_XS %d values over every code, scale bit and edge scale  [ok]" % (nl, len(expected)))


def e2m1(code):
    """The E2M1 element value of a 4-bit code under the OCP MX specification: a sign bit, two exponent bits with bias 1, one mantissa bit, subnormal at exponent 0."""
    sign, exponent, mantissa = code >> 3, code >> 1 & 3, code & 1
    value = Fraction(mantissa, 2) if exponent == 0 else Fraction(2) ** (exponent - 1) * (1 + Fraction(mantissa, 2))
    return -value if sign else value


def mxfp4_value(e, code):
    """The exact value an MXFP4 code decodes to under an E8M0 exponent e, X = 2^(e - 127) times its E2M1 value, with -0 read as +0 and e = 255 read by the same rule; None for an infinity, the value being 2^128 or more."""
    value = Fraction(2) ** (e - 127) * e2m1(code)
    return None if abs(value) >= Fraction(2) ** 128 else value


# MXFP4: every exponent from 0 to 255 in one block each, each block holding every code in both nibbles, and every position every code over each 16 blocks.
# Values of 2^128 or more decode as infinities of their sign, -0 decodes as +0, and exponents 0 and 1 give subnormal scales.
def check_mxfp4(rng):
    payload, expected, cells, runs = b"", [], [], code_runs(rng, 32, 4)
    for e in range(256):
        cs = next(runs)
        payload += bytes([e]) + bytes(cs[j] | cs[j + 16] << 4 for j in range(16))
        cells += [(e, c) for c in cs]
        expected += [mxfp4_value(e, c) for c in cs]
    got = sd.decode(sd.MXFP4, payload, len(expected))
    overflow = 0
    for i, ((e, c), want) in enumerate(zip(cells, expected)):
        if want is None:
            overflow += 1
            assert got[i] == (-math.inf if c & 8 else math.inf), (e, c, got[i])
        else:
            assert got[i] == f32(want), (e, c, got[i], f32(want))
        if c in (0, 8):
            assert struct.pack("<f", got[i]) == b"\0\0\0\0", ("zero code not +0", e, c)
    # 2^128 is reached by E2M1 magnitudes 4 and 6 at e = 253, 2 and up at 254 and 1 and up at 255, each of either sign and twice in a block.
    assert overflow == 2 * 2 * (2 + 4 + 6), overflow
    assert_numpy_matches("MXFP4", sd.MXFP4, payload, len(expected))

    def one(e, code):
        return struct.pack("<f", sd.decode(sd.MXFP4, bytes([e, code]) + bytes(15), 32)[0]).hex()
    hand = {(127, 1): "0000003f", (127, 7): "0000c040", (127, 15): "0000c0c0", (0, 1): "00002000", (1, 1): "00004000",
            (0, 7): "00004001", (254, 1): "0000807e", (253, 5): "0000407f", (253, 6): "0000807f", (255, 1): "0000007f",
            (255, 2): "0000807f", (255, 10): "000080ff", (255, 8): "00000000", (0, 8): "00000000"}
    for (e, code), bits in hand.items():
        assert one(e, code) == bits, (e, code, one(e, code), bits)
    print("raw-blocks: MXFP4 exponents 0 to 255 against every code in every position, %d values, %d infinities, -0 as +0  [ok]"
          % (len(expected), overflow))


def q3_k_block(d, scales, q):
    """A Q3_K block of 6-bit stored scales and 256 codes from -4 to 3."""
    hmask, qs = bytearray(32), bytearray(64)
    for i, v in enumerate(q):
        h, j, l = i // 128, i % 128 // 32, i % 32
        if v >= 0:
            hmask[l] |= 1 << (4 * h + j)
        qs[32 * h + l] |= (v & 3) << 2 * j
    sc = bytearray(12)
    for s, value in enumerate(scales):
        sc[s % 8] |= (value & 15) << 4 * (s // 8)
        sc[8 + s % 4] |= (value >> 4) << 2 * (s // 4)
    return bytes(hmask) + bytes(qs) + bytes(sc) + struct.pack("<H", d)


# Q3_K: every one of the 96 scale bits cleared once, all scales 0 and all 63, and high bits keyed to each value's address, so a decoder that subtracts 4 on the wrong state of the bit, or reads it from another value, changes values.
def check_q3_k(rng):
    patterns = [[63] * 16, [0] * 16, [4 * s for s in range(16)]]
    patterns += [[63 & ~(1 << bit) if s == sub else 63 for s in range(16)] for sub in range(16) for bit in range(6)]
    blocks = [(d, patterns[2]) for d in HALVES] + [(0x3C00, s) for s in patterns] + [(0x0201, patterns[0]), (0xB555, patterns[2])]
    payload, expected, runs = b"", [], code_runs(rng, 256, 2)
    for k, (d, scales) in enumerate(blocks):
        low = next(runs)
        # Block k sets the high bit of value i where bit k % 10 of i is set for k % 10 below 8, every high bit for k % 10 = 8 and none for 9.
        high = [1 if k % 10 == 8 else 0 if k % 10 == 9 else i >> (k % 10) & 1 for i in range(256)]
        q = [c if h else c - 4 for c, h in zip(low, high)]
        payload += q3_k_block(d, scales, q)
        expected += [cell(lambda h: h(d) * (scales[i // 16] - 32) * v) for i, v in enumerate(q)]
    assert_values("Q3_K", expected, sd.decode(sd.Q3_K, payload, len(expected)))
    assert_numpy_matches("Q3_K", sd.Q3_K, payload, len(expected))
    # d 1 and every scale 33, so values are the codes, 2-bit codes 0 to 3 in bit pairs 0 to 3 of every code byte: a set high bit keeps the code and a clear one subtracts 4.
    for fill, want in ((b"\xff", [0, 1, 2, 3]), (b"\x00", [-4, -3, -2, -1])):
        got = sd.decode(sd.Q3_K, fill * 32 + bytes([0b11100100] * 64) + bytes([0x11] * 8 + [0xAA] * 4) + b"\x00\x3c", 256)
        assert got[0:256:32] == want * 2, got[0:256:32]
    print("raw-blocks: Q3_K %d values over every scale bit, address-keyed high bits and edge scales  [ok]" % len(expected))


def q2_k_block(d, dmin, scale_bytes, q):
    qs = bytearray(64)
    for i, v in enumerate(q):
        qs[32 * (i // 128) + i % 32] |= v << 2 * (i % 128 // 32)
    return bytes(scale_bytes) + bytes(qs) + struct.pack("<HH", d, dmin)


# Q2_K: every scale and min byte, and d and dmin paired over every edge value, subnormals and negatives included.
def check_q2_k(rng):
    blocks = [(0x3C00, 0x4AAA, range(16 * k, 16 * k + 16)) for k in range(16)]
    blocks += [(d, dmin, [0xFF, 0x0F, 0xF0, 0x00] * 4) for d in HALVES for dmin in HALVES]
    payload, expected, runs = b"", [], code_runs(rng, 256, 2)
    for d, dmin, scale_bytes in blocks:
        q = next(runs)
        payload += q2_k_block(d, dmin, scale_bytes, q)
        expected += [cell(lambda h: h(d) * (scale_bytes[i // 16] & 15) * v - h(dmin) * (scale_bytes[i // 16] >> 4)) for i, v in enumerate(q)]
    assert_values("Q2_K", expected, sd.decode(sd.Q2_K, payload, len(expected)))
    assert_numpy_matches("Q2_K", sd.Q2_K, payload, len(expected))
    # d 1 and dmin 0.5 with scale 1 and min 2 (byte 0x21): each value is its code less 1.
    got = sd.decode(sd.Q2_K, q2_k_block(0x3C00, 0x3800, [0x21] * 16, [0, 1, 2, 3] * 64), 256)
    assert got[:4] == [-1, 0, 1, 2], got[:4]
    print("raw-blocks: Q2_K %d values over every scale and min byte and every pairing of d and dmin  [ok]" % len(expected))


def q8_0_block(d, q):
    return struct.pack("<H", d) + struct.pack("<32b", *q)


# Q4_0 and Q8_0: every code under every scale of HALVES, so a zero code under a negative scale, and a negative code under +0, decode as -0.
def check_q4_0_q8_0(rng):
    payload, expected, runs = b"", [], code_runs(rng, 32, 4)
    for d in HALVES * 16:
        cs = next(runs)
        payload += iq4_nl_block(d, cs)
        expected += [cell(lambda h: h(d) * (c - 8)) for c in cs]
    assert_values("Q4_0", expected, sd.decode(sd.Q4_0, payload, len(expected)))
    assert_numpy_matches("Q4_0", sd.Q4_0, payload, len(expected))
    q4 = len(expected)
    # Eight blocks for each scale, holding every signed byte between them.
    payload, expected = b"", []
    for d in HALVES:
        signed = list(range(-128, 128))
        rng.shuffle(signed)
        for b in range(8):
            q = signed[32 * b:32 * b + 32]
            payload += q8_0_block(d, q)
            expected += [cell(lambda h: h(d) * v) for v in q]
    assert_values("Q8_0", expected, sd.decode(sd.Q8_0, payload, len(expected)))
    assert_numpy_matches("Q8_0", sd.Q8_0, payload, len(expected))
    # d -1 with nibble 8 or byte 0: d * (nibble - 8) and d * byte are -0.
    assert struct.pack("<f", sd.decode(sd.Q4_0, iq4_nl_block(0xBC00, [8] * 32), 32)[0]) == b"\0\0\0\x80"
    assert struct.pack("<f", sd.decode(sd.Q8_0, q8_0_block(0xBC00, [0] * 32), 32)[0]) == b"\0\0\0\x80"
    print("raw-blocks: Q4_0 %d and Q8_0 %d values over every code and scale, -0 where a negative scale meets a zero code  [ok]"
          % (q4, len(expected)))


# The numpy form against the pure one on the raw blocks tests/roundtrip.py holds llmx's Q4_1, Q4_K, Q5_K and Q6_K decoders to, which reach every scale, min, high bit and nibble.
def check_roundtrip_blocks():
    for type_id, payload in ((sd.Q4_1, roundtrip.q4_1_blocks()), (sd.Q4_K, roundtrip.q4_k_blocks()),
                             (sd.Q5_K, roundtrip.q5_k_blocks()), (sd.Q6_K, roundtrip.q6_k_blocks())):
        _, values, size = sd.TYPES[type_id]
        assert_numpy_matches(sd.type_name(type_id), type_id, payload, len(payload) // size * values)
    print("raw-blocks: the numpy form gives the pure form's bits on the round trip's Q4_1, Q4_K, Q5_K and Q6_K raw blocks  [ok]")


def random_blocks(rng, type_id, count):
    """`count` random blocks of `type_id`, with every f16 scale and min finite."""
    _, _, size = sd.TYPES[type_id]
    halves = {sd.Q8_0: (0,), sd.Q4_0: (0,), sd.Q4_1: (0, 2), sd.IQ4_NL: (0,), sd.IQ4_XS: (0,), sd.Q4_K: (0, 2),
              sd.Q5_K: (0, 2), sd.Q6_K: (208,), sd.Q2_K: (80, 82), sd.Q3_K: (108,)}.get(type_id, ())
    out = bytearray()
    for _ in range(count):
        block = bytearray(rng.getrandbits(8) for _ in range(size))
        for at in halves:
            block[at + 1] &= 0xFB if block[at + 1] & 0x7C == 0x7C else 0xFF
        out += block
    return bytes(out)


# The numpy form against the pure one on random blocks of every type, the types llmx reads today included.
def check_random(rng):
    for type_id in sorted(sd.TYPES):
        count = 64 * sd.TYPES[type_id][1] if sd.TYPES[type_id][1] > 1 else 4096
        assert_numpy_matches("random " + sd.type_name(type_id), type_id, random_blocks(rng, type_id, count // sd.TYPES[type_id][1]), count)
    print("raw-blocks: the numpy form gives the pure form's bits on random blocks of all %d types  [ok]" % len(sd.TYPES))


def load_writer():
    spec = importlib.util.spec_from_file_location("llmx_write_mxfp4", WRITER)
    writer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(writer)
    return writer


# The MXFP4 writer's rule: ties, saturation, a zero block, a clamped exponent and signs, each block decoding to the values the writer intended.
def check_mxfp4_writer(rng):
    writer = load_writer()
    grid = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
    # Scale 1 (a block maximum in [4, 8)): each E2M1 midpoint rounds to the neighbour whose mantissa bit is 0, and 7 saturates at 6.
    ties = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 7]
    want = [0, 1, 1, 2, 2, 4, 4, 6]
    tie_blocks = [[signed * t for t in ties] + [signed * g for g in grid] + [0.3, 0.7, 1.3, 1.7, 2.7, 3.3, 4.9, 5.1] + [4] * 8
                  for signed in (1, -1)]
    for signed, block in zip((1, -1), tie_blocks):
        e, cs, intended = writer.encode_block(block)
        assert e == 127 and intended[:8] == [signed * w if w else 0.0 for w in want], (e, intended[:8])
        assert intended[8:16] == [signed * g if g else 0.0 for g in grid] and intended[16:24] == [0.5, 0.5, 1.5, 1.5, 3, 3, 4, 6]
        assert all(not c & 8 for c, v in zip(cs, intended) if v == 0), "a zero written as -0"
    # The tie blocks lead the samples, so the numpy form meets every midpoint of either sign as the pure form does.
    samples = tie_blocks + [[0.0] * 32, [1e-40] + [0.0] * 31, [-3.0e-39, 2.9e-39] + [1e-45] * 30, [65504.0] * 32, [3.4e38, -1.0] + [0.0] * 30]
    samples += [[rng.gauss(0, 10 ** rng.randint(-8, 8)) for _ in range(32)] for _ in range(200)]
    samples += [[struct.unpack("<f", struct.pack("<f", rng.uniform(-1, 1)))[0] * 2.0 ** rng.randint(-40, 40) for _ in range(32)] for _ in range(100)]
    payload, intended_all = b"", []
    for block in samples:
        block = [struct.unpack("<f", struct.pack("<f", v))[0] for v in block]
        e, cs, intended = writer.encode_block(block)
        payload += bytes([e]) + bytes(cs[j] | cs[j + 16] << 4 for j in range(16))
        intended_all += intended
    assert writer.encode_block([0.0] * 32)[0] == 0, "a zero block takes exponent 0"
    assert writer.encode_block([1e-40] + [0.0] * 31)[0] == 0, "the exponent is clamped at -127"
    got = sd.decode(sd.MXFP4, payload, len(intended_all))
    assert sd.pack_f32(got) == sd.pack_f32(intended_all), "the writer's blocks decode to other values than it intended"
    if sd.np is not None:
        values = sd.np.array([[struct.unpack("<f", struct.pack("<f", v))[0] for v in b] for b in samples], dtype=sd.np.float32)
        data, fast = writer.encode_numpy(values)
        assert data.tobytes() == payload and fast.tobytes() == sd.pack_f32(intended_all), "the writer's numpy form differs from its pure form"
        check_writer_file(writer)
    print("raw-blocks: the MXFP4 writer rounds ties to even, saturates at 6, writes no -0, and %d blocks decode to its values  [ok]"
          % len(samples))


# The writer on a tiny BF16 model: its matrices and tied embedding written as MXFP4, its norms copied, the source's labels of how it was made dropped, and every tensor read back as intended.
def check_writer_file(writer):
    rng = random.Random(7)
    bf16 = lambda n: struct.pack("<%dH" % n, *(struct.unpack("<I", struct.pack("<f", rng.gauss(0, 0.05)))[0] >> 16 for _ in range(n)))
    metadata = {"general.architecture": (8, "qwen3"), "general.file_type": (4, 32), "general.quantized_by": (8, "someone"),
                "qwen3.block_count": (4, 1), "tokenizer.ggml.tokens": (9, (8, ["a", "b"]))}
    norm = struct.pack("<64f", *[1.0 + i / 64 for i in range(64)])
    tensors = [("token_embd.weight", [64, 3], sd.BF16, bf16(192)), ("blk.0.attn_norm.weight", [64], sd.F32, norm),
               ("blk.0.ffn_down.weight", [96, 64], sd.BF16, bf16(96 * 64))]
    with tempfile.TemporaryDirectory(prefix="llmx_mxfp4_") as directory:
        source, output = os.path.join(directory, "source.gguf"), os.path.join(directory, "out.gguf")
        sd.write_gguf(source, metadata, tensors)
        writer.convert(source, output, log=lambda *_: None)
        written = sd.GGUF(output)
        assert list(written.metadata) == ["general.architecture", "qwen3.block_count", "tokenizer.ggml.tokens"]
        assert [(t.name, t.shape, t.type) for t in written.tensors] == [
            ("token_embd.weight", [64, 3], sd.MXFP4), ("blk.0.attn_norm.weight", [64], sd.F32), ("blk.0.ffn_down.weight", [96, 64], sd.MXFP4)]
        assert written.raw(written.tensors[1]) == norm
        source_values = sd.GGUF(source).decode(sd.GGUF(source).tensors[2])
        _, intended = writer.encode_numpy(source_values.reshape(-1, 32))
        assert written.decode(written.tensors[2]).tobytes() == intended.tobytes()
        with open(output, "rb") as f:
            first = hashlib.sha256(f.read()).hexdigest()
        writer.convert(source, output, log=lambda *_: None)
        with open(output, "rb") as f:
            assert hashlib.sha256(f.read()).hexdigest() == first, "the writer is not deterministic"


def run(require=False):
    """The checks above; without numpy the numpy form's are skipped, or with `require` fail."""
    rng = random.Random(20260925)
    check_16bit()
    check_iq4(rng)
    check_mxfp4(rng)
    check_q3_k(rng)
    check_q2_k(rng)
    check_q4_0_q8_0(rng)
    if sd.np is None:
        assert not require, "raw-blocks: numpy is not installed, and the numpy form's checks need it"
        print("raw-blocks: SKIP - the numpy form's checks and the writer's file need numpy")
    else:
        check_roundtrip_blocks()
        check_random(rng)
    check_mxfp4_writer(rng)
    return True


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
