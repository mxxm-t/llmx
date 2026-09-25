"""Decoders for the GGUF tensor types, written from the format descriptions, and a reader and writer for GGUF files.

Each type decodes in two forms: a pure Python form, the readable reference, and a numpy form for whole files, which tests/raw_blocks.py holds to the pure one bit for bit, NaN as NaN.
The pure form's F16 and BF16 NaNs are Python floats, which keep no signalling NaN and, for F16, no payload, so a comparison of bits against the pure form takes a NaN as a NaN; the numpy form keeps every NaN's bits.
The layouts are those of the GGUF type definitions; MXFP4 follows the OCP Microscaling Formats (MX) v1.0 specification.
Every product these layouts form is exact in f32, and a scale times a code minus a min rounds once, so decoding exactly in double and rounding to f32 at the end gives the format's bits.
Neither form flushes subnormals: f16 subnormal scales widen to normal f32 values, and MXFP4's exponents 0 and 1 give subnormal f32 scales.
The pure form needs only the standard library; numpy is imported when present and needed only by the numpy form.
"""

import math
import struct

try:
    import numpy as np
except ImportError:
    np = None

F32, F16, Q4_0, Q4_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS, BF16, MXFP4 = 0, 1, 2, 3, 8, 10, 11, 12, 13, 14, 20, 23, 30, 39

# GGUF type id: name, values per block, bytes per block.
TYPES = {
    F32: ("F32", 1, 4),
    F16: ("F16", 1, 2),
    Q4_0: ("Q4_0", 32, 18),
    Q4_1: ("Q4_1", 32, 20),
    Q8_0: ("Q8_0", 32, 34),
    Q2_K: ("Q2_K", 256, 84),
    Q3_K: ("Q3_K", 256, 110),
    Q4_K: ("Q4_K", 256, 144),
    Q5_K: ("Q5_K", 256, 176),
    Q6_K: ("Q6_K", 256, 210),
    IQ4_NL: ("IQ4_NL", 32, 18),
    IQ4_XS: ("IQ4_XS", 256, 136),
    BF16: ("BF16", 1, 2),
    MXFP4: ("MXFP4", 32, 17),
}

# The 16 values a 4-bit IQ4_NL or IQ4_XS code selects.
IQ4_VALUES = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)

# The E2M1 values doubled, so that a 4-bit MXFP4 code c selects MXFP4_VALUES[c] * 2^(e - 128) for the block's E8M0 byte e.
# Code 8 is -0 in E2M1; it is +0 here, the value every integer path computes.
MXFP4_VALUES = (0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)


def type_name(type_id):
    return TYPES[type_id][0] if type_id in TYPES else "type %d" % type_id


def payload_bytes(type_id, count):
    """The bytes `count` values of `type_id` take; a partial block or an unknown type raises ValueError."""
    if type_id not in TYPES:
        raise ValueError("no decoder for GGUF type %d" % type_id)
    _, values, size = TYPES[type_id]
    if count % values:
        raise ValueError("%d values are not whole %s blocks of %d" % (count, TYPES[type_id][0], values))
    return count // values * size


# The pure form.

def to_f32(x):
    """`x` rounded to the nearest f32, as a Python float; a value past the largest finite f32 becomes an infinity, which struct.pack refuses to make."""
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


def f16(block, at):
    return struct.unpack_from("<e", block, at)[0]


# A BF16 value is the upper half of the f32 with the same bits.
def bf16(block, at):
    return struct.unpack("<f", b"\0\0" + bytes(block[at:at + 2]))[0]


# Q8_0: an f16 scale d, then 32 signed bytes, value i being d * byte i.
def q8_0(block):
    d = f16(block, 0)
    return [d * q for q in struct.unpack_from("<32b", block, 2)]


# Q4_0: an f16 scale d, then 16 bytes; byte j's low nibble is value j and its high nibble value j + 16, each d * (nibble - 8).
def q4_0(block):
    d = f16(block, 0)
    qs = block[2:18]
    return [d * ((q & 15) - 8) for q in qs] + [d * ((q >> 4) - 8) for q in qs]


# Q4_1: an f16 scale d and an f16 min m, then nibbles in Q4_0's order, each d * nibble + m.
def q4_1(block):
    d, m = f16(block, 0), f16(block, 2)
    qs = block[4:20]
    return [d * (q & 15) + m for q in qs] + [d * (q >> 4) + m for q in qs]


# IQ4_NL: an f16 scale d, then nibbles in Q4_0's order, each d * IQ4_VALUES[nibble].
def iq4_nl(block):
    d = f16(block, 0)
    qs = block[2:18]
    return [d * IQ4_VALUES[q & 15] for q in qs] + [d * IQ4_VALUES[q >> 4] for q in qs]


# MXFP4: an E8M0 byte e, then nibbles in Q4_0's order, each MXFP4_VALUES[nibble] * 2^(e - 128).
# Every e decodes the same way: 0 and 1 give subnormal f32 scales, 255 gives 2^127 rather than NaN, and a value of 2^128 or more is an infinity.
def mxfp4(block):
    scale = math.ldexp(1.0, block[0] - 128)
    qs = block[1:17]
    return [MXFP4_VALUES[q & 15] * scale for q in qs] + [MXFP4_VALUES[q >> 4] * scale for q in qs]


# The 6-bit scale and min of sub-block j (0 to 7) of Q4_K and Q5_K, from their 12 bytes.
# Sub-block j < 4 takes scale byte[j] & 63 and min byte[j + 4] & 63; j >= 4 takes scale (byte[j + 4] & 15) | (byte[j - 4] >> 6) << 4 and min (byte[j + 4] >> 4) | (byte[j] >> 6) << 4.
def k_scale_min(sm, j):
    if j < 4:
        return sm[j] & 63, sm[j + 4] & 63
    return (sm[j + 4] & 15) | (sm[j - 4] >> 6) << 4, (sm[j + 4] >> 4) | (sm[j] >> 6) << 4


# Q4_K: 256 values as eight sub-blocks of 32; an f16 scale d, an f16 min dmin, 12 bytes of 6-bit scales and mins, then 128 bytes of nibbles.
# The nibble bytes run in four groups of 32: group g's low nibbles are sub-block 2g and its high nibbles sub-block 2g + 1, each d * scale * nibble - dmin * min.
def q4_k(block):
    d, dmin = f16(block, 0), f16(block, 2)
    sm, qs = block[4:16], block[16:144]
    out = []
    for j in range(8):
        scale, low = k_scale_min(sm, j)
        group = qs[32 * (j // 2):32 * (j // 2) + 32]
        out.extend(d * scale * ((q >> 4 * (j % 2)) & 15) - dmin * low for q in group)
    return out


# Q5_K: Q4_K's d, dmin and scales, then 32 bytes of fifth bits and 128 bytes of nibbles in Q4_K's order.
# Value l of sub-block j takes bit j of fifth-bit byte l as its bit 4.
def q5_k(block):
    d, dmin = f16(block, 0), f16(block, 2)
    sm, qh, qs = block[4:16], block[16:48], block[48:176]
    out = []
    for j in range(8):
        scale, low = k_scale_min(sm, j)
        group = qs[32 * (j // 2):32 * (j // 2) + 32]
        out.extend(d * scale * (((q >> 4 * (j % 2)) & 15) | ((qh[l] >> j) & 1) << 4) - dmin * low
                   for l, q in enumerate(group))
    return out


# Q6_K: 256 values as 16 sub-blocks of 16; 128 bytes of low nibbles, 64 bytes of high bit pairs, 16 signed byte scales, then an f16 scale d.
# Value i, with h = i // 128, k = i % 128 // 32 and l = i % 32, takes nibble k // 2 of low byte 64h + 32(k % 2) + l and bits 2k and 2k + 1 of high byte 32h + l, and decodes as d * scale[i // 16] * (q - 32).
def q6_k(block):
    ql, qh = block[0:128], block[128:192]
    scales = struct.unpack_from("<16b", block, 192)
    d = f16(block, 208)
    out = []
    for i in range(256):
        h, k, l = i // 128, i % 128 // 32, i % 32
        q = (ql[64 * h + 32 * (k % 2) + l] >> 4 * (k // 2)) & 15 | ((qh[32 * h + l] >> 2 * k) & 3) << 4
        out.append(d * scales[i // 16] * (q - 32))
    return out


# Q2_K: 256 values as 16 sub-blocks of 16; 16 bytes of a 4-bit scale (low nibble) and a 4-bit min (high nibble) per sub-block, 64 bytes of 2-bit codes, then an f16 scale d and an f16 min dmin.
# Value i, with h = i // 128, j = i % 128 // 32 and l = i % 32, takes bits 2j and 2j + 1 of code byte 32h + l, and decodes as d * scale * code - dmin * min of sub-block i // 16.
def q2_k(block):
    scales, qs = block[0:16], block[16:80]
    d, dmin = f16(block, 80), f16(block, 82)
    out = []
    for i in range(256):
        h, j, l = i // 128, i % 128 // 32, i % 32
        sc = scales[i // 16]
        out.append(d * (sc & 15) * ((qs[32 * h + l] >> 2 * j) & 3) - dmin * (sc >> 4))
    return out


# The 6-bit scale of sub-block s (0 to 15) of Q3_K from its 12 scale bytes: the low four bits are nibble s // 8 of byte s % 8, and the high two bits are bits 2(s // 4) and 2(s // 4) + 1 of byte 8 + s % 4.
def q3_k_scale(sc, s):
    return (sc[s % 8] >> 4 * (s // 8)) & 15 | ((sc[8 + s % 4] >> 2 * (s // 4)) & 3) << 4


# Q3_K: 256 values as 16 sub-blocks of 16; 32 bytes of high bits, 64 bytes of 2-bit codes in Q2_K's order, 12 bytes of 6-bit scales, then an f16 scale d.
# Value i, with h, j and l as in Q2_K, takes bit 4h + j of high byte l: a set bit leaves the 2-bit code as it is, and a clear one subtracts 4, so codes run from -4 to 3.
# It decodes as d * (scale - 32) * code with the scale of sub-block i // 16.
def q3_k(block):
    hmask, qs, sc = block[0:32], block[32:96], block[96:108]
    d = f16(block, 108)
    out = []
    for i in range(256):
        h, j, l = i // 128, i % 128 // 32, i % 32
        code = (qs[32 * h + l] >> 2 * j) & 3
        if not (hmask[l] >> (4 * h + j)) & 1:
            code -= 4
        out.append(d * (q3_k_scale(sc, i // 16) - 32) * code)
    return out


# IQ4_XS: 256 values as eight sub-blocks of 32; an f16 scale d, a 16-bit word of high scale bits, four bytes of low scale bits, then 128 bytes of codes.
# Sub-block b's 6-bit scale is nibble b % 2 of low byte b // 2, with bits 2b and 2b + 1 of the word above it.
# Its 32 codes are the nibbles of bytes 16b to 16b + 15 in Q4_0's order, each d * (scale - 32) * IQ4_VALUES[nibble].
def iq4_xs(block):
    d = f16(block, 0)
    high = struct.unpack_from("<H", block, 2)[0]
    low, qs = block[4:8], block[8:136]
    out = []
    for b in range(8):
        dl = d * (((low[b // 2] >> 4 * (b % 2)) & 15 | ((high >> 2 * b) & 3) << 4) - 32)
        group = qs[16 * b:16 * b + 16]
        out.extend([dl * IQ4_VALUES[q & 15] for q in group] + [dl * IQ4_VALUES[q >> 4] for q in group])
    return out


PURE = {
    F32: lambda block: [struct.unpack("<f", bytes(block))[0]],
    F16: lambda block: [f16(block, 0)],
    BF16: lambda block: [bf16(block, 0)],
    Q8_0: q8_0, Q4_0: q4_0, Q4_1: q4_1, Q4_K: q4_k, Q5_K: q5_k, Q6_K: q6_k,
    Q2_K: q2_k, Q3_K: q3_k, IQ4_NL: iq4_nl, IQ4_XS: iq4_xs, MXFP4: mxfp4,
}


def decode(type_id, payload, count):
    """The first `count` values of `payload`, blocks of `type_id`, as Python floats holding f32 values; the pure form."""
    size = TYPES[type_id][2]
    blocks = payload_bytes(type_id, count) // size
    out = []
    for b in range(blocks):
        out.extend(to_f32(x) for x in PURE[type_id](payload[b * size:(b + 1) * size]))
    return out


def pack_f32(values):
    """Values from `decode` as little-endian f32 bytes, the form in which two decodes are compared."""
    return struct.pack("<%df" % len(values), *values)


# The numpy form: each function takes a (blocks, bytes per block) uint8 array and returns (blocks, values per block) float32.
# Products of small integers with an f16 scale are exact in f32, so the arithmetic runs in f32 in the pure form's order, and a sum or difference of two exact products rounds once, as there.

def _half(a, at):
    return np.ascontiguousarray(a[:, at:at + 2]).view("<f2").astype(np.float32)


def _nibbles(qs):
    return np.concatenate([qs & 15, qs >> 4], axis=1)


def _np_f32(a):
    return np.ascontiguousarray(a).view("<f4").astype(np.float32)


def _np_f16(a):
    return np.ascontiguousarray(a).view("<f2").astype(np.float32)


def _np_bf16(a):
    return (np.ascontiguousarray(a).view("<u2").astype(np.uint32) << 16).view(np.float32)


def _np_q8_0(a):
    return _half(a, 0) * np.ascontiguousarray(a[:, 2:34]).view(np.int8).astype(np.float32)


def _np_q4_0(a):
    return _half(a, 0) * (_nibbles(a[:, 2:18]).astype(np.float32) - 8)


def _np_q4_1(a):
    return _half(a, 0) * _nibbles(a[:, 4:20]).astype(np.float32) + _half(a, 2)


def _np_iq4_nl(a):
    return _half(a, 0) * np.array(IQ4_VALUES, dtype=np.float32)[_nibbles(a[:, 2:18])]


def _np_mxfp4(a):
    return np.ldexp(np.array(MXFP4_VALUES, dtype=np.float32)[_nibbles(a[:, 1:17])], a[:, 0:1].astype(np.int32) - 128)


def _np_k_scale_min(sm):
    scales, mins = [], []
    for j in range(8):
        if j < 4:
            scales.append(sm[:, j] & 63)
            mins.append(sm[:, j + 4] & 63)
        else:
            scales.append((sm[:, j + 4] & 15) | (sm[:, j - 4] >> 6) << 4)
            mins.append((sm[:, j + 4] >> 4) | (sm[:, j] >> 6) << 4)
    return np.stack(scales, axis=1).astype(np.float32), np.stack(mins, axis=1).astype(np.float32)


def _np_k_codes(qs):
    return np.concatenate([(qs[:, 32 * g:32 * g + 32] >> 4 * t) & 15 for g in range(4) for t in range(2)], axis=1)


def _np_q4_k(a):
    scales, mins = _np_k_scale_min(a[:, 4:16])
    d, dmin = _half(a, 0), _half(a, 2)
    return np.repeat(d * scales, 32, axis=1) * _np_k_codes(a[:, 16:144]).astype(np.float32) - np.repeat(dmin * mins, 32, axis=1)


def _np_q5_k(a):
    scales, mins = _np_k_scale_min(a[:, 4:16])
    d, dmin = _half(a, 0), _half(a, 2)
    qh = a[:, 16:48]
    fifth = np.concatenate([(qh >> j) & 1 for j in range(8)], axis=1)
    codes = _np_k_codes(a[:, 48:176]) | fifth << 4
    return np.repeat(d * scales, 32, axis=1) * codes.astype(np.float32) - np.repeat(dmin * mins, 32, axis=1)


def _np_q6_k(a):
    ql, qh = a[:, 0:128], a[:, 128:192]
    parts = []
    for h in range(2):
        for k in range(4):
            low = (ql[:, 64 * h + 32 * (k % 2):64 * h + 32 * (k % 2) + 32] >> 4 * (k // 2)) & 15
            parts.append(low | ((qh[:, 32 * h:32 * h + 32] >> 2 * k) & 3) << 4)
    codes = np.concatenate(parts, axis=1).astype(np.float32) - 32
    scales = np.ascontiguousarray(a[:, 192:208]).view(np.int8).astype(np.float32)
    return np.repeat(_half(a, 208) * scales, 16, axis=1) * codes


def _np_2bit_codes(qs):
    return np.concatenate([(qs[:, 32 * h:32 * h + 32] >> 2 * j) & 3 for h in range(2) for j in range(4)], axis=1)


def _np_q2_k(a):
    scales = a[:, 0:16]
    d, dmin = _half(a, 80), _half(a, 82)
    return (np.repeat(d * (scales & 15).astype(np.float32), 16, axis=1) * _np_2bit_codes(a[:, 16:80]).astype(np.float32)
            - np.repeat(dmin * (scales >> 4).astype(np.float32), 16, axis=1))


def _np_q3_k(a):
    hmask, sc = a[:, 0:32], a[:, 96:108]
    high = np.concatenate([(hmask >> (4 * h + j)) & 1 for h in range(2) for j in range(4)], axis=1)
    codes = _np_2bit_codes(a[:, 32:96]).astype(np.float32) - 4 * (1 - high).astype(np.float32)
    scales = np.stack([(sc[:, s % 8] >> 4 * (s // 8)) & 15 | ((sc[:, 8 + s % 4] >> 2 * (s // 4)) & 3) << 4
                       for s in range(16)], axis=1).astype(np.float32) - 32
    return np.repeat(_half(a, 108) * scales, 16, axis=1) * codes


def _np_iq4_xs(a):
    high = a[:, 2].astype(np.uint16) | a[:, 3].astype(np.uint16) << 8
    scales = np.stack([(a[:, 4 + b // 2] >> 4 * (b % 2)) & 15 | ((high >> 2 * b) & 3).astype(np.uint8) << 4
                       for b in range(8)], axis=1).astype(np.float32) - 32
    codes = np.concatenate([_nibbles(a[:, 8 + 16 * b:24 + 16 * b]) for b in range(8)], axis=1)
    return np.repeat(_half(a, 0) * scales, 32, axis=1) * np.array(IQ4_VALUES, dtype=np.float32)[codes]


NUMPY = {
    F32: _np_f32, F16: _np_f16, BF16: _np_bf16,
    Q8_0: _np_q8_0, Q4_0: _np_q4_0, Q4_1: _np_q4_1, Q4_K: _np_q4_k, Q5_K: _np_q5_k, Q6_K: _np_q6_k,
    Q2_K: _np_q2_k, Q3_K: _np_q3_k, IQ4_NL: _np_iq4_nl, IQ4_XS: _np_iq4_xs, MXFP4: _np_mxfp4,
}

# Blocks decoded at once, which bounds the temporaries of a large tensor.
NUMPY_CHUNK = 1 << 16


def decode_numpy(type_id, payload, count):
    """The first `count` values of `payload`, blocks of `type_id`, as a float32 array with the bits `decode` gives, a NaN's aside; the numpy form."""
    if np is None:
        raise RuntimeError("the numpy form of the decoders needs numpy")
    _, values, size = TYPES[type_id]
    blocks = payload_bytes(type_id, count) // size
    raw = np.frombuffer(payload, dtype=np.uint8, count=blocks * size).reshape(blocks, size)
    out = np.empty(count, dtype=np.float32)
    # MXFP4 overflows to infinities by definition, and an f16 scale that is an infinity times a zero code is a NaN in both forms.
    with np.errstate(over="ignore", invalid="ignore"):
        for start in range(0, blocks, NUMPY_CHUNK):
            end = min(start + NUMPY_CHUNK, blocks)
            out[start * values:end * values] = NUMPY[type_id](raw[start:end]).reshape(-1)
    return out


# GGUF files: the magic, a version (2 or 3), the tensor and metadata counts, the metadata, the tensor descriptions, then the data at the next multiple of general.alignment (32 when absent).
# Strings are a u64 byte count and UTF-8 bytes; each tensor has a name, a rank, its dimensions (the first varies fastest), a type and an offset into the data.

GGUF_MAGIC = 0x46554747
# Metadata value types: struct formats of the scalars, then 8 (string) and 9 (array: element type, u64 count, elements).
SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
STRING, ARRAY = 8, 9


class Tensor:
    def __init__(self, name, shape, type_id, offset):
        self.name, self.shape, self.type, self.offset = name, shape, type_id, offset

    @property
    def count(self):
        return math.prod(self.shape)

    @property
    def nbytes(self):
        return payload_bytes(self.type, self.count)


class _Reader:
    def __init__(self, f):
        self.f = f

    def take(self, fmt):
        size = struct.calcsize(fmt)
        data = self.f.read(size)
        if len(data) != size:
            raise ValueError("GGUF file ends inside its header")
        return struct.unpack(fmt, data)[0]

    def string(self):
        size = self.take("<Q")
        data = self.f.read(size)
        if len(data) != size:
            raise ValueError("GGUF file ends inside a string")
        return data.decode("utf-8")

    def value(self, vtype):
        if vtype in SCALARS:
            return self.take(SCALARS[vtype])
        if vtype == STRING:
            return self.string()
        if vtype == ARRAY:
            element, count = self.take("<I"), self.take("<Q")
            return element, [self.value(element) for _ in range(count)]
        raise ValueError("unknown GGUF metadata type %d" % vtype)


class GGUF:
    """A GGUF file's metadata, as {key: (value type, value)} in file order with an array's value being (element type, list), and its tensors, whose `offset` is from the start of the file."""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            r = _Reader(f)
            if r.take("<I") != GGUF_MAGIC:
                raise ValueError("%s is not a GGUF file" % path)
            self.version = r.take("<I")
            if self.version not in (2, 3):
                raise ValueError("GGUF version %d is not read" % self.version)
            n_tensors, n_metadata = r.take("<Q"), r.take("<Q")
            self.metadata = {}
            for _ in range(n_metadata):
                key = r.string()
                vtype = r.take("<I")
                self.metadata[key] = (vtype, r.value(vtype))
            infos = []
            for _ in range(n_tensors):
                name = r.string()
                shape = [r.take("<Q") for _ in range(r.take("<I"))]
                infos.append((name, shape, r.take("<I"), r.take("<Q")))
            self.alignment = self.metadata.get("general.alignment", (4, 32))[1]
            start = f.tell() + -f.tell() % self.alignment
        self.tensors = [Tensor(name, shape, type_id, start + offset) for name, shape, type_id, offset in infos]

    def value(self, key, default=None):
        return self.metadata[key][1] if key in self.metadata else default

    def raw(self, tensor):
        with open(self.path, "rb") as f:
            f.seek(tensor.offset)
            data = f.read(tensor.nbytes)
        if len(data) != tensor.nbytes:
            raise ValueError("%s: tensor %s runs past the end of the file" % (self.path, tensor.name))
        return data

    def decode(self, tensor, numpy=True):
        """A tensor's values, flat with the first dimension varying fastest: a float32 array from the numpy form, or a list from the pure one."""
        raw = self.raw(tensor)
        return decode_numpy(tensor.type, raw, tensor.count) if numpy else decode(tensor.type, raw, tensor.count)


def _pack_string(text):
    data = text.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def _pack_value(vtype, value):
    if vtype in SCALARS:
        return struct.pack(SCALARS[vtype], value)
    if vtype == STRING:
        return _pack_string(value)
    if vtype == ARRAY:
        element, items = value
        return struct.pack("<IQ", element, len(items)) + b"".join(_pack_value(element, item) for item in items)
    raise ValueError("unknown GGUF metadata type %d" % vtype)


def write_gguf(path, metadata, tensors):
    """Write a GGUF v3 file: `metadata` as GGUF.metadata holds it, and `tensors` as (name, shape, type, data) with data padded to general.alignment (32 when absent)."""
    alignment = metadata.get("general.alignment", (4, 32))[1]
    header = [struct.pack("<IIQQ", GGUF_MAGIC, 3, len(tensors), len(metadata))]
    for key, (vtype, value) in metadata.items():
        header.append(_pack_string(key) + struct.pack("<I", vtype) + _pack_value(vtype, value))
    offset = 0
    for name, shape, type_id, data in tensors:
        if len(data) != payload_bytes(type_id, math.prod(shape)):
            raise ValueError("tensor %s: %d bytes for %s %s" % (name, len(data), type_name(type_id), shape))
        header.append(_pack_string(name) + struct.pack("<I", len(shape)) + struct.pack("<%dQ" % len(shape), *shape)
                      + struct.pack("<IQ", type_id, offset))
        offset += len(data) + -len(data) % alignment
    with open(path, "wb") as f:
        head = b"".join(header)
        f.write(head + b"\0" * (-len(head) % alignment))
        for _, _, _, data in tensors:
            f.write(data)
            f.write(b"\0" * (-len(data) % alignment))
