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
Q5_0, Q5_1, IQ3_S = 6, 7, 21

# GGUF type id: name, values per block, bytes per block.
TYPES = {
    F32: ("F32", 1, 4),
    F16: ("F16", 1, 2),
    Q4_0: ("Q4_0", 32, 18),
    Q4_1: ("Q4_1", 32, 20),
    Q5_0: ("Q5_0", 32, 22),
    Q5_1: ("Q5_1", 32, 24),
    Q8_0: ("Q8_0", 32, 34),
    Q2_K: ("Q2_K", 256, 84),
    Q3_K: ("Q3_K", 256, 110),
    Q4_K: ("Q4_K", 256, 144),
    Q5_K: ("Q5_K", 256, 176),
    Q6_K: ("Q6_K", 256, 210),
    IQ3_S: ("IQ3_S", 256, 110),
    IQ4_NL: ("IQ4_NL", 32, 18),
    IQ4_XS: ("IQ4_XS", 256, 136),
    BF16: ("BF16", 1, 2),
    MXFP4: ("MXFP4", 32, 17),
}

# Canonical IQ3_S dictionary as decoded bytes, independently pinned from ggml
# eefc4e7321c869496146697d63362f073941aed6 ggml-common.h, SHA256
# bd1af4945a1717c65610b0284e4628b9a1ba3b306fae3a06f6e5f597356e349f.
# Test data is separate from the runtime packed-word dictionary.
IQ3_S_GRID = bytes.fromhex(
    "0101010103010101050101010b0101010f010101010301010303010105030101090301010d03010101050101030501010b050101070701010109010105090101"
    "0b0901010f090101030b0101070b0101010d0101050d0101030f0101090f01010f0f01010101030103010301050103010901030101030301030303010b030301"
    "01050301070503010f050301030703010b07030109090301030d03010b0d0301050f030101010501030105010b0105010f01050101030501070305010d030501"
    "030505010b0505010107050109070501050905010b0905010f090501030b0501070b0501010f0501070f050107010701030307010b0307010105070105050701"
    "03070701070707010d07070109090701010b0701050b07010f0d0701030f07010b0f070101010901070309010f03090103050901090509010507090101090901"
    "07090901030b0901010f090105010b0109010b0101050b0105050b010d050b0107070b0103090b010b090b010f090b010d0d0b01070f0b010d010d0103030d01"
    "07030d0103070d01050b0d01030f0d0101010f0105010f0109010f0101050f0105050f010d050f0107070f01010b0f01090b0f01010101030301010305010103"
    "090101030103010303030103070301030b0301030f030103010501030505010303070103090701030d070103090b01030d0b0103030d0103050f010301010303"
    "03010303070103030d010303010303030903030303050303010703030707030303090303010b0303050b0303010f03030d0f030301010503050305030b030503"
    "0f03050301050503090505030507050301090503070905030b0b0503010d0503050f050303010703090107030f0107030103070307030703030507030f050703"
    "010707030907070303090703050d0703010f0703070109030b01090305030903090309030307090307070903050909030d090903010b0903090b090303010b03"
    "01030b0307030b0303050b0301070b0305070b03030b0b0301050d0309050d030f050d0309090d030d090d0303010f0307010f0301030f0305030f0303050f03"
    "0b070f0303090f03050d0f03010f0f030101010503010105070101050b0101050f0101050103010505030105090301050d03010503050105070501050f050105"
    "010701050507010503090105070901050b090105010b0105050b01050f0d0105010f0105070f01050b0f0105010103050501030501030305070303050f030305"
    "050503050b050305030703050907030505090305030b030503010505090105050f0105050305050507050505010705050f07050503090505070b05050f0b0505"
    "030f0505090f050501010705050107050b010705030307050505070509050705030707050707070505090705010b07050d0d0705030109050f01090501050905"
    "07050905050709050b07090503090905050f09050b0f090509010b0503030b0505050b050f070b0501090b05070b0b05010f0b0501010d0505010d050f010d05"
    "03050d050b0b0d05030d0d050b010f0503030f050d050f0501070f0507090f05010b0f050501010703030107070301070b0301070f0301070505010703070107"
    "070701070b07010705090107090901070f090107030b0107070d0107030f010703010307070103070b01030709030307030503070705030701090307010d0307"
    "050f03070d0f03070101050705030507010505070507050709070507010b050703010707010307070903070703050707070507070f0507070107070703090707"
    "070907070f0907070b0b0707070f070707010907030309070d0309070505090703070907050b0907010d0907090d090703010b0701030b0705030b070b050b07"
    "05070b0709090b070d0b0b07070f0b070d030d0703090d0703010f0707010f0701050f0705050f070b070f070101010909010109050301090105010909050109"
    "0f0501090507010903090109010b0109010f0109050103090f010309030303090703030905050309010703090b07030907090309030b03090b0b030903010509"
    "07010509010305090b0305090305050907070509010905090f0b0509050d0509010f05090901070903030709070307090105070905050709030707090b070709"
    "0101090905010909090509090f07090901090909030f09090b010b090f010b0903050b09050d0b0907030d0909070d09010d0d0901030f090b030f0901070f09"
    "07090f09030b0f090501010b0103010b0903010b0505010b0109010b0909010b0f09010b050b010b0d0d010b090f010b0301030b0701030b0b01030b0503030b"
    "0305030b0507030b050f030b0101050b0303050b0705050b0107050b0d07050b070b050b0501070b0f01070b0103070b0f05070b0909070b030b070b0b0d070b"
    "070f070b0301090b0901090b0105090b0507090b0d09090b05030b0b0d050b0b030b0b0b070b0b0b05090d0b05010f0b09010f0b05050f0b0303010d0703010d"
    "0b03010d0307010d0707010d010d010d0101030d0105030d0f05030d090d030d0503050d0907050d0509050d0b0b050d050d050d010f050d0101070d0903070d"
    "0305070d0109070d0b05090d0709090d050d090d01010b0d07010b0d09070b0d010d0b0d0b010d0d01090d0d03030f0d07030f0d0101010f0901010f0f01010f"
    "0105010f0505010f0d07010f0109010f090b010f050d010f0501030f0303030f0905030f0709030f0b09030f0301050f0901050f0103050f0d03050f0305050f"
    "0107050f030b050f0501070f0507070f0b07070f070b070f0301090f0b01090f0703090f0105090f010b090f05050b0f05090b0f05010d0f03070d0f01010f0f"
)

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


# Q5_0: half scale, fifth bits at bytes 2..5, then nibbles; each d * (code - 16).
def q5_0(block):
    d = f16(block, 0)
    high = int.from_bytes(block[2:6], "little")
    return [d * ((((block[6 + i % 16] >> (4 * (i // 16))) & 15) | ((high >> i) & 1) << 4) - 16)
            for i in range(32)]


# Q5_1: f16 scale and min, four bytes of fifth bits (bit i belongs to value i), then nibbles in Q4_0's order.
def q5_1(block):
    d, m = f16(block, 0), f16(block, 2)
    high = int.from_bytes(block[4:8], "little")
    return [d * (((block[8 + i % 16] >> (4 * (i // 16))) & 15) | ((high >> i) & 1) << 4) + m
            for i in range(32)]


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
# Value i, with h, j and l as in Q2_K, takes bit 4h + j of high byte l, a clear bit subtracting 4 so codes run from -4 to 3, and decodes as d * (scale - 32) * code with the scale of sub-block i // 16.
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
# Sub-block b's 6-bit scale is nibble b % 2 of low byte b // 2 with bits 2b and 2b + 1 of the word above it, and its 32 codes are the nibbles of bytes 16b to 16b + 15 in Q4_0's order, each d * (scale - 32) * IQ4_VALUES[nibble].
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


# IQ3_S: half d, 64 low dictionary-index bytes, 64 high index bits,
# 256 independent sign bits, then eight 4-bit scales (low nibble first).
# A dictionary vector supplies four consecutive positive magnitudes; each
# 32-value group scales by d * (1 + 2 * its nibble), followed by the sign.
def iq3_s(block):
    d = f16(block, 0)
    high = int.from_bytes(block[66:74], "little")
    signs = int.from_bytes(block[74:106], "little")
    scales = int.from_bytes(block[106:110], "little")
    out = []
    for i in range(256):
        group, lane = divmod(i, 4)
        index = block[2 + group] + 256 * ((high >> group) & 1)
        magnitude = IQ3_S_GRID[4 * index + lane]
        value = d * (1 + 2 * ((scales >> (4 * (i // 32))) & 15)) * magnitude
        out.append(-value if (signs >> i) & 1 else value)
    return out


PURE = {
    F32: lambda block: [struct.unpack("<f", bytes(block))[0]],
    F16: lambda block: [f16(block, 0)],
    BF16: lambda block: [bf16(block, 0)],
    Q8_0: q8_0, Q4_0: q4_0, Q4_1: q4_1, Q5_0: q5_0, Q5_1: q5_1, Q4_K: q4_k, Q5_K: q5_k, Q6_K: q6_k,
    Q2_K: q2_k, Q3_K: q3_k, IQ3_S: iq3_s, IQ4_NL: iq4_nl, IQ4_XS: iq4_xs, MXFP4: mxfp4,
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


def _np_q5_0(a):
    high = np.unpackbits(a[:, 2:6], axis=1, bitorder="little")
    codes = _nibbles(a[:, 6:22]) | (high << 4)
    return _half(a, 0) * (codes.astype(np.float32) - 16)


def _np_q5_1(a):
    high = np.unpackbits(a[:, 4:8], axis=1, bitorder="little")
    codes = _nibbles(a[:, 8:24]) | (high << 4)
    return _half(a, 0) * codes.astype(np.float32) + _half(a, 2)


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


def _np_iq3_s(a):
    high = np.unpackbits(a[:, 66:74], axis=1, bitorder="little").astype(np.uint16)
    indices = a[:, 2:66].astype(np.uint16) + 256 * high
    magnitudes = np.frombuffer(IQ3_S_GRID, dtype=np.uint8).reshape(512, 4)[indices].reshape(-1, 256)
    scales = np.stack([a[:, 106 + g // 2] >> (4 * (g % 2)) & 15 for g in range(8)], axis=1)
    values = np.repeat(_half(a, 0) * (1 + 2 * scales).astype(np.float32), 32, axis=1) * magnitudes.astype(np.float32)
    signs = np.unpackbits(a[:, 74:106], axis=1, bitorder="little")
    return np.where(signs != 0, -values, values)


NUMPY = {
    F32: _np_f32, F16: _np_f16, BF16: _np_bf16,
    Q8_0: _np_q8_0, Q4_0: _np_q4_0, Q4_1: _np_q4_1, Q5_0: _np_q5_0, Q5_1: _np_q5_1, Q4_K: _np_q4_k, Q5_K: _np_q5_k, Q6_K: _np_q6_k,
    Q2_K: _np_q2_k, Q3_K: _np_q3_k, IQ3_S: _np_iq3_s, IQ4_NL: _np_iq4_nl, IQ4_XS: _np_iq4_xs, MXFP4: _np_mxfp4,
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

    def raw(self, tensor, first=0, count=None):
        count = tensor.count - first if count is None else count
        if not isinstance(first, int) or not isinstance(count, int) or first < 0 or count < 0 or first + count > tensor.count:
            raise ValueError("tensor %s: invalid element span" % tensor.name)
        skip = payload_bytes(tensor.type, first)
        size = payload_bytes(tensor.type, count)
        with open(self.path, "rb") as f:
            f.seek(tensor.offset + skip)
            data = f.read(size)
        if len(data) != size:
            raise ValueError("%s: tensor %s runs past the end of the file" % (self.path, tensor.name))
        return data

    def decode(self, tensor, numpy=True):
        """A tensor's values, flat with the first dimension varying fastest: a float32 array from the numpy form, or a list from the pure one."""
        return self.decode_span(tensor, 0, tensor.count, numpy)

    def decode_span(self, tensor, first, count, numpy=True):
        """Decode only a flat element span, aligned to stored blocks, without reading the rest of a table or expert stack."""
        raw = self.raw(tensor, first, count)
        return decode_numpy(tensor.type, raw, count) if numpy else decode(tensor.type, raw, count)


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
