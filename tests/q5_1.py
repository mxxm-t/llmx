import struct

import spec_decode as sd
from block_fixture import VARIANTS, packed_hash, run_family
from block_fixture import fixture as block_fixture
from f32 import weight_hash


def raw_block(next_value, b):
    bits = next_value()
    sign = (bits >> 16) & 0x8000
    exponent = 7 + ((bits >> 24) % 3)
    scale = sign | (exponent << 10)
    minimum = (sign ^ 0x8000) | ((exponent + 4) << 10)
    codes = [(next_value() >> 16) & 31 for _ in range(32)]
    if b % 257 in (0, 1, 2):
        scale = sign | (0, 1, 0x03ff)[b % 257]
        minimum = sign ^ 0x8000
    high = sum((code >> 4) << i for i, code in enumerate(codes))
    low = bytes((codes[j] & 15) | ((codes[j + 16] & 15) << 4) for j in range(16))
    return struct.pack("<HHI", scale, minimum, high) + low


def fixture(tied, moe, seed=12345, context=16, split=False):
    """Raw Q5_1 matrices, F32 norms/router, and independently decoded HF weights."""
    return block_fixture(tied, moe, sd.Q5_1, raw_block, seed, context, split)


def run():
    return run_family("q5-1", "Q5_1", fixture, family="q5-1")
