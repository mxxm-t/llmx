import struct

import spec_decode as sd
from block_fixture import VARIANTS, packed_hash, run_family
from block_fixture import fixture as block_fixture
from f32 import weight_hash


def raw_block(next_value, b):
    # Modest signed scales keep the HF model finite; sparse zero/subnormal scales still reach embedding, dense and routed products.
    bits = next_value()
    scale = ((bits >> 16) & 0x8000) | ((3 + ((bits >> 24) % 3)) << 10)
    codes = [(next_value() >> 16) & 15 for _ in range(32)]
    if b % 257 in (0, 1, 2):
        scale = (scale & 0x8000) | (0, 1, 0x03ff)[b % 257]
    return struct.pack("<H", scale) + bytes(codes[j] | (codes[j + 16] << 4) for j in range(16))


def fixture(tied, moe, seed=12345, context=16, split=False):
    """Raw IQ4_NL matrices, F32 norms/router, and independently decoded HF weights."""
    return block_fixture(tied, moe, sd.IQ4_NL, raw_block, seed, context, split)


def run():
    return run_family("iq4-nl", "IQ4_NL", fixture, family=None)
