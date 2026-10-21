import spec_decode as sd
from block_fixture import VARIANTS, packed_hash, run_family
from block_fixture import fixture as block_fixture
from f32 import weight_hash
from raw_blocks import q3_k_block


def raw_block(next_value, b):
    bits = next_value()
    d = ((bits >> 16) & 0x8000) | 0x1400 | ((bits >> 8) & 0x03ff)
    if b % 257 in (0, 1, 2):
        d = (d & 0x8000) | (0, 1, 0x03ff)[b % 257]
    scales = [(next_value() >> 16) & 63 for _ in range(16)]
    codes = [((next_value() >> 16) & 7) - 4 for _ in range(256)]
    return q3_k_block(d, scales, codes)


def fixture(tied, moe, seed=12345, context=16, split=False):
    """Raw Q3_K matrices and independently decoded weights; norms and routers stay F32."""
    return block_fixture(tied, moe, sd.Q3_K, raw_block, seed, context, split)


def run():
    return run_family("q3-k", "Q3_K", fixture, family="q3-k")
