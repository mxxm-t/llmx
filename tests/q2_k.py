import spec_decode as sd
from block_fixture import VARIANTS, packed_hash, run_family
from block_fixture import fixture as block_fixture
from f32 import weight_hash
from raw_blocks import q2_k_block


def raw_block(next_value, b):
    # Signed finite scales exercise both affine terms; sparse edge blocks retain zero and subnormal scale coverage.
    bits = next_value()
    d = ((bits >> 16) & 0x8000) | 0x1400 | ((bits >> 8) & 0x03ff)
    bits = next_value()
    m = ((bits >> 16) & 0x8000) | 0x1400 | ((bits >> 8) & 0x03ff)
    if b % 257 in (0, 1, 2):
        d = (d & 0x8000) | (0, 1, 0x03ff)[b % 257]
    scales = [(next_value() >> 16) & 255 for _ in range(16)]
    codes = [(next_value() >> 16) & 3 for _ in range(256)]
    return q2_k_block(d, m, scales, codes)


def fixture(tied, moe, seed=12345, context=16, split=False):
    """Raw Q2_K matrices, whole-block rows, F32 norms/router and independent decoded weights for HF."""
    return block_fixture(tied, moe, sd.Q2_K, raw_block, seed, context, split)


def run():
    return run_family("q2-k", "Q2_K", fixture)
