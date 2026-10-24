"""Paired raw-block TQ fixtures and independent tiny-model HF consumers."""

import spec_decode as sd
from block_fixture import fixture as block_fixture, run_family
from raw_blocks import tq1_block, tq2_block

TQ2_SEED = 12346


def raw_block(type_id, next_value, block):
    bits = next_value()
    scale = ((bits >> 16) & 0x8000) | ((7 + (bits >> 24) % 3) << 10) | ((bits >> 8) & 1023)
    if block % 257 in (0, 1, 2):
        scale = (scale & 0x8000) | (0, 1, 0x03ff)[block % 257]
    # Raw bytes include patterns that ordinary ternary encoders do not emit.
    count = 52 if type_id == sd.TQ1_0 else 64
    codes = [(next_value() >> 16) & 255 for _ in range(count)]
    if type_id == sd.TQ1_0:
        return tq1_block(codes[:48], codes[48:], scale)
    assert type_id == sd.TQ2_0
    return tq2_block(codes, scale)


def fixture(type_id, tied, moe, seed=12345, context=16, split=False):
    """Exact spec-decoded TQ weights; the shared owner keeps norms/router F32."""
    assert type_id in (sd.TQ1_0, sd.TQ2_0)
    return block_fixture(tied, moe, type_id,
                         lambda next_value, block: raw_block(type_id, next_value, block),
                         seed, context, split)


def fixture_tq1(tied, moe, seed=12345, context=16, split=False):
    return fixture(sd.TQ1_0, tied, moe, seed, context, split)


def fixture_tq2(tied, moe, seed=TQ2_SEED, context=16, split=False):
    return fixture(sd.TQ2_0, tied, moe, seed, context, split)


def run_tq1():
    return run_family("tq1-0", "TQ1_0", fixture_tq1, family="tq1-0")


def run_tq2():
    return run_family("tq2-0", "TQ2_0", fixture_tq2, family="tq2-0")
