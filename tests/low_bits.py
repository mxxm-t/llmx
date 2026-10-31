"""Independent raw Q1_0 tiny-model fixtures and HF consumers."""

import spec_decode as sd
from block_fixture import fixture as block_fixture, run_family
from raw_blocks import q1_0_block


def raw_block(type_id, next_value, block):
    bits = next_value()
    scale = ((bits >> 16) & 0x8000) | ((7 + (bits >> 24) % 7) << 10) | ((bits >> 8) & 1023)
    assert type_id == sd.Q1_0
    return q1_0_block(scale, [(next_value() >> 24) & 1 for _ in range(128)])


def fixture(type_id, tied, moe, seed=12345, context=16, split=False):
    """Build raw blocks, then let spec_decode independently form the HF weights."""
    assert type_id == sd.Q1_0
    return block_fixture(tied, moe, type_id,
                         lambda next_value, block: raw_block(type_id, next_value, block),
                         seed, context, split)


def fixture_q1(tied, moe, seed=12345, context=16, split=False):
    return fixture(sd.Q1_0, tied, moe, seed, context, split)



def run():
    return run_family("q1-0", "Q1_0", fixture_q1, family="q1-0")
