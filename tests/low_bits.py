"""Independent raw Q1_0/Q2_0 tiny-model fixtures and HF consumers."""

import common
import spec_decode as sd
from block_fixture import fixture as block_fixture, run_family
from raw_blocks import q1_0_block, q2_0_block


def raw_block(type_id, next_value, block):
    bits = next_value()
    scale = ((bits >> 16) & 0x8000) | ((7 + (bits >> 24) % 7) << 10) | ((bits >> 8) & 1023)
    if type_id == sd.Q1_0:
        return q1_0_block(scale, [(next_value() >> 24) & 1 for _ in range(128)])
    assert type_id == sd.Q2_0
    return q2_0_block(scale, [(next_value() >> 24) & 3 for _ in range(64)])


def fixture(type_id, tied, moe, seed=12345, context=16, split=False):
    """Build raw blocks, then let spec_decode independently form the HF weights."""
    assert type_id in (sd.Q1_0, sd.Q2_0)
    return block_fixture(tied, moe, type_id,
                         lambda next_value, block: raw_block(type_id, next_value, block),
                         seed, context, split)


def fixture_q1(tied, moe, seed=12345, context=16, split=False):
    return fixture(sd.Q1_0, tied, moe, seed, context, split)


def fixture_q2(tied, moe, seed=12346, context=16, split=False):
    return fixture(sd.Q2_0, tied, moe, seed, context, split)


def run():
    results = [run_family("q1-0", "Q1_0", fixture_q1, family="q1-0"),
               run_family("q2-0", "Q2_0", fixture_q2, family="q2-0")]
    return common.SKIPPED if common.SKIPPED in results else all(results)
