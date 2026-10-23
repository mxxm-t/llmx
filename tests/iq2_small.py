"""Shared tiny HF fixtures for the IQ2_XXS and IQ2_XS raw-block families."""

import spec_decode as sd
from raw_blocks import iq2_xs_block, iq2_xxs_block
from block_fixture import fixture as block_fixture, run_family


def _raw_block(next_value, block, index_mask, scale_count):
    bits = next_value()
    d = ((bits >> 16) & 0x8000) | 0x0400 | ((bits >> 8) & 0x03ff)
    if block % 257 in (0, 1, 2):
        d = (d & 0x8000) | (0, 1, 0x03ff)[block % 257]
    indices = [(next_value() >> 16) & index_mask for _ in range(32)]
    scales = [(next_value() >> 16) & 15 for _ in range(scale_count)]
    codes = [(next_value() >> 16) & 127 for _ in range(32)]
    return d, indices, scales, codes


def raw_block_xxs(next_value, block):
    return iq2_xxs_block(*_raw_block(next_value, block, 255, 8))


def raw_block_xs(next_value, block):
    return iq2_xs_block(*_raw_block(next_value, block, 511, 16))


def fixture_xxs(tied, moe, seed=12345, context=16, split=False):
    """Build IQ2_XXS matrices and independent decoded HF weights."""
    return block_fixture(tied, moe, sd.IQ2_XXS, raw_block_xxs, seed, context, split)


def fixture_xs(tied, moe, seed=12345, context=16, split=False):
    """Build IQ2_XS matrices and independent decoded HF weights."""
    return block_fixture(tied, moe, sd.IQ2_XS, raw_block_xs, seed, context, split)


def run_xxs():
    return run_family("iq2-xxs", "IQ2_XXS", fixture_xxs, family="iq2-xxs")


def run_xs():
    return run_family("iq2-xs", "IQ2_XS", fixture_xs, family="iq2-xs")
