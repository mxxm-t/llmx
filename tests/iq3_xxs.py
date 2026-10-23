"""Independent IQ3_XXS format conversion and tiny-model HF checks, reported separately."""
import hashlib
import json
import os
import tempfile

import common
import spec_decode as sd
from raw_blocks import iq3_xxs_block, iq3_xxs_cases
from block_fixture import fixture as block_fixture, run_family


def raw_block(next_value, b):
    bits = next_value()
    d = ((bits >> 16) & 0x8000) | 0x0400 | ((bits >> 8) & 0x03ff)
    if b % 257 in (0, 1, 2):
        d = (d & 0x8000) | (0, 1, 0x03ff)[b % 257]
    indices = [(next_value() >> 16) & 255 for _ in range(64)]
    scales = [(next_value() >> 16) & 15 for _ in range(8)]
    sign_codes = [(next_value() >> 16) & 127 for _ in range(32)]
    return iq3_xxs_block(d, indices, scales, sign_codes)


def fixture(tied, moe, seed=12345, context=16, split=False):
    """Raw IQ3_XXS matrices and independently decoded HF weights; norms and routers stay F32."""
    return block_fixture(tied, moe, sd.IQ3_XXS, raw_block, seed, context, split)


def check_format():
    blocks = [iq3_xxs_block(*case) for case in iq3_xxs_cases()]
    payload = b"".join(blocks)
    assert len(blocks) == 47 and hashlib.sha256(payload).hexdigest() == "5f7c9bea4301ea265737f3702d39fdea1357d200bcb9b629a4bff65383ac5956", "IQ3_XXS raw inputs changed"
    decoded = sd.pack_f32(sd.decode(sd.IQ3_XXS, payload, 256 * len(blocks)))
    assert hashlib.sha256(decoded).hexdigest() == "868f3ed108fa269ce3e91f0ce383856e9b6fc5dd3c0291bf0e28ada6ad5d3a80", "IQ3_XXS expected decoded bytes changed"
    tensors = [("all_fields", [256, len(blocks)], sd.IQ3_XXS, payload),
               ("multiple_blocks", [512, 3], sd.IQ3_XXS, b"".join(blocks[:6])),
               ("empty_rows", [256, 0], sd.IQ3_XXS, b"")]
    with tempfile.TemporaryDirectory(prefix="llmx_iq3_xxs_format_") as directory:
        source, metadata, output = (os.path.join(directory, "input." + ext) for ext in ("gguf", "json", "bin"))
        for label, fields, expected in (("fields", tensors, decoded + decoded[:6 * 256 * 4]), ("empty", [], b"")):
            sd.write_gguf(source, {}, fields)
            with open(source, "rb") as stream:
                before = hashlib.sha256(stream.read()).hexdigest()
            if fields:
                assert before == "3eb9be262f1fc79fdd7a6345419f15995600049c66765045b845563a19cb6ccc", "IQ3_XXS written GGUF changed"
            else:
                assert before == "a3718c7940c34ff91a906d0758debb656aed43d1aa8b23b99f67cb106cdbc881", "Empty GGUF changed"
            rc, diagnostic = common.run(["dequantize", source, metadata, output])
            assert rc == 0, "IQ3_XXS %s conversion failed: %s" % (label, diagnostic)
            with open(output, "rb") as stream:
                assert stream.read() == expected, "IQ3_XXS %s decoded F32 bytes/signs differ from the independent format decoder" % label
            with open(metadata, encoding="utf-8") as stream:
                assert json.load(stream)["tensors"] == [{"name": name, "shape": shape} for name, shape, _, _ in fields], "IQ3_XXS converted tensor extents changed"
            with open(source, "rb") as stream:
                assert hashlib.sha256(stream.read()).hexdigest() == before, "IQ3_XXS conversion changed its source GGUF"
    print("iq3-xxs: format-only CLI decode, 12032 field values plus multi-block rows, signed zeros and empty rows/model, exact F32 bytes [ok]")
    return True


def run():
    check_format()
    return run_family("iq3-xxs", "IQ3_XXS", fixture, family="iq3-xxs")
