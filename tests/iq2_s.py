"""Independent IQ2_S format conversion and tiny-model HF checks, reported separately."""
import hashlib
import json
import os
import tempfile

import common
import spec_decode as sd
from raw_blocks import iq2_s_block, iq2_s_cases
from block_fixture import fixture as block_fixture, run_family


def raw_block(next_value, b):
    bits = next_value()
    d = ((bits >> 16) & 0x8000) | 0x0400 | ((bits >> 8) & 0x03ff)
    if b % 257 in (0, 1, 2):
        d = (d & 0x8000) | (0, 1, 0x03ff)[b % 257]
    indices = [(next_value() >> 16) & 1023 for _ in range(32)]
    scales = [(next_value() >> 16) & 15 for _ in range(16)]
    signs = [bool((next_value() >> 16) & 1) for _ in range(256)]
    return iq2_s_block(d, indices, scales, signs)


def fixture(tied, moe, seed=12345, context=16, split=False):
    """Raw IQ2_S matrices and independently decoded HF weights; norms and routers stay F32."""
    return block_fixture(tied, moe, sd.IQ2_S, raw_block, seed, context, split)


def check_format():
    blocks = [iq2_s_block(*case) for case in iq2_s_cases()]
    payload = b"".join(blocks)
    assert len(blocks) == 1611 and hashlib.sha256(payload).hexdigest() == "2a66d6243194b3483eb2da6a3e68bef144783ab85b9a36fd661844a543468d10", "IQ2_S raw inputs changed"
    decoded = sd.pack_f32(sd.decode(sd.IQ2_S, payload, 256 * len(blocks)))
    assert hashlib.sha256(decoded).hexdigest() == "426e9fbe558f8678cd2e98753f71ec04a1f6ec521bc5474b56993356f0d7ab39", "IQ2_S expected decoded bytes changed"
    tensors = [("all_fields", [256, len(blocks)], sd.IQ2_S, payload),
               ("multiple_blocks", [512, 3], sd.IQ2_S, b"".join(blocks[:6])),
               ("empty_rows", [256, 0], sd.IQ2_S, b"")]
    with tempfile.TemporaryDirectory(prefix="llmx_iq2_s_format_") as directory:
        source, metadata, output = (os.path.join(directory, "input." + ext) for ext in ("gguf", "json", "bin"))
        for label, fields, expected in (("fields", tensors, decoded + decoded[:6 * 256 * 4]), ("empty", [], b"")):
            sd.write_gguf(source, {}, fields)
            with open(source, "rb") as stream:
                before = hashlib.sha256(stream.read()).hexdigest()
            if fields:
                assert before == "55f8db08847b5dd15861c249e7f113148904030cf35b82f4ad3855d81050c834", "IQ2_S written GGUF changed"
            else:
                assert before == "a3718c7940c34ff91a906d0758debb656aed43d1aa8b23b99f67cb106cdbc881", "Empty GGUF changed"
            rc, diagnostic = common.run(["dequantize", source, metadata, output])
            assert rc == 0, "IQ2_S %s conversion failed: %s" % (label, diagnostic)
            with open(output, "rb") as stream:
                assert stream.read() == expected, "IQ2_S %s decoded F32 bytes/signs differ from the independent format decoder" % label
            with open(metadata, encoding="utf-8") as stream:
                assert json.load(stream)["tensors"] == [{"name": name, "shape": shape} for name, shape, _, _ in fields], "IQ2_S converted tensor extents changed"
            with open(source, "rb") as stream:
                assert hashlib.sha256(stream.read()).hexdigest() == before, "IQ2_S conversion changed its source GGUF"
    print("iq2-s: format-only CLI decode, 412416 field values plus multi-block rows, signed zeros and empty rows/model, exact F32 bytes [ok]")
    return True


def run():
    check_format()
    return run_family("iq2-s", "IQ2_S", fixture, family="iq2-s")
