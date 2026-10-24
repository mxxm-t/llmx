"""IQ1_S/M conversion and independent tiny-model HF checks."""
from functools import partial
import hashlib
import json
from pathlib import Path
import tempfile

import common
import spec_decode as sd
from raw_blocks import iq1_block, iq1_cases
from block_fixture import fixture as block_fixture, run_family


def raw_block(type_id, next_value, b):
    bits = next_value()
    d = ((bits >> 16) & 0x8000) | 0x1800 | ((bits >> 8) & 0x03ff)
    if b % 257 in (0, 1, 2):
        d = (d & 0x8000) | (0, 1, 0x03ff)[b % 257]
    indices = [(next_value() >> 16) & 2047 for _ in range(32)]
    multipliers = [1 + 2 * ((next_value() >> 16) & 7) for _ in range(8 if type_id == sd.IQ1_S else 16)]
    signs = [-1 if next_value() & 65536 else 1 for _ in range(8 if type_id == sd.IQ1_S else 32)]
    return iq1_block(type_id, d, indices, multipliers, signs)


def fixture(type_id, tied, moe, seed=12345, context=16, split=False):
    return block_fixture(tied, moe, type_id, partial(raw_block, type_id), seed, context, split)


def check_format():
    for type_id in (sd.IQ1_S, sd.IQ1_M):
        label = sd.type_name(type_id)
        blocks = [iq1_block(type_id, *case) for case in iq1_cases(type_id)]
        payload = b"".join(blocks)
        decoded = sd.pack_f32(sd.decode(type_id, payload, len(blocks) * 256))
        tensors = [("all_fields", [256, len(blocks)], type_id, payload),
                   ("multiple_blocks", [512, 3], type_id, b"".join(blocks[:6])),
                   ("empty_rows", [256, 0], type_id, b"")]
        with tempfile.TemporaryDirectory(prefix="llmx_iq1_format_") as directory:
            source, metadata, output = [Path(directory) / ("input." + ext) for ext in ("gguf", "json", "bin")]
            for fields, expected in ((tensors, decoded + decoded[:6 * 256 * 4]), ([], b"")):
                sd.write_gguf(source, {}, fields)
                before = hashlib.sha256(source.read_bytes()).hexdigest()
                rc, diagnostic = common.run(["dequantize", str(source), str(metadata), str(output)])
                assert rc == 0, label + " conversion failed: " + diagnostic
                assert output.read_bytes() == expected, label + " converted F32 bytes or zero signs differ"
                assert json.loads(metadata.read_text(encoding="utf-8"))["tensors"] == [
                    {"name": name, "shape": shape} for name, shape, _, _ in fields], label + " converted extents differ"
                assert hashlib.sha256(source.read_bytes()).hexdigest() == before, label + " source changed"
        print("iq1: %s CLI conversion, %d field values, multiple blocks and empty rows/model [ok]" % (label, len(blocks) * 256))
    return True


def run():
    check_format()
    results = [run_family("iq1", label, partial(fixture, type_id), family=family)
               for type_id, label, family in ((sd.IQ1_S, "IQ1_S", "iq1-s"), (sd.IQ1_M, "IQ1_M", "iq1-m"))]
    return common.SKIPPED if common.SKIPPED in results else all(results)
