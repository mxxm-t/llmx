"""Write a GGUF whose matrices, the tied embedding included, are MXFP4, from one that holds them as BF16, F16 or F32.

    python -X utf8 tools/write_mxfp4.py --output Qwen3-0.6B-MXFP4.gguf [--source model.gguf]

The source defaults to the pinned Qwen3-0.6B BF16 file of tests/data/fixtures.json in the HF cache, which must match its SHA-256.
Every tensor of two dimensions is written as MXFP4, rows of 32-value blocks; every other tensor is copied byte for byte.
The metadata is copied in order without general.file_type, general.quantized_by and general.repo_url, which describe how the source was made.

Each block takes the scale rule of the OCP Microscaling Formats (MX) v1.0 specification: its shared exponent is floor(log2(amax)) - 2, amax being the block's largest magnitude and 2 the exponent of E2M1's largest value, 6.
The exponent is clamped at -127, the smallest E8M0 scale, which is also the exponent of a block of zeros, and is stored as the E8M0 byte exponent + 127.
Each value divided by 2^exponent rounds to the nearest E2M1 magnitude {0, 0.5, 1, 1.5, 2, 3, 4, 6}, a tie going to the one whose mantissa bit is 0, and saturates at 6.
Its sign bit is set only when it rounds to a nonzero magnitude, so -0 (code 8) is never written.
The file is written beside the output and replaces it only after tests/spec_decode.py has decoded every tensor to exactly the values the writer intended.
Needs numpy.
"""

import argparse
import hashlib
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from baseline_8b import file_sha256
import spec_decode as sd

np = sd.np

SOURCE_FILE = "Qwen3-0.6B-BF16.gguf"
DROPPED = ("general.file_type", "general.quantized_by", "general.repo_url")
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# Blocks encoded at once, which bounds the temporaries of a large tensor.
CHUNK = 1 << 16


def block_exponent(values):
    amax = max(abs(v) for v in values)
    return -127 if amax == 0 else max(math.frexp(amax)[1] - 1 - 2, -127)


def e2m1_index(a):
    """The index in E2M1 of the magnitude `a` rounds to: the nearest, a tie going to the even index, whose mantissa bit is 0, and 6 for anything past it."""
    for k in range(7):
        if a <= E2M1[k + 1]:
            middle = (E2M1[k] + E2M1[k + 1]) / 2
            return k if a < middle or a == middle and k % 2 == 0 else k + 1
    return 7


def encode_block(values):
    """The E8M0 byte, the 32 codes in value order and the 32 values they stand for, from 32 finite values; the readable form of the rule."""
    exponent = block_exponent(values)
    codes, intended = [], []
    for v in values:
        k = e2m1_index(abs(v) * 2.0 ** -exponent)
        codes.append(k | (8 if v < 0 and k else 0))
        intended.append(math.copysign(E2M1[k], v) * 2.0 ** exponent if k else 0.0)
    return exponent + 127, codes, intended


def encode_numpy(values):
    """The blocks, a (blocks, 17) uint8 array, and the values they stand for, (blocks, 32) float32, from a (blocks, 32) array of finite values; encode_block's rule for whole tensors."""
    x = values.astype(np.float64)
    amax = np.abs(x).max(axis=1)
    exponent = np.where(amax > 0, np.maximum(np.frexp(amax)[1] - 1 - 2, -127), -127).astype(np.int32)
    a = np.abs(x) * np.ldexp(1.0, -exponent)[:, None]
    # The thresholds are E2M1's midpoints; each tie belongs to the side whose index is even.
    k = np.select([a <= 0.25, a < 0.75, a <= 1.25, a < 1.75, a <= 2.5, a < 3.5, a <= 5.0], [0, 1, 2, 3, 4, 5, 6], 7).astype(np.uint8)
    codes = k | np.where((x < 0) & (k > 0), 8, 0).astype(np.uint8)
    magnitude = np.array(E2M1)[k] * np.ldexp(1.0, exponent)[:, None]
    intended = np.where(k > 0, np.where(x < 0, -magnitude, magnitude), 0.0).astype(np.float32)
    blocks = np.empty((len(values), 17), dtype=np.uint8)
    blocks[:, 0] = exponent + 127
    blocks[:, 1:] = codes[:, :16] | codes[:, 16:] << 4
    return blocks, intended


def convert(source, output, log=print):
    """Write `output` from the GGUF `source` by the rule above, check it, and return its SHA-256."""
    if np is None:
        raise SystemExit("write_mxfp4.py needs numpy")
    model = sd.GGUF(source)
    metadata = {key: value for key, value in model.metadata.items() if key not in DROPPED}
    tensors, digests, written = [], {}, 0
    for t in model.tensors:
        digest = hashlib.sha256()
        if len(t.shape) != 2:
            data = model.raw(t)
            digest.update(sd.decode_numpy(t.type, data, t.count).tobytes())
            tensors.append((t.name, t.shape, t.type, data))
        else:
            if t.type not in (sd.F32, sd.F16, sd.BF16) or t.shape[0] % 32:
                raise SystemExit("%s: a %s matrix %d wide is not written as MXFP4; the source must hold BF16, F16 or F32 rows of whole blocks"
                                 % (t.name, sd.type_name(t.type), t.shape[0]))
            values = model.decode(t).reshape(-1, 32)
            if not np.isfinite(values).all():
                raise SystemExit("%s holds a value that is not finite" % t.name)
            parts = []
            for start in range(0, len(values), CHUNK):
                blocks, intended = encode_numpy(values[start:start + CHUNK])
                parts.append(blocks.tobytes())
                digest.update(intended.tobytes())
            tensors.append((t.name, t.shape, sd.MXFP4, b"".join(parts)))
            written += 1
        digests[t.name] = digest.hexdigest()
    partial = output + ".partial"
    try:
        sd.write_gguf(partial, metadata, tensors)
        check = sd.GGUF(partial)
        assert [t.name for t in check.tensors] == list(digests), "the written file lists other tensors"
        for t in check.tensors:
            if hashlib.sha256(check.decode(t).tobytes()).hexdigest() != digests[t.name]:
                raise SystemExit("%s decodes to other values than the writer intended" % t.name)
        os.replace(partial, output)
    finally:
        if os.path.exists(partial):
            os.remove(partial)
    sha256 = file_sha256(Path(output))
    log("wrote %s: %d matrices in MXFP4, %d tensors copied, %d bytes, SHA-256 %s; every tensor decodes to the intended values"
        % (output, written, len(tensors) - written, os.path.getsize(output), sha256))
    return sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description="Write a GGUF with every matrix, the tied embedding included, in MXFP4 by the OCP scale rule, "
                                                 "round to nearest even and saturation at 6, and check it with the spec decoder.")
    parser.add_argument("--source", help="GGUF holding the matrices as BF16, F16 or F32 (default: the pinned %s in the HF cache, "
                                         "verified by its SHA-256)" % SOURCE_FILE)
    parser.add_argument("--output", required=True, help="GGUF to write; replaced only once it is written and checked")
    args = parser.parse_args(argv)
    source = args.source
    if not source:
        from baseline import pinned_fixture, snapshot_path
        spec = pinned_fixture(SOURCE_FILE)
        source = str(snapshot_path(spec["repo"], spec["revision"], spec["file"]))
        if not os.path.isfile(source):
            parser.error("the pinned source is not in the HF cache: %s; fetch it with tools/fetch_test_models.py --all" % source)
        print("verifying %s" % source, flush=True)
        if file_sha256(Path(source)) != spec["sha256"]:
            parser.error("%s does not match its pinned SHA-256" % source)
    convert(source, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
