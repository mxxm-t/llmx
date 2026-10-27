"""Raw NVFP4 matrices and the shared independent tiny-model HF checks."""

import math
import json
import os
import struct
import tempfile

import common
import spec_decode as sd
from block_fixture import VARIANTS, packed_hash, write_fixture
from block_fixture import fixture as block_fixture, run_family
from f32 import weight_hash

NVFP4_SEED = 12346
PROMPT_TEXTS = [("The quick brown fox jumps over the lazy dog. " * 4)[:n]
                for n in (63, 64, 65, 127, 128, 129)]


def raw_block(next_value, block):
    scales = [1 + (next_value() >> 24) % 7 for _ in range(4)]
    codes = [(next_value() >> 16) & 15 for _ in range(64)]
    if block % 257 in (0, 1):
        scales[block % 4] = (0, 127)[block % 257]
    return bytes(scales) + bytes(codes[g * 16 + j] | (codes[g * 16 + j + 8] << 4)
                                for g in range(4) for j in range(8))


def fixture(tied, moe, seed=NVFP4_SEED, context=16, split=False):
    return block_fixture(tied, moe, sd.NVFP4, raw_block, seed, context, split)


def _ue4m3(code):
    exponent, mantissa = code >> 3, code & 7
    return math.ldexp(8 + mantissa, exponent - 10) if exponent else math.ldexp(mantissa, -9)


# Only exact repacking is allowed: these tables contain no nearest-value rounding.
_SCALE_CODES = {_ue4m3(code): code for code in range(127)}
_REPACK = {factor: tuple(code if code in (0, 127) else _SCALE_CODES.get(_ue4m3(code) / factor)
                         for code in range(128)) for factor in (0.5, 0.25)}


def _check_effective(name, shape, data, gains, expected):
    """Decode independently of the repacker, bounding scratch to 256 blocks."""
    per_expert = shape[0] * shape[1]
    for first in range(0, len(expected), 256 * 64):
        # Split the chunk at an expert boundary so its scalar is unambiguous.
        end = min(first + 256 * 64, len(expected))
        at = first
        while at < end:
            expert = at // per_expert if len(gains) > 1 else 0
            stop = min(end, (expert + 1) * per_expert) if len(gains) > 1 else end
            raw = memoryview(data)[at // 64 * 36:stop // 64 * 36]
            if sd.np is None:
                decoded = sd.decode(sd.NVFP4, raw, stop - at)
                have = sd.pack_f32([sd.to_f32(value * gains[expert]) for value in decoded])
            else:
                decoded = sd.decode_numpy(sd.NVFP4, raw, stop - at)
                decoded *= sd.np.float32(gains[expert])
                have = decoded.tobytes()
            assert have == sd.pack_f32(expected[at:stop]), "NVFP4 repack changed effective F32 weights: " + name
            at = stop


def repack(weights, packed):
    """Equivalent NVFP4 bytes with positive scalar/expert companions; HF weights remain exact.
    Negative companions are checked elsewhere: a negative factor changes the sign of zero
    codes and therefore cannot give this strict byte-equivalent fixture transformation."""
    original = {name: values for name, _, _, values in weights}
    result = []
    for index, (name, shape, kind, payload) in enumerate(packed):
        assert kind == sd.NVFP4 and name.endswith(".weight"), "NVFP4 repack needs NVFP4 matrices"
        experts = shape[2] if len(shape) == 3 else 1
        gains = [0.5 if (index + expert) % 2 == 0 else 0.25 for expert in range(experts)]
        per_expert = shape[0] * shape[1]
        data = bytearray(payload)
        for block in range(len(data) // 36):
            table = _REPACK[gains[block * 64 // per_expert]]
            for group in range(4):
                code = data[block * 36 + group]
                replacement = table[code] if code < len(table) else None
                assert replacement is not None, "NVFP4 local scale cannot be repacked exactly"
                data[block * 36 + group] = replacement
        _check_effective(name, shape, data, gains, original[name])
        prefix = name[:-7]
        result += [(name, shape, kind, bytes(data)),
                   (prefix + ".scale", [experts], sd.F32, struct.pack("<%df" % experts, *gains)),
                   (prefix + ".input_scale", [experts], sd.F32,
                    struct.pack("<%df" % experts, *(3.0 + expert for expert in range(experts))))]
    return result


def run_prompts():
    split = common.tensor_width() > 1
    suffix = "_split" if split else ""
    path = os.path.join(os.path.dirname(__file__), "data", "baseline_nvfp4_prompt%s.json" % suffix)
    with open(path, encoding="utf-8") as stream:
        goldens = json.load(stream)
    assert goldens["transformers_version"] == "4.55.2", "NVFP4 prompt reference version changed"
    assert [g["name"] for g in goldens["fixtures"]] == [v[0] for v in VARIANTS], "NVFP4 prompt variants changed"
    comparisons = 0
    with tempfile.TemporaryDirectory(prefix="llmx_nvfp4_prompt_") as directory:
        for (name, tied, moe), golden in zip(VARIANTS, goldens["fixtures"]):
            config, weights, packed = fixture(tied, moe, context=256, split=split)
            assert config == golden["config"] and weight_hash(weights) == golden["weights_sha256"], "NVFP4 prompt weights changed"
            assert packed_hash(packed) == golden["packed_sha256"], "NVFP4 prompt blocks changed"
            assert [c["text"] for c in golden["cases"]] == PROMPT_TEXTS, "NVFP4 prompt lengths changed"
            assert [c["context"] for c in golden["perplexity"]] == [64, 129], "NVFP4 prompt windows changed"
            for scaled in (False, True):
                label = name + ("-scaled" if scaled else "")
                payload = repack(weights, packed) if scaled else packed
                model = write_fixture(os.path.join(directory, label + ".gguf"), config, weights, payload, moe)
                _, count = common.check_hf_fixture("NVFP4 prompt " + label, model, golden["cases"],
                    golden["perplexity"], PROMPT_TEXTS[-1], (1, 64, 129), family="nvfp4")
                comparisons += count
    assert comparisons == 216, "NVFP4 prompt comparison count changed"
    print("nvfp4-prompt: ordinary/scaled dense tied/untied and MoE, 216 full-logit cases and 24 NLL checks  [ok]")
    return True


def run():
    result = run_family("nvfp4", "NVFP4", fixture, family="nvfp4")
    if result is not True:
        return result
    result = run_family("nvfp4-scaled", "NVFP4", fixture, family="nvfp4", repack=repack)
    if result is not True:
        return result
    return run_prompts()
