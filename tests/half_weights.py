import json
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile

import common
import f32
import moe
import spec_decode as sd
from tokenizer import build_byte_vocab


# Compare storage formats through full binary32 model captures, not the CLI's six-decimal logit display.
# Matrix values are exactly representable in both half formats; norms keep their original F32 values.
IDS = list(b"abcdefghijklm")


def shared_weights(weights):
    result = []
    for name, hf, shape, values in weights:
        if len(shape) > 1:
            values = [struct.unpack("<f", struct.pack("<I", struct.unpack("<I", struct.pack("<f", x))[0] & 0xffff0000))[0]
                      for x in values]
        result.append((name, hf, shape, values))
    return result


def write_variant(path, weights, config, arch, variant):
    floats, packed = [], []
    for name, hf, shape, values in weights:
        kind = sd.F32
        if len(shape) > 1:
            if variant == "mixed":
                layer = re.match(r"blk\.(\d+)\.", name)
                # A routed layer's gate and up keep one format, as the Vulkan contract requires.
                kind = (sd.F16, sd.BF16, sd.F32)[int(layer[1]) % 3] if layer else (sd.BF16 if name == "output.weight" else sd.F16)
            elif variant != "f32":
                kind = sd.F16 if variant == "f16" else sd.BF16
        if kind == sd.F32:
            floats.append((name, hf, shape, values))
        elif kind == sd.F16:
            packed.append((name, shape, kind, struct.pack("<%de" % len(values), *values)))
        else:
            words = [struct.unpack("<I", struct.pack("<f", x))[0] >> 16 for x in values]
            packed.append((name, shape, kind, struct.pack("<%dH" % len(words), *words)))
    # Verify the fixture bytes independently before executing either backend.
    originals = {name: values for name, _, _, values in weights}
    for name, _, kind, raw in packed:
        actual = sd.decode(kind, raw, len(originals[name]))
        assert struct.pack("<%df" % len(actual), *actual) == struct.pack("<%df" % len(actual), *originals[name]), name
    vocab = next(shape[1] for name, _, shape, _ in weights if name == "token_embd.weight")
    tokens = build_byte_vocab() + ["<|endoftext|>"] + ["<|reserved_%d|>" % i for i in range(vocab - 257)]
    return f32.write_model(str(path), floats, config=config, arch=arch, quantized=packed, tokens=tokens)


def capture(tool, model, ids, prefix, ubatch, width):
    device = os.environ.get("LLMX_DEVICE", "cpu")
    if width > 1:
        device = ",".join([device] * width)
    command = [str(tool), str(model), str(ids), str(prefix), device, "f32", str(ubatch), "0"]
    if os.environ.get("LLMX_LAYER_SHARES"):
        command.append(os.environ["LLMX_LAYER_SHARES"])
    elif width > 1:
        command.append("-")
    if width > 1:
        command.append(str(width))
    p = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    if common.device_lacks_kernel(p.returncode, p.stderr):
        return None
    assert p.returncode == 0, p.stdout + p.stderr
    doc = json.loads(p.stdout)
    assert doc["tokens"] == IDS and doc["vocab"] == f32.VOCAB + (width > 1) and len(doc["greedy"]) == 64, doc
    payloads = []
    for phase, rows in (("batched", len(IDS)), ("decode", len(IDS)), ("greedy", 64)):
        data = Path(str(prefix) + "." + phase + ".bin").read_bytes()
        assert len(data) == rows * doc["vocab"] * 4, (phase, len(data))
        payloads.append(data)
    return doc, payloads


def run(require=False):
    if common.f32_cache_skip("half-weights"):
        return common.SKIPPED
    for option in ("LLMX_DTYPE", "LLMX_LOAD_MODE"):
        if os.environ.get(option, "auto") != "auto":
            print("half-weights: SKIP - the capture tool uses auto activation and loading (%s=%s)" % (option, os.environ[option]))
            return common.SKIPPED
    tool = Path(common.exe_path()).with_name("llmx-model-logits" + (".exe" if os.name == "nt" else ""))
    if not tool.exists():
        assert not require, "half-weights: missing " + str(tool)
        print("half-weights: SKIP - missing " + str(tool))
        return common.SKIPPED
    checked, skipped = 0, set()
    with tempfile.TemporaryDirectory(prefix="llmx_half_weights_") as directory:
        root = Path(directory)
        ids = root / "prompt.ids"
        ids.write_text(" ".join(map(str, IDS)), encoding="ascii")
        fixtures = (("qwen3", f32.CONFIG, f32.tensors(False)), ("qwen3moe", moe.CONFIG, moe.tensors()))
        widths = (1, 2) if os.environ.get("LLMX_DEVICE", "cpu") == "cpu" else (1,)
        for arch, config, original, width in ((*fixture, width) for fixture in fixtures for width in widths):
            config = dict(config, context_length=128)
            if width > 1:
                config.update(embedding_length=40, feed_forward_length=20)
                if arch == "qwen3moe":
                    config["expert_feed_forward_length"] = 12
                    original = moe.tensors(config=config, vocab=258)
                else:
                    original = f32.tensors(False, config=config, vocab=258)
            for tied in (False, True):
                weights = shared_weights([w for w in original if not tied or w[0] != "output.weight"])
                tag = arch + ("-tied" if tied else "-head") + "-width" + str(width)
                models = {variant: write_variant(root / (tag + "-" + variant + ".gguf"), weights, config, arch, variant)
                          for variant in ("f32", "f16", "bf16", "mixed")}
                previous = None
                for ubatch in (3, 16):
                    reference = capture(tool, models["f32"], ids, root / "reference", ubatch, width)
                    assert reference is not None, "half-weights: F32 control was refused"
                    if previous is not None:
                        assert reference[0]["greedy"] == previous[0]["greedy"], (tag, "F32 greedy ids depend on ubatch")
                        assert reference[1] == previous[1], (tag, "F32 binary32 logits depend on ubatch")
                    previous = reference
                    for variant in ("f16", "bf16", "mixed"):
                        if variant in skipped or (variant == "mixed" and skipped):
                            continue
                        got = capture(tool, models[variant], ids, root / "candidate", ubatch, width)
                        if got is None:
                            skipped.add(variant)
                            print("half-weights: %s weight kernels unavailable on selected device, skipped" % variant)
                            continue
                        assert got[0]["dtype"] == reference[0]["dtype"], (tag, variant, "policy differs")
                        assert got[0]["greedy"] == reference[0]["greedy"], (tag, variant, ubatch, "greedy ids differ")
                        for phase, actual, expected in zip(("batched", "decode", "greedy"), got[1], reference[1]):
                            assert actual == expected, (tag, variant, ubatch, phase, "binary32 logits differ")
                        checked += 1
    if not checked:
        print("half-weights: SKIP - no half-weight model executed")
        return common.SKIPPED
    print("half-weights: %d exact full-vocabulary capture sets and 64-token replies; dense/MoE, tied/untied, F16/BF16/mixed, identical across ubatch 3/16; auto activation policy, CPU tensor widths 1/2 where selected, %d unavailable formats  [ok]"
          % (checked, len(skipped)))
    return True
