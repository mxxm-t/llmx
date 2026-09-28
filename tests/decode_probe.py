import json
import math
import os
import re
import struct
import subprocess
import tempfile

import common
import f32
import spec_decode


# llmx-decode-probe (tools/decode_probe.cpp) found beside the executable, on tiny models: the ranking it prints, a fixture it must refuse, and a vocabulary smaller than the five ids it lists.
# The fixtures' prompt and ids read as bytes, since the tiny models' vocabulary is one token per byte.
PROMPT = "abc"
# Each malformed fixture, and the entry its refusal must name.
MALFORMED = ((["ids", ["x"]], "ids[0]"), (["ids", [1, 1.5]], "ids[1]"), (["ids", [-1]], "ids[0]"),
             (["ids", [4294967296]], "ids[0]"), (["ids", [1e300]], "ids[0]"), (["ids", [2, True]], "ids[1]"),
             (["ids", [257]], "ids[0]"), (["tokens", [0, "y"]], "tokens[1]"), (["tokens", [2.5, 0]], "tokens[0]"),
             (["tokens", [0, 257]], "tokens[1]"))
# A model whose vocabulary holds four tokens, fewer than the five best ids the tool lists.
SMALL_VOCAB = ["a", "b", "c", "d"]


def tool_path():
    return os.path.join(os.path.dirname(common.exe_path()), "llmx-decode-probe" + (".exe" if os.name == "nt" else ""))


def device():
    """The configured device as the tool names one: a Vulkan index, or the CPU for the CPU and for a device list."""
    configured = os.environ.get("LLMX_DEVICE", "cpu")
    return configured.split(":", 1)[1] if configured.startswith("vulkan:") and "," not in configured else "cpu"


def probe(tool, model, directory, fixture):
    path = os.path.join(directory, "fixture.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fixture, f)
    return subprocess.run([tool, model, path, device()], capture_output=True, encoding="utf-8", errors="replace", timeout=120)


def best(out, forced, vocab):
    """The ids and logits of the tool's last step's best list, which must be min(5, vocab) distinct ids inside the vocabulary, the largest logit first."""
    line = re.search(r"^step %d best((?: \d+ \S+)+)$" % forced, out, re.M)
    assert line, out
    fields = line.group(1).split()
    ids, values = [int(i) for i in fields[0::2]], [float(v) for v in fields[1::2]]
    assert len(ids) == min(5, vocab) and len(set(ids)) == len(ids) and all(0 <= i < vocab for i in ids), out
    assert all(math.isfinite(v) for v in values) and values == sorted(values, reverse=True), out
    return ids


def write_small_vocab_model(path):
    """A one-layer qwen3 model 32 wide whose vocabulary is SMALL_VOCAB, its weights F32 from a fixed generator."""
    width, vocab = 32, len(SMALL_VOCAB)
    state = 97531

    def values(count, norm=False):
        nonlocal state
        out = []
        for _ in range(count):
            state = (1664525 * state + 1013904223) & 0xffffffff
            value = (((state >> 16) & 1023) - 512) / 8192
            out.append(1.0 + value if norm else value)
        return struct.pack("<%df" % count, *out)

    tensors = [("token_embd.weight", [width, vocab], spec_decode.F32, values(width * vocab)),
               ("output_norm.weight", [width], spec_decode.F32, values(width, True))]
    for norm in ("attn_norm", "ffn_norm", "attn_q_norm", "attn_k_norm"):
        tensors.append(("blk.0.%s.weight" % norm, [width], spec_decode.F32, values(width, True)))
    for matrix in ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"):
        tensors.append(("blk.0.%s.weight" % matrix, [width, width], spec_decode.F32, values(width * width)))
    tensors.append(("output.weight", [width, vocab], spec_decode.F32, values(width * vocab)))
    config = {"block_count": 1, "embedding_length": width, "feed_forward_length": width, "attention.head_count": 1,
              "attention.head_count_kv": 1, "attention.key_length": width, "context_length": 16}
    metadata = {"general.architecture": (spec_decode.STRING, "qwen3")}
    metadata.update(("qwen3." + key, (4, value)) for key, value in config.items())
    metadata["tokenizer.ggml.tokens"] = (spec_decode.ARRAY, (spec_decode.STRING, SMALL_VOCAB))
    spec_decode.write_gguf(path, metadata, tensors)
    return path


def run(require=False):
    tool = tool_path()
    if not os.path.exists(tool):
        assert not require, "decode-probe: %s not found beside the executable" % tool
        print("decode-probe: SKIP - %s not found beside the executable" % tool)
        return common.SKIPPED
    with tempfile.TemporaryDirectory(prefix="llmx_probe_") as directory:
        model = f32.write_model(os.path.join(directory, "tiny-f32.gguf"), f32.tensors(False))
        # The greedy path two steps long, read from the tool itself: each step's best id is the next step's forced id.
        forced = []
        for step in range(3):
            p = probe(tool, model, directory, {"prompt": PROMPT, "ids": forced, "tokens": [0, 1]})
            assert p.returncode == 0 and "every forced id is its step's greedy token" in p.stdout, p.stdout + p.stderr
            forced.append(best(p.stdout, step, f32.VOCAB)[0])
        # A forced id off the greedy path is reported, exiting 1.
        wrong = (forced[1] + 1) % f32.VOCAB
        p = probe(tool, model, directory, {"prompt": PROMPT, "ids": [forced[0], wrong], "tokens": [0, 1]})
        assert p.returncode == 1 and "the forced ids left the greedy path" in p.stdout, p.stdout + p.stderr
        # A malformed fixture is refused, naming the entry, before any step runs.
        for (key, value), where in MALFORMED:
            fixture = {"prompt": PROMPT, "ids": [forced[0]], "tokens": [0, 1]}
            fixture[key] = value
            p = probe(tool, model, directory, fixture)
            assert p.returncode == 2 and where in p.stderr and not p.stdout.strip(), \
                "decode-probe took %s = %r: exit %d\n%s%s" % (key, value, p.returncode, p.stdout, p.stderr)
        # A vocabulary of four lists four ids.
        small = write_small_vocab_model(os.path.join(directory, "tiny-vocab4.gguf"))
        p = probe(tool, small, directory, {"prompt": "ab", "ids": [2, 3], "tokens": [0, 3]})
        assert p.returncode in (0, 1), p.stdout + p.stderr
        best(p.stdout, 2, len(SMALL_VOCAB))
    print("decode-probe: the greedy path and a step off it, %d malformed fixtures refused by entry, and a vocabulary of %d  [ok]"
          % (len(MALFORMED), len(SMALL_VOCAB)))
    return True


if __name__ == "__main__":
    run()
