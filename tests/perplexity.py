import math
import os
import struct
import tempfile

from common import run as cli
from tokenizer import build_byte_vocab, w_str


def build_model(path):
    # Constant activations and a nonuniform output distribution give an analytic
    # scoring oracle; different target IDs expose miscounted window boundaries.
    config = {"block_count": 1, "embedding_length": 32, "feed_forward_length": 32,
              "attention.head_count": 1, "attention.head_count_kv": 1,
              "attention.key_length": 32, "context_length": 4}
    tensors = [("token_embd.weight", [32, 256], [1.0] * 8192),
               ("output.weight", [32, 256], [i / 1024 for i in range(256) for _ in range(32)]),
               ("output_norm.weight", [32], [1.0] * 32)]
    for name in ("attn_norm", "attn_q_norm", "attn_k_norm", "ffn_norm"):
        tensors.append(("blk.0." + name + ".weight", [32], [1.0] * 32))
    for name in ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"):
        tensors.append(("blk.0." + name + ".weight", [32, 32], [0.0] * 1024))
    payloads = []
    for _, shape, data in tensors:
        if len(shape) == 2:
            payloads.append(b"".join(struct.pack("<e", data[i]) + bytes([1]) * 32
                                     for i in range(0, len(data), 32)))
        else:
            payloads.append(struct.pack("<%df" % len(data), *data))
    with open(path, "wb") as f:
        f.write(struct.pack("<IIQQ", 0x46554747, 3, len(tensors), len(config) + 1))
        for key, value in config.items():
            w_str(f, "qwen3." + key)
            f.write(struct.pack("<II", 4, value))
        w_str(f, "tokenizer.ggml.tokens")
        f.write(struct.pack("<IIQ", 9, 8, 256))
        for token in build_byte_vocab():
            w_str(f, token)
        offset = 0
        for i, (name, shape, _) in enumerate(tensors):
            w_str(f, name)
            f.write(struct.pack("<I", len(shape)))
            f.write(struct.pack("<%dQ" % len(shape), *shape))
            f.write(struct.pack("<IQ", 8 if len(shape) == 2 else 0, offset))
            offset += len(payloads[i])
        f.write(b"\0" * (-f.tell() % 32))
        for data in payloads:
            f.write(data)


def run():
    with tempfile.TemporaryDirectory(prefix="llmx_ppl_boundaries_") as directory:
        model = os.path.join(directory, "analytic.gguf")
        build_model(model)
        logits = [i / 32 / math.sqrt(1 + 1e-6) for i in range(256)]
        logsum = math.log(sum(math.exp(x) for x in logits))
        cases = [("ab", 4, 0), ("abcd", 4, 0), ("abcde", 4, 0),
                 ("abcdef", 4, 0), ("abcdefgh", 4, 0), ("abcdefghi", 4, 0),
                 ("abcdefghi", 4, 1), ("abcdefghi", 3, 2), ("abcde", 2, 0)]
        for text, context, limit in cases:
            flags = ["--threads", "1"]
            if context != 4:
                flags += ["-c", str(context)]
            if limit:
                flags += ["--chunks", str(limit)]
            rc, out = cli(["perplexity", model, text] + flags)
            assert rc == 0, out
            fields = dict(line.split(":", 1) for line in out.splitlines())
            windows = [text[i:i + context] for i in range(0, len(text), context)]
            windows = [w for w in windows if len(w) >= 2][:limit or None]
            targets = [ord(t) for w in windows for t in w[1:]]
            expected = sum(logsum - logits[t] for t in targets) / len(targets)
            assert int(fields["tokens"]) == len(text)
            assert int(fields["used tokens"]) == sum(map(len, windows))
            assert int(fields["scored tokens"]) == len(targets)
            assert int(fields["chunks"]) == len(windows)
            assert int(fields["context size"]) == context
            assert abs(float(fields["mean NLL"]) - expected) < 1e-5
            assert math.isclose(float(fields["perplexity"]), math.exp(expected), rel_tol=1e-5)
        path = os.path.join(directory, "corpus \u00fc.txt")
        with open(path, "wb") as f:
            f.write(b"ab\r\ncd\nef")
        inline = cli(["perplexity", model, "ab\r\ncd\nef", "--threads", "1"])
        for flag in ("-f", "--file"):
            assert cli(["perplexity", model, flag, path, "--threads", "1"]) == inline
        for flag in ("-c", "--ctx-size", "--chunks"):
            for value in ("", "0", "-1", "x", "2x", "1.5", "2147483648", "9" * 40):
                assert cli(["perplexity", model, "abcd", flag, value])[0] != 0
            assert cli(["perplexity", model, "abcd", flag])[0] != 0
        for text, flags in (("", []), ("a", []), ("abcd", ["-c", "1"]),
                            ("abcd", ["-c", "5"]), ("abcd", ["--file", path])):
            assert cli(["perplexity", model, text] + flags)[0] != 0
        assert cli(["perplexity", model, "--file", path + ".missing"])[0] != 0
    print("perplexity: analytic NLL, window boundaries, file parity and invalid flags  [ok]")
    return True


if __name__ == "__main__":
    run()
