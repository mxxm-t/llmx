import os
import sys
import struct
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import run as cli

# Tokenizer round-trip gate. Builds a minimal GGUF with a tiny GPT-2-style BPE
# vocab + merges (via the CLI's own quantize path is overkill, so we write the
# GGUF directly), then exercises encode/decode incl. unicode and specials.

ALIGN = 32


def w_str(f, s):
    b = s.encode()
    f.write(struct.pack("<Q", len(b)))
    f.write(b)


def build_byte_vocab():
    # Mirrors the C++ GPT-2 bytes_to_unicode() in tokenizer.hpp. The C++
    # utf8_encode(cp) returns the UTF-8 bytes of codepoint cp, which as a Python
    # string is simply chr(cp) (w_str then encodes it back to those bytes).
    m = {}
    for b in range(33, 127):
        m[b] = chr(b)
    for b in range(161, 256):
        if b != 173:
            m[b] = chr(b)
    n = 0
    for b in range(256):
        if not ((33 <= b <= 126) or (161 <= b <= 255 and b != 173)):
            m[b] = chr(256 + n)
            n += 1
    return [m[b] for b in range(256)]


def build_tokenizer_gguf(path, tokens, merges, specials):
    # Header with 0 tensors, 3 kv pairs.
    with open(path, "wb") as f:
        f.write(struct.pack("<IIQQ", 0x46554747, 3, 0, 3))  # magic, ver, 0 tensors, 3 kv

        # tokenizer.ggml.tokens (array of string)
        w_str(f, "tokenizer.ggml.tokens")
        f.write(struct.pack("<I", 9))  # V_ARRAY
        f.write(struct.pack("<I", 8))  # element type = STRING
        f.write(struct.pack("<Q", len(tokens)))
        for t in tokens:
            w_str(f, t)

        # tokenizer.ggml.token_type (array of u32)
        types = [1] * len(tokens)
        for t, tt in specials:
            types[t] = tt
        w_str(f, "tokenizer.ggml.token_type")
        f.write(struct.pack("<I", 9))
        f.write(struct.pack("<I", 4))  # UINT32
        f.write(struct.pack("<Q", len(types)))
        for t in types:
            f.write(struct.pack("<I", t))

        # tokenizer.ggml.merges (array of string)
        w_str(f, "tokenizer.ggml.merges")
        f.write(struct.pack("<I", 9))
        f.write(struct.pack("<I", 8))
        f.write(struct.pack("<Q", len(merges)))
        for m in merges:
            w_str(f, m)

        # align to 32
        while f.tell() % ALIGN:
            f.write(b"\0")
    return path


def run():
    d = tempfile.mkdtemp(prefix="llmx_tok_")
    try:
        # Tiny vocab: ascii bytes 33..126 mapped via the GPT-2 byte scheme, plus
        # a couple of merged words and a special control token.
        tokens = build_byte_vocab() + ["he", "llo", "world", "<|endoftext|>"]
        bos_id, eos_id = 0, len(tokens) - 1

        # merges: "h e" -> "he", "he llo" -> "hello", "hello world" -> "helloworld"
        merges = ["h e", "he llo", "hello world"]
        specials = [(eos_id, 3)]  # GGUF token types: 3 = control

        gguf = os.path.join(d, "tok.gguf")
        build_tokenizer_gguf(gguf, tokens, merges, specials)

        # encode
        rc, out = cli(["tokenize", gguf, "hello"])
        assert rc == 0, "tokenize failed"
        ids = [int(x) for x in out.split(",") if x.strip()]
        # decode
        rc, out2 = cli(["detokenize", gguf, ",".join(str(i) for i in ids)])
        assert rc == 0, "detokenize failed"
        assert out2.strip() == "hello", "round-trip '%s' != 'hello'" % out2.strip()

        # unicode
        rc, out3 = cli(["tokenize", gguf, "h\u00e9llo"])
        assert rc == 0, "unicode tokenize failed"
        rc, out4 = cli(["detokenize", gguf, out3.strip()])
        assert rc == 0, "unicode detokenize failed"
        assert out4.strip() == "h\u00e9llo", \
            "unicode round-trip %r != %r" % (out4.strip(), "h\u00e9llo")

        # special token: encodes to a single id and decodes back unchanged
        rc, out5 = cli(["tokenize", gguf, "<|endoftext|>"])
        assert rc == 0, "special tokenize failed"
        special = [int(x) for x in out5.split(",") if x.strip()]
        assert len(special) == 1, "special token split into %d ids" % len(special)
        rc, out6 = cli(["detokenize", gguf, str(special[0])])
        assert rc == 0, "special detokenize failed"
        assert out6.strip() == "<|endoftext|>", \
            "special round-trip %r" % out6.strip()

        print("tokenizer: encode->decode round-trip 'hello' -> %s, unicode and "
              "special token  [ok]" % out2.strip())
        return True
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
