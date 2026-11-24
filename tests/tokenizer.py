import io
import json
import os
import sys
import struct
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import run as cli, tokenize_failures

# Tokenizer round-trip gate: builds a minimal GGUF with a tiny GPT-2-style BPE vocab and merges, written directly since going through the CLI's quantize path is overkill, then exercises encode/decode incl. unicode and specials.
# It also requires a file naming another tokenizer or pretokenizer to be refused, and holds the qwen35, joyai-llm and minimax-m2 pretokenizers to HF's ids and the Unicode class table and NFC to unicodedata (docs/TESTS.md, Suite components).

ALIGN = 32
QWEN35_GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "baseline_tokenizer_qwen35.json")
# The goldens of tokenizers GGUF files name, each with the pretokenizer its files name (tools/gen_baseline.py, FILE_TOKENIZERS).
FILE_GOLDENS = {family: (os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "baseline_tokenizer_%s.json" % family), pre)
                for family, pre in (("deepseek41", "joyai-llm"), ("minimax", "minimax-m2"))}
UNICODE_CLASS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "core", "unicode_class.hpp")
# The code points the White_Space property holds, which has not changed since Unicode 6.3.
WHITE_SPACE = set(range(0x9, 0xE)) | {0x20, 0x85, 0xA0, 0x1680} | set(range(0x2000, 0x200B)) | {0x2028, 0x2029, 0x202F, 0x205F, 0x3000}
# Code points whose class group the running Python's Unicode version gives another way than 16.0 does, by version; between 15.0 and 16.0 no assigned code point moved.
UNICODE_MOVED = {"15.0.0": {}, "15.1.0": {}, "16.0.0": {}}


def w_str(f, s):
    b = s.encode()
    f.write(struct.pack("<Q", len(b)))
    f.write(b)


def build_byte_vocab():
    # Mirrors the C++ GPT-2 bytes_to_unicode() in tokenizer.hpp.
    # The C++ utf8::encode(cp) returns the UTF-8 bytes of codepoint cp, which as a Python string is simply chr(cp) (w_str then encodes it back to those bytes).
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


def build_tokenizer_gguf(path, tokens, merges, specials, strings=()):
    # Header with 0 tensors and 3 kv pairs, plus one per (key, value) in `strings`.
    with open(path, "wb") as f:
        f.write(struct.pack("<IIQQ", 0x46554747, 3, 0, 3 + len(strings)))  # magic, ver, 0 tensors, kv count

        for key, value in strings:
            w_str(f, key)
            f.write(struct.pack("<I", 8))  # STRING
            w_str(f, value)

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


def check_qwen35(d):
    """The qwen35 golden's texts through a file named as the Qwen3.5 files name their tokenizer, holding the part of the pinned HF vocabulary those texts reach: each must give HF's ids.
    The file also holds the control tokens only tokenizer_config.json adds, as the GGUF files do, and each one's text, alone and all side by side, must give its one id.
    The file lists the tokens in id order, so a token's position in it maps back to its id."""
    with io.open(QWEN35_GOLDEN, encoding="utf-8") as f:
        doc = json.load(f)
    vocab = dict(doc["tokens"], **doc["config_tokens"])
    ids = sorted(int(i) for i in vocab)
    path = build_tokenizer_gguf(os.path.join(d, "qwen35.gguf"), [vocab[str(i)] for i in ids], doc["merges"],
                                [(p, doc["token_types"][str(i)]) for p, i in enumerate(ids) if str(i) in doc["token_types"]],
                                [("tokenizer.ggml.model", "gpt2"), ("tokenizer.ggml.pre", "qwen35")])
    config = [{"text": text, "ids": [int(i)]} for i, text in doc["config_tokens"].items()]
    config.append({"text": "".join(case["text"] for case in config), "ids": [case["ids"][0] for case in config]})
    for cases, what in [(doc["cases"], "texts differ from HF"), (config, "control-token texts differ from their ids")]:
        failures = tokenize_failures(path, cases, ids)
        assert not failures, "qwen35: %d/%d %s:" % (len(failures), len(cases), what) + \
            "".join("\n  %s: want %s, got %s" % f for f in failures)
    return len(doc["cases"]), len(doc["config_tokens"])


def check_unicode_class():
    """The table's classes against the running Python's unicodedata: each code point that version assigns must have the bits of its general category's group and, for a cased letter, its case, but for the listed moves, and White_Space must be the property's code points.
    Code points only 16.0 assigns are left out, since unicodedata cannot name them."""
    import re
    import unicodedata
    version = unicodedata.unidata_version
    assert version in UNICODE_MOVED, "unicodedata is at Unicode %s, which the table check does not list" % version
    with io.open(UNICODE_CLASS, encoding="ascii") as f:
        runs = [(int(a, 16), int(b, 16), int(c)) for a, b, c in re.findall(r"\{0x([0-9a-f]+), 0x([0-9a-f]+), (\d+)\}", f.read())]
    groups = {"L": 1, "M": 2, "N": 4, "P": 8, "S": 16}
    cases = {"Lu": 64, "Lt": 64, "Ll": 128}
    bits = {}
    for first, last, value in runs:
        assert first <= last and value and (not bits or first > max(bits)), "unicode_class.hpp: run %x-%x out of order" % (first, last)
        for cp in range(first, last + 1):
            bits[cp] = value
    space = {cp for cp, value in bits.items() if value & 32}
    assert space == WHITE_SPACE, "unicode_class.hpp: White_Space differs at %s" % sorted(space ^ WHITE_SPACE)
    moved = UNICODE_MOVED[version]
    wrong = []
    for cp in range(0x110000):
        category = unicodedata.category(chr(cp))
        if category == "Cn":
            continue
        want = moved.get(cp, groups.get(category[0], 0) | cases.get(category, 0))
        if bits.get(cp, 0) & ~32 != want:
            wrong.append("U+%04X %s" % (cp, category))
    assert not wrong, "unicode_class.hpp: %d code points differ from Unicode %s, first %s" % (len(wrong), version, wrong[:5])
    return version, len(runs)


def nfc_items():
    """Texts whose NFC differs from them or could: every code point the running Python's unicodedata assigns with a canonical decomposition, a combining class or a quick check other than Yes, alone and decomposed, every Hangul syllable decomposed, and seeded runs of combining marks out of order after a base."""
    import random
    import unicodedata
    items, marks = [], []
    for cp in range(0x110000):
        c = chr(cp)
        if unicodedata.category(c) in ("Cn", "Cs"):
            continue
        if unicodedata.combining(c):
            marks.append(c)
        decomposition = unicodedata.decomposition(c)
        if (decomposition and not decomposition.startswith("<")) or unicodedata.combining(c) or not unicodedata.is_normalized("NFC", c):
            items += [c, unicodedata.normalize("NFD", c)]
    items += [unicodedata.normalize("NFD", chr(cp)) for cp in range(0xAC00, 0xAC00 + 11172)]
    rng = random.Random(1234)
    bases = "aeoAOUnsz\u0915\u0b47\u1100\uac00"
    items += [rng.choice(bases) + "".join(rng.choice(marks) for _ in range(rng.randint(1, 4))) for _ in range(3000)]
    return items


def check_nfc(d, tokens):
    """NFC as the qwen2 and qwen35 pretokenizers apply it, against the running Python's unicodedata: a file holding the bytes alone gives back each text's bytes, so its ids are the normalized text.
    Code points only 16.0 assigns are left out, as in the class check, and joyai-llm, whose tokenizer.json names no normalizer, must keep a decomposed accent as it is."""
    import unicodedata
    path = build_tokenizer_gguf(os.path.join(d, "nfc.gguf"), tokens[:256], [], [], [("tokenizer.ggml.pre", "qwen2")])
    items, chunks, chunk = nfc_items(), [], []
    for item in items:
        chunk.append(item)
        if sum(len(x) + 1 for x in chunk) > 4000:
            chunks.append(chunk)
            chunk = []
    chunks.append(chunk)

    def tokenized(model, text):
        rc, out = cli(["tokenize", model, text])
        assert rc == 0, "nfc: tokenize failed: %s" % out.strip()[:200]
        return bytes(int(i) for i in out.replace(",", " ").split())

    def nfc(text):
        return unicodedata.normalize("NFC", text).encode("utf-8")

    wrong = []
    for chunk in chunks:
        if tokenized(path, " ".join(chunk)) != nfc(" ".join(chunk)):
            wrong += [x for x in chunk if tokenized(path, x) != nfc(x)]
            if len(wrong) >= 5:
                break
    assert not wrong, "nfc: texts differ from unicodedata %s, first %s" % (unicodedata.unidata_version,
                                                                       [x.encode("unicode_escape").decode("ascii") for x in wrong[:5]])
    joyai = build_tokenizer_gguf(os.path.join(d, "nfc_joyai.gguf"), tokens[:256], [], [], [("tokenizer.ggml.pre", "joyai-llm")])
    assert tokenized(joyai, "cafe\u0301") == "cafe\u0301".encode("utf-8"), "nfc: joyai-llm normalized a decomposed accent"
    return len(items)


def check_file_tokenizer(d, family):
    """A family's golden texts through a file naming its pretokenizer, holding the part of the pinned vocabulary those texts reach and the GGUF file's token types: each must give HF's ids."""
    golden, pre = FILE_GOLDENS[family]
    with io.open(golden, encoding="utf-8") as f:
        doc = json.load(f)
    ids = sorted(int(i) for i in doc["tokens"])
    path = build_tokenizer_gguf(os.path.join(d, family + ".gguf"), [doc["tokens"][str(i)] for i in ids], doc["merges"],
                                [(p, doc["token_types"][str(i)]) for p, i in enumerate(ids) if str(i) in doc["token_types"]],
                                [("tokenizer.ggml.model", "gpt2"), ("tokenizer.ggml.pre", pre)])
    failures = tokenize_failures(path, doc["cases"], ids)
    pieces = {case["text"].encode("unicode_escape").decode("ascii"): case.get("pieces") for case in doc["cases"]}
    assert not failures, "%s: %d/%d texts differ from HF:" % (family, len(failures), len(doc["cases"])) + \
        "".join("\n  %s: want %s, got %s, HF pieces %s" % (f + (pieces[f[0]],)) for f in failures)
    return len(doc["cases"])


def run():
    d = tempfile.mkdtemp(prefix="llmx_tok_")
    try:
        # Tiny vocab: ascii bytes 33..126 mapped via the GPT-2 byte scheme, plus a couple of merged words and a special control token.
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

        # The tokenizer and pretokenizer the file names: the implemented ones give the same ids as a file naming neither, and any other is refused, since encoding would still succeed with wrong ids.
        named = os.path.join(d, "named.gguf")
        build_tokenizer_gguf(named, tokens, merges, specials,
                             [("tokenizer.ggml.model", "gpt2"), ("tokenizer.ggml.pre", "qwen2")])
        rc, out7 = cli(["tokenize", named, "hello"])
        assert rc == 0 and out7 == out, "named gpt2/qwen2 tokenizer: %r, unnamed %r" % (out7, out)
        # The refusal names the key and every implemented value.
        for key, value, implemented in [("tokenizer.ggml.pre", "gpt-2", ("qwen2", "qwen35", "joyai-llm", "minimax-m2")), ("tokenizer.ggml.pre", "qwen3", ("qwen2", "qwen35", "joyai-llm", "minimax-m2")),
                                        ("tokenizer.ggml.model", "llama", ("gpt2",))]:
            other = os.path.join(d, "other.gguf")
            build_tokenizer_gguf(other, tokens, merges, specials, [(key, value)])
            rc, out8 = cli(["tokenize", other, "hello"])
            assert rc != 0 and key in out8 and all("'%s'" % name in out8 for name in implemented), \
                "%s = %s was not refused naming %s: %r" % (key, value, implemented, out8)

        n, controls = check_qwen35(d)
        nfc = check_nfc(d, tokens)
        version, runs = check_unicode_class()
        n41 = check_file_tokenizer(d, "deepseek41")
        nmm = check_file_tokenizer(d, "minimax")
        print("tokenizer: encode->decode round-trip 'hello' -> %s, unicode and special token, other tokenizers refused, "
              "qwen35 %d/%d texts match HF and its %d file-only control tokens are one id each, "
              "the Unicode class table's %d runs match Unicode %s, NFC of %d texts matches it, deepseek41 %d/%d and minimax %d/%d texts match HF  [ok]" % (out2.strip(), n, n, controls, runs, version, nfc, n41, n41, nmm, nmm))
        return True
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
