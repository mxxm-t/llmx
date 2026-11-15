# `src/core/unicode_nfc.hpp` - Unicode NFC over UTF-8

Normalization Form C of Unicode 16.0.0 (UAX #15), in namespace `unicode_nfc`, as the NFC normalizer of a Hugging Face `tokenizer.json` applies it.
The qwen2 and qwen35 pretokenizers of `tokenizer/tokenizer.hpp` ([tokenizer-tokenizer.md](tokenizer-tokenizer.md)) are its one user: `encode` normalizes the text between special tokens before the split, as HF does.

- `normalize(text)`: the text in NFC.
  One pass reads the text: ASCII bytes are skipped, and from the first other byte on each code point takes the quick check, so an all-ASCII or NFC-clean text costs that pass alone.
  The quick check passes when no code point is in `kNotQuick` and the nonzero combining classes never fall, and the text is then returned unchanged.
  Otherwise each run of valid UTF-8 is normalized, and a byte that starts no valid sequence (`utf8::valid_length`) is copied as it is and ends the run before it, since HF only ever sees valid text.
- `normalize_run(code points, out)`: full canonical decomposition, canonical ordering (a stable sort of each run of nonzero combining classes), then canonical composition, where a mark composes with the last starter unless a character between them has a class of 0 or at least its own.
- `decompose(cp, out)`, `compose(a, b)`, `ccc(cp)`: one step each, by binary search over the tables; Hangul syllables are decomposed and composed arithmetically, as section 3.12 of the Unicode standard gives.

The tables are `core/unicode_nfc_table.inc` ([core-unicode_nfc_table.md](core-unicode_nfc_table.md)), included inside the namespace after the four types they use.
`tests/tokenizer.py` holds the whole path to the running Python's `unicodedata.normalize("NFC")` through `llmx tokenize` on a file of the 256 byte tokens alone, whose ids are the normalized text's bytes ([TESTS.md](../TESTS.md), Suite components).
