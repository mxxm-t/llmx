# `src/core/unicode_class.hpp` - Unicode code-point classes

The code-point classes a pretokenizer regex reads, generated from the Unicode 16.0.0 character database, in namespace `unicode_class`.
The `joyai-llm` pretokenizer of `tokenizer/tokenizer.hpp` ([tokenizer-tokenizer.md](tokenizer-tokenizer.md)) is its one user; the qwen2 and qwen35 pretokenizers keep their byte classes.

- `L`, `M`, `N`, `P`, `S`: one bit for each general-category group (letters, marks, numbers, punctuation, symbols), and `W` for the White_Space property, which is what the regex engine's `\s` holds.
- `kRanges`: 1586 sorted, disjoint runs of code points with the same bits; a code point in none has no class, which is every unassigned code point and every other category (controls but the White_Space ones, format characters, separators other than spaces, surrogates and private use).
- `of(cp)`: the bits of a code point, by binary search over the runs.

The file is written whole by `python tools/gen_unicode_table.py --ucd DIR` from `DerivedGeneralCategory.txt` and `PropList.txt`, each refused unless it has the pinned SHA-256 ([ASSETS.md](../ASSETS.md#the-unicode-class-table)), and is not edited by hand.
`tests/tokenizer.py` holds it to the running Python's `unicodedata`: every code point that version assigns must have its category group's bit, and White_Space must be the property's 25 code points.
