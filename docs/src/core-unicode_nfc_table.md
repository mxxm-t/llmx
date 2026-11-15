# `src/core/unicode_nfc_table.inc` - Unicode NFC tables

The Unicode 16.0.0 data `core/unicode_nfc.hpp` ([core-unicode_nfc.md](core-unicode_nfc.md)) reads, included inside its namespace, which declares the types.

- `kCombining`: 393 sorted runs of code points with the same nonzero canonical combining class, from `UnicodeData.txt`.
- `kDecompositions`: the 2081 canonical decompositions of `UnicodeData.txt` by code point, one or two code points each, the second 0 for a singleton; compatibility decompositions and the algorithmic Hangul syllables are left out.
- `kCompositions`: the 961 primary composites, each two-code-point decomposition whose code point `DerivedNormalizationProps.txt` does not list under `Full_Composition_Exclusion`, sorted by first then second half.
- `kNotQuick`: 119 sorted runs of code points whose `NFC_QC` is No or Maybe.

The file is written whole by `python tools/gen_unicode_table.py --ucd DIR`, with `core/unicode_class.hpp`, from files each refused unless it has the pinned SHA-256 ([ASSETS.md](../ASSETS.md#the-unicode-class-table)), and is not edited by hand.
