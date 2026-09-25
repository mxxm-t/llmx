# `src/tokenizer/tokenizer.hpp` - byte-level BPE with Qwen2/Qwen3 pretokenization

From-scratch byte-level BPE tokenizer in namespace `bpe`. The byte encoding and
merge algorithm are GPT-2's; the pretokenizer is not.

- `utf8_encode` / `utf8_char_len`: code-point <-> UTF-8 helpers.
- `build_byte_encoder`: GPT-2 `bytes_to_unicode()` byte <-> printable char map.
- `SpecialToken`: named control/user-defined token.
- `pretokenize(text)`: splits text before BPE, implementing the Qwen2/Qwen3
  `Split` regex from `tokenizer.json` rather than GPT-2's. Alternatives are
  ordered and the first match wins. Three differences from GPT-2 change the
  resulting ids: a leading punctuation character or underscore binds to the
  word that follows (`_snake`, `(x`), digits are emitted one at a time, and a
  whitespace run ending in newlines stays one piece. Using the GPT-2 regex here
  produces valid-looking but wrong ids, which the HF tokenizer fixtures catch.
- `Tokenizer`: reads tokenizer metadata from a GGUF model
  (`tokenizer.ggml.model/tokens/token_type/merges/bos/eos_token_id`).
  - `encode(text) -> vector<uint32_t>`: pretokenize, byte-encode, BPE merge by
    rank, map to ids.
  - `decode(ids) -> string`: reverse; an id outside the vocabulary throws.
  - `bpe(word)`, `byte_encode(raw)`.
  - Tracks `bos_id`, `eos_id` and the special tokens.

Character classes are approximated for UTF-8 without a Unicode table: any byte
at or above 0x80 counts as a letter. This matches Qwen's classes on the scripts
the fixtures cover; it is not a general `\p{L}` implementation.

`tests/baseline.py` compares encoded ids against the HF tokenizer for pinned
real models, which is the external gate for this file.
