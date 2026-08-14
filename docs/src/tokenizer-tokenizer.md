# `src/tokenizer/tokenizer.hpp` — GPT-2 byte-level BPE

From-scratch GPT-2 style byte-level BPE tokenizer in namespace `bpe`.

- `utf8_encode` / `utf8_char_len`: code-point <-> UTF-8 helpers.
- `build_byte_encoder`: GPT-2 `bytes_to_unicode()` byte <-> printable char map.
- `SpecialToken`: named control/user-defined token.
- `Tokenizer`: reads tokenizer metadata from a GGUF model
  (`tokenizer.ggml.model/tokens/token_type/merges/bos/eos/pad_token_id`,
  `add_bos_token`).
  - `encode(text) -> vector<uint32_t>`: byte-encode, BPE merge by rank, map to ids.
  - `decode(ids) -> string`: reverse.
  - `token_id(s)`, `bpe(word)`, `byte_encode(raw)`.
  - Tracks `bos_id`, `eos_id`, `pad_id`, `add_bos`, special tokens.
