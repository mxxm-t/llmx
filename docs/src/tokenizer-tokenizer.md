# `src/tokenizer/tokenizer.hpp` - byte-level BPE with Qwen2, Qwen3 and Qwen3.5 pretokenization

From-scratch byte-level BPE tokenizer in namespace `bpe`.
The byte encoding and merge algorithm are GPT-2's; the pretokenizer is not.

- UTF-8 comes from `core/utf8.hpp` ([core-utf8.md](core-utf8.md)): `utf8::encode` builds the byte map and `utf8::lead_length` steps through text and tokens.
  The lead byte's length is lenient, so `pretokenize`, `bpe` and `decode` move on over invalid input rather than refusing it.
- `build_byte_encoder`: GPT-2 `bytes_to_unicode()` byte <-> printable char map.
- `SpecialToken`: named control/user-defined token.
- `pretokenize(text)`: splits text before BPE, implementing the Qwen2/Qwen3 `Split` regex from `tokenizer.json` rather than GPT-2's.
  Alternatives are ordered and the first match wins.
  Three differences from GPT-2 change the resulting ids: a leading punctuation character or underscore binds to the word that follows (`_snake`, `(x`), digits are emitted one at a time, and a whitespace run ending in newlines stays one piece.
  Using the GPT-2 regex here produces valid-looking but wrong ids, which the HF tokenizer fixtures catch.
  The Qwen3.5 regex (pre `qwen35`) adds combining marks (`\p{M}`) to the letter runs of the second alternative and to the class the fourth excludes, which the byte classes below already give, so one routine serves both.
- `Tokenizer`: reads tokenizer metadata from a GGUF model (`tokenizer.ggml.tokens/token_type/merges/bos/eos_token_id`).
  - Refuses a file that names a `tokenizer.ggml.model` other than `gpt2` or a `tokenizer.ggml.pre` other than `qwen2` or `qwen35`, and the error names the key and the implemented values.
    Without the check, such a file would encode to valid-looking but wrong ids.
    A key the file omits is not checked, so the synthetic test models, which name neither, are read as this tokenizer.
  - `encode(text) -> vector<uint32_t>`: pretokenize, byte-encode, BPE merge by rank, map to ids.
  - `decode(ids) -> string`: reverse; an id outside the vocabulary throws.
  - `bpe(word)`, `byte_encode(raw)`.
  - Tracks `bos_id`, `eos_id` and the special tokens.
  - `is_eos(id)`: whether a token ends a generation, for `generate` and the server alike; a model without an EOS id has no stop token.
    `ignore_eos` masks the same id before sampling (`infer::sample`).

Character classes are approximated for UTF-8 without a Unicode table: any byte at or above 0x80 counts as a letter.
This matches Qwen's classes on the scripts the fixtures cover; it is not a general `\p{L}` implementation.

Known differences from HF's `tokenizer.json`, shared by Qwen3 and Qwen3.5 files unless noted:

- No NFC normalization: a decomposed accent (`e` followed by U+0301) encodes differently from HF, which composes it first.
- Non-ASCII characters that are not letters or marks read as letters, so llmx cuts the text differently around them, and the ids differ where a merge joins across a cut only one side makes:
  - a run of non-ASCII spaces, such as two U+3000 before a word, stays one run, where HF splits off all but the last space;
  - a non-ASCII number such as U+00BD (one half) joins the space before it, where HF emits it alone;
  - non-ASCII punctuation joins the letters around it, where HF groups it with neighbouring punctuation and following newlines.
    The common English case is a closing curly quote (U+201D) before a period or a comma, one HF token and two in llmx; U+00BB before a comma is the same, and U+3002 before two newlines is one token in Qwen3's vocabulary, which llmx cannot form;
  - non-ASCII symbols are read the same way, and HF groups them with the punctuation and newlines after them: U+00AE (registered sign) before a comma is one HF token and two in llmx in both vocabularies, as are U+00B0 (degree) and U+20AC (euro) before a period or a comma in Qwen3.5's.
- Qwen3 only: the qwen2 regex leaves combining marks out of letter runs, so HF splits Thai into more tokens than llmx gives (6 against 2 for the first Thai text of the qwen35 fixture); qwen35 keeps marks in letter runs, as llmx does.
- Qwen3.5 only: the GGUF files mark 7 tokens as control tokens that `tokenizer.json` lacks (ids 248070 to 248076: `<|audio_start|>`, `<|audio_end|>`, `<tts_pad>`, `<tts_text_bos>`, `<tts_text_eod>`, `<tts_text_bos_single>` and `<|audio_pad|>`).
  llmx encodes each as its one id, as transformers' tokenizer does from `tokenizer_config.json`, while `tokenizer.json` alone splits the text into several ordinary tokens.

`tests/baseline.py` compares encoded ids against the HF tokenizer for pinned real Qwen3 models.
`tests/tokenizer.py` does the same for the qwen35 pretokenizer on every build, from `tests/data/baseline_tokenizer_qwen35.json`: HF's ids for 37 texts (the 20 of the Qwen3 fixture, Thai, Devanagari, CJK punctuation and every added token) and the part of the pinned HF vocabulary those texts reach.
It also requires one id for each of the 7 control tokens above, which the golden keeps apart with the types the GGUF files give them.
These are the external gates for this file.
