# `src/tokenizer/tokenizer.hpp` - byte-level BPE with Qwen2, Qwen3, Qwen3.5 and joyai-llm pretokenization

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
- `pretokenize_joyai(text)`: the `joyai-llm` pretokenizer (DeepSeek-V4.1's files), its `tokenizer.json`'s three `Isolated` splits in turn, each on the previous one's pieces: number runs of 1 to 3 (`\p{N}{1,3}`), runs of CJK ideographs U+4E00 to U+9FA5 and kana U+3040 to U+30FF, then a split whose alternatives are an ASCII punctuation character before ASCII letters, an optional character that is no newline, letter, punctuation or symbol before a run of letters and marks, an optional space before punctuation and symbols and their trailing newlines, and the three whitespace rules of the qwen2 regex.
  Each split's matches are pieces and so is each run of text between them, so a digit run of the first split stays whole through the third.
  The classes are Unicode 16.0's, read from `core/unicode_class.hpp` ([core-unicode_class.md](core-unicode_class.md)), and `\s` is White_Space; a byte that does not begin a valid UTF-8 sequence is read as one character of no class.
  The file's `tokenizer.ggml.pre` chooses it (`joyai`), and `encode` runs it in place of `pretokenize`, whose qwen2 and qwen35 path it does not touch.
- `Tokenizer`: reads tokenizer metadata from a GGUF model (`tokenizer.ggml.tokens/token_type/merges/bos/eos_token_id`).
  - Refuses a file that names a `tokenizer.ggml.model` other than `gpt2` or a `tokenizer.ggml.pre` other than `qwen2`, `qwen35` or `joyai-llm`, and the error names the key and the implemented values.
    Without the check, such a file would encode to valid-looking but wrong ids.
    A key the file omits is not checked, so the synthetic test models, which name neither, are read as this tokenizer.
  - Refuses a `tokenizer.ggml.tokens` array whose elements are not strings, and a `tokenizer.ggml.token_type` array with fewer entries than tokens; a file without `token_type` gives every token the normal type.
  - `encode(text) -> vector<uint32_t>`: the control and user-defined tokens' texts matched first, the longest at each place, each as its one id, and the text between them pretokenized, byte-encoded, BPE-merged by rank and mapped to ids; a symbol the vocabulary lacks throws.
    The specials are kept ordered by their first byte with `special_at` giving each byte's range, so a position compares only against the specials that begin with its byte, and a word merged once is not merged again within the call (a map local to it, so concurrent calls share nothing), since BPE gives a word the same ids wherever it stands.
    On a 159-message chat of 4.1 MB and 954025 ids the lead-byte filter and the memo took the encode from about 950 ms to about 100 ms on the Windows development machine, 40 percent of it having been the scan and 60 percent BPE run again on words already merged.
  - `decode(ids) -> string`: reverse; an id outside the vocabulary throws.
  - `bpe(word)`: the merge loop over a byte-mapped word, which merges the adjacent pair of lowest rank, the leftmost on a tie, until no pair has a rank.
    The symbols are spans of the word in a doubly linked list and the candidate pairs a min-heap keyed by rank and the left span's index; a merge pops the top, skips it when either span was merged since it was pushed (its recorded lengths no longer match), joins the spans and pushes the two new neighbour pairs.
    A merged span keeps its left half's index and the survivors keep their order, so the lowest index among equal ranks is the leftmost pair, as a scan chooses it, and the ids are the plain loop's.
    A word of n symbols takes O(n log n) with one `merge_rank` lookup per candidate; the scan it replaced rescanned every pair per merge, and a one-word text of 64 KiB, which a client of the server can send, took it 84 s on the Windows development machine, four times the 32 KiB word's 21 s, where the heap takes 1.1 to 1.6 s for 2 MiB.
  - `byte_encode(raw)`.
  - Tracks `bos_id`, `eos_id` and the special tokens.
  - `is_eos(id)`: whether a token ends a generation, for `generate` and the server alike; a model without an EOS id has no stop token.
    `ignore_eos` masks the same id before sampling (`infer::sample`).

In `pretokenize`, character classes are approximated for UTF-8 without a Unicode table: any byte at or above 0x80 counts as a letter.
This matches Qwen's classes on the scripts the fixtures cover; it is not a general `\p{L}` implementation.

Before the split, `encode` puts the text between special tokens in NFC (`unicode_nfc::normalize`, [core-unicode_nfc.md](core-unicode_nfc.md)) for `qwen2` and `qwen35`, whose `tokenizer.json` names the NFC normalizer, as HF does; `joyai-llm`, whose `tokenizer.json` names an empty normalizer sequence, keeps the text as it is.
Special tokens are matched before it, as HF matches its added tokens, which are not normalized.

Known differences from HF's `tokenizer.json`, shared by Qwen3 and Qwen3.5 files unless noted:

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
`tests/tokenizer_encode.cpp` (CTest `tokenizer-encode`) holds `encode` to the encode it replaced, written out in the test, on a small vocabulary with merges and special tokens, holds `bpe` to a plain lowest-rank-leftmost loop on random words over 4000 generated merges, bounds the time of two one-word texts of 2 MiB and checks the two load refusals above.
`tests/tokenizer.py` does the same for the qwen35 pretokenizer on every build, from `tests/data/baseline_tokenizer_qwen35.json`: HF's ids for 42 texts (the 25 of the Qwen3 fixture, five of them texts NFC changes, Thai, Devanagari, CJK punctuation and every added token) and the part of the pinned HF vocabulary those texts reach.
It also requires one id for each of the 7 control tokens above, which the golden keeps apart with the types the GGUF files give them.
It holds the joyai-llm pretokenizer to HF's ids from `tests/data/baseline_tokenizer_deepseek41.json` the same way: 103 texts (scripts, digits of several kinds, punctuation and symbols, emoji, Unicode-version and whitespace probes, every added token in groups and a 6000-character excerpt with a multilingual tail), with the token types of DeepSeek-V4.1's GGUF file.
These are the external gates for this file.
