# `src/tokenizer/hf_tokenizer.hpp` - Hugging Face tokenizer files to the GGUF tokenizer keys

`bpe::hf::read_keys(dir, config, n_vocab)` reads a Hugging Face directory's `tokenizer.json`, `tokenizer_config.json` and, when present, `chat_template.jinja`, and returns the `tokenizer.*` keys the GGUF converter writes for the same checkpoint, so `bpe::Tokenizer` ([tokenizer](tokenizer-tokenizer.md)) and the chat template read a directory as they read its GGUF file.
`config` is the directory's `config.json` and `n_vocab` the embedding's rows.

What it writes:
- `tokenizer.ggml.model` `gpt2` and `tokenizer.ggml.pre` `qwen2`.
- `tokenizer.ggml.tokens` and `tokenizer.ggml.token_type` (INT32), `n_vocab` long: the model's vocabulary as normal tokens (1), added tokens as control (3) when `special` or written `<|name|>`, else user-defined (4), and every id neither uses as `[PAD<id>]`, unused (5).
- `tokenizer.ggml.merges`, each `"a b"`, from either form a file uses (`"a b"` or `["a", "b"]`).
- `bos`, `eos`, `unknown` and `padding` token ids: the added token whose text `tokenizer_config.json` names, else the id `config.json` gives below `n_vocab`; a pad text that names no token, as gptqmodel writes `<unk>`, gives no id from the text, as HF's tokenizer gives none.
- `tokenizer.ggml.add_bos_token` and `add_eos_token` when `tokenizer_config.json` sets them.
- `tokenizer.chat_template`: `chat_template.jinja` when present, else `tokenizer_config.json`'s `chat_template`.

What it refuses, each by name: a key of either file it does not list; truncation or padding; a normalizer other than NFC, which the tokenizer applies for `qwen2`; a pre-tokenizer other than the qwen2 Split (`Isolated`, not inverted, the pattern byte for byte) followed by ByteLevel; a ByteLevel step, post-processor or decoder that adds a prefix space or uses its own regex; a post-processor other than ByteLevel or null; a model other than plain byte-level BPE (dropout, unknown token, prefixes, suffixes, byte fallback, ignored merges); a token id at or past `n_vocab` or given twice; an added token that takes another token's id, strips, matches single words or is normalized; a merge not of two non-empty parts without spaces; `add_bos_token`, `add_eos_token`, `add_prefix_space`, `clean_up_tokenization_spaces` or `split_special_tokens` set, which the tokenizer does not implement; a bos, eos or unknown token text that is not an added token; an `extra_special_tokens` that is not a map or a list, or a text in it that is not an added token; a list of named chat templates; an `added_tokens_decoder` entry that differs from `tokenizer.json`'s token.

The converter also makes control tokens of a few other forms, such as a name between `<` U+FF5C and U+FF5C `>` (fullwidth bars); this reader does not, so a file whose specials take such a form, none of the qwen2 files, would give type 4 where its GGUF file gives 3.

On Qwen/Qwen3-0.6B at c1899de the keys equal those of `Qwen/Qwen3-0.6B-GGUF`'s Q8_0 file but its older chat template, and the directory's tokenizer gives HF's ids on the 20 texts of `tests/data/baseline_tokenizer.json`.
`unsloth/Qwen3-0.6B-BF16.gguf` also differs in having no BOS id and a padding id of 151654, edits of that file.
