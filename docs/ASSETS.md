# llmx — Test assets

Where the real models and corpora used for manual verification live. These are
environment-specific paths (this dev machine); the automated suite
(`tests/run_tests.py`) generates its own synthetic fixtures and needs none of
them.

## Model locations

Models are kept in the LM Studio model directory:

```
C:\Users\Marko\.lmstudio\models\
```

llmx currently loads **Q8_0 / F32** GGUF only (see `docs/src/format-gguf.md`).
Of the models on this machine, exactly one is usable today:

| Model                                            | Format | Status                       |
|--------------------------------------------------|--------|------------------------------|
| `Qwen\Qwen3-8B-GGUF\Qwen3-8B-Q8_0.gguf` (8.11 GB)| Q8_0   | **Usable** — the real-model gate |
| `Qwen\Qwen2-0.5B-Instruct-GGUF\...fp16.gguf`     | FP16   | Not yet supported            |
| `lmstudio-community\...\Qwen3-30B...Q4_K_M.gguf` | Q4_K_M | Not yet supported            |
| `lmstudio-community\...\Qwen3-Coder...Q4_K_M.gguf`| Q4_K_M | Not yet supported            |
| `unsloth\...\Qwen3.5-4B-BF16.gguf`               | BF16   | Not yet supported            |
| `unsloth\...\mmproj-F32.gguf`                    | F32    | Multimodal projector (not a main model) |

The Q4_K_M / FP16 / BF16 models are relevant to `docs/ROADMAP.md` #1 (more quant
formats): once those block types are supported, the same directory provides real
models to validate them against.

Use the Qwen3-8B Q8_0 model for the manual real-model checks that the suite
can't cover — e.g. the lossless correctness gate (path-controlled perplexity)
and real throughput:

```
llmx.exe generate C:\Users\Marko\.lmstudio\models\Qwen\Qwen3-8B-GGUF\Qwen3-8B-Q8_0.gguf "The capital of France is" -n 32
```

## Wiki text location

The wikitext corpus used for corpus-level perplexity is committed to the test
data directory:

```
tests\data\wiki.test.raw
```

This is the wikitext-2-raw test split (4358 articles, ~1.28 MB). It is a plain
text dump — the `@-@` split tokens are present, matching the wikitext corpus
format expected by the path-controlled perplexity gate in `docs/ROADMAP.md`.

> **Limitation:** `llmx.exe perplexity` currently takes the text as a single
> argv element, so a ~1.28 MB file can't be passed on the command line (arg
> length + whitespace). A `--file <path>` option to read the corpus from disk is
> the intended follow-up; until then, run perplexity on small excerpts only.
