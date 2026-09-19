# llmx — Test assets

Where the real models and corpora used for manual verification live. These are
environment-specific paths (this dev machine); the automated suite
(`tests/run_tests.py`) generates its own synthetic fixtures and needs none of
them.

> Superseded once `llmx pull` lands (`docs/ROADMAP.md` #9a): the hardcoded paths
> below become a cache the tool manages. Until then, this file is the record of
> what is on this machine.

## Model locations

Models are kept in the LM Studio model directory:

```
C:\Users\Marko\.lmstudio\models\
```

llmx reads **Q8_0 / Q4_0 / Q4_1 / Q4_K / Q5_K / Q6_K / F32** tensors (see
`docs/src/format-gguf.md`). Dense Qwen3 Q4_K_M and Q5_K_M mixtures are supported;
support for their tensor encodings does not add new architectures. F32 norms
work, but F32 embeddings/matrix operations remain an inference gap.

| Model                                            | Format | Status                       |
|--------------------------------------------------|--------|------------------------------|
| `Qwen\Qwen3-8B-GGUF\Qwen3-8B-Q8_0.gguf` (8.11 GB)| Q8_0   | **Usable** — the real-model gate |
| `Qwen\Qwen2-0.5B-Instruct-GGUF\...fp16.gguf`     | FP16   | Not yet supported            |
| `lmstudio-community\...\Qwen3-30B...Q4_K_M.gguf` | Q4_K_M | Quant supported; architecture not validated (MoE is unsupported) |
| `lmstudio-community\...\Qwen3-Coder...Q4_K_M.gguf`| Q4_K_M | Quant supported; architecture not validated (MoE is unsupported) |
| `unsloth\...\Qwen3.5-4B-BF16.gguf`               | BF16   | Not yet supported            |
| `unsloth\...\mmproj-F32.gguf`                    | F32    | Multimodal projector (not a main model) |

These assets exercise both tensor-format coverage and architecture support
(`docs/ROADMAP.md` #1-3). Check both before selecting a validation model.

Use the Qwen3-8B Q8_0 model for the manual real-model checks that the suite
can't cover — e.g. the lossless correctness gate (path-controlled perplexity)
and real throughput:

```
llmx.exe generate C:\Users\Marko\.lmstudio\models\Qwen\Qwen3-8B-GGUF\Qwen3-8B-Q8_0.gguf "The capital of France is" -n 32
```

## Gate fixture models (HF cache)

`tests/baseline.py` looks these up in the Hugging Face cache automatically, and
skips if they are absent. `LLMX_BASELINE_GGUF` overrides the lookup.
Keep the fixture's original filename when using the override: it selects the
quantization-specific logit/PPL bounds. Only that model's numerical checks run
when an override is set; unsupported filenames are rejected.

| Repo / file | Why this one |
|---|---|
| `Qwen/Qwen3-0.6B-GGUF` / `Qwen3-0.6B-Q8_0.gguf` | Small enough to gate on, and the tokenizer golden's model |
| `unsloth/Qwen3-0.6B-GGUF` / `Qwen3-0.6B-Q4_0.gguf` | **Load-bearing.** Mixed Q4_0/Q4_1/Q6_K/F32, and its Q6_K `token_embd` has a subnormal super-block scale. The Q8_0 fixture has almost no subnormal scales (0.0061% of blocks against 5.89% in Qwen3-8B), so without this model the logit gate is blind to the f16 subnormal bug class - it passed with that bug deliberately reintroduced until this was added. |

Fetch and SHA-256 verify the pinned snapshots with
`python tools/fetch_test_models.py` (Python standard library only, about 1 GB
combined). Revisions and digests are recorded in `tests/baseline.py`; the
numerical checks use those exact snapshots unless explicitly overridden.

### Fixed-excerpt HF perplexity gate

`tests/data/baseline_perplexity.json` records an HF float32 reference from
`Qwen/Qwen3-0.6B` revision `c1899de289a04d12100db370d81485cdf75e47ca`.
Regenerate it with `python tools/gen_baseline.py perplexity` in the isolated
HF environment described by that script. Generation uses CPU eager attention;
the JSON records torch/transformers versions, the exact text and its SHA-256,
token IDs, target count, mean NLL and PPL.

The text is the first 1024 Unicode characters of `wiki.test.raw`, with line
endings normalized to LF before extraction. Its 247 tokens form one continuous
sequence with no added BOS/EOS. Every token after the first is scored against
the preceding tokens (246 targets); log-softmax and the reduction use float64
on the HF float32 logits. The suite writes the stored text bytes to a temporary
file and invokes `perplexity --file`, so checkout newline settings do not change
the test input.

The HF reference is mean NLL **3.360285580**, PPL **28.797413678**. Two repeated
llmx measurements per quant gave Q8_0 NLL **3.36166** / PPL **28.8371** and mixed
Q4_0 NLL **3.49184** / PPL **32.8463**. The absolute mean-NLL bounds are **0.01**
and **0.16**, respectively (about 1.01% and 17.35% relative PPL). These bounds
allow quantization error; they do not establish lossless inference. The gate
also requires exact HF token IDs/count and finite, mutually consistent NLL/PPL.

The same pinned reference also scores disjoint 64-token windows (all four, or
the first two) and 123-token windows (two, omitting the singleton tail).
Positions/KV reset each window, and NLL is weighted by scored targets.

| Context / limit | HF NLL | Q8_0 NLL | Mixed Q4_0 NLL |
|---|---:|---:|---:|
| 64 / all | 4.030360346 | 4.037034329 | 4.197941852 |
| 64 / 2 | 3.710157365 | 3.712550543 | 3.779062509 |
| 123 / all | 3.630793905 | 3.643202014 | 3.751242730 |

Chunked mean-NLL bounds are **0.02** for Q8_0 and **0.20** for mixed Q4_0;
the tighter continuous bounds above remain unchanged. An independent control
running the previous scoring loop on the same token windows matched total
NLL exactly for both quants. This establishes unchanged window arithmetic,
not equivalence to the full-precision weights.

A diagnostic with two-token windows gave HF NLL **11.543536540**, Q8_0
**11.737387723**, Q4_0 **9.244354366**, identically under old/new scoring.
Those large quantized/full-precision differences are not covered by the
useful bounds above; the real-model gate uses contexts 64 and 123. Minimum
context handling is tested separately using mathematically known, nonuniform
probabilities in `tests/perplexity.py`. These short excerpts do not establish
full-corpus or long-context numerical correctness.


## Wiki text location

The wikitext corpus used for corpus-level perplexity is committed to the test
data directory:

```
tests\data\wiki.test.raw
```

This is the wikitext-2-raw test split (4358 articles, ~1.28 MB). It is a plain
text dump — the `@-@` split tokens are present, matching the wikitext corpus
format expected by the path-controlled perplexity gate in `docs/ROADMAP.md`.

`llmx.exe perplexity <model.gguf> --file tests/data/wiki.test.raw --ctx-size 512`
reads UTF-8 text and scores disjoint 512-token windows. Add `--chunks 4` to
evaluate only the first four windows. See `docs/USAGE.md` for target selection;
line endings are preserved, so use identical bytes and scoring policies for
both arms of a comparison.
