# Image input

This page is the plan for image input, ROADMAP item 10, as F2DEV approved it as the plan of record on 2026-10-25.
The coordinator and F2DEV answered its open questions the same day, on the owner's delegation.
Nothing in `src/` reads an image yet; the sections below say what will, in the order it is built.
Audio follows on the same encoder contract once images are done.

## 1. The models and the weights (planned)

One vision encoder serves Qwen 3.5, 3.6, 3.8-27B and Qwen3.8-Flash-Next (`qwen4exp`).
- At transformers commit `93c8b7b` (v5.16.1, the commit [QWEN38-FLASH](QWEN38-FLASH.md) pins), `Qwen4ExpVisionModel` and `Qwen3_5VisionModel` are the same code but for one decorator, and `get_rope_index` is the same in both.
- Their configs agree: depth 27, width 1152, MLP 4304, 16 heads of 72, patch 16, spatial merge 2, temporal patch 2, 2304 learned positions (a 48 by 48 grid), `gelu_pytorch_tanh` in the blocks and no deepstack layers.
  Only the output width differs: 2560 for Flash-Next, 5120 for 3.8-27B and 2048 for 3.6-35B-A3B.
- Qwen3.5-0.8B has the same structure at depth 12, width 768, 12 heads of 64 and output 1024.

llmx reads the GGUF `mmproj` file that ships beside a text GGUF (`general.architecture` `clip`, `clip.projector_type` `qwen3vl_merger`).
HF safetensors are read only by the reference generator; HF directories wait for ROADMAP items 3 and 9b.

The Flash-Next file's tensors, in GGUF order (input dimension first):

| GGUF tensor | Shape | HF parameter |
|---|---|---|
| `v.patch_embd.weight`, `v.patch_embd.weight.1` | [16, 16, 3, 1152] F32 each | `model.visual.patch_embed.proj.weight` [1152, 3, 2, 16, 16], one tensor per temporal frame |
| `v.patch_embd.bias` | [1152] | `patch_embed.proj.bias` |
| `v.position_embd.weight` | [1152, 2304] F32 | `pos_embed.weight` |
| `v.blk.N.ln1`, `v.blk.N.ln2` (weight, bias) | [1152] | `blocks.N.norm1`, `blocks.N.norm2` |
| `v.blk.N.attn_qkv` | [1152, 3456] BF16, bias [3456] | `blocks.N.attn.qkv` |
| `v.blk.N.attn_out` | [1152, 1152] BF16, bias [1152] | `blocks.N.attn.proj` |
| `v.blk.N.ffn_up` | [1152, 4304] BF16, bias [4304] | `blocks.N.mlp.linear_fc1` |
| `v.blk.N.ffn_down` | [4304, 1152] BF16, bias [1152] | `blocks.N.mlp.linear_fc2` |
| `v.post_ln` (weight, bias) | [1152] | `merger.norm`, before the 2 by 2 merge |
| `mm.0` | [4608, 4608] BF16, bias [4608] | `merger.linear_fc1` |
| `mm.2` | [4608, 2560] BF16, bias [2560] | `merger.linear_fc2` |

The file carries neither the processor's pixel bounds nor the image token ids.
The bounds are the family's documented defaults, as the precision policy takes an architecture's default for a GGUF; the token ids are read from the text file's vocabulary by their texts.

The files, all from the `unsloth` repositories unless named:

| File | Size | For |
|---|---|---|
| Qwen3.5-0.8B `mmproj-BF16.gguf`, revision `6ab46149` | 207,346,528 B | the released-model gate, beside the pinned Q8_0 text file |
| Qwen3.8-Flash-Next `mmproj-BF16.gguf`, revision `38bb39ee` | 907,542,944 B | the target, beside production's text files |
| `ggml-org` Qwen3.8-Flash-Next `mmproj-...-Q8_0.gguf` | 616,703,104 B | not used in the first steps |

## 2. Preprocessing (planned)

It follows transformers' `load_image` and `Qwen2VLImageProcessor` (torchvision backend) at the pinned commit, and must give their bytes.
1. Decode PNG or JPEG.
2. Apply the EXIF orientation (`ImageOps.exif_transpose`).
3. Convert to RGB as Pillow's `convert("RGB")` does: alpha dropped without compositing, palettes expanded, gray replicated.
4. Choose the size with `smart_resize`: factor 32 (patch 16 times merge 2), at least 65536 and at most 16777216 pixels, Python's `round` (halves to even), and an aspect ratio past 200 refused.
5. Resize with torchvision's v2 `resize` on uint8, bicubic with antialiasing (the separable kernel with a = -0.5, its support widened by the downscale factor), rounded and clamped to uint8; an unchanged size is the identity.
6. Normalize in F32 as `(x - 127.5) / 127.5`, HF's mean 0.5 and std 0.5 fused with its 1/255.
7. Cut 16 by 16 patches in merge-block order (block row, block column, row in block, column in block), each 768 values in (channel, row, column) order; the temporal axis repeats the frame.

An image gives gh * gw / 4 tokens, gh and gw being its patch grid: 64 at least, 1024 for 1024 by 1024 pixels and 16384 at most.
`--image-max-pixels`, on the CLI and the server, is HF's `max_pixels` processor argument with HF's default, 16777216, so llmx's output equals HF's for any image a user gives; the fit charges the encoder's scratch at the bound, and an image whose tokens do not fit the context is refused by name.

The decoders are written here (`format/`), PNG with its inflate and JPEG with libjpeg-turbo's integer IDCT, fancy upsampling and color tables, since the reference decodes through Pillow's libjpeg-turbo.

## 3. The encoder (planned)

Per HF `Qwen4ExpVisionModel.forward`:
- Patch embedding: a matrix product of each 768-value patch row with frame 0's kernel, a product added with frame 1's, and the bias.
- Learned positions: the 48 by 48 grid resampled bilinearly with aligned corners to the patch grid, four taps a patch in merge-block order, the taps and weights a function of the grid computed on the host.
- 27 blocks, each a LayerNorm (weight, bias, eps 1e-6), the qkv product and bias, a 2D rope on q and k (the 36 frequencies of a 72-wide head, 18 by patch row and 18 by patch column, NeoX pairs, no norm), bidirectional attention within the image at scale 1/sqrt(72), the output product and bias added to the residual, then a LayerNorm, the first MLP product and bias, GELU in its tanh form, the second product and bias added to the residual.
- The merger: a LayerNorm over 1152, four consecutive patch rows read as one 4608-wide row, a product and bias, exact (erf) GELU, a product and bias, one row per image token at the text model's width.

The backend gains, each with a CPU implementation and a test against a reference written from the math:
- a row-broadcast bias add, a LayerNorm with weight and bias, and `gelu` with its form (tanh or erf) as an argument;
- a weighted gather of table rows, for the learned positions;
- `norm_rope_partial`'s rule extended to per-row position streams with a per-frequency stream map, and its rope taken without the norm, rather than a second rope;
- non-causal attention over one image's rows: the existing `attention` taken as a non-causal view at head widths 64 and 72, or a second kernel only on a stated reason it cannot, meeting the Vulkan head-width rule.

The encoder has no sequences and no caches, so it has a contract of its own, `model/encoder.hpp`, holding only what the image encoder reads, with its module at `model/arch/qwen3vl_vision.hpp` chosen in the registry by `clip` and `clip.projector_type`.
It runs on the device that holds the embedding, counted in that device's fit, one image a call, so an image's rows never depend on another image.

## 4. A prompt of token ids and encoder rows (planned)

- The Qwen templates render an image part as `<|vision_start|><|image_pad|><|vision_end|>` (ids 248053, 248056 and 248054 in the 3.5 and 3.8 vocabularies), and refuse an image in a system message.
- `chat::Message` carries content parts, and the renderer hands the template the parts.
- After tokenizing, each pad id becomes gh * gw / 4 pad ids in image order; the pad count must equal the images given, as HF checks.
- The history keeps the pad ids as tokens, which Flash-Next's per-layer embedding hashes as HF hashes them, and beside them media runs: the first row, the rows, the merged grid and the image's digest.
- The embedding part writes each encoder row over its pad row; Flash-Next spreads it into its four streams as any row.
- Positions are HF's `get_rope_index`: a text row is (p, p, p) with p counting up; an image starting at p puts the merged patch at (row, column) at (p, p + row, p + column), and p then grows by the larger of the merged grid's sides.
  A row's position is its index plus a delta that is a function of the media runs below it; with no image the delta is 0 and nothing changes.
- The sections are interleaved [11, 11, 10] over the 32 rotary frequencies, a constant the module declares beside the sections the GGUF records.
- Flash-Next's sparse attention ropes each pooled key group at its first token's position: once its pooled keys are computed once in the pass that completes a group, that pass's call carries the groups' positions, so this step builds on that change and stores no positions.

## 5. History and cache (planned)

- A history's identity is its token ids, its row classes and its media runs; a text-only history has none and keeps its bytes, its disk digests and the saved caches.
- An image's digest is SHA-256 over the encoder file's identity, the preprocessing settings, the grid and the resized pixels, so another `mmproj` reuses nothing.
- The disk tier's block digests hash each media run a block overlaps; the device donors, host entries, boundaries and jobs compare media runs beside the row classes, with one owner beside the row-class comparison, and a fork inside an image needs the whole image to match.
- A request keeps its resized pixels for its life and its encoder rows until its prompt is read; a resume that recomputes encodes again, to the same bits.
- A context cut never falls inside an image, and the lookup drafter skips media runs.

## 6. The routes (planned)

- `/v1/chat/completions` and `/v1/chat` take `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}`; a `detail` other than absent or `auto` is refused with the field named; `/v1/completions` stays text.
- Only `data:` URLs of `image/png` and `image/jpeg`, base64 decoded strictly and the format read from its first bytes; http, https and file URLs are refused.
- The limits: the body limit, 20 MiB decoded per image, 89,478,485 pixels, and the image's tokens inside the context; 16-bit PNG, CMYK, arithmetic-coded and 12-bit JPEG, GIF and WebP are refused by name.
- `/v1/tokenize` counts an image's tokens.
- The CLI takes `--mmproj FILE` on its model commands and `--image FILE` on `generate` and `logits` with `--chat`, so a served reply is held to the CLI's.

## 7. The references and gates (planned)

- The reference environment gains torchvision and Pillow, pinned (the coordinator's and F2DEV's decision of 2026-10-25): the Pillow wheel by version, Python tag and platform with the libjpeg-turbo and zlib versions it reports, recorded in each fixture's metadata; the generator refuses other versions; decoded bytes and arrays are committed so the suite needs neither package; no text reference is regenerated.
- The F32 paths are held to 2e-5 a logit and 1e-5 NLL on tiny random-weight fixtures of both families built by HF; device paths below F32 are held to a budget calibrated as [PRECISION](PRECISION.md) sets out.
- Qwen3.5-0.8B's `mmproj` beside its pinned Q8_0 file is checked against HF on the files' own weights in the hosted HF job; Flash-Next's against HF's vision model and end to end through the layered file reference, by hand.
- Each step that touches the runtime, the renderer or the cache keeps text-only `generate`, `logits`, `perplexity`, `chat`, `tools/server_mix_check.py` and the disk digests byte-identical to main on the CPU and a device.
- The comparison with the reference runtime's image encoding and first token is the phase's final gate, at image grids both runtimes take unchanged.

## 8. Order of work (planned)

Each step is a branch of its own, its failing test first.
0. This page and the STATUS block.
1. `feat/image-decode`: PNG and JPEG decoding, EXIF orientation, RGB conversion, `smart_resize` and the resize, with their byte checks.
2. `feat/vision-ops`: the backend ops on the CPU.
3. `feat/vision-encoder`: the encoder contract and module, `--mmproj`, and a tool that writes an image's rows.
4. `feat/image-prompt`: media runs, positions, the embedding's rows and `--image`, after the sparse attention's pooled-key change.
5. `feat/image-cache-identity`: media runs in every tier.
6. `feat/image-routes`: content parts, limits and errors.
7. `feat/vision-vulkan`: the device ops, identity and speed.
8. `feat/image-flash`: Flash-Next's released checks, the encoder's work split under pacing, then production.
