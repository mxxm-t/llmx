# Image input

This page is the plan for image input, ROADMAP item 10, as F2DEV approved it as the plan of record on 2026-10-25.
The coordinator and F2DEV answered its open questions the same day, on the owner's delegation.
The image decoder and standalone Qwen vision encoder are implemented, along with `llmx-encode-image` for checking encoder rows against the reference. Image rows are not yet wired into text prompts, chat templates or the server. The sections below retain the broader image-input plan and distinguish its implemented pieces from the remaining integration work.
Audio follows on the same encoder contract once image input is integrated.

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

The decoders are written here (`image/`, section 9), PNG with its inflate and JPEG with libjpeg-turbo's integer IDCT, fancy upsampling and color tables, since the reference decodes through Pillow's libjpeg-turbo.

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

The encoder has no sequences and no caches, so it has a contract of its own, `model/encoder.hpp`, sized for the known vision models (section 9), with its module at `model/arch/qwen3vl_vision.hpp` chosen in the registry by `clip` and `clip.projector_type`.
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

## 9. Addendum of 2026-10-25: where the code lives, and other vision models (planned)

F2DEV's decisions of 07:45 and 07:46, on the owner's rules that each concern is kept in one place and that image input serves future vision models too.

**Five owners, no more.**
- `src/image/`, beside `tokenizer/` on `core/` alone: everything from a file's bytes to what an encoder reads (decode, orient, size, resize, normalize, patchify, and the digest of the resized pixels), under parameters the encoder module hands it.
- The encoder: `model/encoder.hpp`, the one interface the rest of the runtime calls, and one module a vision model family, `model/arch/qwen3vl_vision.hpp` first, chosen by a registry from the mmproj's `general.architecture` (`clip`) and `clip.projector_type`; the backend ops it needs live in the backends beside the ops they resemble.
- `inference/media.hpp`: a rendered prompt with image parts made into ids with pad runs and media runs, and the pad-count rule; `chat.hpp` only carries parts to the template, and the positions' delta is `model/history.hpp`'s.
- `server/api.hpp`: the data URL, the limits and the refusals, and one call into `image/`; `/v1/tokenize` counts through `media.hpp`.
- The cache: media runs compared in the one place row classes are, and hashed into `DiskIndex::digests` there and nowhere else.

**The encoder interface, sized for the known kinds.**
The projector types the reference runtime reads show what the next modules need: Qwen 2.5 and 3 (`qwen2.5vl_merger`, `qwen3vl_merger`), MiMo-VL (`mimovl`, Qwen-style sizing with banded window attention), Gemma 3 (`gemma3`, SigLIP at a fixed 896 by 896 with a 4 by 4 average pool to 256 tokens and bilinear resizing), Gemma 4 (`gemma4v`, variable resolution of 70 to 1120 tokens, a 3 by 3 pool, a 2D rope and a learned normalization), and the InternVL, Pixtral, Idefics and GLM families.
`model/encoder.hpp` therefore carries what every one of them must give, and the Qwen module implements each member now:
- the preprocessing parameters `image/` applies: patch size, merge, the temporal repeat, mean and std, the size rule (HF's `smart_resize` with its factor and pixel bounds, or a fixed square) and the resample (antialiased bicubic, or bilinear);
- the tokens an image's grid gives;
- the positions its rows take: Qwen's three components with the delta after the image, or a plain run of text positions;
- the texts of the tokens a template marks an image with, read from the text file's vocabulary;
- `encode`: one image's prepared rows to embedding rows at the text model's width, on the device that holds the embedding.

A member only a later module needs comes with that module and no earlier, as ADDING-AN-ARCHITECTURE has it: Gemma's attention that sees an image's rows both ways inside the text model, and MiMo's banded window attention, are the first such members on the list.
A new vision model is then one module, one registry row and its fixtures, with no change to `image/`, `media.hpp`, the routes or the cache.

## 10. Speed (planned)

The owner's gate for the phase: image input beats the reference runtime's mtmd path heavily, the owner's fork and the castagna-veloce build, on the same MI50s, model, mmproj and images at default clocks, in three figures:
- the encode time of one image;
- the time to the first token of a request with images;
- tokens a second with several images in one request, and with several requests with images at once.

The references' figures are recorded first, on one image set whose grids both runtimes take unchanged: sides that are multiples of 32, between 64 and 4096 tokens an image (512 by 512, 1024 by 1024, 1280 by 768, 2048 by 1024 and 2048 by 2048), on Qwen3.8-Flash-Next and Qwen3.5-0.8B with their BF16 mmproj files, each figure with its conditions.
Every step's design states its cost per image; the CPU steps record their encode time as a figure the device step must beat as well; step 7 measures the gate and the phase's final gate holds it.

Speed designed in from the start:
- Decoding, resizing and the digest run on the connection's thread, never on the scheduler's.
- As section 5 has it, a request keeps its resized pixels for its life and its encoder rows until its prompt is read, and prefix reuse covers the same image at the same place; a cache of encoder rows keyed by digest is a tier of its own and comes only on a measured gain, since the gate's workloads, fresh images a request, gain nothing from it.
- Images of one shape in a request go to the encoder in one pass, their attention kept apart by segment, so a pass's bits stay each image's alone.
- On the device an image stays on the device from its patches to its rows, with no host round trip a layer, and few launches an image: the patch embedding and position mix in one kernel, a layer's norm and projections fused where the kernels allow, attention in one kernel a layer, the merger fused; about 6 launches a layer and under 200 an image.
- The encoder's work in a server is cut into pieces that the scheduler's pacing admits beside generating requests (CACHE-PLAN P6): the patch embedding, each block's projections, its attention in runs of query rows of at most 2^37 multiply-adds and its rest, and the merger. An image's pieces are its request's prompt work, so a request at an image without rows spends its prompt slice turn on one piece, one a round, and runs every piece at once while no other request could use the stages; one image encodes at a time, and a request that ends mid-encode frees the encoder at the next piece boundary (F2DEV's DECISION of 2026-10-26 12:24). A job reading a conversation's reply again on a fork inside its image encodes that image again in background turns, which come only while no prompt waits, so its encode gives way at a piece boundary to a request at an image and starts again later.

The cost of each step at one 1024 by 1024 image (1024 tokens, 4096 patches) on the 27-layer, 1152-wide tower, before any measurement:

| Step | What it costs an image | Expected |
|---|---|---|
| 1, decoding | a 1024 by 1024 JPEG decoded, an unchanged size resized (the identity) | about 10 to 20 ms, on the connection's thread |
| 2, CPU ops | the ops alone, no image path | the ops' own rates in `vision-ops` |
| 3, CPU encoder | about 5.5 TFLOP: 3.4 in the blocks' products, 2.1 in attention over 4096 patches, 0.08 in the patch embedding and the merger | about 10 to 15 s on the rig's 16 cores, the figure step 7 must beat |
| 4, prompt | 1024 encoder rows written over the pad rows, positions from the runs | well under 1 ms; the text model's prefill of 1024 tokens dominates the first token |
| 5, cache | SHA-256 of 3 MB of pixels; a prefix hit skips the image's encode and its prefill | about 10 ms |
| 6, routes | base64 of about 1.4 MB | about 2 ms |
| 7, Vulkan encoder | the 5.5 TFLOP at F16 on one MI50 | under 0.4 s, under 200 launches; owns the speed gate |
| 8, Flash-Next | encoder pieces under pacing; the first token after the image's 1024 prompt rows | the gate's three figures on production's placement |
