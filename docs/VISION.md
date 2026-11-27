# Image input

This page is the plan for image input, ROADMAP item 10, as F2DEV approved it as the plan of record on 2026-10-25.
The coordinator and F2DEV answered its open questions the same day, on the owner's delegation.
The image decoder and the Qwen and DeepSeek-V4.1 vision encoders (section 11) are implemented, along with `llmx-encode-image` for checking encoder rows against the reference. Image rows are wired into CLI prompts, chat templates, server requests and cache identity. The sections below describe the approved design and retain the original work order in section 8. Current qualification and performance results belong in STATUS; implementation alone does not close those gates.
Audio remains planned on the same encoder contract.

## 1. The models and the weights

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

## 2. Preprocessing

It follows transformers' `load_image` and `Qwen2VLImageProcessor` (torchvision backend) at the pinned commit, and must give their bytes.
1. Decode PNG or JPEG.
2. Apply the EXIF orientation (`ImageOps.exif_transpose`).
3. Convert to RGB as Pillow's `convert("RGB")` does: alpha dropped without compositing, palettes expanded, gray replicated.
   Inkling's processor composites instead: an image with transparency is converted to RGBA and laid onto a grey checkerboard of 8 px cells, 204 and 153 from the turned image's top left, as (rgb a + cell (255 - a) + 127) / 255 (`image::Matte::checker`). That is the processor given a Pillow image; HF's path and URL route decodes to RGB and drops alpha, so llmx follows the model's own processor and differs from that route only on images with transparency.
4. Choose the size with `smart_resize`: factor 32 (patch 16 times merge 2), at least 65536 and at most 16777216 pixels, Python's `round` (halves to even), and an aspect ratio past 200 refused.
5. Resize with torchvision's v2 `resize` on uint8, bicubic with antialiasing (the separable kernel with a = -0.5, its support widened by the downscale factor), rounded and clamped to uint8; an unchanged size is the identity.
6. Normalize in F32 as `(x - 127.5) / 127.5`, HF's mean 0.5 and std 0.5 fused with its 1/255.
7. Cut 16 by 16 patches in merge-block order (block row, block column, row in block, column in block), each 768 values in (channel, row, column) order; the temporal axis repeats the frame.

An image gives gh * gw / 4 tokens, gh and gw being its patch grid: 64 at least, 1024 for 1024 by 1024 pixels and 16384 at most.
`--image-max-tokens`, on the CLI and the server, bounds the tokens of a resized image, by default at the encoder's own bound, 16384 tokens for Qwen's: the encoder turns N tokens into its resize bound, for Qwen's HF's `max_pixels` processor argument of N * (patch * merge)^2 pixels, so the default is HF's 16777216 and llmx's output equals HF's for any image a user gives; the fit charges the encoder's scratch at the bound, and an image whose tokens do not fit the context is refused by name.

The decoders are written here (`image/`, section 9), PNG with its inflate and JPEG with libjpeg-turbo's integer IDCT, fancy upsampling and color tables, since the reference decodes through Pillow's libjpeg-turbo.

## 3. The encoder

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
With `--mmproj` the fit charges that device, before it sizes the KV, with the encoder's weights, its tables, its arena and its F32 K and V for the largest image `--image-max-tokens` allows, and the scratch of its passes: products over every patch and attention a piece at a time (`VisionEncoder::memory`, `PlacementRequest::encoder`).
A device makes the arena and the K and V at that bound as the encoder opens, so no encode grows them; a lower `--image-max-tokens` leaves more room for the KV.

## 4. A prompt of token ids and encoder rows

- The Qwen templates render an image part as `<|vision_start|><|image_pad|><|vision_end|>` (ids 248053, 248056 and 248054 in the 3.5 and 3.8 vocabularies), and refuse an image in a system message.
- `chat::Message` carries content parts, and the renderer hands the template the parts.
- After tokenizing, each pad id becomes gh * gw / 4 pad ids in image order; the pad count must equal the images given, as HF checks.
- The history keeps the pad ids as tokens, which Flash-Next's per-layer embedding hashes as HF hashes them, and beside them media runs: the first row, the rows, the merged grid and the image's digest.
- The embedding part writes each encoder row over its pad row; Flash-Next spreads it into its four streams as any row.
- Positions are HF's `get_rope_index`: a text row is (p, p, p) with p counting up; an image starting at p puts the merged patch at (row, column) at (p, p + row, p + column), and p then grows by the larger of the merged grid's sides.
  A row's position is its index plus a delta that is a function of the media runs below it; with no image the delta is 0 and nothing changes.
- The sections are interleaved [11, 11, 10] over the 32 rotary frequencies, a constant the module declares beside the sections the GGUF records.
- Flash-Next's sparse attention ropes each pooled key group at its first token's position: once its pooled keys are computed once in the pass that completes a group, that pass's call carries the groups' positions, so this step builds on that change and stores no positions.

## 5. History and cache

- A history's identity is its token ids, its row classes and its media runs; a text-only history has none and keeps its bytes, its disk digests and the saved caches.
- An image's digest is SHA-256 over the encoder file's identity, the preprocessing settings, the grid and the resized pixels, so another `mmproj` reuses nothing.
- The disk tier's block digests hash each media run a block overlaps; the device donors, host entries, boundaries and jobs compare media runs beside the row classes, with one owner beside the row-class comparison, and a fork inside an image needs the whole image to match.
- A request keeps its resized pixels for its life and its encoder rows until its prompt is read; a resume that recomputes encodes again, to the same bits.
- A context cut never falls inside an image, and the lookup drafter skips media runs.

## 6. The routes

- `/v1/chat/completions` and `/v1/chat` take `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}`; a `detail` other than absent or `auto` is refused with the field named; `/v1/completions` stays text.
- Only `data:` URLs of `image/png` and `image/jpeg`, base64 decoded strictly and the format read from its first bytes; http, https and file URLs are refused.
- The limits: the body limit, 20 MiB decoded per image, 89,478,485 pixels, and the image's tokens inside the context; 16-bit PNG, CMYK, arithmetic-coded and 12-bit JPEG, GIF and WebP are refused by name.
  A request's images are held to what a request may hold in tokens (the server's request limit) before any is decoded, at the fewest tokens an image takes under the encoder's sizing (64 for Qwen's at its defaults), and as each is decoded, at their tokens so far, refused with 413 and the numbers, so a request makes no more rasters than its tokens allow.
  A decoder spends no more than a real file costs: a JPEG of more than 64 scans and a PNG whose compressed texts inflate past 1 MiB together are refused, and a JPEG's planes are allocated at their first scan, never for a header alone.
- `/v1/tokenize` counts an image's tokens.
- The CLI takes `--mmproj FILE` on its model commands and `--image FILE` on `generate` and `logits` with `--chat`, so a served reply is held to the CLI's.

## 7. The references and gates

- The reference environment gains torchvision and Pillow, pinned (the coordinator's and F2DEV's decision of 2026-10-25): the Pillow wheel by version, Python tag and platform with the libjpeg-turbo and zlib versions it reports, recorded in each fixture's metadata; the generator refuses other versions; decoded bytes and arrays are committed so the suite needs neither package; no text reference is regenerated.
- The F32 paths are held to 2e-5 a logit and 1e-5 NLL on tiny random-weight fixtures of both families built by HF; device paths below F32 are held to a budget calibrated as [PRECISION](PRECISION.md) sets out.
- Qwen3.5-0.8B's `mmproj` beside its pinned Q8_0 file is checked against HF on the files' own weights in the hosted HF job; Flash-Next's against HF's vision model and end to end through the layered file reference, by hand.
- Each step that touches the runtime, the renderer or the cache keeps text-only `generate`, `logits`, `perplexity`, `chat`, `tools/server_mix_check.py` and the disk digests byte-identical to main on the CPU and a device.
- The comparison with the reference runtime's image encoding and first token is the phase's final gate, at image grids both runtimes take unchanged.

## 8. Original order of work (2026-10-25)

The approved sequence below records the original branches and dependencies. It is not a list of unfinished steps; STATUS records the implemented work and its gate results. Each step has its failing test first.
0. This page and the STATUS block.
1. `feat/image-decode`: PNG and JPEG decoding, EXIF orientation, RGB conversion, `smart_resize` and the resize, with their byte checks.
2. `feat/vision-ops`: the backend ops on the CPU.
3. `feat/vision-encoder`: the encoder contract and module, `--mmproj`, and a tool that writes an image's rows.
4. `feat/image-prompt`: media runs, positions, the embedding's rows and `--image`, after the sparse attention's pooled-key change.
5. `feat/image-cache-identity`: media runs in every tier.
6. `feat/image-routes`: content parts, limits and errors.
7. `feat/vision-vulkan`: the device ops, identity and speed.
8. `feat/image-flash`: Flash-Next's released checks, the encoder's work split under pacing, then production.

## 9. Addendum of 2026-10-25: where the code lives, and other vision models

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

## 10. Speed targets and qualification

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
- The encoder's work in a server is cut into pieces that the scheduler admits beside generating requests (CACHE-PLAN P6): the patch embedding, each block's projections, its attention in runs of query rows of at most 2^37 multiply-adds and its rest, and the merger. An image's pieces are its request's prompt work, so a request at an image without rows spends its prompt slice turn on one piece, one a round, and runs every piece at once while no other request could use the stages; a piece goes to the thread that records the first stage, which records the embedding's backend, as an item between the passes, so no piece stops the stages, and beside replies a replies' pass meets the pieces the chunk budget holds at a piece's measured time, one at least, as a prompt takes a chunk beside them, and the rows come back through host-visible storage once the last piece's submission has retired, which the round looks at again a millisecond on; one image encodes at a time, and a request that ends mid-encode frees the encoder at the next piece boundary (F2DEV's DECISION of 2026-10-26 12:24). A history forked inside an image, a follow-up turn's or a job's, takes the image's rows past its last whole block, less than a block of rows, from the entry holding the history it forks, which keeps them beside its history and frees them with it, so it encodes nothing; only where those rows are gone does a job reading a conversation's reply again on a fork inside its image encode that image again in background turns, which come only while no prompt waits, so its encode gives way at a piece boundary to a request at an image and starts again later.

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

## 11. DeepSeek-V4.1 (`deepseek41v`)

DeepSeek-V4.1-Flash's `-vl` text files take images through an mmproj whose `clip.projector_type` is `deepseek41v`, the second module on the contract of section 9 (`model/arch/deepseek41_vision.hpp`), with no second path in `image/`, `media.hpp`, the routes or the cache.
Its reference is DeepSeek's own inference/image_processor.py, vision.py and model.py at the revision [DEEPSEEK4](DEEPSEEK4.md) pins, run in the deepseek41 reference environment.

- **The files:** the mmproj holds a ViT of patch 14, 1024 wide, 32 blocks of 16 heads of 64, RMSNorm (eps 1e-6, weight only), qkv and output products with bias and a SwiGLU feed-forward of 2816 without bias, then `v.post_ln`, the aligner's `mm.1` (9216 to 5120) and `mm.2` (5120 to 5120) with bias, and the block's marker rows `v.token_embd.img_start`, `img_end` and `v.image_newline`.
  A `-vl` text file is the old file with `blk.N.exp_probs_b_vl.bias` beside each MoE layer's `exp_probs_b`; without an image it renders, routes and decodes as the old file, byte for byte.
- **Preprocessing:** the padded fit of image_processor.py (`image::Fit::padded`): an image below 544 * 544 pixels raised to it, its sides ceiled to whole patches, a block past its row bound solved down by the processor's own rule, then Pillow's `ImageOps.pad` (bicubic with Pillow's 22-bit weights, centred, gray 127), normalized as section 2 and rounded to BF16, cut in rows of patches.
- **Tokens:** an image of h by w aligner cells takes the block [START] ([IMAGE] * w [NEWLINE]) * h [END], h * (w + 1) + 2 rows, every one with the id 129264; `--image-max-tokens N` bounds those rows (1024 by default, the processor's `vision_max_n_token`), N below 4 is refused; a grid thinner than one aligner cell (3 patches a side), which a raised image under 29 pixels a side (an aspect past about 350:1) or a small budget gives, is padded to whole tokens as the official aligner pads it.
  Positions are plain text positions.
- **The encoder:** the 2D rope of section 3's stream map on the whole 64-wide head, bidirectional attention within the image, then the aligner: the grid padded to multiples of 3 with zero rows, a 3 by 3 unfold read as one 9216-wide row, `mm.1`, erf GELU and `mm.2`.
  `mm.1`'s input axis is regrouped once at load from the converter's channel-major order to patch-major, so the unfold is a gather of patch rows with zero copies over the pad, on every backend.
- **The text model:** each pass marks the rows holding 129264 once, and every MoE layer routes them with `exp_probs_b_vl` in place of `exp_probs_b` (the weights still from the raw scores); Engram skips image rows and its n-grams stop at them, as model.py and engram.py have it.
  DSpark's stages route their draft rows with their own `exp_probs_b`, never a `bias_vl`, as `forward_spec` does, and a draft row is a drafted text token, never an image row.
- **Chat:** the file's template reads content as a string, so an image part renders as the marker text and a message's parts are joined with a blank line, as encoding.py joins its blocks; the marker typed in text, in a message, a raw prompt or a CLI prompt, is refused, since only an image part may place it.
- **Gates:** the padded fit, the tiny tower's rows and a tiny end-to-end golden through model.py come from the official code (docs/ASSETS.md, The deepseek41 tiny reference); the tiny tower's and the release mmproj's rows are held per image to twice the official F32 forward's error with the attention sums in llmx's order, the error of the official F32 forward as torch orders it recorded beside it ([DEEPSEEK4](DEEPSEEK4.md), section 5); the image rows' correctness on the real text file waits for the layered reference of DEEPSEEK4 section 5.3.

## 12. Inkling-2 (`inkling2v`)

Inkling-2-Mini's mmproj (`mmproj-BF16.gguf`, projector `inkling2v` under `clip.vision.projector_type`) is the third module on the contract of section 9 (`model/arch/inkling2_vision.hpp`), with its preprocessing in section 2's `pad_end` fit and checker matte.
Its reference is transformers' `Inkling2VisionModel` and `Inkling2ImageProcessor`.

- **The files:** a ViT of patch 20, 1024 wide, 23 blocks of 16 heads of 64, LayerNorm (eps 1e-5) with weight and bias, separate q, k and v products with bias, an MLP of 4096 with bias and erf GELU, a learned 32 by 32 position grid, a class row (`v.class_embd`, `v.class_position_embd.weight`), `v.pre_ln` before the blocks and no norm after, per-block layer scale `ls1` and `ls2`, and the merger's `mm.1` (4096 to 4096) and `mm.2` (4096 to 2304) without bias.
  The patch kernel is stored `[20, 20, 3, 1024]`, a patch read (c, y, x) as llmx's patches are, where HF's linear reads (y, x, c).
- **Tokens:** an image of h by w 40 px tokens takes h * w rows, one text position each; `--image-max-tokens N` bounds them, 4096 by default and at most, and an image past the bound is refused with a message naming the flag, since the processor never resizes.
  The default is the model's practical bound, so 1080p (1296 tokens) and 1440p (2304 tokens) screenshots are read as they are; a smaller `--image-max-tokens` refuses more images.
- **The encoder:** the patch product and the positions' bilinear taps at half-cell centres (align_corners false, border clamp), the class row after the patches (`EncoderPlan::extra_rows`), LayerNorm over every row, then in each block the 2D rope on adjacent pairs of the whole head, the column stream first, positions from 1 and the class row at (0, 0), base 10000; layer scale as `add_bias`'s per-column scale, `(W x + b) * ls`, before each residual add; the merger reads the patch rows only, four consecutive rows as one.
- **Gates:** `tools/inkling2_vision_reference.py --encode` runs HF's tower on the mmproj's own weights in float64 and float32 (eager attention) for five images (one token, a transparent one, 2 by 1, 3 by 2 and a 40-token photo) and keeps the float64 rows and the bound, twice HF float32's worst per-row error (7.51e-06); llmx on the CPU at F32 measured 2.25e-06 at worst on 2026-11-23, below HF float32's own error on every image.
  No HF safetensors were on disk, so the patch kernel's layout is held only by the converter's convention and this comparison.
- **The text model:** the rows take text positions at `<|unused_200054|>`, the id `inkling2_loop` declares as `text_position_images`, and replace bundle row 0 of the normed embedding with the other five rows zero (`ModelPlan::embed_width`), as `Inkling2LoopModel` scatters `pooler_output` into its normed embeddings; the template renders an image part as `<|content_image|>`, the token and `<|end_message|>`.
  `tools/inkling2_file_reference.py image-rows` renders chat rows holding images with HF's processor and keeps the tower's float32 rows on the mmproj's weights; `run` and `greedy` scatter them into HF's text model on the GGUF's weights for the logits and greedy tokens llmx is held to.
