# Inkling-2-Mini (`inkling2_loop`), planned

Inkling-2-Mini is a 24-layer mixture-of-experts model whose middle twelve layers run three times per token.
This page is the design of its support: what the model computes, how the loop maps onto the runtime's layers, caches, split and fit, which backend ops it needs, how it is held to HF, and the order of the work.
The stream is tracked in `docs/STATUS.md` (Inkling-2-Mini).
Text comes first, then the DSpark drafter, then vision.

## 1. The files

The supported layout is the `inkling2_loop` GGUF that mainline llama.cpp reads (unsloth's `Mainline_Support` folder).
Unsloth's root-folder looped layout names the same tensors differently and is refused: llmx keeps one loader path.
The UD-Q8_K_XL file holds 744 tensors: the routed experts of layers 0-19 as NVFP4 with a per-expert F32 `.scale` companion, the experts of layers 20-23 as BF16, the projections as BF16 and the norms, router, convolution taps, clips and queries as F32.
NVFP4 weights with expert companions already load on the CPU and Vulkan (`src/model/weights.hpp`); the GPU core for ROCm and CUDA gains NVFP4 as a step of its own (8. Order of work, below).
The unsloth UD files are the main target and the file the HF-exact gate runs on.
ggml-org's `Inkling-2-Mini-Q8_0.gguf` holds the same names, shapes and metadata keys with every matrix, the experts included, as Q8_0 and no `.scale` companions (704 tensors), so it loads through the same path and brings up ROCm on the existing Q8_0 kernels before NVFP4 is there.

Measured against the original checkpoint (`thinkingmachines/Inkling-2-Mini`, read by HTTP ranges), the file's layouts are:

- Every short-convolution weight, the full-attention layers' static taps included, is `[4, C]` with a channel's four taps contiguous and the last one on the current token, so the full-attention file tap `t` is HF's `static_weight[3 - t]`.
- The full-attention layers' `attn_k_conv_proj` and `attn_v_conv_proj` keep HF's output order, head times four plus lag, lag 0 on the current token: the static and the per-token taps index lags in opposite orders.
- `attn_q_clip` and `attn_rel_clip` are HF's per-head scales unchanged, a head's values contiguous.
- `attn_rel_bias` is HF's `rel_logits_proj.proj` `[R, extent]` transposed, stored `ne [R, extent]` with a position's R values contiguous, equal to the checkpoint's BF16 values once transposed (layers 0 and 5 checked).
- `ffn_gate_up_exps` holds HF's de-interleaved order, the gate rows first; the model reads it as two stack roles over runs of its rows (`Role::first_row`), the gate's and the up's.
- `blk.N.loop_msg_query` of loop layer N holds HF's merged message queries of that layer, query, key, value and feed-forward in that order, bit for bit.
- The NVFP4 experts' `.scale` companion is the checkpoint's `amax / outer_scale_denominator` in F32 bit for bit, and the decoded experts equal HF's dequantization of the original codes and block scales value for value, differing only in the sign of zero.
- `output.weight` has 201024 rows where HF's head has the 200058 real ones; the logits beyond them are not the model's.

The drafter is `dspark-Inkling-2-Mini-Q8_0.gguf` (`general.architecture = dflash`, 54 tensors) and the vision tower `mmproj-BF16.gguf`.

## 2. What the model computes

The reference is HF transformers' `Inkling2LoopForCausalLM` (`models/inkling2_loop`, transformers 5.24.0).
Hidden 2304, 24 query and 8 key-value heads of 128, no RoPE, vocabulary 201024 of which 200058 are real tokens.
The file's tokenizer has the 200058 tokens; the 966 head rows past them belong to no token and are not zero in the unsloth files (unit-norm like the real rows), so HF's head and llmx's read only the first 200058 rows and no padded id can be scored or sampled.
Layers 5, 11, 17 and 23 attend over the whole context; the others over a window of 1024 positions.

- **Embedding:** the token row, then an RMSNorm with its own weights.
- **Attention residuals:** a token carries up to four block states besides its hidden row.
  At each block boundary (layers 0, 6, 12 and 18) the layer's input joins the block states and the residual stream restarts from the attention output.
  The attention and feed-forward inputs of a layer are a softmax mix, with learned queries, of the unweighted-RMS-normed block states and the current row; a full-attention layer mixes three inputs, one each for queries, keys and values.
- **Attention:** queries are scaled per head and clipped to L2 norm 22.63, keys clipped to the same norm, values RMS-normed without weights.
  Keys and values pass a causal short convolution of four taps first, `(conv(x) + x) / sqrt(5)`; on the full-attention layers the taps add per-token, per-head weights projected from the layer input, `(conv(x) + x) / 3`.
  A per-token, per-head relative state of 16 values, scaled and clipped to norm 8, times a learned 16 x 1024 table gives an additive bias for distances 0 to 1023, zero beyond.
  Each token and head adds a sink logit with a zero value to the softmax.
  On the full-attention layers queries and relative states are scaled by `1 + 0.1 log(max((pos + 1) / 8192, 1))`.
  The output is gated by a sigmoid of a projection of the query input before the output projection.
- **Convolutions after attention and after the feed-forward part:** the same four-tap short convolution over the hidden width.
- **Experts:** 128 routed, 8 per token, no shared expert.
  The router is F32: sigmoid scores, picked by score plus bias, weighted by score, normalised and scaled by 8.
  The expert activation is `4 tanh(g / 4) * sigmoid(g) * 25 tanh(u / 25)` over a fused gate-up projection.
- **Loop:** layers 6 to 17 run three passes from the state after layer 5.
  Passes 2 and 3 take messages from the previous pass: four learned queries per layer mix the previous pass's input to layer 12 and its output of layer 17, and a gate of rank 64, `(1 - tanh(f)) * h + tanh(g) * m`, blends each message into the query, key, value and feed-forward inputs.
  Layers 18 to 23 continue from the third pass.
- **Head:** a last attention-residual mix, the output norm, the output projection and a softcap of 30.

The exact formulas, tensor names and shapes are in the HF file and are restated in `tests/` fixtures as they are built.

## 3. The loop in the runtime

The plan lists 48 execution layers in the order they run: 0-5, 6-17 three times, 18-23.
Each has its own KV and convolution state, as HF gives each pass its own cache layers.
Execution layers of the same physical layer name the same tensors, which the runtime adopts once per device (`resolve_tensors` caches by tensor and device).
A layer's `kind` carries whether it attends globally, is a block boundary, is in pass 2 or 3, or opens a pass.

The restart needs no saved row: layer 6 is a block boundary, so each pass stores its input, the output of layer 5, as a block state.
A pass restarts from that state and drops the later ones.
The messages read the previous pass's input to layer 12 and output of layer 17, which the restart copies to two rows of their own before the new pass overwrites them; each layer computes its four messages when it runs.

The residual row is a bundle of six hidden-width rows, `[h, B0, B1, B2, B3 or M0, M1]`, as deepseek4's hyper-connection streams are, so handoffs and crossings carry the block states with no new runtime code.
Block state b sits in bundle row 1 + b, and the messages from the row after the loop's last block, so the post-loop block appended at layer 18 overwrites M0 only once the loop is done.
The embedding writes the token row and zeros the other five, `copy_rows` moves rows between the bundle and contiguous slots, and `query_mix` reads the bundle in place, its items `h` first.
The last loop layer of a pass before the last ends the pass: its output and the pass's later blocks become the messages, and `h` restarts from the loop's first block state.
Carried slots that cross only after the layer that writes them are the cheaper form, kept for when crossings show in profiles.

A tensor group splits a layer's attention by heads, a member's q heads with the KV heads they read, and the experts by their hidden rows; each part's output is summed over the group and each member closes it with its convolution and residual (`Inkling2::join`).
A member's slot holds the rings of its own KV heads, in a slot as wide as one card's.

Two runtime changes come first, in a contract commit of their own that leaves other models byte-identical:

- The fit counts a tensor that several execution layers share once (`place.hpp` dedupes within one layer only), and so does a tensor group's shards.
- A placement keeps the three passes of the loop on one device: a split point inside execution layers 6 to 41 is refused, as it would cross between devices six times a token.

## 4. Caches

Every execution layer is an entries layer (`Cache::entries`), the kind deepseek4's window layers use.

- A sliding-window layer keeps a ring of [k|v] rows, 1023 positions plus the headroom, and four convolution rings of three rows plus the headroom (keys 1024, values 1024, after attention 2304, after the feed-forward part 2304 wide), in its state slot.
- A full-attention layer keeps its keys and values as pooled [k|v] rows of one position in the KV blocks and its convolution rings in its slot. They are to move to ordinary K/V heads (`kv_write`, `KVViews`, per-layer heads and width through `KVShape` once it lands from the DSpine stream), so the cache types and the device attention layout apply to them.
- The sliding layers keep their rings for memory: paged K/V over every position would hold about 10 GB at 32k tokens against 0.36 GB of rings (Fable estimate, unmeasured), until KV storage can release the blocks below a window.
- The headroom is 64 rows, the most a verified draft takes, or the window where that is smaller, as the fixture's is.
  `depthwise_conv` over a ring and `attention_rel_sink` read a pass's own rows from their inputs and write only the last rows of a pass that fit a ring, after reading the history, so a pass longer than a ring needs no sub-blocks.
- A rejected draft or a failed pass of at most the headroom rolls back by length, so no layer needs a rerun; a longer failed pass rewinds the history.
- At F32 a sequence slot takes about 0.44 GB: 8.9 MB of [k|v] ring in each of the 40 sliding execution layers and 1.8 MB of convolution rings in each of the 48.
- Prefix reuse stores checkpoints of the state slots as qwen35 does.
  A checkpoint copies every ring, so the server keeps fewer of them than for deepseek4; an f16 format would halve both.

The rings are F32 until a native f16 format is calibrated per `docs/PRECISION.md`.
A block-shared windowed KV that checkpoints for free is the alternative, recorded in the ideas ledger.

## 5. Backend ops

New ops, each on the CPU, Vulkan and once in the GPU core for ROCm and CUDA:

- `attention_rel_sink`: attention over a ring or pooled rows with a per-row, per-head bias table of 1024 floats indexed by distance, a per-row, per-head sink and a window.
  The bias table itself is a matmul of the relative states by the 16 x 1024 table.
  Its optional `RelClip` clips q and rel first as `l2_clip` clips them; the GPU row kernel applies it to the heads it stages with the same sums where a block holds one head and the row one key part, so the shallow decode clip takes no launch of its own, and deeper rows, the tile path and the other backends run `l2_clip` first.
  When the full-attention layers move to ordinary K/V heads, the bias and the sink become an optional extras struct on `attention` (Fable ac643fa4): a precomputed bias `[rows][heads][extent]` from that matmul, its extent and a sink `[rows][heads]`, built into the device attention kernels as a variant the way the window is.
  `attention_rel_sink` then keeps only the sliding layers' ring and drops its pooled rows, the CPU sharing one softmax helper between the two; the full layers pin f32 K/V until an f16 cache is calibrated for them.
- `l2_clip`: per-head scale, clip to a threshold, and the optional position log scale.
- `depthwise_conv` over a ring (`Op::depthwise_conv_ring`, the PLE convolution's op with a ring history, no activation, the input as residual and a divisor): a ring of past rows, static taps, optional per-row, per-head taps, and the residual over the square root.
- `query_mix`: the attention-residual mix of up to six normed rows by up to four learned queries, used for the layer inputs, the head and the messages; a later pass's layer passes its message mix and its residual mix in one call, which the GPU runs in one launch; a first-pass sliding layer and the head pass their norm with the mix (`QueryMixRows::norm`), which the GPU runs in the mix's launch.
- `tanh_gate_mix`: the message gate's tail.
- A softcapped GLU form `(4, 25)` over a fused gate-up expert stack, `GluForm::softcap` with `up_limit` beside the limit, and `sigmoid_routing`, which the CPU and Vulkan have and the GPU core is to have; the router's 1e-20 is `RouteRule::eps`, 0 for MiniMax-M3.1.
- `softcap`: the final logit softcap of 30, in place on the logits after the head's product.
- `copy_rows`: a strided row copy or add, which moves rows between the residual bundle and contiguous slots.

The output gate reuses `sigmoid_mul`, the unweighted value norm a group RMS norm with unit weights.

### Row classes

- The CPU `attention_rel_sink` sums its softmax denominator and its values by blocks of 128 absolute positions, each from zero into an f32 total, through the helper the generic attention uses, the sink after the blocks (docs/PRECISION.md, Attention value sums).
- Vulkan `attention_rel_sink` outside the tile splits every row's keys into parts of 64 and merges them in part order, whatever the call's rows; empty parts of shorter rows weigh zero, so a decode row has the same bits alone, beside other sequences and in any slice.
- The GPU core's `attention_rel_sink` outside the tile splits a row of n keys into parts and merges them in part order, by a rule of n alone, so a decode row has the same bits alone, beside other sequences and in any slice.
  Where a kv head serves three or four query heads, heads are 128 wide and a row has at most 16 relative values, a block takes the kv head's heads in a lanes build (`llmx_attention_rel_lanes_g3`, `llmx_attention_rel_lanes_g4`), ring or pooled keys alike, with parts of `gpu_shape::rel_lane_keys(n)` keys.
  - Sixteen lanes take a key, eight dimensions a lane, so a block scores 16 keys a step; lane x also takes relative value x, so the bias joins the dot in one 16-lane sum.
  - The views and block table go in device memory (`pass_arg`), since a pooled row reads a block index a key.
  - After the q loads each thread prefetches a line of the kernel's own code into L2, since its merge tail is cold by the next layer (docs/src/backends-gpu.md, Kernel notes).
  - Each 16-lane group keeps its own online softmax; the groups merge by shuffles within a wave, the waves in LDS, and the parts in the last block to finish, which loads every part's maximum and sum at once.
  - On one MI50 at tensor width 2's shape (12 heads, 4 kv heads), against the earlier one-key-a-thread G = 3 ring build: 35.8 to 16.8, 37.9 to 22.9, 98.7 to 43.2 and 259 to 128 us at 512, 1024, 4096 and 16384 keys; at one card's (24, 8) 54.2 to 31.6 us at 1024 keys (inkspd2ag mb2, f32 rows).
  Other shapes run a one-head instance of the row kernel (`llmx_attention_rel_sink`), a thread a key and parts of `gpu_shape::rel_part_keys(n)` keys; a row of at most 128 keys is one part.
- A view whose prompt's extent reaches the profile's `attention_tile_rows` takes the tile, which sums in another order: a class of its own, keyed by `row_class`. The extent is the call's `RowRuns` run holding the view's first row, so a prompt takes the same build over a ring or pooled rows, alone, beside other sequences or sliced over calls, on Vulkan (a tile dispatch a view) and on the GPU core (the view table's seventh word).
- Every multiply-add in the Inkling shaders and kernels is an explicit fma and the sums whose order matters are `precise` (docs/ADDING-AN-ARCHITECTURE.md).

## 6. Correctness

The reference is `tools/inkling2_file_reference.py`, after `tools/minimax_file_reference.py`: HF's own class in F32 with eager attention, its weights decoded from the GGUF by `tests/spec_decode.py`, every weight but the embedding and the experts resident and the experts decoded one at a time.
Its `names` command checks every tensor the reference reads, by name, shape and type, and that no file tensor goes unread, without reading tensor data; a job runs it before the long rows.
It is F32 on the file's values, not HF as shipped, which dequantizes the experts to bf16.
The 512-token rows run whole; the 16k rows run in chunks of 1024 tokens through HF's own cache, keeping the last 512 rows whole and every position's top 20 logits, log-sum-exp and target NLL, since a whole 16k row's scores would not fit 31 GB.
Router ties follow MiniMax's rule (docs/MINIMAX.md, Classes and checks) through the same `RouteFollower` of tools/minimax_file_reference.py: `run --routes DIR` reads llmx-route-probe's `NAME.routes`, keyed by the router call's place in the pass (loop passes included), takes llmx's pick wherever every expert the two picks differ in scores within `MARGIN_ULPS` ulps of HF's k-th score, weighted as llmx's `RouteRule` weighs it (sum plus 1e-20, times `route_scale`), and `compare` reports such picks and fails a pick beyond it.
`MARGIN_ULPS` is sized once, on HF's own scores, at twice the largest deviation of llmx's CPU F32 router scores over raw[:64], and frozen before any rerun: 3072 ulps, from a largest deviation of 1536 ulps (1.79e-6) with the weights exact (2026-11-23, llmx 8a94f84d); a width or device arm never sizes it, and a pick beyond it is a question for the F32 class, never a reason to widen it.
Its `fixture` command runs the same reference on the fixture's file against the golden: 4.9e-8 whole and 4.0e-6 in chunks of 7, as close as the golden's own chunked run.
`run --kv f16` holds the default cache's arm: every new key after its convolution and clip and every new value after its norm is rounded to binary16 and widened back before HF's cache takes it, the row's own key included, as the IQ1 correction rounds K and V (docs/PRECISION.md, 4. The test side); on the fixture it moves the logits by 2.8e-3 from the F32 golden, whole and in chunks, and leaves `--kv f32` bit for bit as before.
A synthetic fixture shrinks the window, the relative extent, the log-scale floor and the headrooms so every path, ring wrap and the log scale included, runs within a few dozen tokens, with three passes.
Gates follow `docs/ADDING-AN-ARCHITECTURE.md`: the fixture's logits within 2e-5 and NLL within 1e-5 of HF, the split check, slice invariance, and the real file against file-exact HF at 512 tokens and at 16k raw and chat-template tokens, past the 8192-position log-scale floor and many ring wraps.

The tap order is settled by the file layouts (1. The files, above).
The full-attention queries and relative states are scaled, clipped and then log-scaled, as HF computes them; the file stores only the scales, so the order is the model's and not the file's.

## 7. Chat format

The file's template renders once the chat renderer takes Jinja's `float` filter (fix/chat-filter-float, its own branch off main), byte for byte as transformers renders it.
It has no generation prompt: the model writes `<|message_model|>` and then a content tag itself, and only a reply ends with the end token, `<|content_model_end_sampling|>` (id 200006, the file's `eos_token_id` and `eot_token_id`).
A reply is a run of segments, `<|message_model|>` [name] tag body `<|end_message|>`, with the tag `<|content_thinking|>` for reasoning, `<|content_text|>` for text and `<|content_invoke_tool_json|>` for a call whose body is `{"name":...,"args":{...}}`, closed by the end token.
Every concern below has an owner main already has; each is extended there, with no second path (Fable a985d458, the plan reviewed by Fable ad26f586), on feat/chat-inkling (gitea PR 570, stacked on fix/chat-filter-float):

- Generation prompt: nothing is appended, so the prompt stays the reference's bytes; `chat::ChatFormat::opens_reasoning` is false under segments, so a `<think>` a user types opens no reasoning, and `chat::stable_prefix` holds since the renders with and without a generation prompt are equal.
- Stop: `is_eos` on 200006 ends a reply in `generate` and the scheduler. A second stop id is added only if a run shows the model writing `<|endoftext|>` or `<|message_user|>`, and then as a stop-id set `bpe::Tokenizer` reads from the file, not as server stop strings.
- Reply split: `chat::Segments` holds the grammar as data (`chat::tagged_segments`), which `chat_format` reads where the template writes the open and reasoning tags (`ChatFormat::segments`); `chat::ReplySplit` takes it beside `opened`, so the server's chat routes and `chat::reply_message` reach it. Thinking segments are reasoning, text segments content, and call segments pass whole to `ToolCallSplit`.
- Tool calls: a fourth `ToolFormat` value, `segment`, read from the template writing `<|content_invoke_tool_json|>` under segments, and its reader in `ToolCallSplit`, the name before the tag and the body's `args` the arguments; the chat fixture's `tool_format` for the template is `segment`, and the server's tool round trip runs under it.
- The next turn: `ChatFormat::assistant` splits a reply by the same grammar into reasoning, text and calls, for `chat` and the server to keep the same turn; while a reply is written, `ChatFormat::reply_unsettled` keeps the server from predicting a next turn outside an open text segment. The fixture's turns under the template are the replies its model writes, cut from transformers' renders.
- Reasoning effort: `ChatFormat::reasoning_effort` and `TemplateVar` carry it, as a string on the server's top-level field and as any value through `chat_template_kwargs`; `chat` passes none, so its effort is the template's 0.9. Fixture cases with numbers and the template's names (`none` to `max`) are untried.
- Message ends: the prompt holds the end token only after replies, so the server's `MessageEnds` places message ends by rendering each prefix, which is correct but renders once per message on each fit; `message_spans` gives no spans for a miss note, and the text route cuts only after replies. The scheduler's cache cut points are to be read before that is called harmless.
- The tags reach the split as text: `bpe::Tokenizer::decode` writes each token's characters back as bytes, special tokens included.

Untried: a reply prefix the format owns (`<|message_model|><|content_text|>` to turn reasoning off, `<|content_thinking|>` to force it); a split by token id, so text a user types cannot open a segment; cut points from a message-end token the format names (`<|end_message|>` beside the end token), with no renders; `<|content_invoke_tool_text|>` and `<|content_tool_error|>` readers or a refusal; images through the parts path.

## 8. Order of work

1. This design and the STATUS block.
2. The contract commit: shared tensors counted once by the fit and the shards, the loop kept on one device.
3. The synthetic fixture (`tests/fixture_inkling2.py`, its HF-to-file map stating the layouts above once), its HF golden (`tools/inkling2_reference.py`) and the file-exact reference (`tools/inkling2_file_reference.py`).
4. Configuration, plan and refusals, with `tests/arch_inkling2.cpp`.
5. The CPU ops with unit tests, then the CPU model on the fixture and on the real file against HF.
6. Vulkan, then the GPU core for ROCm and CUDA, brought up on ggml-org's Q8_0 file.
7. NVFP4 in the GPU core, its own branch so every NVFP4 model gains it: raw-block spec-decoder tests (`tests/spec_decode.py`, the round-trip component's raw blocks), device identity against the CPU on ROCm and CUDA, then the UD files on ROCm and CUDA and their speed against the Q8_0 file.
8. Serving, then the DSpark drafter (with the dspine-drafter stream's block-drafter pairing, `src/model/arch/dflash.hpp`), then vision.
9. Speed against mx-llama.cpp on the same files, cards and prompts.
