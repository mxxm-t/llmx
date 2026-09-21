# llmx - Development Status

Current implementation and remaining work. Historical checkpoints, failed
experiments and raw evidence remain in [ASSETS](ASSETS.md) and
`docs/benchmarks/`; their dated next steps are not current blockers.

## Vulkan backend, sub-step 1 of docs/VULKAN.md (2026-09-21)

- **Goal:** the first vendor backend over the Radeon VII: storage and
  submission first, kernels in the following sub-steps.
- **Done:** `src/backends/vulkan/vulkan_backend.cpp`, the one translation
  unit outside the header-only runtime, built as a static library only
  with `LLMX_HAS_BACKEND_VULKAN=ON`. The loader is loaded at run time and
  every entry point fetched through it, so nothing links against
  `vulkan-1` and a build without the option is byte-for-byte what it was.
  One instance, the physical device by index, one compute queue without
  graphics, timeline semaphores required and 8- and 16-bit storage and
  arithmetic enabled where present, push descriptors enabled where
  present. Buffers are one `VkBuffer` on their own memory: device-local
  and not host-visible for `Memory::device`, host-visible, coherent and
  cached for `Memory::host_visible`, mapped for their lifetime. `adopt`,
  `write` and `read` go through a 64 MiB staging buffer in chunks;
  `alloc` zero-fills in stream order; `copy` is a device copy. `submit`
  ends the open command buffer and signals the timeline with the ticket,
  `wait` blocks on the value, `sync` submits what is open and waits on the
  last ticket, and a ring of four command buffers is reused as tickets
  retire. One full barrier between consecutive commands. The compute ops
  throw naming their sub-step. `backend-vulkan` passes 13 checks on the
  Radeon VII and exits 77, which CTest reports as skipped, without a
  loader or device. The CMake option-to-flag conversion that the generated
  config needed is exercised by every configure and the default tree is
  unchanged: 19/19 native, Python 12/12.
- **Slip, recorded:** these files were swept into `757dee9`, the commit
  that recorded the fork gate, before they had been built, by a `git add
  -A` that should have been scoped. The default tree was verified within
  minutes and was never broken, since the option is off; the Vulkan tree
  was built and its test run right after, and this commit carries the
  description the previous one lacked.
- **Done: sub-step 2.** Six GLSL compute shaders under
  `src/backends/vulkan/shaders/`, compiled by `glslc` at build time into
  the generated include directory as numeric arrays and embedded, with a
  shared `q.glsl` for block decoding. Pipelines are built on first use
  with a push-descriptor set layout and 128 bytes of push constants; a
  dispatch binds, pushes buffers and constants, launches and fences.
  Elementwise kernels are one invocation per element; the norms are one
  workgroup per row or per (row, head) with a shared-memory reduction, so
  they do not depend on the subgroup size; `embed` decodes F32 and Q8_0
  rows, the table bound once as floats and once as bytes. `backend-vulkan`
  compares every kernel against the CPU backend on random inputs, bounds
  fixed in the test before the first run: exact for add, gather and both
  embed paths, 1e-6 relative for SiLU, 1e-5 for the norms and RoPE;
  160,688 outputs match on the Radeon VII, and out-of-range rows, positions
  and ids are refused on the host. The default tree is untouched by any of
  it.
- **Done: sub-step 3, the row kernel.** `matmul` for F32 and Q8_0: one
  subgroup per output row, lanes striding over blocks, batch columns in
  chunks of eight held in registers, one `subgroupAdd` per column. Q8_0
  rows with an even block count are read as 32-bit words over block pairs
  with the activations as 16-byte vectors, which took the 4096-square Q8_0
  matvec from 32 GB/s to 201 GB/s on the Radeon VII; odd block counts keep
  the 16-bit path. The subgroup size is queried and must divide 256.
  Checked against the CPU backend over batch widths 1, 3, 8 and 13 and
  both parities at 1e-4 relative; 167,388 kernel outputs match in all.
- **Done: sub-step 4.** KV blocks on the device, one K and one V buffer
  per layer holding `[kv_head][token][head_dim]` in 64-token blocks, grown
  by allocate and copy on the queue with the buffers the copy reads from
  kept alive until it retires. `kv_write` scatters a view's rows; `kv_copy`
  copies a block within each layer's buffer; `attention` is one workgroup
  per (row, head) with the subgroups taking tokens round robin, one
  `subgroupAdd` per token for the score, an online softmax so no score
  array is needed, and a shared-memory merge of the subgroups. Checked
  against the CPU backend through each backend's own storage and block
  size on histories of 0, 63, 64, 65 and 131 tokens with 1 and 3 queries,
  two views in one call, and a copied block attending like its source, at
  1e-4 relative; 179,228 kernel outputs match in all. The device's bounds
  checks caught a query offset in the test that the CPU backend reads
  through silently.
- **Done: sub-step 5, all but the floor.** `--device cpu|vulkan:N` on
  `generate`, `chat`, `logits`, `perplexity` and `bench`; a build without
  the backend says so rather than falling back. The Python suite takes
  `--device` and passes it through `LLMX_DEVICE`, test configuration like
  `LLMX_BASELINE_GGUF`; a fixture the device has no kernel for is reported
  as skipped, with the reason, rather than failed. On the Radeon VII the
  whole suite passes with every command on the device: the synthetic F32
  and sharded HF cases, the chat and thread goldens, and the real Qwen3
  Q8_0 baselines, whose top logits and all four perplexity cases match the
  CPU's numbers to the digit (PPL 28.8371 continuous, 38.2140 at c=123).
  Q4_0 was skipped until sub-step 6 gave the device its Q4_1 and Q6_K
  decoders; it passes now, see below.

  Throughput after the arena fix, 247-token prompt and 32 greedy tokens,
  one run each, no gate: 0.6B prefill 382 tok/s and decode 67.6, against
  the CPU's 431 and 46.7 at six threads; 8B prefill 40.3 and decode 21.3,
  against the CPU's 37 and about 4. The first numbers were 19 and 10.9
  tok/s decode, because every dispatch taking ids or positions allocated
  a device buffer, over a hundred `vkAllocateMemory` calls per token; a
  host-visible arena per ring slot removed that. Prefill on the device is
  the row kernel streaming the weights once per eight columns, which the
  tile kernel is for.
- **The floor, measured.** The pinned mx revision's Vulkan backend did
  not build here: its shader generator's nested configure fails under
  CMake 4.x with "CMAKE_C_COMPILER not set", with both the pip and the
  Visual Studio CMake, and is left for a CMake 3.x retry. The reference
  used instead is today's upstream llama.cpp Windows Vulkan release,
  b11075, which is a stricter bar since its kernels are newer. Same card,
  same files, 247-token prompt and 32 generated tokens; llmx numbers are
  single runs after the changes below, the reference is `llama-bench`
  with three repeats.

  | model | phase | llama.cpp b11075 Vulkan | llmx Vulkan | llmx share |
  |---|---|---:|---:|---:|
  | Qwen3-0.6B-Q8_0 | prefill | 660 tok/s | 1195 (three runs within 1%) | 181% |
  | Qwen3-0.6B-Q8_0 | decode | 198 tok/s | 111 | 56% |
  | Qwen3-0.6B-Q4_0 | prefill | 636 tok/s | 1110 (997, 1110, 1110) | 175% |
  | Qwen3-0.6B-Q4_0 | decode | 209 tok/s | 107 (90, 106, 108) | 51% |
  | Qwen3-8B-Q8_0 | prefill | 99 tok/s | 220 | 222% |
  | Qwen3-8B-Q8_0 | decode | 39.7 tok/s | 32.7 | 82% |

  Prefill clears the reference on both models since the tile kernel;
  decode is at about half of it, and the roadmap's bar is the reference,
  not the CPU. Where the time goes, measured per kernel at the 0.6B shapes rather
  than guessed: a near-empty dispatch with its barrier costs 6.3 us, so
  the four hundred dispatches of a token are under 3 of its milliseconds;
  the Q8_0 matvec has a floor of about 17 us at 1 MB and streams at 65 to
  125 GB/s on the 1 to 3 MB layer matrices and about 210 GB/s at 8B's
  sizes, on a card whose memory does 1 TB/s; decode attention was 134 us
  per layer with one workgroup per head walking the whole history. Three
  changes so far: the inter-dispatch barrier names the compute and
  transfer stages instead of all commands (decode +14% on 0.6B); a
  subgroup takes several rows with a lane cluster each, so a 1024-wide
  row no longer idles three quarters of its lanes (neutral, so occupancy
  was not the limit); and attention splits the history into 32-token
  chunks across workgroups with a merge kernel, 134 to 36 us per layer,
  decode 78 to 99.5 tok/s on 0.6B. `backend-vulkan` reports the per-shape
  timings so the next change is measured against them. Fourth, the tile
  kernel for wide batches: a workgroup computes a 64 x 64 output tile
  with the weights dequantized once into shared memory and every thread
  accumulating a 4 x 4 micro-tile, so a weight is read once per pass
  rather than once per eight columns; batches of 16 and up take it.
  Prefill 449 to 1025 tok/s on 0.6B and 40 to 220 on 8B, checked against
  the CPU at batch widths 16, 64, 100 and 247 at 1e-4.
  Fifth, the matvec's access pattern: eight consecutive lanes share one
  block pair, lane t loading words t, t + 8 and, for lane 0, word 16, so
  a load instruction touches 32 contiguous bytes per pair instead of
  four, and every lane reads the two scales directly, which the hardware
  serves as one broadcast and which beat a shuffle. The 8B matrices went
  from 210 to about 295 GB/s and 8B decode from 21.8 to 28.1 tok/s. Two
  variants measured and rejected: a generic loop over words with
  accumulator arrays indexed by column spilled to scratch and ran at 27
  GB/s, and two pairs per iteration cost occupancy and lost a few percent.
  Sixth, grouped projections: the row kernel takes up to three
  projections of one X in one dispatch, workgroups handed to projections
  in order so a workgroup's buffers are selected once, which is dynamic
  indexing of a storage buffer array and a device feature the backend
  now requires. q, k and v are one dispatch, gate and up another, checked
  bit for bit against the same projections one at a time. Decode 104 to
  110 tok/s on 0.6B and 28.1 to 29.3 on 8B.
  Seventh, the lanes per block pair swept: one, two, four, eight and
  sixteen give 200, 190, 336, 295 and 185 GB/s at the 8B shapes, so four
  it is, each lane with five loads in flight over 16 contiguous bytes.
  8B decode 29.3 to 32.7 tok/s.
  Eighth, sub-step 6 for the Q4_0 fixture: Q4_0, Q4_1 and Q6_K in
  `embed`, the tile kernel and the row kernel, checked against the CPU
  (exact for embed, 1e-4 for the matmuls; 599,588 outputs), and the
  fixture's HF baselines pass on the device with the CPU's numbers
  (PPL 32.8463 continuous, 42.5740 at c=123). Two findings on the way.
  The first version put every type's branch in the one row module and
  Q8_0 decode fell from 105 to 65 tok/s with no change to any executed
  Q8_0 instruction, the register demand of the whole module setting the
  occupancy of every path; the row kernel is now one module per family
  of types, F32 and Q8_0, Q4_0 and Q4_1, Q6_K, built from one source with
  a define, and Q8_0 is back at 110 to 112 against the parent commit's
  110 to 112 in an interleaved run. The second: byte loads with scalar
  activation loads gave Q4_0 17 GB/s, Q4_1 35 and Q6_K 6 to 17 at the 8B
  shapes, and the Q6_K head of the fixture, 151,936 rows, took 22 ms of a
  38 ms token. The word paths read the quants as 32-bit words, Q4_0 as a
  block per lane over the 9-word pair with the first block's nibble
  words assembled from two loads, Q4_1 as one lane per 5-word block,
  Q6_K as sixteen lanes per block each holding three words of quants,
  two of sub-scales and the scale, with every other block's words
  assembled from two loads since 210 bytes is not a multiple of four;
  and the activations as aligned 16-byte vectors, one load for four.
  Now 149, 188 and 170 GB/s at the 8B shapes and the head at 102 GB/s,
  1.25 ms. Q4_0 decode 26 to 107 tok/s; llama.cpp does 209 on the same
  file.
  Ninth, the rest of sub-step 6: Q4_K and Q5_K in `embed`, the tile
  kernel and the row kernel, checked against the CPU the same way
  (723,796 outputs across the seven types). The row kernel is now five
  modules from one source: F32 and Q8_0, Q4_0 and Q4_1, Q4_K, Q5_K,
  Q6_K. Q4_K and Q5_K beside Q6_K cost Q6_K 40 percent (170 to 99 GB/s)
  the way every type beside Q8_0 had cost Q8_0, and Q4_K alone gained 30
  percent over Q4_K beside Q5_K (104 to 135 GB/s); at the 8B shapes the
  row kernel reads Q4_K at 135 GB/s, Q5_K at 123 and Q6_K at 163. Eight
  lanes share a Q4_K or Q5_K block, each taking the sixteen nibble bytes
  of one half of one 64-value chunk, with the three packed sub-scale
  words read by every lane. `Qwen3-0.6B-Q5_K_M.gguf` (168 Q5_K, 29 Q6_K,
  113 F32; same repo and revision as the Q4_0 file) is a third
  `BASELINE_MODELS` fixture, as the fused Q5_K block below proposed, with
  bounds set from the CPU's measured HF deltas plus margin: top-5 overlap
  4, NLL delta 0.05 continuous (measured 0.026) and 0.16 per chunk
  (measured 0.130, 0.068, 0.024). On the device its logits and all four
  perplexity cases match the CPU's numbers to the digit (PPL 29.5612
  continuous, 38.6676 at c=123).

  A confirmation run of the reference at the end of the session, asked
  for by the user, moved with the machine: llama.cpp 0.6B Q8_0 measured
  602 prefill and 182 decode against 660 and 198 three hours earlier,
  and llmx in the same minutes 1000 to 1157 and 102.7 to 106.5 against
  1195 and 111; Q4_0 the same, 624/209 against 636/209 and llmx 878 to
  1056 and 95.7 to 97.4 against 1110 and 107. Both arms fell together, so
  the shares hold and neither run is a regression; the table keeps the
  earlier pairs and the session-end pairs side by side.

  | model | phase | llama.cpp b11075 Vulkan | llmx Vulkan | llmx share |
  |---|---|---:|---:|---:|
  | Qwen3-0.6B-Q5_K_M | prefill | 492 tok/s | 640 (533, 638, 645) | 130% |
  | Qwen3-0.6B-Q5_K_M | decode | 205 tok/s | 87 (81, 88, 87) | 42% |
  | Qwen3-0.6B-Q8_0, session end | prefill | 602 tok/s | 1000 to 1157 | 166 to 192% |
  | Qwen3-0.6B-Q8_0, session end | decode | 182 tok/s | 102.7 to 106.5 | 56 to 59% |
  | Qwen3-0.6B-Q4_0, session end | prefill | 624 tok/s | 878 to 1056 | 141 to 169% |
  | Qwen3-0.6B-Q4_0, session end | decode | 209 tok/s | 95.7 to 97.4 | 46% |
  Tenth, two decode wins found by timing the host against the device.
  Temporary instrumentation put a 0.6B decode token at 0.5 ms of host
  recording and 7.4 ms of device time over 396 dispatches, so the
  device is the story, and the row kernel's final reduction turned out
  to run its xor-shuffle chain over all eight column slots whether one
  column was live or eight: 48 shuffles per lane after five loads.
  Reducing only the live columns took the 1024-square Q8_0 matvec from
  17.1 to 13.2 us, 1024x3072 from 33 to 22 us and the 8B shapes from 313
  to 373 GB/s, and 0.6B decode from 122 to 145 tok/s over 128 tokens.
  Then the pass is submitted in chunks of 64 dispatches as it is
  recorded, so the device starts while the host records the rest: 145
  to 158 tok/s; chunks of 16, 32, 64, 128 and 256 gave 150, 156, 158,
  152 and 149, and 8B prefill is unchanged by it (203 to 208 either
  way). Both changes leave every kernel output identical and the greedy
  text the same. Three interleaved rounds of the four files afterwards:

  | model | phase | llama.cpp b11075 Vulkan | llmx Vulkan | llmx share |
  |---|---|---:|---:|---:|
  | Qwen3-0.6B-Q8_0 | prefill | 660 tok/s | 1029, 1105, 1109 | 156 to 168% |
  | Qwen3-0.6B-Q8_0 | decode | 198 tok/s | 137.9, 137.1, 136.5 | 69% |
  | Qwen3-0.6B-Q4_0 | prefill | 636 tok/s | 934, 962, 1040 | 147 to 164% |
  | Qwen3-0.6B-Q4_0 | decode | 209 tok/s | 128.2, 129.4, 127.3 | 61% |
  | Qwen3-0.6B-Q5_K_M | prefill | 492 tok/s | 681, 736, 742 | 138 to 151% |
  | Qwen3-0.6B-Q5_K_M | decode | 205 tok/s | 122.1, 110.5, 118.4 | 54 to 60% |
  | Qwen3-8B-Q8_0 | prefill | 99 tok/s | 208, 208, 209 | 210% |
  | Qwen3-8B-Q8_0 | decode | 39.7 tok/s | 35.5, 35.5, 35.7 | 89% |

  Decode over 128 tokens reads higher than over 32 (158 against 138 on
  0.6B Q8_0) because the first tokens carry the clock ramp; llama-bench
  warms up before its 32, so the 32-token llmx figure is the
  conservative one and the table keeps it.
  Eleventh, the residual add folded into the matmul's store. The model
  asks for `matmul_add`, Y += W X, and each backend produces it its own
  way: the CPU computes into scratch and adds, the same arithmetic as
  before to the bit, the device sets a flag in the row and tile kernels'
  push constants and the store becomes an accumulate. Two dispatches
  fewer per layer; 0.6B Q8_0 decode 137 to 140-146 tok/s, 8B 35.5 to
  35.8, checked against the CPU at batch widths 1, 3 and 64.

  Twelfth, and the one that changes the reading of every table above:
  the user asked whether the measurement was right, and it was not
  matched. The reference's bench tool warms up, then averages three
  repeats of prompt processing into an empty history and of generating
  32 tokens from an empty history, model time only. `generate --verbose`
  gave one cold run whose decode came after the 247-token prompt, so
  with attention over 250 to 280 tokens every step, with sampling and
  text output inside the timer and the clock ramp in the first tokens.
  `llmx bench --model` now runs the reference's protocol (docs/USAGE.md),
  and the same binary that read 138 tok/s under `generate` reads 202
  under it. Both arms in the same minutes, same card and files:

  | model | phase | reference b11075 Vulkan | llmx Vulkan | llmx share |
  |---|---|---:|---:|---:|
  | Qwen3-0.6B-Q8_0 | pp247 | 651.2 +- 4.3 tok/s | 1507.6 +- 5.1 | 232% |
  | Qwen3-0.6B-Q8_0 | tg32 | 195.1 +- 0.6 tok/s | 201.9 +- 0.2 | 103% |
  | Qwen3-0.6B-Q4_0 | pp247 | 669.7 +- 3.1 tok/s | 1367.1 +- 5.0 | 204% |
  | Qwen3-0.6B-Q4_0 | tg32 | 220.8 +- 0.9 tok/s | 194.8 +- 1.5 | 88% |
  | Qwen3-0.6B-Q5_K_M | pp247 | 518.1 +- 1.6 tok/s | 875.1 +- 3.5 | 169% |
  | Qwen3-0.6B-Q5_K_M | tg32 | 219.8 +- 1.0 tok/s | 169.4 +- 0.7 | 77% |
  | Qwen3-8B-Q8_0 | pp247 | 97.0 +- 0.5 tok/s | 232.5 +- 0.4 | 240% |
  | Qwen3-8B-Q8_0 | tg32 | 38.8 +- 0.1 tok/s | 40.6 +- 0.1 | 105% |

  Q8_0 decode is at the floor on both models under the matched protocol.
  The earlier tables stand as what `generate --verbose` measured, both
  arms' figures at the time; they are not the floor comparison.

  Thirteenth, the attention inputs as one op. `norm_rope_kv` is what the
  model asks for at every layer: q normed and rotated in place, k normed
  and rotated and written with v into the view's KV blocks. The base
  class default is the three ops it replaces, which is what the CPU
  runs, so its arithmetic is unchanged; the device runs one kernel, a
  workgroup per (row, head) over the q, k and v heads, the k heads
  written straight into their blocks, three dispatches fewer per layer.
  Checked against the CPU's three ops with a 70-token history, q
  directly at 1e-5 and k and v through attention at 1e-4. Under the
  matched protocol, same session as the table above:

  | model | test | reference b11075 Vulkan | llmx before | llmx now | llmx share |
  |---|---|---:|---:|---:|---:|
  | Qwen3-0.6B-Q8_0 | tg32 | 195.1 tok/s | 201.9 | 221.3 +- 0.8 | 113% |
  | Qwen3-0.6B-Q4_0 | tg32 | 220.8 tok/s | 194.8 | 205.4 +- 0.4 | 93% |
  | Qwen3-0.6B-Q5_K_M | tg32 | 219.8 tok/s | 169.4 | 176.7 +- 0.4 | 80% |
  | Qwen3-8B-Q8_0 | tg32 | 38.8 tok/s | 40.6 | 41.0 +- 0.0 | 106% |

  Prefill moved within noise (1494, 1352, 873 and 230 tok/s).

  Fourteenth, the norm folded into the matmul, tried three ways and
  rejected, not committed. The op was `matmul_group_normed`, the RMS
  norm of X against w and the projections in one call, the CPU norming
  into scratch first and the device folding it into the row kernel for
  one column. (1) The column staged in 16 KB of shared memory per
  workgroup, normed there, every read from it: 8B decode 41.0 to 13.4
  tok/s and 0.6B 221 to 101, with the plain modules carrying the array
  too; in separate normed modules, so the plain ones had none, 8B 22.3
  and 0.6B 165. Sixteen KB of shared memory per workgroup caps what a
  compute unit holds. (2) No staging, each read scaled by the row's
  factor and w on the way in: 8B 12.5, 0.6B 137; with the reduction
  removed and the scale alone left, 8B 11.7, so the per-read scale and
  its second load are what the tuned Q8_0 load pattern cannot absorb.
  (3) A 4 KB staging for rows up to 1024 wide: 0.6B Q8_0 203, Q4_0 199,
  Q5_K_M 176 against 221, 205 and 177 unfused. Every variant also
  moved the prefill's norm into a backend scratch buffer, and that
  alone cost 8B prefill 232 to 166 tok/s; the buffer was device-local
  and allocated once, and the cause was not found before the whole
  change was reverted, so a backend-allocated activation buffer is a
  thing to measure before using again. The two norm dispatches per
  layer stay: on 0.6B they are about a tenth of a decode token and no
  fusion tried gets them back.

  Fifteenth, the greedy-output hash the user asked for: `generate` with
  the 247-token excerpt, 128 greedy tokens, on the CPU and on the
  device, output hashed. Qwen3-0.6B Q8_0, Q4_0 and Q5_K_M give the same
  hash on both backends (e41aee746cc31abf, c19367d54fb40354,
  b18a64cda0208547), so the device's reduction-order differences never
  flipped an argmax over 128 steps on any of the three quantizations,
  and Qwen3-8B-Q8_0 gives the same hash on both over 64 tokens
  (379c733a478ec58a). The check is CPU against device on the same file:
  a greedy sequence from a quantized file cannot be checked against the
  fp32 reference beyond its first tokens, since quantization legitimately
  moves the argmax, so HF stays the ranking and NLL gate.

  Sixteenth, the long-context case the user asked for, `bench --model`
  with 16384 prompt tokens and 512 generated, two repeats after a
  warm-up, both arms in the same minutes:

  | model | test | reference b11075 Vulkan | llmx Vulkan | llmx share |
  |---|---|---:|---:|---:|
  | Qwen3-0.6B-Q8_0 | pp16384 | 514.1 +- 1.7 tok/s | 155.2 +- 3.7 | 30% |
  | Qwen3-0.6B-Q8_0 | tg512 | 177.2 +- 0.1 tok/s | 192.5 +- 0.2 | 109% |
  | Qwen3-8B-Q8_0 | pp16384 | 97.3 +- 0.5 tok/s | bad allocation | - |
  | Qwen3-8B-Q8_0 | tg512 | 37.6 +- 0.1 tok/s | bad allocation | - |

  Two findings. Decode over a 512-token history still leads, but a
  16384-token prompt runs at 30 percent of the reference: the device
  attention kernel is one workgroup per (query row, head) with an
  online softmax, which is the flash form for decode, but prefill has
  no query tiling, so every query row re-reads its whole K/V history
  and the traffic is quadratic. At 247 tokens that is invisible behind
  the matmuls; at 16384 it is the prompt. And Qwen3-8B does not fit a
  16k context on the 16 GB card: the KV cache is F32, 16896 tokens of it
  are about 5 GB on top of 8.7 GB of weights, and the allocation
  failed, where the reference's f16 KV fits.
  Seventeenth, the tiled prefill attention. A wide pass of 128-wide
  heads takes `attention_tile`: a workgroup per 32 query rows and head,
  the head's K and V streamed through shared memory in tiles of 16
  tokens, so a tile is read once per 32 rows instead of once per row;
  eight lanes share a row, each holding 16 elements of it, a score is
  three xor shuffles, the softmax is online per row as before. Rows
  past the causal limit are masked inside the tile that holds them and
  later tiles are not read. Checked against the CPU at 32 and 45 rows
  after 0 and 70 tokens at 1e-4. Same session as the table above, the
  8B case at 8k because 16k does not fit its f32 cache:

  | model | test | reference b11075 Vulkan | llmx before | llmx now | llmx share |
  |---|---|---:|---:|---:|---:|
  | Qwen3-0.6B-Q8_0 | pp16384 | 514.1 +- 1.7 tok/s | 155.2 | 513.3 +- 0.2 | 100% |
  | Qwen3-0.6B-Q8_0 | tg512 | 177.2 +- 0.1 tok/s | 192.5 | 191.9 +- 0.5 | 108% |
  | Qwen3-8B-Q8_0 | pp8192 | 115.7 +- 0.2 tok/s | - | 185.9 +- 1.3 | 161% |
  | Qwen3-8B-Q8_0 | tg512 | 37.3 +- 0.1 tok/s | - | 39.2 +- 0.0 | 105% |
  | Qwen3-0.6B-Q8_0 | pp247 | 656.0 +- 2.1 tok/s | 1507.6 | 1610.5 +- 4.1 | 246% |
  | Qwen3-0.6B-Q8_0 | tg32 | 198.3 +- 0.3 tok/s | 221.3 | 226.9 +- 0.4 | 114% |

  The 16k prompt went from 30 percent of the reference to level with
  it, and the kernel does not yet share a K/V tile across the query
  heads of a KV group, which is the next factor available there.
  Eighteenth, the f16 cache sides. `--cache-type-k` and `--cache-type-v`
  take `f32` or `f16` for `generate`, `chat`, `logits`, `perplexity` and
  `bench --model`, the same on the CPU and the device; the type reaches
  each backend through `kv_alloc`, the CPU converts with F16C on write
  and read, the device builds four variants of every cache kernel and
  the storage picks one, and the device writes halves with an explicit
  round-to-nearest-even in the bits because `packHalf2x16` leaves the
  rounding to the driver and the first build differed from the CPU by
  an f16 ulp. The two backends now hold identical cache bytes; checked
  for every combination of K and V types through `kv_write`,
  `norm_rope_kv`, `kv_copy` and both attention kernels, device against
  CPU at 1e-4 and both against the f32 cache at 2e-2. The HF gate with
  both sides f16 (`run_tests.py --cache-type f16`) passes on both
  backends with the same numbers to three places: Q8_0 NLL delta
  0.001254 against 0.001374 with f32 caches, Q4_0 0.131584, Q5_K_M
  0.026174; the synthetic F32 and shard gates, which compare exact f32
  arithmetic, skip under f16 caches by design. Halving the cache lets
  Qwen3-8B run the 16k context that failed to allocate with f32:

  | model | test | reference b11075 Vulkan | llmx f32 cache | llmx f16 cache | llmx share (f16) |
  |---|---|---:|---:|---:|---:|
  | Qwen3-8B-Q8_0 | pp16384 | 97.3 +- 0.5 tok/s | bad allocation | 141.7 +- 0.3 | 146% |
  | Qwen3-8B-Q8_0 | tg512 | 37.6 +- 0.1 tok/s | bad allocation | 38.7 +- 0.0 | 103% |
  | Qwen3-8B-Q8_0 | pp8192 | 115.7 +- 0.2 tok/s | 185.9 +- 1.3 | 189.9 +- 2.7 | 164% |
  | Qwen3-8B-Q8_0 | tg512 (8k) | 37.3 +- 0.1 tok/s | 39.2 +- 0.0 | 39.5 +- 0.0 | 106% |
  | Qwen3-0.6B-Q8_0 | pp16384 | 514.1 +- 1.7 tok/s | 513.3 +- 0.2 | 512.9 +- 0.7 | 100% |
  | Qwen3-0.6B-Q8_0 | tg512 | 177.2 +- 0.1 tok/s | 191.9 +- 0.5 | 203.6 +- 0.1 | 115% |
  | Qwen3-0.6B-Q8_0 | pp247 | 656.0 +- 2.1 tok/s | 1610.5 +- 4.1 | 1620.9 +- 3.7 | 247% |
  | Qwen3-0.6B-Q8_0 | tg32 | 198.3 +- 0.3 tok/s | 226.9 +- 0.4 | 225.1 +- 0.5 | 113% |

  The default stays f32 for now: the flags exist, the gate passes with
  f16, and switching the default is a separate decision recorded when it
  is taken.
  Nineteenth, mixed groups partitioned by type. A Q5_K_M layer's q and k
  are Q5_K and its v is Q6_K, and a group of mixed types fell back to
  one dispatch per projection; it is now one dispatch per type, two for
  that group instead of three. Q5_K_M tg32 176.7 to 185.7 +- 0.5 tok/s
  (84 percent of the reference's 219.8); Q4_0, whose groups are pure,
  read 211.2 +- 0.3 in the same minutes against 205.4 earlier, which is
  the session's drift, not the change. After the reduction fix the row
  kernel's per-type readings at the 8B shapes are Q8_0 415 GB/s, Q6_K
  215, Q5_K 179, Q4_0 178, Q4_K 167, and the Q6_K head of the 0.6B
  files, 151,936 rows of 1024, takes 693 us at 184 GB/s, 7 percent of a
  Q5_K_M token; at 1024 wide every row's lanes re-read the whole
  activation row, 620 MB of cache traffic against 127 MB of weights.
  Twentieth, two rows per lane cluster in the Q6_K module, tried and
  rejected, not committed: each lane decoded its sixteenth of two rows'
  blocks and applied both to the activation vectors it had loaded once,
  halving the activation re-read. The second row's quants and scales
  doubled the live registers and the occupancy lost outweighed the
  traffic saved: the head went from 693 to 1184 us, the 8B shape from
  215 to 129 GB/s, and Q5_K_M tg32 from 185.7 to 161.1 tok/s. The
  activation re-read is the traffic, but the answer is not more state
  per lane. Twenty-first, the other way round, a 4 KB shared-memory
  copy of the activation row per workgroup in the Q6_K module for
  one-column passes up to 1024 wide, every lane reading from it: also
  rejected, the head 693 to 771 us, Q5_K_M tg32 185.7 to 180.1, and
  even the 4096-wide shape, which was not staged, fell from 215 to 175
  GB/s from the select on every read. The K-quant row kernels at these
  shapes are not limited by activation traffic in a way either fix can
  reach; what remains there is the per-dispatch floor and the decode
  work per lane, and it is left at 84 percent for now.
- **Left:** decode on the 4- and 5-bit files, at 93 and 84 percent of
  the reference under the matched protocol. The tiled attention does not yet share a K/V tile
  across the query heads of a KV group. And the question of the default
  cache type, f16 being the reference's default and passing the gate
  here.

## KV cache fork, step 2 of the KV design (2026-09-21)

- **Goal:** a second history with the same committed tokens, sharing every
  full block and copying the partial tail, which the server needs for
  prefix reuse and branching and which the fork rule in KV-CACHE.md had
  described without implementing.
- **Done:** `KVSequence::fork` retains every full block and takes a fresh
  block for a partial tail, reporting the ids so the caller fills it;
  `Backend::kv_copy` copies every layer's K and V of one block into
  another, because only the backend knows the layout; `Model::fork` does
  both on every storage and the fork inherits the tickets of the passes
  that wrote what it shares. Shared blocks are read-only: `prepare`
  refuses to append into one, which a history truncated into a shared
  block would do, so such a history is forked again instead. `kv-cache`
  checks a fork across a block edge (shared block at refcount two, tail
  private and copied, both histories diverging without touching the
  shared block, a boundary fork taking no tail, the refused append,
  releases following the refcounts) and, through the model, that a forked
  sequence continues exactly as a fresh one fed the same tokens while the
  original continues exactly as if never forked. Native 19/19, Python
  12/12 with both HF models.
- **Done: gate**, base `c85aa9d`, candidate `215ad24`, layout control
  `d8d508e`. Five 0.6B cells at 15 pairs, one 8B at 9. System CPU averaged
  47 and 43 percent in the first two 0.6B cells and 38 to 39 in the rest,
  against the benchmark's own 37.5, so other activity was present during
  the two cells that moved most. Evidence in
  `docs/benchmarks/kv-fork-20260921/`, raw monitors archived and hashed;
  every sample kept.

  | cell | phase | candidate mean / median / base wins | control mean / median / base wins |
  |---|---|---|---|
  | 0.6B-1 | prefill | -1.39% / -2.54% / 9/15 | -1.59% / -1.05% / 9/15 |
  | 0.6B-1 | decode  | +3.38% / +2.15% / 5/15 | +3.23% / +0.95% / 7/15 |
  | 0.6B-2 | prefill | **-3.92% / -3.51% / 11/15 FAIL** | +1.53% / +2.26% / 5/15 |
  | 0.6B-2 | decode  | **-3.66% / -4.01% / 12/15 FAIL** | +1.31% / -0.16% / 8/15 |
  | 0.6B-3 | prefill | -0.76% / -2.13% / 9/15 | +1.45% / +1.36% / 6/15 |
  | 0.6B-3 | decode  | -1.12% / -0.06% / 8/15 | -0.24% / -0.39% / 8/15 |
  | 0.6B-4 | prefill | -2.20% / -2.35% / 11/15 | -1.45% / -0.74% / 9/15 |
  | 0.6B-4 | decode  | +0.01% / +0.55% / 7/15 | +0.28% / -0.21% / 8/15 |
  | 0.6B-5 | prefill | -1.77% / -1.53% / 10/15 | -0.48% / -0.97% / 8/15 |
  | 0.6B-5 | decode  | -0.60% / -0.40% / 10/15 | -1.96% / -2.10% / 10/15 |
  | 8B | prefill | -0.23% / +1.08% / 4/9 | +0.01% / +0.55% / 4/9 |
  | 8B | decode  | +0.31% / +0.00% / 4/9 | -1.00% / -0.22% / 6/9 |

  Cell 2 failed both phases at about -3.8 percent with the control positive
  in the same cell; the three cells after it pass and 8B is flat. What the
  measured path gained from this change is one refcount comparison per
  pass in `prepare`. Two things are written down rather than argued away.
  0.6B decode, the phase that comparison sits on, is flat over five cells:
  +3.38, -3.66, -1.12, +0.01, -0.60, mean -0.4 against a control mean of
  +0.5. 0.6B prefill is under base in all five cells, mean -2.0 against a
  control mean of -0.1, which is the same size and sign as the step 2 gate
  and the opposite of the step 6 gate on changes of the same character;
  it is inside the band AGENTS.md documents for this file and this model,
  and it joins the running list of candidates if 0.6B prefill is ever
  found a few points low against an older baseline.
- **Left:** nothing. The prefix index is step 4 of the KV design and lands
  with the server.

## Execution model for batching and placement (ROADMAP #5, #7) (2026-09-21)

- **Goal:** fix what the backend interface and the model layer need for the
  per-layer and per-tensor splits and for continuous batching, now that both
  are scoped and per-row split is dropped, so the first vendor backend
  implements each signature once.
- **Done:** the design, `docs/EXECUTION.md`: `submit`/`wait` tickets with
  no events, host-visible memory and `write` returning with the transfer as
  its caller, per-row positions with the RoPE table as a buffer, batched KV
  views with `gather_rows`, and `Model` / `Sequence` / `ExecContext` /
  `Batch`. Seven ordered steps; steps 1 to 4 change the interface and go
  before Vulkan. ROADMAP #4a marked done, #5 rewritten without per-row, #7
  pointed at the design; stale "steps 5 and 6 remain" claims corrected in
  ARCHITECTURE, DEVICE-EXECUTION, README and the backend page.
- **Done:** amended for architectures beyond dense Qwen (hybrid compressed
  attention, MoE, lookup-table memory, residual mixing): tables are per
  storage counted in that storage's entries, `kv_alloc` describes an entry
  by key and value widths, placement is per tensor role with attention and
  feed-forward separate, and a sequence may hold private unpaged state.
  The "Beyond dense Qwen" section lists the assumptions steps 3, 4 and 6
  must not make.
- **Done: step 1.** `norm_rope_rows` takes the cos/sin tables as buffers
  and one position per row; the model adopts its table once and passes a
  slice of an identity position table. `Backend::rope` is deleted; `bench`
  times `norm_rope_rows` instead of a function the runtime never called.
  `backend-group` checks three rows at positions 5, 2 and 9 against a
  double-precision reference reading the table at each row's own position.
  Native 18/18, Python 12/12 with both HF models.
- **Done: step 1 gate**, three arms built the same way from detached
  worktrees: base `f1e0be4`, candidate `7519170`, and a layout control
  `769761e` that is base plus an unused function appended to each of the
  two files the change edits (`cpu_backend.hpp`, `arch_qwen.hpp`), per the
  AGENTS rule that the control must perturb the same files. Qwen3-0.6B-Q8_0
  five times at 15 pairs, Qwen3-8B-Q8_0 once at 9, six threads, 250-token
  prompt, 32 generated tokens. Evidence in
  `docs/benchmarks/rope-positions-20260921/`; raw monitors archived beside
  the repo and hashed in each cell's `monitor-summary.json`. System CPU
  averaged 40 to 46 percent across cells against the benchmark's own 37.5,
  so other activity was present throughout; every sample is kept.

  | cell | phase | candidate mean / median / base wins | control mean / median / base wins |
  |---|---|---|---|
  | 0.6B-1 | prefill | +3.83% / +3.84% / 4/15 | -2.61% / -0.42% / 9/15 |
  | 0.6B-1 | decode  | +1.47% / +1.80% / 5/15 | +0.33% / +3.32% / 5/15 |
  | 0.6B-2 | prefill | -0.29% / -0.06% / 8/15 | -1.33% / +0.16% / 7/15 |
  | 0.6B-2 | decode  | +0.46% / +0.91% / 5/15 | +1.45% / +0.87% / 6/15 |
  | 0.6B-3 | prefill | -0.35% / +2.29% / 6/15 | +0.73% / +0.03% / 7/15 |
  | 0.6B-3 | decode  | **-2.31% / -1.75% / 13/15 FAIL** | -2.22% / -1.61% / 10/15 |
  | 0.6B-4 | prefill | +1.60% / +0.06% / 7/15 | +0.78% / +0.38% / 7/15 |
  | 0.6B-4 | decode  | -0.08% / -0.27% / 8/15 | -1.29% / -1.39% / 10/15 |
  | 0.6B-5 | prefill | -0.86% / -0.81% / 8/15 | +0.44% / +0.09% / 7/15 |
  | 0.6B-5 | decode  | +0.51% / +0.34% / 6/15 | +2.06% / +1.84% / 6/15 |
  | 8B | prefill | +1.85% / +0.23% / 3/9 | +2.01% / +0.08% / 4/9 |
  | 8B | decode  | +0.14% / +0.45% / 4/9 | +0.75% / +0.45% / 2/9 |

  Fails at 12 of 15 and 8 of 9 base wins. One cell failed, 0.6B-3 decode,
  on the win count. In that same cell the control, which executes the
  same instructions as base, lost 2.22 percent with 10 base wins, so the
  cell moved against both later arms rather than against the change, and
  the two reruns the goal requires came back at -0.08 and +0.51 with 8 and
  6 base wins. Over five 0.6B cells the candidate's decode mean is
  +0.01 percent. Recorded as noise confirmed by rerun, with the failing
  cell kept. The +3.83 prefill in cell 1 is not claimed either; the
  control spans -2.61 to +0.78 on the same measurement.
- **Done: step 2.** `submit()` returns a monotonic ticket and `wait()`
  blocks on one; `sync()` stays what the failure paths call, because a
  failed pass has ops queued behind no ticket. `alloc` takes a `Memory`
  kind; the logits buffer is host visible and the host reads it in place
  after the wait, so no read op copies a row it can already see. The model
  submits once per pass, waits on that ticket for the logits, and `reset`
  waits on the last ticket instead of draining. `kv-cache` counts the
  calls: one submission waited on once per step, no read op for logits, a
  three-token prompt at ubatch 2 submitting twice and waiting once, and a
  reset that waits without a sync. Native 18/18, Python 12/12 with both HF
  models.
- **Done: step 2 gate**, same shape as step 1: base `2bbfd34`, candidate
  `4ee6f48`, layout control `bd0cdc7` perturbing `cpu_backend.hpp` and
  `arch_qwen.hpp`. Three 0.6B cells at 15 pairs, one 8B at 9. System CPU
  averaged 37 to 40 percent per cell against the benchmark's own 37.5, so
  the machine was close to quiet; every sample is kept. Evidence in
  `docs/benchmarks/tickets-20260921/`, raw monitors archived and hashed.

  | cell | phase | candidate mean / median / base wins | control mean / median / base wins |
  |---|---|---|---|
  | 0.6B-1 | prefill | -1.72% / -2.45% / 10/15 | -1.03% / -1.20% / 9/15 |
  | 0.6B-1 | decode  | +0.58% / +1.04% / 4/15 | +0.79% / +0.21% / 7/15 |
  | 0.6B-2 | prefill | -1.10% / -1.54% / 9/15 | -1.98% / -0.24% / 9/15 |
  | 0.6B-2 | decode  | +1.02% / -0.04% / 8/15 | -0.84% / -1.84% / 9/15 |
  | 0.6B-3 | prefill | -2.02% / -2.84% / 10/15 | +0.21% / -0.43% / 9/15 |
  | 0.6B-3 | decode  | +0.67% / +1.03% / 6/15 | +2.02% / -0.54% / 8/15 |
  | 8B | prefill | +0.54% / +0.36% / 3/9 | +0.74% / +1.00% / 1/9 |
  | 8B | decode  | +0.86% / +0.66% / 3/9 | +0.98% / +0.66% / 2/9 |

  All eight cells pass. The candidate's 0.6B prefill is below base in all
  three cells, by 1.1 to 2.0 percent on the mean, which is inside the band
  the control itself spans (-1.98 to +0.21) and the change adds nothing a
  prefill executes beyond one counter increment per pass. Recorded rather
  than explained away; if a later step finds 0.6B prefill a point or two
  low against an older baseline, this is a candidate along with the code
  read's three points.
- **Done: step 3.** `attention` and `kv_write` take an array of `KVView`s
  laid out in row order, and a view carries `nq`, the rows of the pass that
  are its own; `pos` and `batch` are gone because both follow from the
  view, and the sequence fills `nq` from what it has prepared and not
  committed. The CPU backend takes the views one after another around the
  code it had, so one view computes what it did before. `kv-cache` checks
  two sequences with histories of different lengths across a block edge,
  written and attended in one call, against the same two taken separately,
  bit for bit. `gather_rows` is deferred to step 4, where a `Batch` gives
  it a real caller. Native 18/18, Python 12/12 with both HF models.
- **Done: step 3 gate**, base `c850149`, candidate `8a3ff82`, layout
  control `a09c584` perturbing the two edited files. Three 0.6B cells at
  15 pairs, one 8B at 9. System CPU averaged 37 to 39 percent per cell
  against the benchmark's own 37.5; every sample is kept. Evidence in
  `docs/benchmarks/views-20260921/`, raw monitors archived and hashed.

  | cell | phase | candidate mean / median / base wins | control mean / median / base wins |
  |---|---|---|---|
  | 0.6B-1 | prefill | -0.60% / -0.32% / 8/15 | +0.97% / +0.81% / 6/15 |
  | 0.6B-1 | decode  | **-1.21% / -1.59% / 12/15 FAIL** | -0.76% / -0.53% / 8/15 |
  | 0.6B-2 | prefill | +1.83% / +2.08% / 5/15 | +2.98% / +1.27% / 7/15 |
  | 0.6B-2 | decode  | +0.10% / +0.18% / 7/15 | +0.52% / +0.81% / 5/15 |
  | 0.6B-3 | prefill | +1.82% / +1.37% / 4/15 | +2.45% / +2.25% / 4/15 |
  | 0.6B-3 | decode  | +1.75% / +2.59% / 5/15 | +2.12% / +1.98% / 5/15 |
  | 8B | prefill | +1.60% / +0.03% / 4/9 | +1.42% / -0.31% / 6/9 |
  | 8B | decode  | -0.10% / +0.22% / 4/9 | -0.08% / -0.65% / 5/9 |

  One cell failed, 0.6B-1 decode, on the win count alone at exactly the
  threshold; its mean and median are inside the noise band and the control
  was negative in the same cell. The two cells run after it, identical in
  plan, came back at +0.10 and +1.75 with 7 and 5 base wins, and 8B decode
  is flat. Recorded as noise confirmed by rerun, with the failing cell
  kept.
- **Done: step 4.** `Model` owns the weights, the cache's pool and physical
  storage and the backend, read-only after construction apart from pool
  bookkeeping. `Sequence` is one request's history and its last ticket,
  `ExecContext` one pass in flight, and `Model::forward` runs one pass over
  a batch of entries, each a sequence with tokens to append and whether it
  wants logits; rows that want logits are compacted by `gather_rows`, which
  gets its first caller, and the head runs once over them. `forward`
  submits and returns; the context waits on the ticket the first time its
  logits are read, so a caller with two contexts can submit the next pass
  before reading this one. `step` and `prefill` are wrappers on a sequence
  and a context the model keeps, so the CLI, `generate`, `chat` and
  `perplexity` are unchanged. Decode runs the same row-batched graph as
  prefill with one row, which retires the separate decode arena and the
  single-row helpers. `kv-cache` runs two sequences in one pass, one
  decoding over a history while the other prefills, against the two alone,
  and refuses a sequence listed twice; `backend-group` checks the gather.
  Native 18/18, Python 12/12 with both HF models.
- **Done: step 4 gate**, base `56ee2d1`, candidate `18b9c74`, layout
  control `1a05110` perturbing `cpu_backend.hpp` and `arch_qwen.hpp`.
  Three 0.6B cells at 15 pairs, one 8B at 9. Evidence in
  `docs/benchmarks/model-split-20260921/`, raw monitors archived and
  hashed; every sample is kept.

  | cell | phase | candidate mean / median / base wins | control mean / median / base wins |
  |---|---|---|---|
  | 0.6B-1 | prefill | **-2.29% / -2.21% / 12/15 FAIL** | -0.54% / -1.08% / 10/15 |
  | 0.6B-1 | decode  | +0.84% / +0.77% / 5/15 | +0.19% / +0.37% / 6/15 |
  | 0.6B-2 | prefill | -0.66% / -0.70% / 8/15 | +0.27% / +0.40% / 5/15 |
  | 0.6B-2 | decode  | +1.35% / +1.18% / 5/15 | +0.66% / +1.00% / 5/15 |
  | 0.6B-3 | prefill | +1.57% / +2.50% / 5/15 | +3.01% / +0.97% / 5/15 |
  | 0.6B-3 | decode  | +3.03% / +1.45% / 2/15 | +1.68% / +1.88% / 6/15 |
  | 8B | prefill | -0.05% / +0.54% / 4/9 | -0.28% / -0.03% / 5/9 |
  | 8B | decode  | +0.32% / +0.44% / 3/9 | +0.27% / +0.22% / 2/9 |

  One cell failed, 0.6B-1 prefill, on the win count at exactly the
  threshold with mean and median inside the band; the two identical cells
  after it came back at -0.66 and +1.57 with 8 and 5 base wins, and 8B
  prefill is flat. Recorded as noise confirmed by rerun, failing cell kept.
  Decode is positive in every cell, which is the number this step could
  have moved: the decode loop now runs the row-batched graph with one row
  rather than its own path, and it did not cost anything measurable.
- **Done: the Vulkan backend page**, `docs/VULKAN.md`, designed against
  the Radeon VII's reported properties: wave64, 8- and 16-bit storage,
  timeline semaphores behind `submit`/`wait`, push descriptors, a 4-byte
  storage offset alignment that closes the `adopt` alignment question, no
  cooperative matrix so matmul is subgroup dots, and the memory types each
  `Memory` kind maps to. Loader loaded at run time so one binary runs
  without Vulkan; GLSL in the tree compiled by `glslc` at build time and
  embedded. Seven sub-steps with a CTest that skips without a device.
- **Done: step 6, placement**, taken ahead of the Vulkan backend because
  its first test needs no device. `Model` takes several backends and a
  `Placement`, a device per tensor role: each layer's attention and
  feed-forward block, the embedding table and the output head. Each weight
  is adopted by the backend that hosts its role; each device that runs
  attention gets a `KVStorage` for exactly its layers with its own pool,
  block size and adopted RoPE tables; a `Sequence` holds a table per
  storage and a ticket per device, and records the model that made it.
  Wherever the placement changes, the residual stream crosses through the
  context's staging vector, a `read` then a `write`, which gives `write`
  its caller. The `placement` CTest splits a two-layer model over two CPU
  backends so both crossings fall inside a layer and requires the bytes of
  one backend for a prompt, decode steps, a history across a block edge, a
  reset and a two-sequence pass; it counts reads and writes so crossings
  are exactly where the placement changes and absent on one device. No
  flag selects a placement yet: with the CPU as the only backend there is
  nothing to place, so `--device` and `--n-gpu-layers` land with Vulkan.
  Native 19/19, Python 12/12 with both HF models.
- **Done: step 6 gate**, base `6514b17`, candidate `3a5aa7b`, layout
  control `c6743a1`. Three 0.6B cells at 15 pairs, one 8B at 9, all eight
  pass. Evidence in `docs/benchmarks/placement-20260921/`, raw monitors
  archived and hashed; every sample kept.

  | cell | phase | candidate mean / median / base wins | control mean / median / base wins |
  |---|---|---|---|
  | 0.6B-1 | prefill | +2.92% / +3.09% / 4/15 | +0.44% / +0.61% / 7/15 |
  | 0.6B-1 | decode  | -1.16% / +0.08% / 7/15 | -1.05% / -1.28% / 9/15 |
  | 0.6B-2 | prefill | +2.03% / +1.37% / 3/15 | +0.70% / -0.15% / 10/15 |
  | 0.6B-2 | decode  | +0.96% / +0.55% / 6/15 | +1.08% / +0.46% / 6/15 |
  | 0.6B-3 | prefill | +2.61% / +3.05% / 1/15 | +0.46% / +0.97% / 7/15 |
  | 0.6B-3 | decode  | +0.32% / +0.12% / 6/15 | -0.35% / -1.07% / 8/15 |
  | 8B | prefill | +0.28% / +0.67% / 3/9 | +0.51% / +0.92% / 3/9 |
  | 8B | decode  | -0.21% / -0.43% / 5/9 | +0.04% / +0.22% / 4/9 |

  0.6B prefill is two to three percent up in every cell against a control
  under one percent, with base wins of 4, 3 and 1 of 15. Not claimed as a
  win: the single-device path adds a role lookup per layer and nothing it
  could have saved, and the step 2 gate had the same measurement two
  percent the other way on a change of the same character. It is the
  layout band again, this time in the candidate's favour, and the two
  cancel.
- **Left:** the Vulkan backend, sub-step 1 of `docs/VULKAN.md`. This
  workstation has the loader and `vulkaninfo` but not the SDK: `VULKAN_SDK`
  is unset and there is no `glslc`, so the LunarG SDK has to be installed
  before sub-step 1 can build here. Then the placement flags with it.
- **Gotchas:** `sync()` stays `noexcept`; `wait` is too. The other
  developer's last recorded position predates the last five merges to
  main; the design is posted for review but does not wait on it.

## ROCm on Windows is not available for this hardware (2026-09-21)

Checked before planning any GPU work on the workstation. The Windows HIP SDK
supports RDNA3, RDNA3.5 and RDNA4 only, lists no Instinct card, and states
that it does not support gfx906 (Vega 20). gfx906 entered ROCm maintenance
mode in 5.7 and is deprecated on Linux too, where it still runs but AMD no
longer builds for it.

Consequence for the plan, which it confirms rather than changes: the ROCm
backend is developed and validated on the Linux machine. Windows keeps the
CPU backend. If this workstation ever needs GPU acceleration, the route is
Vulkan, already the roadmap's portability target and supported by AMD's
Windows drivers.

## Device execution model complete (ROADMAP #4a) (2026-09-21)

All six steps are merged. Weights and activations are `Buffer` handles, every
op takes a buffer and a float offset, KV blocks are buffers the backend
allocates, and ops enqueue with one sync per forward pass. The model layer
holds no host address and computes no offset into KV storage. Per-step blocks
are deleted per the checkpoint rule; the plan and the step table live in
[DEVICE-EXECUTION](DEVICE-EXECUTION.md) and every gate's evidence is under
`docs/benchmarks/`.

What a vendor backend supplies: `alloc`, `adopt`, `read` and `copy` over its
own allocator, `kv_alloc` and a `KVStorage` it owns, the ops, and a `sync()`
that means it. What it does not have to invent is a place for activations or
a cache layout.

Three things are worth carrying forward rather than rediscovering.

- **`sync()` is `noexcept` by contract.** Three of its four callers are
  exception paths releasing KV blocks, so a sync that threw would replace the
  error that got there and leave the sequence half-released. A backend that
  cannot establish that its work finished must fail hard.
- **A few percent of Qwen3-0.6B prefill means nothing on this tree without a
  control.** The same measurement moved -3 points during the code read and +6
  at step 6, both times from code placement rather than work. AGENTS.md
  carries the rule and the evidence; the short version is that the control
  has to perturb the same file the change does.
- **`Backend::write` was deleted** after step 5 declined to give it a caller.
  Weights arrive through `adopt` and every other value is produced by an op.
  The first backend that genuinely needs a host-to-device write adds it back
  alongside that caller.


## Full code read before the first vendor backend (2026-09-21)

- **Goal:** read every line of `src/`, `tests/` and `tools/` before starting a
  GPU backend, fix what is actually broken, record the rest. Steps 1 to 4 of
  the device execution migration had landed and it was the right point to stop
  and look at the whole tree.
- **Done:** the read. `src/model/kv_cache.hpp` is the only file with nothing
  to report.
- **Done: two real bugs.** `f32_to_f16` OR-ed the rounding carry into the
  exponent field instead of adding it, so any value whose mantissa rounded up
  out of ten bits came back exactly half the right size whenever the exponent
  was odd; 1.999756 encoded to 1.0 rather than 2.0. It reached only
  `llmx quantize`, where it scales a whole block, and nothing had ever
  compared that direction against a reference. `tests/fp16.cpp` now does,
  using binary16 as its own oracle over the whole finite range. Separately,
  `bpe::Tokenizer::decode` indexed the vocabulary with an unchecked id, which
  `llmx detokenize <id>` passes straight from the command line.
- **Done: two hangs and a silent truncation reachable from a model file.**
  A chat template calling `replace`, `count` or `split` with an empty needle
  looped forever, because each advanced its cursor by the needle's length.
  The template is GGUF metadata, so that was untrusted input reaching an
  unkillable loop. A model declaring no EOS id had token zero, an ordinary
  token, treated as the stop token; and a reasoning-start marker with no
  matching end marker suppressed the entire reply.
- **Done: three tests asserted less than they claimed.** Two were coverage
  this project lost to its own activation-arena change and did not notice:
  `prefill-scope` counted 64- and 96-byte allocations that stopped existing
  when nine vectors became one arena, and `kv-cache` aimed an injected
  allocation failure at "the second large batch buffer" for the same reason.
  The third predates it: the tokenizer fixture typed its special token 2,
  which is GGUF's *unknown* rather than *control*, so the special-token path
  its docstring advertises was never exercised, and its unicode case checked
  only an exit code.
- **Done:** the vocabulary vector was zero filled twice per forward pass with
  a `read` overwriting all of it immediately after, an extra 608 KB per
  decoded token on Qwen3-8B; comments across the backend, the CPU backend and
  the model still described the pre-migration interface; `ARCHITECTURE`,
  `ROADMAP`, `DEVICE-EXECUTION` and the backend page said the interface takes
  raw host pointers, which stopped being true at step 4.
- **Done: the gate failed, and the cause was code layout, not the change.**
  Worth reading before the next few-percent argument. The first 0.6B prefill
  cell came in at -4.58% and failed the advance rule. Reruns gave -0.81,
  -2.52, -1.92 and -2.33, so two thirds failed and it was clearly not simple
  noise. A layout control built by appending an unused function to
  `cpu_backend.hpp` moved the same number by only -1.16/+0.63/-1.13/+0.33,
  which made the candidate look like a real regression sitting outside the
  band.

  Bisecting the seven commits found it. The f16 commit alone was clean at
  -0.56% mean. Adding the next commit, a four-line bounds check inside
  `Tokenizer::decode`, produced -3.13/-2.17/-3.09. `Model::prefill` never
  calls `decode`; the prefill timer cannot execute one instruction of it. The
  entire measured difference came from where those four lines pushed the code
  that follows them in the single translation unit.

  Two things follow. The check is now `vocab.at(id)`, one token instead of
  four lines, with the readable message at the CLI call site where the caller
  knows the vocabulary size. And the control has to perturb the same file as
  the change: appending to a different header understated the band by more
  than a factor of two. AGENTS.md carries this now; cells are `06-layout*`,
  `06-fp16*`, `06-tok*`, `06-pre*` and `06-mid*` under
  `docs/benchmarks/code-read-20260921/`.
- **Merged with one performance cell unresolved.** Recorded so nobody later
  reads this as four green cells. Correctness was green throughout: native
  18/18, Python 12/12 with both HF models, 260-token logits byte-identical.
  0.6B prefill came in at -3.87/-1.59/-3.71 across three runs, failing the
  advance rule twice, while 8B prefill gained 3.56% and both decode cells
  were flat. The bisection above is why it merged anyway: the loss tracks
  where the compiler places the code, not work the change added, and the
  clearest single case is a four-line guard in a function that prefill never
  calls moving the same number by three points. If a later step finds 0.6B
  prefill about three points low against an older baseline, this is where it
  went.
- **Left:** nothing blocking.
- **Gotchas:** `Backend::write` and `Backend::copy` are implemented and have
  no caller anywhere, which AGENTS.md forbids. They stay only because step 5
  is their consumer and is next; if step 5 does not use `write`, delete it
  there. `Backend::rope` is the last op taking raw host pointers, and its only
  caller outside the CPU backend is `bench`, which therefore measures a
  function the runtime never calls; `norm_rope_rows` is what the model runs
  and it still takes raw `cos`/`sin` pointers. Moving both is a measured
  change and belongs in its own gated commit, not this one.

  Known and deliberately not fixed: `strip_ws` in the template renderer eats a
  leading unary minus, so `{{ -1 }}` renders `1`; `{% for k, v in x %}` binds
  one variable literally named "k, v" rather than unpacking; `quant/` includes
  `format/gguf.hpp`, which reaches up one layer and is already recorded in
  ARCHITECTURE as a known exception; `quantize_row_q4_1` has no production
  path, since the CLI writes only q8_0 and q4_0; `metadata_u64`,
  `Tokenizer::token_id` and `pad_id` have no callers.

## Code layout moves this benchmark more than the rule allows (2026-09-21)

Moving the embedding gather into the backend measured -8.03% on 0.6B prefill,
twice. The gather itself measures 0.0014 ms at that shape and the logits are
byte-identical, so the work cannot account for 38 ms. Builds that behave
identically, 15 pairs each, against one reference build:

| Build | Prefill | Decode |
|---|---:|---:|
| baseline + unused function | -1.34% | -1.35% |
| baseline + small unused function | -0.50% | -1.18% |
| baseline + larger unused function | +2.52% | -2.00% |
| embed as written | -8.03% | -0.84% |
| embed + that function as dead code | -1.89% | +0.42% |
| embed, gather forced out of line | -3.75% | -0.90% |
| embed, prefill body forced out of line | -4.36% | -1.89% |
| embed, token id in a member slot | -0.60% | -2.21% |

Three things follow, and the second one matters most.

**Layout alone spans about four points of prefill**, wider than the runner's
3% band. The first probe, an unused function and nothing else, **failed the
advance rule** on decode at 12 of 15 baseline wins. The rule can therefore
report a regression for a relink.

**A single-build A/B cannot gate a few-percent change on this cell.** A
candidate needs comparing against several perturbed builds of the same
behaviour, not one. Earlier few-percent prefill conclusions on this model
carry that uncertainty, as does the 0.6B prefill cell's own A/A failure at
-3.53%.

**The -8% build was still real.** It sits outside the layout band, so it was
a genuinely poor layout, not noise: one extra local in a very large inline
body was enough. Forcing functions out of line made it worse; moving the
local to a member slot restored it to -0.60%, inside the band. That is the
shipped form, with the reason in the code.

Root cause is structural. The whole runtime is one translation unit of
headers and the forward pass is one enormous inline body, so any edit can
reshuffle it. Splitting the model layer into its own translation unit would
bound this; AGENTS already allows `.cpp` files with one per logical unit.
Evidence in `docs/benchmarks/layout-sensitivity-20260921/`.

**Then the confound in my own comparison.** The build embeds the Git
revision and a dirty marker. The candidate had been built from a dirty tree
and the baseline from a clean worktree, so the two binaries differed in an
embedded string as well as in code. Rebuilding both the same way, from
detached worktrees at their two commits, the same change reads:

| Cell | Paired mean | Paired median | Baseline wins |
|---|---:|---:|---:|
| 0.6B prefill | +1.09% | -0.08% | 8/15 |
| 0.6B decode | +0.13% | -1.37% | 8/15 |
| 8B prefill | +0.35% | +0.83% | 3/9 |
| 8B decode | +0.51% | +0.46% | 4/9 |

Dead level, against -8.03% when the arms also differed in build identity.
The rule for every comparison from here: **build both arms the same way**,
same source of the version string and same tree state, so the only
difference is the change. An embedded string is enough to move this
benchmark by several points.

## Matched mx gate on a quiet machine (2026-09-21)

The comparison repeated under the rule below, with recorded system CPU of
39% on 8B and 43% on 0.6B against the benchmark's own 37.5%. Eight pairs per
model, alternating arms, llmx from `0c0dec7`, mx `5542318e74`. Evidence in
`docs/benchmarks/kv-mx-quiet-20260921/`.

| Model | Phase | llmx | mx | Paired median | llmx wins |
|---|---|---:|---:|---:|---:|
| 0.6B Q8_0 | prefill | 524.57 | 277.54 | +85.64% | 8/8 |
| 0.6B Q8_0 | decode | 49.09 | 47.77 | +3.91% | 7/8 |
| 8B Q8_0 | prefill | 40.77 | 22.16 | +84.54% | 8/8 |
| 8B Q8_0 | decode | 4.64 | 4.67 | -1.53% | 3/8 |

Three of four cells clear the floor, prefill by a wide margin on both models.
8B decode is 1.53% under it, against a harness A/A of 0.06% on that cell, so
the sign is outside harness noise; but llmx wins 3 of 8 pairs and the
per-pair spread runs -7.5% to +1.2%, so the size is not well determined. The
earlier figure of -3.5% came from the loaded run and overstated it.

This is the current state of ROADMAP #8's performance gate: met everywhere
except 8B decode, which is under by roughly one and a half percent.

## The matched mx comparison ran under background load (2026-09-20 run)

`tools/compare_cpu.py` passes its own A/A: one binary as both arms moves at
most 0.80%, and 0.06% on the 8B decode cell. The harness is sound.

**Corrected cause.** This block first blamed memory contention between the
two arms, because llmx copies the model into the heap while mx maps it. That
was a hypothesis and it is wrong. Two things refute it. The same depression
appears on Qwen3-0.6B, where both processes together need about 1.2 GB on a
32 GB machine, so there is no pressure to have. And the recorded monitors
give the real answer: the comparison ran at 55% mean system CPU against the
benchmark's own 37.5%, while the later A/A runs sat at 36 to 40%. Roughly a
fifth of the machine was doing something else during the comparison, which
is why both arms were slow. Medians, tok/s:

| Condition | 0.6B pp | 0.6B tg | 8B pp | 8B tg |
|---|---:|---:|---:|---:|
| llmx alternating with mx | 464.92 | 43.56 | 37.34 | 3.89 |
| mx alternating with llmx | 259.21 | 41.62 | 20.58 | 4.02 |
| llmx alternating with itself | 548.42 | 50.94 | 41.49 | 4.73 |
| mx alternating with itself | 278.13 | 49.64 | 21.45 | 4.78 |

Both binaries lose about a fifth under that load, llmx 3.89 against 4.73 and
mx 4.02 against 4.78, and no drift appears across pairs, so it is not heat
building up. Pairing is what makes the comparison survive this: both arms
alternate inside the same conditions, so the relative result holds even
though the absolute rates do not represent a quiet machine. The 8B decode
deficit reads -3.2% paired under load against -1.0% unpaired when quiet, and
the unpaired figure compares two runs three hours apart, which is the
comparison this project has been burned by. The honest statement is a
deficit of roughly 1% to 3.5%, still under the floor, smaller than first
reported, and not caused by the KV work.

Prefill is unaffected: llmx is roughly double mx in every condition measured.

What this changes for method: the runner already records activity, but
nothing acted on it. A comparison worth publishing should be repeated when
the recorded system CPU is near the benchmark's own share. Evidence in
`docs/benchmarks/mx-harness-aa-20260921/`.

## A/A calibration of the A/B runner (2026-09-20)

Run after the screening and sampler results below, not before them, which is
the wrong order and is why this block exists. The same binary published as
both arms, `tools/ab_runner.py`, 247-token prompt, 6 threads, evidence in
`docs/benchmarks/aa-calibration-20260920/`.

| Model | Phase | Paired mean | Paired median | Baseline wins | Per-pair spread |
|---|---|---:|---:|---:|---|
| 0.6B Q8_0 | prefill | -2.04% | -3.53% | 7/9 | -8.09% to +8.96% |
| 0.6B Q8_0 | decode | -0.03% | +0.30% | 4/9 | -5.94% to +3.03% |
| 8B Q8_0 | prefill | -0.00% | +0.07% | 2/5 | -4.49% to +2.57% |
| 8B Q8_0 | decode | +1.93% | +2.27% | 2/5 | -2.24% to +5.67% |

**0.6B prefill fails its own A/A**: identical code reports a 3.53% median
loss, past the runner's 3% band. 8B decode moves 2.27% on identical code.
So on this machine today, a single cell below about 4% in those two places
is not evidence either way. What survives: 0.6B decode, where the A/A holds
to 0.30% median.

Consequences, applied to the blocks below rather than left for a reader to
work out:

- The sampler's 0.6B decode gain, +32.04% with 0 of 9 baseline wins, is far
  outside this band and stands.
- The sampler's prefill cells, +2.79% and +2.17%, are inside the band and
  are **withdrawn**; the change has no measured prefill effect.
- The sampler's 8B decode cell was already reported as unresolved and stays
  so.
- In the KV screening, per-cell prefill differences under about 4% carry no
  weight. The rejection of 64 rested on five of eight plans losing prefill,
  two of them by 6.5% and 7.8%, which is a pattern rather than one cell, but
  it is weaker evidence than that table implied. The choice of 128 over 256
  rests on allocated-versus-used bytes, which is counted rather than timed
  and is unaffected.
- The matched mx comparison uses a different harness
  (`tools/compare_cpu.py`, in-process timing) which has **not** been A/A
  calibrated. Its 8B decode cell of -3.5% should be treated as provisional
  until it is.

## Greedy sampling cost (2026-09-20)

Greedy sampling sorted the whole 151936-token vocabulary before reading one
element. `sample` now takes a linear maximum at temperature zero and a
partial sort over the top-k window above it; the repetition penalty is read
through a lambda instead of materialized. Ties now take the lowest token id,
where the unstable sort left them unspecified.

| Paired A/B, generate --temp 0, 247-token prompt, 6 threads | Prefill | Decode |
|---|---:|---:|
| Qwen3-0.6B Q8_0, 9 pairs | +2.79% | +32.04% |
| Qwen3-8B Q8_0, 5 pairs | +2.17% | -0.72% |

The saved work is a constant per token, about 7 ms, so it dominates a 0.6B
token and is inside the runner's noise band on 8B at five pairs, where it is
reported as unresolved rather than as a win. Evidence in
`docs/benchmarks/sampler-greedy-20260920/`. The matched mx gate is unaffected:
that harness times model inference only and excludes sampling.

## Paged KV cache design (2026-09-20)

- **Goal:** replace the single-sequence contiguous `HostKVCache` with a paged
  cache whose logical view (per-sequence block table, positions, refcounts) is
  backend-neutral and whose physical block size and layout are backend-owned,
  so the multi-user server and GPU backends do not inherit a CPU layout.
- **Done:** isolated cold, shuffled paging microbenchmark, seven paired
  repeats, two geometries, three lengths: block 16 costs +25-33% on decode
  attention, 128 costs +5-10%, 256 costs +1-3%. Design in
  [KV-CACHE](KV-CACHE.md), opened for review by the second developer.
- **Done (step 1):** `model/kv_cache.hpp` (`BlockPool`, `KVSequence`),
  backend `kv_layout`/`kv_alloc`/`kv_write` and view-form `attention`,
  `CpuKVStorage` backed on demand, `HostKVCache` and the raw-pointer
  attention removed. The review findings are folded in: each step and each
  prompt is one transaction, bookkeeping vectors are reserved so failure
  paths never allocate, pool and sequence own their ids (non-copyable,
  release on destruction), retain rejects free ids, growth copies into
  exact-size buffers with retained and peak bytes reported, the budget
  crosses the seam in tokens, arithmetic is checked. `generate --verbose`
  prints `kv: allocated/peak/used`. Native suite 17/17; Python suite with
  both HF models passes; main and paged logits byte-identical on 0.6B and
  8B (11 to 841 tokens) and greedy text identical. Provisional block size
  128 behind the temporary `LLMX_KV_BLOCK` knob.
- **Done (screening):** 64/128/256 against contiguous main on 0.6B and 8B
  Q8_0, eight frozen plans, all samples kept, tables in
  [KV-CACHE](KV-CACHE.md). Block fixed at 128; the knob is deleted.
- **Done (mx gate):** matched comparison on the paged runtime, tables in
  [KV-CACHE](KV-CACHE.md). Prefill +79.8% (0.6B) and +82.7% (8B) over mx,
  8/8 pairs; decode -0.2% and -3.5%. The 8B decode cell is under the floor
  and is the existing decode bandwidth item, not a paging cost.
- **Left:** re-review of the final revision; a decision on the 8B
  decode cell; then main integration. Fork/COW and device buffers are later
  steps. F16 KV is out of scope.
- **Gotchas:** the microbenchmark is isolated attention with a cold cache and
  is not an end-to-end decode cost. Memory waste cuts against large blocks:
  224 KiB per token on 0.6B means a partial 256-token tail wastes up to
  55.8 MiB per sequence, 27.8 MiB at 128.

## Native HF download checkpoint (2026-09-20)

ROADMAP #9a is implemented: native `llmx pull`, immutable revision resolution,
verified cache, `HF_TOKEN` credentials, bounded parallel downloads (four streams
by default) and aggregate sharded GGUF loading. This checkpoint is locally
validated and prepared for main publication; hosted CI is still pending.
HF native safetensors/tokenizer/config support and device kernels remain planned.

| Local check | Windows | Linux |
|---|---:|---:|
| Native suite | 16/16 | 15/15 with UBSan |
| Required-HF Python components | 12/12 | 12/12 with UBSan |
| Shard/Unicode fixtures | 66/66 | 66/66 with ASan/UBSan |
| Transport fixtures | 58/58 | 58/58 |
| Sharded HF full-logit cases | 20/20 | 20/20 |

The real 639,446,688-byte public Q8 download passes independent SHA256 after
four-stream assembly. Cache reuse passes on Windows and Linux. Five measured
loader pairs average 427.246 ms before and 428.494 ms after; paired speed change
is -0.283% with a descriptive interval [-3.706%, +3.140%]. This tiny observed
cost is unresolved; all activity-flagged samples are retained. Complete loaded
payload/tensor/metadata hashes match in all 12 processes. No inference arithmetic
changed, and this is not a fresh mx inference comparison. Full evidence and
limitations are in the [checkpoint](ASSETS.md#native-hf-pull-checkpoint-2026-09-20).
All 28 Markdown files were reviewed. Live gated-repository use and controlled
credential redirects remain untested; native credential fixtures and public
HTTPS pass. A maintained curl 8.4+ supplies HTTPS and redirect behavior.

## Prefill placement checkpoint (2026-09-20)

Backend-owned prefill placement is merged and published on both main remotes
at `3c5d4b9`, under the performance tradeoff policy. All five hosted jobs pass in
[run 35516912422](https://github.com/mxxm-t/llmx/actions/runs/35516912422): Windows,
macOS Intel, Linux, Linux UBSan and required HF. The new native targets are
included (12 Windows, 11 Linux/macOS).
On supported Windows topology, six-worker prefill uses separate physical cores
and checks restoration before decode. Other configurations fall back. There is
no runtime affinity flag, NUMA memory policy, arithmetic change or concurrent
submission support. Persistent OS refusal to restore is reported as an error.

| Final source check | Result |
|---|---:|
| Windows native / Linux native | 12/12 / 11/11 |
| Required HF Python suites, Windows / Linux | 11/11 / 11/11 |
| Active-placement HF cases | 28/28 |
| Exact Q8 follow-up pairs | 6/6 |
| F32/Q8 long-continuation vectors | 33/33 each |
| Q8 monitored blocks / processes | 24/24 / 72/72 |
| F32 monitored blocks / processes | 4/4 / 12/12 |
| Frozen identities rechecked | 79/79 in each run |

| Primary model | Paired prefill gain vs main | Paired combined gain vs mx |
|---|---:|---:|
| 0.6B Q8 | +26.40% | +37.49% |
| 8B Q8 | +32.84% | +34.84% |
| 0.6B F32 | +13.70% | +2.86% |

All three measured rounds are retained. Competing developer builds/model jobs
overlapped Q8 despite the reservation and materially limit causal claims.
Observed Q8 follow-up combined means range from -4.39% to +1.44% versus main,
with wide uncertainty; F32 decode is -1.26%, while its combined measurement is
+1.27% versus main. These costs are accepted alongside the primary gains, not
relabelled as zero or used to claim universal parity. Full phase, absolute-time,
activity and uncertainty tables are in the
[final comparison](ASSETS.md#final-prefill-placement-comparison-2026-09-20).

Independent review verified all 84 outputs/vectors and reproduced the statistics
and activity summaries. The single-worker performance smoke passes both arms;
its one-pair numbers are diagnostic. Source `250846a` is unchanged since final
correctness and timing. All 26 Markdown files were reviewed for this
checkpoint. The original working tree and executable remain untouched.

## Runtime base release checkpoint (2026-09-20)

The completed CPU runtime stack is accepted for release under the user's
performance tradeoff policy: large gains may justify smaller costs elsewhere,
with HF correctness and matched mx comparisons retained. This is a scoped
release decision, not a claim of universal per-phase or per-quant superiority.
The optional placement candidate is not part of that published base; its
completed integration is recorded in the checkpoint above.

Release `08351b0` combined runtime checkpoint `0d41a7b` with public main `9511a4a`.
All `src/` files at that reconciliation matched the runtime parent after Git
newline normalization. CI retry/cache handling and plain Windows build-header error
checks are retained. No inference arithmetic changed during reconciliation.
The original Windows working tree, user files and its root executable are not
replaced by this release.

| Final reconciliation check | Windows MSVC | Linux GCC 13.3 / WSL |
|---|---:|---:|
| CMake Release build | Pass | Pass |
| Native CTest | 10/10 | 10/10 |
| Offline downloader cases | 15/15 | 15/15 |
| Python suite with both HF models required | 11/11 | 11/11 |
| Tiny F32 full-logit maximum HF error | 0.00000070 | 0.00000070 |

These are correctness/build checks; simultaneous jobs and diagnostic synthetic
timings do not supply new performance measurements. The rig's HF container
lacked CMake, so the Linux run used the configured WSL toolchain instead.
Independent merge review verifies that native tests, all eleven Python
components, UBSan, required HF fixtures and downloader checks remain wired.
All 25 Markdown files were reviewed for source alignment, ASCII and local links.

## Base release performance decision

The base release's Q8 evidence is the fixed monitored comparison: six workers,
ubatch 128, F32 KV, eight measured rounds per model/workload, with all samples
retained. The table reports paired combined-work speed changes for production
versus pinned mx `5542318e74`; positive means faster. Follow-up prefix loading
is outside the timer. See the [complete phase tables and monitoring limits](ASSETS.md#monitored-prefill-comparison-2026-09-20).

| Model / workload | Combined speed vs mx | Approximate 95% interval |
|---|---:|---:|
| 0.6B Q8, primary 215+32 | +19.69% | +15.87% to +23.50% |
| 0.6B Q8, follow-up 1+32 | -2.58% | -12.82% to +7.67% |
| 0.6B Q8, follow-up 9+32 | +12.46% | -15.74% to +40.66% |
| 8B Q8, primary 215+32 | +24.41% | +10.65% to +38.17% |
| 8B Q8, follow-up 1+32 | -4.72% | -16.01% to +6.58% |
| 8B Q8, follow-up 9+32 | -2.87% | -7.61% to +1.88% |

The primary gains support release while the smaller follow-up costs remain
explicit. Zero-crossing intervals do not prove parity or make the costs zero.
Background imbalance, competing-agent jobs and coarse telemetry limit causal
attribution. No sample was discarded or corrected. An idle PC is not required;
future timing must keep monitoring and must not overlap competing agent work.

The base release used the separate nine-round F32 comparison from the
[ordered-reduction study](ASSETS.md#ordered-prefill-accumulator-reductions-2026-09-20):

| 0.6B F32 phase | llmx tok/s | mx tok/s | Mean-rate difference |
|---|---:|---:|---:|
| Primary prefill | 382.921 | 364.721 | +4.99% |
| Decode | 13.735 | 13.524 | +1.56% |

The base release's backend, inference and quantization sources match that
`bf122fd` study; later model edits validate construction. This is applicable
historical evidence, not a new F32 measurement on the reconciled tree. Older
F32 deficits and the earlier sub-percent Q8 decode veto are superseded as
release blockers by the later evidence and clarified tradeoff policy. Their
original results remain in ASSETS. Other hardware, quants and workloads need
their own measurements; K-quant optimization remains separate work below.

## Status table

| Feature                                  | Status   |
|------------------------------------------|----------|
| Layered restructure                      | Done     |
| Build config (config.hpp + CMake + build.bat) | Done |
| Test suite (roundtrip / perf / tokenizer)| Done     |
| Perf `bench` command                     | Done     |
| CPU backend optimization                 | Done     |
| More quant formats (Q4_0/Q4_1/Q4_K/Q5_K/Q6_K read) | Done |
| More model architectures (Llama, ...)    | Planned  |
| More formats (safetensors, ...)          | Planned  |
| JSON syntax and Unicode validation      | Done |
| GGUF reader size and tensor extent validation | Done |
| JSON quantize tensor validation | Done |
| Qwen model construction validation | Done |
| Paged KV cache (block pool, backend-owned blocks) | Done |
| Device execution model (ROADMAP #4a)     | Done     |
| Execution model: tickets, batched views, placement (`docs/EXECUTION.md`) | Steps 1 to 4 and 6 of 7 done; Vulkan sub-step 1 of 7 done |
| KV cache fork (KV-CACHE step 2)          | Done     |
| Multi-device split (per-layer, per-tensor) | Placement done over CPU backends; flags wait for a device backend |
| GPU backends (Vulkan first to write, ROCm first-class) | Planned |
| Multi-device split (per-layer, per-tensor) | Planned  |
| Multi-node / cluster                     | Planned  |
| Multi-user server                        | Planned  |
| Chat follow-up cache validation          | Done |
| Correctness baseline vs HF reference     | In Progress |
| Pinned HF reference generation           | Done |
| Optional Qwen3-8B HF consumer             | Done |
| HF fixed-excerpt PPL baseline            | Done     |
| Performance floor vs mx-llama.cpp        | In Progress |
| Matched CPU comparison thread selection | Done |
| Perplexity text-file input (-f/--file)    | Done     |
| Chunked corpus perplexity               | Done     |
| F32 embedding/matrix inference          | Done |
| CPU attention in backend (ROADMAP #4a)  | Done |
| CPU row streaming / parallel prefill   | Done |
| CPU attention value accumulation      | Done |
| CPU grouped projections              | Done |
| CPU Q8 scale / load scheduling       | Done |
| Head-major CPU KV storage             | Done |
| CPU worker exception safety           | Done |
| CPU worker cost profile                 | Done |
| CPU ordered prefill reductions          | Done |
| Backend-owned prefill placement | Done (main `3c5d4b9`, five hosted jobs green) |
| CLI thread settings                    | Done |
| Automatic build identification          | Done (main `9511a4a`) |
| Live generation and loading progress     | Done |
| GitHub CPU CI                          | Done     |
| HF fixture download retries and CI cache | Done |
| Hosted numeric/path portability repair | Done (five jobs green at `851d375`) |
| HF model download and sharded GGUF (ROADMAP #9a) | Done (local gates pass; hosted CI pending publication) |
| HF native formats (ROADMAP #9b)          | Planned  |
| HF Hub kernels (additional, after #4a)   | Planned  |

`Done` denotes implemented and validated functionality in this release tree.
The earlier runtime base `08351b0` was published on both main remotes. Its initial five-check
hosted run `35512421834` passed ordinary Ubuntu and required HF, but failed
Windows reference-generator path spelling, UBSan exact scalar-tail comparison,
and macOS JSON subnormal conversion. Repair `851d375` passed all five jobs in
[run 35512954742](https://github.com/mxxm-t/llmx/actions/runs/35512954742):
Windows, macOS Intel, Linux, Linux UBSan and required HF. The repair changes
JSON conversion plus test portability, preserving inference kernels and bounds.
Evidence is in `docs/benchmarks/ci-portability-20260920.json`; prior four-check
passes at `b266650` and `9511a4a` cover those smaller releases.
See [CI](CI.md) for the precise workflow scope and local reproduction commands.

## Active feature blocks

### External floor of merged main (2026-09-20)

Matched against mx `5542318e74`, six threads, eight rounds, identical committed
token IDs in both arms. Runtime is `main` after the device-execution steps and
the fused Q5_K/Q6_K decode dots; the quantized-activation kernels are NOT in it.

| Model / phase | llmx | mx | ratio | |
|---|---:|---:|---:|---|
| Q8_0 prefill | 495.23 | 271.75 | 1.82x | above |
| Q8_0 decode | 46.01 | 44.94 | 1.02x | above |
| Q5_K_M prefill | 408.70 | 242.41 | 1.69x | above |
| Q5_K_M decode | 34.92 | 60.93 | 0.57x | **below** |

Three of four clear the floor, and Q8_0 clears both phases: decode crossed from
0.99x earlier today to 1.02x. **ROADMAP #8 is met for Q8_0 and NOT met
overall** - K-quant decode is the single remaining failure, with mx 1.75x
faster. Absolute rates are higher than this afternoon on both arms because host
load fell; the ratios barely moved (Q5_K decode 0.589x then, 0.573x here).
The 1.02x is inside the range an A/A can produce and should not be leaned on;
the 1.69x-1.82x margins and the 0.57x shortfall are not.
Evidence: `benchmarks/main-external-floor-20260920.json`.


### Scoped correctness coverage and remaining HF work

- **Goal:** keep independent HF ground truth and extend coverage where the roadmap requires it.
- **Done:** exact tokenizer fixtures; tiny tied/untied F32 full logits and NLL; real Q8/Q4 ranking and excerpt PPL; HF/Jinja2 follow-up chat fixtures; pinned reference generation and strict consumers. The unchanged real 8B consumer previously passed 37/37 on Windows and Linux with frozen bounds. Real 0.6B F32/Q8 1,943-token plus 32-step continuation checks are archived in ASSETS.
- **Left:** broader full-corpus, maximum-context and per-layer references, plus prospective numerical bounds for any new lossy kernels. Short 8B rankings/excerpts are not deep-context validation.
- **Gotchas:** self-consistency is supplementary. Exact comparison against another llmx path cannot replace HF. Model construction validation does not establish finite weights, arbitrary token-ID safety, request budgets or failed-session recovery.

### K-quant and device execution work (separate developer branch)

- **Goal:** improve the remaining K-quant decode path and continue ROADMAP #4a without overlapping this release/placement work.
- **Done:** the separate `design/device-execution-model` branch and the K-quant experiments are not incorporated by this release. Their measurements and source identities must be reviewed before adoption. Quantized-activation commits `357d68d` and `97d52e8` fail `backend-group`; they remain isolated and are not merge-ready. Arithmetic-preserving dispatch work is being separated onto a passing base.
- **Left:** prospective correctness/performance validation for any new quantized-activation path. The device execution model is complete on main (see the 2026-09-21 block above); K-quant decode remains. Coordinate rebases and announce timing reservations.
- **Gotchas:** earlier grouped Q16 failed the unchanged native double-dot accuracy contract. Do not reuse it as a lossless baseline or weaken bounds after observing results. CPU Q8 results do not establish K-quant parity.

## Working rules and ownership

One developer owns native HF download/cache and sharded GGUF loading; the other owns its
separate K-quant/device work. Coordination happens in a shared log outside
this repository. Builds and tests
may run in parallel when no timing reservation is active. Keep every planned
performance sample, record ordinary machine activity, and report missing
telemetry honestly. GitHub receives main only; feature checkpoints stay on Gitea.
- **Goal:** reject malformed lengths, dimensions, arithmetic overflow and tensor
  extents before allocating payload storage or reporting loading progress;
  honor the file's declared alignment. This closes the documented format-layer
  error-handling gap and supports future Hub/sharded-format work.
- **Done:** bounded reads, checked size arithmetic and subtraction-based file
  ranges reject malformed input before payload allocation/progress. Reader and
  writer honor positive uint32 alignments divisible by eight, including 24.
  Quantized row widths must contain whole blocks. Array depth is limited to
  256 and tensor rank to four; valid empty tensors retain mathematical size zero.
  Isolated branch `fix/gguf-tensor-extents` starts at `5859762`; placement
  experiments and fixed binaries remain separate.
- **Done:** 122 GGUF cases pass on Windows/Linux: independently constructed
  fixtures plus the writer-alignment round trip. All nine
  native tests and all eleven required-HF suite components pass on both.
  Current-reader ASan+UBSan passes both format/progress tests. The initial full
  Windows build lacked the MSVC include environment; that failure is retained
  and the complete run passes after initializing vcvars64. No source workaround.
- **Done:** the pinned 8B file loads completely with byte-identical `info`
  output against the prior validated control. All six fresh short HF cases
  match top-1 and top-5 overlap 5/5. This is not a new 8B NLL/long-context gate.
  All jobs are terminal. All 25 project Markdown files reviewed and stale
  `info`, quantized-row and loader validation descriptions corrected.
- **Left:** merge with the runtime stack once its separate performance gate
  passes; the root executable and main/GitHub remain unchanged. No inference
  hot path changed and no new performance result is claimed. Evidence:
  [`gguf-reader-validation-20260920.json`](benchmarks/gguf-reader-validation-20260920.json).
- **Gotchas:** token IDs, numerical weight contents, writer validation and future
  request recovery remain separate validation work. The later Qwen construction
  block checks configuration geometry and required tensor layouts.
  JSON conversion dimensions are covered by the later block
  above. Do not
  claim that file-extent checks make arbitrary models executable. Preserve
  nested GGUF arrays within a documented depth limit and valid non-power-of-two
  alignments that are multiples of eight.

### Prefill placement reassessment with machine activity monitoring

- **Method clarification after review:** the frozen runner rotates and
  reverses the three arm orders inside each model/workload block; model and
  workload order also rotate across one warmup and eight measured rounds.
  Report mean/median rates and elapsed time, paired ranges, sample deviation
  and approximate paired 95% intervals, retaining all originals. A third
  contaminated block ends the study as inconclusive; it is not accepted.
  Benchmark and recorder CPU are excluded by PID plus creation time. All arms
  use the same model path/hash for a given model. Activity telemetry does not
  measure DRAM bandwidth or prove absence of short/inaccessible activity;
  sub-percent observer effects also remain unresolved. No clean-preflight
  result alone establishes that a small performance difference is real.
- **Goal:** complete the reopened whole-prefill placement assessment against
  production and matched mx, including HF/lossless and short follow-ups.
- **Done:** preserved the historical candidate and its failed original screen;
  JSON checkpoint `a61c414` passes Windows/Linux correctness suites.
- **Done:** fresh scratch control and default-enabled candidate retain current
  JSON/CLI fixes and pass 8/8 Windows CTests each. The candidate full required-HF
  suite passes 11/11; timings are diagnostic only. Rebuilt callback contracts
  pass 17 cases/4,626 exact values and Windows lifecycle checks pass 27
  cases/1,176 exact values. All 28 active-prefill HF logit cases pass: ten tiny
  F32, six real 0.6B Q8, six real 0.6B F32 and six 8B Q8. Every candidate
  process verifies six applies/restores on distinct target CPUs, no errors or
  leftover restriction, and exact printed logits against current production.
  Tiny F32 maximum HF error is 6.991024018e-7 against the unchanged 2e-5 bound;
  all 18 real-model cases match top-1 and all five top-5 IDs.
- **Done:** the active 1,943-token prefill plus 32 forced continuation steps
  passes for real 0.6B F32 and Q8. Each model's 33 full vectors (5,013,888
  finite floats per arm) are byte-identical to current production, with exact
  per-target NLL. Both candidate processes verify six applies/restores and
  zero decode setters. All 36 HF/prompt/continuation inputs match the earlier
  committed evidence before execution. F32 maximum HF logit error is
  0.000126362 <= 0.001; absolute mean continuation NLL differences are
  0.000000645211 <= 0.0001 (F32) and 0.007011817 <= 0.01 (Q8).
- **Done:** `tools/monitor_windows.py` records timestamped system CPU/disk/GPU
  counters and per-process CPU deltas keyed by PID plus creation time. The
  24-sample controlled-load check detects the known CPU process at a median
  99.995% of one logical CPU; recorder CPU is 0.53125 s over 24.01487 s,
  including initialization. Intervals and query errors remain in the log.
- **Left:** complete prospectively planned matched comparisons and check observer
  effects once a quiet measurement window is available. The driver records
  machine activity before and throughout every matched block; no model timing
  has yet passed its preflight screen.
- **Done:** fresh Linux candidate pass-through build, native 8/8 and full
  required-HF suite 11/11 pass with unchanged snapshot source. Matched mx
  primary/one-token/nine-token continuation harness builds against the pinned
  CPU DLLs. The activity evaluator passes 20 synthetic/known-load checks.
- **In progress:** prospective three-arm comparison plan uses one outer
  warmup round plus eight measured rounds per model/workload, with complete
  matched-block replacement for detected contention (at most two replacements).
  The first preflight defers before any model launch: accessible unrelated CPU
  is 23-44% of one logical CPU and physical disk busy is 29-41% across eleven
  samples. Raw logs are retained; all 56 frozen identities recheck unchanged.
  No performance result or candidate rejection follows. Small observer effects
  remain an explicit unresolved limit.
- **Done:** fresh enabled-candidate optional 8B regression passes 37/37 checks
  against unchanged HF fixture bounds. This includes serial-step NLL coverage;
  active placement is established by the separate callback witnesses, not by
  the serial NLL path. Its timings are not performance evidence.
- **Done:** primary, one-token and nine-token follow-up correctness passes
  for both Q8 models: 12 processes, six byte-identical full-vector pairs and
  911,616 finite floats. All 20 callbacks independently verify six distinct
  physical cores and full restoration (120 applies and 120 restores total),
  with zero decode setters, placement errors or leftover restrictions.
  These fixed-token prefix/suffix checks do not replace interactive chat tests.
- **Checkpoint:** all 25 project Markdown files reviewed; stale status wording
  corrected. Supplemental commands, sources and results are archived in
  [`prefill-contention-followup-20260920.json`](benchmarks/prefill-contention-followup-20260920.json).
  All correctness jobs are terminal. Timing remains deferred; preserve the
  first preflight and use a fresh output directory for the next comparison.
- **Latest preflight:** the next attempt also defers before launching a model:
  sustained unrelated CPU exceeds the unchanged screen. Recorder exit is zero;
  all 58 identities recheck unchanged. A minimally changed runner now requires
  a fresh `--output` path, preserving both attempts and the original runner.
  [Second preflight evidence](benchmarks/prefill-preflight-02-20260920.json).
  Turning off the only recorder would remove during-run contention evidence;
  an extra-recorder sensitivity diagnostic would not prove zero-recorder cost.
  Keep that limit explicit and assess whether such a diagnostic is useful after
  the matched comparison, without creating more measurement infrastructure now.
- **Gotchas:** activity monitoring is evidence, not proof of no interference.
  Keep observer overhead and unavailable counters explicit; no automatic
  adoption or retroactive noise claim follows from the policy change.
  The local check retains 124-126 inaccessible processes as unknown; system
  counters remain available. Controlled disk/GPU saturation and benchmark
  timing perturbation are untested. This continuation check is not full-corpus,
  maximum-context or 8B long-context HF coverage. Production runtime source
  remains unchanged; the candidate is still scratch-only.
  Full 25-file Markdown review and saved checkpoint evidence:
  [`prefill-reassessment-correctness-20260920.json`](benchmarks/prefill-reassessment-correctness-20260920.json).

### JSON validation, Unicode decoding and output escaping

- **Goal:** fix current model-description JSON handling and the documented
  ROADMAP #9b prerequisite for HF metadata/safetensors, without dependencies.
- **Done:** strict number/literal/escape syntax, classic-locale finite-double
  conversion, validated UTF-8 and UTF-16 surrogate-pair decoding, and a
  256-container nesting bound. Value/API and duplicate get-first behavior stay
  unchanged. A small string quoting helper now escapes both dequantize path
  and tensor-name fields, including quotes, backslashes and control bytes.
- **Done:** final Windows/Linux builds and native tests pass 8/8; both full
  required-HF Python suites pass 11/11. Native JSON has 1,033 checks, including
  independent expected bytes, numeric/locale limits and malformed input.
  Linux ASan+UBSan passes the same checks. Actual Q8/Q4 CLI round trips retain
  BMP/supplementary Unicode, quotes, backslashes, newline and source paths.
  Old parser/CLI regressions fail as expected. The initial new-reader Windows
  suite exposed the existing output-escaping bug; its failed result remains
  archived alongside successful final runs. A missing temporary Linux build
  directory caused one later launch to fail before executing any build/test;
  final Linux validation uses a persistent owned build directory.
- **Left:** include the validated fix with the runtime stack when its external
  requirements pass; hosted macOS execution remains unobserved locally.
  All project Markdown is reviewed at this checkpoint. Evidence:
  [`json-validation-20260920.json`](benchmarks/json-validation-20260920.json).
- **Gotchas:** 1,033 counts checks, not independent input documents. Numeric
  storage is double, not exact arbitrary-precision integers. Lone surrogates,
  invalid UTF-8, overflow and nonzero underflow to zero are rejected by policy.
  Quoting expects valid UTF-8. The later JSON conversion block checks tensor
  dimensions and byte extents; the Qwen block validates construction geometry
  and tensor layouts. Token/request checks and general filesystem-path handling
  remain separate. No inference arithmetic changes.
  Suite timings are diagnostic; existing HF/mx merge requirements stay open.

### Native CPU decode sampling (diagnostic complete)

- **Goal:** identify sampled native instruction/function locations during current
  8B decode without adding timers to runtime source. This is a separate
  diagnostic after caller attribution proved no production saving.
- **Done:** optimized PDB harness builds against 17 unchanged runtime files.
  Session 65416 completes the discarded pair, then stops on xperf's unquoted
  commas in C++ symbol fields. A separate parser recovers the saved trace;
  the initial failure and all 41 original identities remain unchanged.
  Independent recovery review pins 51 additional identities. Session 21434
  runs exactly the remaining six invocations and terminates with exit 0.
  All eight planned invocations and 16 finite full vectors pass; vectors are
  byte-identical to the historical unchanged-runtime reference.
- **Done:** independent audit rehashes both manifests, checks collector ownership,
  Running/Stopped states, fixed ready/done holds, and reconstructs all target
  samples directly from saved exports: 39,209 discarded, then 40,115, 39,733
  and 39,571 measured. Every trace has zero lost events/buffers and 100%
  named application-symbol coverage. All unknown/system samples are retained.
  Of 119,419 measured samples, 116,184 (97.291051%) land in `dot_row_impl`.

| 8B phase, mean elapsed ms | Plain | Sampled | Sampled/plain change |
|---|---:|---:|---:|
| Prefill, 215 tokens | 7060.177800 | 7022.348833 | -0.535808% |
| Decode, 32 tokens | 6970.996900 | 7034.000533 | +0.903797% |

- **Done:** exact-binary mapping verifies 345 instructions and all 1,476 code
  bytes against the frozen executable. Actual image bases and PE exception
  ranges resolve every dot sample to an instruction start; padding is excluded.
  All non-dot and caller/other-thread counts remain in the evidence.
- **Left:** external HF/mx requirements remain open. This diagnostic selects no
  production optimization and does not reopen rejected studies. Main/GitHub
  remain unchanged. Full checkpoint evidence and Markdown review are archived
  in [`cpu-native-decode-sampling-20260920.json`](benchmarks/cpu-native-decode-sampling-20260920.json).
- **Gotchas:** three measured pairs, six threads, ubatch 128 and F32 KV; one
  internal warmup per process. Decode paired elapsed changes are +2.915900%,
  -0.611112% and +0.453332%. These combine profiler/handshake/state effects
  and variability, not pure tool overhead. Samples include outside-clock gate
  activity and identify execution locations, not hardware-stall causes or
  elapsed per-operation costs. This exact-vector check supplements earlier
  HF evidence; it does not replace the independent correctness gate.

### Current decode caller-cost attribution (diagnostic complete)

- **Goal:** separate caller work from the previously mixed dispatch residual
  before selecting another optimization. No placement study is reopened.
- **Done:** unchanged production, legacy worker probes and added caller probes
  complete all 12 fixed 8B invocations in session 21493, exit 0. Six threads,
  ubatch 128, F32 KV, 215 prompt plus 32 forced tokens; one outer triplet and
  each process's first iteration are prospective warmups. All 24 full vectors
  are finite and byte-identical to the pinned unchanged-runtime vector. All 93
  frozen identities recheck unchanged. Each of four attributed traces has 32
  steps, 5,792 dispatches and 37,152 caller events, with zero accounting gap.
  Builds, native grouped/error checks, active-probe error/reuse, synthetic
  accounting and independent malformed-trace checks pass. Native Windows
  sampling resolves 6,205 of 6,221 samples to two named C++ test functions
  using local PDBs; this earlier capability checkpoint does not sample a model
  or establish a memory-stall diagnosis. The later model study is above.
- **Finding:** observed means are 217.602291 ms/token in dispatch, 1.972685
  model-side, 0.080984 backend caller work, 0.015842 harness and 0.067954
  explicit caller-observer brackets, plus 0.000014 outer timer fringe.
  Model-side includes serial backend norms/RoPE. SwiGLU is 1.157311 ms/token
  within the model-side total. Attributed decode elapsed is 2.140880% above
  plain and 1.863856% above legacy spans; paired differences against spans
  change sign. Perturbation/variation is comparable to or larger than the
  individual residual regions. Explicit brackets do not capture all observer
  effects, and standalone calibration is not subtracted from model timings.
- **Decision:** no recoverable production saving is proved. Stop the
  outside-kernel optimization direction without probe tuning, a repeat timing
  screen or runtime implementation. Keep all samples. Native function sampling
  is exercised in the separate model study above; none was selected or run
  in this earlier caller-attribution study. External HF/mx gates remain open.
- **Left:** the existing external decode performance gaps remain open. Any
  separate native model sampling needs its own fixed plan, output checks and
  unprofiled control. No production implementation follows this diagnostic.
  Independent terminal audit rechecks all 93 identities, 24 vectors and
  148,608 caller events, reproducing every integer time partition.
- **Gotchas:** the old 2.010823 ms/token residual is historical motivation,
  not a current serial-cost estimate. Region times are instrumented intervals,
  not production savings; exact final vectors are not an external HF gate.
  Production source, tests, build/CI and root executable remain unchanged.
  Full evidence: [`cpu-decode-caller-cost-20260920.json`](benchmarks/cpu-decode-caller-cost-20260920.json).

### Synchronous CPU prefill placement (old screen failed; reassessment reopened)

- **Goal:** test one backend-owned synchronous callback around the complete
  prefill, after the smaller operation-local candidate failed. Keep platform
  details below Model and include setup, callback and checked cleanup costs.
- **Done:** all 36 primary invocations completed with exit 0. Both prefill cases
  pass against fresh production and disabled prototype, with 5/5 wins against
  each. However, 8B decode mean loses 0.573915% against disabled, beyond the
  frozen 0.5% limit. Reject this integration and stop placement adoption.
  Under that original rule, follow-up timing and conditional HF/mx runs did
  not proceed. The user has since requested evaluating large gains against
  minor losses: adoption is reopened for further validation, with this failed
  screen and all samples preserved.
  Windows native checks pass 7/7; callback/tiny-model contracts pass 17 cases
  and 4,626 exact values, plus 27 Windows placement cases and 1,176 exact
  values. Linux unchanged native assertions and callback/no-op guards pass.
  Both one-thread perf smoke floors pass (placement inactive).
  All 18 separate real-model witnesses pass: exact fresh-production vectors
  for initial and one/nine-token continuation cases on both models. Enabled
  processes have 12 applies/restores for primary and 24 for follow-ups across
  two iterations; disabled witnesses and decode have zero setters. Production
  has no instrumentation. All 36 timed vectors match their witness controls.
  Preflight passes 103 checks; 216 identities were frozen before timing.
  Evidence: `docs/benchmarks/cpu-prefill-callback-20260920.json`.
  Independent timing/archive audit and the full 25-file Markdown checkpoint
  review are complete.
- **Left:** fresh active-path HF/lossless checks pass in the reassessment block;
  short-follow-up and fresh matched mx validation remain open under the user's
  clarified tradeoff preference. Retain the
  bounded callback and existing mapping. Report uncertainty for small decode
  differences and gains across phases; the old cutoff failure stays recorded.
  Production scheduling remains unchanged until adoption is validated.
- **Gotchas:** scratch Backend/Model/CPU only; production source, tests, build
  files, public API and root executable are unchanged. Successful placement
  uses one apply and one checked restore pool dispatch. Partial application
  restores all changed participants before unbound fallback. Cleanup retries
  once but reports its first error; persistent refusal does not establish safe
  pool/model reuse. Initial scope is six supported Windows participants,
  mechanically quant-independent, with no wider performance claim. Source
  guards reject nesting/thread changes but do not make Model concurrent.
  Timed policy permits fallback; witness counters prove their own processes.
  Exact vectors are not an independent HF gate. The failed primary screen
  originally prevented the planned short-follow-up performance runs, so their
  timing remains unverified despite passing numerical/activation witnesses.

| Model / phase | Production mean tok/s | Disabled mean tok/s | Enabled mean tok/s | Mean vs production | Mean vs disabled | Median vs disabled |
|---|---:|---:|---:|---:|---:|---:|
| 0.6b / pp | 474.695274 | 472.415155 | 548.326819 | +15.511% | +16.069% | +17.878% |
| 0.6b / tg | 48.835786 | 50.004903 | 50.342125 | +3.084% | +0.674% | +0.222% |
| 8b / pp | 30.392742 | 30.891701 | 41.452967 | +36.391% | +34.188% | +34.854% |
| 8b / tg | 4.628661 | 4.649671 | 4.622986 | -0.123% | -0.574% | +0.004% |

### CPU-local batched-matmul placement (scratch candidate screened out)

- **Goal:** keep placement inside the CPU backend and existing callbacks,
  avoiding a generic Backend phase API or model-level platform code.
- **Done:** all 24 invocations completed with exit 0. Prefill meets the frozen
  >=5% mean/median and 4/5-win screen in both models, but 0.6B decode loses
  2.62% mean and 2.72% median throughput, beyond the 0.5% limit. Reject this
  candidate; no sample removal, rerun or inherited phase-wide result.
  All final vectors match within model and against prior unchanged controls.
  Fresh Windows lifecycle gate passes 41 cases and 11,849 exact values;
  native backend-group passes unchanged 540 cases per arm. Linux native/no-op
  checks and both one-thread perf smoke floors pass. Four real-model witness
  processes verify exact prior outputs, candidate apply/restore counts of
  4,704 for 0.6B and 6,048 for 8B across two iterations, and zero decode/control
  setters. All 130 identities were frozen before timing.
  Evidence: `docs/benchmarks/cpu-matmul-placement-20260920.json`.
  Independent timing audit and full 25-file Markdown checkpoint review pass.
- **Left:** the later whole-prefill callback above is the preferred reopened
  assessment under the user's clarified tradeoff preference. This per-operation
  arm and its failure remain archived. Production is unchanged; its prepared
  active-path HF plan was not run.
- **Gotchas:** six-thread Q8_0 nbatch > 1 scope only; other paths fall back.
  Every eligible operation queries topology and each nonempty callback pays
  apply/checked restore. Timing compares enabled/disabled prototype policy,
  with inert scaffolding in control; it is not the exact production baseline.
  Instrumented witness activation covers its own processes, not every timed
  callback. Allocation failure before dispatch propagates; dual body/restore
  failure can replace the task payload with cleanup failure. Persistent restore
  refusal is reported and only the synthetic test performs manual rescue.
  No production source, public flag, Backend API or root executable changed.

| Model / phase | Disabled mean tok/s | Enabled mean tok/s | Mean change | Median change | Enabled wins |
|---|---:|---:|---:|---:|---:|
| 0.6b / pp | 471.700388 | 504.994000 | +7.06% | +8.73% | 4/5 |
| 0.6b / tg | 49.597335 | 48.299841 | -2.62% | -2.72% | 1/5 |
| 8b / pp | 30.040204 | 40.211011 | +33.86% | +34.11% | 5/5 |
| 8b / tg | 4.502460 | 4.575066 | +1.61% | +1.60% | 5/5 |

### Prefill placement without diagnostic observers (scratch screening passed)

- **Goal:** establish whether the prefill placement benefit survives removal
  of shared observer dispatches before considering production integration.
- **Done:** all 24 invocations completed with exit 0, and the frozen screen
  passes. Both prefill means/medians improve at least 5%, with 5/5 wins per
  model; decode means/medians stay within the 0.5% regression limit.
  Each model's 12 finite final vectors match exactly, with internal warmup
  identity also checked. All 48 source/build/check identities were frozen.
  The new parser accepts seven archived lifecycle records and rejects 34
  corruptions; six rule-boundary checks pass. Unchanged helper lifecycle
  evidence is reused explicitly, not claimed as fresh execution.
  Independent timing audit and review of all 25 Markdown files are complete.
  Full results are in `docs/benchmarks/cpu-prefill-observer-free-20260920.json`.
- **Left:** the CPU-local batched-matmul prototype above is rejected by its
  separate performance screen. The synchronous callback is implemented only
  in scratch and failed its old primary cutoff. Its adoption decision is now
  reopened above; the production Backend API remains unchanged.
  Independent HF and fresh matched mx gates remain required.
- **Gotchas:** scheduler mode constructs no placement Session. Candidate
  construction/apply/prefill/verify/checked restoration are timed; decode
  follows immediately. Stored pre-decode witnesses are serialized afterward,
  and the already-restored destructor performs no pool/affinity call.
  Formatting and inert object disposal are outside timing. There is no
  post-decode mask observation. Six threads, one prompt, Windows only;
  this remains scratch evidence, not production or HF/mx acceptance.

| Model / phase | Scheduler mean tok/s | Prefill-only mean tok/s | Mean change | Median change | Candidate wins |
|---|---:|---:|---:|---:|---:|
| 0.6b / pp | 479.228537 | 529.632147 | +10.52% | +11.02% | 5/5 |
| 0.6b / tg | 49.518222 | 49.633298 | +0.23% | +0.10% | 3/5 |
| 8b / pp | 29.027817 | 40.985134 | +41.19% | +40.29% | 5/5 |
| 8b / tg | 4.581855 | 4.600768 | +0.41% | +0.79% | 4/5 |

### Prefill-only CPU placement (scratch screening passed)

- **Goal:** test the measured prefill opportunity while restoring normal
  scheduling before decode, including recurring placement costs in timing.
- **Done:** the fixed 24-invocation comparison completed with exit 0 and passes
  its frozen screen. Both models improve prefill mean/median by at least 5%
  with 5/5 wins, and decode mean/median stay within the 0.5% regression limit.
  All 12 saved final vectors per model match exactly; each internal warmup
  also matches its measured iteration. Lifecycle tests pass 3,636 exact Q8 value
  comparisons, fresh iterations, exception cleanup and same-pool reuse.
  The parser accepts seven lifecycle records and rejects fifteen corruptions.
  Independent preflight passes; 46 identities were frozen before timing.
  Independent post-run audit and all 25 Markdown checkpoint reviews pass.
  Complete results are in `docs/benchmarks/cpu-prefill-placement-20260920.json`.
- **Left:** the separate observer-free study above passes its frozen screen.
  Broader thread/phase lifecycles, independent HF correctness and
  fresh matched mx performance remain required before adoption.
- **Gotchas:** this is a screening pass, not production readiness. Restoring
  masks does not reset cache, boost or scheduling state. The full candidate
  Session lifecycle, including report serialization and destruction, is timed
  as prefill. Common observation work outside both clocks can influence decode.
  The result covers one 215-token prompt, 32 forced decode steps, six threads,
  ubatch 128, F32 KV and Windows Ryzen 7 5800X only. No HF/mx floor follows.

| Model / phase | Scheduler mean tok/s | Prefill-only mean tok/s | Mean change | Median change | Candidate wins |
|---|---:|---:|---:|---:|---:|
| 0.6b / pp | 467.836623 | 535.466490 | +14.46% | +11.93% | 5/5 |
| 0.6b / tg | 49.293600 | 49.418118 | +0.25% | +0.39% | 3/5 |
| 8b / pp | 29.521998 | 40.948992 | +38.71% | +40.36% | 5/5 |
| 8b / tg | 4.594650 | 4.623768 | +0.63% | -0.30% | 3/5 |

### Explicit CPU worker placement (all-phase candidate screened out)

- **Goal:** compare scheduler-selected placement with six workers on six
  distinct queried physical cores, including caller worker zero, without
  changing kernels or production options.
- **Done:** helper success/unbound, caller/worker failure cleanup, destructor
  restoration and 1,818 exact grouped Q8 value checks pass. Independent
  preflight verifies source fidelity, masks, actual CPU witnesses and cleanup.
  All 24 real-model invocations pass. Fixed-arm before/after snapshots verify
  logical CPUs 0/2/4/6/8/10 from the same queried topology and process mask;
  every original thread mask is restored. Saved finite final vectors match
  byte-for-byte within each model, across both arms and all invocations.

  | Model / phase | Scheduler mean tok/s | Fixed mean tok/s | Mean change | Median change | Fixed wins |
  |---|---:|---:|---:|---:|---:|
  | 0.6B prefill | 479.071849 | 541.704220 | +13.07% | +11.58% | 5/5 |
  | 0.6B decode | 49.314603 | 47.022156 | -4.65% | -5.70% | 0/5 |
  | 8B prefill | 30.183415 | 40.902861 | +35.51% | +35.72% | 5/5 |
  | 8B decode | 4.564495 | 4.649338 | +1.86% | +2.00% | 5/5 |

  The frozen rule requires at least 0.5% higher decode mean and median with
  4/5 wins in both models, and no prefill mean/median regression above 3%.
  Small-model decode fails; the all-phase candidate stays outside production.
  All samples are retained; no new mx or independent HF gate was run.
- **Left:** retain scheduler-selected production behavior. The separate
  prefill-only study above passed its own frozen screen with transition costs
  included; this all-phase candidate remains rejected.
  Matched external decode requirements remain open.
- **Gotchas:** topology and allowed mask are queried in each owned child;
  adjacent CPU IDs are observed, not assumed. Core placement includes serial
  caller work and warmup first-touch, and cannot isolate migration/SMT effects.
  Final-position vectors do not prove full-corpus/deep-context correctness.
  No mapping search, production flag, API or cross-platform affinity promise.
  Evidence: `benchmarks/cpu-worker-placement-20260920.json`.

### Exact decode SwiGLU callback fusion (screened out)

- **Goal:** measure unchanged SwiGLU inside the existing equal-row Q8 gate/up
  worker callback, preserving all buffers, arithmetic and projection order.
- **Done:** MSVC and GCC each pass 1,440 matrix arm comparisons, 126,990
  finite bit comparisons and 27,054 nonfinite classifications. A separate
  instrumented copy witnesses 16,920 rows exactly once; an arithmetic mutant
  compiles and is rejected numerically. Actual unchanged grouped projections
  plus an independent literal SwiGLU expression supply the synthetic reference.
  Assembly confirms original dots and scalar exp/divide/multiply order.
  All 40 fixed timing samples complete with exact gate/up/output checks.

  | Synthetic shape | Serial mean ms/call | Fused mean ms/call | Mean change | Median change | Fused wins |
  |---|---:|---:|---:|---:|---:|
  | Small gate/up + SwiGLU | 0.089504 | 0.085051 | -4.98% | -5.91% | 7/9 |
  | Large gate/up + SwiGLU | 2.442995 | 2.383465 | -2.44% | +0.25% | 7/9 |

  The frozen screen requires lower mean and median in both shapes with at
  least 6/9 wins each, plus at least 2% lower mean and median in one shape.
  The large median fails; no model integration follows. All samples retained.
- **Left:** retain current production behavior; do not weaken the prospective
  screen or repeat this experiment merely to obtain a passing sample.
  Independent HF and matched external decode requirements remain open.
- **Gotchas:** both timed arms use one concrete Q8 helper, not the generic
  production dispatch. Repeated synthetic weights can remain cached. This is
  neither a model slowdown finding nor an HF/external performance result.
  Numerical checks use the standard floating environment and compare within
  each compiler; NaN payload and alternate rounding-mode identity are unclaimed.
  Evidence: `benchmarks/q8-swiglu-fusion-screening-20260920.json`.

### Native Q8 bounded inner-loop follow-up (rejected)

- **Goal:** test one ordinary two-trip loop with local accumulators, preserving
  one copy of the native block body and every arithmetic operation.
- **Done:** MSVC emits two block bodies without accumulator stack traffic.
  Independent instruction review finds 55 F16C-path instructions per pair
  versus 54 for two control iterations. Arithmetic and branch counts are
  unchanged; the extra instruction reloads the feature flag inside the loop.
  No second-block work moves before the first block's final FMA/exit check.
  This fails the useful-scheduling gate; no numerical/model/timing runs follow.
- **Left:** stop this unrolling exploration. Historical and current-source 8B
  whole-operation diagnostics are complete below. Within-group attribution
  remains unresolved; use the matrix-cost findings to select the next study.
- **Gotchas:** instruction counts are not micro-op counts or measured latency.
  No hints, forced inlining, feature specialization or duplicated kernel bodies
  were used. Production remains unchanged. Evidence:
  `benchmarks/q8-bounded-inner-loop-rejection-20260920.json`.

### Native Q8 block scheduling study (rejected at codegen gate)

- **Goal:** expose two consecutive native Q8 blocks to compiler scheduling while
  preserving every weight product, four FMA chains and the original reduction.
- **Done:** isolated control/candidate comparators build with MSVC. Independent
  scalar/control oracles pass 612,267 finite bit checks and 1,939 nonfinite
  classifications on both MSVC and GCC. A repeated-block mutant compiles and
  fails numerically. The two-block lambda preserves the original HADD epilogue,
  block order and odd tail, but MSVC emits eight unconditional 32-byte
  accumulator stack stores per two-block iteration, including the F16C path.
  This fails the planned no-spill codegen gate; the formulation is rejected.
- **Left:** no adoption or model timing for this formulation. Continue the
  external decode performance work from the unchanged production kernel.
- **Gotchas:** this is a codegen rejection, not a measured slowdown. The oracle
  is scoped numerical evidence, not an HF/model validation claim. Prior
  pointer, feature-specialization and integer-unroll studies remain distinct.
  Evidence is in
  `benchmarks/q8-native-block-scheduling-rejection-20260920.json`.

### Exact Q8 horizontal reduction study (rejected)

- **Goal:** reduce Q8 row-dot epilogue cost while preserving all float products,
  FMA chains and contributing addition order.
- **Done:** the scratch shuffle/add sequence emits the intended instructions.
  Native CTest passes 7/7 and required-HF suites pass 11/11 for control/candidate
  on Windows/Linux. MSVC scalar checks pass 612,267 finite bit comparisons and
  1,939 nonfinite classifications; a wrong shuffle is rejected. Windows repeated
  control and candidate match 5,013,888 long-prompt logits and four serial NLL
  cases, within HF bounds. Windows candidate 8B HF checks pass 37/37.
  All 60 fixed timing processes completed: one discarded warmup plus nine
  measured rounds per model/arm, with every sample retained.

  | Mean tok/s | Current | Candidate | mx | Candidate/current | Candidate/mx |
  |---|---:|---:|---:|---:|---:|
  | 0.6B prefill | 473.669 | 453.120 | 277.206 | -4.34% | +63.46% |
  | 0.6B decode | 48.966 | 48.723 | 49.969 | -0.50% | -2.49% |
  | 8B prefill | 29.646 | 29.936 | 21.348 | +0.98% | +40.23% |
  | 8B decode | 4.579 | 4.549 | 4.610 | -0.65% | -1.31% |

  Both decode medians also trail current/mx. Candidate wins only 3/9 decode
  pairs against current and 0/9 against mx on each model. The reduction is
  rejected; production stays at `bf122fd`, and the native regression proposal
  remains unapplied. Validation evidence is in
  `docs/benchmarks/q8-exact-reduction-validation-20260920.json`; completed timing
  is in `docs/benchmarks/q8-exact-reduction-performance-20260920.json`.
- **Left:** close the separate production decode floor. Current control means
  in this session trail mx by 2.01% (0.6B) and 0.67% (8B); do not pool this with
  earlier sessions or revive the rejected epilogue based on selected samples.
- **Gotchas:** the low 0.6B candidate prefill sample (357.111 tok/s) stays in the
  mean. Its median is 471.589 versus current 476.152; do not describe the mean
  gap as a universal causal slowdown. Correctness alone does not justify this
  performance change. NaN payloads and alternate rounding modes remain unclaimed.


### Optional Qwen3-8B HF consumer

- **Goal:** compare the verified local Q8_0 GGUF against the independent pinned
  8B HF tokenizer, logit and PPL goldens without adding large CI downloads.
- **Done:** original HF generation and provenance evidence are committed in
  `bf122fd`. Before any llmx 8B comparison, declare exact tokenizer/input IDs,
  six exact top-1 matches and top-5 set overlap 5/5; all ten printed logits must
  be finite, sorted, unique-token and within absolute magnitude 100. NLL delta
  limits are 0.01 continuous and 0.02 for each windowed case, prospectively
  reusing the existing Q8 quality budget, not calibrated from 8B results.
  Consumer implementation and independent review are complete. Both platforms pass
  all 37 checks: 20 tokenizer cases, six prompt-ID/ranking pairs, PPL IDs and
  four NLL cases. Largest NLL delta is 0.002185355, under its 0.02 limit. The
  full required-HF suite passes 11/11 on each platform, including consumer
  rejection tests (Linux perf is diagnostic only). The official pinned GGUF is
  downloaded and hash-verified in the Linux cache documented in ASSETS. The
  successful Linux gate used an identical staged copy, removed only after its
  tests finished. Interrupted mounted-file results and separate intervention
  metadata are preserved. Copy plus verification took 59.17 seconds; direct
  download plus verification took 486.61 seconds. These are operational I/O
  observations, not inference measurements. Prefer an existing verified copy
  when faster and reuse the completed Linux cache. Windows original untouched.
  All 25 Markdown files reviewed and stale performance wording corrected.
  Evidence: `benchmarks/hf-8b-validation-20260920.json`.
- **Left:** merge the validated consumer with the runtime stack after its
  external performance gates pass. Keep failures with original bounds;
  investigate rather than relaxing thresholds to fit observations.
- **Gotchas:** exact original GGUF conversion revision is undocumented. The
  official model-family link and file hash do not prove identical source
  weights. Fixed excerpt/rank checks do not cover full corpus or all logits.

### CPU ordered prefill reductions

- **Goal:** determine whether the four-row/three-column prefill kernel's
  addressable accumulator array adds avoidable stack traffic or reduction
  overhead, without changing per-lane FMA or final addition order.
- **Done:** explicit ordered reductions match 1,824 scalar-FMA outputs across
  dimension tails and unaligned inputs. MSVC assembly removes most epilogue
  accumulator stack traffic; the FMA loops do not spill in either arm. The
  function grows from 1,119 to 2,931 bytes. Nine alternating rounds improve
  mean Q8/F32 prefill by 4.59%/7.01%, winning 8/9 and 9/9 pairs. A separate
  nine-round Q8 follow-up repeats the prefill gain (+6.25%, 9/9 pairs).
  Q8 decode changes from -1.81% to +0.71% versus control between sessions;
  no stable decode regression or universal external parity is established.
  All control/candidate final-vector hashes match. Windows/Linux native checks
  pass 7/7 and required-HF suites pass 10/10. Linux real-F32 tokenizer/logit/NLL
  checks also pass. Both platforms check each arm against the same scalar-FMA
  oracle; an MSVC mutant swapping final additions is rejected. Fresh long
  comparisons have 5,013,888 byte-identical logits per F32/Q8 model. F32 maximum
  HF error is 0.000126362 <= 0.001; prefilled continuation NLL deltas are
  0.000000645 <= 0.0001 (F32) and 0.007011817 <= 0.01 (Q8). Four separate serial
  NLL cases equal control and pass existing HF bounds. All 25 Markdown files
  reviewed; evidence: `benchmarks/prefill-ordered-reduction-20260920.json`.
- **Left:** retain this validated feature checkpoint on Gitea; merge with the
  runtime stack only when broader external performance requirements pass.
- **Gotchas:** no reassociation or new activation quantization. A synthetic
  gain alone does not establish the external floor. Keep worker implementation
  unchanged and isolate timing from other builds/tests and user inference.

### Pinned HF reference generation

- **Goal:** reuse independent HF tokenizer/logit/PPL generation for explicitly
  pinned models, keeping larger-model fixtures separate from the existing suite.
- **Done:** model/revision/output and associated GGUF labels are explicit;
  numerical loaders share pinned CPU float32 eager execution. Alternate models
  require separate output and cannot overwrite the default fixture directory.
  Windows full required-HF suite passes 10/10 components against the unchanged
  validated 9cfe43f executable; Linux generator safeguards pass 2/2 tests.
  Two offline generations from cached 0.6B HF weights reproduce every numerical
  field. Against committed goldens: 20 tokenizer cases, six top-10 ID lists,
  247 PPL token IDs, four NLL values and synthetic F32 JSON match exactly.
  Rounded top-10 logit values differ by at most 0.0001. Existing fixtures,
  acceptance bounds and default CI model downloads are unchanged. Evidence:
  `benchmarks/hf-reference-tools-20260919.json`.
  Parallel rig work generated actual 8B tokenizer/logit/PPL references from
  verified original `Qwen/Qwen3-8B` at pinned `b968826d9c46dd6066d109eabc6255188de91218`.
  All three modes pass with CPU FP32 eager execution. Measured memory reaches
  the owned container's 40 GiB cap including file cache (3,098 limit events,
  zero OOM/kill). Evidence: `benchmarks/hf-8b-reference-20260920.json`.
- **Left:** merge the tooling with the runtime stack after its external gates
  pass. The separate 8B consumer passes Windows and Linux checks under
  predeclared bounds. Official GGUF metadata links
  the base model and matches the local Q8 digest, but exact original conversion
  revision is undocumented. Default CI downloads remain unchanged.
- **Gotchas:** do not overwrite small-model goldens with another model or expand
  default CI downloads. Tooling support alone is not an 8B correctness result.
  The tooling-only checkpoint changed no hot path and ran after worker timing.
  The parallel 8B HF work ran on the separate rig; it is not a performance gate.

### Separate Q8 scale/payload storage study (screened out)

- **Goal:** test a scratch storage view with original half-scale bytes separate
  from contiguous 32-byte weight blocks, preserving all arithmetic and values.
- **Done:** MSVC and GCC each pass 612,267 finite bit comparisons, 1,939
  nonfinite classifications and 141 packing cases. Grouped/standalone witnesses
  confirm the split dot runs; a corrupted-scale mutant compiles then fails.
  Native assembly preserves the FMA chains/HADD with no loop accumulator spills.
  Fixed synthetic timing completes all six shapes and 96 samples, with exact
  output checks throughout. No samples are dropped.

  | Shape | Original mean ms/call | Split mean ms/call | Mean change | Median change | Split wins |
  |---|---:|---:|---:|---:|---:|
  | Small up | 0.047025 | 0.047276 | +0.53% | -0.51% | 4/7 |
  | Small gate/up | 0.078415 | 0.078916 | +0.64% | +0.69% | 1/7 |
  | Small down | 0.042013 | 0.041720 | -0.70% | -1.84% | 3/7 |
  | Large up | 0.865281 | 0.844761 | -2.37% | -6.07% | 5/7 |
  | Large gate/up | 2.406585 | 2.359071 | -1.97% | -2.83% | 6/7 |
  | Large down | 0.859395 | 0.850048 | -1.09% | -0.20% | 3/7 |

  The frozen advancement rule requires at least 3% mean and median improvement
  with at least 5/7 wins in every large case, and no small-case mean/median
  regression above 3%. It fails; no model integration follows this study.
- **Left:** retain production storage. Revisit only with a distinct hypothesis;
  do not weaken the screening rule or infer a real-model improvement from these
  short cached matrix measurements. External decode requirements remain open.
- **Gotchas:** packing takes 8.96/19.04/9.09 ms in the three large cases and
  adds another weight-sized retained allocation in this diagnostic. Logical
  retained bytes exclude allocator overhead and an additional unpack-validation
  temporary. No peak-RSS, model-loading, HF or external-performance claim.
  Evidence: `benchmarks/q8-split-storage-screening-20260920.json`.

### Current 8B worker-span diagnostic

- **Goal:** measure instrumentation impact on the current 8B runtime and
  attribute decode dispatch intervals to operations before selecting a change.
- **Done:** current source snapshots, plain/instrumented builds and fault check
  pass. All eight fixed processes pass: one discarded outer warmup pair and
  three alternating measured pairs. Every process internally warms up. Saved
  151,936-float final vectors match byte-for-byte across all eight invocations;
  internal warmups use FNV64. Each instrumented trace has 865 prefill and 5,792
  decode records with valid ordered timestamps and no overflow.

  | Phase time, ms | Plain mean | Plain median | Spans mean | Spans median | Mean change |
  |---|---:|---:|---:|---:|---:|
  | Prefill, 215 tokens | 7224.912 | 7215.189 | 7188.779 | 7186.463 | -0.50% |
  | Decode, 32 tokens | 6955.863 | 6934.154 | 6887.725 | 6864.781 | -0.98% |

  Paired changes reverse direction in both phases; no probe speedup is claimed.
  Instrumented decode dispatch is 213.231 ms/token within 215.241 ms/token phase
  time. Gate/up accounts for 46.82% and FFN down for 23.76% of dispatch time;
  all matrix projections total 98.54%, attention 1.46%.
- **Left:** the separate Q8 scale/payload study above missed its screening
  rule. Do not change workers or extrapolate an external performance pass from
  this three-pair diagnostic. No model integration is selected.
- **Gotchas:** all builds/tests finished before timing; all measured samples
  remain. No mx comparison or independent HF gate was run. Last-finisher entry
  can overlap other workers' compute and callback intervals can include
  descheduling. These intervals are not all recoverable overhead. Evidence:
  `benchmarks/current-8b-worker-spans-20260920.json`.

### Archived decode operation attribution

- **Goal:** label the existing 0.6B decode worker spans by operation, using the
  exact archived model/backend source and shapes to prove dispatch order.
- **Done:** all 12 traces and 54,144 decode records map to 141 dispatches per
  token: five per layer across 28 layers, then vocabulary projection. Archived
  source, guards and shapes prove the order. Exact last-finisher decomposition
  passes per operation and sums back to the original dispatch totals. In the
  historical instrumented current Q8 arm, gate/up accounts for 26.07% and
  vocabulary projection for 21.90% of dispatch time. Each of the five layer
  operations changes its control/current delta sign across three pairs.
- **Left:** within-group Q/K/V and gate/up member timing remains unresolved.
  Do not rewrite workers on this evidence. The separate current-source 8B
  plain/spans diagnostic is complete in the block above.
- **Gotchas:** archived current is `9cfe43f`, not a fresh `bf122fd` measurement.
  Model and decode paths are unchanged, but prefill source/compiler layout
  differs. Instrumentation perturbs timings; these are neither external-floor
  results nor evidence of a causal regression. Evidence:
  `benchmarks/worker-decode-operation-attribution-20260920.json`.

### CPU worker cost profile

- **Goal:** localize the remaining prefill/decode costs before selecting another
  hot-path change; compare the pre-error worker control with the retained runtime.
- **Done:** operation-level profiles cover 24 processes: Q8/F32, one/six
  workers, both source arms, three alternating pairs and two instrumented
  sequences after an uninstrumented warmup. Final-vector hashes agree with
  warmups, across source arms and thread counts. The initial instrument double
  counted attention's nested parallel_for; its consistency check rejected the
  run and the corrected probe excludes nested calls. Full samples and sources:
  `benchmarks/cpu-worker-profile-20260919.json`.
  Release definitions, DLL imports and the archived DLL hash establish OpenMP
  workers/barriers in the mx reference. Historical measurements are retained;
  ASSETS and the original artifact now carry a dated interpretation correction.
  The separate per-participant probe is complete: 24 serial Q8/F32 processes,
  six participants, three alternating pairs of plain/instrumented builds for
  both snapshots. Final full vectors match byte-for-byte across every arm and
  mode; all traces have 673 prefill and 4512 decode dispatches with ordered
  timestamps. The instrumented current failure/drain/reuse check passes.
  Evidence: `benchmarks/cpu-worker-spans-20260919.json`.
- **Left:** whole decode operations are now labeled by the verified archived
  dispatch order (see attribution block above). Individual projections inside
  grouped callbacks remain unresolved, and instrumentation changes timing
  materially. No production runtime change is selected.
  Keep the existing worker implementation and the external performance gate;
  do not repeat rejected dispatch variants on this evidence.
- **Gotchas:** instrumentation changes timing. The three-process comparison
  locates costs but does not clear a small regression or prove causality.
  Keep profiles isolated from builds/tests and user inference. Reference source
  may be inspected but cannot be copied into llmx.

| Instrumented phase mean ms, six workers | Before worker fix | Retained runtime |
|---|---:|---:|
| Q8 prefill | 562.43 | 573.38 |
| Q8 decode, 32 steps | 764.09 | 762.44 |
| F32 prefill | 617.72 | 626.08 |
| F32 decode, 32 steps | 2443.77 | 2431.62 |

Q/K/V and gate/up account for 10.57 ms of the 10.95 ms mean Q8 prefill difference
and 7.94 ms of the 8.36 ms F32 difference. These include dispatch/wait time.
Q8 prefill medians reverse the small mean ordering (569.89 vs 565.93 ms), so
this diagnostic does not establish a stable regression magnitude. The next
table is the separate completed span probe; do not pool the two sessions.

| Span-probe phase mean ms | Before plain | Retained plain | Before instrumented | Retained instrumented |
|---|---:|---:|---:|---:|
| Q8 prefill | 632.945 | 567.707 | 591.131 | 574.741 |
| Q8 decode, 32 steps | 803.943 | 762.698 | 779.716 | 783.178 |
| F32 prefill | 638.665 | 627.730 | 645.636 | 635.732 |
| F32 decode, 32 steps | 2463.925 | 2479.028 | 2458.478 | 2562.645 |

Q8 decode's control/current ordering reverses with instrumentation. F32 decode
differs by +0.61% in plain builds but +4.24% in instrumented builds. The exact
last-finisher decomposition separates entry, callback and final completion;
it does not turn overlapping participant spans into additive phase costs.
These results do not identify a stable worker regression or prove its absence.
No mx benchmark or new independent HF gate was run by this scratch probe.

### Live generation and loading progress

- **Goal:** stream generated text immediately in chat/generate, show prompt
  processing before the first token, and make loader progress reusable by
  current CLI consumers and future serving (ROADMAP #7).
- **Done:** optional synchronous loader byte-progress and inference text callbacks
  are implemented; CLI generate/chat owns terminal detection, stderr status and
  stdout flushing. Prompt-processing status appears before prefill. Normal
  Qwen3 output and --think stream before the next model step; legacy retroactive
  filters retain their prior buffered behavior. Stop/EOS and follow-up cache
  accounting remain unchanged.
  Windows/Linux full required-HF suites pass all nine components. After review
  tightened completion ordering, final native checks pass 7/7 on both platforms;
  Linux follow-up chat/progress and version checks pass again. Four MSVC mutants
  fail as intended: delayed delivery, missing flush, ignored read failures and
  completion before a failing trailing seek. Full documentation review covers
  all 25 Markdown files, including current capabilities, CLI defaults and test
  scope. Evidence: `benchmarks/live-generation-20260919.json`.
- **Left:** merge with the enclosing runtime stack only after its external
  performance gates pass. Gitea holds feature checkpoints; GitHub remains
  main-only. Next runtime work should profile the remaining CPU costs.
- **Gotchas:** callbacks are synchronous and do not provide concurrent execution
  or resumable-session recovery. Byte chunks can split UTF-8 characters. Loading
  counts tensor payload bytes, not metadata/padding or model preparation; final
  completion follows every read/seek. The later GGUF validation checkpoint
  checks file extents before progress starts, and the Qwen checkpoint validates
  construction geometry/layouts. Token/request checks remain separate work.
  Legacy filtering buffers text when future markers can
  retroactively discard it; no server framework is added.

| Qwen3-0.6B Q8_0, 64 greedy tokens, median of 3 pairs | Before 8226e17 | Streaming |
|---|---:|---:|
| First visible text from process start (s) | 2.663 | 0.662 |
| Whole process elapsed (s) | 2.738 | 2.741 |
| CLI generation (tok/s) | 31.48 | 31.77 |

Same model/prompt, six CPU workers, stdout pipe and --think; one outer warmup
pair excluded. Every output byte matches. Builds/tests were stopped during
these runs. This is end-user delivery latency, including loading and prefill,
not a kernel-speedup or external mx-llama.cpp parity claim.

### Automatic build identification

- **Goal:** identify each CMake/plain MSVC build by release version plus Git
  revision, with a dirty marker for tracked changes and a clear archive fallback.
- **Done:** CMake refreshes build revision on each build, without rewriting an
  unchanged header. Plain build.bat emits the same metadata. --version and the
  usage banner show release plus Git revision and tracked-dirty state, with
  unknown fallback outside a checkout. Windows plain/CMake clean, dirty,
  new-commit, archive and no-op cases pass in a path containing spaces; Linux
  clean/dirty/new-commit rebuilds and version smoke pass. The current project
  build also reports its actual HEAD plus dirty state. Windows full required-HF suite passes with
  the new version regression. The Windows for/f equals-sign parsing issue was
  caught and fixed before the passing rerun.
  README now separates implemented CPU capabilities from future execution,
  serving and HF goals. All source/comments/docs and new messages use ASCII;
  Unicode fixture data is preserved. The requirement is recorded in AGENTS.
- **Left:** merge with the validated stack after its external gates pass.
  Live generation/progress reached checkpoint 9cfe43f; subsequent CPU cost
  profiles are recorded above. Full evidence for build identification is in
  `benchmarks/build-version-20260919.json`; all 25 Markdown files were reviewed
  for current capabilities, future goals, build behavior and ASCII compliance.
- **Gotchas:** untracked files do not mark a build dirty. Source archives report
  unknown even when nested in another repo. No timestamps, automatic release
  increments, commits/tags or new build/runtime dependencies are introduced.

### CLI thread settings

- **Goal:** make existing auto, decode and prefill thread flags work consistently
  for generate, follow-up chat and bench without changing kernel arithmetic.
- **Done:** generate/chat capture the resolved decode count, apply the prefill
  count and restore decode for every turn. Bench keeps auto selection; actual
  phase counts are visible through verbose/benchmark output. The new regression
  covers 32 generation/chat configurations and four bench cases per platform,
  with 64 HF-golden replies, and rejects all three reintroduced bug mutants.
  Windows full required-HF suite passes. Linux native tests and every correctness
  component pass; its initial automatic-thread synthetic floor fails. Pinning
  that test to its original one-worker conditions passes on both platforms,
  without changing floors. Logs retain the initial failure and the targeted
  reruns. Kernel/model comparator code is byte-identical to c072af2.
- **Left:** merge with the enclosing validated runtime stack once its external
  performance gates pass. Evidence: `benchmarks/cli-threads-20260919.json`.
- **Gotchas:** changing phase counts recreates the CPU pool. Automatic counts
  can be slower for tiny synthetic jobs, particularly under WSL; do not compare
  old serial-default benchmark results with new auto-default results. GPU
  backends will retain CPU-worker meaning for these flags; ubatch remains
  prompt tokens per forward pass. No GPU execution interface was introduced.

### CPU worker exception safety

- **Goal:** propagate CPU task failures after every participant finishes, with
  safe job lifetime and reusable dispatch state; clean up partial pool startup.
  This repairs the existing backend before ROADMAP #4a/#7 execution work.
- **Done:** dispatch catches caller/worker failures and waits for completion
  before rethrowing; startup joins partially created pools. Windows/Linux
  native checks, full suites with required real HF fixtures, Linux UBSan native
  tests and allocation/task fault sweeps pass. The original pool terminates on
  the task and partial-startup regressions. Independent real F32 HF checks pass;
  long F32/Q8 vectors and continuous/window NLL are exact against the control.
  Full samples, hashes, commands, logs and harnesses are archived in
  `docs/benchmarks/worker-errors-cpu-20260919.json`.
- **Left:** investigate the Q8 prefill cost before
  adoption, and close the external Q8 decode floor. No merge. The old root
  executable was retained at that checkpoint; current deployment is recorded above. Paired candidate Q8 prefill loses
  eight of nine rounds despite overlapping ranges; do not dismiss that as noise.
- **Gotchas:** dispatch recovery does not roll back partially written outputs
  or establish Model/session recovery. No concurrent submissions are supported.
  Later GGUF and Qwen checkpoints address file extents, configuration geometry
  and required tensor layouts; token/request checks remain open. Control is
  `3a82284`; no merge. Gitea holds
  the feature checkpoint, while the public/default branch remains unchanged.

- **Post-reboot decision:** retain the existing c072af2 implementation. The
  stored-call variant failed its longer comparison. Failure-only exception
  bookkeeping loses all five paired decode rounds. A shared non-template
  dispatch body improves exploratory prefill but does not improve decode;
  it is not adopted. No more dispatch variants are planned without profiling
  evidence. The user explicitly asked to keep the implementation simple.
  Full samples and source patches for the two latest scratch studies are in
  `docs/benchmarks/worker-cold-errors-cpu-20260919.json` and
  `docs/benchmarks/worker-shared-dispatch-cpu-20260919.json`. Both pass initial
  MSVC allocation/task-fault and grouped-kernel checks; neither entered full
  HF/platform adoption gates. Those experiments changed neither runtime source
  nor the root executable.
  Next performance work should localize the remaining cost before changing code.
  User authorization includes merging main and publishing GitHub once the
  requirements pass; current performance evidence does not clear that gate.

Stored-call experiment is **not adopted**. Full HF, exact vectors/NLL,
Windows/Linux suites and Linux UBSan native checks pass, but the longer run
does not establish a performance gain. Evidence and commands:
`docs/benchmarks/worker-invocation-cpu-20260919.json`.

| Mean tok/s, nine rounds | Before errors | Error checkpoint | Stored call | mx |
|---|---:|---:|---:|---:|
| Q8 prefill | 422.70 | 406.76 | 404.32 | 262.84 |
| Q8 decode | 44.69 | 44.93 | 44.53 | 46.32 |
| F32 prefill | 354.95 | 346.77 | 343.63 | 369.07 |
| F32 decode | 13.44 | 13.43 | 13.30 | 13.33 |

All ranges overlap. Stored-call Q8 prefill loses every pair against the
pre-error control, and no mean beats the error checkpoint. Preserve the early
five-round result as exploratory, not a reason to select the variant.

| Mean tok/s, nine matched rounds | Control | Candidate | mx-llama.cpp |
|---|---:|---:|---:|
| Q8 prefill | 408.39 | 386.18 | 263.80 |
| Q8 decode | 43.88 | 43.94 | 45.81 |
| F32 prefill | 362.85 | 377.11 | 370.26 |
| F32 decode | 13.91 | 14.06 | 13.69 |

Q8 prefill mean changes by -5.44%; Q8 decode remains -4.09% below mx. All
candidate/control ranges overlap; F32 means lead in this session without an
equivalence claim. Separate synthetic timing also has lower prefill/decode
means (see ASSETS). All samples are retained; builds/tests do not overlap timing.

One block per in-flight feature. A block is what lets a fresh agent pick a
feature back up with a "continue feature X" prompt, so keep it current. When the
feature ships, delete its block and mark the row `Done` above.

### F32 embedding/matrix inference

- **Goal:** load and run F32 embeddings and matrices in dense Qwen3, with
  external HF numerical validation and measured CPU performance (ROADMAP #8).
- **Done:** direct F32 rows, float-aligned tensor blobs, deterministic HF
  full-logit/NLL fixtures (tied/untied, odd widths, batches, threads). Windows
  and Linux full suites pass; UBSan passes and catches the pre-fix alignment
  fault. New sanitizer CI job passes workflow lint. All 311 real Qwen3-0.6B
  F32 tensors verified against original HF weights; all real HF checks pass.
  Copy/direct controls match matrix hashes and all logits over a 1,943-token
  prompt plus 32 greedy tokens. Measurements/provenance are in ASSETS.
- **Left:** close the measured F32 CPU gap versus public mx-llama.cpp
  `5542318e74`, then merge and observe the expanded five-job hosted CI.
  Current F32 results are in the KV and worker-error blocks; the older
  scale/load comparison records that earlier checkpoint. Sustained external
  parity remains unproven. Loading is excluded and each process warms up.
- **Findings:** activation tiling and a fully spinning worker pool did not
  establish a win. Profiling instead identifies scalar attention as roughly
  170-180 ms of prefill. Backend attention now improves prefill by 24.8%
  against its interleaved control and passes HF bounds, while changing
  summation order. Remaining work is the CPU performance gap, not numerical
  validation of this attention implementation.
  `tools/compare_cpu.cpp` / `.py` preserve the matched measurement procedure;
  commands and limitations are in ASSETS.
- **Gotchas:** do not treat success on quantized weights widened to F32 as
  parity with the original HF weights. Validate tied/untied output and prefill.

### CPU attention in backend

- **Goal:** move causal GQA attention out of the model and into the backend
  (ROADMAP #4a), sharing the decode and prefill implementation and improving
  CPU throughput with measured, numerically bounded vectorization.
- **Done:** `Backend::attention` now handles both forward paths, with CPU-owned
  score scratch, causal GQA, AVX2 dots and value accumulation, and scalar tails.
  Deleted duplicate model-layer attention loops. Regenerated the tiny HF
  fixture at head width 10 to cover vector tails. Windows and Linux full
  suites pass, including real Q8/Q4 HF checks; Linux UBSan synthetic suite
  passes. Tiny maximum HF logit error is 6.2e-7. A future-token attention
  mutant fails with error 0.03787. Real F32 excerpt/window NLL differs from HF
  by at most 4.5e-6. At 1,943 prompt tokens plus 32 generated tokens, greedy
  output matches HF and all 5,013,888 logits are within 0.001 (max 0.0001252).
  Eight interleaved rounds give F32 prefill 286.84 -> 357.88 tok/s (+24.8%);
  decode 13.34 -> 13.57 tok/s (small change with overlapping ranges).
  Q8 synthetic guardrails pass; matmul 123.07 -> 121.80 GFLOPS, prefill
  4828 -> 4800 and decode 4522 -> 4875 tok/s, with overlapping ranges.
- **Left:** close the remaining external CPU floor gap, then merge and run
  hosted CI. See the KV and worker-error blocks for later measurements.
  Profiling after vectorization finds prefill attention around 60-67 ms and
  decode attention around 97-100 ms; matrix operations now dominate decode.
  Sequential F32 row streaming and parallel batched elementwise work are
  implemented on the next feature branch; see its validation block below.
- **Gotchas:** SIMD dot reductions change summation order. A matching top token
  or logit sum is not a numerical gate. This step does not implement device
  buffers, resident activations or async execution, which remain prerequisites
  for GPU backends.

### CPU row streaming / parallel prefill

- **Goal:** close the F32 CPU performance gap without changing weights or
  weakening the HF numerical gate (ROADMAP #8).
- **Done:** contiguous single-row F32 decode and parallel batched
  norm/RoPE/SiLU work, guarded to keep fewer than two rows per worker serial.
  Final eight interleaved rounds: prefill 371.65 -> 383.06 tok/s (+3.1%,
  overlapping ranges), decode 13.87 -> 14.68 (+5.8%, disjoint ranges).
  Same-run mx is 398.97 / 14.89: both floors remain open (-4.0% / -1.4%).
  Windows and Linux full suites pass with required Q8/Q4 HF fixtures; UBSan
  synthetic suite passes. Real F32 HF logits/excerpt/chunked NLL pass.
  Long-prompt full logits have maximum error 0.00012636 under 0.001, with
  all greedy IDs matching HF. Q8 step guard passes with overlapping ranges.
  The batched guard caught tiny-prompt scheduling regressions; after the
  serial guard, small-batch ranges overlap the control and B=64 retains a
  17% latency improvement. ASSETS and the row-scheduling JSON contain all
  initial/final samples and validation scope.
- **Left:** close both remaining external gaps before merge and hosted CI.
  Instrumented pool profiling finds 26-28 ms after worker callbacks finish
  during 32 decode steps. Bounded completion polling did not establish a win
  over its atomic-only control (eight rounds); no pool change was adopted.
  Full spinning was already rejected. Query scheduling remains scratch-only;
  the value-accumulation kernel below is the next validated checkpoint.
- **Gotchas:** single-row dots change reduction order. Printed sums alone
  are not the numerical gate. Initial unconditional scheduling appeared above
  mx for prefill, but final guarded measurements did not; use the final run.
  The legacy bench "prefill" is repeated step(), not batched prefill, so keep
  the separate batched regression guard. Reader alignment was not adopted.

### CPU attention value accumulation

- **Goal:** reduce CPU attention output loads/stores by retaining value-sum
  lanes in SIMD registers, while preserving per-lane summation order (#4a/#8).
- **Done:** normalized coefficients and register value sums implemented, with
  32-lane blocks, eight-lane remainders and scalar tails. Expanded the tiny HF
  fixture to head width 42, deriving its tensor shapes and HF config together.
  SIMD and forced scalar-value branches pass at maximum HF error 0.00000070
  under 0.00002; a missing SIMD-block output mutant fails at 0.19888665.
  Windows/Linux full suites pass with required real Q8/Q4 fixtures; Linux
  UBSan synthetic suite passes. Real F32 HF logits and excerpt/window NLL pass.
  All 5,013,888 logits over 1,943 prompt plus 32 greedy tokens are byte-identical
  to c4436fd; HF maximum error 0.00012636 under 0.001, greedy IDs 32/32 exact.
  Q8 step and batched guards show overlapping control/candidate ranges.
  Final eight interleaved rounds:

  | Mean tok/s | Control c4436fd | Value kernel | mx | Gap vs mx |
  |---|---:|---:|---:|---:|
  | Prefill | 384.86 | 397.63 | 397.94 | -0.08% |
  | Decode | 14.56 | 14.57 | 14.89 | -2.10% |

  Prefill improves 3.32% with narrowly overlapping ranges; decode is unchanged
  within noise. ASSETS and `benchmarks/attention-values-cpu-20260919.json`
  retain all samples, hashes, validation logs and reproduction harnesses.
- **Left:** close the external performance gap before merge/hosted CI.
  Instrumented profiling attributes about 2,073 of 2,187 ms decode to matmul,
  including 528 ms in the output projection; attention is about 87 ms.
  Investigate matrix operations next, using matched controls and HF gates.
- **Gotchas:** means close to mx are not proof of parity. This is an interactive
  workstation and no outliers were discarded. Query scheduling and polling
  remain scratch-only. The value kernel keeps sequence order per lane; byte
  identity is established for the recorded Windows long-prompt case, not all
  inputs or compilers. Forced scalar values do not prove no-AVX ISA support.

### CPU grouped projections

- **Goal:** reduce worker-pool dispatches for Q/K/V and FFN gate/up projections
  sharing activations, preserving float arithmetic (ROADMAP #4a/#8).
- **Done:** direct grouped decode through existing F32/Q8_0/Q4_K row kernels.
  Other types, batches and small jobs use the sequential fallback. No TLS,
  nested dispatch or activation conversion. Added CTest coverage to CI.
  Windows/Linux full suites pass with required real Q8/Q4 HF fixtures;
  real F32 HF and UBSan synthetic/backend checks pass. Backend tests cover
  540 cases and 141,750 outputs per platform; a missing-row mutant fails.
  Instrumented Q8 run observes 3,584 groups avoiding 5,376 dispatches across
  warmup plus measured sequences. Four excerpt/window NLL cases per model
  match the previous runtime exactly. All 5,013,888 long-prompt logits per
  model (F32 and Q8) are byte-identical to the previous runtime. F32 remains
  within the independent HF bound, with identical greedy continuation.
  Final eight interleaved rounds after a discarded warmup round:

  | Mean tok/s | Previous 5a9518c | Grouped | mx | Gap vs mx |
  |---|---:|---:|---:|---:|
  | Q8_0 prefill | 418.59 | 419.06 | 267.76 | +56.51% |
  | Q8_0 decode | 43.92 | 45.18 | 48.34 | -6.55% |
  | F32 prefill | 397.41 | 394.25 | 390.07 | +1.07% |
  | F32 decode | 14.47 | 14.64 | 14.86 | -1.49% |

  Q8 decode mean/median improve 2.86%/3.19%; ranges overlap due to one slower
  candidate sample, retained in the result. F32 prefill/decode changes are
  within overlapping ranges. Synthetic matmul mean 129.57 -> 127.64 GFLOPS
  (median 129.78 -> 129.51, ranges overlap); step-based prefill/decode improve
  4998/4928 -> 6861/6921 tok/s. Batched guards retain overlapping ranges except
  the faster six-thread single-token case. Raw samples, hashes, exact-output
  gates and reproduction sources are in ASSETS and
  `benchmarks/grouped-projections-cpu-20260919.json`.
  The follow-up real 8B Q8_0 diagnostic also preserves all 5,013,888 logits
  on its matched 215+32-token history. Three measured rounds, after warmup:

  | 8B mean tok/s | Previous | Grouped | mx |
  |---|---:|---:|---:|
  | Prefill | 29.12 | 29.25 | 21.20 |
  | Decode | 4.31 | 4.33 | 4.49 |

  Control/group ranges overlap; this does not establish a small speedup.
  Decode remains below mx. Exact-vector equality is against previous llmx,
  not an independent 8B HF baseline. Evidence and the full-vector control are
  in `benchmarks/grouped-projections-8b-20260919.json`.
- **Left:** close the external decode floors before merge and hosted CI.
  Native Q8 scale/load scheduling is implemented and validated below.
- **Gotchas:** exact equality is scoped to tested inputs/platform, not a
  full-corpus or maximum-context proof. The legacy bench prefill uses step();
  batched prefill has a separate guard. All outliers retained; do not pool
  absolute rates from separate sessions. Candidate is validated but unmerged.

### CPU Q8 scale / load scheduling

- **Goal:** reduce native Q8 decode instruction overhead while preserving
  float activations, per-lane accumulation order and exact weight scales (#8).
- **Done:** selected direct memory half broadcast plus direct byte-load sign
  extension. Assembly confirms the intended instructions; scalar fallback and
  accumulator order are unchanged. Feature specialization did not establish a
  further gain and is excluded. Exhaustive finite-half scale/signed-weight
  regression passes, and a wrong-half-offset mutant fails. Windows/Linux full
  suites with required Q8/Q4 HF fixtures, real F32 HF checks, backend CTest and
  UBSan synthetic/backend checks pass. Instrumentation confirms real Q8 use.
  Recorded F32/Q8 long vectors and excerpt/window NLL are exactly unchanged
  from `b6a890f`; the same-weight 8B vector comparison also passes.

  | Eight-round mean tok/s | Before b6a890f | Candidate | mx |
  |---|---:|---:|---:|
  | Qwen3-0.6B Q8 prefill | 422.99 | 418.43 | 273.51 |
  | Qwen3-0.6B Q8 decode | 45.30 | 46.54 | 48.35 |
  | Qwen3-0.6B F32 prefill | 398.17 | 395.23 | 394.92 |
  | Qwen3-0.6B F32 decode | 14.70 | 14.77 | 14.98 |

  Q8 prefill and F32 ranges overlap. Q8 decode improves but remains below mx.
  Root CLI and production comparator code hashes match their validated arms.
  The larger-model diagnostic also improves decode but does not establish
  external parity:

  | Qwen3-8B Q8 mean tok/s | Before | Candidate | mx |
  |---|---:|---:|---:|
  | Prefill | 28.97 | 29.04 | 21.13 |
  | Decode | 4.29 | 4.45 | 4.49 |

  The short synthetic guard's threaded single-token regression does not
  retain disjoint ranges in the longer follow-up. Paired changes run in both
  directions; the higher candidate mean remains visible:

  | Synthetic prefill mean ms | Before | Candidate |
  |---|---:|---:|
  | 1 thread, 1 token | 0.16995 | 0.12580 |
  | 6 threads, 1 token | 0.15864 | 0.16167 |

  ASSETS and `benchmarks/q8-scale-load-cpu-20260919.json` retain all samples,
  controls, hashes, HF/platform logs, prototype patches and reproduction code.
- **Left:** close the external decode floors before merge and hosted CI.
  Continue matrix-operation work with matched model measurements; retain the
  threaded tiny-prompt guard because zero slowdown has not been established.
- **Gotchas:** retain F16 subnormal/negative-scale behavior and scalar fallback;
  do not multiply activations by scales instead, which changes rounding.
  Exact equality is scoped to recorded inputs/platforms. The 8B equality check
  is against previous llmx, not an independent HF 8B reference. All samples
  are retained; no external parity claim.

### CPU comparison thread scaling

- **Goal:** locate the remaining CPU performance gap using matched thread
  counts and matrix-shape measurements, following ROADMAP #8.
- **Done:** explicit `--threads` in the comparator and runner, with requested
  counts echoed and checked. Windows llmx/mx and Linux llmx builds pass;
  invalid arguments and missing/wrong thread metadata are rejected. A real-model
  runner smoke passes. Matched Q8/F32 scaling is complete; those measurements
  used runtime `475f312`, before the KV and worker-error changes.
  Full vectors are byte-identical across all measured counts; continuous NLL
  is unchanged from the prior runtime and passes the independent HF fixture.
  Projection and matrix probes are complete. F32 matrix ranges overlap mx;
  Q8 matrix latency remains higher at the default comparison count. ASSETS and
  `benchmarks/cpu-thread-scaling-20260919.json` contain complete results.
- **Left:** merge the tool with the validated stack once its external floor
  is met. The resulting contiguous per-head KV change is implemented and
  validated below; it is no longer a pending experiment.
- **Gotchas:** use the same thread count in both arms and record it with every
  result. Keep the pinned model, tokens, reference revision, warmup and KV
  settings. Scaling diagnostics do not waive the existing external floor.

### Head-major CPU KV storage

- **Goal:** make each KV head's history contiguous to improve attention reads,
  preserve arithmetic order, and separate concrete CPU storage from logical
  sequence state without adding speculative device/server interfaces.
- **Done:** `HostKVCache` owns bounded growth and token-major projection writes;
  `Model` owns valid length/reset; backend attention receives an explicit head
  stride. Promoted the exact validated headers from the scratch candidate.
  Growth/reset/mixed histories match all 229,758 control values on Windows
  and Linux (the latter with nonrecovering UBSan). The direct storage oracle
  is in CTest and rejects a wrong-head relocation mutant.
  Real F32 HF logits and continuous/window NLL pass; long HF error is at most
  0.00012636185 under 0.001, with all 32 greedy IDs matching. F32 and Q8 each
  retain all 5,013,888 long-history logits and four full-precision NLL cases
  exactly versus `475f312`. Windows/Linux full suites with required real HF
  fixtures and Linux UBSan native/synthetic suites pass. Integrated MSVC code
  has the same `.text` hash as the validated candidate; final native CTest
  integration passes on Linux and UBSan.
  Nine interleaved matched rounds, all outliers retained:

  | Mean tok/s | Control | Head-major KV | mx | Gap vs mx |
  |---|---:|---:|---:|---:|
  | Q8 prefill | 366.46 | 416.37 | 262.34 | +58.72% |
  | Q8 decode | 42.36 | 44.41 | 45.51 | -2.42% |
  | F32 prefill | 345.42 | 361.38 | 362.44 | -0.29% |
  | F32 decode | 13.06 | 13.47 | 13.27 | +1.55% |

  Candidate/control ranges overlap; paired candidate wins are 9/9, 7/9,
  7/9 and 8/9 respectively. These results support an incremental selection,
  not a claim that the external floors are closed. The separate tiny synthetic
  decode mean declines 2.88% with overlapping ranges; that remains a recorded
  limitation. ASSETS and `benchmarks/head-major-kv-cpu-20260919.json` preserve
  all samples, hashes, scopes and reproduction sources. Earlier short-run
  diagnostics remain in `head-major-kv-initial-20260919.json`.
- **Left:** close the remaining external Q8 decode and F32 prefill gaps, then
  merge the validated stack and observe hosted CI. No merge or publish yet.
- **Gotchas:** reset retains allocation but must not expose stale tokens.
  Growth temporarily holds old and replacement storage together; future
  multi-user memory budgets must account for that peak. A CPU head-major layout
  is not a requirement for future device buffers, paging or shared prefixes.

### Chat follow-up cache validation

- **Goal:** preserve correct conversation history across follow-up prompts,
  reusing KV only when its exact token prefix matches the rendered transcript.
- **Done:** review found that `cmd_chat` skips cached tokens by count alone,
  renders twice per turn, and can pass empty logits to generation when the
  template does not add a generation suffix. Generated stop tokens and the
  unconditional EOS step also need accurate cache accounting.
  The new HF-backed chat regression reproduced a crash on the old binary.
  It also exposed double consumption of template block terminators, which
  skips adjacent content and breaks nested conditionals/loops.
  Implemented exact fed-token tracking and reset/refill for changed prefixes;
  removed unconditional EOS insertion and the redundant prefill/render pass.
  Fixed block terminator consumption and first keyword argument parsing. The
  latter affected Qwen namespace state and removal of old reasoning. New
  end-to-end HF reply fixtures pass; a token-count-only reuse mutant fails.
  The real Qwen template matches Jinja2 across initial and follow-up histories.
  Windows and Linux full suites with required real HF fixtures pass. MSVC
  renderer test, Linux CTest and Linux UBSan native/synthetic suites pass.
  Linux first exposed a missing `<cmath>` include in the standalone renderer;
  that is fixed. Chat coverage is 54 runs of nine scenarios across thread and
  batch settings; the real template has 12 independent Jinja2 cases. Logs,
  fixture hashes and reproduction commands are in
  `benchmarks/chat-followup-validation-20260919.json`.
- **Left:** merge with a validated runtime stack and observe hosted CI. The
  enclosing stack still needs its external performance floors; chat fixes do
  not establish those floors. Numerical kernels and model forward paths are
  unchanged by this fix.
- **Gotchas:** prefill continuation tests alone do not validate chat-template
  reuse. Keep messages separate from cached tokens; rendered text may change
  earlier turns or retokenize their boundary. This is single-user chat, not
  multi-user or concurrent request support.

### Correctness baseline vs HF reference

- **Goal:** give the suite an external ground truth. Correctness is measured
  against the HF reference, never against llmx itself (`docs/ROADMAP.md` #8).
- **Done:**
  - The tokenizer mode of `tools/gen_baseline.py` emits golden fixtures using
    `tokenizers` + `huggingface_hub`; numerical modes additionally need torch
    and transformers. Tokenizer output is committed to
    `tests/data/baseline_tokenizer.json`.
  - `tests/baseline.py` compares llmx against the committed goldens and is
    wired into `tests/run_tests.py`. It SKIPS when no fixture model is on disk,
    so the rest of the suite still runs anywhere.
  - Tokenizer parity: **20/20 cases match pinned Qwen/Qwen3-0.6B.**
  - HF fp32 top-10 logit goldens for six prompts landed in `1c2e102`.
    The baseline checks top-1, top-5 set overlap and a magnitude bound on
    Qwen3-0.6B Q8_0 and mixed Q4_0 fixtures. Both passed all six prompts in
    that commit; the mixed fixture catches the f16 subnormal regression.
  - Local golden generation works in an isolated environment with
    numpy<2.3, torch 2.5.1+cpu and transformers 4.55.2.
  - Pinned HF fp32 PPL golden for a 247-token wikitext excerpt: exact token
    IDs/count and finite NLL/PPL checks, with per-quant absolute NLL bounds.
    HF PPL 28.7974; Q8_0 28.8371 (NLL delta 0.001374 <= 0.01); mixed Q4_0
    32.8463 (delta 0.131554 <= 0.16). Both llmx arms repeated identically at
    printed precision. Generator, provenance and bounds are in `docs/ASSETS.md`.
    Eight injected bad-output cases were rejected; build and full suite pass.
  - Nine bugs found and fixed via this path, all of which survived a green
    suite: attention missing 1/sqrt(head_dim); temperature cancelling in the
    sampler; RoPE read past context_length; the GPT-2 whitespace guard that
    could never fire; attention width hardcoded to n_embd; tied embeddings
    unsupported; Windows argv delivered in the ANSI codepage so any non-ASCII
    prompt was mangled before llmx saw it; and the pretokenizer implementing
    the GPT-2 regex instead of the Qwen2/Qwen3 one; the byte encoder incorrectly
    including soft-hyphen byte 0xAD in its printable set. Four new HF cases
    reject the preserved old encoder. Whole-wikitext file input now tokenizes
    298,938 tokens and scores the requested window limit successfully.
- **Left:**
  - Per-layer activation and full-corpus PPL goldens, plus maximum-context validation.
    The 1,943-token/32-step independent F32 HF check is complete; its scope and
    full-vector quantized diagnostics are recorded in ASSETS.
    Continuous and chunked excerpt gates are implemented, with an explicit
    disjoint-window scoring policy (`docs/USAGE.md`). They are not full-corpus
    coverage. Two-token windows show large quantized/HF deviations even under
    the unchanged old scorer; diagnostics and bounds are in `docs/ASSETS.md`.
    The existing ranking gate does not bound full-vector numerical error.
  - Existing tiny F32 full-vector and real-model excerpt bounds are implemented.
    Broader Q8 full-vector and full-corpus acceptance bounds remain open.
- **Gotchas:**
  - Round-trip and synthetic tests alone are not an external correctness gate;
    preserve the HF tokenizer/logit checks and extend their coverage.
  - The 8B hides bugs the 0.6B exposes: `n_head * head_dim == n_embd` holds for
    Qwen3-8B (32*128 == 4096) and fails for 0.6B/1.7B/4B.
  - torch is a fixture-GENERATION dependency only, never needed to run the
    suite and never at runtime.
  - F32 embedding/matrix support and a tight HF numerical gate are implemented
    on the active feature branch. Its external performance gate remains open;
    see the F32 block above before merging.
  - Generation's legacy reasoning filter searches `thinking_start/end`, not
    Qwen3's actual `<think>` / `</think>` markers. Its docs now state that limit.
  - The CLI thread-settings block records the validated auto/prefill/decode
    corrections found during the documentation review.
  - JSON syntax/Unicode and conversion tensor dimensions/extents are implemented
    in their active blocks above. Qwen construction now validates configuration
    geometry and required tensor layouts; token/request checks remain separate. See
    docs/src/core-json.md.
  - Qwen3 does NOT use the GPT-2 pretokenizer regex. Read the Split pattern out
    of `tokenizer.json` before touching `pretokenize`.

### Performance floor vs mx-llama.cpp

- **Goal:** llmx must be at least as fast as mx-llama.cpp on the same model,
  quant, prompt and hardware (`docs/ROADMAP.md` #8), pp and tg both reported.
- **Preceding two-arm result (2026-09-20):** unchanged validated `bf122fd`
  runtime versus pinned mx `5542318e74`, nine alternating measured pairs per
  model after one discarded warmup pair; six workers, ubatch 128, 215 prompt
  plus 32 forced tokens and F32 KV. Every process exits zero and llmx final
  output hashes repeat within each model. No timing overlaps other compute.

  | Q8 model / phase | llmx mean tok/s | mx mean tok/s | Mean gap | llmx / mx median | Paired wins |
  |---|---:|---:|---:|---:|---:|
  | 0.6B prefill | 440.479 | 274.766 | +60.31% | 448.159 / 277.473 | 9/9 |
  | 0.6B decode | 47.280 | 47.589 | -0.65% | 47.362 / 48.315 | 3/9 |
  | 8B prefill | 30.170 | 22.124 | +36.37% | 30.481 / 21.994 | 9/9 |
  | 8B decode | 4.403 | 4.440 | -0.82% | 4.465 / 4.501 | 2/9 |

  Both decode means and medians remain below mx. The external gate remains
  open; close results do not meet the required floor. Preserve all measured
  rounds, including the slower first measured round, and do not pool older
  sessions. Evidence: `benchmarks/q8-current-floor-20260920.json`.
  Independent review of all 27 pending commits found no additional concrete
  publication blocker; the existing Windows/Linux HF/native evidence remains
  valid for this unchanged runtime. Hosted CI follows eventual publication.
  Earlier ordered-prefill comparisons also exceed mx in F32 prefill/decode;
  their initial and Q8 follow-up sessions retain their separate scope.
- **Earlier instruction study (no runtime change):** paired native Q8 rows regress;
  direct pointer increments and explicit row-kernel inlining do not establish
  a decode gain. Exact row/tail/fallback checks pass. Assembly confirms shared
  activation loads without inner-loop spills, removal of native-loop address
  multiplication, and removal of row calls in the respective prototypes.

  | Separate Qwen3-0.6B Q8 studies, mean decode tok/s | Control | Prototype | mx |
  |---|---:|---:|---:|
  | Paired rows | 46.00 | 43.81 | 48.31 |
  | Pointer increments, longer run | 46.22 | 45.86 | 48.22 |
  | Explicit inlining | 46.56 | 46.04 | 48.46 |
  | Inlining plus pointers | 46.56 | 46.38 | 48.46 |

  No prototype is adopted. Do not pool absolute rates across these sessions.
  ASSETS and `benchmarks/q8-row-instructions-20260919.json` retain patches,
  assembly, exact checks, complete samples, code hashes and reproduction.
  These scratch prototypes did not enter the full HF/platform adoption gate.
- **Previous investigation:** AVX2 integer dots with vectorized activation packing
  were tested at 8-bit and 16-bit precision. Q16 improves matched mean decode
  by 2.79% on 0.6B and 4.06% on 8B, but still trails mx by 5.12% / 1.21%.
  Both variants pass existing Windows HF fixture bounds; Q16 stays much closer
  to the current float path. Independent packing/product controls pass,
  including signed weight extremes, half subnormals and fallback cases.
  Across four excerpt/window cases, Q16's maximum absolute NLL change versus
  current llmx is 0.00003155. Across 5,013,888 logits on a 1,943+32-token
  forced-HF continuation, maximum change is 0.002213; maximum and RMS error
  against HF are slightly lower in this case. This is a nonzero precision
  change, not a lossless result. Full corpus, independent 8B HF, maximum
  context and cross-platform validation remain open. No candidate was adopted;
  all patches, samples and numerical controls are preserved in
  `benchmarks/q8-integer-activation-20260919.json` and ASSETS.
- **Earlier investigations:** paired F32 decode rows did not improve throughput;
  packed F32 prefill variants regressed. None was adopted. Exact patches,
  samples and diagnostics are in ASSETS and the paired-decode/packed-prefill
  benchmark JSON files. These experiments used the `5a9518c` runtime.
  Profiling scalar exponentials finds only 7.45 ms SiLU wall time and 9.42 ms
  summed softmax worker time during 2,237.50 ms decode; these are not the main
  remaining cost. The summed worker measurement is not wall time.
  The preceding matched Q8 comparison covers both the HF fixture and real 8B
  model against pinned public mx `5542318e74`: three alternating pairs,
  215+32 pinned tokens, six threads, ubatch 128 and F32 KV.

  | Q8_0 model / phase | llmx mean tok/s | mx mean tok/s | Gap |
  |---|---:|---:|---:|
  | Qwen3-0.6B prefill | 380.50 | 265.35 | +43.39% |
  | Qwen3-0.6B decode | 42.73 | 46.81 | -8.71% |
  | Qwen3-8B prefill | 25.94 | 20.07 | +29.27% |
  | Qwen3-8B decode | 4.10 | 4.39 | -6.64% |

  Prefill exceeds the reference on both models, but decode remains below it;
  both comparisons have disjoint arm ranges in each phase. Raw samples,
  hashes, flags and scope are in `benchmarks/q8-external-floor-20260919.json`.
  Source inspection confirms mx uses quantized Q8 activations and integer
  dots, while llmx retains float activations. Any analogous optimization
  needs a measured numerical cost bound before adoption.
- **Historical stand-in comparison (different conditions; not pooled):**
  Qwen3-8B Q8_0, 343-token wikitext prompt, -t 16, this workstation. Reference
  is the CPU AVX2 llama.cpp shipped with LM Studio, stock `llama-server`, same
  machine, so no rig time was used.
    - pp   llmx 37.23 / 37.86   llama.cpp 37.70 / 37.34   -> parity
    - tg   llmx  3.91           llama.cpp  4.99 / 5.02    -> 22% under
  These are the original matched measurements. Later Q8_0 decode commits
  report 4.12 tok/s with independent accumulators (`54ea063`) and 4.24/4.18
  with F16C (`0c12570`); these are not a new matched mx-llama.cpp comparison.
  Separate decode/prefill thread flags landed in `dada6c0`.
- **How prefill got there, 3.89 -> 37.5 tok/s (9.6x), each step A/B measured:**
  - persistent worker pool instead of spawning threads per call (decode -23%)
  - attention through `Backend::parallel_for` instead of its own threads
  - batched prefill: matrix-matrix instead of one token at a time (3.89 -> 13.2)
  - dequantize each weight row once per batch, not once per column (-> 15.3)
  - four independent accumulators in the f32 dot (-> 17.0)
  - fused 4-row kernel sharing one activation load (-> 24.0)
  - two activation columns per four rows, 0.75 loads/FMA (-> 33.7)
  - three activation columns per four rows, 0.58 loads/FMA (-> 37.5)
- **The lesson worth keeping:** the kernel was LOAD bound, not FMA bound. Each
  naive dot needs 2 loads per FMA and Zen3 sustains about 2 loads/cycle against
  2 FMAs/cycle, so it ran at half of FMA peak no matter how the batch was
  blocked. Every win after the first came from raising the FMA:load ratio.
- **Left:**
  - Close the current Q8 decode deficits while retaining prefill gains. The
    latest fixed `bf122fd` control/candidate/mx session measures control mean
    gaps of -2.01% (0.6B) and -0.67% (8B), with both medians below mx. The
    preceding two-arm session measured -0.65%/-0.82%; keep the sessions separate. Earlier bandwidth/thread observations and null allocation,
    fragmentation and prefetch experiments do not predict this current gap;
    mmap has not been established as a throughput improvement.
  - Grouped Q16 activation packing is now rejected. Its own scalar/integer
    arithmetic and activation tests passed, but unchanged native accuracy
    checks fail on MSVC and GCC (control 7/7, candidate 6/7). First diagnosed
    absolute error is 0.000219106674 against the existing 0.000206180004 limit,
    on a standalone one-worker Q8 dot. No test tolerance was changed; model
    HF-cost/performance runs were stopped before execution. Evidence:
    `benchmarks/q16-group-native-rejection-20260920.json`. Bounded-cost research
    does not satisfy AGENTS' lossless requirement. Preserve the rejection;
    do not tune a new tolerance to that failing case.
  - The exact Q8 horizontal reduction is rejected by complete timing despite
    passing numerical gates. Read-only review also found two redundant `h_`
    clears in `Model::step`, but no evidence that their cost closes the gap.
    The native two-block lambda is rejected at the no-spill codegen gate.
    The ordinary inner-loop follow-up also fails the useful-scheduling gate:
    no spills, but one extra feature reload and no useful cross-block work.
    Unrolling remains closed. Caller attribution and native sampling are
    complete above; neither established a recoverable production cost. The
    separately reopened prefill-placement assessment remains active. Prior
    F16C specialization, pointer increments and row pairing are nulls.
  - Thread and matrix-shape diagnostics are complete (see CPU comparison
    thread scaling above). F32 matrix ranges overlap mx, while Q8 matrix
    latency still trails it. The resulting head-major KV layout is now validated.
    Further changes should follow profiling of the current runtime, not repeat
    completed instruction/dispatch studies. The larger-model diagnostic did
    not close the external gap.
- **Gotchas:**
  - Synthetic `bench` throughput does not establish real-model speed. Small
    projections may stay serial depending on thread count. Grouping improves
    the six-thread synthetic case, but the real-model external floor still
    fails; use the matched model measurements.
  - Do not tune the row block as a byte budget. Measured worse at every size
    (64/128/196/256 KB gave 22.37/22.04/23.68/21.12 against 24.04 for a flat
    4); the knee follows the fused kernel width, so it is `DOT_ROWS`.
  - ubatch barely matters once the kernel is right, and 343 vs 512 on a
    343-token prompt is the SAME computation - do not read noise as signal.

### Device execution model (ROADMAP #4a)

- **Process note:** this block was opened after the code was written, which
  `AGENTS.md` forbids.
- **Goal:** land the backend-agnostic execution model #4b depends on, in the
  six steps of `docs/DEVICE-EXECUTION.md`. Bar per step is no measured
  regression, not a win.
- **Done:** the design, `b1e4904`. Step 1, weights resolved once at load,
  `edd617f`: suite green, HF logits and PPL unchanged on Q8_0 and Q4_0. Doc
  page refreshed in `1d4ffa4`.
- **Done:** step 2, batched elementwise ops, committed on the feature branch.
  Suite green and HF perplexity bit-identical to step 1.
- **Done:** step 2 measured under protocol after 57294 went terminal, with the
  plan, advance rule and contamination criteria frozen and hashed before any
  timing. Nine measured pairs, arms alternating and reversing, every arm under
  `monitor_windows.py`, no sample dropped. **Result: does not advance.**

  | Phase | base mean | cand mean | Mean | Median | Baseline wins |
  |---|---:|---:|---:|---:|---:|
  | Prefill | 337.91 | 346.55 | +2.56% | +2.66% | 2/9 |
  | Decode | 27.48 | 27.69 | +0.79% | -0.50% | 5/9 |

  Prefill passes every criterion. Decode fails paired wins (5/9 against a
  rule of <= 4) while its mean and median both sit inside the 1% noise band,
  so that phase is better described as indistinguishable than regressed. The
  rule is not weakened and the run is not repeated to obtain a pass. Evidence:
  `benchmarks/device-exec-step2-20260920.json`.
- **Left:** decide step 2's disposition. The frozen rule is stricter than
  `AGENTS.md`'s own tradeoff principle, which says a large gain can justify a
  minor loss and warns against rejecting on an isolated per-case cutoff. That
  tension is a judgement call and must not be resolved by editing the rule
  afterwards. Options: re-measure with more pairs under a NEW prospective plan,
  or keep decode on the single-row ops so only prefill changes.
- **Left:** step 1 still has no admissible measurement of its own; it was timed
  off-protocol during 57294 (13:33:05-13:52:07 +0300, disclosed at the time).
  Expected neutral, unproven.
- **Left:** the external mx-llama.cpp floor is untouched by this runner and
  still applies to any step that advances.
- **Left:** steps 3-6 (buffers, arena, KV on buffers, sync) untouched. No
  vendor backend is writable until step 6. Step 3 was started as interface
  plumbing only and reverted: `Buffer` with no caller is a speculative seam,
  which `AGENTS.md` forbids. It must land together with the weight conversion
  that uses it, which means touching `arch_qwen.hpp` and waiting for the other
  developer's
  stack.
- **Gotchas:**
  - Step 2's first decode reading of +3.40% was an artifact: its two winning
    pairs were the two lowest-throughput rounds, the other six gave +0.63%.
    Decode runs B=1 and cannot benefit. Prefill is the real effect, 8/8 paired
    wins. All samples retained.
  - Not a SwiGLU result. The screened-out "Exact decode SwiGLU callback fusion"
    measured a different change and its verdict stands.

### Fused Q5_K and Q6_K row dots (ROADMAP #1)

- **Goal:** give Q5_K and Q6_K the decode row dot Q4_K already has, so no
  dequantized value is materialised. Q4_K works because `d*q - m` factorises
  into `d*sum(q*x) - m*sum(x)`; Q5_K carries the same scale/min pair plus a
  high bit, and Q6_K has signed group scales and no min.
- **Done:** Q5_K fused dot, adopted. `dot_row_q5_K` mirrors `dot_row_q4_K`
  with the fifth bit taken from `qh`, whose mask shifts two places every 64
  values while `qh` itself does not advance. Dispatch now covers both types
  through one threading block rather than a copy.
- **Done:** correctness against the fp32 HF golden, not against llmx. The
  ordinary suite's fixtures are Q8_0 and Q4_0, so it does not cover this path
  at all; `Qwen3-0.6B-Q5_K_M.gguf` was scored directly against
  `tests/data/baseline_logits.json`: top-1 6/6, worst top-5 overlap 4/5, mean
  4.50, max absolute logit 24.46 against the 100 bound. The file is 29 tensors
  of Q6_K as well, so that path is exercised incidentally.
- **Done:** throughput, plan frozen before timing, 15 measured pairs.

  | Phase | base mean | cand mean | Mean | Median | Baseline wins |
  |---|---:|---:|---:|---:|---:|
  | Decode | 8.424 | 12.751 | +51.369% | +51.12% | 0/15 |
  | Prefill | - | - | -0.93% | -0.64% | 10/15 |

  Evidence: `benchmarks/fused-q5k-decode-20260920.json`.
  **Corrected 2026-09-20 by an independent recomputation of all 15 pairs**: the arm
  means were 8.424 and 12.751333 tok/s, not 8.96 and 13.33. The paired
  +51.369% is unchanged, and Q6_K recomputes to +68.0186%. The original
  figures were read from a single round rather than the arm means.
- **Done:** the matched external floor, which the above does NOT establish.
  `tools/compare_cpu.py`, mx `5542318e74`, six threads, eight matched rounds,
  identical committed HF token IDs in both arms:

  | Phase | llmx | mx | Ratio | Gate |
  |---|---:|---:|---:|---|
  | Prefill | 324.33 | 230.33 | 1.41x | above |
  | Decode | 15.98 | 58.30 | 0.27x | **BELOW** |

  Q5_K decode does not meet ROADMAP #8. The fused dot is a real 51% gain over
  the previous llmx build and still leaves mx 3.65x faster. Reporting the
  self-comparison alone would have read as success.
  Evidence: `benchmarks/q5k-external-floor-20260920.json`.
- **Done:** located the gap. Same machine, session, binaries and harness, two
  quants:

  | | llmx | mx | ratio |
  |---|---:|---:|---:|
  | Q8_0 decode | 40.93 | 41.44 | 0.99x |
  | Q5_K decode | 15.98 | 58.30 | 0.27x |

  Decode is bandwidth bound, so Q5_K (about 0.69 bytes/weight) should beat
  Q8_0 (about 1.06). mx does that: 41.44 -> 58.30, +41%. llmx goes backwards:
  40.93 -> 15.98, -61%. A smaller quant making llmx slower means the K-quant
  decode path is limited by per-weight unpacking and float conversion, not by
  memory traffic. Q8_0 sits at parity because its fused dequant+FMA matvec is
  cheap enough to stay bandwidth bound.
  Evidence: `benchmarks/kquant-decode-bound-20260920.json`.
- **Left:** prefill work cannot close this. Both quants are already above the
  floor in prefill, 1.50x and 1.41x in the same runs. The whole gap is K-quant
  decode.
- **Left:** test whether quantizing the activation vector to 8 bits and taking
  an integer dot product restores bandwidth-bound behaviour, as mx does. That
  is a much larger change than a fused dot, affects every quant type, and
  changes arithmetic, so it needs its own prospective plan and its own HF
  gate. It is the highest-value remaining decode work.
- **Left:** the same treatment for Q6_K, which has signed group scales and no
  min, so the dot is `sum(d_g * sum(q*x))` with no `sum(x)` term. Written, not
  built or measured. Expect the same floor gap to remain afterwards.
- **Done 2026-09-21:** `Qwen3-0.6B-Q5_K_M.gguf` is in `BASELINE_MODELS`
  (see the Vulkan block above), so the external gate covers Q5_K/Q6_K
  permanently on both backends. It is a third download for
  `tools/fetch_test_models.py`, 419 MB from the same repo and revision as
  the Q4_0 file.
- **Gotchas:**
  - Prefill does not benefit and cannot: the fused dot only fires at
    `nbatch == 1`, and the batched path already dequantizes each row once and
    reuses it across the batch. That made prefill an accidental control in the
    Q5_K run - identical code measured -0.93% mean and -0.64% median, which
    puts this setup's noise floor near 1%. Useful when reading small effects.
  - Qwen3-0.6B-Q5_K_M is 168 Q5_K / 29 Q6_K / 113 F32, so it exercises both
    paths in one model. A fused Q5_K dot alone will show a partial effect.
  - The roadmap's 2.16 -> ~2.6 tok/s is Q4_K's result on its own model, not a
    target for these types.

Nothing else is in flight. When you start a feature, open a block above
before writing code - see `AGENTS.md` -> "Starting a feature".
