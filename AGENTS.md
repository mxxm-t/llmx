# AGENTS.md

Guidance for AI agents (and humans) working in this repo. Read this before
making changes.

## What this is

**llmx** - a ground-up, dependency-free LLM inference runtime. It reads/writes
GGUF v3, runs quantized or F32 Qwen3, Qwen3-MoE and the Qwen 3.5 hybrid (qwen35, qwen35moe) transformers on x86 CPU with AVX2/FMA/F16C
or on a Vulkan device, and is structured so formats, quantizations, backends, and multi-device / cluster
serving can be added later without touching the core.

## Build

`docs/BUILD.md` is the one page on building: the prerequisites and commands on Windows, Linux and macOS, the Vulkan backend, the Docker image, where the binary lands, the build identifier and the test commands.
In short, `build.bat` writes `llmx.exe` to the repo root, and `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release` followed by `cmake --build build --config Release` builds the binary and the native tests.
The notes below are for working on llmx itself.

For MSVC builds in temporary worktrees, use a fresh build directory or `cmake --build build --config Release --clean-first` after header changes.
An incremental invocation here emitted MSB8029 and left the executable older than edited headers, despite exiting successfully.
Preserve the compile log and verify that the changed source was actually rebuilt before claiming a gate.

The build dir's `generated/config.hpp` is produced from `cmake/llmx-config.hpp.in`; the checked-in `src/config.hpp` is the fallback used by the plain `build.bat` path.
Keep the two in sync when you add build knobs.

The Linux MI50 machine builds and runs the Vulkan backend in the image from `docker/Dockerfile` (`docs/BUILD.md`, Docker), which carries the loader, the Mesa driver, the headers and the shader compiler and takes the cards from the host through `/dev/dri`.
That leaves the shared machine's packages untouched.

## Principles

- **Performance first.** llmx is a *runtime*: a slow-but-correct implementation
  is not enough. Every change to a hot path (quantized matmul, RMSNorm, RoPE,
  attention, KV cache) should state the perf impact and be benchmarked, not just
  verified for correctness. When correctness and speed trade off, prefer the
  fast path and prove it is lossless (see `docs/ROADMAP.md` correctness gate).
- **Assess performance tradeoffs.** Report gains and regressions together across
  phases and models. A large gain can justify a minor loss elsewhere; do not
  automatically reject it on an isolated per-case cutoff. Preserve all results
  and explain the workload tradeoff. HF correctness and matched mx comparisons
  remain required.
- **Build both arms the same way, and run them in the same place.** A
  comparison is only about the change if nothing else differs between the
  binaries, and the same applies to where they run: a figure taken on one
  machine, driver and operating system does not subtract from one taken on
  another. Two reference figures that differed by build, compiler, operating
  system and driver at once were briefly read here as a driver effect; they
  said nothing. Compare llmx before, llmx after and the reference within one
  environment, and quote across environments only as separate results. The build embeds the Git
  revision and a dirty marker, and that string alone moved 0.6B prefill by
  several percent: the same change measured -8.03% with mismatched build
  identity and -0.08% with both arms built from detached worktrees at their
  own commits. Build both arms the same way, from the same kind of tree, and
  record each binary's `--version` alongside its hash.
- **Code layout is part of the measurement.** In a header-only runtime with
  one translation unit, an unrelated edit can reshuffle the hot path.
  Behaviourally identical builds, differing only by an unused function, span
  about four points of prefill on Qwen3-0.6B, and one of them failed the
  advance rule outright. Before believing a few-percent result on a hot-path
  cell, compare against a perturbed build of the same behaviour and check the
  difference is outside that band. Evidence in
  `docs/benchmarks/layout-sensitivity-20260921/`.

  **How big the band is depends on where you edit, so choose the control to
  match.** Appending an unused function to the end of `cpu_backend.hpp` moved
  0.6B prefill by -1.16/+0.63/-1.13/+0.33 percent over four runs. Adding a
  four-line bounds check inside `Tokenizer::decode`, a function that
  `Model::prefill` never calls, moved the same measurement by
  -3.13/-2.17/-3.09 percent and failed the advance rule twice. Same tree, same
  binaries built the same way, no change to any executed instruction of the
  measured path. A control that perturbs a different file than your change
  will understate the band and convince you a layout artifact is a
  regression; that is what happened here before the cause was bisected.
  Evidence in `docs/benchmarks/code-read-20260921/`, cells `06-layout*` and
  `06-tok*`.
- **Check machine contention.** Record timestamped background process CPU use
  and system CPU, disk and GPU activity before and throughout performance runs,
  using the same low-overhead monitoring in every arm. Separate benchmark and
  monitor activity from unrelated work. Do not run competing builds, tests or
  downloads. An idle machine is not required. Define activity flags before
  measuring, retain every planned matched block and report potentially affected
  runs alongside the complete results. Interleave and repeat arms to assess
  noise and activity imbalance; do not automatically stop, discard or replace
  a run because background activity is present. Do not discard isolated slow
  samples. If monitoring is unavailable, report that limitation instead of
  assuming the machine was idle or treating unknown activity as zero.
- **Dependency-free.** No external libs. The whole point is to control the full
  stack; reaching for a library erodes that.
- **Lean, not clever.** Add a seam only when a second implementation is on the
  roadmap. No speculative abstraction, no empty stubs.
- **One owner per concern, kept tight.** A feature is implemented once, in the
  lowest layer that holds what it needs, and callers reach it through one call
  (`docs/ARCHITECTURE.md`, Each concern has one owner). Before adding code,
  find the concern's owner; if a second caller needs the same steps, move them
  into the owner instead of copying them. The CLI and the server read flags
  and requests and call down; they hold no model, placement or loading logic.
  Remove what nothing reaches (functions, flags, kernels, diagnostic switches,
  experiment code) in the change that leaves it unreached.
  Every job checks this (Dead code and stale docs, below).
- **Not overengineer, not underdo, no bloat.** This is a ground-up runtime with a
  deliberately small surface, so the default is to build *only* what the current
  feature needs and no more. But "lean" is not an excuse to ship a half-built
  feature: if a piece is claimed or a path is promised, it must actually work and
  be covered. Prefer the simplest thing that fully solves the stated problem;
  leave generic seams, flags, and scaffolding out until a concrete second use
  exists. When in doubt, ask whether the extra code pays for itself today.

## Configuration: flags, not environment variables

Runtime knobs are **CLI flags**. Environment variables are not used for runtime
configuration, because they are invisible in a command line, do not appear in
`--help`, and silently change results between runs - which is exactly what you
do not want while measuring.

- **A knob a user would set** is a flag: `--threads`, `--ubatch`. Choose the
  name that fits it best. An established name is a good candidate where it
  fits, but no name is taken only because another runtime uses it, and no
  behaviour is copied with it (llmx has no `n_batch`, see below).
- **A value the code can determine** is not a knob at all. Row blocking in the
  CPU matmul was briefly `LLMX_ROW_BLOCK`; it is now `DOT_ROWS`, the width the
  fused kernel handles, because measurement showed the knee follows the kernel
  and not the machine. A constant that is a property of the code does not
  belong in the environment.
- **Temporary A/B knobs get deleted** once they have answered their question.
  Both `LLMX_ROW_BLOCK` and `LLMX_ROW_BLOCK_BYTES` existed only to find a
  number and were removed with the finding recorded in `docs/STATUS.md`.
- **The runtime credential exception is `HF_TOKEN`** for `llmx pull` gated-repo
  access. It is not a tuning knob and must not appear in child argv or logs.
- **Test configuration**: `LLMX_BASELINE_GGUF` points `tests/baseline.py` at a fixture model; `LLMX_DEVICE`, set by `run_tests.py --device`, appends `--device` to every command that takes it so the suite runs on a device backend, except where a component names its own devices (`threads` names the CPU, whose thread counts it checks, and `split` names the CPU backends its tool splits over); `LLMX_CACHE_TYPE`, set by `run_tests.py --cache-type`, appends `--cache-type-k` and `--cache-type-v` the same way so the HF gate runs with a chosen cache type; `LLMX_LAYER_SHARES`, set by `run_tests.py --layer-shares`, appends `--layer-shares` so a device list is tested at a split the fit would not choose; `LLMX_TENSOR_WIDTH`, set by `run_tests.py --tensor-width`, appends `--tensor-width` so a device list runs as tensor groups, and `tensor-split` then captures on that list (docs/TENSOR-SPLIT.md); the components whose synthetic fixtures a group refuses, an odd feed-forward width or one head (`perplexity`, `f32`, `moe`, `mxfp4`, `qwen35`, `drafters`, `shards`, `server` and `chat`), skip under a width above 1 (`common.tensor_width_skip`).
  The synthetic bench (`bench` without `--model`) takes only `--device` and `--threads`, so it gets neither shares, cache types nor a load mode, and `split` gives its tool equal shares and the cache types it names itself.
  The runtime stores f16 by default, so the components that check exact f32 arithmetic against independent fixtures (`f32`, `moe`, `qwen35`, `shards`, `server`) ask for f32 sides themselves and skip when `--cache-type` asks for another type (`common.f32_cache_skip`).
  `LLMX_LOAD_MODE`, set by `run_tests.py --load-mode`, appends `--load-mode` to every model command the same way, so the suite runs with the weights read in any mode.
  `LLMX_DTYPE`, set by `run_tests.py --dtype`, appends `--dtype` to model commands that do not name it themselves; it never reaches the synthetic benchmark or metadata-only commands. The split component passes it through its tool's optional dtype argument to both placements.
  That is test configuration, not runtime configuration, and it reaches the binary only as the flags.

If you add a flag, add it to `docs/USAGE.md` and to `print_usage` in the same
change, or it does not exist as far as a user is concerned.

## Verify

The round-trip component (`tests/roundtrip.py`, under Tests below) checks the format and quantization paths.
It builds a random F32 model, quantizes it to Q8_0 and Q4_0 through the CLI, dequantizes it back and bounds the error, and decodes the written blocks with the spec decoders of `tests/spec_decode.py`, written from the format descriptions, rather than with llmx's own reader.
Q4_1, Q4_K, Q5_K and Q6_K, which `quantize` does not write, and Q8_0 and Q4_0 under negative scales, which their quantizers never write, are decoded the same way from raw blocks the test writes, chosen so every scale, min, high bit and nibble reaches a decoded value, and `dequantize` must match bit for bit, the sign of zero included.
Under a negative scale Q4_0's nibble 8 must decode as -0, as the format's d*(nibble - 8) gives it (`docs/ASSETS.md`).
Run it alone against the root `llmx.exe` that `build.bat` writes, or as part of the suite for a CMake build:
```
python tests/roundtrip.py
python tests/run_tests.py --exe build/Release/llmx.exe
```
`llmx.exe info FILE` lists the metadata and tensors of a written GGUF.
On Windows with `LongPathsEnabled` set, the round-trip component also reads a GGUF through a path longer than 260 characters and checks identical decoded payload bytes and the preserved full model path in its metadata; it records a skip of this case if the policy is disabled or unavailable. Both MSVC build routes embed the shared `cmake/windows.manifest`; no test or runtime enables the system policy.

## Tests

Run the native backend, model, chat-template, KV storage, loader and streaming checks
after a CMake build:
```
ctest --test-dir build -C Release --output-on-failure
```

`json` checks syntax, numeric/locale boundaries, UTF-8 and escaped Unicode,
malformed input, nesting limits and JSON output string escaping.
A float written by `number` is its shortest decimal in the C locale's form under a comma locale too, reads back through `parse` as the same float for every finite value among a million random bit patterns, and is `null` when JSON cannot hold it.
The Q8/Q4 round-trip test also checks escaped Unicode tensor names through the actual CLI.

`format-output` checks conversion staging, serialization and stream failure,
refusal of premature publication, a later publication failure after the first
complete file was published, and temporary cleanup. The Python round-trip adds
real child-only file-size-limit failures on POSIX, existing-file preservation,
a failed second output open and aliased raw output refusal.

`gguf-validation` checks independent binary fixtures for field lengths/counts, array depth, tensor arithmetic, byte counts that overflow although the element count fits, in one row or across rows, file extents, tensor types and quantized row widths, each refused as such when the element count also overflows, custom alignment and a tensor name repeated in one file.
Its independent layouts cover all 35 known storage types, including those without decoders, and refuse unknown and removed IDs even in empty tensors. It reads the type IDs and block declarations in `src/backends/vulkan/shaders/q.glsl` and holds them to `quant/types.hpp`, without running a shader. CTest supplies both arguments: `llmx-gguf-validation-test <fixture.gguf> <q.glsl>`.
These are format checks; they do not establish model-schema safety.
`load-progress` reads, maps and reads in a file as the loader does, and checks the progress, each tensor's file span, that reading the headers maps nothing, that a model not mapped is neither written nor read in, early rejection, and a file truncated before loading or whose size changes between reading and mapping, refused before any progress.
The loader's readers check a file's size against its header with the mapping's own check, so a changed size is refused the same way.
It then writes a tiny Qwen model with tokenizer metadata and loads it through `infer::load_model` in each load mode: on the CPU the payload is kept, and on a CPU backend that copies what it adopts and reports `reads_in_place()` false the host copy is released (`payload_size()` is 0), streamed from one file in `auto`; a direct load on the CPU maps nothing and its copy holds each weight where the file does.
Every load's progress starts at 0, only rises and ends at the payload, and every load gives logits bit-identical to the same model built in memory; a load on the CPU also holds the tokenizer and the chat format beside the model.
Each load runs on a copying backend that reads the read ring in place, which takes every part as a copy out of it, and on one that does not, which takes writes.
A write that fails part way through a streamed load, and a progress callback that throws, stop it with their errors on the asynchronous test backend `model-validation` uses (`tests/loading_backend.hpp`), and no buffer is freed before it drains.
A qwen3moe model with its experts on the CPU beside a copying backend loads in each mode; so does a split over two copying backends with a tied head, whose embedding reaches both and whose every write goes to its own backend's storage, and the CPU beside a copying backend, whose shared embedding is read and reported once.
A set of three shards with a metadata-only first loads in each mode, streamed from the two shards holding tensors.
The planned reads are checked on spans with no file: pieces on the granule within the limit, a gap of one granule read through and a longer one starting a new piece, a tensor longer than a piece, a new file, and every tensor's bytes in exactly one part; so are the rule for reading around the cache, which reads around it when the bytes streamed are more than the memory available and not when they equal it or the memory is unknown, and a reader that falls back to the cache where direct reads are refused.
The stream itself runs in reads of one page, so tensors cross reads and reads hold several tensors: every copy, two of one tensor included, holds the file's bytes, and a file cut after its header was read stops it with where the file ended.
In dozens of 64-byte reads with slow fills the readers fill the ring and wait, and every copy still holds the file's bytes, filled by copies, by writes, and by copies that run only when their submission is waited for, as a device's do, which a slot refilled before its copies retire would spoil; a fill failing late stops the stream; two members of a tensor group take their shards of every role through the same reads (`Upload::runs`), columns a run a row, and each member's storage holds exactly its packed shard filled each of the three ways; where direct reads are taken, a read that fails on a reader thread comes out of the stream.
`disk-store` checks the disk tier's store alone (`server::DiskStore`, `docs/DISK-TIER.md`), every wait on a store call bounded at a minute: CRC32C's standard check value; an entry of two runs over several slabs, not whole ones, read back bit for bit with its blob, its file the size its layout gives, in a directory its owner's alone; a read into another layout refused and the entry deleted, a missing entry refused, a write cancelled while queued leaving nothing and an evicted entry gone; with keep, the directory left marked kept, a store of another identity adopting nothing and removing it, and one of the same identity adopting the entry, which reads back; a payload byte flipped on disk failing its checksum and a truncated file refused, each deleted; a store without keep leaving nothing; the sweep removing a crashed directory with a temporary file and a junk entry and a kept one past the age limit while keeping a young kept one, adoption leaving out an entry past the age limit; a live server's directory, its lock held by a child process, left alone; and the free-space floor stopping a write.
`file-reader` checks `format::FileReader` and `core::HostPages`: owned pages and their moves, a reservation's rounding, a size past the largest whole number of pages a `size_t` holds refused by name by both and the largest that rounds refused by the operating system, commits that start and end inside a page and repeat, commits outside it refused, and a decommit that keeps the pages it covers only in part; reads at any offset and length on a file of several granules, a read past the end short by exactly what the file lacks, empty and tiny files, and eight threads reading one reader at once; where the test's file system takes direct reads, aligned direct reads, a rounded one past the end, and the refusal of a misaligned one, and where it does not, the refusal naming the file.
The loader's checks run in `direct` too where the file system takes direct reads, and check its refusal where it does not.

`file-reader` also checks `format::file_sha256` against `core::Sha` over a file past one 64 MiB piece and an empty one, and `format::cached_file_sha256`, which returns a planted digest while the file's stamp holds and reads the file again once it changes.
`hub-manifest` checks SHA-1 and SHA-256 against standard vectors and, where the CPU has the SHA extensions, their compression against the portable code's over 1 to 33 random blocks.

`gguf-shards` covers complete shard sets, metadata-only first shards, exact
payloads and the shard each tensor's span names (a zero-sized tensor that ends
a shard included), inconsistent metadata,
duplicate tensor names, truncation before reading, a size that changes between reading and
mapping, aggregate progress and Unicode file paths. `hub-manifest`, `hub-pull` and `hub-transport` are offline tests of
variant selection/hash vectors, concurrent range assembly/cache repair and
native curl child lifetime/response handling. Real transfers are separate
integration checks. The Python `shards` component compares sharded synthetic
logits and NLL against the independent HF fixture.

`model-validation` checks Qwen configuration ranges/defaults, required tensor layouts and in-memory storage before model execution buffers are allocated.
It covers tied/untied output, supported matrix types and singleton axes, the qwen3moe keys and routed roles, placements the model refuses, `place_model`'s refused requests and a pass past the context.
It also requires an unsupported embedding, tied head on another device or late
mixer weight to fail before any adoption or allocation. Two routed layers check
that a destination missing a copied router or windowed expert type leaves only
that layer on its host, with exact prompt and follow-up logits; a host missing
the type is refused even when streaming is requested.
Mixed gate/up expert types also require the backend's `mixed_experts` operation: an incapable resident backend is refused before adoption or allocation, and an incapable stream destination leaves that layer on the CPU, with identical prompt and follow-up logits. A separately typed down projection stays eligible, and missing gate/up tensors keep their existing refusal text. Both architecture plans are checked; the tiny qwen35 plan check alters storage tags only and does not execute those views.
The fit counts what the model reads: a routed layer that also carries dense matrices is fitted without their width in its activations and without their bytes among its weights.
A tensor width a model's shards cannot take is refused naming the projection (`model/shard.hpp`): a width that does not divide the heads, the K heads or the vocabulary rows, one that neither divides the KV heads nor is a multiple of them, a column split off whole Q4_K blocks, and a routed layer or an embedded drafter, which a group does not split yet.
Every refusal it provokes through its checks of the file, the placements, the context, the device types, the tensor widths and a split's loading must give the label and text listed in `tests/data/model_refusals.txt`, in order, so a change to a refusal's text or to which defect a file is refused for changes that list, while the context storage limit and the other loading, window and hook failures are held to their texts in the test itself; `llmx-model-validation-test --write FILE` writes the refusals it sees.
An asynchronous test backend (`tests/loading_backend.hpp`, which `load-progress` also uses), which plays a device, so it copies what it adopts and says it is not the CPU, also checks loading failure and model teardown drain pending work before releasing buffers, including split placements and backend reuse.
Through the loader's own adoption hook (`infer::planning_adopt`), it checks that nothing is read in place on that backend, and that a host running a streamed layer's experts beside it reads exactly that layer's feed-forward norm, router and three expert stacks, the norm and router taken by both.
It checks the hook both ways: deferring the copies, as the streamed load does, every copied weight gets storage and no weight is written or adopted while the model is built, only the position tables the model computes itself, and a model missing a tensor, or whose cache does not fit, fails after storage but before any weight is uploaded; without deferring, as the mapped load does, every weight is adopted as the model resolves it.
It does not validate numeric weights, arbitrary token IDs or failed-session recovery.
The models it builds sit on backends of one thread, but for the fit check's, which takes the CPU backend's automatic count, and those the construction, loading and reader checks build must start no worker threads.

`shard` checks how a model splits over a tensor group (`model/shard.hpp`, docs/TENSOR-SPLIT.md, section 4.2) on a dense qwen3 and a dense qwen35 plan built in memory, whose every split falls on whole 256-value blocks at widths 2 and 4.
Spans written out by hand: a linear-attention layer's `attn_qkv` q and k rows by K head and its V rows from each of three tiles, the conv's channels with them, `ssm_out`'s columns and the decay's elements by those V heads, a gated q head, `attn_output`'s columns, a KV head replicated over two members at width 4, the head's vocabulary rows, `ffn_down`'s columns, the whole axis at width 1 and `ssm_norm` whole on every member.
For every storage type a matrix may have, at widths 1, 2 and 4, tied and untied, each member's packed bytes of every role must be the bytes an oracle in the test gathers from the declarations' meaning alone, whole rows it owns on the row axis and each row's blocks it owns on the column axis, its shape must hold them, its runs must follow the tensor's order packed back to back, and unless a KV head is replicated the members must hold each byte once.
A member's footprint (`footprint` with a width and a member) must equal counts made in the test: its rows and columns of every Q8_0 matrix, the norms and the embedding whole, its vocabulary rows' logits, its KV heads' cache and its heads' recurrent state, and at width 1 the footprint of one device.

`mxfp4-vulkan` checks MXFP4 precision paths through public calls: every scale byte and code in embedding against an independent field decoder, dense/grouped/routed products, logits, accumulation, guards, extreme finite products and wide split partials. BF16 uses an independently rounded-input reference; integer results include an independent block-quantization error allowance, while boundary cases require exact bits. Matrix products pin scale bytes 0, 1, 103, 104, 143 through 146, 252 and 255, and completed matrix paths must match the selected policy. Mixed prompt/decode dense, logits, additive and grouped calls must agree bit for bit across microbatches of one and three at both sides of the tile threshold. Isolated overflow-cancellation and subnormal fixtures require the integer tile and split reduction together on integer-tile profiles; F32 intermediate-storage mutations were rejected independently. `docs/STATUS.md` records the released dtype qualification and remaining model and reference-speed work; these native checks alone do not establish those gates.

`backend-group` checks mixed types, uneven rows, batches, thread counts,
output boundaries and fallback behavior against separate calls and double dots.
It also checks every finite f16 scale against signed Q8 weight extremes using
one-hot inputs with exact expected products, and every column of the three-, two- and one-column prefill dots against one ordered scalar FMA oracle across dimension tails and unaligned inputs.
Q8 products also read a tiny original F32 input beside a zero-weight block peak exactly, through plain, grouped, routed and residual calls, across row classes, widths, signed weights/scales and thread counts.
The test is built without contraction, so a tail the kernels leave to the compiler fails it.
Its arithmetic-oracle checks request F32 on each matrix call, so they pin the float paths, and a grouped call there is the base's call of each projection in turn; the CPU backend's own grouped dispatch is checked by `q8-dots`.
It also sweeps activations from 1e-30 to 1e36 over every type it reads, where a row whose exact result fits must come out finite and within the double oracle's bound, checks `norm_rope_partial` at out-of-order positions and `gather_rows` in pick order, and refuses a row beyond the source, an unknown type and a projection without storage.
It holds the CPU's collective of a tensor group (docs/TENSOR-SPLIT.md, section 4.3) at widths 2, 3 and 4 and at 1 and 6 workers: every member's residual, at an offset in its storage, must gain ((p0 + p1) + ...) + p(W-1) of the members' partial rows with the float bits the test computes in that order, at every row count up to the reservation, and a collective joined by a member not first or over a backend that says it is not the CPU, and a sum of fewer members, more rows, another width or past a residual's storage, are refused.
Every pair of extents `Backend::row_class` puts in one class, at extents on each side of every device crossover and tile split from 1 to 1000, gives the same bits for four rows of one prompt through a matmul of F32, Q8_0, Q4_0, Q4_1, Q4_K, Q5_K, Q6_K and MXFP4 at rows 256 and 4096 wide and the routed gate and down projections of each type, under each of the F32, F16 and BF16 matrix policies. The shared check also requires finite outputs and checks dtype-independent attention of 128-wide heads after a 70-token history once (`tests/row_classes.hpp`); a generated token is a class of its own.
The independent HF fixtures below remain the external correctness gate.

`fused-dot-overflow` pins the two kernel families apart. `dot_row_impl` folds
the scale into each weight before the activation, avoiding the demonstrated
scale-after-sum overflow; accumulation can still overflow under cancellation.
The fused K-quant dots accumulate first and apply the scale after,
which is what makes them fast and what lets a large activation reach infinity
before a small scale could bound it. Those rows fall back to dequantizing.
Sixteen cases, four for each of Q8_0/Q4_K/Q5_K/Q6_K (a tiny and a zero scale against a huge input, a tiny and an ordinary scale against an ordinary one), run with quantized activation dots enabled and disabled.
Q8_0 keeps the float dots in both settings; the other types also run decode dots over quantized activations, scaled per block and bounded against the sum of magnitudes.

`quantize-range` checks Q4_0 packing from finite F32 inputs against exact integer expectations: every code from -7 to 7 at every position, with power-of-two scales across exponents -149 through 124, half-way rounding, rounded subnormal scales, output guards and a scale that underflows even in F32. Binary16 scale underflow and overflow remain format limits; this checks defined packed codes, not finite decoded weights for every magnitude. It runs with gradual underflow and does not establish nonfinite-input handling.

`matrix_precision.hpp` is the shared CPU and Vulkan unit-weight regression: two signed inputs beside a peak cover small-value preservation and distinguish F32 from block-int16 arithmetic through each of the six matrix calls, prompt/decode/mixed runs, widths 256/3840/4096/4352 and output tails for Q4_0/Q4_1/Q4_K/Q5_K/Q6_K. It checks output guards and actual execution witnesses. CPU K-quant products must also match the independently expected F32 or block-int16 reconstruction for each row, beside their exact path lists, at one and four threads.

`q8-dots` checks those decode dots (`backends/cpu/q8_dots.hpp`), 16-bit for
Q4_0/Q4_1/Q4_K/Q5_K/Q6_K/MXFP4 (Q8_0 uses F32 inputs), against a double-precision
reference fed the same quantized activations, and that a decode row computes
the same alone, beside other rows and in a grouped call, bit for bit. It also
checks activation quantization independently against the original inputs over
float exponent boundaries, tiny/subnormal blocks and reciprocal-overflow
thresholds: packed range/sign/zero, exact integer sums, nearest reconstruction,
ties to even, and the blocks beside a range that crosses a row boundary left untouched, with the input unchanged.
It holds the K-quants' prompt dots at a width of 4096 to the same reference, a decode row beside a prompt's rows to its bits alone, and the routed products of every type: each generated token's entry the bits of a one-column matmul of its expert, a prompt's entries within a stated bound of them, and the down projection their weighted sum.
Dense Q4_K/Q5_K prompts also match their one-column prompt calls bit for bit at
the eight-column packing threshold, two-output-row by sixteen-column tiles,
four-output-row by eight-column tails, widths 4096, 4352 and 14336, and 1, 3
and 6 threads. Column counts on both sides of 16 and 32 cover repeated tiles
and partial columns, with even and odd output rows. Signed finite half-scale extremes
and input magnitudes from zero through 1e18 cover grouped projections, residual
adds, a generated row beside prompt rows, unchanged inputs and output guards.
Fallback reconstruction checks are
strict; ordinary SIMD controls allow its existing float-rounding error.
The shared matrix-precision check above holds the K-quants at their prompt-width boundary numerically as well as by their execution witnesses.
These run under gradual underflow;
they do not establish nonfinite-input handling or flush-to-zero behavior.

`qwen35-ops` checks the CPU backend's ops of the qwen35 layers (`backends/backend.hpp`, `docs/QWEN35.md`) against references the test writes from the math in double precision, each within a bound stated beside it as a count of F32 units of the values' magnitude.
It covers the causal conv and the raw rows it carries, which must be the inputs bit for bit, and the gated delta rule's rows and state, at Hv = Hk and Hv = 3 Hk with the tiny fixtures' widths, at a width that takes the 32-column blocks and at the files' 128 by 128: fresh sequences whose slots hold NaN, histories shorter than the conv's window, one-token entries and a verify that reads one slot and writes another, side by side in one call.
Each V column of the recurrence must give the same bits in a 32-column block, in an 8-column block and alone, and q and k heads near 1e-4 hold the L2 norms' epsilon and its place inside the root.
A decay factor below 2^-126 must be flushed to 0 and one just above it kept, on a state whose decayed values stay normal, so the check holds under any denormal handling.
It also checks the gated norm, the partial rope, reading q between its gates and k in place, and `sigmoid_mul` as the output gate and as a scale of one value per row; the norms each with a head near 1e-4 that holds their epsilon.
The partial rope's reference rotates every pair at the token's position, as the rope sections give it for text, so it holds nothing of the sections themselves, and at the full width over contiguous heads, as Qwen3 calls it, the op must meet the same reference; it must rotate a pair in its scalar tail with the bits of the same pair in its vector body, and an RMSNorm row 12 wide must scale each element of its scalar tail with the bits of the same element in its vector body.
Every result must be the same bit for bit at 1, 2, 3, 5, 8 and 16 threads, for a row alone and beside others, with the views in another order, and with a sequence cut into passes of 1, 2 and 3 rows that carry its state in its slot, so k one-row calls equal one call of k rows.
`state_alloc` must zero-fill its slots, `state_copy` copy one slot in every layer and a storage whose buffers cannot hold its slots be refused; malformed views, among them each of the three ways one view's slots can meet another's, must be refused before anything is written.
The embedded drafter's two ops: `argmax_rows` over rows beside each other, a tie taken at its lower id and an infinity, a NaN first or after the largest, a row of -infinity and a row whose prior id is invalid each given the invalid id, and `embed_ids` giving the table's rows for valid ids and a zero row for an id past the table.

`arch-qwen35` checks the qwen35 module (`model/arch/qwen35.hpp`) on tiny models it builds in memory, four layers of linear then full attention twice.
Every key the reader refuses must be refused with its text: two MTP blocks or one for every block, a value width apart from the key width, an odd rotary width or one past the head, rope sections that do not cover the rotary width, a conv of other than four taps, V heads that are no multiple of the K heads, an inner size of part of a V head, KV heads that do not divide the query heads, no layer kinds, a recurrent-layers array of another length or with a recurrent MTP block, and scaled rope.
The plan must give each layer its kind, its cache (a state or KV), its roles and ops, the conv's taps as a table, the state's shape, rope tables of half the rotary width a position and the arena's slots, and take the kinds from the recurrent-layers array when a file has no interval, leaving its MTP block unplanned; a layer holding the other kind's first projection, a router, missing conv taps, conv taps with HF's middle axis and a decay of the wrong width are refused with their texts.
The footprint must count a KV layer's cache by the budget and a state layer's by the slots, and not count the conv's taps as a product.
The runtime's rules for a model that keeps a state: a backend without one of its ops refused at load naming the layer and the op, `keeps_state`, a fork without a checkpoint refused, a second sequence refused while the one slot is held, a failed first pass leaving no loss, and a failed step without a checkpoint going back to 0, the history then giving a fresh one's bytes.
Its checkpoints, on a model of a 512-token context whose prompts pass a CPU block: a 200-token prompt keeping its state at 128 gives the logits of one that does not; a retract to 150 reaches 128, and the prompt continued from there gives the same logits and decode; a keep entry's checkpoint, forked and read in place beside its source's continuation in one pass, gives them in both, a fork elsewhere being refused; `keep` makes the live state at 200 the checkpoint in the one slot, replacing the one at 128, decode after it unchanged and a retract past it reaching it; a failed step goes back to the checkpoint and the history goes on with the same bits; and the slot is taken and given back as counted.
Its 128 tokens and checkpoint copied to host memory (`Model::save_host`) are restored (`Model::restore_host`) into a checkpoint slot of their own once the one slot is free, and continue the prompt with the bits of the prompt read in one; a copy where no checkpoint is refused, and a restore with every checkpoint slot held throws and takes nothing.
Its state alone at 128 (`Model::save_host` without blocks), whose runs add up to its bytes (`HostHistory::device_bytes`), once the history has kept its state at 200 instead, continues the prompt from a fork of the history at 128 with that state (`Model::fork` with a state) with the bits of the prompt read in one, beside the history's own continuation in the same pass; a fork with a state at another length, with a whole history's copy or with every checkpoint slot held, and a restore of a state alone, are refused.
The automatic checkpoint count (`PlacementRequest::fit_checkpoints`) is tried through the fit: on a split at shares 1 and 3, whose first device holds only a linear-attention layer's states and has room for the live states and not one checkpoint more, the model is placed with no checkpoint slot, and on roomy devices with the four asked for; on one device with room for the whole budget and no more, the checkpoint slots take at most a quarter of it in whole blocks of 128, the four asked for leaving 384 of 512 tokens and none taken of 384, and a request for as many as a size holds ends with a count that fits; a device whose free memory reads one byte for its first two reads and then has room for everything takes the four asked for and the whole budget, the count chosen once that memory has settled; two cards fitted to their free memory, the first's coming back after eight reads, give the plan, the KV budget and the checkpoints of idle cards; and the KV tokens the checkpoints took (`PlacedModel::checkpoint_kv_tokens`) are 128 where four slots took a block of the 512, and none where they took none or the budget stayed whole; live and checkpoint slots whose sum a size cannot hold are refused.
Marks past the first take only the room the budget leaves (`PlacementRequest::fit_marks`): of six asked for, one device takes one where it holds the whole 512-token budget with one and no more, three where it holds three and where it holds one byte short of four, and six where it is roomy, the budget whole each time.
Drafting takes only the room the no-drafter fit leaves: on one device that holds the MTP fixture's 512-token budget and nothing more, the embedded drafter with a mark, and a mark alone as lookup asks, are refused by their text, and on one that holds the budget with the drafter and one mark beside it the budget is the 512 tokens and one mark is taken; one byte short of the budget, the drafter, a mark and one checkpoint, the automatic checkpoint gives way and the budget stays 512 tokens, as the fit without drafts keeps it with its checkpoint, while a checkpoint asked for by number is refused; and roomy devices of a split whose first stage holds only a linear-attention layer's states give the budget and the four checkpoints of the fit without drafts.
A pass refused for its rows or its logits rows, for a sequence listed twice, through `begin_pass` and `forward`, or for fewer free slots than its fresh sequences need takes no slot, nor does one each of whose planning allocations fails in turn, so a valid pass of another fresh sequence follows each, and a sequence with a history keeps its slot through them and continues with the bytes of one never refused.
A prompt in slices of 1, 3 and 16 tokens with its decode, two sequences in one pass against each alone, and a split over two CPU backends whose first stage holds only a linear-attention layer, which keeps a state and no KV, against one CPU, must all be bit-identical.
The same model as a qwen35moe file, each dense block replaced by 4 experts, 2 a token, and a shared expert with its gate, must refuse a missing shared expert width, more experts a token than a layer has and another gating function by their texts, plan every layer routed with its stacks streamed into a window and its norm, router and shared expert copied, and give the same bytes over slices, two sequences, that split and one that runs every feed-forward part on the second backend, as experts on the CPU beside a device run.
With an MTP block after its four layers and a context of 512, as an embedded drafter (docs/SPECULATIVE.md, section 7): the drafter's plan, a file without the block and a backend without `argmax_rows` refused by name, the logits of the prompt and of decode unchanged with the drafter loaded, a draft of an empty history and one without a drafter refused; and a history's drafts and every draft row's logits must be those of a fresh sequence fed the same tokens in the same row classes, bit for bit, for a prompt in slices, a history beside another sequence in every pass, three histories drafted in one batch with chains of every length and one of none, each row's logits read by its sequence, a verify after a mark retracted to each of its 0 to 5 kept rows, a fork at a checkpoint at a CPU block's end and a retract to it, a reset, a history continued after a failed pass, and over four CPU stages; a sequence in the slot a NaN carried row was left in reads a zero row in its first pass; a history copied to host memory at its checkpoint and promoted back into the other checkpoint slot drafts as one never evicted, the carried row going with the slot, and so does a fork from its checkpoint's state alone in host memory (`Model::fork` with a state); an argmax marking the second draft invalid keeps one draft; and on two KV blocks, one held by another sequence, a draft inside its last block takes none, a draft needing another fails and changes nothing, and once the block is free it drafts as a fresh history; a drafter away from the last stage is refused; and every product of the drafter's context rows, draft steps and head takes the activation dtype the model was asked for, f32, f16 or bf16, as the target's do.

`prefill-scope` uses self-generated model fixtures to check caller-once execution,
nesting/thread guards, allocation and microbatch boundaries, error draining and
scope reuse. Windows-only `prefill-placement` covers real eligible topology,
unsupported topology fallback and synthetic apply/restore failures without
requiring a six-core hosted runner. Neither replaces the independent HF gate.

`mxfp4` checks CPU decoding at every exponent and code in every position against an independent oracle, scale-product boundaries, one-hot dots, odd block counts, extreme-scale cancellation and guarded outputs. `tests/mxfp4.py` checks dense tied/untied and MoE raw-block fixtures against pinned HF, including prompt lengths 39/40/41 and 63/64/65 around the two measured narrow GPU tile thresholds at microbatches 1/3/65 and NLL windows 40/64/128. The existing short references are unchanged. It also requires 150 original-F32 control cases at 2e-5, whose tool requests F32 explicitly and requires only completed F32 matrix paths; production logits and NLL use the frozen bounds of the completed matrix paths through `common.hf_bounds`. `tests/baseline_mxfp4.py` checks the pinned real writer output against file-exact HF and its separate approved quality bounds (docs/ASSETS.md, MXFP4 CPU fixtures). On a selected device without the type, the component reports SKIP; requiring MXFP4 turns that refusal into failure. The server's MXFP4 subcheck uses the same refusal policy.

`dtype` checks exact BF16 conversion against mathematical nearest neighbours over every finite encoding and both signs, plus overflow, infinity and NaN. Independent unit-weight products distinguish F32, BF16 and block-int16 activation arithmetic across dense, grouped, additive, head and routed calls, prompt/decode/mixed rows and thread counts. Dense and MoE models with different policies alternate on one shared backend and match isolated instances and split execution; this establishes policy application and isolation, not the independent HF gate or full F16 conformance. It holds actual dispatch evidence against these products and requires evidence to clear between measurements; the shared matrix-precision check above owns the K-quant threshold cases. It also checks automatic resolution for declared F32, supported declarations, mixed-device intersections, missing devices and the record, then proves placement applies its selection with a model whose BF16 and F32 logits differ. Backend path catalogs name packed, BF16 and retained F32 matrix families without changing native preferences or manufacturing execution witnesses; the CPU placement/health record carries those families, and backend-vulkan holds its catalog to the device profile's tile choice.
It also holds the int8 dtype to the rule every value follows: auto never chooses it, a placement whose every device lists it resolves to it, and a device that does not list it runs the next wider dtype it has, F16 where it has it and F32 otherwise, the record naming that dtype with the warning, while a run whose every device widened to one dtype is that dtype's.

`fp16` checks binary32 to binary16 conversion without an oracle library:
every finite half must encode back to its own bits, and floats at a fixed stride through the magnitudes below 2 of both signs, with eight chosen values such as the largest half and the smallest normal and subnormal ones, must each encode to the nearest half, ties to even.
Named cases pin the mantissa's carry into the exponent, overflow by rounding to infinity, both zeros, an underflow to zero, infinity and a NaN that keeps a payload.

`cpus` holds the automatic worker count of `core/cpus.hpp`, and the cgroup reading of `core/cgroup.hpp` it goes through, on file texts and field values rather than a real cgroup or job object.
It reads cgroup v2's `cpu.max` ("max 100000" as no limit, "600000 100000" as 6 CPUs, "150000 100000" rounded up to 2) and v1's `cpu.cfs_quota_us` over `cpu.cfs_period_us` (-1 as no limit), every malformed text as nothing, the process's cgroup in `/proc/self/cgroup` for v2 and for v1's cpu controller, refusing a path that is not absolute or climbs, and the cgroup mounts in `/proc/self/mountinfo` with their roots and escapes.
Over a file system held in a map it reads the quota in the process's own cgroup, found under its mount by taking the mount's root off its path, and in each cgroup above it up to the mount point, v1, v2 and both, the smallest winning: a child cgroup inside a container is read, a cgroup inside one that shares a name with the container's path on the host is not, and a path no mount's root holds reads nothing.
It reads a job object's CPU rate hard cap as CPUs of the active processors, rounded up, and takes the automatic count as the fewest of the hardware threads, the affinity and the quota, 4 when none was read, from 1 to 64.

`host-memory` holds what `core/host_memory.hpp` counts as the memory a process can still take, on file texts and field values rather than a real cgroup or job object.
It reads cgroup v2's `memory.max` ("max" as no limit) and v1's `memory.limit_in_bytes` (2^62 bytes or more as none) less the working set, the usage (`memory.current`, `memory.usage_in_bytes`) less the inactive file pages `memory.stat` gives, `inactive_file` on v2 and `total_inactive_file` on v1 and never the other's key or `inactive_anon`.
A working set below 0 is 0, a room past the limit is 0, a `memory.stat` that is missing, lacks the key or gives it as anything but one decimal number leaves the usage whole, and every other malformed text reads as nothing.
Over a file system held in a map it reads the room in the process's own cgroup and in each above it up to the mount point, v1, v2 and both, the smallest winning: a parent's tighter limit is taken, each cgroup's `memory.stat` is read at its own level, v1 is read in the memory controller's cgroup and not the cpu controller's, and a limit without its usage or a malformed file is passed over, the limit above a malformed `memory.max` still read.
It reads a job object's process and job memory limits less the process's and the job's commit, and takes the figure as the fewer of the host's available memory and that room, each only when it was read, so a container limited to 8 GiB is given what its limit leaves and, without the limit's files, the host's figure stands.

`sampler` calls `infer::sample` on hand-picked logits with expectations taken from the definitions.
Temperature 0 takes the largest score and the lowest id on a tie.
The repetition penalty divides a seen token's positive score and multiplies a negative one, by the penalty and once however often the token was seen, so a repeated leader loses to the runner-up.
`top_k` 1 is greedy at any temperature, `top_k` 3 never draws outside the three best, and a `top_p` between the first probability and the sum of the first two keeps exactly those two.
The nucleus is measured after the temperature and within the `top_k` window, which default requests combine: at temperature 2 a `top_p` of 0.7 keeps three tokens, and `top_k` 2 with `top_p` 0.6 keeps one.
Draws at temperatures 1 and 0.5 fall within three binomial standard deviations of the softmax at that temperature, a bound that rejects the temperature applied twice or ignored.
A seed repeats its sequence and seed 0 keeps the default state; every seed is fixed, so the frequency checks draw the same tokens on every run.
A masked id, the end of text under `ignore_eos`, does not exist for the draw: greedy takes the best of the rest, the penalty cannot bring it back, `top_k` 3 keeps the three best of the rest, a `top_p` of 0.7 is measured without it, draws follow the softmax of the rest, an id past the row changes no draw, a masked id 0 is not given beside scores that are all negative infinity or NaN, greedy or drawn at top-k 0 and 1 with top-p 0.95 and 1, and the settings' overload masks the end id only with `ignore_eos`.
Draws on random rows are also held, each with the generator's state after it, to a slow reference in the test that sorts the whole row but a masked id by score and then id and sums the softmax in id order.
The draws pin the ranking, the tie rule, the order a draw walks the tokens in and the generator's use; a sum in another order than id order differs only in its last bits, so no draw shows that order, and the reference only follows it.
Its rows are random, 1000 scores on five or twelve levels so that ties fall across every cut, or spread as a model's are, at top-k 0, 1, 40 and 1000, top-p 0.1, 0.95 and 1, temperatures 0, 0.2, 0.8 and 1.5 and penalties 1 and 1.1, each cell twelve draws from the next seed of a fixed sequence with the history growing, and half the cells masking the row's leader, each top-k, temperature and top-p under one penalty.
Rows of 40000, on five levels and spread, at top-k 0, 40, 5000 and 40000 with three draws a cell, take a nucleus past the 512 tokens the sampler ranks a nucleus with by heap and through the selections that double it, and a top-k of 5000, past the 4096 of a top-k set, through its selection over every key.

`infer::accept` is held there too, against the loop without drafts written out in the test, over 3000 random verifies of 1 to 8 drafts whose twelve scores sit on five levels: greedy, seeded at top-k 0 with top-p 0.9, and top-k 5 with a penalty, an end token and a limit falling at every row, drafts the loop's own picks with one replaced at a random place; it must stop at the loop's row with its pick, and leave the generator and the history where the loop does.
`spec::Acceptance` is held to its rule: ten empty verifies from its start still draft, the eleventh rests the request, 16 tokens stepped end the rest, and a verify keeping three moves the same average back above the break-even; its counts by draft position are every verify's fed positions and the prefix each kept; and `spec::draft_length` gives no more drafts than the decode columns it is given; a position's chance of being kept (`Acceptance::keeps`) is its share weighed with four verifies at a prior's share, or one half.
The pass price: `spec::PassTimes` fitted to passes of 34 ms and 10 ms a row and chains of 4 ms a step and 1 ms a chain gives those numbers, passes of one width price nothing, and a thousand passes of one width forget the spread; `spec::draft_depths` gives a lone decoder that keeps most drafts all three, one that keeps few none, eight decoders keeping half none, three requests beside each other the depth the most tokens a millisecond give within each cap, leaving out the one that keeps few, and every cap where the price is not known, but none where every pass measured had the rows the caps give, so another width is measured.

`spec` holds speculative decoding's round (docs/SPECULATIVE.md, section 3) to the run without drafts: `infer::generate` with test-only proposers that keep every draft, miss at the first, second, third or last draft of every verify, draw at random, or propose the end token, the vocabulary's last id and one past it, and with lookup, at 1, 4 and 16 drafts, must give the ids, the fed history and the logits after one more token of the run without drafts on one CPU.
It runs on the synthetic Q8_0 model over one and two CPU stages, a tiny qwen3moe model over one and three and the hybrid qwen35 model of `server-resume` over one, two and four, greedy, seeded at top-k 40 and top-p 0.95, and at temperature 1.5 with a penalty, after prompts of 10, 127, 128 and 129 tokens, so verifies start on each side of a 128-token block, with and without an end token that the greedy reply meets; a stop text inside a verify and a limit at each of 12 tokens end a reply as without drafts, and drafts that are all kept must take fewer than a quarter of the passes.
The hybrid model with an MTP block after its layers, whose weights follow theirs, drafts with its embedded drafter over one, two and four CPU stages at 1, 3 and 8 drafts, greedy and seeded, from prompts of 10, 127 and 129 tokens, each run feeding drafts, and must give the run without drafts of the file without the block, which the file with it gives too.
Then the history calls: on the hybrid model over one, two and four CPU stages and the dense one over two, rounds through 300 tokens that mark, feed 2 to 17 tokens as one verify and retract to keep none of them, some or all, each verify row and each step after a retract to the mark the bits of single steps, so the state's rerun from the mark is held at every length a verify can reach.
`arch-qwen35` holds the mark's ownership: a marked sequence destroyed at length 0 and holding the live slot leaves the one mark and the one live slot for the next, a mark moved by construction and by assignment goes with its sequence so a reset of the one moved from releases nothing, a marked sequence assigned over returns its mark, and a mark whose allocations fail in turn changes nothing.
A rerun that fails, injected into the conv, fails its retract, leaves its backend recording in order and keeps the mark, and the retract called again reaches its length with the bits of single steps; a second mark, a pass of more rows than a mark saves and a second pass after a mark are refused, a retract keeping every row leaves the history, and a model without mark slots gives none.

`logprobs` checks `inference/logprobs.hpp` against a double-precision reference that shifts by the row's maximum and sums with compensation: every value of 151936-token rows of three spreads, a row with one certain token, logits too large for `exp` unshifted, logits wholly below -1e30, equal and tied logits, within half a float step, the probabilities summing to one, `token_nll` the same quantity in double, and the top lists of 0, 1, 5, 20 and more tokens than the row in the order of a full sort by logit and then id.
It then runs the scheduler over `tiny_qwen` with a tokenizer of its 16 tokens and no end token: a greedy request with `top_logprobs` 5, read from its token channel, must carry at every position the log-softmax and top five of the logits row a second model gives through the same passes (the prompt at its extent, then one decode entry a token), and the same request without logprobs the same ids and no values; so must a request drawn at temperature 1.5 with penalty 1.3 and a seed, whose values are the raw row's though some of its draws are not greedy's choice.
The same request left unread until it ends must hold exactly `Request::kRowsWaiting` rows on its channel and then give the same values, those past the rows computed by the scheduler; cancelled before it is read, its waiting rows come without values and the rest with the scheduler's.

`server-resume` runs uncapped greedy requests with `top_logprobs` 5 through the scheduler over the synthetic Q8_0 model with a token list and no end token, whose decode and prompt rows take different CPU float reductions, on a pool too small for them: each alone first, where it never pauses, then together, where the scheduler pauses and resumes them, and every id, logprob and top entry must equal its run alone, a difference naming its first token.
Once every request has ended, the blocks the scheduler holds reserved must be its donors' blocks and never more than a pool has (`Stats::reserved`, `Stats::donor_blocks`).
With three requests the resumes must have recomputed more rows than each pause's longest prompt (`Stats::recomputed`), so generated tokens among them.
The cases: three requests on 8 blocks of 128 tokens, a victim paused with generated tokens in its partial block whose donor the growth that paused it then takes; a prompt read one token a pass (`ubatch` 1) paused while it is still prefilling, which takes nothing back and recomputes at least the two blocks of its prompt it had read; a follow-up turn that forked the previous turn's reply rows, paused with every donor gone, which recomputes its whole 512-token history; the three requests over a two-CPU layer split; and paused requests cancelled, after which no request is active or queued and the blocks held reserved are the donors', and, once stopped, the pool is empty.
The cases of a single pause end the paused request with a stop string found from its reply alone, or cancel it, so exactly one pause happens: an uncapped request paused by an earlier one's growth with its donor intact, which takes the donor back whole when a capped request ends (`Stats::taken_back` 1, nothing recomputed); the same for a follow-up turn whose history holds the first block of the previous turn's history, forked at its first admission; a request whose own donor went while a capped request that forked the same setup prefix kept that block, which it forks and then recomputes the other 384 rows by class; the same where the capped request's prompt holds the paused request's prompt and its first 80 generated tokens, so its donor matches that history by tokens past the first block but holds those tokens as prompt rows, and the resume still forks only the first block; and a paused request cancelled while its donor waits, which takes nothing back and leaves the pool empty once stopped.
Three cases hold room to first admission: a resumed request whose next growth step finds only room that capped requests hold sits out passes for it rather than being paused again (one pause, one donor taken back, each reply its reply alone), a request refused room evicts no donor and starts only once the capped request holding the room has ended, so a prompt repeating that donor's prompt still forks its block, and with a queue of one a request submitted while another is paused is queued rather than refused and starts only once the paused request has resumed.
A request whose growth step a capped request's room holds sits out more than a block of passes, every reply its reply alone and ending by its length or stop text, and a capped request submitted meanwhile starts only once the capped request holding the room has ended.
An older request's growth step that falls due in the iteration a newer capped request could first be admitted takes the room first, so no request sits a pass out, and a paused request cancelled while its donor holds less than a block leaves no donor behind.
Two cases act from inside a pass's stage through a CPU backend that runs a hook as the model submits its work: on two and on three CPU stages of a four-layer model, a request cancelled while its pass is in flight ends only once that pass has retired, its history kept as a donor that a request repeating its prompt forks (128 tokens reused); and on one and on three stages, a stop while a pass is in flight ends the request cancelled with nothing left reserved.
After either, a prompt as long as the pool runs on the model's own history, so every block came back, a block of a request whose handle is still held included.
In a build with Vulkan, when device 0 opens, it repeats the three requests on the device and adds a 100-token prompt sharing the first block of a 600-token prompt's history, whose tile split another way, so it forks nothing, paused with every donor gone, which takes nothing back and recomputes at least its first 384-token reservation.
A hybrid model, a qwen35 of four layers with Q8_0 matrices whose linear-attention layers keep a recurrent state, holding three states and no checkpoint slots, keeps no donor: three uncapped requests on one CPU and over a two-CPU split with two passes in flight, a victim still prefilling and a follow-up turn repeating a finished request's history each give their replies alone with no donor, nothing taken back and no prefix reused.
With checkpoint slots it keeps a request's state at its prompt's last whole block: a follow-up turn repeating a 300-token prompt and its reply forks the 256 tokens there and gives the ids and values of its prompt on a fresh model, on one CPU and over a two-CPU split; with one slot a newer conversation's checkpoint takes the older's, so the older's follow-up reuses nothing and the newer's 256 tokens, each equal to its prompt on a fresh model; and a paused request keeps its whole history as its checkpoint and takes it back whole on resuming, as the take-back case above has it (one pause, one taken back, nothing recomputed), every reply its reply alone.
A donor parked at its checkpoint holds no live slot: with one live slot the same 300-token prompt again forks its 256 tokens and gives the same reply, and with three, three requests at once beside the donor each give their reply alone.
Paused requests cancelled, a request cancelled in flight and a stop in flight on three CPUs leave every state slot and every block free, so three requests at once then run whole, and a scheduler of four requests at once over the model is refused as it is made.
Donors kept in host memory (docs/SPECULATIVE.md, section 2, Host tier): two conversations of a 300-token prompt alternate on a pool of 4 blocks of 128, so each turn evicts the other's donor; with a host tier each follow-up promotes its own donor back and forks its 256 tokens with the reply it gives on a fresh model, on the synthetic Q8_0 model and on the hybrid one with a checkpoint slot, on one CPU and over two, and without one reuses nothing; a write-back whose copy throws keeps nothing, and a promotion whose copy throws keeps its entry and takes nothing, every reply still its reply alone.
With one request at a time and host memory for one copy, a promotion whose room evicts the device donor the request would otherwise fork, and whose copy then takes the promoted entry's host memory, fails, and the request runs from what is left with its reply alone.
A re-prefill job's donor supersedes its conversation's other donors: on 16 blocks of 128 a request needing one donor's room evicts the request's donor the job forked, though an unrelated conversation's is older, and with a host tier copies nothing to host memory, so a prompt repeating the unrelated conversation forks its 256 tokens and the follow-up forks the job's 384; after two turns each read again, a request needing the whole pool evicts every donor and host memory keeps the unrelated conversation's and the second job's alone, the first job's superseded by its tokens, so both later prompts fork from host memory; every reply its reply alone.
Host memory keeps the conversations that came back against newcomers: one request at a time, a prompt and its repeat with 200 more tokens, then three new prompts, each evicting the donor before it; with room for two copies the repeat's copy stays while the newcomers take each other's room, so a prompt repeating it forks its prompt's 384 tokens, and with room for one the first newcomer is refused and the second, the tier having refused as many as it holds, evicts the repeat's copy, so a prompt repeating the second newcomer forks its 256 tokens and one repeating the repeat forks nothing; every reply its reply alone.
With two donors at most, an unrelated conversation's and a conversation's request's, the conversation's job completing gives up its request's donor and not the unrelated one, so both later prompts fork, and with a host tier nothing has been copied once the job completes.
A host copy renewed by a job donor of the same history stands for it: a conversation's job donor evicted to host memory, promoted for a regenerated reply that is the same reply and superseded by that reply's job, is renewed when long requests evict the new job's donor, and two more long requests pressing host memory for two copies leave it, so the follow-up forks its 384 tokens from host memory.
Message boundaries, on the hybrid model with three checkpoint slots and a host tier, on one CPU and over two: a conversation of six turns, each reply read again, keeps each job's state in host memory, thinned to four with the first kept; after a 2000-token request evicts every donor, an edit of turn 2 forks the first boundary with the last job's blocks promoted from host memory and a regenerated turn 6 the newest, each giving its reply on a fresh model.
With copies to host memory failing no boundary or host copy is kept, and with copies from it failing an edit of turn 2 whose boundary beats every donor fails its fork with the state and reads its prompt from the start, its reply alone either way.
A disk tier under the host tier (docs/DISK-TIER.md, Demotion), its root a temporary directory: eight conversations taking turns one at a time on the synthetic Q8_0 model, on one CPU and over two, into a host tier of four copies, so after turn k the k - 3 oldest copies are on disk, written while still in host memory and released at once when room is needed, the entry files in the server's directory those counted with their bytes and none temporary, and a clean exit leaves no directory; with disk room for two entries the oldest files go and the two newest stay, all four written; a conversation's first job's donor written to disk, promoted for its second turn and superseded by that turn's job leaves the disk, nothing else written; and with every write held in flight a minute a chunk (`DiskOptions::pace`), the turn whose copy needs the room of the copy being written runs to its end in under 20 seconds, the write cancelled, its temporary file gone and nothing kept, the copy that needed the room kept; every reply its reply alone.
Read back from disk (docs/DISK-TIER.md, Restore): after six such turns the first conversation's copy is on disk alone, and its follow-up waits for the read, is promoted and forks its 256 tokens with the reply it gives on a fresh model, on the synthetic Q8_0 model and on the hybrid one with checkpoint slots, on one CPU and over two, prompts read in passes of 16 rows so no prompt rate bounds the wait; with reads held two seconds a chunk, an unrelated request submitted after the follow-up runs to its end while the follow-up waits and is admitted after; a follow-up whose file has a byte flipped computes its history, the failure counted; with prompts read whole, so the measured rate bounds the wait, a read held two seconds is given up and the follow-up computes and ends first; and message boundaries on the hybrid model under a host tier of three copies a device, the edit of turn 2 and the regenerated turn 6 reading their boundaries back from disk; every reply its reply alone.
Kept across a restart (docs/DISK-TIER.md, Keeping entries across a restart): under `--disk-cache-keep` three conversations into a host tier of four copies write nothing until the scheduler stops, which writes the two host copies and the device donor and leaves its directory marked kept; a second scheduler on a fresh model of the same file adopts the three, and two follow-ups read theirs back and fork 256 tokens with the replies they give on a fresh model; a third leaves out an entry made two days old; and with an age limit of five seconds the entries six turns leave on disk are deleted, their files with them.
A reply read again as prompt rows (`Scheduler::follow`, docs/SPECULATIVE.md, section 2): a 300-token prompt's reply and two closing ids, given as the ids its next turn begins with, are read by a job on a fork of the request's history and kept as a donor at their last whole block, on the synthetic Q8_0 model and on the hybrid one with checkpoints, on one CPU and over two, so a follow-up turn of those ids forks all of them, past the reply, and gives its reply on a fresh model.
Idle after a 100-token reply the job reads 128 rows in passes of 16 (384 tokens forked), and a regenerated reply, the prompt sent again, forks the request's own donor, which stays beside the job's, at the prompt's 256 tokens and gives the same reply; while a 400-token reply is written, ids given as 100 and 380 tokens are written let the job fork the running request and read all 640 tokens beside its decode rows, so the whole ids leave nothing to read, which a job made whole after its last pass must still complete; and given only the first, with the follow-up sent as the whole ids are, the follow-up forks the running job's 384 tokens with one pass in flight.
A job whose ids grow from 384 to 896 tokens while a 700-token reply is written reserves the blocks they take first, so on 16 blocks a request needing 5 makes it give way rather than running the pool out, and every reply is its reply alone.
Over a two-CPU split with two passes in flight and a 256-row budget, where a request is in flight in the pass beside the one formed: a job begun at idle beside a request decoding 200 tokens reads nothing from that request's first pass to its last and completes once it has ended, and a job begun while its reply is written keeps reading beside that request at most 64 rows a pass; each request gives its reply alone.
On the hybrid model a request that needs the room the job holds, submitted as each of the job's seven pass boundaries retires, cancels it there, gives its reply alone and lets the job complete after it, and one that fits beside the job leaves it running.
Over tensor groups of two CPUs (docs/TENSOR-SPLIT.md, step 2), each reply against its run alone on one group: three uncapped requests paused and resumed on one group and on two stages of groups, a paused request taking its donor back, a follow-up turn's fork against its prompt on a fresh model, and donors in host memory, each member's storage copied on its own.
`llmx-server-resume-test cpu` or `device` runs one half.
A first admission forks only rows of the classes it computes them in (`Model::row_class`): a follow-up turn repeating a prompt and its reply, on the CPU and on a device, gives the ids and values of the same prompt on a fresh model, reusing the prompt's whole blocks and not the reply's, and on a device a prompt sharing a block of a 600-token prompt of another tile split reuses nothing while one of the same split reuses its 448 shared tokens, each equal to its prompt on a fresh model.
The shared wave harness holds pass retirement while a wave is queued, so a descheduled submitting thread cannot let its first request run ahead and change the pause scenario; the first pass may start before queuing finishes.

`server-spec` holds drafting in the scheduler (docs/SPECULATIVE.md, section 3) to the run without drafts: requests verifying drafts beside each other must give, token for token, the ids, log-probabilities and top entries each gives alone without drafts, and drafts must have been fed and kept.
Over the synthetic Q8_0 model with lookup, and the hybrid model with an MTP block with its embedded drafter and with lookup, on one CPU and over a two-CPU split at a pass a stage, on CPU backends that report 16 decode columns so several requests draft in one pass: greedy and sampled, capped and uncapped, one ended by a stop string, each also alone with drafts, and three uncapped requests on 8 blocks that pause and resume; three requests each a lone decoder in a pass of its own over three CPU stages at three passes, each verifying 63 drafts, keep the draft rows in flight within the 64 the logits rows hold; the schedulers draft by the decode columns alone, and the hybrid model's embedded drafter runs again priced by the passes' timing, which need not draft, with every reply its reply alone.
A request cancelled while its pass is in flight, at each of several submissions so some cancels find a verify, on the Q8_0 model with lookup and the hybrid one with its drafter, ends cancelled, a request after it gives its reply on a fresh model without drafts, the ledger holds only donors and every block comes back.

`server-passes` drives the scheduler's policy core (`server/policy.hpp`), first by hand.
The host tier's default size (`host_cache_default`): `--max-seqs` histories below half the free memory, half the free memory at and past it, a sum that would overflow taken as past it, and none where the free memory is unknown or a history takes nothing.
The growth rule (`Growth`): admission reserves a capped request's history and what it may still generate and an uncapped one's history and a step, a step falls due only for an uncapped decoding request whose next position passes its blocks and reaches a step past that position, never past what a pool holds, the logits rows a context reserves are one pass's alone and twice that once passes overlap, and the decode share is the decoding requests over the passes rounded up once the passes fill the stages and every ready one with fewer; a prompt slice takes a ubatch, but for a request alone, nothing else active, queued or paused, about a 2 * stages-th of what it lacks and 128 rows at least, and beside other requests first the rows that bring what it read alone back to a whole ubatch, before any other prompt's slice and closing the pass to other prompts, and then a ubatch again.
`make_room`, the one owner of who gives up blocks for whom: a request that fits or cannot fit takes nothing, the donor it forks goes last or first, growth pauses the latest uncapped requests admitted after it, headroom before donor, and never one admitted before it or a capped one, and a plan that would pause a request in flight waits and takes nothing while one pausing only requests not in flight, or holding a capped request in flight, does not.
`round_steps`: each stage records its oldest waiting pass from the last stage down, a free slot none, and passes past their last stage retire oldest first.
The logits rows: runs follow the newest and wrap to row 0 but never over the oldest run, come back oldest first, a run given back early waiting for the ones before it, and an empty ring starts at row 0.
Then through the scheduler's round under a simulated executor, the pools, the growth rule, `make_room`, `round_steps` and the logits rows being the scheduler's own functions and the round's order restated from `Scheduler::run`: once by hand, where a growth step due beside a queued request that fits only in the room the step needs takes that room and the request waits, and in random schedules, 2000 by default (`llmx-server-passes-test N` runs N), over 1 to 4 stages, 1 to twice that many pass slots and two pools of different block sizes.
One schedule in four serves a model that keeps a recurrent state, where no request leaves a donor.
One request in three waits for a read of its history from disk (docs/DISK-TIER.md, Restore), of a random number of rounds, failing one time in four, under a random bound, and the round admits through the scheduler's own `admit_waiting`: a request admitted before one submitted earlier must find that one still waiting for its read, and across 1000 schedules or more requests must pass one waiting, reads must be given up at their bound and reads must fail.
The round is the scheduler's at any number of slots: a pass formed in every free slot while a request is ready, each taking the decode share of the decoding requests, those that left flight earliest first, each request in one pass at a time, and the host's stages recorded after the round's device stages.
A simulated executor runs stage s on device s, the host waiting for a stage's handoff before it records the next and for a pass's last stage before it samples, each stage taking a random time, on the host's thread for a stage on the host, one stage or every one of them in some schedules; submissions arrive with the host's time, and requests are cancelled wherever they are, in flight and in a pass formed but not yet recorded, passes fail at a random stage or as their logits are read, and each schedule ends in a stop at a random round or, one in eight, once every request has ended.
After every event a request is in at most one pass and no pass is empty, each device runs its passes in formation order, a slot and a run of logits rows belong to one pass until it ends, no pool is over-reserved and the reservations and donors add up, no more requests are active than there are slots, a model keeping a state holds no donor, nothing in flight is paused, parked or ended, a request refused room is refused only when donors and the uncapped requests after it could not give it, the oldest only when a capped request holds the rest, admission is first-come, a pass always finds its logits rows, and a free slot stays empty only when no active request is ready or a prompt read alone, in flight, holds the passes for its slice back to a whole ubatch.
A growth plan pauses one request at most, since one uncapped request's reservation holds a step, and waits only on a request in flight, and the oldest request's plan that waits takes that request in the round its pass retires, within a lap (S + P - 1 rounds) of the first wait; a younger request's plan may turn to another request in flight when an older one's step takes the one it waited on, and the drained schedules check that every request still ends.
Every decoder that is not stalled gets a token within a lap and a cancellation ends within a lap; a cancelled request in flight is not sampled when its pass retires, a failed pass ends its own requests alone and leaves every other pass in flight, and after a stop the ledger and the logits rows hold nothing; across 1000 schedules or more requests must pause, stall, take donors back and wait on requests in flight, the oldest request's wait must end, passes must fail, and requests must be cancelled in flight and before a pass's first stage, and a model keeping a state must pause.

`server-passes-cpu` runs the scheduler itself over the synthetic Q8_0 model on one CPU and split over two and three, at P = 1, S, S + 1 and 2S (one on one CPU, where two are refused as the scheduler is made), with a ubatch of 16, every request's ids, logprobs and top entries against its run alone on one CPU at one pass in flight.
A paused load, two uncapped requests, one sampled, beside four capped ones, three with prompts past the ubatch and some sampled, on 8 blocks, must pause on every run; a held load, two uncapped requests on 9 blocks whose older one's growth step falls due at 380 generated tokens and the younger's at 384, must pause once on every run and, at P = S on a split, wait on the younger in flight (`Stats::waits`).
A steady load that never pauses has each run's passes recorded as they retire (`Scheduler::on_retire`) and replayed in their order through `Model::forward` on a fresh model of the same placement, every logits row bit for bit; asked without log-probabilities, so no row is copied and no values are computed, it gives every id its run alone with log-probabilities gives, and no values.
Through a CPU backend that runs a hook as the model submits its work: a request cancelled at the last stage's twelfth submission, and in a second run at its thirteenth, ends cancelled with a prefix of its ids alone while the three others run whole; the last stage's twentieth submission throwing ends its pass's requests with the error, a prefix of their replies alone, and the others run whole, some always surviving on a split at P = S or more; and a stop at the first stage's twelfth submission ends every request cancelled with nothing reserved or in flight; after each a prompt as long as the pool runs, so every block came back.
With a ubatch of 256 on two CPU stages, a burst of four 512-token prompts queued together takes whole ubatches, never a cut slice, and a lone 768-token prompt's first slice is 192 rows, a quarter of it, and once a request submitted as that slice's pass retires its next is 64 rows in a pass no other prompt shares, before the new request's first slice, then whole ubatches; every reply its reply alone.
Every case runs again over the hybrid model of `server-resume`, whose layers alternate linear and full attention, so a stage of a split may hold only states.
The paused, held and steady loads also run over 1, 2 and 3 stages of tensor groups of two CPUs (docs/TENSOR-SPLIT.md, step 2) at the same P, every reply against its run alone on one group of two at one pass in flight, since a group gives its own bits, not one CPU's.

`sampling-pool` runs the scheduler's sampling threads (`server/sampling_pool.hpp`) with no thread and with one, three and four: every index of a job once, for jobs of 0, 1, 2, 5, 64 and 1000 indices and 2000 jobs back to back of 1 to 17, a single index on the calling thread, every thread and the calling one taking an index at once, each on a thread of its own, and calls that throw rethrown by `run` only once every call has returned, after which the next job runs whole.

`backend-errors` injects task and startup-allocation failures, checks completion
before error propagation, and exercises pool reuse and thread reconfiguration.
It also checks valid empty CPU transfers, rejected offsets/null sources,
unchanged storage and a zero thread hint preserving the current pool.
It pins that construction and count changes start no threads, and that the first dispatch at a count starts one pool of that size, which later dispatches reuse.
A start that fails partway fails its dispatch and keeps the count, and the next dispatch starts the whole pool without a new count.
It checks `quant::row_bytes` against the block layouts over one row and over several, zero rows taking zero bytes however wide a whole-block row, and its refusals of an unknown type and a partial block, with rows or with none, and of a size that wraps in one row or across rows.
It checks sizing of known layouts without decoders, refusal order and exact messages, and that the registry's metadata agrees with `quant::storage_type`. The decoder registry and CPU support must remain exactly F32, Q8_0, Q4_0, Q4_1, Q4_K, Q5_K, Q6_K and MXFP4; recognizing more storage layouts must not enable their execution.
Decode and batched `matmul` and `embed` must refuse a Q8_0 row that ends inside a block, and `matmul` and `matmul_group` must refuse row runs that reach past the call, in order or not, are out of order or fall short of it, all before writing any output.
It does not establish recovery of partially executed model sessions.

`backend-vulkan` exists only in a build with `LLMX_HAS_BACKEND_VULKAN=ON`.
Where the device runs int8, Q8_0, Q4_0, Q4_1, Q4_K, Q5_K and Q6_K rows 512 and 1280 wide take their 8-bit builds at every row count, 1, 3, 8, 13 and 64 generated tokens through the row kernels and prompts of 40, 64, 128 and 200 tokens, within 1e-3 of the CPU fed the activations rounded to 8 bits per block of 32, chosen at least 0.2 of a step from a rounding tie so the device's reciprocal of a block's peak rounds them as the host does, and witness `block-int8`; a prompt gives the same bits as one call and as two slices naming its extent; F32 and MXFP4 weights, which have no 8-bit build, give f16's bits under int8 and witness no `block-int8`.
The decode column checks below run again under int8, where the device runs it, over every 8-bit build.
Where a second device opens and the platform shares memory and semaphores between them, it first holds a tensor group's collective over devices 0 and 1 (docs/TENSOR-SPLIT.md, step 3): six sums in a row at 1, 7 and 64 rows, so each parity of the inboxes is written three times, every member's residual at an offset gaining the partials' sum in member order with the bits a float sum gives on the host, and the refusals of a collective joined by a member not first, over the CPU and over one device twice; with one device, or on Windows, it says why it skips them.
Its dtype workspace regression uses a 256 by 32769 F32 matrix and 513 sparse BF16 input columns, whose four split partials cross the 256 MiB target. Dense, head, residual and grouped calls check all 84052485 products against exact two-term arithmetic, with odd row/column counts, aligned input/output offsets and untouched output gaps. This tests column slicing, not a total device-memory bound.
It opens device 0, checks that its identity (`Backend::identity`) names the device and is the same for a second backend of it, round-trips buffers through adopt, copy, write and read, checks that a weight written in pieces into `alloc_weight` storage, or, where the device imports host pages, copied in pieces into it out of them, holds its bytes and gives the products of the same weight adopted, and that memory off the page or a part of a page is not imported, checks zeroed allocations, host-visible memory read in place after a wait, monotonic tickets, copies submitted around holds between submissions, after an idle gap and with a hold pending at teardown, and the refusal of a KV budget that overflows, then runs every implemented kernel against the CPU backend on random inputs with bounds fixed in the test: exact where the arithmetic is the same operation in the same order, a stated relative tolerance where a transcendental or a reduction order differs.
The row kernels and the integer-dot tile read quantized rows against 16-bit integer activations under f16, so the CPU reference is fed the activations quantized the same way and
the comparison is about the dots; the norm, SiLU and attention kernels' twin
of their output is checked through a matmul from it. Among the kernel checks it
prints, without asserting them, the time of a tiny dispatch, the matvec bandwidth per type and shape, the prefill tile's rate and the time of decode attention and the small kernels, and after every check, when the device reports them,
the driver's per-kernel statistics (registers, shared memory, scratch);
`--isa DIR` additionally writes the driver's disassembly of each kernel
to that directory.
A backend made to time its work must count in its device time every dispatch between two readings, 6000 of them past the 4096 one query pool holds, since the server's `--timing` reads each stage every 32 rounds.
With `--isa` every row kernel build must then hold its one-column build's counts of float multiplies, multiply-adds that round the product first, fused ones, adds and adds over shuffled lanes: the same counts where the driver keeps the column loop rolled, else those and whole columns of one column's, and for the Q8_0 decode kernel, whose builds also differ in rows, steps, copies of their products and forms, the counts its shape and forms give as the first line of its representation reports them: for each product in the code two multiplies and a separate add, with no fused multiply-add, the half-block order's step among them where the build takes that order, which must be where its one-column build does, and, beside those product adds, per row and column the one-column build's shuffled adds and one residual add, or with the transposed reduction that reduction's pairs and one residual add; a grouped build of the other row kernels must hold its wide build's counts, and the two-row builds of the Q4 and K-quant families, whose first line reports their columns and rows, must differ from each other by whole copies of one row and column's products, the same number a column in every pair, no more than the one-column build holds and in its proportion of fused products.
The Q8_0 decode builds are held to those counts only where the one-column build reduces over both shuffled and plain adds, and elsewhere to its kinds of operation alone; their multiplies also count the one-column build's multiplies beyond its products, and the transposed reduction is accepted only where the one-column build's reduction takes six levels.
The counts are a screen on how the driver contracts and reduces a column's sums; a reassociation that keeps them shows only in the decode-column comparison below, which is what holds batch invariance.
The instruction counter excludes the reciprocal scale of integer address division and its captured RADV five-instruction normalization sequence only when all registers match. Its ordinary native run checks this parser on captured instruction forms, retaining neighboring matrix products and unknown sequences, and checks Q8 shape counts with missing-multiply, missing-add and extra-fusion negative controls.
Where the captured K4 ISA contains native 16-bit integer dots, every captured Q6 row build must retain those dots too. This catches operand widening during centering even when the resulting scalar integer arithmetic is numerically exact; drivers without that recognized ISA witness are not covered by this instruction check.
CTest runs `backend-vulkan` without `--isa`, and no hosted runner has a GPU, so `llmx-backend-vulkan-test --isa DIR` is run by hand on an MI50 under RADV and on the Radeon VII under the AMD proprietary driver at every change to a row kernel, its builds or how a pass is chunked.
The ordinary test also checks the weight dispatch descriptor through timestamped kernel names: dense and routed calls on both sides of their crossovers, F16/F32/BF16 requests, odd and paired blocks, and unsupported storage IDs refused before operand access with their exact text. Expected names are independent of the descriptor; a device without timestamps reports this coverage unavailable, while its numerical tests still run.
Tiled attention covers heads 128 wide, and 256 wide six to a KV head as qwen35's 27B has them, over 32, 45 and 100 query rows after histories of 0, 70 and 600 tokens in f32 and f16 caches.
Attention additionally covers 120 combinations of head widths 32/40/64/128/256,
query/KV head ratios 1/2/3/4/6/8 and all four F32/F16 cache-side pairs, with nonzero
inputs at long histories. Both rows of a mixed short/long pass must equal the
same rows taken separately, bit for bit; CPU comparisons retain the bound
`1e-4 * (1 + abs(reference))`.
Every decode column of every row kernel build must be, bit for bit, the same column computed alone: calls of 1 to 64 generated tokens, and the residual add and a group of three projections at 1, 2, 3, 8, 9, 13, 16, 18, 29, 32, 33, 34, 36, 40, 48 and 64, whose remainders past the widest Q8_0 decode build and the widest two-row build take each of their builds, over F32, Q8_0, Q4_0, Q4_1, Q4_K, Q5_K and Q6_K rows 4096 and 1280 wide, the first four also 224 wide and Q8_0 also 2560 wide, the output head of Q4_0, Q4_1 and Q6_K on its own twin, and 8 experts routed 2 a token over 1 to 32 tokens on rows 4096 and 1280 wide, each token's gate, up and down against the token alone.
Rows 1280 wide give some lanes of a 64-lane subgroup three steps and every lane of a 32-lane one five, so a build that takes steps in pairs also takes one after them.
A group of a Q8_0 and a Q4_0 projection whose batch reaches the 8-bit tile crossover but not the other types' must equal each type alone forced onto the row kernel, bit for bit, on a device whose profile puts batches between the two.
A matmul whose row runs are out of order must be refused even when every run takes the same kernel.
An `embed` whose F32 or Q8_0 table holds fewer rows than the call names must be refused, even when every id is inside the table.
Every kind of refusal of `matmul`, `matmul_add`, `matmul_group`, the routed products and `embed` is also made in two passes, and a valid call after each pass must give what it gave before.
`alloc` and `adopt` leave a new buffer held by the command-buffer slot that fills or copies it, so the first pass submits until no slot holds the call's operands, makes the call and then drops them, and a command the call left naming one fails the next submission.
The second pass writes into an output that must keep what it held.
What a paused request's resume relies on is checked the same way: 40 generated rows of one sequence in one run of extent 1 beside a prompt's rows must equal, at rows 0, 1, 7, 8, 15, 31 and 39, the row in a matmul of its own, and their attention after a 500-token history in one view of extent 1 beside a 60-row prompt's view must equal every row decoded one call at a time.
The integer-dot tile and row kernels of every quantized type must hold the precision of 16-bit activations: against a double product of the unquantized inputs, whose every block holds one value 30 times the others, each output within half a 16-bit step of each block's peak times that block's weights, where 8-bit activations miss by the 8-bit step.
The qwen35 layers' ops meet the CPU: `norm_rope_partial` at the full width in place and reading q between its gates with a quarter of a 256-wide head rotated, at 1e-5; `gated_rms_norm` and `sigmoid_mul`, in place too, at 1e-5 and 1e-6; and the conv and the gated delta rule at 1e-5 and 1e-4 over the mixed sequences of `qwen35-ops` (fresh ones on slots of NaN, short histories, one-token entries and a verify), at the tiny fixtures' shapes, a 64 by 40 matrix, 128 by 128 and the 0.8B's and the 27B's heads, their carried conv rows bit for bit.
The embedded drafter's ops meet the CPU id for id and bit for bit: `argmax_rows` over rows of 37 and 248320 logits with a tie, an infinity, a NaN first and last, a row of -infinity and a row whose prior id is invalid, with and without prior ids, and `embed_ids` of F32 and Q8_0 tables with ids past the table, which write zero rows; and on the device `embed_ids` of every type it embeds, MXFP4 through its own build, must give `embed`'s rows bit for bit.
On the device a sequence's conv and recurrence rows and state must be the same bits alone, beside others, in reverse view order and cut into passes of 1, 2 and 3 rows, which compares the delta rule's short build, taken by a call whose views hold at most 8 rows each, with its 16-token build at 8 and 9 rows, a fresh slot's contents must change no bit, a decay below 2^-126 must be flushed and one just above it kept on a state whose decayed values stay normal, and `state_alloc` must zero every slot and `state_copy` copy one slot in every layer.
Gated attention's tail at head width 256 runs as the layer does, attention after a 90-token history, the output gated in place by `sigmoid_mul`, then a Q8_0 and a Q4_K output projection added to a residual, on the row kernel for three decode rows and on the tile for a prompt's rows, against the CPU fed the activations each kernel reads, so the copy the gate writes, not attention's of the ungated output, is what the projection reads.
Every pair of extents `Backend::row_class` puts in one class gives the same bits through the same matmul, routed products and attention as `backend-group` checks (`tests/row_classes.hpp`), at extents on each side of every crossover of the device's profile and every tile split.
It exits 77, which CTest reports as skipped, when there is no loader, the loader has no driver or is older than Vulkan 1.2, or device 0 is missing or lacks a feature the backend requires.

`vulkan-quantization` reads the packed activation buffers before consumer arithmetic.
It checks the 16-bit twin against the original finite inputs, using a representable-scale reconstruction bound independent of the runtime quantizer.
Inputs cover every f32 exponent, reciprocal and normalization boundaries, seeded finite peaks and mixed exponents within a block.
It checks the stored whole sums against double products of the encoded scale and integer sums, output guards, and exact agreement of the word-wise and lane-wise 16-bit writers.
The 8-bit twin of --dtype int8, which the quantizers' second builds (`TWIN8`) write after the 16-bit one, is held to the same reconstruction and sum bounds at 127 levels over the same inputs, with its guards; its lane-wise and word-wise writers must agree word for word, and the 16-bit twin beside it must be the one the first builds write alone.
It also sends 285 peaks, one block of each at width 32 and two at width 64, through raw Q8_0 identity matrices, directly and through SiLU and RMSNorm, comparing to each producer's actual float output.
On a device with both float-preservation properties the first backend's 82080 reconstructed outputs are bounded by half a 16-bit step plus float-rounding allowance, with the independent f32 scale rounded upward to fit the block peak and floored at the smallest positive f32.
On devices without both 32-bit float-preservation properties, consumer cases needing subnormal scales are explicitly skipped; the packed-twin checks still cover every finite exponent.
A second backend selects the fallback while its module-creation call rejects those optional execution modes, then runs the consumer cases that do not need subnormal scales.
It needs a Vulkan device; without one it exits 77. These producer and Q8 consumer checks do not replace model correctness against HF.

On a device reporting both float-preservation properties, the same test compares the preserved Q8 consumers against the ordinary modules on normal inputs, bit for bit: 50676 outputs across widths 32/64/96/128/2080/2112, 41 output rows and 11 column counts from 1 through 64. Decode row runs keep every case on the row kernels, including odd/even block counts and row/column tails. A mismatch fails the test; this is compatibility with the ordinary modules on that driver, not an independent numerical reference. F32 rows also keep the ordinary module: 1599 outputs over widths 33/128/2048 and columns 1/3/9 must match with preservation available or disabled, covering scalar and vector loads.

The same test checks quantized decode and prompt range arithmetic through all six matrix calls: zero rows and one independently encoded unit weight, minimum-positive/unit/maximum-finite half scales, negative scales, cancelling affine minima, eleven activation exponents, both signs and valid widths 256/288/4096. Three decode columns and 65 prompt columns exercise tails; the prompt calls require an actual tile name from the diagnostic timing API. Exact zeros, bounded unit products, expected infinities, additive/routed behavior and output guards are checked. These focused checks do not replace the calibrated HF and long-context gates.

`vulkan-buffer` checks constructor cleanup on a fake device that supplies every Vulkan call it makes, so it needs no loader and runs wherever the backend builds.
On property and feature values alone it checks that every device need the kernels declare is refused by name when missing, among them subgroup shuffles and 8- and 16-bit storage and integers, and that heads 128 and 256 wide fit the attention kernels on 32- and 64-lane devices alike, while a head of another width fits only up to four subgroup lanes' elements, 128 on a 32-lane device, and no head wider than 256 fits.
It also checks that a device without the integer dot product gets a profile that takes no integer-dot kernel (the Q8_0 decode kernel and the integer-dot tile), even where its device's row asks for the integer dot.
`vulkan-lifetime` opens a device, intercepts transfers and injects allocation failures to check queued storage ownership during KV growth, padded-copy creation/replacement/invalidation and argument-arena overflow. It also varies the integer tile preference independently of the MXFP4 row decoder preference over three free-memory budgets: a profile that cannot create a copied MXFP4 tile must release that copy capacity from its scratch reserve.
It checks retry and unchanged KV accounting after failed growth.
A buffer dropped right after `alloc` or `adopt` must outlive its zero fill or its upload.
Hold setup failing at its timeline or at each of its events destroys what it made, and the request made again holds, with copies around the holds right and nothing left after teardown.
`alloc_weight` storage written in pieces that end inside a row holds the bytes and gets the float tile's padded copy, which storage from `alloc` does not.
Five kernel-construction cases substitute calls to check cleanup, poisoned failure outputs, retry and cache reuse.
Two query cases use a real diagnostic add dispatch to check creation failure/retry and destruction after device idle; these cases skip if diagnostic timestamps are unavailable.
Transfer ownership cases intercept copies so old failures cannot submit references to freed memory.
Where devices 0 and 1 form a tensor group's collective, a join whose semaphore creation fails at each of its four semaphores destroys what it made, seven sums back to back with nothing read between give the chain of sums in member order once the collective is destroyed, no semaphore going before both members are idle, and a sum whose sync file fails at each export and each import leaves both members drained, its semaphores made again and the next sums right; without a second device or the exchange these cases are skipped and say so.
Broad device arithmetic remains covered by `backend-vulkan` and HF.

`http` starts the server's HTTP layer (`src/server/http.hpp`) on a system-chosen port from a thread and drives it with the layer's own client: a whole response, a body echoed back, a chunked stream whose chunks arrive as written, a whole response refused inside a stream, an oversized body refused with 413, a malformed request line refused with 400, an unknown route 404, a client seen by `peer_closed` as open while it waits for its answer, also after one urgent (out-of-band) byte, and as closed once it leaves, a write to it then throwing `ClientGone`, and the listener closed from the main thread ending the accept loop, while its thread serves a client and while it waits in accept, after which an accept returns no connection at once.
The hosted TSan job runs it and `server-passes-cpu` under ThreadSanitizer, which sees a socket one thread closes while another reads it.
It runs on Linux, Windows and macOS.

`server-utf8` checks that the server's `utf8_sanitize` (`src/server/api.hpp`) turns a surrogate (ED A0 80), an overlong form (E0 80 80) and a value above U+10FFFF (F4 90 80 80) into U+FFFD, one per byte, as it does truncated and stray bytes, while valid text of every length stays unchanged, and that `utf8_complete` holds back a character whose bytes have not all arrived and lets a whole one or a stray continuation byte through; `error_json` repairs a stray byte in an error message in both reply shapes; and `compat_number` writes a log-probability below -9999, minus infinity and a NaN as -9999, where the native shapes' `jmini::number` writes null for the last two.

`placement` splits a two-layer model over two CPU backends with a device per
tensor role (`docs/EXECUTION.md`) and requires the bytes of the same model on
one backend for a prompt, decode steps, a history across a block edge, a
reset and a two-sequence pass. It counts copies, writes and submissions so
the residual stream crosses exactly where the placement changes and never on
one device, and each stage and crossing submits the devices it records on.
With the embedding alone on one backend and every layer on the other it
gives the bytes of one backend, and only the backend the residual leaves
keeps a handoff buffer. Each backend of a split is asked to hold between submissions once per model made over it and given the request back when the model goes, a single device not, nor either of two listed devices when every role sits on one, nor the devices of a model whose construction is refused. It refuses malformed placements, among them a
device whose attention layers are not one run, and sequences of another
model.
A server's fitted budget (`PlacementRequest::fit_kv`) on a device reporting its free memory: the request's budget where it fits, backed whole as the model is made, the bytes a history grown to the context backs; a budget past the device cut to whole blocks, which fit, one more block not fitting; a device whose free memory rises between reads, as one reclaiming an ended process's memory does, read again until it settles and given the budget of the memory it settled at, also where it stays level for four seconds before rising, and a split over two such devices placed once theirs has; a device that cannot hold the model and one that holds it without one block of KV each refused by its own text; and beside experts on the CPU, a device that copies its weights is not charged for the routed feed-forward blocks.
It checks the fit to device budgets (`model/layer_split.hpp`): even shares where room allows, a device without room left out, a host device given only what the others cannot hold, layers placed by their own sizes and their own caches, a cache for some layers and not others refused, the busiest device given as few layers as fit, tied weights counted once, the host's tables, handoff buffers and staging counted where they sit, shares honored or refused, and the fitted placement exact against one device.
`place_model`, the one placement entry, asks the backends for their budgets, applies the request's ubatch, and places a layer split at the given shares exactly as one device computes.
A three-layer model placed by `place_model` over two and three CPU backends at ubatch 3 takes a 13-token prompt in five chunks, more than the stages, so the pipelined prefill reuses its pass slots and both handoff buffers; the prompt, three decode steps, a second prompt continuing the history, every row of `score()` and a two-sequence pass must be exact against one backend, with the same `n_tokens` and `kv_used_bytes`, and each stage but the last must keep two handoff buffers and the last none.
A backend on the last stage then fails while the first stage is chunks ahead, on top of a history, once at an attention mid-prompt and once at the head on the last chunk (`FailingCpu` in `tests/tiny_qwen.hpp`, which `kv-cache` also uses): every storage's length and `kv_used_bytes` must be back at the history, and the same prompt again must be exact.
`place_model` refuses experts on the CPU beside several devices, on a routed model, and on a model without routed layers, on the CPU as beside a device that reads host memory in place and is not the CPU, each time by the name of the flag given, and a stream point without experts on the CPU; on a routed model, experts on the CPU beside a CPU leave that CPU alone, nothing crossing, with the logits of the model placed without them, and beside that device they run on a CPU placed beside it, the residual crossing each way, with the same logits.
A tensor split (docs/TENSOR-SPLIT.md, step 2) of a three-layer model, tied and untied: one stage of width 2 on two CPU backends against two stages of width 2 on four, at ubatch 3, must give the same bits for a 13-token prompt, three decode steps, a second prompt continuing the history, every row of `score()` and a two-sequence pass, the prompt in one pass the bits of the prompt in slices, and each prompt and step one backend's logits within 1e-4 times one plus their magnitude, the sums regrouped.
Passes in flight go through the pass API (`reserve_passes`, `begin_pass`, `run_pass_stage`, `pass_logits`, `end_pass`, `abort_pass`) over two, three and four CPU stages of a four-layer model, tied and untied, at S, S + 1 and 2S pass slots for S stages: five requests, one a fork of another's first block, their prompts sliced at random and their generated tokens given, are formed into passes of random groups, advanced a stage and retired in random order, so passes wait between stages at once and take logits rows anywhere in the reserved range, and every row must be the bytes of its request run alone through `prefill` and `step` on one backend.
The reservation keeps a handoff buffer per slot, two at least, on each stage but the last, and none on the last.
Once in each run a backend fails a pass of several sequences at its next stage (`FailingCpu`): that pass leaves flight with its histories where it found them, the others go on, and its requests formed again still give their rows alone.
The pass API's refusals are checked one by one, each leaving nothing changed: a context not reserved, reserved twice, used before or on one device for two slots; `forward` through a reserved context; a slot beyond the reservation or in use; more rows or logits rows than reserved; a sequence listed twice or in flight, and `reset`, `fork` and `forward` of one; stages out of order or twice, and logits or an end before the last stage.
A reservation whose allocations the backends refuse (`TightCpu`) fails with their error and leaves the context fresh, and a smaller reservation of the same context then succeeds and runs the passes that follow.
A request's pass slots reach the fit, which holds a handoff buffer per slot, two at least, on each device but the last, which sends nothing: four slots over two devices hold four buffers, and over three devices eight.
A request's histories grow the cache budget only where the context's blocks cannot hold them, each counted up to the context, and three histories that need three blocks run in one pass on one backend and over a split.
Over two CPU stages, the synthetic Q8_0 model's 40-token prompt and its 199 greedy tokens recomputed by class, the prompt at its extent in slices of 16 and the tokens as entries of extent 1 of up to 64 rows, must give the logits one backend gives after the prompt and 199 single decode steps, bit for bit, whole and from a fork at the first block.

Run the Python suite (synthetic fixtures are generated locally; real-model HF
checks skip when their models are absent):
```
python tests/run_tests.py
```

For a CMake build, pass `--exe <path-to-built-llmx>`.
`--only` runs just the components it names, comma separated (`--only baseline`, `--only split,server`), and refuses a name the suite does not have.
CI uses `--no-perf-floor` for shared runners, `--require-tools` on every CMake build it runs the suite on so a tool missing beside the executable, or numpy for raw-blocks, fails rather than skips (the `build.bat` binary has no tools beside it), and `--require-baseline` in its real-model job so missing fixtures fail.
That job also runs `--only baseline` with `--cache-type f32`, `llmx-split-check` on the Q8_0 over two CPU backends and `tools/server_mix_check.py` on the Q8_0; the Vulkan job runs the suite with `--device cpu` on the Vulkan-enabled binary.
The TSan job builds `http` and `server-passes-cpu` with `-fsanitize=thread` and runs them with address randomization off (`setarch -R`), which ThreadSanitizer needs on the runner's kernel.
Local performance floors remain enabled by default. `--require-device-types Q8_0,Q4_K` makes a selected backend's refusal of any named weight type fail rather than skip through `common.device_lacks_kernel`; names are the case-sensitive storage names in `tests/spec_decode.py`.
Types not named keep their ordinary skips, and naming a type does not prove it was exercised, require a missing model, or change native CTest's driverless skips.
The `reference-consumer` component checks early and backend refusals, unrelated failures, required and optional types, runner exit status and invalid selections.
See `docs/CI.md` for workflow coverage and reproduction commands.

- **Dead code** (`tests/dead_code.py`, component `dead-code`): what no product path reaches, from the source, standard library only; the product is `llmx` and the tools, so code only tests reach is a finding too.
  Its findings and the list they are held to are described in Dead code and stale docs, below.
  `unused` is a C++ declaration in `src/`, `tests/` or `tools/` whose name no other token names, and `test-only` one in `src/` that only `tests/` names; the fields of a struct a `static_assert` pins and the values of a named enum cast from an integer are left out, a cast to a template parameter of the same name not counting, and a dead overload, or a name shared with a live one, is not seen.
  `override` is an override in `src/` that product code never calls, through an object or from a member of its class or a base.
  `shader` holds the Vulkan kernels' chain: every `.comp` compiled by a CMake entry and every entry's `.comp` there, every `.glsl` included, every module embedded and in the kernel table, every kernel id named at a dispatch or derived by `kv_variant`, every variant's define tested by its source, every tested define set, and every `#ifndef` default overridden by some entry.
  `flag` is a flag `src/cli/main.cpp` parses that stores nothing, or stores into a field or local nothing uses: the help printing its default, a refusal's condition and a copy into a field of the same name are no use, and a field of a struct `main.cpp` declares is looked for in `main.cpp` until a copy passes it on.
  `python` is a function, class, method or module constant in `tests/` or `tools/` that nothing reaches from a module's top level, an argparse option that neither its module nor a module its parsed namespace goes to reads, or a test module that no suite import, workflow step or command line in a live section of a doc runs.
  `file` is a `src/` file no translation unit CMake builds includes, a test or tool source neither CMake nor the linked check builds, or a header nothing includes; `macro` is a macro defined and never used, unless a system header after it reads it, or an `LLMX_` macro tested but set nowhere, in the code, the build files or a doc's `-D`.
  Planted faults show each check catching what it is for: a dead function, one after a function with a braced default argument, a function only a test calls, an unused value of an enum a template parameter shares a name with, a macro never used, an override nothing calls, one declared over several lines and one with a braced default argument, an unbuilt shader, a flag whose value only the help prints, a flag parsed and dropped, an unreached Python function, an option only another module's namespace reads, an unbuilt test source a record names, and a listed finding that no longer occurs.
  `python -X utf8 tests/dead_code.py --linked DIR`, which the Vulkan job runs, builds every target with the Vulkan backend into `DIR` at -O0 with every inline and static function emitted, links each executable with `--gc-sections` and reports a `src/` function no executable keeps (`linked-unreached`) or only tests keep (`linked-test-only`), which tells the overloads and shared names apart that the name check cannot.
  It needs Linux, GCC, GNU ld, binutils, CMake, the Vulkan headers and glslc, and leaves out lambdas and classes local to a function, virtual functions, templates never instantiated, special members nobody wrote or that are defaulted, constexpr functions, code a Linux build does not compile, a function whose name only such code calls, and a function the name check already reports.
- **Docs** (`tests/docs_check.py`, component `docs`): the Markdown, and the code's comments, against the tree and the binary's help pages, standard library only.
  In every section a relative link and its anchor resolve (`link`), a line number names its commit (`line-pin`), and a heading or a ROADMAP item a reference names is there (`section`): `(AGENTS.md, Tests)`, with the file backticked or not, or `(Tests, above)` for a heading or bold label of the same doc.
  In live sections a repository path, glob or `path:symbol` exists (`path`); a qualified name, a call with arguments and each name among the arguments are the code's, a name only a string holds not counting (`name`); and a command line of `llmx`, of a built tool or test, or of a script in `tests/` or `tools/` uses flags that program takes, a script without a parser taking those of the module beside it that it hands its arguments to, and a `-DLLMX_` option exists (`command`).
  The comments and docstrings of the code, the build files and the workflow are held to the same: a `docs/` path they name exists (`path`) and a heading they name is there (`section`).
  `usage`: each flag a command's help page lists appears in its section of `docs/USAGE.md`, a synopsis without `[flags...]` lists them all, synopses and table rows name only flags the command takes, a numeric default in a table equals the help's, every command has a section and every section a command, and the load tool's table lists exactly its options, with its defaults.
  `test-name`: a name a live doc lists after CTest is a CTest and one beside component a suite component, a line of this list that opens with a test's name and what it does names one, another test-like name this list or `docs/CI.md` gives is named somewhere in the tree, and every CTest and suite component is described in this list.
  `src-page`: every `src/` file is named by exactly one `docs/src` page title, and every title names a path that exists.
  Planted faults show each check catching what it is for: a dangling path, a broken link, an unpinned line number, a command line with a flag its program does not take, a script given a flag of a module it imports without handing it its arguments, a heading reference without backticks in a record, a missing heading of the same doc, a comment naming a doc that does not exist, a qualified name whose member only a string holds, a wrong flag and a wrong default in `docs/USAGE.md`, a page for a file that does not exist, two CTests removed while this list describes them, and a listed finding that no longer occurs; a correct pinned line reference must give no finding.
- **Arch boundary** (`tests/arch_boundary.py`, component `arch-boundary`): the runtime names no architecture. It reads the registered names from `src/model/arch/registry.hpp` and fails if one of them stands as a word, in its own case anywhere, comments and string literals included, and in any case outside a string literal, or a string literal begins with `blk.` or ends in `.weight` or `.bias`, in `src/model/*.hpp`, `src/inference/*.hpp`, `src/server/*.hpp`, `src/cli/*.cpp` or a header under `src/model/arch/` other than the registry and the modules it includes, which the modules share. The layers below the model are not read, the tokenizer among them, whose pretokenizer names match architecture names.
  A self-test first holds it to a planted text and to one name for each row of the registry's table, so a check that finds nothing because it reads nothing fails; it runs no binary.
- **Version** (`tests/version.py`): `--version` matches the CMake project
  version and build identifier format, and the usage banner starts with it.
- **Dtype calibration reference** (`tools/calibrate_dtype.py`): the frozen budget of `docs/PRECISION.md`, reproduced in pinned HF with one CPU thread, the 16-bit classes on the tiny fixtures and the int8 class on `tests/int8.py`'s 512-token Q8_0 fixtures, its inputs rounded per block of 32 by the twin's rule at 127 levels. The `reference-generator` component checks rounding hooks exclude routers and norms, cleanup after a reference failure, version refusal and preservation of an existing output. CI uses doubles for these safeguards; actual calibration is a separate reference run, not generated during CI.
- **Device reference** (`tests/device_reference.py`): the quantization plan's shared CPU/device criterion in `tests/common.py`, with deliberate ranking, NLL, calibration, shape, nonfinite and greedy faults.
  CMake's `llmx-model-logits` captures every full-vocabulary row of an excerpt in batched and per-token execution, then 64 argmax steps after prefill, without stopping at EOS.
  Captures also retain actual per-device matrix paths separately for batched, decode and greedy phases; tiny-HF bounds consume these witnesses, refusing missing or incompatible evidence and keeping F32-only work at its original bounds.
  Tiny F32 captures are checked against independent HF rows; the tests also check Unicode paths, malformed IDs, explicit layer-share capture equivalence and share refusals, incomplete files and the caller's complete flow, with a damaged low-ranked logit that must fail the calibrated maximum.
  The control measures the largest logit gap and reports its other disagreements; the candidate retains every acceptance check. Both roles reject malformed or nonfinite captures.
  `tools/check_device.py` runs the real-model comparison with an existing-type control selected first and retains raw results; see `docs/CI.md`, Device versus CPU numerical checks.
  A missing capture tool fails with `--require-tools` and otherwise reports SKIP after the criterion tests.
  A ninth argument, `-` for the automatic placement, and a tenth, a tensor width, form tensor groups of the listed devices, each a backend of its own, so `cpu,cpu` is a group of two CPU backends, which the command line's listed-once rule does not form.
- **Tensor split** (`tests/tensor_split.py`, component `tensor-split`): the fixtures of docs/ASSETS.md's tensor-split fixtures, whose every split falls whole at widths 2 and 4, against `tests/data/baseline_tensor_split.json`, their shapes and weight hashes checked first.
  `llmx-model-logits` captures the dense model, tied and untied, on one CPU backend and on groups of two and four (docs/TENSOR-SPLIT.md, step 2): every batched and per-token row at a golden text's last position must be HF's within the bound of the path's precision and the longest text's NLL within the NLL bound, and each group's rows and 64 greedy steps must pass the device-reference criterion of `tests/common.py` against one backend; the qwen35 fixture runs on one backend until a group splits a layer that keeps a state.
  It skips without the tool, unless `--require-tools` is given, and under `--cache-type f16`.
- **CLI** (`tests/cli.py`): the command-line surface the numerical components do not reach.
  A Vulkan device is refused with an error and nothing on stdout, never run on the CPU instead, through a model command and through the synthetic bench.
  A build without the Vulkan backend refuses it, and so does a Vulkan build that cannot open it; where device 0 opens, an index no machine has stands in for the missing device.
  `info` on the synthetic MoE model names its architecture and layer count, and lists every tensor written with its type, shape and size.
  A known Q2_K layout, as an embedding and as an unused tensor, opens through `info`, `tokenize` and `detokenize`, with exact type/size and tokenizer output. CPU generation refuses each by tensor name and type, at its role or as unused by the model. `dequantize` refuses it before changing either existing raw output.
  `--n-cpu-moe` and `--cpu-moe`, alone and beside `--moe-stream-from`, are refused on the synthetic dense model with status 1 and the name of the flag given by `generate`, `chat`, `logits`, `perplexity`, `bench --model` and `serve`, on the CPU and on the configured device when that is one device other than the CPU.
  It also checks the CLI's usage errors, all refused before a model is opened: an unknown command, `serve` and `pull` without arguments, missing and extra arguments, unknown flags and flags without a value, a chat positional argument, a second prompt, `--file` or `-f` anywhere but right after the model, `--depth` with more than one sequence, a pull target without its quant, a bench `--size` that is not a multiple of 32, a value after the `--ignore-eos` switch (checked by its reason, which a parser without the switch would not give), the synthetic bench's flags with `--model` and the model run's without it, `--profile` off a single Vulkan device, `--moe-stream-from` without experts on the CPU, an unknown cache type or load mode, and numbers out of their form or range.
  So are, each checked by its reason so that a usage error for another cause does not pass for it: anything after `--version`, `-tb` with `perplexity --per-token`, `perplexity -c 1`, an empty value that would read as the flag not given (`pull --file`, `pull --cache-dir`, `--stop`, `--then-ids`, `--layer-shares`, `bench --model`), a second value for a flag in one spelling or in two, `--cpu-moe` with `--n-cpu-moe`, and `--tensor-width` over a list that is not whole groups, above 4, with a share count other than the groups or over a Vulkan device and the CPU in one group.
  Each but the unknown command, which is checked for its status, an empty stdout and its name on stderr, exits with status 2, nothing on stdout and the command's page, or the overview for `--help` or `--version` followed by anything, then the reason on stderr.
  The help check reads the overview and every command's page, each shown by `--help` and `-h` alike with status 0 and without a model.
  Every flag a page lists, in each spelling, is taken by its command: the line is read in full and fails only on a missing input file, or pull on its empty quant, or with `--size` and `--iters` runs the synthetic bench.
  The same line giving a flag with a value again is refused as a usage error, in the same spelling and in each other spelling its page lists, while the same line giving a switch again is taken, since the second changes nothing.
  The devices are made before the model is opened, so `--profile`'s line, which names `vulkan:0`, fails on that device instead where the build or the machine has none.
  Every flag another page lists, and one no page lists, is refused by a command whose page does not list it, as a usage error.
  It needs no device, so it runs in every job.
- **Round-trip** (`tests/roundtrip.py`): build a random F32 model and quantize it to Q8_0 and Q4_0 through the CLI, each dequantized and checked against its own bound.
  Regression gate for `quant/` + `format/`.
  Also checks quantize's JSON tensor schema/dimension and binary-length rejection,
  output preservation on validation failure, and valid one-to-four-dimensional
  conversion for both writable types.
  `dequantize` must match, bit for bit, the spec decoders of `tests/spec_decode.py`: Q8_0 and Q4_0 on the blocks quantize writes, and Q4_1, Q4_K, Q5_K, Q6_K, Q8_0 and Q4_0 on raw blocks that reach every scale, min, high bit and nibble, Q8_0's and Q4_0's under negative scales, which their quantizers never write, so the references of `q8-dots` and `backend-group`, which take them from llmx's decoders, rest on an independent decode for all six types. MXFP4 additionally checks every exponent and code in every position through the actual dequantize CLI, against the independent spec decoder.
  Q4_0's raw blocks take each of the test's scales twice, negatives included, so every position takes every nibble, and under a negative scale nibble 8 must decode as -0, as the format's d*(nibble - 8) gives it (`docs/ASSETS.md`).
  Both types quantize and dequantize under a non-ASCII directory to the same bytes as under an ASCII one, and a type name quantize does not write is refused with exit status 2.
- **Raw blocks** (`tests/raw_blocks.py`): the spec decoders of `tests/spec_decode.py`, one for each GGUF type the pinned fixtures and the MXFP4 writer's file hold (F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS and MXFP4), each in a pure form, the readable reference, and a numpy form for whole files.
  They are written from the GGUF type layouts and, for MXFP4, the OCP Microscaling Formats (MX) v1.0 specification; the module also reads and writes GGUF files.
  Blocks built from chosen fields must decode to the values those fields define, computed with exact fractions and rounded once to f32.
  A value that is zero must carry the sign the format's own arithmetic gives it, the same expression evaluated in Python floats in the format's order: -0 where a negative scale meets a zero code or a scale of +0 meets a negative one.
  F16 and BF16 run over all 65536 bit patterns.
  Q4_0, Q8_0, IQ4_NL and IQ4_XS take every code, in both nibbles where there are two, under scales positive, negative, zero, subnormal and at the largest finite value, and each of IQ4_XS's 48 scale bits is cleared once.
  MXFP4 takes every exponent from 0 to 255, each block holding every code and each position taking every code over each 16 blocks: exponents 0 and 1 give subnormal scales, 255 decodes as 2^127 times the doubled E2M1 value like any other exponent, a value of 2^128 or more is an infinity, and -0 decodes as +0.
  Q3_K has each of its 96 scale bits cleared once and high bits keyed to each value's address, which catch a -4 subtracted on the wrong state of the bit; Q2_K takes every scale and min byte, with d and dmin paired over subnormals and negatives.
  A few blocks of these types decode to values written out by hand.
  The numpy form must give the pure form's bits, NaN as NaN, on all of these, on the round trip's Q4_1, Q4_K, Q5_K and Q6_K raw blocks and on random blocks of every type.
  The pure form's F16 and BF16 NaNs are Python floats, which keep no signalling NaN and, for F16, no payload, so only the numpy form keeps a NaN's bits.
  The MXFP4 writer's pure and numpy forms must write the same blocks, the E2M1 midpoints of both signs included, which decode to the values it intended.
  It runs no llmx binary; without numpy the numpy checks and the writer's file are skipped and reported, and under `--require-tools`, as in CI, which installs numpy, they fail instead.
- **MXFP4 writer** (`tools/write_mxfp4.py`): writes a GGUF with every matrix, the tied embedding included, in MXFP4, from one holding them as BF16, F16 or F32, by default the pinned Qwen3-0.6B BF16 file, verified by its SHA-256.
  Each block takes the OCP scale rule, floor(log2(amax)) - 2 clamped at -127, and each value rounds to the nearest E2M1 value, a tie to the even code, saturating at 6 and never written as -0 (`docs/ASSETS.md`).
  The output replaces nothing until the spec decoder has read every tensor back as exactly the values the writer intended.
  It is a fixture tool that needs numpy, not a quantizer llmx offers.
- **Perf** (`tests/perf.py`): the synthetic `bench` on one thread times a 2048-wide matmul, RMSNorm, the fused norm and RoPE, and the synthetic model's prefill and decode, and prints them, so perf-first changes can be checked for regressions.
  Generous floors on the matmul's GFLOPS and on prefill and decode tokens per second make catastrophic slowdowns fail loudly without being flaky; `--no-perf-floor` reports the timings without them.
- **Tokenizer** (`tests/tokenizer.py`): encode/decode round-trips incl. unicode and special tokens, and refusal of a file naming another tokenizer or pretokenizer, with an error naming the key and the implemented values.
  The qwen35 pretokenizer must give HF's ids for the 37 texts of `tests/data/baseline_tokenizer_qwen35.json` (the Qwen3 fixture's 20, Thai, Devanagari, CJK punctuation and every added token), read through a file the test writes from the part of the pinned HF vocabulary those texts reach, so it needs no model.
  The 7 control tokens only the GGUF files and `tokenizer_config.json` add must each give their one id, alone and side by side.
  `python tools/gen_baseline.py tokenizer-qwen35` regenerates the file, keeping every merge that forms a substring of a text, so the ids equal the whole vocabulary's however llmx cuts the text.
- **Perplexity** (`tests/perplexity.py`): a synthetic model with an analytic
  scoring oracle checks window boundaries, chunk limits, target counts, file
  and inline input parity and invalid flags.
  Under `--verbose` each window's line must give its scored targets and the oracle's mean NLL for it, in order, with the totals unchanged.
  On the same model `logits` gives the same output for inline text and `--file` or `-f`, appends the ids of the file `--then-ids` names, separated by commas or whitespace, and refuses any other separator and an id past the vocabulary, 2^32 plus a valid id included.
- **Chat** (`tests/chat.py`): follow-up replies against independent HF goldens, including changed prefixes, stop/EOS and token-limit endings.
  Quiet chat prints exactly its one dtype report on stderr; verbose chat adds the loading and per-turn progress there.
  A template the renderer refuses stops `chat` and `serve` before either takes a turn or listens, and `generate` and `logits` with `--chat`, while `generate` without it still runs on the file.
  `generate --chat` and `logits --chat` must give what the same commands give the text the template renders for one user message with the generation prompt and no system message, `--then-ids` and `--last` included.
  CTest also runs `chat-template` on `tests/data/baseline_chat_template.json`: the pinned real Qwen templates (Qwen2.5, the Qwen3 variants, and every Qwen 3.5, 3.6 and 3.8 template found in GGUF files and the official repositories), each held to its SHA-256, over 34 conversations each, tools, tool calls and content given as parts among them, with whether a conversation keeps an assistant turn split under each and a two-turn conversation of seven replies kept that way, and small feature templates, every case byte for byte against transformers' own chat template renderer, a failure where it fails, with its message where the reference raises the template's own error, an undefined value or a type or division error; templates the renderer must refuse; templates past its nesting and value limits, a `map` filter naming `map` 5000 times among them, which must be refused or fail without ending the process; filters and tests given more positional arguments than they read, which must fail as Jinja fails them, `join`'s attribute, which must fail where Jinja would render other text, and the tests Jinja names by operator, which must select as Jinja's do; the texts `chat::assistant_turn` must split as the Qwen templates split them; replies `chat::ReplySplit` must split into reasoning and content the same way at every cut into one, two or three pieces and byte by byte, inside a `<think>` the template opened or the reply opens after newlines, cut off while reasoning, and without a `<think>`, which stays whole, and the prompts `chat::opens_reasoning` must find open; the two conversations of each qwen35 real-model chat golden (`tests/data/qwen35-0.8b` and `tests/data/qwen35-4b`) under its file's template, against transformers' render; and a conversation ending in an assistant turn under the Qwen3 template of the official repositories, held in the test's source, whose reasoning the old renderer dropped; and `chat::stable_prefix` over a tokenizer of one token a byte, the ids of a conversation rendered without the generation prompt as far as they prefix the prompt's, and 0 where the template raises on that render or the tokenizer refuses the text, and after a reply the ids a next turn begins with: up to the next user turn's text, up to the reply's end while it is written, without the reasoning a template renders only for the last turn, and none where the tokenizer refuses the reply.
  Regenerate both fixtures with `python tools/gen_chat_baseline.py` in the reference environment of `docs/ASSETS.md`; running them needs no external libraries.
  The same tool's `--extract` and `--scan` check every template on a machine by hand, through the same test binary.
  `tests/chat.py` also checks that a device that cannot be made is refused before the model file is read.
- **Thread controls** (`tests/threads.py`): actual auto/explicit phase counts,
  restoration after prefill, follow-up chat and HF-golden replies.
  The automatic count must be the one the test reads itself, apart from llmx, from the hardware threads, the process's affinity and its CPU quotas (the cgroup's on Linux, the job object's CPU rate hard cap on Windows), so a run in a CPU-limited container checks the quota.
  Perplexity also checks batched/per-token counts, both batch-thread aliases
  and automatic/zero selection against an independent HF NLL fixture.
- **Loading and streaming** (CTest `load-progress`, `generation-stream`, `cli-output`):
  read-in byte reporting, file spans, files truncated before loading or whose size changes between reading and mapping, callback failures, the loader's host copy and logits in each load mode, over splits and shards, the planned reads and the stream's reads, copies and failures, early text delivery, split UTF-8 bytes and stop/EOS accounting, with `ignore_eos` a reply running past the masked EOS to its limit or to a stop text;
  `cli-output` also reads `--device` lists as the commands do (canonical spellings, a device once, malformed entries refused), and the cache types and load modes as `exec_flag` reads them (one spelling each, an empty or unknown name refused before any model file is read), as it reads `--layer-shares`, whose empty list it refuses.
  It checks that `pull`'s page prints `--cache-dir`'s default from `hub::cache_in_home`, that `exec_flag` reads `--threads-batch` and `-tb` only where the command asks for them, and that `GivenFlags` refuses a second value for a flag, in one spelling or two, and `--cpu-moe` with `--n-cpu-moe`, takes a switch given again, and keeps the spelling a line gave, and that `long_spelling` gives each short form's long spelling.
  The synthetic bench's timed steps (`time_steps`) allocate nothing over a CPU backend that counts its allocations, so the history's storage, which a model grows on its first steps, is not in the timed prefill.
  It runs the CLI's number readers (`int_arg`, `float_arg` with the ranges of `infer::Sampling`, `--seed`'s decimal 64-bit read, `seconds_arg`'s seconds with an optional unit) and `token_ids` over every malformed form: a missing value, a sign, space, base prefix, fraction or trailing character, infinity and NaN, a value past its range or its type, and an id that would narrow into the vocabulary.
- **KV cache** (`tests/kv_cache.cpp`, CTest `kv-cache`): block pool reuse and
  exhaustion, sequence prepare/commit/abort/reset, on-demand storage growth
  and retained reset across block boundaries, paged attention over two
  block tables against a double-precision reference, two sequences batched
  in one attention call against the same two taken separately, the ticket
  and release contract (one submission per pass, waits and syncs counted on
  every release path), the model transaction on failure, a two-entry
  `forward` against the entries run alone, and forks: a whole-block length
  sharing its blocks without allocating one, a length inside a block or
  past the history refused, refused appends into shared blocks, refcounted
  release, and a sequence forked at a block boundary continuing exactly as
  a fresh one fed the same history. This oracle supplements the
  independent HF gate.
  The shared KV storage (`BlockKVStorage`) is held to the doubling rule it replaced, written out in the test: every growth step and the peak, with out-of-order ids and mixed cache types, an overflowing budget refused at allocation, attention over a block no write backed refused, and the growth hooks, each old buffer retired once, the backend drained and the accounting unchanged when a growth fails, and a retry.
  What a copy to host memory holds and how (`Model::host_identity`): one identity for two models of one file on two CPU backends, another for another V cache type, another activation dtype, a backend of another identity, a backend of other row classes and a split over two backends, and a copy's runs adding up to its bytes.
  A history's two blocks copied to host memory (`Model::save_host`), reset and restored into other physical blocks (`Model::restore_host`), continue as the history never copied does, bit for bit; a copy inside a block, past the history or past its limit and a restore into another model are refused, the refused copy keeping no slab, and a second copy takes the slabs the first left; and which idle slabs a copy frees on a split before it allocates (`infer::detail::slabs_to_free`), so the slabs alive stay within the limit, a copy whose shortfall the spare slabs cannot cover refused.
  A history recomputed by class, as a paused request's resume recomputes it, must give the logits its decode gave, bit for bit: on the synthetic Q8_0 model, a 40-token prompt at its extent in slices of 16 and its 199 greedy tokens as entries of extent 1 of up to 64 rows, logits only on the last, against the prompt and 199 single decode steps; the same from a fork at the first block, and beside another sequence's decode row and a third's prompt slice.
- **Server** (`tests/server.py`): `llmx serve` on a system-chosen port against the CLI on the same file, the synthetic F32 model without a download and the Q8_0 fixture when present: greedy through `/v1/generate` equals `generate --temp 0` alone and four at a time, a stream carries the same ids, a seeded request repeats, refusals, a client leaving mid-stream leaves nothing active, a chat turn and a follow-up whose assistant turn comes with its reasoning inline, in `reasoning_content` or null, each rendering as `chat` renders its own reply and counted alike by `/v1/tokenize`, kept whole by both under a template without reasoning, and two `chat` turns under a Qwen 3.8 template of the chat fixture as long as the reference's renders with the reply split, under a template that opens the reply inside `<think>` a `/v1/chat/completions` reply whose reasoning comes as `reasoning_content` and its answer as `content`, whole and streamed, while `/v1/chat` keeps the text whole, and `chat_template_kwargs` with `enable_thinking` false closing the `<think>` in the prompt, counted alike by `/v1/tokenize`, with a value that is not an object, a name the render sets and a nested value refused with 400, the compatible `/v1/completions` and `/v1/chat/completions` whole and streamed in the OpenAI clients' shape, a prompt repeating a finished request's tokens reuses its blocks with the CLI's greedy text, and the limits: a KV budget below the context bounds a request and a full queue answers 503.
  The health dtype record names the actual requested flag, including one supplied by the test runner. Explicit CPU policies are held to their native, emulated or fallback records and give the CLI's replies; auto must take the available native policy.
  A sampling field outside the range the CLI's flag takes is refused with 400 on `/v1/generate`, and a penalty below 1 on `/v1/completions` under `penalty` and `repetition_penalty` alike, and a `top_k` of -1 is refused on the native route and sampled as `top_k` 0 on the compatible one.
  `/v1/tokenize` gives the ids `llmx tokenize` prints, with `add_special` absent, false, true and 1, which it does not read, for text beyond ASCII, a special token's text and an empty text (with the Q8_0 fixture, the tokenizer golden's texts), and `/v1/detokenize` gives those ids back as the text; ids that end or start inside a character, alone, together and reversed, give the bytes `llmx detokenize` prints with the U+FFFD repair.
  A generating route reads as many tokens as `/v1/tokenize` counts for its prompt, `tools/server_load.py` finds the route and counts each text as the CLI does, and a whole reply's ids detokenize to its text; with the Q8_0 fixture, `messages` give the CLI's ids for the chat fixture's renders of the cases carried over from the Jinja2 goldens under that file's template, and a chat request with them reads that many tokens and replies as that text does through `/v1/generate`; while `--max-seqs 1 --max-queue 1` holds one request and queues another, both routes answer, and a text of more tokens than `--ctx-size 512` is counted.
  Both routes refuse a body that is not a JSON object, a missing or mistyped field, both `text` and `messages`, an id that is not a whole number or lies outside the vocabulary, whose edge the CLI confirms (2^32 past a valid id included), and a body past the size limit, in the native error shape.
  On a synthetic model whose vocabulary lacks the byte token `q`, `/v1/tokenize` refuses a text holding it, as a text and as messages, with the 400 and the message `/v1/generate` and `/v1/chat` give.
  With the Q8_0 fixture, seeded requests on each of the sampler's four paths, four at once with the other settings at their defaults, give the text `generate` gives alone with the same settings and seed, repaired as the server writes a character a reply ends partway through: the defaults, `top_k` 40 with `top_p` 1, `top_k` 0 with `top_p` 0.95, and `top_k` 0 with `top_p` 1.
  The synthetic model's file name holds a byte that is not UTF-8 on Linux, and elsewhere characters beyond ASCII whose UTF-8 bytes code pages 932, 936, 949, 950 and 1257 cannot map, so a name read in the system code page there fails; `/v1/health` and `/v1/models` must name it as UTF-8, with a U+FFFD for each byte that belongs to no UTF-8 character.
  Log-probabilities on `/v1/generate` and `/v1/completions`, and with the Q8_0 fixture on `/v1/chat` and `/v1/chat/completions`, whole and streamed: a request asking for none by name gets the bytes of one that never names them, and one asking gets that reply's ids and text with its values in the route's shape, the same bytes on a second run, the stream carrying the whole reply's values, greedy's token first among the five listed and every list most likely first; the completions and chat routes give `/v1/generate`'s values for the same prompt, each of four prompts gets its values alone while the others run beside it, 20 tokens are listed when asked, and 21, a `top_logprobs` without `logprobs` and a `logprobs` of the wrong type are refused with 400; a seeded draw at temperature 1.5 with penalty 1.3 gives the same values on `/v1/generate` with one token listed and with none, which then carries no `top_logprobs`, and on `/v1/completions` with `logprobs` 1 and 0, whose maps hold the sampled token's text at every position, listed or not.
  With the Q8_0 fixture, uncapped requests share a pool too small for all of them: a request is paused when it runs out and resumes from its history, and the check confirms the server counts a pause and each request runs to its own end with the text and every log-probability it gives alone, `/v1/health` then showing no active, queued or paused request, a new pause and an increase since the concurrent group started in device takebacks, host promotions or recomputed rows; a long prompt read one token a pass is paused while it is still prefilling and, resumed, gives the CLI's greedy text and the values it gives alone on a server of its own; a conversation of six turns on a 1024-token pool, its history growing past half the pool, forks the last turn's prompt on every follow-up, never its reply, which decode computed (its `reused_tokens` within the last turn's prompt, growing with the server's `prefix_tokens` once the last turn's prompt passes 512 tokens, where a device's tile takes one split) with each turn's greedy text equal to the CLI's; and a follow-up that fits the pool only once one donor goes, beside an unrelated donor, consumes the turn it repeats, so a later prompt repeating the unrelated request's history still reuses it with the CLI's greedy text.
  With the Q8_0 fixture, two conversations of about 500 tokens alternating on a 1024-token pool with `--host-cache-bytes`, each turn evicting the other's donor, promote their own donors from host memory and reuse their prefixes with the CLI's greedy text, and reuse nothing with `--host-cache-bytes 0`.
  With the Q8_0 fixture too, a chat turn of about 500 tokens and a 160-token reply without reasoning on `/v1/chat/completions`, once `/v1/health` counts the reply read again, lets a follow-up turn reuse more tokens than the first turn's prompt, into its reply, with the reply the same request gives on a fresh server.
  With the Q8_0 fixture too, a client that leaves a whole reply while it is generated, a streamed prompt while it is read one token a pass, or a request waiting for the one slot is noticed within seconds, though nothing written to it fails, and the server then holds nothing active or queued and starts a request reaching the whole pool at once; a client that shuts only its sending side during a whole reply gets no answer, the connection just closing; and each request left behind would run for thousands of passes, so a server that noticed nothing fails on any device.
  The synthetic MoE model's prompts must each give the same ids alone and four at a time, where a pass routes one request's prompt rows beside another's decode rows.
  With `ignore_eos` a greedy reply that ends early at the model's end token runs exactly to its limit, and the CLI's `--ignore-eos` gives the same ids, greedy and seeded: on the synthetic model written with an end token from its own greedy reply, through `/v1/generate` whole and streamed and `/v1/completions`, uncapped to the 16-token context, which `generate --ignore-eos` asked for the room its prompt leaves also fills with the server's ids, and on the Q8_0 fixture through `generate` and `chat` and all four routes, a stop text reached past the masked token still ending the reply; requests with and without it give their own ids alone and at once, and a value other than `true` or `false` is refused with 400 on every route.
  The disk tier over a server's life, on POSIX, where SIGTERM reaches the server: on the synthetic model, SIGTERM exits with status 0 and leaves no directory, a killed server's directory is swept by the next, under `--disk-cache-keep` SIGTERM leaves the directory kept and the next server under keep adopts it, and a tier past the free space or without a host tier is refused as the server starts; with the Q8_0 fixture, three conversations written at SIGTERM under keep are adopted by the next server and read back with the CLI's greedy text, and a server killed right after SIGTERM leaves only whole entries, which the next adopts without an error.
  On the CPU it runs as it is, so every job checks it; on a single device other than the CPU it runs with its experts on the host and prompts from three tokens streamed, where a pass holds streamed prompt rows beside host decode rows.
  There too, with prompts from 100 tokens streamed, a 120-token prompt that forks the first 64-token block of a finished prompt's history must give the ids and values it gives on a server of its own, since the path follows the whole prompt and not the 56 tokens the fork leaves to read.
  A device list skips it, and the whole component skips under `--cache-type f16`.
  Throughput is measured separately with `tools/server_load.py`.
- **Serving load** (`tools/server_load.py`, suite component `server-load`): a running server's figures under closed-loop levels of concurrent users and open-loop Poisson request rates, through `/v1/generate`, `/v1/completions` or the reference server's `/completion`.
  Prompts are counted through the server's tokenize route where it has one, `/tokenize` or `/v1/tokenize`, and by two probe requests otherwise.
  Prompts of an exact token length come from a seeded word list, their lengths from the seed alone, replies of a fixed length ask to ignore the end of text, and each level reports time to first token, time per output token, inter-token and end-to-end latency percentiles, output and total tokens per second, and failures by reason; `--json` keeps every request's record, and the tool exits 3 when any timed request failed.
  The suite runs its `--self-test`, which needs no model: the stream reading of every API and the figures on made-up arrival times exactly, then every API in both loads against an in-process server, `ignore_eos` sent by a workload read with `--output-len` and not by one read with `--tokens`, temperature 0 and no seed sent greedy and llmx's defaults with seeds from 1 sent with `--sampled`, with a refusal, a dropped stream, both timeouts and a short reply, checked on counts, reasons and time bounds a busy machine cannot break, then the whole tool's table, notes and exit status.
- **Many users** (`tools/server_mix_check.py`): a real model through
  `llmx serve` on any placement, a layer split above all. Requests mixing
  short and long prompts with short and long replies are run alone, then
  all at once, then skewed: long prompts land while others decode and every
  fourth client leaves mid-stream. Each request that finishes must give its
  ids alone, a client that fails to leave as planned fails the run, clients that left must leave nothing active, and the first
  requests must give the same text through `generate --temp 0`, whose
  prompt a split pipelines over its stages.
  The HF job runs it on the Q8_0 fixture on the CPU with `--requests 8 --cli 2`.
  `--ids` writes every phase's ids, so two builds can be held byte-equal.
  `--logprobs` has every request of those phases ask for its log-probabilities and top five, compared with its ids and written beside them, and `--passes N` serves with N passes in flight.
  `--drafter lookup|embedded|PATH` serves with drafts while the CLI phase runs without them, so every phase holds the server's drafts to the replies without them; it prints the drafts the server fed and kept and fails where it fed none.
  `--sampled` draws every request of those phases at the sampler's defaults (temperature 0.8, top-k 40, top-p 0.95) with seeds 1 to the request count in place of greedy, and the CLI phase gives `generate` the same settings and seeds, so the seeded replies are held to their runs alone and to the CLI as the greedy ones are.
  `--uncapped` runs other phases on a pool too small for its requests: 12 uncapped greedy requests through `/v1/completions`, streamed with `logprobs` 5, with `--max-seqs 6 --ctx-size 4096` unless given, each alone and then all at once, where requests are paused and resumed; each must give its tokens and every value alone, and it reports the pauses, the tokens resumes recomputed, the wall time together and its inter-token p50 and p99.
  `--fresh-phases` starts each capped phase and the repeat on a fresh server, refusing reuse, pauses or unfinished requests. It is incompatible with `--uncapped`; `--ctx-size` also sets the capped pool.
  `--cache-type f16|f32` selects both KV sides for every server and CLI comparison; omitting it keeps the runtime default, and `--tensor-width W` groups the devices in tensor groups of W for both.
  The server component runs `tests/server_mix_tool.py` offline, checking shared/fresh lifetimes, cleanup, forwarding, changed replies, reuse and unfinished-work refusals.

- **Long context** (`tools/long_context_check.py`): one 16k-token
  request, a user message asking for a summary of an extract of
  `tests/data/wiki.test.raw`, rendered by the model's chat template as
  `/v1/chat` renders it, greedy, 512 generated tokens by default, sent to
  `llmx serve` on the device under test twice from fresh servers, which
  must give the same tokens; then the CPU reads the rendered prompt and
  those tokens (`llmx logits --chat --last`), and at every generated
  position the device's token must be the CPU's top choice or within
  `--margin` (0.5) logits of it. As raw text an instruct model continued
  the extract and looped, and the check compared near-ties inside the loop.
  Nothing else here reaches a
  prompt that fills thousands of KV blocks and then decodes from that
  history. A hash across two backends is not the check: their activations
  round differently, and after a long prompt greedy decoding meets
  near-ties where either token is right, so identical text is only
  required of one backend against itself. It needs a real model and is run
  by hand, not by `run_tests.py`.
  `--dtype auto|f16|bf16|f32|int8` applies the same request to prompt sizing, both fresh generations and both backends' scoring passes; `reference-consumer` checks its forwarding through the CLI and server routes. This backend comparison retains its own margin and is not the independent HF gate.
  Independent HF depth uses `common.hf_depth_top1` and its separate `HF_DEPTH_TOP1_MARGIN`, as agreed in `docs/PRECISION.md`: exact, accepted-tie and miss counts stay separate, with every position and HF gap retained. `reference-consumer` holds the boundary, top-two membership and malformed-row refusals; short-fixture and NLL bounds stay unchanged.
  `--cli` sends the message through two fresh `llmx generate --chat --file`
  runs instead, whose `--verbose` output gives the prompt's token count and
  the generated ids.
  A position past the margin prints both backends' top candidates there, the
  device reading the same tokens, so a near-tie is told from a wrong kernel.
  The device reads them before the baseline does, so its work ends before
  the baseline's long reading starts.
- **Decode probe** (`tools/decode_probe.cpp`, target `llmx-decode-probe`, built beside `llmx-split-check`): a reply's decode path on a real model, the prompt read as one prefill and each forced id of a fixture fed as a decode step, as a request alone runs through the server.
  Each step prints its greedy token, the runner-up and the forced id with their logits, and the tool exits 1 where a forced id is not its step's greedy token; after the last forced id it prints the step's five best, ranked by `infer::top_logprobs`, or every id of a smaller vocabulary, and the logits of the fixture's two tokens.
  `tests/data/decode_probe_30b_a3b.json` holds a prompt, the 55 ids Qwen3-30B-A3B Q8_0's greedy replies share on an MI50 in the Q8_0 decode kernel's quarter layout and in its half-block order, and the two tokens where they part, with each order's gap between them (docs/STATUS.md, the half-block order).
  It needs that model and is run by hand; the suite's `decode-probe` component (`tests/decode_probe.py`) runs the tool found beside `--exe` on the tiny F32 model and on a model of four tokens, on the configured device where that is one Vulkan device and on the CPU otherwise.
  The greedy path, read from the tool's own lists over runs of none, one and two forced ids, must exit 0 and a step off it exit 1; the last step's best list must hold five distinct ids inside the vocabulary, largest logit first, and four on the model of four tokens.
  A fixture whose id or token is not a whole number, is negative, past 2^32 - 1 or past the vocabulary, or is a string or a boolean, must be refused with exit status 2, nothing on stdout, and the entry named, as `ids[1]`.
  A fixture's `draft`, a count, loads the file's embedded drafter, or with a `drafter_file` the MTP blocks that file holds beside the model, and prints its drafts after the last step's greedy token, each row's id and every logit, and its `cache` stores both cache sides as `f16` or `f32`; the `qwen35` and `drafters` components read them so.
  It skips when the tool is not there, unless `--require-tools` is given.
- **Drafters** (`tests/drafters.py`, with the tool `tools/drafter_pack.cpp`, target `llmx-drafter-pack`, which splits a model's MTP blocks into a drafter file beside it and embeds them back, every tensor's bytes copied unchanged): drafter files beside a model (docs/SPECULATIVE.md, step 6), on tiny fixtures.
  The tiny qwen35 model's MTP block split out by the tool leaves a model whose logits are the file without a block's, and drafts from beside it, `--drafter PATH`, as embedded: `llmx-decode-probe`'s drafts and every draft row's logits, after prompts of 1, 5 and 10 tokens, are the embedded drafter's bit for bit, and so are those of the block embedded back by the tool.
  With it beside the model, `generate`, greedy and seeded at 1 to 16 drafts, and a chat of two turns at a context of 1024, print what they print without drafts and feed drafts, `bench --model` runs its rows and times its rollback, and `serve` gives each reply alone and at once as without drafts, greedy and seeded, with drafts fed.
  A qwen35 and a qwen3 draft model sharing the target's tokenizer give `generate`'s and chat's output without drafts and feed drafts, and `serve` and `bench` refuse a draft model by name.
  Every pairing check (`infer::spec::pair`) refuses its case with the text it names: another token list, another end of text, MTP blocks of another width or counted otherwise, blocks that are not MTP blocks, a tensor past the blocks, blocks beside a model that carries its own, an architecture llmx does not run, a draft model without the target's end of text, and a DFlash drafter of another width, with taps out of order or past the layers, with a plain mask token, or beside a qwen3 model; a DFlash drafter that pairs is refused as nothing runs it yet, and a file cut short is refused.
  It skips when either tool is not beside the executable, unless `--require-tools` is given, and under a cache type other than f32.
- **Layer split** (`tools/split_check.cpp`, target `llmx-split-check`, built in every configuration with tests, which is the default, and linked to the Vulkan backend when that is on): a model on one device against the same model split in equal shares over a comma-separated list of devices of the same kind (default `0,1`; a device is `cpu` or a Vulkan index), with optional decode steps, ubatch, cache type (`f16`, the default, or `f32`, both sides of both models) and dtype (`auto`, the default, or `f16`, `bf16`, `f32`, `int8`), as raw float logits compared with `memcmp`: every position of a scored text through the prompt path, the prefill in chunks of the ubatch, which a split pipelines over its stages, and greedy decode steps, then three passes of a decoding sequence beside a fresh prompt, every row's logits.
  On a split that takes passes in flight, with a text of 12 tokens at least, it then runs them through the pass API at P = S, S + 1 and 2S: 2P sequences, each a prompt cut from the text in chunks of up to 32 tokens and then 8 tokens of the text as generated ones, four entries to a pass, each stage recorded after a random host delay of up to 3 ms, and every logits row must equal the same passes run one after another through `forward` on the split and on the single device.
  After the decode steps, the prompt and the steps are recomputed by class on each model, as a paused request's resume recomputes them: the prompt at its extent in ubatch slices and the steps as entries of extent 1 of up to 64 rows, whole and from a fork at the last whole block when the history passes one, which must give the decode's last logits; it prints how many of these ran from a fork.
  The split must be bit-identical, since each layer runs the same kernels on the same rows wherever it sits; a split over different backends is held to the HF bounds instead.
  Both placements report the requested and resolved dtype, including fallback; after the checks, each device reports its completed matrix paths. The component requires those records and validates the paths against the resolved dtype. Invalid dtype names fail before the model is read.
  Each listed device is a backend of its own, without the CLI's listed-once rule, so `cpu,cpu` is two CPU backends.
  A ninth argument, a tensor width above 1 (docs/TENSOR-SPLIT.md), makes the single device a list of that many devices forming one group and the split's devices groups of that many, its stages, which must give the one group's bits.
  The `split` component (`tests/split.py`) runs the tool found beside `--exe` on the tiny F32 model, tied and untied, the tiny MoE model, the three MXFP4 dense tied/untied and routed fixtures, and the tiny qwen35 models of Hv = Hk and Hv = 3 Hk, one CPU against `cpu,cpu`, the MoE also against `cpu,cpu,cpu` and the qwen35 models against `cpu,cpu,cpu,cpu`, a layer a stage, where two stages hold only a linear-attention layer's state and no KV, at ubatch 1, 3 and 16 and with f16 and f32 caches, with 3 decode steps after a 13-token text, which fills their 16-token context.
  The tool forks no model that keeps a recurrent state, which forks only at a checkpoint, so it recomputes such a model by class from no fork.
  After the decode steps it also runs verifies of drafts on both models: rounds that mark the history, feed 2 to 17 of the decode's tokens in one pass of generated tokens and retract to keep none, some or all of them, every verify row and the step after each retract to the mark compared between the split and the single device; the `split` component requires rounds on the Q8_0 model's 40-step runs.
  It also writes a synthetic Q8_0 model with a 256-token context, whose prompt and decode rows take their respective CPU float reductions, and runs it against `cpu,cpu` after a 100-token text with 40 steps and a 150-token text with 8, histories that pass a 128-token block, so every run recomputes from a fork on one backend and on the split, which the component requires.
  It skips when the tool is not there, unless `--require-tools` is given, and the configured device, shares, cache type and load mode do not reach it.
  The HF job runs it on the Q8_0 fixture over the perplexity excerpt, `cpu` against `cpu,cpu` with 8 steps and 64-token chunks; other real models and splits over devices are run by hand.
- **F32** (`tests/f32.py`): deterministic small-model weights with full logits
  and windowed NLL generated independently by HF. Covers tied/untied weights,
  odd dimensions, batch tails and threads without downloading a model,
  and `bench --model` at a depth and with `--seqs 2`, two sequences where the model's context fills one cache block.
  `logits --file` must print what the same prompt inline does.
  `generate`, greedy and seeded, with `--drafter lookup` at `--draft-max` 1, 4, 8 and 16 must print the text and the ids it prints with drafts off (`common.check_drafts`), as on the MoE and qwen35 fixtures.
  The `--last` rows of the prompt, in one pass and in several, and of its first three tokens continued by `--then-ids`, are held to the HF bound at their positions.
  Each row is printed once, and the rows over several passes and after `--then-ids` are the bytes of the one-pass rows at the same positions.
  The same rows read with `--per-token`, every token through a decode step, are held to the HF bound too, the prompt's and its head's continued by `--then-ids` the same bytes, and without `--last` it prints the last row's list.
- **MoE** (`tests/moe.py`): the same for a tiny `qwen3moe` model against HF `Qwen3MoeForCausalLM` (`tools/gen_baseline.py moe`), two routed layers and one dense, across batch widths and threads and, on a device, with the experts of one or every routed layer on the CPU (`--n-cpu-moe`, `--cpu-moe`).
  On a device it also streams those layers to the device for prompts from a length on (`--moe-stream-from`): every routed layer's from 1, which streams what 2 does since neither a generated token nor a one-token prompt streams, and the first routed layer's from 4, which only the longer prompts reach, while the placements that do not stream give 0, which never streams.
  A second model is Q8_0 with every row width, the experts' included, a multiple of 64, so a device's Q8_0 decode kernel reads its rows in the order it keeps for an even block count; its goldens (`tests/data/baseline_moe_q8.json`, `tools/gen_baseline.py moe-q8`) are HF holding the file's own weights as `tests/spec_decode.py` decodes them, for three prompts and HF's greedy continuation of 16 ids, read batched, three rows a pass and with `logits --per-token` through decode steps.
  Its gated variant's router is kept at least 0.1 router logits from a top-k tie and its near-tie variant's within 1e-5 of one.
  On each path the gated variant's 51 x 257 logits and each prompt's forced mean NLL must stay within the frozen bounds of its actual execution dtype, from the completed matrix-path record, and its greedy id must equal HF's wherever the top two part by more than twice that logit bound. Every prompt must include such a row. The former coarse 8-bit allowance is retained only as history (docs/STATUS.md, the half-block order).
  The near-tie variant's errors are printed and held to nothing.
- **Int8** (`tests/int8.py`): tiny Q8_0 models with a 512-token context and every width a multiple of 64, dense untied, dense tied and a qwen3moe model whose every token takes all eight experts, so no routing near a tie turns over under rounding, with texts of 160, 200 and 256 tokens and windows of 128 and 256.
  Their goldens are HF float32 on each file's own weights (`tests/data/baseline_int8.json`, `tools/gen_baseline.py int8`); the logits of every text and the windowed NLL, scored in batched passes and one token a step, must stay within the frozen budget of the paths each run witnessed, under f16 and under int8.
  Where the device runs int8 every int8 run must witness `block-int8`; where it does not, the record must name the f16 it widened to, with the warning, and no run witness `block-int8`.
- **Qwen 3.5** (`tests/qwen35.py`): the same for tiny `qwen35` models against HF `Qwen3_5ForCausalLM` (`tools/gen_baseline.py qwen35-tiny`), whose goldens come from HF's token-by-token cached forward, which runs the recurrence llmx runs per token.
  The native qwen35moe checks run prompt slices, mixed-sequence passes, layer splits and experts on a separate CPU backend at F32, F16 and BF16, requiring byte identity within each policy. A separate call-boundary check observes every router and shared expert gate during prompt and decode: routers stay F32, and shared gates receive the selected policy.
  The fixtures are Hv = Hk with a tied head, Hv = 3 Hk with its own head, that model with one MTP block, whose file must print the bytes the file without it prints, and a qwen35moe model of Hv = 3 Hk with 4 experts, 2 a token, and a shared expert against HF `Qwen3_5MoeForCausalLM`, whose router the generator holds at least 1e-3 logits from a tie for every token.
  The writer makes the weights as HF holds them and applies the converter's transforms itself (docs/QWEN35.md, GGUF conventions).
  Beyond the F32 checks, the NLL is scored in passes of three tokens and one token at a time, and greedy decode after a prefill must give HF's greedy tokens.
  On the models of a 1024-token context the serve checks write, the Hv = 3 Hk one and the qwen35moe one, 60 tokens of `generate` with `--drafter lookup` and a chat of two turns must give what they give with drafts off, the verifies reaching the state's rerun from the mark.
  Served with a context of 1024, the Hv = 3 Hk model must give the same greedy ids through `/v1/generate` alone, four at once and from `generate`, a follow-up turn repeating a 500-byte prompt and its reply must fork the state the first turn kept, reusing fewer tokens than that prompt, with `generate`'s ids for its whole prompt and checkpoints held in `/v1/health`, on that model and on a qwen35moe one, and four uncapped requests on a pool of 1024 tokens must be paused and resumed with the text each gives alone, with `--state-checkpoints 0` keeping no donor and taking nothing back, and with the default checkpoints too; `bench --seqs 3` must hold a state slot for each of its sequences.
  The MTP block's file as an embedded drafter (docs/SPECULATIVE.md, section 7): its drafts and every draft row's logits at steps 1 and 2 from prompts of 1, 2, 5 and 12 tokens must be the assembled HF reference's (`tests/data/baseline_qwen35_mtp.json`, `tools/gen_baseline.py qwen35-mtp`, docs/ASSETS.md) within the F32 bound, read through `llmx-decode-probe` with f32 caches, which skips without the tool unless `--require-tools` is given; `generate` greedy and seeded with `--drafter embedded` at 1, 4, 8 and 16 drafts, feeding drafts, must print what it prints with drafts off, as must 60 tokens and a chat of two turns on its 1024-context model; served with `--drafter embedded`, and the Hv = 3 Hk model with `--drafter lookup`, each reply greedy and seeded alone and with the others at once must be its reply without drafts, and `/v1/health` must count drafts fed; `bench --model --drafter embedded` must run, its rollback timed at a prompt of 11 tokens and skipped with its reason at 12 and 16, past the 16-token context, and the file without the block be refused by name.
  On a device whose backend lacks the linear attention's ops the files are refused as they load, and the component reports SKIP, which `run_tests.py` counts as neither a pass nor a failure (`common.SKIPPED`), as it does where llmx refuses the architecture (`qwen35.REFUSALS`).
- **Baseline** (`tests/baseline.py`): real-model EXTERNAL ground truth.
  Compares llmx against golden fixtures generated once from the HF reference by `tools/gen_baseline.py` and committed to `tests/data/`.
  Needs a real model, so it SKIPS when none is on disk; point it at one with `LLMX_BASELINE_GGUF`.
  Otherwise each check, the tokenizer's included, reads its model from the HF cache at the revision `tests/data/fixtures.json` pins (repo, revision, file, SHA-256 and size) for each qwen3 entry marked `gate`, read into `BASELINE_MODELS`, the path `tools/fetch_test_models.py` downloads to.
  A qwen3 gate model's bounds sit in `tests/baseline.py` and a qwen35 one's in `tests/baseline_qwen35.py`, and `tests/baseline.py` refuses to load unless each file is pinned once, each gate model has bounds in its family's module, each bounded qwen3 file is a gate model, and each gate model is marked `hosted`, since the hosted HF job downloads and requires every one.
  Every perplexity cell is scored twice, in batched passes (the default) and with `--per-token`, so the prompt and decode kernels both meet the reference.
  Its logit and PPL outputs go through the validators the 8B check uses, `common.check_logits` and `common.check_ppl`, at each fixture model's bounds: the exact prompt token count, ten unique in-vocabulary IDs with finite logits sorted from the top, and exactly the PPL fields with every count exact.
  Regenerating tokenizer fixtures needs `tokenizers` and `huggingface_hub`; numerical fixtures also need `torch` and `transformers`.
  RUNNING the suite needs none of these packages.
  `python -X utf8 tests/baseline.py --file-exact DIR --model FILE` holds one file to the goldens `tools/gen_baseline.py file-exact` made from it, HF run on that file's own weights as the numpy spec decoder reads them, at the Q8_0 fixture's bounds whatever the file's type (`FILE_EXACT_BOUNDS`); goldens made from another file are refused by SHA-256.
  A device with no kernel for the file fails a file-exact run, where the gate's own checks skip.
  Making those goldens also needs numpy.
  The other entries of `tests/data/fixtures.json` pin the Qwen3 models of tensor types llmx does not read yet, each with `gate` false until its type has bounds, and the Qwen3.5-4B Q4_K_M, which is checked by hand; `hosted` marks the ones the hosted HF job is to download then; `tools/fetch_test_models.py --all` fetches them with the gate's models.
  Each entry names its family: `qwen3`, whose goldens are the ones above, or `qwen35`, whose goldens, vocabulary and context `tests/baseline_qwen35.py` holds, with the qwen35 gate files' bounds.
  The 0.8B Q4_K_M first runs its committed file-exact goldens at the Q8_0 file's bounds, and only if they pass is it held to its model's goldens at its own quality bounds.
  For each qwen35 file of the gate on disk it runs that check at 512-token windows: the family's tokenizer golden, the file's chat template, the ids of its model's chat renders, prompts and excerpt, then logits and perplexity, with one skip line where llmx does not run the architecture, on a device whose backend lacks its ops; a file without bounds is measured and fails.
- **Reference generator** (`tests/reference_generator.py`): standard-library checks for pinned reference selection, separate alternate-model output and forwarding the revision/float32/eager settings to the HF loaders.
  Actual reference generation and model correctness remain separate checks.
  Qwen3-MoE file-exact mapping shares the tiny Q8 mapper and checks mixed F32/BF16/Q8 expert stacks, router names and pinned-version refusal. The layered CLI checks offline import/load behavior, cached tokenizer writing and native peak memory reporting with Linux/macOS unit controls. The separate `python tests/layered_moe_reference.py` hand check uses the pinned HF environment for full-versus-streamed tiny equality, global rotary buffers, one-layer residency and failure cleanup (docs/ASSETS.md, Qwen3-MoE file-exact references). Hosted jobs do not provision that environment.
  For `file-exact` it checks the argument combinations it refuses, that `tests/baseline.py --file-exact` fails when the device has no kernel for the file, the HF parameters a few GGUF tensor names take under the one map `tests/f32.py` holds for the tiny models and file-exact alike, and a tiny GGUF's tensors reaching their parameters with reversed dimensions and unchanged values.
  Every qwen35 reference runs through one check of its environment (`qwen35_environment` in `tools/gen_baseline.py`), whose doubles must see a torch, transformers or tokenizers version other than the pinned ones refused, transformers pinned as the chat renderer pins it, and an installed `kernels`, `fla` or `causal_conv1d` package refused, since HF would run it in place of its torch functions.
  For `qwen35-tiny` its doubles hold the generator offline, to float32 and eager attention, and to its key checks: only `mtp.*` and `model.visual.*` keys unused, and none missing.
  The converter's tiled V-head order has one owner, `tests/qwen35.py`, for the tiny writer, file-exact and the layered reference alike, and the test maps HF's `dt_bias` of one Qwen3.5-4B layer onto that layer's `ssm_dt.bias` in a 4B GGUF bit for bit through it (`tests/data/qwen35_4b_dt_bias.json`), so a misreading of the order that the writer and the kernels share cannot pass the Hv = 3 Hk fixture.
  It also checks `tests/data/fixtures.json`: each file pinned once with every field, the gate's models those with bounds, and of the six pinned ahead of their types the hosted ones exactly UD-Q8_K_XL, IQ4_XS and Q2_K, and of the qwen35 files the two 0.8B ones hosted and in the gate and the 4B checked by hand.
  The qwen35 tokenizer golden, on a made-up vocabulary, must keep a merge that joins across a cut HF makes and drop one no text reaches.
  It must give an added token the GGUF files' type, control for a special one or one written `<|name|>` and user-defined otherwise, and keep a token only the config adds apart.
  The generator must refuse a tokenizer file whose SHA-256 is not the pinned one, and the committed golden must hold the generator's texts, commit and digests, so neither changes without regenerating it.
  For `qwen35` it checks the selections it refuses, a tiny qwen35 file written as the converter writes one (norms as 1 + w, `ssm_a` as -exp(A_log), three V heads to each K head in the tiled order, and an MTP block) coming back as HF's tensors, as does a qwen35moe file's routed layer, its gate and up stacks joined into HF's `gate_up_proj` whether decoded whole or a tensor at a time and its shared expert's gate vector as a row, and the loader against doubles: the versions, float32, eager attention, no `kernels`, `fla` or `causal_conv1d` package, the keys a load may leave out, the recurrence refused and the checkpoint files held to their digests.
  The committed qwen35 goldens must hold the generator's checkpoints, files, prompts, conversations, templates and excerpts, the committed file-exact ones the weights of the file they are for, and the consumer must pin every one of them.
  For the layered qwen35 reference (`tools/gen_layered_reference.py`) it checks the arguments it refuses, that it runs offline in that environment check on its threads and refuses what the check refuses, the model a checkpoint's config chooses, dense or routed, and a layer count past the checkpoint's refused, the checkpoint keys it refuses (a stray key, a missing one, a converted one and two for one parameter) and the later layers' keys it leaves to a model cut to its first layers, a file-exact run's arguments and its reading of one routed layer of a GGUF as HF's parameters, a tensor either side lacks refused, its count of float32 steps between two values, that it names an F32 tensor it had nothing to compare with, that the committed Qwen3.5-0.8B record, and the Qwen3.6-35B-A3B record cut to its first four layers, show every input's logits and every parameter equal to HF's full forward, and that the committed 9B, 27B and 35B-A3B goldens hold the generator's texts, windows and versions, the 35B-A3B's also the experts implementation HF's default ran.
- **Reference consumer** (`tests/reference_consumer.py`): standard-library rejection tests for changed 8B fixtures, damaged logits/PPL, top-5 boundary swaps beyond those `common.top5_overlap` forgives, wrong model identity and failed launches, and a passing run over simulated outputs that must have 41 checks with each NLL case scored in both modes.
  For `tests/baseline_qwen35.py` it checks changed goldens, the digest that stands for the excerpt's ids, one skip after 47 checks when llmx refuses the architecture or a device refuses its ops, a run of 59 checks failing without bounds and passing with them, a wrong file digest or chat template, and file-exact goldens made from another file.
  For the Qwen3.5-0.8B Q4_K_M it checks that its quality bounds are that SHA-256's alone and no other file's keep fewer than every top-1, that a run of 114 checks passes with two of its eight top-1 swapped and fails on the top-1 count with three, that a file-exact ranking's swap fails the run before any of the model's rankings runs, and that its NLL bounds hold at their edges.
  The suite's form must skip only a file not on disk or refused, and fail a file without bounds; `--require-baseline` must count a qwen35 gate model as it counts a Qwen3 one.
  For the layered qwen35 goldens (`tests/baseline_layered.py`) it requires each model's goldens to record its GGUF's SHA-256 and every F32 tensor of the GGUF compared and equal to the checkpoint's but for `ssm_a` values one float32 step off, which the model's provenance line counts, a passing run of 41 checks per model without file-exact goldens, one skip line for either refusal, and a failure for a model with no goldens, for another error and for a check failed before the refusal.
  A model with file-exact goldens must have them made from its file and held to the family's file-exact bounds: answers from them pass their 21 checks and then the model's, 63 checks, where the Qwen3.6-35B-A3B Q4_K_M keeps its quality bound of five top-1 of six and fails with four, answers from the model's goldens fail the file-exact checks before any of the model's rankings runs, and file-exact goldens made from another file are refused before they run.
  Only that file's SHA-256 keeps fewer than every top-1: each other file with layered goldens fails on one ranking turned over.
  A component that compared nothing must return `common.SKIPPED`, which the runner prints as SKIP rather than PASS: `split` and `decode-probe` without their tool, `f32`, `moe`, `server` and `shards` under another cache type, the components `common.tensor_width_skip` names under a tensor width, and `baseline` with no fixture model on disk.
  It is included in the ordinary suite; it does not load or download the 8B model or the qwen35 models.
- **Fixture downloader** (`tests/fetch_models.py`): seventeen offline tests
  of `tools/fetch_test_models.py` against simulated responses: a verified
  download and its cached reuse, a corrupt cached file replaced, bounded
  retries with backoff on 429, transient server errors and network
  failures (Retry-After as seconds or a date, HF reset headers, malformed
  headers), no early retry when the server asks for too long a wait, no
  retry on permanent HTTP or local write errors, interrupted and short
  reads restarted from a clean temporary file, and a hash mismatch failing
  with the existing file kept.
  A run downloads the gate's models and `--all` every pinned model, and `--key` prints the HF job's cache key, a hash of the gate's pins that a model pinned ahead of the gate leaves as it is and a new pin of a gate model changes.
  It is not a `run_tests.py` component; CI runs it as a step of its own (`docs/CI.md`).

The optional real 8B check is separate from the ordinary suite and default CI:

```
python -X utf8 tests/baseline_8b.py --exe build/Release/llmx.exe --model path/to/Qwen3-8B-Q8_0.gguf --output-dir hf-8b-review
```

Use a new output directory; on Linux use `--exe build/llmx`.
The consumer verifies the model/fixture hashes and writes raw outputs plus `report.json`, including failures.
It requires exact tokenizer/input IDs, six top-1 matches, top-5 overlap 5/5 and valid top-10 logits; absolute NLL bounds are 0.01 for the continuous excerpt and 0.02 per windowed case.
Each NLL case is scored twice, in batched passes (`ppl-NN`) and with `--per-token` (`ppl-NN-per-token`), as in `tests/baseline.py`, so a run has 41 checks.
These prospective Q8 bounds were frozen before the 8B comparison.
Since 2026-09-25 the overlap, here and in `tests/baseline.py`, counts a swap at the 5th place as agreement when the reference puts both tokens within 0.1 logits of its 5th value (`common.top5_overlap`): such near ties reorder with any summation order.
It also counts the reference's 5th and 6th trading places when llmx's own logits for the two are within 0.1, a tie the quantized weights can create where the reference has none: on an MI50 and CPU split, 0.6B Q8_0 puts " black" and " orange" 0.076 apart for "The three primary colors are red," against the reference's 0.203.
See `docs/ASSETS.md` for provenance and scope: short rankings/excerpts do not establish full-corpus or deep-context correctness, and the exact original GGUF conversion revision is undocumented.

The qwen35 files are checked the same way by hand, one pinned file at a time, against the goldens of the checkpoint their entry names (`docs/ASSETS.md`, The qwen35 real-model references):

```
python -X utf8 tests/baseline_qwen35.py --exe build/llmx --model path/to/Qwen3.5-4B-Q4_K_M.gguf --output-dir qwen35-review --context 4096
```

`--context` picks the 512-token or the 4096-token windows, and `--file-exact DIR` holds the file to the goldens `tools/gen_baseline.py file-exact` made from it instead.
It prints one skip line where llmx does not run the architecture, on a device whose backend lacks its ops, and a file without bounds is measured and fails, so the first measurement sets them.

`tests/baseline_layered.py` runs the same checks on the Qwen3.5-9B, Qwen3.6-27B and Qwen3.6-35B-A3B Q4_K_M files against the layered HF reference's goldens, chosen by the file's SHA-256, and skips in one line where llmx does not run the qwen35 architecture (`docs/ASSETS.md`, The layered qwen35 reference).

The two project gates are external and are defined in `docs/ROADMAP.md` #8:
**correctness is the HF reference**, and **performance must be at least
mx-llama.cpp** on the same model, quant, prompt and hardware. Do not substitute
a self-consistency check for either. llmx passed a fully green suite while eight
correctness bugs were live, because every test compared llmx against llmx.

When you change a hot path, run `tests/perf.py` and note the before/after in the
commit message. The lossless correctness gate (path-controlled perplexity on a
real model) is tracked in `docs/ROADMAP.md`.

Report numerical results in comparison tables: before / candidate / mx-llama.cpp
for performance, and observed error / HF bound for correctness. Include units
and distinguish measured results from targets or unverified claims.

Real models and corpora for manual verification (the Qwen3-8B Q8_0 model, the
wikitext test set) are documented in `docs/ASSETS.md`.

The shared tiny-HF consumer reads exactly one completed `matrix-paths:` JSON record from logits/perplexity stderr. It refuses missing, repeated, malformed or unqualified paths and applies the frozen bounds of the actual execution; an all-F32 measurement keeps 2e-5 logits and 1e-5 NLL regardless of the requested policy. The shared dense, MoE, MXFP4, sharded and Qwen3.5 checks, F32 fixture row checks and gated Q8 MoE continuation use this owner. The server MXFP4 check independently validates a full CLI row at that witnessed bound, then requires each checked server logprob to match its log-softmax within six-decimal CLI output and F32 rounding before comparing it with HF. This is a composition through numerical equivalence, not a server kernel witness. Concurrent and streamed values must still equal the requests run alone; a static health catalog never chooses the bound. The former weight-based MXFP4 allowance is removed.

The thread-count component also reads the completed perplexity matrix-path witness through that tolerance owner: F32 execution keeps its 1e-5 NLL bound even under a narrower request, while a reached F16/BF16 path uses its frozen budget. Thread-count assertions remain exact, and missing or unqualified witnesses fail.

The dtype tests cover explicit overrides, emulation and wider fallback separately from auto; mixed-device placement must execute the per-device choices and refuse an incomplete device policy before allocation. The CLI checks all four dtype values on every non-server model command, plus missing, invalid and repeated values on all model commands. The server checks the exact health record and CLI/server generation parity at each value on the CPU. These focused checks do not replace conformance, independent HF or final merge gates.

The `vulkan-quantization` offset-range regression checks zero rows beside a unit-weight row, independently encoded in Q4_0/Q4_1/Q4_K/Q5_K/Q6_K/Q8_0, through all six matrix calls with output guards. Nonzero cancelling minima, negative scales, odd block counts, guard boundaries and both activation signs are covered. The F16 policy is checked over finite binary16 inputs, including subnormal/normal boundaries and 65504, preserving zero exactly and the unit product within its independently bounded 16-bit reconstruction error. The original F32 extreme inputs are checked under explicit F32 on decode and prompts; decode must dispatch a row kernel and products keep F32 bounds. The backend test checks all six quant types on original F32 inputs against double dots, with no activation-rounding allowance, at one, three and nine decode columns and a tile-width prompt; each decode column must equal the same column alone bit for bit. BF16 is checked through the same six matrix calls and through the double-dot and decode-batch checks against rounded inputs. A separate nearest-neighbour reference checks all finite BF16 encodings with five F32 tails and both signs, plus infinities and NaNs, using unit and non-BF16 F32 weights to prove that weights stay exact. It then runs F32 on the same backend as a control. Matrix sums can turn -0 into +0, so this checks zero values rather than a conversion's zero sign. This is additional range evidence, not the independent HF or full dtype-conformance gate. The BF16 batch-split regression holds F32 and seven quantized weight types (Q8_0, Q4_0, Q4_1, Q4_K, Q5_K, Q6_K and MXFP4) bit for bit across batches 63/64/65 and 127/128/129, logical extents 1/3/65/129, mixed decode and prompt runs, and ordinary, logits, additive and grouped matrix calls. Each call must witness BF16; the same logical row alone supplies its identity reference.

## Architecture

See `docs/ARCHITECTURE.md` for the layer diagram and rules. The rule that
matters: **each layer depends only on the layers below it** -
`cli > server > inference > model > backends > tokenizer > format > quant > core`;
`server/` uses the inference layer and adds only scheduling and transport.

| Directory    | Contents                                        |
|--------------|-------------------------------------------------|
| `core/`      | fp16 and bf16 <-> f32, JSON parser, UTF-8, file hashes, the host memory a process can still take and owned pages, comma-separated lists, the CPUs a process may use and the cgroups its limits are read from |
| `hub/`       | CLI acquisition path: Hub metadata, curl HTTPS and verified multi-stream cache |
| `quant/`     | storage metadata and checked row sizes in types.hpp, decoder registry + Q8_0/Q4_0/Q4_1/Q4_K/Q5_K/Q6_K/MXFP4 kernels |
| `format/`    | GGUF v3 reader/writer (headers, then mapping, then reading in), file spans, a file read at offsets, output files published whole, raw F32 tensors to and from GGUF |
| `tokenizer/` | byte-level BPE, Qwen2/Qwen3/Qwen3.5 pretokenizer |
| `model/`     | runtime (sequences and their histories, passes, stages, the arena, placement), one module per architecture under `arch/` chosen by the registry (qwen3 and qwen3moe, qwen35 and qwen35moe) with their shared graph pieces, KV cache and recurrent state slots, layer split over devices, tensor shards |
| `backends/`  | Backend interface + cpu/ (AVX2) and vulkan/ impls; one worker pool; `device_profile.hpp`, the device numbers a GPU backend shapes its kernels by |
| `inference/` | model loading, sampler, log-probabilities, generate, perplexity, chat template renderer |
| `server/`    | multi-user server (`docs/SERVER.md`): HTTP layer, scheduler with prefix reuse, its policy core and sampling threads, routes |
| `cli/`       | thin argument parsing + dispatch               |

## Starting a feature

A fresh agent (or human) can jump straight into a feature by reading, in order:
1. `AGENTS.md` - this file: what the project is, how to build and verify.
2. `docs/ARCHITECTURE.md` - the layers and the dependency rule.
3. `docs/ROADMAP.md` - the stable plan; pick or confirm the feature there.
4. `docs/STATUS.md` - what is already in flight and where each feature stands.

When you start (or pick up) a feature:
- A new model architecture is a module under `src/model/arch/`, built in the order and held to the gates `docs/ADDING-AN-ARCHITECTURE.md` gives.
- Keep each independent feature on its own branch based on current main.
  Do not base unrelated work on another unmerged feature. If a dependency is
  necessary, name it in STATUS and keep the dependent change separate.
  A feature's own documentation ships in the same commit as the code it
  describes; a documentation correction unrelated to the feature goes on a
  branch of its own.
- Open a new per-feature block in `docs/STATUS.md` (or update the existing one)
  **before** writing code: **Goal / Done / Left / Gotchas**. That block is what
  lets the next agent pick the feature back up with a "continue feature X"
  prompt, so keep it current.
- A feature lands with a test the hosted workflow runs, or its STATUS block names the hand check that covers it and says why no hosted runner can run it.
- A bug fix lands its failing test first: a commit that adds the test, failing on the unfixed code, then the fix that makes it pass.
- A branch lands as at most two commits: the failing test, where it has one, then the change with its tests, its documentation and its STATUS entry.
  Its working commits, review fixes and checkpoints are squashed into those before it lands, so main's history holds what changed and why, not how the branch got there; a docs-only branch lands as one commit.
- Don't invent new directions - follow the roadmap. When the feature ships,
  delete its block and mark the row `Done` in the STATUS table.

## Merge gates

A branch runs the checks of what it can break, once, when it is complete; between its commits only the builds, CTest and a quick Qwen3-0.6B identity on the CPU run.

- Docs only: the suite's `docs` and `dead-code` components, run on the tree that lands; the hosted run is not waited for, since only those two components read the Markdown.
- Tests or tools only: CTest and the suite components they touch, on the CPU, and the hosted run.
- CLI, server or other host logic: CTest, the CPU suite, Qwen3-0.6B byte identity against main, and the hosted run; host code specific to Windows also runs CTest and the suite on Windows.
  A change to the server, the scheduler or the suite's `server` component also runs that component on an MI50 (`--device vulkan:0 --only server`): its checks that send signals do not run on Windows, so the Radeon VII does not cover them, and no hosted job has a device.
- Model, kernel, device, loader or placement code: byte identity against main on the CPU and a device for the models the change reaches, the suite on a device, the Radeon VII check on Windows for device code, one timing round against main, and the hosted run.
  A change of numerics also records its error against the HF reference and the headroom to its bound.
- A rebase without a conflict in code reruns the builds, CTest and the hosted run; a fix after review reruns what it touches.

A branch lands when its checks pass at the head that lands (the hosted run green, but for docs only) and main has not moved since: main is fast-forwarded to it.
The landing commit's own STATUS entry is the record: it gives what the gates measured and says the change lands by fast-forward, and the merge adds no commit of its own; the commit's hash is in git, and its hosted run is found on GitHub by that commit.
Comparisons against the reference runtimes belong to the phase's final gate, not to each branch.

### Traps the gates exist for

Each of these let a wrong result or a wrong measurement through here once.
They are part of the gates, not advice.

- **A check counts only for the path it reaches.** A tiny model below a fast path's size threshold runs another kernel, and a green result says nothing about the path it skipped. A numerics change names the path each check exercised and proves it ran, through a counter, a log line or a forced-path unit check, and exercises the threshold's edges.
- **Short checks do not bound long ones.** A device prompt path with 8-bit activations passed every tiny fixture and the 512-token HF windows, then missed 6 to 8 of 512 top-1 tokens on a 9B model at 16k tokens against HF on the file's own weights. A change of prompt or decode numerics on a device runs the long-context checks, the raw and the chat-template 16k sequences against file-exact HF, before it merges.
- **A bound follows the precision a path computes in, never the weight type.** F32 bounds hold F32 paths. A path in another precision is held to a budget calibrated for that precision: a reference run with the exact weights, activations rounded to that precision and F32 sums, measured against file-exact HF over the dense, tied and MoE fixtures and extra seeds, then frozen with its headroom stated before any candidate is rerun. A budget is never set just above a failing candidate, and the real-model bounds are not widened to let a path through.
- **Correctness is not traded for speed by default.** A faster path in a lower precision than the rule gives (8-bit activations, say) is an opt-in with its own bound, never the default, and never decides a merge of the default path. A fallback is never narrower than the precision asked for, and says so when taken.
- **Measure what a user runs.** Timing runs at the device's default clocks and power state. Setting performance levels or holding clocks high flatters a kernel and makes its figures incomparable with any other run; every arm, llmx before, llmx after and the reference, runs on the same device in the same environment (Principles, above).
- **An open speed gate is recorded, not waived.** First support of a model or a type may merge while it is below the reference, with the cells below it listed in STATUS and the recovery work named, and that work follows at once. It never covers a slowdown on a path main already has, which the timing round against main catches at every merge.
- **A review finding is rechecked before it is acted on.** A reviewer's finding, a person's or a tool's, is checked against the code, then fixed or rejected with the evidence beside it.

## Checkpoints

Update `docs/STATUS.md` and commit at each meaningful checkpoint - at minimum
when a feature, a milestone, or a discrete chunk of work is complete. Each
commit should leave `docs/STATUS.md` accurate: `Done`/`Left` reflect reality,
the build passes, and tests are green. A fresh agent should be able to read
STATUS.md and resume exactly where the last commit left off.
These checkpoints live on the branch; it is squashed as it lands (Starting a feature, above).

At every completed checkpoint, review all project Markdown files against the
current code, tests, CLI, build configuration and recorded results. Correct stale
claims and links, distinguish historical measurements from current status, and
record the review in STATUS.md. This includes README, AGENTS, all docs/ pages
and per-source documentation; update only what needs changing.

Build identification is automatic: both supported build paths combine the
explicit release version with Git revision and tracked-dirty state. Update
CMake's project version and the fallback config together for a release; do not
increment versions or create tags merely because a build ran. `--version`
reports the embedded value, with `unknown` for builds without Git metadata.

## Dead code and stale docs

What nothing reaches is removed in the change that leaves it unreached (Principles), and a doc names only what the tree has: a change that removes or renames a path, a flag, a command, a test, a heading or a name in the code corrects every doc that names it.
Every hosted job holds the tree to both, and a finding fails the job:

- The compilers: an unused function of one translation unit, or an unused local, is an error in the CMake builds and in `build.bat`.
- The suite's first two components, `dead-code` and `docs` (Tests, above), read the source, the Markdown and the comments in a few seconds.
- The Vulkan job also runs `python -X utf8 tests/dead_code.py --linked build-linked`, which builds every target once more to see which functions the product's executables keep.

**A finding is fixed in the change that makes it**: remove the code, or correct the doc.
Only a finding that has to stay is listed, in `tests/data/known_findings.txt`, as `check | file | name | times | reason`, `times` being how often it occurs and the reason saying what it is and what removes it, such as the branch it goes with.
A listed finding that no longer occurs, or occurs another number of times, fails its check too, and so does a line whose check neither component has, so the change that removes one also removes its line, and the list does not rot.
A new line needs a reason a reviewer accepts, not a wish to merge first.
An interface that a later step of an approved plan gives its product caller is such a reason: the line names the plan and the step, as `layer split phase 3, step 2`, and goes in that step.

**What the checks rely on:**
- A struct whose fields mirror a layout something else reads, such as a kernel's structure, is pinned by a `static_assert` with `offsetof` or `sizeof`, which tells the name check its fields are read by position.
- An enum that a value is cast to from an integer has its values reached by arithmetic, so the name check leaves them out; kernel ids are the shader check's.
- A doc's sections: a heading carrying a date opens a record, and a heading saying "planned", "Order of work", "Open questions", "Not chosen", "Out of scope", "Non-goals" or "Migration order", like all of `docs/ROADMAP.md` and `docs/QWEN35.md`, a plan.
  A record's or a plan's names may be gone or not exist yet, so only their links and line numbers are checked.
  In `docs/STATUS.md` only the status table, the active blocks and the working rules are live; elsewhere a section is live unless a heading above it opens a record or a plan, and live text is held to the tree.
- A line number in a doc names the commit it points into, `vulkan_backend.cpp:581 at 0123abc`, since the line moves with the next edit.
- A reference to a heading of the same doc says where it is, `(Tests, above)`, and one to another doc's names the doc, `(AGENTS.md, Tests)`; a heading's name in prose without either is not seen.
- A source CMake does not build is built by the linked check beside its targets (`HAND_BUILT` in `tests/dead_code.py`); a doc naming a source does not count as building it.
- The tree the checks read is what a commit would hold: the files git tracks and the new ones it does not ignore, or every file of a copy without `.git`.
- A `docs/src` page's title names its file or its directory (`` # `src/core/json.hpp` - minimal JSON parser ``), and each source file is named by exactly one title.

**What no mechanical check sees** is left to the review of the Markdown at each checkpoint (Checkpoints, above) and at each merge.
In the docs: whether prose about behaviour, a refusal, a crossover or an exit status is still true; numbers not written beside the constant they are; a counted fact repeated in several docs (the suite's components, the consumer's checks); whether a measurement is current or history; history paragraphs in live text, which move under a dated heading; names of other programs; planned names in live design text; whether a page describes its file as it is now; what a pinned line says; a finished plan still headed "planned"; and one concern described differently in two docs.
In the code: code that runs but serves nothing, paths for inputs or devices that never occur, whether a function only tests call is a probe worth keeping, a virtual function whose name is too common for the override check to tell its calls apart, generality nobody uses (a default never overridden, a template parameter with one instantiation, a seam with one implementation), table rows and enum values no path selects, and a feature implemented twice.

## Build-time vs runtime

- **Backends** are the only compile-time concern (GPU SDKs are heavy). Gated by
  `LLMX_HAS_BACKEND_*` in `src/config.hpp` (see `cmake/llmx-config.hpp.in`).
  Vulkan is the only optional backend today; ROCm, CUDA and SYCL are planned and each gets its option with its implementation.
- **Model architectures** are compiled in and selected from metadata by
  `src/model/arch/registry.hpp`, one module per architecture under `src/model/arch/`
  on a runtime they share; today qwen3, qwen3moe, qwen35 and qwen35moe. A new one follows
  `docs/ADDING-AN-ARCHITECTURE.md`.
- **Split mode** is a runtime flag: a `--device` list selects a layer split
  (`docs/MULTI-DEVICE.md`); the tensor split and node count are planned. See
  `docs/ROADMAP.md`.

## Conventions

- mx-llama.cpp may be inspected as a reference, but do not copy its source.
  Implement llmx changes independently in this project's architecture and style.

- Use ASCII characters in code, comments, documentation and commit messages.
  Preserve Unicode test coverage using escaped literals and fixture data.

- Code comments are short and carry only what helps read the code: what a thing is, an invariant it keeps, or a non-obvious reason, in a sentence or two.
- No walls of text in code. Measurements, timings, rejected alternatives and history belong in the docs (`docs/STATUS.md`, `docs/VULKAN.md` and the like); a comment may point there.
- Do not break a sentence across lines in code comments or commit messages. A line ends where a sentence ends; a long sentence stays on one line rather than wrapping at a column.
- A file's comments follow these rules once the file is touched, and a branch's files are swept before it merges.

- CPU and shared runtime code use headers (`#pragma once` + `inline`),
  compiled via `src/cli/main.cpp`. The optional Vulkan backend has its own
  compiled translation unit. Keep one TU per logical unit when adding `.cpp` files.
- Include paths are relative to `src/` root: `#include "format/gguf.hpp"`.
- Every source file under `src/` is described in `docs/src/`: a page per file
  (`<dir>-<file>.md`), or one page for a directory whose files form one unit
  (`server.md`, `hub.md`, `backends-vulkan.md`). A page changes in the same
  commit as its file whenever what it says changes; a new file comes with its
  description, and a removed file takes it with it.
- No comments in code unless they explain a non-obvious decision or algorithm (e.g. the fp16 rounding, the GGUF padding rules, the AVX2 dequant+FMA path).
- Cross-platform (Windows / Linux / macOS): guard MSVC-vs-GCC intrinsics with `#if defined(_MSC_VER)`.
  AVX2, FMA and F16C are the build baseline (`docs/BUILD.md`), so the x86 kernels include `<immintrin.h>` and use them with no runtime check; a kernel beyond that set, such as the deferred AVX-512 path (`docs/src/backends-cpu.md`), needs its own runtime dispatch.
- Don't overengineer. Add a seam (interface) only when a second implementation
  is actually on the roadmap. Empty stubs are discouraged.

## Roadmap

`docs/ROADMAP.md` lists: more quant formats, more architectures, more formats,
backends (Vulkan written first, ROCm first-class on Linux), per-layer and
per-tensor multi-device split, multi-node cluster, a multi-user server, and
Hugging Face integration (`llmx pull` plus reading what the Hub actually
hosts). Follow the roadmap before inventing new directions.

GPU backends are **not** drop-in the way a quant type is. The device
execution model (`docs/ROADMAP.md` #4a, `docs/DEVICE-EXECUTION.md`) and the
Vulkan backend over it (`docs/VULKAN.md`) are complete; a further vendor
backend implements the whole `Backend` interface over its own allocator,
kernels and queue, and the Vulkan backend is the measure of what that
takes. Don't pick up "add the ROCm backend" expecting the "one file + one
registry entry" experience Q4_0 had.
