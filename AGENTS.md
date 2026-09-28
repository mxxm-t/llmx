# AGENTS.md

Guidance for AI agents (and humans) working in this repo. Read this before
making changes.

## What this is

**llmx** - a ground-up, dependency-free LLM inference runtime. It reads/writes
GGUF v3, runs quantized or F32 Qwen3-style transformers on x86 CPU with AVX2/FMA/F16C
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
- **Test configuration**: `LLMX_BASELINE_GGUF` points `tests/baseline.py` at a fixture model; `LLMX_DEVICE`, set by `run_tests.py --device`, appends `--device` to every command that takes it so the suite runs on a device backend, except where a component names its own devices (`threads` names the CPU, whose thread counts it checks, and `split` names the CPU backends its tool splits over); `LLMX_CACHE_TYPE`, set by `run_tests.py --cache-type`, appends `--cache-type-k` and `--cache-type-v` the same way so the HF gate runs with a chosen cache type; `LLMX_LAYER_SHARES`, set by `run_tests.py --layer-shares`, appends `--layer-shares` so a device list is tested at a split the fit would not choose.
  The synthetic bench (`bench` without `--model`) takes only `--device` and `--threads`, so it gets neither shares, cache types nor a load mode, and `split` gives its tool equal shares and the cache types it names itself.
  The runtime stores f16 by default, so the components that check exact f32 arithmetic against independent fixtures (`f32`, `moe`, `qwen35`, `shards`, `server`) ask for f32 sides themselves and skip when `--cache-type` asks for another type (`common.f32_cache_skip`).
  `LLMX_LOAD_MODE`, set by `run_tests.py --load-mode`, appends `--load-mode` to every model command the same way, so the suite runs with the weights read in any mode.
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

## Tests

Run the native backend, model, chat-template, KV storage, loader and streaming checks
after a CMake build:
```
ctest --test-dir build -C Release --output-on-failure
```

`json` checks syntax, numeric/locale boundaries, UTF-8 and escaped Unicode,
malformed input, nesting limits and JSON output string escaping.
A float written by `number` is its shortest decimal in the C locale's form under a comma locale too, reads back through `parse` as the same float for a million random bit patterns, and is `null` when JSON cannot hold it.
The Q8/Q4 round-trip test also checks escaped Unicode tensor names through the actual CLI.

`format-output` checks conversion staging, serialization and stream failure,
refusal of premature publication, a later publication failure after the first
complete file was published, and temporary cleanup. The Python round-trip adds
real child-only file-size-limit failures on POSIX, existing-file preservation,
a failed second output open and aliased raw output refusal.

`gguf-validation` checks independent binary fixtures for field lengths/counts, array depth, tensor arithmetic, byte counts that overflow although the element count fits, in one row or across rows, file extents, tensor types and quantized row widths, each refused as such when the element count also overflows, custom alignment and a tensor name repeated in one file.
These are format checks; they do not establish model-schema safety.
`load-progress` reads, maps and reads in a file as the loader does, and checks the progress, each tensor's file span, that reading the headers maps nothing, that a model not mapped is neither written nor read in, early rejection, and a file truncated before loading or whose size changes between reading and mapping, refused before any progress.
The loader's readers check a file's size against its header with the mapping's own check, so a changed size is refused the same way.
It then writes a tiny Qwen model with tokenizer metadata and loads it through `infer::load_model` in each load mode: on the CPU the payload is kept, and on a CPU backend that copies what it adopts and reports `reads_in_place()` false the host copy is released (`payload_size()` is 0), streamed from one file in `auto`; a direct load on the CPU maps nothing and its copy holds each weight where the file does.
Every load's progress starts at 0, only rises and ends at the payload, and every load gives logits bit-identical to the same model built in memory, with the tokenizer and the chat format loaded beside them.
Each load runs on a copying backend that reads the read ring in place, which takes every part as a copy out of it, and on one that does not, which takes writes.
A write that fails part way through a streamed load, and a progress callback that throws, stop it with their errors on the asynchronous test backend `model-validation` uses (`tests/loading_backend.hpp`), and no buffer is freed before it drains.
A qwen3moe model with its experts on the CPU beside a copying backend loads in each mode; so does a split over two copying backends with a tied head, whose embedding reaches both and whose every write goes to its own backend's storage, and the CPU beside a copying backend, whose shared embedding is read and reported once.
A set of three shards with a metadata-only first loads in each mode, streamed from the two shards holding tensors.
The planned reads are checked on spans with no file: pieces on the granule within the limit, a gap of one granule read through and a longer one starting a new piece, a tensor longer than a piece, a new file, and every tensor's bytes in exactly one part; so are the rule for reading around the cache, at, below and above the memory available, and a reader that falls back to the cache where direct reads are refused.
The stream itself runs in reads of one page, so tensors cross reads and reads hold several tensors: every copy, two of one tensor included, holds the file's bytes, and a file cut after its header was read stops it with where the file ended.
In dozens of 64-byte reads with slow fills the readers fill the ring and wait, and every copy still holds the file's bytes, filled by copies, by writes, and by copies that run only when their submission is waited for, as a device's do, which a slot refilled before its copies retire would spoil; a fill failing late stops the stream; where direct reads are taken, a read that fails on a reader thread comes out of the stream.
`file-reader` checks `format::FileReader` and `core::HostPages`: owned pages and their moves, a reservation's rounding, a size past the largest whole number of pages a `size_t` holds refused by name by both and the largest that rounds refused by the operating system, commits that start and end inside a page and repeat, commits outside it refused, and a decommit that keeps the pages it covers only in part; reads at any offset and length on a file of several granules, a read past the end short by exactly what the file lacks, empty and tiny files, and eight threads reading one reader at once; where the test's file system takes direct reads, aligned direct reads, a rounded one past the end, and the refusal of a misaligned one, and where it does not, the refusal naming the file.
The loader's checks run in `direct` too where the file system takes direct reads, and check its refusal where it does not.

`gguf-shards` covers complete shard sets, metadata-only first shards, exact
payloads and the shard each tensor's span names (a zero-sized tensor that ends
a shard included), inconsistent metadata,
duplicate tensor names, truncation before reading and between reading and
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
The fit counts what the model reads: a routed layer that also carries dense matrices is fitted without their width in its activations and without their bytes among its weights.
Every refusal it provokes must give the label and text listed in `tests/data/model_refusals.txt`, in order, so a change to a refusal's text or to which defect a file is refused for changes that list; `llmx-model-validation-test --write FILE` writes the refusals it sees.
An asynchronous test backend (`tests/loading_backend.hpp`, which `load-progress` also uses), which plays a device, so it copies what it adopts and says it is not the CPU, also checks loading failure and model teardown drain pending work before releasing buffers, including split placements and backend reuse.
Through the loader's own adoption hook (`infer::planning_adopt`), it checks that nothing is read in place on that backend, and that a host running a streamed layer's experts beside it reads exactly that layer's feed-forward norm, router and three expert stacks, the norm and router taken by both.
It checks the hook both ways: deferring the copies, as the streamed load does, every copied weight gets storage and nothing is written or adopted while the model is built, and a model missing a tensor, or whose cache does not fit, fails after storage but before any weight is uploaded; without deferring, as the mapped load does, every weight is adopted as the model resolves it.
It does not validate numeric weights, arbitrary token IDs or failed-session recovery.
Each model it builds runs on a one-thread CPU backend that must start no worker threads.

`backend-group` checks mixed types, uneven rows, batches, thread counts,
output boundaries and fallback behavior against separate calls and double dots.
It also checks every finite f16 scale against signed Q8 weight extremes using
one-hot inputs with exact expected products, and every column of the three-, two- and one-column prefill dots against one ordered scalar FMA oracle across dimension tails and unaligned inputs.
The test is built without contraction, so a tail the kernels leave to the compiler fails it.
The independent HF fixtures below remain the external correctness gate.

`fused-dot-overflow` pins the two kernel families apart. `dot_row_impl` folds
the scale into each weight before the activation, avoiding the demonstrated
scale-after-sum overflow; accumulation can still overflow under cancellation.
The fused K-quant dots accumulate first and apply the scale after,
which is what makes them fast and what lets a large activation reach infinity
before a small scale could bound it. Those rows fall back to dequantizing.
Sixteen cases across Q8_0/Q4_K/Q5_K/Q6_K cover tiny and zero scales against
huge and ordinary inputs, once on the float dots and once on the decode dots
over quantized activations, which cannot overflow since each block is scaled
first and are bounded against the sum of magnitudes.

`quantize-range` checks Q4_0 packing from finite F32 inputs against exact integer expectations: every code from -7 to 7 at every position, with power-of-two scales across exponents -149 through 124, half-way rounding, rounded subnormal scales, output guards and a scale that underflows even in F32. Binary16 scale underflow and overflow remain format limits; this checks defined packed codes, not finite decoded weights for every magnitude. It runs with gradual underflow and does not establish nonfinite-input handling.

`q8-dots` checks those decode dots (`backends/cpu/q8_dots.hpp`), 8-bit for
Q8_0/Q4_K/Q5_K and 16-bit for Q4_0/Q4_1/Q6_K, against a double-precision
reference fed the same quantized activations, and that a decode row computes
the same alone, beside other rows and in a grouped call, bit for bit. It also
checks activation quantization independently against the original inputs over
float exponent boundaries, tiny/subnormal blocks and reciprocal-overflow
thresholds: packed range/sign/zero, exact integer sums, nearest reconstruction,
ties to even and partial-block guards. Fallback reconstruction checks are
strict; ordinary SIMD controls allow its existing float-rounding error.
These run under gradual underflow;
they do not establish nonfinite-input handling or flush-to-zero behavior.

`qwen35-ops` checks the CPU backend's ops of the qwen35 layers (`backends/backend.hpp`, `docs/QWEN35.md`) against references the test writes from the math in double precision, each within a bound stated beside it as a count of F32 units of the values' magnitude.
It covers the causal conv and the raw rows it carries, which must be the inputs bit for bit, and the gated delta rule's rows and state, at Hv = Hk and Hv = 3 Hk with the tiny fixtures' widths, at a width that takes the 32-column blocks and at the files' 128 by 128: fresh sequences whose slots hold NaN, histories shorter than the conv's window, one-token entries and a verify that reads one slot and writes another, side by side in one call.
Each V column of the recurrence must give the same bits in a 32-column block, in an 8-column block and alone, and q and k heads near 1e-4 hold the L2 norms' epsilon and its place inside the root.
A decay factor below 2^-126 must be flushed to 0 and one just above it kept, on a state whose decayed values stay normal, so the check holds under any denormal handling.
It also checks the gated norm, the partial rope, reading q between its gates and k in place, and `sigmoid_mul` as the output gate and as a scale of one value per row; the norms each with a head near 1e-4 that holds their epsilon.
The partial rope's reference rotates every pair at the token's position, as the rope sections give it for text, so it holds nothing of the sections themselves; the op must equal `norm_rope_rows` at the full rotary width and rotate a pair in its scalar tail with the bits of the same pair in its vector body.
Every result must be the same bit for bit at thread counts from 1 to 16, for a row alone and beside others, with the views in another order, and with a sequence cut into passes of 1, 2 and 3 rows that carry its state in its slot, so k one-row calls equal one call of k rows.
`state_alloc` must zero-fill its slots, `state_copy` copy one slot in every layer and a storage whose buffers cannot hold its slots be refused; malformed views, among them each of the three ways one view's slots can meet another's, must be refused before anything is written, and `Backend`'s own form of each op, which the Vulkan backend runs, must refuse it by name.

`prefill-scope` uses self-generated model fixtures to check caller-once execution,
nesting/thread guards, allocation and microbatch boundaries, error draining and
scope reuse. Windows-only `prefill-placement` covers real eligible topology,
unsupported topology fallback and synthetic apply/restore failures without
requiring a six-core hosted runner. Neither replaces the independent HF gate.

`fp16` checks binary32 to binary16 conversion without an oracle library:
every finite half must encode back to its own bits, and every encoded float
must be at least as close as either neighbouring half, ties to even.

`cpus` holds the automatic worker count of `core/cpus.hpp`, and the cgroup reading of `core/cgroup.hpp` it goes through, on file texts and field values rather than a real cgroup or job object.
It reads cgroup v2's `cpu.max` ("max 100000" as no limit, "600000 100000" as 6 CPUs, "150000 100000" rounded up to 2) and v1's `cpu.cfs_quota_us` over `cpu.cfs_period_us` (-1 as no limit), every malformed text as nothing, the process's cgroup in `/proc/self/cgroup` for v2 and for v1's cpu controller, refusing a path that is not absolute or climbs, and the cgroup mounts in `/proc/self/mountinfo` with their roots and escapes.
Over a file system held in a map it reads the quota in the process's own cgroup, found under its mount by taking the mount's root off its path, and in each cgroup above it up to the mount point, v1, v2 and both, the smallest winning: a child cgroup inside a container is read, a cgroup inside one that shares a name with the container's path on the host is not, and a path no mount's root holds reads nothing.
It reads a job object's CPU rate hard cap as CPUs of the active processors, rounded up, and takes the automatic count as the fewest of the hardware threads, the affinity and the quota, 4 when none was read, from 1 to 64.

`host-memory` holds what `core/host_memory.hpp` counts as the memory a process can still take, on file texts and field values rather than a real cgroup or job object.
It reads cgroup v2's `memory.max` ("max" as no limit) and v1's `memory.limit_in_bytes` (2^62 bytes or more as none) less the working set, the usage (`memory.current`, `memory.usage_in_bytes`) less the inactive file pages `memory.stat` gives, `inactive_file` on v2 and `total_inactive_file` on v1 and never the other's key or `inactive_anon`.
A working set below 0 is 0, a room past the limit is 0, a `memory.stat` that is missing, lacks the key or gives it as anything but one decimal number leaves the usage whole, and every other malformed text reads as nothing.
Over a file system held in a map it reads the room in the process's own cgroup and in each above it up to the mount point, v1, v2 and both, the smallest winning: a parent's tighter limit is taken, each cgroup's `memory.stat` is read at its own level, v1 is read in the memory controller's cgroup and not the cpu controller's, and a limit without its usage or a malformed file is passed over while the cgroups above are still read.
It reads a job object's process and job memory limits less the process's and the job's commit, and takes the figure as the fewer of the host's available memory and that room, each only when it was read, so a container limited to 8 GiB is given what its limit leaves and, without the limit's files, the host's figure stands.

`sampler` calls `infer::sample` on hand-picked logits with expectations taken from the definitions.
Temperature 0 takes the largest score and the lowest id on a tie.
The repetition penalty divides a seen token's positive score and multiplies a negative one, by the penalty and once however often the token was seen, so a repeated leader loses to the runner-up.
`top_k` 1 is greedy at any temperature, `top_k` 3 never draws outside the three best, and a `top_p` between the first probability and the sum of the first two keeps exactly those two.
The nucleus is measured after the temperature and within the `top_k` window, which default requests combine: at temperature 2 a `top_p` of 0.7 keeps three tokens, and `top_k` 2 with `top_p` 0.6 keeps one.
Draws at temperatures 1 and 0.5 fall within three binomial standard deviations of the softmax at that temperature, a bound that rejects the temperature applied twice or ignored.
A seed repeats its sequence and seed 0 keeps the default state; every seed is fixed, so the frequency checks draw the same tokens on every run.
A masked id, the end of text under `ignore_eos`, does not exist for the draw: greedy takes the best of the rest, the penalty cannot bring it back, `top_k` 3 keeps the three best of the rest, a `top_p` of 0.7 is measured without it, draws follow the softmax of the rest, an id past the row changes no draw, a masked id 0 is not given beside scores that are all negative infinity or NaN, greedy or drawn at top-k 0 and 1 with top-p 0.95 and 1, and the settings' overload masks the end id only with `ignore_eos`.
Every draw is also held, with the generator's state after it, to a slow reference in the test that sorts the whole row but a masked id by score and then id and sums the softmax in id order.
The draws pin the ranking, the tie rule, the order a draw walks the tokens in and the generator's use; a sum in another order than id order differs only in its last bits, so no draw shows that order, and the reference only follows it.
Its rows are random, 1000 scores on five or twelve levels so that ties fall across every cut, or spread as a model's are, at top-k 0, 1, 40 and 1000, top-p 0.1, 0.95 and 1, temperatures 0, 0.2, 0.8 and 1.5 and penalties 1 and 1.1, each cell twelve draws from the next seed of a fixed sequence with the history growing, and half the cells masking the row's leader, each top-k, temperature and top-p under one penalty.
Rows of 40000 take a nucleus past the 512 tokens the sampler ranks a nucleus with by heap and through the selections that double it, and a top-k of 5000, past the 4096 of a top-k set, through its selection over every key.

`logprobs` checks `inference/logprobs.hpp` against a double-precision reference that shifts by the row's maximum and sums with compensation: every value of 151936-token rows of three spreads, a row with one certain token, logits too large for `exp` unshifted, logits wholly below -1e30, equal and tied logits, within half a float step, the probabilities summing to one, `token_nll` the same quantity in double, and the top lists of 0, 1, 5, 20 and more tokens than the row in the order of a full sort by logit and then id.
It then runs the scheduler over `tiny_qwen` with a tokenizer of its 16 tokens and no end token: a greedy request with `top_logprobs` 5, read from its token channel, must carry at every position the log-softmax and top five of the logits row a second model gives through the same passes (the prompt at its extent, then one decode entry a token), and the same request without logprobs the same ids and no values; so must a request drawn at temperature 1.5 with penalty 1.3 and a seed, whose values are the raw row's though some of its draws are not greedy's choice.
The same request left unread until it ends must hold exactly `Request::kRowsWaiting` rows on its channel and then give the same values, those past the rows computed by the scheduler; cancelled before it is read, its waiting rows come without values and the rest with the scheduler's.

`server-resume` runs uncapped greedy requests with `top_logprobs` 5 through the scheduler over the synthetic Q8_0 model with a token list and no end token, whose decode rows take the CPU's 8-bit dots and whose prompt rows the float path, on a pool too small for them: each alone first, where it never pauses, then together, where the scheduler pauses and resumes them, and every id, logprob and top entry must equal its run alone, a difference naming its first token.
Once every request has ended, the blocks the scheduler holds reserved must be its donors' blocks and never more than a pool has (`Stats::reserved`, `Stats::donor_blocks`).
With three requests the resumes must have recomputed more rows than each pause's longest prompt (`Stats::recomputed`), so generated tokens among them.
The cases: three requests on 8 blocks of 128 tokens, a victim paused with generated tokens in its partial block whose donor the growth that paused it then takes; a prompt read one token a pass (`ubatch` 1) paused while it is still prefilling, which takes nothing back and recomputes at least the two blocks of its prompt it had read; a follow-up turn that forked the previous turn's reply rows, paused with every donor gone, which recomputes its whole 512-token history; the three requests over a two-CPU layer split; and paused requests cancelled, after which the scheduler holds nothing and, once stopped, the pool is empty.
The take-back cases end the paused request with a stop string found from its reply alone, so exactly one pause happens: an uncapped request paused by an earlier one's growth with its donor intact, which takes the donor back whole when a capped request ends (`Stats::taken_back` 1, nothing recomputed); the same for a follow-up turn whose history holds the previous turn's forked blocks; a request whose own donor went while a capped request that forked the same setup prefix kept that block, which it forks and then recomputes the other 384 rows by class; the same where the capped request's prompt holds the paused request's first 80 tokens, so its donor matches that history by tokens past the first block but holds those tokens as prompt rows, and the resume still forks only the first block; and a paused request cancelled while its donor waits, which takes nothing back and leaves the pool empty once stopped.
Three cases hold room to first admission: a resumed request whose next growth step finds only room that capped requests hold sits out passes for it rather than being paused again (one pause, one donor taken back, each reply its reply alone), a request refused room evicts no donor and starts only once the capped request holding the room has ended, so a prompt repeating that donor's prompt still forks its block, and with a queue of one a request submitted while another is paused is queued rather than refused and starts only once the paused request has resumed.
A request whose growth step a capped request's room holds sits out more than a block of passes, every reply its reply alone and ending by its length or stop text, and a capped request submitted meanwhile starts only once the capped request holding the room has ended.
An older request's growth step that falls due in the iteration a newer capped request could first be admitted takes the room first, so no request sits a pass out, and a paused request cancelled while its donor holds less than a block leaves no donor behind.
Two cases act from inside a pass's stage through a CPU backend that runs a hook as the model submits its work: on two and on three CPU stages of a four-layer model, a request cancelled while its pass is in flight ends only once that pass has retired, its history kept as a donor that a request repeating its prompt forks (128 tokens reused); and on one and on three stages, a stop while a pass is in flight ends the request cancelled with nothing left reserved.
After either, a prompt as long as the pool runs on the model's own history, so every block came back, a block of a request whose handle is still held included.
In a build with Vulkan, when device 0 opens, it repeats the three requests on the device and adds a 100-token prompt that forked the first block of a 600-token prompt's history, whose tile splits its sums otherwise, paused with every donor gone, which takes nothing back and recomputes at least its first 384-token reservation.
`llmx-server-resume-test cpu` or `device` runs one half.
The shared wave harness holds pass retirement while a wave is queued, so a descheduled submitting thread cannot let its first request run ahead and change the pause scenario; the first pass may start before queuing finishes.

`server-passes` drives the scheduler's policy core (`server/policy.hpp`), first by hand.
The growth rule (`Growth`): admission reserves a capped request's history and what it may still generate and an uncapped one's history and a step, a step falls due only for an uncapped decoding request whose next position passes its blocks and reaches a step past that position, never past what a pool holds, the logits rows a context reserves are one pass's alone and twice that once passes overlap, and the decode share is the decoding requests over the passes rounded up once the passes fill the stages and every ready one with fewer.
`make_room`, the one owner of who gives up blocks for whom: a request that fits or cannot fit takes nothing, the donor it forks goes last or first, growth pauses the latest uncapped requests admitted after it, headroom before donor, and never one admitted before it or a capped one, and a plan that would pause a request in flight waits and takes nothing while one pausing only requests not in flight, or holding a capped request in flight, does not.
`round_steps`: each stage records its oldest waiting pass from the last stage down, a free slot none, and passes past their last stage retire oldest first.
The logits rows: runs follow the newest and wrap to row 0 but never over the oldest run, come back oldest first, a run given back early waiting for the ones before it, and an empty ring starts at row 0.
Then through the scheduler's round under a simulated executor, the pools, the growth rule, `make_room`, `round_steps` and the logits rows being the scheduler's own functions and the round's order restated from `Scheduler::run`: once by hand, where a growth step due beside a queued request that fits only in the room the step needs takes that room and the request waits, and in random schedules, 2000 by default (`llmx-server-passes-test N` runs N), over 1 to 4 stages, 1 to twice that many pass slots and two pools of different block sizes.
The round is the scheduler's at any number of slots: a pass formed in every free slot while a request is ready, each taking the decode share of the decoding requests, those that left flight earliest first, each request in one pass at a time, and the host's stages recorded after the round's device stages.
A simulated executor runs stage s on device s, the host waiting for a stage's handoff before it records the next and for a pass's last stage before it samples, each stage taking a random time, on the host's thread for a stage on the host, one stage or every one of them in some schedules; submissions arrive with the host's time, and requests are cancelled wherever they are, in flight and in a pass formed but not yet recorded, passes fail at a random stage or as their logits are read, and each schedule ends in a stop at a random round or, one in eight, once every request has ended.
After every event a request is in at most one pass and no pass is empty, each device runs its passes in formation order, a slot and a run of logits rows belong to one pass until it ends, no pool is over-reserved and the reservations and donors add up, nothing in flight is paused, parked or ended, a request refused room is refused only when donors and the uncapped requests after it could not give it, the oldest only when a capped request holds the rest, admission is first-come, a pass always finds its logits rows, and a free slot stays empty only when no active request is ready.
A growth plan pauses one request at most, since one uncapped request's reservation holds a step, and waits only on a request in flight, and the oldest request's plan that waits takes that request in the round its pass retires, within a lap (S + P - 1 rounds) of the first wait; a younger request's plan may turn to another request in flight when an older one's step takes the one it waited on, and the drained schedules check that every request still ends.
Every decoder that is not stalled gets a token within a lap and a cancellation ends within a lap; a cancelled request in flight is not sampled when its pass retires, a failed pass ends its own requests alone and leaves every other pass in flight, and after a stop the ledger and the logits rows hold nothing; across 1000 schedules or more requests must pause, stall, take donors back and wait on requests in flight, the oldest request's wait must end, passes must fail, and requests must be cancelled in flight and before a pass's first stage.

`server-passes-cpu` runs the scheduler itself over the synthetic Q8_0 model on one CPU and split over two and three, at P = 1, S, S + 1 and 2S (one on one CPU, where two are refused as the scheduler is made), with a ubatch of 16, every request's ids, logprobs and top entries against its run alone on one CPU at one pass in flight.
A paused load, two uncapped requests, one sampled, beside capped ones with prompts past the ubatch, some sampled, on 8 blocks, must pause on every run; a held load, two uncapped requests on 9 blocks whose older one's growth step falls due four tokens before the younger's, must pause once on every run and, at P = S on a split, wait on the younger in flight (`Stats::waits`).
A steady load that never pauses has each run's passes recorded as they retire (`Scheduler::on_retire`) and replayed in their order through `Model::forward` on a fresh model of the same placement, every logits row bit for bit.
Through a CPU backend that runs a hook as the model submits its work: a request cancelled at the last stage's twelfth or thirteenth submission ends cancelled with a prefix of its ids alone while the three others run whole; the last stage's twentieth submission throwing ends its pass's requests with the error, a prefix of their replies alone, and the others run whole, some always surviving once the passes fill the stages; and a stop at the first stage's twelfth submission ends every request cancelled with nothing reserved or in flight; after each a prompt as long as the pool runs, so every block came back.

`backend-errors` injects task and startup-allocation failures, checks completion
before error propagation, and exercises pool reuse and thread reconfiguration.
It also checks valid empty CPU transfers, rejected offsets/null sources,
unchanged storage and a zero thread hint preserving the current pool.
It pins that construction and count changes start no threads, and that the first dispatch at a count starts one pool of that size, which later dispatches reuse.
A start that fails partway fails its dispatch and keeps the count, and the next dispatch starts the whole pool without a new count.
It checks `quant::row_bytes` against the block layouts over one row and over several, zero rows taking zero bytes however wide a whole-block row, and its refusals of an unknown type and a partial block, with rows or with none, and of a size that wraps in one row or across rows.
Decode and batched `matmul` and `embed` must refuse a Q8_0 row that ends inside a block, and `matmul` and `matmul_group` must refuse row runs that reach past the call, are out of order or fall short of it, all before writing any output.
It does not establish recovery of partially executed model sessions.

`backend-vulkan` exists only in a build with `LLMX_HAS_BACKEND_VULKAN=ON`.
It opens device 0, round-trips buffers through adopt, copy, write and read, checks that a weight written in pieces into `alloc_weight` storage, or copied in pieces into it out of host pages the device imports, holds its bytes and gives the products of the same weight adopted, and that memory off the page is not imported, checks zeroed allocations, host-visible memory read in place after a wait, monotonic tickets and the refusal of a KV budget that overflows, then runs every implemented kernel against the CPU backend on random inputs with bounds fixed in the test: exact where the arithmetic is the same operation in the same order, a stated relative tolerance where a transcendental or a reduction order differs.
The decode row kernel reads quantized rows against 16-bit integer activations, and on
a device whose profile prefers the integer dot the wide tile reads 8-bit
ones, so the CPU reference is fed the activations quantized the same way and
the comparison is about the dots; the norm, SiLU and attention kernels' twin
of their output is checked through a matmul from it. After the checks it
prints the matvec bandwidth per type and, when the device reports them,
the driver's per-kernel statistics (registers, shared memory, scratch);
`--isa DIR` additionally writes the driver's disassembly of each kernel
to that directory.
With `--isa` every row kernel build must then hold its one-column build's counts of float multiplies, multiply-adds that round the product first, fused ones, adds and adds over shuffled lanes: the same counts where the driver keeps the column loop rolled, else those and whole columns of one column's, and for the Q8_0 decode kernel, whose builds also differ in rows, steps, copies of their products and forms, the counts its shape and forms give as the first line of its representation reports them: for each product in the code a multiply and a multiply-add, or a fused one in the one-column build's proportion, the half-block order's step among them where the build takes that order, which must be where its one-column build does, and per row and column the one-column build's shuffled adds and one plain add, the residual add's, or with the transposed reduction that reduction's pairs and one residual add; a grouped build must hold its wide build's counts.
The counts are a screen on how the driver contracts and reduces a column's sums; a reassociation that keeps them shows only in the decode-column comparison below, which is what holds batch invariance.
CTest runs `backend-vulkan` without `--isa`, and no hosted runner has a GPU, so `llmx-backend-vulkan-test --isa DIR` is run by hand on an MI50 under RADV and on the Radeon VII under the AMD proprietary driver at every change to a row kernel, its builds or how a pass is chunked.
Attention additionally covers 80 combinations of head widths 32/40/64/128/256,
query/KV head ratios 1/2/4/8 and all four F32/F16 cache-side pairs, with nonzero
inputs at long histories. Both rows of a mixed short/long pass must equal the
same rows taken separately, bit for bit; CPU comparisons retain the bound
`1e-4 * (1 + abs(reference))`.
Every decode column of every row kernel build must be, bit for bit, the same column computed alone: calls of 1 to 64 generated tokens, and the residual add and a group of three projections at 1, 2, 3, 8, 9, 13, 16, 18, 29, 32, 33, 34, 36, 40, 48 and 64, whose remainders past the widest Q8_0 decode build take each of its builds, over F32, Q8_0, Q4_0, Q4_1, Q4_K, Q5_K and Q6_K rows 4096 and 1280 wide, the first four also 224 wide and Q8_0 also 2560 wide, the output head of Q4_0, Q4_1 and Q6_K on its own twin, and 8 experts routed 2 a token over 1 to 32 tokens on rows 4096 and 1280 wide, each token's gate, up and down against the token alone.
Rows 1280 wide give some lanes of a 64-lane subgroup three steps and every lane of a 32-lane one five, so a build that takes steps in pairs also takes one after them.
A group of a Q8_0 and a Q4_0 projection whose batch reaches the 8-bit tile crossover but not the other types' must equal each type alone forced onto the row kernel, bit for bit, on a device whose profile puts batches between the two.
A matmul whose row runs are out of order must be refused even when every run takes the same kernel.
An `embed` whose F32 or Q8_0 table holds fewer rows than the call names must be refused, even when every id is inside the table.
Each refusal it makes of `matmul`, `matmul_add`, `matmul_group`, the routed products and `embed` is made in two passes, and a valid call after each pass must give what it gave before.
`alloc` and `adopt` leave a new buffer held by the command-buffer slot that fills or copies it, so the first pass submits until no slot holds the call's operands, makes the call and then drops them, and a command the call left naming one fails the next submission.
The second pass writes into an output that must keep what it held.
What a paused request's resume relies on is checked the same way: 40 generated rows of one sequence in one run of extent 1 beside a prompt's rows must equal each row in a matmul of its own, and their attention after a 500-token history in one view of extent 1 beside a prompt's view must equal the rows decoded one call at a time.
It exits 77, which CTest reports as skipped, when there is no loader, no
device or a driverless loader.

`vulkan-buffer` checks constructor cleanup on a fake device that supplies every Vulkan call it makes, so it needs no loader and runs wherever the backend builds.
`vulkan-lifetime` opens a device, intercepts transfers and injects allocation failures to check queued storage ownership during KV growth, padded-copy creation/replacement/invalidation and argument-arena overflow.
It checks retry and unchanged KV accounting after failed growth.
A buffer dropped right after `alloc` or `adopt` must outlive its zero fill or its upload.
`alloc_weight` storage written in pieces that end inside a row holds the bytes and gets the float tile's padded copy, which storage from `alloc` does not.
Five kernel-construction cases substitute calls to check cleanup, poisoned failure outputs, retry and cache reuse.
Two query cases use a real diagnostic add dispatch to check creation failure/retry and destruction after device idle; these cases skip if diagnostic timestamps are unavailable.
Transfer ownership cases intercept copies so old failures cannot submit references to freed memory.
Broad device arithmetic remains covered by `backend-vulkan` and HF.

`http` starts the server's HTTP layer (`src/server/http.hpp`) on a system-chosen port from a thread and drives it with the layer's own client: a whole response, a body echoed back, a chunked stream whose chunks arrive as written, a whole response refused inside a stream, an oversized body refused with 413, a malformed request line refused with 400, an unknown route 404, a client seen by `peer_closed` as open while it waits for its answer, also after one urgent (out-of-band) byte, and as closed once it leaves, a write to it then throwing `ClientGone`, and the listener closed from the main thread ending the accept loop.
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
keeps a handoff buffer. It refuses malformed placements, among them a
device whose attention layers are not one run, and sequences of another
model.
It checks the fit to device budgets (`model/layer_split.hpp`): even shares where room allows, a device without room left out, a host device given only what the others cannot hold, layers placed by their own sizes, the busiest device given as few layers as fit, tied weights counted once, the host's tables, handoff buffers and staging counted where they sit, shares honored or refused, and the fitted placement exact against one device.
`place_model`, the one placement entry, asks the backends for their budgets, applies the request's ubatch, and splits by shares exactly as one device computes.
A three-layer model placed by `place_model` over two and three CPU backends at ubatch 3 takes a 13-token prompt in five chunks, more than the stages, so the pipelined prefill reuses its pass slots and both handoff buffers; the prompt, three decode steps, a second prompt continuing the history, every row of `score()` and a two-sequence pass must be exact against one backend, with the same `n_tokens` and `kv_used_bytes`, and each stage but the last must keep two handoff buffers and the last none.
A backend on the last stage then fails while the first stage is chunks ahead, on top of a history, once at an attention mid-prompt and once at the head on the last chunk (`FailingCpu` in `tests/tiny_qwen.hpp`, which `kv-cache` also uses): every storage's length and `kv_used_bytes` must be back at the history, and the same prompt again must be exact.
`place_model` refuses experts on the CPU beside several devices, on a routed model, and on a model without routed layers, on the CPU as beside a device that reads host memory in place and is not the CPU, each time by the name of the flag given, and a stream point without experts on the CPU; on a routed model, experts on the CPU beside a CPU leave that CPU alone, nothing crossing, with the logits of the model placed without them, and beside that device they run on a CPU placed beside it, the residual crossing each way, with the same logits.
Passes in flight go through the pass API (`reserve_passes`, `begin_pass`, `run_pass_stage`, `pass_logits`, `end_pass`, `abort_pass`) over two, three and four CPU stages of a four-layer model, tied and untied, at 2 to 8 pass slots: five requests, one a fork of another's first block, their prompts sliced at random and their generated tokens given, are formed into passes of random groups, advanced a stage and retired in random order, so passes wait between stages at once and take logits rows anywhere in the reserved range, and every row must be the bytes of its request run alone through `prefill` and `step` on one backend.
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
Local performance floors remain enabled by default. `--require-device-types Q8_0,Q4_K` makes a selected backend's refusal of any named weight type fail rather than skip through `common.device_lacks_kernel`; names are the case-sensitive storage names in `tests/spec_decode.py`.
Types not named keep their ordinary skips, and naming a type does not prove it was exercised, require a missing model, or change native CTest's driverless skips.
The `reference-consumer` component checks early and backend refusals, unrelated failures, required and optional types, runner exit status and invalid selections.
See `docs/CI.md` for workflow coverage and reproduction commands.

- **Dead code** (`tests/dead_code.py`, component `dead-code`): what no product path reaches, from the source, standard library only; the product is `llmx` and the tools, so code only tests reach is a finding too.
  Its findings and the list they are held to are described in Dead code and stale docs, below.
  `unused` is a C++ declaration in `src/`, `tests/` or `tools/` whose name no other token names, and `test-only` one in `src/` that only `tests/` names; the fields of a struct a `static_assert` pins and the values of a named enum cast from an integer are left out, a cast to a template parameter of the same name not counting, and a dead overload, or a name shared with a live one, is not seen.
  `override` is an override in `src/` that product code never calls, through an object or from a member of its class or a base.
  `shader` holds the Vulkan kernels' chain: every `.comp` compiled by a CMake entry, every `.glsl` included, every module embedded and in the kernel table, every kernel id named at a dispatch or derived by `kv_variant`, every variant's define tested by its source, every tested define set, and every `#ifndef` default overridden by some entry.
  `flag` is a flag `src/cli/main.cpp` parses that stores nothing, or stores into a field or local nothing uses: the help printing its default, a refusal's condition and a copy into a field of the same name are no use, and a field of a struct `main.cpp` declares is looked for in `main.cpp` until a copy passes it on.
  `python` is a function, class, method or module constant in `tests/` or `tools/` that nothing reaches from a module's top level, an argparse option that neither its module nor a module its parsed namespace goes to reads, or a test module that no suite import, workflow step or command line in a live section of a doc runs.
  `file` is a `src/` file no translation unit CMake builds includes, a test or tool source neither CMake nor the linked check builds, or a header nothing includes; `macro` is a macro defined and never used, unless a system header after it reads it, or an `LLMX_` macro tested but set nowhere, in the code, the build files or a doc's `-D`.
  Planted faults show each check catching what it is for: a dead function, one after a function with a braced default argument, a function only a test calls, an unused value of an enum a template parameter shares a name with, a macro never used, an override nothing calls, one declared over several lines and one with a braced default argument, an unbuilt shader, a flag whose value only the help prints, a flag parsed and dropped, an unreached Python function, an option only another module's namespace reads, an unbuilt test source a record names, and a listed finding that no longer occurs.
  `python -X utf8 tests/dead_code.py --linked DIR`, which the Vulkan job runs, builds every target with the Vulkan backend into `DIR` at -O0 with every inline and static function emitted, links each executable with `--gc-sections` and reports a `src/` function no executable keeps (`linked-unreached`) or only tests keep (`linked-test-only`), which tells the overloads and shared names apart that the name check cannot.
  It needs Linux, GCC, GNU ld, binutils, CMake, the Vulkan headers and glslc, and leaves out lambdas, virtual functions, templates never instantiated, special members nobody wrote or that are defaulted, constexpr functions, code a Linux build does not compile, and a function whose name only such code calls.
- **Docs** (`tests/docs_check.py`, component `docs`): the Markdown, and the code's comments, against the tree and the binary's help pages, standard library only.
  In every section a relative link and its anchor resolve (`link`), a line number names its commit (`line-pin`), and a heading or a ROADMAP item a reference names is there (`section`): `(AGENTS.md, Tests)`, with the file backticked or not, or `(Tests, above)` for a heading or bold label of the same doc.
  In live sections a repository path, glob or `path:symbol` exists (`path`); a qualified name, a call with arguments and each name among the arguments are the code's, a name only a string holds not counting (`name`); and a command line of `llmx`, of a built tool or test, or of a script in `tests/` or `tools/` uses flags that program takes, a script without a parser taking those of the module beside it that it hands its arguments to, and a `-DLLMX_` option exists (`command`).
  The comments and docstrings of the code, the build files and the workflow are held to the same: a `docs/` path they name exists (`path`) and a heading they name is there (`section`).
  `usage`: each flag a command's help page lists appears in its section of `docs/USAGE.md`, a synopsis without `[flags...]` lists them all, synopses and table rows name only flags the command takes, a numeric default in a table equals the help's, every command has a section, and the load tool's table lists exactly its options, with its defaults.
  `test-name`: a name a live doc lists after CTest is a CTest and one beside component a suite component, a line of this list that opens with a test's name and what it does names one, another test-like name this list or `docs/CI.md` gives is named somewhere in the tree, and every CTest and suite component is described in this list.
  `src-page`: every `src/` file is named by exactly one `docs/src` page title, and every title names a path that exists.
  Planted faults show each check catching what it is for: a dangling path, a broken link, an unpinned line number, a command line with a flag its program does not take, a script given a flag of a module it imports without handing it its arguments, a heading reference without backticks in a record, a missing heading of the same doc, a comment naming a doc that does not exist, a qualified name whose member only a string holds, a wrong flag and a wrong default in `docs/USAGE.md`, a page for a file that does not exist, two CTests removed while this list describes them, and a listed finding that no longer occurs; a correct pinned line reference must give no finding.
- **Arch boundary** (`tests/arch_boundary.py`, component `arch-boundary`): the runtime names no architecture. It reads the registered names from `src/model/arch/registry.hpp` and fails if one of them stands as a word (in any case outside a string literal), or a string literal begins with `blk.` or ends in `.weight` or `.bias`, in `src/model/*.hpp`, `src/inference/*.hpp`, `src/server/*.hpp`, `src/cli/*.cpp` or a header under `src/model/arch/` that the registry does not include, which the modules share. The tokenizer is left out, since its pretokenizer names match architecture names.
  A self-test first holds it to a planted text and to one name for each row of the registry's table, so a check that finds nothing because it reads nothing fails; it runs no binary.
- **Version** (`tests/version.py`): `--version` matches the CMake project
  version and build identifier format, and the usage banner starts with it.
- **CLI** (`tests/cli.py`): the command-line surface the numerical components do not reach.
  A Vulkan device is refused with an error and nothing on stdout, never run on the CPU instead, through a model command and through the synthetic bench.
  A build without the Vulkan backend refuses it, and so does a Vulkan build that cannot open it; where device 0 opens, an index no machine has stands in for the missing device.
  `info` on the synthetic MoE model names its architecture and layer count, and lists every tensor written with its type, shape and size.
  `--n-cpu-moe` and `--cpu-moe`, alone and beside `--moe-stream-from`, are refused on the synthetic dense model with status 1 and the name of the flag given by `generate`, `chat`, `logits`, `perplexity`, `bench --model` and `serve`, on the CPU and on the configured device when that is one device other than the CPU.
  It also checks the CLI's usage errors, all refused before a model is opened: an unknown command, `serve` and `pull` without arguments, missing and extra arguments, unknown flags and flags without a value, a chat positional argument, a second prompt, a value after the `--ignore-eos` switch (checked by its reason, which a parser without the switch would not give), the synthetic bench's flags with `--model` and the model run's without it, `--profile` off a single Vulkan device, `--moe-stream-from` without experts on the CPU, an unknown cache type or load mode, and numbers out of their form or range.
  So are, each checked by its reason so that a usage error for another cause does not pass for it: anything after `--version`, `-tb` with `perplexity --per-token`, `perplexity -c 1`, an empty value that would read as the flag not given (`pull --file`, `pull --cache-dir`, `--stop`, `--then-ids`, `--layer-shares`, `bench --model`), a second value for a flag in one spelling or in two, and `--cpu-moe` with `--n-cpu-moe`.
  Each exits with status 2, nothing on stdout and the command's page, or the overview for `--help` or `--version` followed by anything, then the reason on stderr.
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
  `dequantize` must match, bit for bit, the spec decoders of `tests/spec_decode.py`: Q8_0 and Q4_0 on the blocks quantize writes, and Q4_1, Q4_K, Q5_K, Q6_K, Q8_0 and Q4_0 on raw blocks that reach every scale, min, high bit and nibble, Q8_0's and Q4_0's under negative scales, which their quantizers never write, so the references of `q8-dots` and `backend-group`, which take them from llmx's decoders, rest on an independent decode for all six types.
  Q4_0's raw blocks take each of the test's scales twice, negatives included, so every position takes every nibble, and under a negative scale nibble 8 must decode as -0, as the format's d*(nibble - 8) gives it (`docs/ASSETS.md`).
  Both types quantize and dequantize under a non-ASCII directory to the same bytes as under an ASCII one, and a type name quantize does not write is refused with exit status 2.
- **Raw blocks** (`tests/raw_blocks.py`): the spec decoders of `tests/spec_decode.py`, one for each GGUF type the pinned fixtures hold (F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS and MXFP4), each in a pure form, the readable reference, and a numpy form for whole files.
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
- **Perf** (`tests/perf.py`): time matmul / RMSNorm / RoPE hot paths and print
  throughput, so perf-first changes can be checked for regressions. Assert a
  generous floor so catastrophic slowdowns fail loudly without being flaky.
- **Tokenizer** (`tests/tokenizer.py`): encode/decode round-trips incl. unicode and special tokens, and refusal of a file naming another tokenizer or pretokenizer, with an error naming the key and the implemented values.
  The qwen35 pretokenizer must give HF's ids for the 37 texts of `tests/data/baseline_tokenizer_qwen35.json` (the Qwen3 fixture's 20, Thai, Devanagari, CJK punctuation and every added token), read through a file the test writes from the part of the pinned HF vocabulary those texts reach, so it needs no model.
  The 7 control tokens only the GGUF files and `tokenizer_config.json` add must each give their one id, alone and side by side.
  `python tools/gen_baseline.py tokenizer-qwen35` regenerates the file, keeping every merge that forms a substring of a text, so the ids equal the whole vocabulary's however llmx cuts the text.
- **Perplexity** (`tests/perplexity.py`): a synthetic model with an analytic
  scoring oracle checks window boundaries, chunk limits, target counts, file
  and inline input parity and invalid flags.
  Under `--verbose` each window's line must give its scored targets and the oracle's mean NLL for it, in order, with the totals unchanged.
  On the same model `logits` gives the same output for inline text and `--file` or `-f`, appends `--then-ids` ids separated by commas or whitespace, and refuses any other separator and an id past the vocabulary, 2^32 plus a valid id included.
- **Chat** (`tests/chat.py`): follow-up replies against independent HF goldens, including changed prefixes, stop/EOS and token-limit endings.
  A template the renderer refuses stops `chat` and `serve` before either takes a turn or listens, while `generate` still runs on the file.
  CTest also runs `chat-template` on `tests/data/baseline_chat_template.json`: the pinned real Qwen templates (Qwen2.5, the Qwen3 variants, and every Qwen 3.5, 3.6 and 3.8 template found in GGUF files and the official repositories), each held to its SHA-256, over 34 conversations each, tools, tool calls and content given as parts among them, with whether a conversation keeps an assistant turn split under each and a two-turn conversation of seven replies kept that way, and small feature templates, every case byte for byte against transformers' own chat template renderer, a failure where it fails with its message; templates the renderer must refuse; templates past its nesting and value limits, a `map` filter naming `map` 5000 times among them, which must be refused or fail without ending the process; the texts `chat::assistant_turn` must split as the Qwen templates split them; and a conversation ending in an assistant turn under the Qwen3 template of the official repositories, held in the test's source, whose reasoning the old renderer dropped.
  Regenerate both fixtures with `python tools/gen_chat_baseline.py` in the reference environment of `docs/ASSETS.md`; running them needs no external libraries.
  The same tool's `--extract` and `--scan` check every template on a machine by hand, through the same test binary.
  `tests/chat.py` also checks that a device that cannot be made is refused before the model file is read.
- **Thread controls** (`tests/threads.py`): actual auto/explicit phase counts,
  restoration after prefill, follow-up chat and HF-golden replies.
  The automatic count must be the one the test reads itself, apart from llmx, from the hardware threads, the process's affinity and its CPU quotas (the cgroup's on Linux, the job object's CPU rate hard cap on Windows), so a run in a CPU-limited container checks the quota.
  Perplexity also checks batched/per-token counts, both batch-thread aliases
  and automatic/zero selection against an independent HF NLL fixture.
- **Loading and streaming** (CTest `load-progress`, `generation-stream`, `cli-output`):
  read-in byte reporting, file spans, files truncated before loading or between reading and mapping, callback failures, the loader's host copy and logits in each load mode, over splits and shards, the planned reads and the stream's reads, copies and failures, early text delivery, split UTF-8 bytes and stop/EOS accounting, with `ignore_eos` a reply running past the masked EOS to its limit or to a stop text;
  `cli-output` also reads `--device` lists as the commands do (canonical spellings, a device once, malformed entries refused), and the cache types and load modes as `exec_flag` reads them (one spelling each, an empty or unknown name refused before any model file is read), as it reads `--layer-shares`, whose empty list it refuses.
  It checks that `pull`'s page prints `--cache-dir`'s default from `hub::cache_in_home`, that `exec_flag` reads `--threads-batch` and `-tb` only where the command asks for them, and that `GivenFlags` refuses a second value for a flag, in one spelling or two, and `--cpu-moe` with `--n-cpu-moe`, takes a switch given again, and keeps the spelling a line gave, and that `long_spelling` gives each short form's long spelling.
  It runs the CLI's number readers (`int_arg`, `float_arg` with the ranges of `infer::Sampling`, `--seed`'s decimal 64-bit read) and `token_ids` over every malformed form: a missing value, a sign, space, base prefix, fraction or trailing character, infinity and NaN, a value past its range or its type, and an id that would narrow into the vocabulary.
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
  A history recomputed by class, as a paused request's resume recomputes it, must give the logits its decode gave, bit for bit: on the synthetic Q8_0 model, a 40-token prompt at its extent in slices of 16 and its 199 greedy tokens as entries of extent 1 of up to 64 rows, logits only on the last, against the prompt and 199 single decode steps; the same from a fork at the first block, and beside another sequence's decode row and a third's prompt slice.
- **Server** (`tests/server.py`): `llmx serve` on a system-chosen port against the CLI on the same file, the synthetic F32 model without a download and the Q8_0 fixture when present: greedy through `/v1/generate` equals `generate --temp 0` alone and four at a time, a stream carries the same ids, a seeded request repeats, refusals, a client leaving mid-stream leaves nothing active, a chat turn and a follow-up whose assistant turn comes with its reasoning inline, in `reasoning_content` or null, each rendering as `chat` renders its own reply and counted alike by `/v1/tokenize`, kept whole by both under a template without reasoning, and two `chat` turns under a Qwen 3.8 template of the chat fixture as long as the reference's renders with the reply split, the compatible `/v1/completions` and `/v1/chat/completions` whole and streamed in the OpenAI clients' shape, a prompt repeating a finished request's tokens reuses its blocks with the CLI's greedy text, and the limits: a KV budget below the context bounds a request and a full queue answers 503.
  A sampling field outside the range the CLI's flag takes is refused with 400 on every route, `repetition_penalty` on the compatible routes included, and a `top_k` of -1 is refused on the native route and sampled as `top_k` 0 on the compatible one.
  `/v1/tokenize` gives the ids `llmx tokenize` prints, with `add_special` absent, false, true and 1, which it does not read, for text beyond ASCII, a special token's text and an empty text (with the Q8_0 fixture, the tokenizer golden's texts), and `/v1/detokenize` gives those ids back as the text; ids that end or start inside a character, alone, together and reversed, give the bytes `llmx detokenize` prints with the U+FFFD repair.
  A generating route reads as many tokens as `/v1/tokenize` counts for its prompt, `tools/server_load.py` finds the route and counts each text as the CLI does, and a whole reply's ids detokenize to its text; with the Q8_0 fixture, `messages` give the CLI's ids for the chat fixture's renders of the cases carried over from the Jinja2 goldens under that file's template, and a chat request with them reads that many tokens and replies as that text does through `/v1/generate`; while `--max-seqs 1 --max-queue 1` holds one request and queues another, both routes answer, and a text of more tokens than `--ctx-size 512` is counted.
  Both routes refuse a body that is not a JSON object, a missing or mistyped field, both `text` and `messages`, an id that is not a whole number or lies outside the vocabulary, whose edge the CLI confirms (2^32 past a valid id included), and a body past the size limit, in the native error shape.
  On a synthetic model whose vocabulary lacks the byte token `q`, `/v1/tokenize` refuses a text holding it, as a text and as messages, with the 400 and the message `/v1/generate` and `/v1/chat` give.
  With the Q8_0 fixture, seeded requests on each of the sampler's four paths, four at once with the other settings at their defaults, give the text `generate` gives alone with the same settings and seed, repaired as the server writes a character a reply ends partway through: the defaults, `top_k` 40 with `top_p` 1, `top_k` 0 with `top_p` 0.95, and `top_k` 0 with `top_p` 1.
  The synthetic model's file name holds a byte that is not UTF-8 on Linux, and elsewhere characters beyond ASCII whose UTF-8 bytes code pages 932, 936, 949, 950 and 1257 cannot map, so a name read in the system code page there fails; `/v1/health` and `/v1/models` must name it as UTF-8, with a U+FFFD for each byte that belongs to no UTF-8 character.
  Log-probabilities on `/v1/generate` and `/v1/completions`, and with the Q8_0 fixture on `/v1/chat` and `/v1/chat/completions`, whole and streamed: a request asking for none by name gets the bytes of one that never names them, and one asking gets that reply's ids and text with its values in the route's shape, the same bytes on a second run, the stream carrying the whole reply's values, greedy's token first among the five listed and every list most likely first; the completions and chat routes give `/v1/generate`'s values for the same prompt, each of four prompts gets its values alone while the others run beside it, 20 tokens are listed when asked, and 21, a `top_logprobs` without `logprobs` and a `logprobs` of the wrong type are refused with 400; a seeded draw at temperature 1.5 with penalty 1.3 gives the same values on `/v1/generate` with one token listed and with none, which then carries no `top_logprobs`, and on `/v1/completions` with `logprobs` 1 and 0, whose maps hold the sampled token's text at every position, listed or not.
  With the Q8_0 fixture, uncapped requests share a pool too small for all of them: a request is paused when it runs out and resumes from its history, and the check confirms the server counts a pause and each request runs to its own end with the text and every log-probability it gives alone, `/v1/health` then showing nothing paused and resumes that took their donors back or recomputed what they lacked; a long prompt read one token a pass is paused while it is still prefilling and, resumed, gives the CLI's greedy text and the values it gives alone on a server of its own; a conversation of six turns on a 1024-token pool, its history growing past half the pool, reuses the last turn's history on every follow-up (its `reused_tokens` and the server's `prefix_tokens` grow each time) with each turn's greedy text equal to the CLI's; and a follow-up that fits the pool only once one donor goes, beside an unrelated donor, consumes the turn it repeats, so a later prompt repeating the unrelated request's history still reuses it with the CLI's greedy text.
  With the Q8_0 fixture too, a client that leaves a whole reply while it is generated, a streamed prompt while it is read one token a pass, or a request waiting for the one slot is noticed within seconds, though nothing written to it fails, and the server then holds nothing active or queued and starts a request reaching the whole pool at once; a client that shuts only its sending side during a whole reply gets no answer, the connection just closing; and each request left behind would run for thousands of passes, so a server that noticed nothing fails on any device.
  The synthetic MoE model's prompts must each give the same ids alone and four at a time, where a pass routes one request's prompt rows beside another's decode rows.
  With `ignore_eos` a greedy reply that ends early at the model's end token runs exactly to its limit, and the CLI's `--ignore-eos` gives the same ids, greedy and seeded: on the synthetic model written with an end token from its own greedy reply, through `/v1/generate` whole and streamed and `/v1/completions`, uncapped to the 16-token context, which `generate --ignore-eos` asked for the room its prompt leaves also fills with the server's ids, and on the Q8_0 fixture through `generate` and `chat` and all four routes, a stop text reached past the masked token still ending the reply; requests with and without it give their own ids alone and at once, and a value other than `true` or `false` is refused with 400 on every route.
  On the CPU it runs as it is, so every job checks it; on a single device other than the CPU it runs with its experts on the host and prompts from three tokens streamed, where a pass holds streamed prompt rows beside host decode rows.
  There too, with prompts from 100 tokens streamed, a 120-token prompt that forks the first 64-token block of a finished prompt's history must give the ids and values it gives on a server of its own, since the path follows the whole prompt and not the 56 tokens the fork leaves to read.
  A device list skips it, and the whole component skips under `--cache-type f16`.
  Throughput is measured separately with `tools/server_load.py`.
- **Serving load** (`tools/server_load.py`, suite component `server-load`): a running server's figures under closed-loop levels of concurrent users and open-loop Poisson request rates, through `/v1/generate`, `/v1/completions` or the reference server's `/completion`.
  Prompts are counted through the server's tokenize route where it has one, `/tokenize` or `/v1/tokenize`, and by two probe requests otherwise.
  Prompts of an exact token length come from a seeded word list, their lengths from the seed alone, replies of a fixed length ask to ignore the end of text, and each level reports time to first token, time per output token, inter-token and end-to-end latency percentiles, output and total tokens per second, and failures by reason; `--json` keeps every request's record, and the tool exits 3 when any timed request failed.
  The suite runs its `--self-test`, which needs no model: the stream reading of every API and the figures on made-up arrival times exactly, then every API in both loads against an in-process server, `ignore_eos` sent by a workload read with `--output-len` and not by one read with `--tokens`, with a refusal, a dropped stream, both timeouts and a short reply, checked on counts, reasons and time bounds a busy machine cannot break, then the whole tool's table, notes and exit status.
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
  `--uncapped` runs other phases on a pool too small for its requests: 12 uncapped greedy requests through `/v1/completions`, streamed with `logprobs` 5, with `--max-seqs 6 --ctx-size 4096` unless given, each alone and then all at once, where requests are paused and resumed; each must give its tokens and every value alone, and it reports the pauses, the tokens resumes recomputed, the wall time together and its inter-token p50 and p99.
- **Long context** (`tools/long_context_check.py`): one 16k-token
  summarization prompt from `tests/data/wiki.test.raw`, greedy, 512
  generated tokens by default, sent to `llmx serve` on the device under
  test twice from fresh servers, which must give the same tokens; then the
  CPU reads the prompt and those tokens (`llmx logits --last`), and at
  every generated position the device's token must be the CPU's top choice
  or within `--margin` (0.5) logits of it. Nothing else here reaches a
  prompt that fills thousands of KV blocks and then decodes from that
  history. A hash across two backends is not the check: their activations
  round differently, and after a long prompt greedy decoding meets
  near-ties where either token is right, so identical text is only
  required of one backend against itself. It needs a real model and is run
  by hand, not by `run_tests.py`.
- **Decode probe** (`tools/decode_probe.cpp`, target `llmx-decode-probe`, built beside `llmx-split-check`): a reply's decode path on a real model, the prompt read as one prefill and each forced id of a fixture fed as a decode step, as a request alone runs through the server.
  Each step prints its greedy token, the runner-up and the forced id with their logits, and the tool exits 1 where a forced id is not its step's greedy token; after the last forced id it prints the step's five best, ranked by `infer::top_logprobs`, or every id of a smaller vocabulary, and the logits of the fixture's two tokens.
  `tests/data/decode_probe_30b_a3b.json` holds a prompt, the 55 ids Qwen3-30B-A3B Q8_0's greedy replies share on an MI50 in the Q8_0 decode kernel's quarter layout and in its half-block order, and the two tokens where they part, with each order's gap between them (docs/STATUS.md, the half-block order).
  It needs that model and is run by hand; the suite's `decode-probe` component (`tests/decode_probe.py`) runs the tool found beside `--exe` on the tiny F32 model and on a model of four tokens, on the configured device where that is one Vulkan device and on the CPU otherwise.
  The greedy path three steps long, which it reads from the tool's own lists, must exit 0 and a step off it exit 1; each step's best list must hold five distinct ids inside the vocabulary, largest logit first, and four on the model of four tokens.
  A fixture whose id or token is not a whole number, is negative, past 2^32 - 1 or past the vocabulary, or is a string or a boolean, must be refused with exit status 2, nothing on stdout, and the entry named, as `ids[1]`.
  It skips when the tool is not there, unless `--require-tools` is given.
- **Layer split** (`tools/split_check.cpp`, target `llmx-split-check`, built in every configuration with tests, which is the default, and linked to the Vulkan backend when that is on): a model on one device against the same model split in equal shares over a comma-separated list of devices of the same kind (default `0,1`; a device is `cpu` or a Vulkan index), with optional decode steps, ubatch and cache type (`f16`, the default, or `f32`, both sides of both models), as raw float logits compared with `memcmp`: every position of a scored text through the prompt path, the prefill in chunks of the ubatch, which a split pipelines over its stages, and greedy decode steps, then three passes of a decoding sequence beside a fresh prompt, every row's logits.
  On a split that takes passes in flight, with a text of 12 tokens at least, it then runs them through the pass API at P = S, S + 1 and 2S: 2P sequences, each a prompt cut from the text in chunks of up to 32 tokens and then 8 tokens of the text as generated ones, four entries to a pass, each stage recorded after a random host delay of up to 3 ms, and every logits row must equal the same passes run one after another through `forward` on the split and on the single device.
  After the decode steps, the prompt and the steps are recomputed by class on each model, as a paused request's resume recomputes them: the prompt at its extent in ubatch slices and the steps as entries of extent 1 of up to 64 rows, whole and from a fork at the last whole block when the history passes one, which must give the decode's last logits; it prints how many of these ran from a fork.
  The split must be bit-identical, since each layer runs the same kernels on the same rows wherever it sits; a split over different backends is held to the HF bounds instead.
  Each listed device is a backend of its own, without the CLI's listed-once rule, so `cpu,cpu` is two CPU backends.
  The `split` component (`tests/split.py`) runs the tool found beside `--exe` on the tiny F32 model, tied and untied, and the tiny MoE model, one CPU against `cpu,cpu` and the MoE also against `cpu,cpu,cpu`, at ubatch 1, 3 and 16 and with f16 and f32 caches, with 3 decode steps after a 13-token text, which fills their 16-token context.
  It also writes a synthetic Q8_0 model with a 256-token context, whose decode rows take the CPU's 8-bit dots and prompt rows the float path, and runs it against `cpu,cpu` after a 100-token text with 40 steps and a 150-token text with 8, histories that pass a 128-token block, so every run recomputes from a fork on one backend and on the split, which the component requires.
  It skips when the tool is not there, unless `--require-tools` is given, and the configured device, shares and cache type do not reach it.
  The HF job runs it on the Q8_0 fixture over the perplexity excerpt, `cpu` against `cpu,cpu` with 8 steps and 64-token chunks; other real models and splits over devices are run by hand.
- **F32** (`tests/f32.py`): deterministic small-model weights with full logits
  and windowed NLL generated independently by HF. Covers tied/untied weights,
  odd dimensions, batch tails and threads without downloading a model,
  and `bench --model` at a depth and with `--seqs 2`, two sequences where the model's context fills one cache block.
  `logits --file` must print what the same prompt inline does.
  The `--last` rows of the prompt, in one pass and in several, and of its first three tokens continued by `--then-ids`, are held to the HF bound at their positions.
  Each row is printed once, and the rows over several passes and after `--then-ids` are the bytes of the one-pass rows at the same positions.
  The same rows read with `--per-token`, every token through a decode step, are held to the HF bound too, the prompt's and its head's continued by `--then-ids` the same bytes, and without `--last` it prints the last row's list.
- **MoE** (`tests/moe.py`): the same for a tiny `qwen3moe` model against HF `Qwen3MoeForCausalLM` (`tools/gen_baseline.py moe`), two routed layers and one dense, across batch widths and threads and, on a device, with the experts of one or every routed layer on the CPU (`--n-cpu-moe`, `--cpu-moe`).
  On a device it also streams those layers to the device for prompts from a length on (`--moe-stream-from`): from 0, from 1, which streams what 2 does since neither a generated token nor a one-token prompt streams, and from 4, which only the longer prompts reach.
  A second model is Q8_0 with every row width, the experts' included, a multiple of 64, so a device's Q8_0 decode kernel reads its rows in the order it keeps for an even block count; its goldens (`tests/data/baseline_moe_q8.json`, `tools/gen_baseline.py moe-q8`) are HF holding the file's own weights as `tests/spec_decode.py` decodes them, for three prompts and HF's greedy continuation of 16 ids, read batched, three rows a pass and with `logits --per-token` through decode steps.
  Its gated variant's router is kept at least 0.1 router logits from a top-k tie and its near-tie variant's within 1e-5 of one.
  On each path the gated variant's 51 x 257 logits must be within 0.127 of HF's, twice a pinned CPU calibration's largest error, each prompt's mean forced NLL within 0.01 of HF's, and its greedy id HF's at the 4, 5 and 5 rows HF's top two part by more than 0.254; a coarse fixture correctness bound, not losslessness (docs/STATUS.md, the half-block order).
  The near-tie variant's errors are printed and held to nothing.
- **Qwen 3.5** (`tests/qwen35.py`): the same for tiny `qwen35` models against HF `Qwen3_5ForCausalLM` (`tools/gen_baseline.py qwen35-tiny`), whose goldens come from HF's token-by-token cached forward, which runs the recurrence llmx runs per token.
  The fixtures are Hv = Hk with a tied head, Hv = 3 Hk with its own head, and that model with one MTP block, whose file must print the bytes the file without it prints.
  The writer makes the weights as HF holds them and applies the converter's transforms itself (docs/QWEN35.md, GGUF conventions).
  Beyond the F32 checks, the NLL is scored in passes of three tokens and one token at a time, and greedy decode after a prefill must give HF's greedy tokens.
  Until llmx runs the architecture it refuses the files, and the component reports SKIP, which `run_tests.py` counts as neither a pass nor a failure (`common.SKIPPED`).
- **Baseline** (`tests/baseline.py`): real-model EXTERNAL ground truth.
  Compares llmx against golden fixtures generated once from the HF reference by `tools/gen_baseline.py` and committed to `tests/data/`.
  Needs a real model, so it SKIPS when none is on disk; point it at one with `LLMX_BASELINE_GGUF`.
  Otherwise each check, the tokenizer's included, reads its model from the HF cache at the revision `tests/data/fixtures.json` pins (repo, revision, file, SHA-256 and size) for each entry marked `gate`, read into `BASELINE_MODELS`, the path `tools/fetch_test_models.py` downloads to.
  Their bounds sit in `tests/baseline.py`, which refuses to load unless each file is pinned once, each gate model has bounds and each bounded file is a gate model, and each gate model is marked `hosted`, since the hosted HF job downloads and requires every one.
  Every perplexity cell is scored twice, in batched passes (the default) and with `--per-token`, so the prompt and decode kernels both meet the reference.
  Its logit and PPL outputs go through the validators the 8B check uses, `common.check_logits` and `common.check_ppl`, at each fixture model's bounds: the exact prompt token count, ten unique in-vocabulary IDs with finite logits sorted from the top, and exactly the PPL fields with every count exact.
  Regenerating tokenizer fixtures needs `tokenizers` and `huggingface_hub`; numerical fixtures also need `torch` and `transformers`.
  RUNNING the suite needs none of these packages.
  `python -X utf8 tests/baseline.py --file-exact DIR --model FILE` holds one file to the goldens `tools/gen_baseline.py file-exact` made from it, HF run on that file's own weights as the numpy spec decoder reads them, at the Q8_0 fixture's bounds whatever the file's type (`FILE_EXACT_BOUNDS`); goldens made from another file are refused by SHA-256.
  A device with no kernel for the file fails a file-exact run, where the gate's own checks skip.
  Making those goldens also needs numpy.
  The other entries of `tests/data/fixtures.json` pin the models of tensor types llmx does not read yet, each with `gate` false until its type has bounds, and `hosted` marks the ones the hosted HF job is to download then; `tools/fetch_test_models.py --all` fetches them with the gate's models.
  Each entry names its family: `qwen3`, whose goldens are the ones above, or `qwen35`, whose goldens, vocabulary and context `tests/baseline_qwen35.py` holds, with the qwen35 files' bounds once they are set.
  For each hosted qwen35 file on disk it runs that check at 512-token windows: the family's tokenizer golden, the file's chat template, the ids of its model's chat renders, prompts and excerpt, then logits and perplexity, with one skip line while llmx refuses the architecture; a file without bounds is measured and fails.
- **Reference generator** (`tests/reference_generator.py`): standard-library checks for pinned reference selection, separate alternate-model output and forwarding the revision/float32/eager settings to the HF loaders.
  Actual reference generation and model correctness remain separate checks.
  For `file-exact` it checks the argument combinations it refuses, that `tests/baseline.py --file-exact` fails when the device has no kernel for the file, the HF parameters a few GGUF tensor names take under the one map `tests/f32.py` holds for the tiny models and file-exact alike, and a tiny GGUF's tensors reaching their parameters with reversed dimensions and unchanged values.
  Every qwen35 reference runs through one check of its environment (`qwen35_environment` in `tools/gen_baseline.py`), whose doubles must see a torch, transformers or tokenizers version other than the pinned ones refused, transformers pinned as the chat renderer pins it, and an installed `kernels`, `fla` or `causal_conv1d` package refused, since HF would run it in place of its torch functions.
  For `qwen35-tiny` its doubles hold the generator offline, to float32 and eager attention, and to its key checks: only `mtp.*` and `model.visual.*` keys unused, and none missing.
  The converter's tiled V-head order has one owner, `tests/qwen35.py`, for the tiny writer, file-exact and the layered reference alike, and the test maps HF's `dt_bias` of one Qwen3.5-4B layer onto that layer's `ssm_dt.bias` in a 4B GGUF bit for bit through it (`tests/data/qwen35_4b_dt_bias.json`), so a misreading of the order that the writer and the kernels share cannot pass the Hv = 3 Hk fixture.
  It also checks `tests/data/fixtures.json`: each file pinned once with every field, the gate's models those with bounds, and of the six pinned ahead of their types the hosted ones exactly UD-Q8_K_XL, IQ4_XS and Q2_K.
  The qwen35 tokenizer golden, on a made-up vocabulary, must keep a merge that joins across a cut HF makes and drop one no text reaches.
  It must give an added token the GGUF files' type, control for a special one or one written `<|name|>` and user-defined otherwise, and keep a token only the config adds apart.
  The generator must refuse a tokenizer file whose SHA-256 is not the pinned one, and the committed golden must hold the generator's texts, commit and digests, so neither changes without regenerating it.
  For `qwen35` it checks the selections it refuses, a tiny qwen35 file written as the converter writes one (norms as 1 + w, `ssm_a` as -exp(A_log), three V heads to each K head in the tiled order, and an MTP block) coming back as HF's tensors, and the loader against doubles: the versions, float32, eager attention, no `kernels`, `fla` or `causal_conv1d` package, the keys a load may leave out, the recurrence refused and the checkpoint files held to their digests.
  The committed qwen35 goldens must hold the generator's checkpoints, files, prompts, conversations, templates and excerpts, and the consumer must pin every one of them.
  For the layered qwen35 reference (`tools/gen_layered_reference.py`) it checks the arguments it refuses, that it runs offline in that environment check on its threads and refuses what the check refuses, the checkpoint keys it refuses (a stray key, a missing one, a converted one and two for one parameter), its count of float32 steps between two values, that it names an F32 tensor it had nothing to compare with, that the committed Qwen3.5-0.8B record shows every input's logits and every parameter equal to HF's full forward, and that the committed 9B and 27B goldens hold the generator's texts, windows and versions.
- **Reference consumer** (`tests/reference_consumer.py`): standard-library rejection tests for changed 8B fixtures, damaged logits/PPL, top-5 boundary swaps beyond those `common.top5_overlap` forgives, wrong model identity and failed launches, and a passing run over simulated outputs that must have 41 checks with each NLL case scored in both modes.
  For `tests/baseline_qwen35.py` it checks changed goldens, the digest that stands for the excerpt's ids, one skip when llmx refuses the architecture after 47 checks, a run of 59 checks failing without bounds and passing with them, a wrong file digest or chat template, and file-exact goldens made from another file.
  The suite's form must skip only a file not on disk or refused, and fail a file without bounds; `--require-baseline` must count a qwen35 gate model as it counts a Qwen3 one.
  For the layered qwen35 goldens (`tests/baseline_layered.py`) it requires each model's goldens to record its GGUF's SHA-256 and every F32 tensor of the GGUF compared and equal to the checkpoint's but for `ssm_a` values one float32 step off, which the model's provenance line counts, a passing run of 41 checks per model, one skip line while llmx refuses the architecture, and a failure for a model with no goldens, for another error and for a check failed before the refusal.
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
It prints one skip line while llmx refuses the architecture, and until a file has bounds it measures every check and fails, so the first measurement sets them.

`tests/baseline_layered.py` runs the same checks on the Qwen3.5-9B and Qwen3.6-27B Q4_K_M files against the layered HF reference's goldens, chosen by the file's SHA-256, and skips in one line while llmx refuses the qwen35 architecture (`docs/ASSETS.md`, The layered qwen35 reference).

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

## Architecture

See `docs/ARCHITECTURE.md` for the layer diagram and rules. The rule that
matters: **each layer depends only on the layers below it** -
`cli > server > inference > model > backends > tokenizer > format > quant > core`;
`server/` uses the inference layer and adds only scheduling and transport.

| Directory    | Contents                                        |
|--------------|-------------------------------------------------|
| `core/`      | fp16 <-> f32, JSON parser, UTF-8, common types  |
| `hub/`       | CLI acquisition path: Hub metadata, curl HTTPS and verified multi-stream cache |
| `quant/`     | type ids and block sizes, QuantType registry + Q8_0/Q4_0/Q4_1/Q4_K/Q5_K/Q6_K kernels |
| `format/`    | GGUF v3 reader/writer (headers, then mapping, then reading in), file spans, a file read at offsets, raw F32 tensors to and from GGUF |
| `tokenizer/` | byte-level BPE, Qwen2/Qwen3/Qwen3.5 pretokenizer |
| `model/`     | runtime (sequences, passes, stages, the arena, placement), one module per architecture under `arch/` chosen by the registry (qwen3 and qwen3moe), KV cache, layer split over devices |
| `backends/`  | Backend interface + cpu/ (AVX2) and vulkan/ impls; one worker pool; `device_profile.hpp`, the device numbers a GPU backend shapes its kernels by |
| `inference/` | model loading, sampler, log-probabilities, generate, perplexity, chat template renderer |
| `server/`    | multi-user server (`docs/SERVER.md`): HTTP layer, scheduler with prefix reuse, routes |
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
  Unrelated documentation corrections belong in a separate commit; a feature's
  own documentation ships with that feature.
- Open a new per-feature block in `docs/STATUS.md` (or update the existing one)
  **before** writing code: **Goal / Done / Left / Gotchas**. That block is what
  lets the next agent pick the feature back up with a "continue feature X"
  prompt, so keep it current.
- A feature lands with a test the hosted workflow runs, or its STATUS block names the hand check that covers it and says why no hosted runner can run it.
- A bug fix lands its failing test first: a commit that adds the test, failing on the unfixed code, then the fix that makes it pass.
- Don't invent new directions - follow the roadmap. When the feature ships,
  delete its block and mark the row `Done` in the STATUS table.

## Merge gates

A branch runs the checks of what it can break, once, when it is complete; between its commits only the builds, CTest and a quick Qwen3-0.6B identity on the CPU run.

- Docs only: the suite's `docs` and `dead-code` components and the hosted run.
- Tests or tools only: CTest and the suite components they touch, on the CPU, and the hosted run.
- CLI, server or other host logic: CTest, the CPU suite, Qwen3-0.6B byte identity against main, and the hosted run; host code specific to Windows also runs CTest and the suite on Windows.
- Model, kernel, device, loader or placement code: byte identity against main on the CPU and a device for the models the change reaches, the suite on a device, the Radeon VII check on Windows for device code, one timing round against main, and the hosted run.
  A change of numerics also records its error against the HF reference and the headroom to its bound.
- A rebase without a conflict in code reruns the builds, CTest and the hosted run; a fix after review reruns what it touches.

A branch lands when the hosted run is green at the head that lands and main has not moved since: main is fast-forwarded to it, and STATUS records the merge in a commit of its own.
Comparisons against the reference runtimes belong to the phase's final gate, not to each branch.

## Checkpoints

Update `docs/STATUS.md` and commit at each meaningful checkpoint - at minimum
when a feature, a milestone, or a discrete chunk of work is complete. Each
commit should leave `docs/STATUS.md` accurate: `Done`/`Left` reflect reality,
the build passes, and tests are green. A fresh agent should be able to read
STATUS.md and resume exactly where the last commit left off.

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
  on a runtime they share; today qwen3 and qwen3moe. A new one follows
  `docs/ADDING-AN-ARCHITECTURE.md`.
- **Split mode** is a runtime flag: a `--device` list splits by layers
  (`docs/MULTI-DEVICE.md`); tensor groups and node count are planned. See
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
