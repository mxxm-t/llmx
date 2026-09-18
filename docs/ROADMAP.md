# llmx — Roadmap

> **Dev status:** `docs/STATUS.md` tracks what's done / in flight / planned per
> feature. This roadmap is the stable long-term plan; STATUS is the living
> scratchpad. When a feature ships, its STATUS block collapses to a `Done` row
> here — this file never carries dev noise.

Ordered roughly by dependency and value. Items marked **[design]** are
specified in `docs/ARCHITECTURE.md` but not yet implemented.

## 1. More quantization formats
The `quant::Registry` makes new types drop-in: one file + one registry entry.
- `Q4_0`, `Q4_1`, `Q6_K`, `Q5_K`, `IQ2/IQ3/IQ4` blocks
- K-quant support (shared second scale)
- Type-aware `data_size()` already reads from the registry rather than a switch
- Extend `make_test.py` / `verify_gguf.py` to round-trip each new type
- K-quants are what most GGUF on the HF Hub actually uses; see #9b

## 2. More model architectures
The `infer::Model` layer is Qwen3-specific today. Generalize to an architecture
registry keyed by `general.architecture`:
- Llama (GQA + RoPE, close to Qwen3)
- Mistral, Gemma (rotary/context differences), Phi
- Each arch = a forward-graph file under `model/`, selected at load from metadata

## 3. More formats
`format::ModelFormat` is the seam; GGUF is the first impl.
- safetensors, raw `.bin`+`.json`, ONNX export path
- A `format::open()` that sniffs magic and dispatches (GGUF check already stubbed)
- safetensors is HF-native and unlocks most of the Hub; see #9b

## 4. Backends **[design]**
GPU backends are the only compile-time concern (heavy SDKs); `config.hpp`
`LLMX_HAS_BACKEND_*` gates each. This splits into two phases — the device
execution model has to land before any vendor backend is worth writing.

### 4a. Device execution model (prerequisite, backend-agnostic)
Today's `Backend` takes raw host pointers and returns scalars synchronously, so
a device backend would re-upload weights and round-trip activations on every
call. Before any GPU work:
- **Device buffers**: allocate / upload / free handles on `Backend`, so weights
  are uploaded once at load and stay resident. Today `Model::matvec` passes a
  fresh host pointer per call (`model/arch_qwen.hpp`).
- **Resident activations**: the elementwise work in `Model::step` (SiLU, the two
  residual adds, per-head q/k norms) must run device-side, or every layer pays a
  host round trip. Either add ops to `Backend` or move the graph down a layer.
- **Attention in the backend**: `attend_head` is scalar CPU code in the model
  layer; on GPU it is a large share of decode time at depth.
- **Async**: a submit / sync concept. `dot_q8_0` returning `float` by value is a
  per-row kernel launch.
- **Type-generic matmul**: drop the `_q8_0` suffix and dispatch through
  `quant::Registry`, so Q4_0 and later types stop bypassing the backend.
- **Batched prefill**: `infer::prefill` feeds one token at a time, so prompt
  processing offers a GPU no matrix-matrix work. Batch-1 matvec caps the
  achievable win to decode.

The CPU backend stays correct and fast through this refactor: it is the A/B
reference for every GPU claim (see #8).

### 4b. Vendor backends
- **ROCm (HIP)**: first-class target, matches the MI50 (gfx906) rig.
  `LLMX_HAS_BACKEND_ROCM`
- **Vulkan**: portability backend, explicitly *not* first-class. One set of
  compute shaders covering NVIDIA, Intel, and AMD as a fallback. Lower peak
  throughput than HIP on the rig — the point is breadth, not speed. Developed
  and validated on the MI50 (gfx906 supports Vulkan), so it needs no new
  hardware; NVIDIA and Intel stay listed as *untested* until that hardware
  exists. `LLMX_HAS_BACKEND_VULKAN`
- **CUDA**: only if Vulkan proves insufficient on NVIDIA. Three vendor paths is
  more surface than this project should carry (`AGENTS.md` — no bloat)
- **oneAPI / SYCL**: dropped. Vulkan is the Intel answer
- Runtime device selection: `--device rocm:0`, `--device vulkan:0`
- Kernels are hand-written — no cuBLAS / rocBLAS / CLBlast. The SDK is the
  dependency exception `config.hpp` already carves out; vendor math libraries
  are not

## 5. Multi-device split **[design]**
Split a single model across several backends on one machine.
- **per-layer**: consecutive layers to different devices (pipeline)
- **per-tensor**: independent tensors (embeddings, norms) on different devices
- **per-row**: row-parallel matmul across devices
- BackendManager owns one Backend per device; model routes tensors by placement
- All are runtime parameters, not build options

## 6. Multi-node / cluster **[design]**
- `node_id` on each Backend, message layer for cross-node tensor exchange
- Per-layer pipeline across nodes; gradient/activation shipping
- Split mode + node count chosen at launch (env / config file), not compile time

## 7. Multi-user server **[design]**
- HTTP/WS server front-end sharing the model + KV cache (batching, KV reuse)
- Continuous batching, generation queues, `/generate` streaming
- `server/` directory is the planned home (not yet created — avoid empty stubs)

## 8. Correctness & perf gates
Two standards, both EXTERNAL. Neither may be replaced by a self-consistency
check: llmx passed a fully green suite while six correctness bugs were live,
because every test compared llmx against itself.
- **Correctness is the HF reference.** Golden fixtures are generated once with
  HF tooling (`tools/gen_baseline.py`) and committed; `tests/baseline.py`
  compares llmx against them with no torch at test time. A round-trip that a
  consistently wrong encoder also passes is not a correctness test.
- **Performance floor is mx-llama.cpp.** llmx must be at least as fast as the
  gfx906 fork on the same model, quant, prompt and hardware, measured as pp and
  tg tok/s. A ground-up runtime slower than the thing it replaces has no claim
  to being a runtime. Report both arms; never a single number.
- Path-controlled perplexity on real text as the lossless gate (see
  `gfx906-correctness-gate` skill)
- Large-context output hashing to prove KV cache + RoPE correctness at depth
- Every GPU kernel claim gated by a CPU-vs-GPU A/B on identical inputs; the CPU
  backend is the reference implementation (see #4a)
- Micro-benchmarks per backend/quant, stored for regression comparison

## 9. Hugging Face integration
The Hub is where the models are, so "runs what the Hub hosts" is the practical
bar. Two halves with different dependencies: the download + format-coverage half
is gated on nothing and can start immediately; only the kernel half is gated (on
#4a). The position in this list is by dependency, not by priority.

### 9a. `llmx pull` — model download
- Hub REST: `GET /api/models/{repo}` for the file list and metadata,
  `GET /{repo}/resolve/{rev}/{file}` for bytes
- Pin a commit SHA rather than `main`, so a pull is reproducible
- `HF_TOKEN` via `Authorization: Bearer` for gated repos
- A local cache with a documented layout; `llmx pull <repo>:<quant>` resolves a
  quant variant to a concrete file
- **Sharded GGUF**: the Hub splits large models into `-0000N-of-0000M.gguf`.
  `gguf::read_gguf` assumes a single file — a real gap, not a detail
- **TLS is the dependency problem.** There is no HTTPS in the C++ stdlib, and
  this is the second carve-out from "dependency-free" after GPU SDKs. Shelling
  out to `curl` (present on Win10+, Linux and macOS) adds zero link-time deps
  and is the default choice; a stdlib-only `tools/hf_pull.py` is the fallback.
  Do not link OpenSSL, and do not implement TLS.

### 9b. Reading what the Hub actually hosts
`docs/ASSETS.md` records 5 of 6 local models as unusable — that is this
problem in miniature. Downloading a model is worth little if llmx cannot read
it.
- **K-quants** (`Q4_K_M` and friends): the dominant GGUF quant on the Hub — #1
- **safetensors**: HF-native, and trivial to read — u64 header length + JSON
  header + raw tensor bytes. No new dependency; `core/json.hpp` already parses
  the header — #3
- **BF16 / F16 tensors**: most HF safetensors are BF16. `core/fp16.hpp` covers
  f16 <-> f32, but there is no bf16 path and no F16 case in
  `gguf::TensorInfo::data_size()`
- **`tokenizer.json`**: the HF tokenizer format. `bpe::Tokenizer` reads only
  GGUF-embedded `tokenizer.ggml.*`, so safetensors repos have no tokenizer path
- **`config.json`**: architecture config. `infer::load_config` reads only
  `qwen3.*` GGUF metadata keys

### 9c. Hub kernels (additional, not a primary target) **[design]**
Gated on #4a: a kernel registry with no device execution model behind it is
scaffolding with nothing to plug into.

`kernels-community` artifacts are **not** consumable by llmx. Hub kernels are
PyTorch extensions, not portable shared libraries: they include `<torch/all.h>`
and `<torch/library.h>`, register ops through `TORCH_LIBRARY_EXPAND` into
`torch.ops`, are loaded by the `kernels` Python package, and are built per
(torch version x CUDA version x C++ ABI x Python version). Consuming them would
mean libtorch plus a Python runtime. Their *source* is ordinary HIP/CUDA and can
be read and ported by hand — that is the only supported use of them here.

What would work instead, if this is ever picked up:
- llmx defines its own kernel repo layout on the Hub and ships kernel *source*
- `hiprtc` / `nvrtc` compile at first use — both already ship in the SDK that a
  GPU backend needs, so this adds no new dependency
- Cache the result content-addressed by (source hash, GPU arch, driver version)
- Vulkan skips runtime compilation entirely by shipping precompiled `.spv`
- Payoff: no build step for users, and kernels updatable without shipping a new
  llmx binary

## Non-goals (for now)
- Training / fine-tuning in-tree
- Dependencies — keep the "no external libs" property as long as practical
