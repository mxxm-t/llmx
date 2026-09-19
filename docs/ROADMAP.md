# llmx - Roadmap

> **Dev status:** `docs/STATUS.md` tracks what's done / in flight / planned per
> feature. This roadmap is the stable long-term plan; STATUS is the living
> scratchpad. When a feature ships, its STATUS block collapses to a `Done` row
> in STATUS - this file never carries dev noise.

Ordered roughly by dependency and value. Items marked **[design]** are
specified in `docs/ARCHITECTURE.md` but not yet implemented.

## 1. More quantization formats
`quant::Registry` is the only integration point: a new type is a pair of block
kernels plus one registry entry, and nothing outside `quant/` and the GGUF type
constants has to change. Block kernels and the registry live in
`quant/quant.hpp`; shared K-quant kernels live in `quant/k_quants.hpp`.
- Done: `Q8_0`, `Q4_0`, `Q4_1`, plus `Q4_K` and `Q6_K` read-only. Real files
  are MIXED: Qwen3-0.6B-Q4_0 is 193 Q4_0 / 113 F32 / 3 Q4_1 / 1 Q6_K, and
  Qwen3-8B-Q4_K_M is 217 Q4_K / 37 Q6_K / 145 F32. A type on its own loads
  nothing; the set is what matters.
- K-quant kernels live in `quant/k_quants.hpp`, split out when `Q4_K` brought
  the shared 6-bit sub-scale decoder (`get_scale_min_k4`) that `Q5_K` reuses.
- `Q5_K` done (read-only). Qwen3-0.6B-Q5_K_M is 168 Q5_K / 29 Q6_K / 113 F32,
  so dense Qwen3 `Q4_K_M` and `Q5_K_M` mixtures can be read. Other mixtures
  may still require missing types such as Q3_K.
- A fused Q4_K row dot landed for decode: no dequantized value is materialised,
  because d*q - m factorises the dot into d*sum(q*x) - m*sum(x). 2.16 -> ~2.6
  tok/s. Q5_K and Q6_K still take the generic path and would benefit the same
  way; measure before assuming, and note prefill is a different question since
  the batched path already reuses the dequantized row.
- `IQ2/IQ3/IQ4` are deliberately NOT next. Every quant type multiplies the
  per-backend kernel work later (see #4b), and these are both rarer on the Hub
  and harder to implement. Hold them until a GPU backend exists and that cost
  is visible.
- K-quants are what most GGUF on the Hub actually uses; see #9b
- `TensorInfo::data_size()` still switches on type in `format/gguf.hpp` rather
  than reading the registry, because `quant/` includes `format/` and not the
  other way round. Adding a type means touching both.
- Extend `tests/roundtrip.py` to cover each new type that has a quantizer

## 2. More model architectures
The `infer::Model` layer is Qwen3-specific today. Generalize to an architecture
registry keyed by `general.architecture`:
- Llama (GQA + RoPE, close to Qwen3)
- Mistral, Gemma (rotary/context differences), Phi
- Each arch = a forward-graph file under `model/`, selected at load from metadata

## 3. More formats
`format::ModelFormat` has a GGUF adapter and a magic-sniffing `format::open()`
implemented in `gguf.hpp`. CLI/model code currently consumes `gguf::GGUFModel`
directly; a second format needs integration through this seam.
- safetensors, raw `.bin`+`.json`, ONNX export path
- Extend `format::open()` beyond its current GGUF magic check
- safetensors is HF-native and unlocks most of the Hub; see #9b

## 4. Backends **[design]**
GPU backends are the only compile-time concern (heavy SDKs); `config.hpp`
`LLMX_HAS_BACKEND_*` names are reserved for those gates; the existing options
do not build GPU code yet. The vendor targets are ROCm, CUDA, SYCL
(Intel) and Vulkan. This splits into two phases - the device
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
- **Attention in the backend (CPU implementation in progress)**: causal GQA
  now goes through `Backend::attention` for both decode and prefill. The CPU
  backend owns score scratch and vectorized computation; model code still
  owns the host KV cache. A GPU implementation needs device buffers below it.
- **Async**: a submit / sync concept. `dot_q8_0` returning `float` by value is a
  per-row kernel launch.
- **Type-generic matmul (done for supported quants)**: dispatch through
  `Backend::matmul` and `quant::Registry`; F32 matrices use direct rows.
- **Batched prefill (done on CPU)**: `Model::prefill` batches tokens with
  `--ubatch`. Device-resident execution still needs the refactor above.

The CPU backend stays correct and fast through this refactor: it is the A/B
reference for every GPU claim (see #8).

### 4b. Vendor backends
Four vendor targets. Each is opt-in at build time because its SDK is heavy, and
each is gated by its own `LLMX_HAS_BACKEND_*` in `config.hpp`.
- **ROCm (HIP)**: first-class target, matches the MI50 (gfx906) rig.
  `LLMX_HAS_BACKEND_ROCM`
- **CUDA**: NVIDIA. `LLMX_HAS_BACKEND_CUDA`
- **SYCL**: Intel, through oneAPI/DPC++ over Level Zero. This is what llama.cpp
  calls its SYCL backend. `LLMX_HAS_BACKEND_SYCL`
- **Vulkan**: the portability backend, explicitly *not* first-class. One set of
  compute shaders that runs anywhere, used as the fallback where no vendor
  backend is built or available. Lower peak throughput than a vendor path -
  the point is breadth, not speed. It can be developed and validated on the
  MI50 (gfx906 supports Vulkan), so it needs no new hardware.
  `LLMX_HAS_BACKEND_VULKAN`
- Hardware availability sets what can be *claimed*, not what can be written:
  only AMD gfx906 is testable here today, so any NVIDIA or Intel result stays
  marked **untested** until that hardware exists. Do not claim a backend works
  on hardware nobody has run it on.
- Runtime device selection: `--device rocm:0`, `--device cuda:0`,
  `--device sycl:0`, `--device vulkan:0`.
- Kernels are hand-written - no cuBLAS / rocBLAS / oneMKL / CLBlast. The SDK is
  the dependency exception `config.hpp` already carves out; vendor math
  libraries are not.

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
- Split mode + node count chosen at launch (CLI flags), not compile time

## 7. Multi-user server **[design]**
- Shared inference events: token delivery must support live CLI chat/generation
  and future server streaming. Loading progress should report completed work
  from the loader, leaving presentation to the CLI or server. Implement the
  current CLI consumers first; avoid server-specific stubs.

- HTTP/WS server front-end sharing read-only model weights, with independent
  sequence state and mutable KV histories (see ARCHITECTURE.md, KV state and
  concurrent execution). Prefix reuse shares immutable KV only, with explicit
  lifetime tracking; it must not share a user's mutable history.
- Continuous batching, generation queues, `/generate` streaming
- Separate execution scratch ownership and safe backend scheduling; internal
  worker parallelism does not make the current `Model` concurrently callable.
- `server/` directory is the planned home (not yet created - avoid empty stubs)

## 8. Correctness & perf gates
Two standards, both EXTERNAL. Neither may be replaced by a self-consistency
check: llmx passed a fully green suite while six correctness bugs were live,
because every test compared llmx against itself. The subsequent bug tally
and implemented HF coverage are recorded in STATUS.
- **Correctness is the HF reference.** Golden fixtures are generated once with
  HF tooling (`tools/gen_baseline.py`) and committed; `tests/baseline.py`
  compares llmx against them with no torch at test time. A round-trip that a
  consistently wrong encoder also passes is not a correctness test.
- **Performance floor is mx-llama.cpp.** llmx must be at least as fast as the
  gfx906 fork on the same model, quant, prompt and hardware, measured as pp and
  tg tok/s. A ground-up runtime slower than the thing it replaces has no claim
  to being a runtime. Report both arms; never a single number.
- Path-controlled perplexity on real text as the lossless gate (see
  `correctness-gate` skill)
- Large-context output hashing to prove KV cache + RoPE correctness at depth
- Every GPU kernel claim gated by a CPU-vs-GPU A/B on identical inputs; the CPU
  backend is the reference implementation (see #4a)
- Micro-benchmarks per backend/quant, stored for regression comparison

## 9. Hugging Face integration
The Hub is where the models are, so "runs what the Hub hosts" is the practical
bar. Two halves with different dependencies: the download + format-coverage half
is gated on nothing and can start immediately; only the kernel half is gated (on
#4a). The position in this list is by dependency, not by priority.

### 9a. `llmx pull` - model download
- Hub REST: `GET /api/models/{repo}` for the file list and metadata,
  `GET /{repo}/resolve/{rev}/{file}` for bytes
- Pin a commit SHA rather than `main`, so a pull is reproducible
- `HF_TOKEN` via `Authorization: Bearer` for gated repos
- A local cache with a documented layout; `llmx pull <repo>:<quant>` resolves a
  quant variant to a concrete file
- **Sharded GGUF**: the Hub splits large models into `-0000N-of-0000M.gguf`.
  `gguf::read_gguf` assumes a single file - a real gap, not a detail
- **TLS is the dependency problem.** There is no HTTPS in the C++ stdlib, and
  this is the second carve-out from "dependency-free" after GPU SDKs. Shelling
  out to `curl` (present on Win10+, Linux and macOS) adds zero link-time deps
  and is the default choice; a stdlib-only `tools/hf_pull.py` is the fallback.
  Do not link OpenSSL, and do not implement TLS.

### 9b. Reading what the Hub actually hosts
`docs/ASSETS.md` records local model coverage. Quant support alone does not
make a model usable: its architecture and tokenizer must also be implemented.
- **K-quants** (`Q4_K_M` and friends): the dominant GGUF quant on the Hub - #1
- **safetensors**: HF-native: u64 header length + JSON header + raw tensor
  bytes. Reuse `core/json.hpp` after fixing its validation/Unicode gaps; validate
  tensor extents and dtypes before exposing data. No new dependency; see #3.
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
be read and ported by hand - that is the only supported use of them here.

What would work instead, if this is ever picked up:
- llmx defines its own kernel repo layout on the Hub and ships kernel *source*
- `hiprtc` / `nvrtc` compile at first use - both already ship in the SDK that a
  GPU backend needs, so this adds no new dependency
- Cache the result content-addressed by (source hash, GPU arch, driver version)
- Vulkan skips runtime compilation entirely by shipping precompiled `.spv`
- Payoff: no build step for users, and kernels updatable without shipping a new
  llmx binary

## Non-goals (for now)
- Training / fine-tuning in-tree
- Dependencies - keep the "no external libs" property as long as practical
