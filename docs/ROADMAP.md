# llmx — Roadmap

Ordered roughly by dependency and value. Items marked **[design]** are
specified in `docs/ARCHITECTURE.md` but not yet implemented.

## 1. More quantization formats
The `quant::Registry` makes new types drop-in: one file + one registry entry.
- `Q4_0`, `Q4_1`, `Q6_K`, `Q5_K`, `IQ2/IQ3/IQ4` blocks
- K-quant support (shared second scale)
- Type-aware `data_size()` already reads from the registry rather than a switch
- Extend `make_test.py` / `verify_gguf.py` to round-trip each new type

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

## 4. Backends **[design]**
The `Backend` interface (matmul / RMSNorm / RoPE) is the compute seam.
- **ROCm (HIP)**: primary GPU target, matches the MI50 rig. `LLMX_HAS_BACKEND_ROCM`
- **CUDA**, **Vulkan**, **oneAPI** as secondary backends
- Runtime device selection: `--device 0`, `--device rocm:0`
- `config.hpp` `LLMX_HAS_BACKEND_*` already gates each

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
- Path-controlled perplexity on real text as the lossless gate (see
  `gfx906-correctness-gate` skill)
- Large-context output hashing to prove KV cache + RoPE correctness at depth
- Micro-benchmarks per backend/quant, stored for regression comparison

## Non-goals (for now)
- Training / fine-tuning in-tree
- Dependencies — keep the "no external libs" property as long as practical
