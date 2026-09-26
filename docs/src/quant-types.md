# `src/quant/types.hpp` - type ids and block sizes

The storage types llmx reads, in namespace `quant`, and nothing else: the
lowest header of the quant layer, which every layer above names types
through.

- Type ids, the numbers GGUF files and the backends' kernels name the types
  by: `GGML_TYPE_F32=0`, `GGML_TYPE_Q4_0=2`, `Q4_1=3`, `Q8_0=8`, `Q4_K=12`,
  `Q5_K=13`, `Q6_K=14`.
- The block of each quantized type, values per block and bytes per block
  (`*_BLOCK`, `*_TYPESIZE`): 32/18 (Q4_0), 32/20 (Q4_1), 32/34 (Q8_0),
  256/144 (Q4_K), 256/176 (Q5_K), 256/210 (Q6_K).
- A `static_assert` that `size_t` holds 64 bits: `row_bytes` sizes a file's
  64-bit dimensions in `size_t`, so a narrower one would cut them short
  before its overflow check, and llmx does not build there.

`quant::Registry` (`quant/quant.hpp`) pairs each id with its block and its
kernels, F32 as a block of one value in 4 bytes, and `quant::row_bytes`
sizes rows from it. The format layer sizes a file's tensors through
`row_bytes` too (`gguf::TensorInfo::data_size`), so a file's tensors and a
backend's rows are sized from the same numbers. The Vulkan kernels carry
their own copy of the ids and block sizes in `q.glsl`
([backends-vulkan](backends-vulkan.md)).

A new type takes its id and block here and in `q.glsl`, a registry entry
with its kernels in `quant.hpp` or `k_quants.hpp`, a kernel on each backend
that runs it, and its tests: `tests/roundtrip.py` holds the CPU's decode to a
decoder written from the format description, and `backend-vulkan` holds the
device's kernels to the CPU's. The format layer reads it with no change of
its own.
