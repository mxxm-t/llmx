# `src/quant/types.hpp` - storage metadata and row sizes

GGUF storage metadata in namespace `quant`, independent of which types a
backend can execute. This is the lowest header of the quant layer.

- Type ids, the numbers GGUF files and the backends' kernels name the types
  by: `GGML_TYPE_F32=0`, `F16=1`, `GGML_TYPE_Q4_0=2`, `Q4_1=3`, `Q8_0=8`, `Q4_K=12`,
  `Q5_K=13`, `Q6_K=14`, `BF16=30`, `MXFP4=39`.
- The block constants of each implemented quantized type, values per block and bytes per block
  (`*_BLOCK`, `*_TYPESIZE`): 32/18 (Q4_0), 32/20 (Q4_1), 32/34 (Q8_0),
  256/144 (Q4_K), 256/176 (Q5_K), 256/210 (Q6_K), 32/17 (MXFP4).
- `StorageType` holds a name, values per block and bytes per block.
  `storage_type(id)` returns that metadata for 35 active GGML layouts through
  ID 42, including F16, BF16, the K-quants, IQ and ternary formats, integer
  types, F64, NVFP4, Q1_0 and Q2_0. It returns null for an unknown ID or a
  removed layout: 4, 5, 31-33 and 36-38. A name and byte layout do not imply
  a decoder or backend support.
- `row_bytes(type, nin, rows = 1)` sizes rows from this metadata. It refuses
  an unknown or removed type, then a width that ends inside a block, then
  a byte count that would overflow. Zero rows take zero bytes, but still
  require a known type and whole-block width.
- A `static_assert` that `size_t` holds 64 bits: `row_bytes` sizes a file's
  64-bit dimensions in `size_t`, so a narrower one would cut them short
  before its overflow check, and llmx does not build there.

`quant::Registry` (`quant/quant.hpp`) contains only implemented types and
takes their metadata from this table. The format layer sizes a file's
tensors through `row_bytes` (`gguf::TensorInfo::data_size`), so it can open
metadata for a known layout without an inference kernel. Inference and
raw conversion still check their own support. File extents and backend
rows use the same sizes.

The Vulkan kernels carry their own declared IDs and block sizes in `q.glsl`
([backends-vulkan](backends-vulkan.md)); `gguf-validation` reads those
declarations and checks them against this table without changing shader
expressions. Its independent binary fixtures cover every active layout,
payload bounds, partial rows, removed IDs and overflow. `backend-errors`
holds registry metadata to the table and ensures metadata-only types do not
become executable.

A new executable type adds its decoder to the registry and kernels to the
backends that run it. `tests/roundtrip.py` holds block decoding to an independent
format decoder; `half-weights` holds finite F16/BF16 decoding to a mathematical
oracle and CPU products to exactly widened F32 weights. `backend-vulkan` holds device kernels to the CPU's. A
previously unknown layout also needs storage metadata here; a known layout
does not need another sizing or format implementation.
