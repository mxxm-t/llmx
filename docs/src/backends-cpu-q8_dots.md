# `src/backends/cpu/q8_dots.hpp` - CPU dots over quantized activations

The dots the CPU backend takes for packed weights against activations quantized per block of 32 values (`backend::q8`), included by `cpu_backend.hpp`, whose page says when each path is taken ([backends-cpu](backends-cpu.md)).

- `Rows16`, `size_rows`, `quantize16`: a call's activations as 16-bit integers with each block's scale and integer sum, sized apart from quantizing so blocks can be filled from several threads; a tiny block rounds against a representable scale (`quantize_small`).
- The decode dots, one weight row against one activation row: `dot_q4_0`, `dot_q4_1`, `dot_q45_K` (Q4_K and Q5_K), `dot_q6_K` and `dot_mxfp4`, each summing a block's products exactly in integers and applying the scales once a block; `dot_mxfp4_wide` takes decoded weights with double products where the scales would lose range.
- The prompt dots (`dot_block`): a block of consecutive weight rows against several activation rows, 256 values at a time, each weight row unpacked once and met by every column.
- Q8_0 is not here: it keeps the original F32 inputs in the float dots of `cpu_backend.hpp`.

`q8-dots` holds these to a double-precision reference fed the same quantized activations (`AGENTS.md`, Tests).
