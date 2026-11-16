# Third-party notices

llmx is licensed under the MIT License (LICENSE).
This page keeps the notice of the one third-party material llmx contains.

## ggml

The fixed codebook tables of the IQ1, IQ2 and IQ3 quantization formats are part of those GGUF formats and are copied from ggml's `ggml/src/ggml-common.h` and `ggml/src/ggml-quants.c` at commit eefc4e7321c869496146697d63362f073941aed6.
They are the six files `src/quant/iq1_grid.inc`, `iq2_xxs_grid.inc`, `iq2_xs_grid.inc`, `iq2_s_grid.inc`, `iq3_xxs_grid.inc` and `iq3_s_grid.inc`, each of which names that commit in its first line, and the test suite's copies of the same tables in `tests/spec_decode.py` and `tests/iq1_grid.inc`.
The 16 signed values of the IQ4_NL and IQ4_XS formats (`IQ4_NL_VALUES` in `src/quant/types.hpp`, and the tests' copies in `tests/spec_decode.py`, `tests/raw_blocks.py`, `tests/iq4_nl.cpp` and `tests/iq4_nl_vulkan.cpp`) are ggml's `kvalues_iq4nl` table and count as copied material too.
llmx's decoders, kernels and the rest of its code are written independently.
The notice below is ggml's `LICENSE` file at that commit, unchanged.

```
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
