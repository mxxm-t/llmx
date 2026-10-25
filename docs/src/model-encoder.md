# `src/model/encoder.hpp` - the vision encoder's contract and runner

The one interface the runtime calls for image input ([VISION](../VISION.md), sections 3 and 9).

`VisionArchitecture` is what a vision model family's module implements: `preprocess()`, the `image::Params` that `image::prepare` applies; `tokens(grid_h, grid_w)`, the rows an image's patch grid gives; `positions()`, how those rows are placed among a prompt's positions; `token_texts()`, the texts of the tokens a chat template marks an image with; `plan(TensorIndex)`, its roles, the ops some backends lack, the floats a patch takes in each arena slot, its attention's heads, the width of an image token's row and its tables; `fill_tables`; `inputs(grid_h, grid_w)`, each patch's position streams and learned-position taps computed on the host; and `encode(EncodeStep)`, its math as backend ops.
A member a later module needs comes with that module.

`VisionEncoder` is the runner.
It plans the module over an mmproj's tensors, refuses a backend without one of the plan's ops by name, checks each role with the runtime's `role_fits` (a convolution's kernel stored `[kw, kh, c, out]` read as the matrix `[kw * kh * c, out]`), adopts the weights and fills the tables.
`encode(prepared)` encodes one image a call, so an image's rows never depend on another image: it writes the patches into the arena's input slot, attends through a non-causal `KVView` over the image's rows, and returns the image's rows to the host.
Its arena and its one layer of K and V grow to the largest image encoded and never shrink; the K and V are scratch of one encode, F32 always, outside the model's cache types.
