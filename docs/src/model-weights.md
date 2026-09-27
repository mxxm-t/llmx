# `src/model/weights.hpp` - what a model is built from

The part of the architecture contract that the loader and a file's reader use as well, in namespace `infer`.

- `TensorView`: one tensor as a reader hands it to the model: its `name`,
  its `shape` with the fastest dimension first, its storage `type` (the GGUF
  type id) and its `bytes`, with `data` null when they are not in memory.
- `ModelWeights`: what a model is built from, whatever file it came from:
  the architecture that reads it (`arch`, [architecture](model-architecture.md)),
  and one view per tensor in the file's order, so tensor i is the file's
  tensor i, with unique names: `gguf::read_gguf` refuses a repeated name,
  and the model refuses one among views that reach it another way. A second
  format is a reader that produces this, as `infer::gguf_weights`
  ([registry](model-arch-registry.md)) does for GGUF. Two models built from one
  `ModelWeights` share the architecture object and nothing else.
- `TensorIndex`: the views by name, which `plan_model` builds once over a
  `ModelWeights`'s tensors for the architecture's plan and for setting each
  role's tensor. It refuses a repeated name ("duplicate tensor"), `find`
  gives an absent name as `nullopt` and `at` refuses it with `missing(name)`
  ("missing tensor"), the text the model's resolution also gives for a role
  without a tensor, so each has one owner. It lives no longer than the views.
- `AdoptWeight`: `std::function<BufferPtr(size_t tensor, Backend&)>`, how
  the model's builder puts a tensor on a backend. The model calls it once
  for each backend that hosts a weight's role, and without one it calls
  `Backend::adopt(view.data, view.bytes)`. The loader's hook
  (`infer::planning_adopt`) adopts a weight on a backend that reads in place
  and, in the streamed loads `auto` and `direct`, gives a copying backend
  unfilled storage (`Backend::alloc_weight`), which the loader fills once the
  model is built ([load](inference-load.md)); in a mapped load it adopts
  through a copying backend too. The model reads no weight's bytes while it
  is built.
- `Weight`: a tensor resolved once at load - type, a buffer handle from the
  backend that hosts it and the role's two dimensions, one expert's for a
  stack. The model keeps a row of them per layer indexed by role id, and one
  for the pass's roles, and hands a row to each part it calls.
  `Weight::slice()` names the weight's location; the model never
  dereferences it. The forward pass indexes a row by role id instead of
  rebuilding `"blk.N."` and hashing a tensor name for every projection of
  every layer of every token, and a device backend recognizes the same weight
  across calls. See `docs/DEVICE-EXECUTION.md` step 1.
