# `src/model/weights.hpp` - what a model is built from

`ModelWeights::declared_dtype` carries source precision independently of tensor storage, defaulting to F32 for an unspecified in-memory model. The GGUF architecture registry assigns its documented architecture default when reading a file; the placement owner resolves its execution policy.

The part of the architecture contract that the loader and a file's reader use as well, in namespace `infer`.

- `TensorView`: one tensor as a reader hands it to the model: its `name`,
  its `shape` with the fastest dimension first, its storage `type` (the GGUF
  type id) and its `bytes`, with `data` null when they are not in memory.
  An optional `row_source` retains a shared [WeightRows](model-weight_rows.md)
  source for a declared mixer lookup instead of ordinary weight adoption.
  The ordinary GGUF loader sets it for architecture-declared lookup tensors:
  disk mode retains a file reader, while host-backed modes retain the mapping
  or host allocation that owns the table bytes.
- `ModelWeights`: what a model is built from, whatever file it came from:
  the architecture that reads it (`arch`, [architecture](model-architecture.md)),
  and one view per tensor in the file's order, so tensor i is the file's
  tensor i, with unique names: `gguf::read_gguf` refuses a repeated name,
  and the model refuses one among views that reach it another way. A second
  format is a reader that produces this, as `infer::gguf_weights`
  ([registry](model-arch-registry.md)) does for GGUF. Two models built from one
  `ModelWeights` share the architecture object and any explicit row sources;
  their execution storage and histories remain separate.
- `TensorIndex`: the views by name, which `plan_model` builds once over a
  `ModelWeights`'s tensors for the architecture's plan and for setting each
  role's tensor. It refuses a repeated name ("duplicate tensor"), `find`
  gives an absent name as `nullopt` and `at` refuses it with `missing(name)`
  ("missing tensor"), the text the model's resolution also gives for a role
  without a tensor, so each has one owner. It lives no longer than the views.
- `AdoptWeight`: `std::function<BufferPtr(size_t tensor, Backend&, const shard::Held* shard)>`, how
  the model's builder puts a tensor on a backend: the whole tensor where `shard` is null, or a tensor group's member's shard, the copy `shard` describes ([shard](model-shard.md)). The model calls it once
  for each backend that hosts a weight's role, and without one it calls
  `Backend::adopt(view.data, view.bytes)`, or adopts a packed copy of a member's shard that it keeps. The loader's hook
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
  For an external lookup, `row_source` is retained and the backend handle is
  empty; dimensions and type still describe the original table. The runtime
  validates the source against that table and bypasses `AdoptWeight` for it.
  The prepared lookup supplied to the graph holds only the requested rows.

File-backed sources own their shared reader. A memory-backed source borrows its
span unless given a shared owner. The GGUF loader supplies that owner for
host-backed sources. Without an owner, the caller must retain the allocation
or mapping through every read; keeping `row_source` alive alone does not retain
those borrowed bytes.

`TensorView.scale` optionally names a companion by its index in the same `ModelWeights` tensor array. `weight_scale_count` accepts only an NVFP4 weight with an ordinary F32 companion holding one value or one per expert, with consistent byte size, no row source and no nested scale. `Weight` retains the adopted scale buffer and count; `Weight.operand()` returns the shared backend `WeightView`, keeping scale association out of callers. Tensor shards retain the whole expert-scale vector. Independent operator/model checks cover scalar and expert scales, tied use, layer/tensor splits and loading modes. Qualification of scaled released models remains separate.

The model-weight-scales test also loads tied and untied files over two copying backends in layer and tensor splits, in each supported loading mode. It compares with independently constructed F32 weights and requires both backends to execute scaled matrix products using their own companion-scale storage.
