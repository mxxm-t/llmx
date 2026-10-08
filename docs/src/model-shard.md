# `src/model/shard.hpp` - how a model splits over a tensor group

The one owner of how a model splits over a tensor group's members (`docs/TENSOR-SPLIT.md`, section 4.2), whatever the backend, in namespace `infer::shard`.
The runtime calls `check_plan` as a group is placed, and it and the loader's adoption hook pack each member's shards.

An architecture declares each role's split on the role (`Role::shard`, [architecture](model-architecture.md)): an `Axis` - none, output rows or input columns, a vector's elements being its columns - and the `ShardSection`s that cover it in order, each `tiles` tiles of `units` units of `unit` rows or columns.
A member takes the same share of the units in every tile, so one tile is a contiguous split and several are Qwen3.5's tiled V heads, which a member takes from every tile by K head.
An expert stack's rows are tiles too, one an expert, so a member takes the same hidden rows of every expert.
A section with an alignment (`ShardSection::align`, an expert's hidden rows under its down stack's block size) is held by cover: a member owns its even share and holds the whole runs of `align` that cover it in every tile, which is the same thing where the share is whole blocks.
A section marked `replicate`, the KV heads, also takes a width that is a multiple of its units, each unit then held by width / units members.

- `parts(role, width, member)`: per tile of each section, what the member owns and the run it holds, the two the same but under an alignment.
- `spans(role, width, member)`: the runs of the role's axis a member holds, its covers, in order, adjacent ones joined; the whole axis at width 1 or for a role whose axis is none.
  A width that neither divides a section's units nor, where it replicates, is a multiple of them is refused by name ("a tensor width of 3 does not divide the 8 heads of blk.0.attn_q.weight"); sections that do not cover the axis are the architecture's error (`std::logic_error`).
- `check_plan(plan, views, width)`: the legality of a width over a model, before anything is placed: every split role's sections, and on the column axis every span of every member on whole blocks of the tensor's storage type, refused naming the projection and the type, and a state layer's saved row the width does not divide, refused naming the layer; an embedded drafter's block is checked as a layer is, and a routed layer's expert stacks as any role's, whose covers fall on whole blocks by construction, so an expert width is not refused.
- `runs(role, view, width, member)`: the bytes of the tensor a member holds, as `Run`s of the tensor's bytes in the order they are packed, each with its offset in the member's copy: whole rows of each span on the row axis, and each row's spans on the column axis, which is the packed column shard; one run of the whole tensor at width 1 or for a role whose axis is none.
  On the row axis a covered row the member does not own is a `zero` run, zero bytes at its place in the member's copy, listed after the file's runs, which stay in the tensor's order; `pack` writes zeros there, and the loader writes them as it places the storage.
  `bytes(runs)` is the size of the member's copy, `pack(runs, src, dst)` makes it from the tensor's bytes, `share(role, width, member)` its rows or columns along the axis, an expert stack's those of one expert, and `shape(role, view, width, member)` its shape.
  The loader streams a member's runs straight into its storage (`Upload::runs`, [load](inference-load.md)).
- `kv_heads(plan, width)` and `state(plan, width)`: the KV heads a member keeps, its share or one where they are replicated, and its share of the recurrent state's K and V heads, which the runtime allocates and the fit counts for a member (`footprint`, [place](model-place.md)).
- `saved(v, width)` and `saved_floats(layer, width)`: a row a state layer saves for a mark as a member holds it, its share of the row's width, and the floats a member saves a row.
