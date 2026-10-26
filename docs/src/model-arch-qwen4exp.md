# `src/model/arch/qwen4exp.hpp` - Qwen3.8-Flash-Next config and graph pieces

The module in namespace `infer::qwen4exp` reads and validates Qwen4Exp GGUF configuration, builds a complete `Architecture` plan and runs the graph through the model runtime.
`src/model/arch/registry.hpp` registers it as `qwen4exp`, so ordinary GGUF loading and model commands select it from `general.architecture`.
The pinned text schema, its source provenance and the released file's qualification are in [QWEN38-FLASH](../QWEN38-FLASH.md).

A layer split runs it as any model.
On a tensor group the plan shards attention heads, GDN head tiles, expert hidden dimensions and vocabulary rows, and keeps hyper-connection mixing, QSA selection and PLE replicated.
Injection uses the residual collective for full-stream partials.
External PLE rows are read once by the group's first member and uploaded to each member.
Legal widths depend on the plan's head, vocabulary and quant-block spans.

- `Config`: decoder and optional MTP counts, MoE dimensions, attention and GDN geometry, normalized layer kinds, QSA compression and indexer geometry, hyper-connection dimensions and optional PLE hash metadata.
  PLE integer arrays stay `uint64_t`, since the GGUF writer emits `ARRAY<UINT64>` and the shared `metadata::counts` helper would narrow the multipliers.
- `read_config(file, prefix)`: reads the `block_count`, text widths, expert keys, attention head and RoPE keys, SSM keys, hyper-connection keys, per-layer recurrent flags and compression ratios, QSA indexer keys, and the optional PLE keys under the registry prefix.
  A missing recurrent array uses `full_attention_interval` (default 4), and its full-attention entries normalize to `qwen_sparse_attention`.
  A present array must cover all blocks, including a non-recurrent MTP block.
  Compression ratios are zero for linear attention and one common value above one for QSA, and the budget must divide by that value.
- PLE is absent when `ple.layers` is absent.
  A present PLE description has one zero-based decoder-layer index, targets a linear-attention layer, uses an n-gram size of at least two and a positive number of heads per order, and carries exact-width multiplier, offset and vocabulary-size arrays.
  Counts and products are checked against the integer widths the config and row indices use.
  The table row width times its head count gives the PLE embedding input width.
- `check_config(config)`: the geometry constraints, apart from metadata parsing: positive dimensions, divisible query/KV and V/K heads, an even rotary width no wider than an attention head, four MRoPE sections covering the rotary width, valid HC dimensions, and products and cache sizes within supported integer and allocation bounds.
  PLE may only be attached to a linear layer.
- `fill_tables`: builds the F32 rotary frequency and position tables; the `arch-qwen4exp` CTest checks cells of them against values from the pinned HF rotary module.
- `ple_rows(config, vocab_size, previous, current, output)`: computes token-major PLE gather rows without allocating.
  The oldest-first history tail may contain at most `ngram_size - 1` tokens.
  Missing predecessors and history at or before EOS are padded with EOS; the current EOS token still hashes its own context.
  All token IDs, output capacity and signed hash-product bounds are checked before output is written.
- `lookup_ids(step)`: applies those rules to each entry's token snapshot and fills the runtime's ID scratch without backend work or I/O.
  The PLE layer declares its table role and head count in `lookup`.
  With an external row source, `apply_ple` gathers the runtime's packed unique rows through remapped IDs; a resident table uses the same hash and the backend's embedding path.
- `ple(config, step, weights, embeddings, out, scratch)`: projects the gathered n-gram embeddings into per-stream keys and a value, normalizes the keys and current streams, applies the signed square-root gate, then runs the causal dilated convolution, SiLU and residual add.
  Its convolution tail is the layer's F32 `state_tail`; `ple_recur` replays that update from saved normalized and gated rows during a recurrent rerun.
- `mix(config, step, weights, out, scratch)` and `inject(config, step, weight, normed, block, gates)`: mix normalized streams through the learned down and up projections, keeping the normalized streams in `scratch.normed`, then inject a block result into the original streams through learned per-stream gates.
  The architecture also wires the GDN recurrence and convolution, QSA index selection and selected-KV attention, routed and shared experts, PLE lookup and convolution, per-block HC and the final HC head.
  The routed and shared experts are one `blocks::moe_step` call ([blocks](model-arch-blocks.md)) into the block's output rows, cleared first from the model's zero row (`tables[2]`); the expert slots hold k routed rows and the shared expert's rows after them.
  On a tensor group, injection writes zeroed full-stream partials for the runtime's residual sum (`Backend::repeat_row` from the same zero row), so the original residual is not added once per member.
  PLE replay inputs are saved whole on each member, while GDN inputs follow its sharded head tiles.
- The recurrent output uses sigmoid gating through `NormGate::sigmoid`, where Qwen35's gated normalization keeps SiLU; the released GGUF carries no key for it, so the module fixes sigmoid.
- `plan_drafter`, `draft_rows` and `draft`: the optional MTP block uses the target's raw HC-wide residual and the residual-sized carried rows, marks and KV layer.
  Its input normalizes each hidden stream with `hnorm` and the token embedding with `enorm`, then runs a fused `[embedding; hidden]` projection or the older pair of square projections.
  Context rows write only K and V and, for QSA, raw indexer keys.
  Draft rows run both HC blocks, the routed and shared FFN, then the block's own HC head and the target output matrix.
  The next step carries the raw draft residual before that head.
  An MTP compression of 0 selects ordinary causal attention, and a value above 1 selects QSA with its own ratio; the selected-row slot covers both trunk and MTP ratios.
- `normalize_drafter`: adapts a sidecar's private and shared converter headers through the registry callback after tokenizer equality.
  It validates before it swaps headers, keeps private globals under the MTP namespace, fills a complete PLE omission and restores only the legacy zero compression prefix.
  File spans, tensor order, offsets and payload owners are unchanged.

The model takes resident PLE tables and explicit file-backed row sources.
The loader builds bounded sources from declared lookup tensor spans, and `PlacementRequest::lookup_storage` selects `auto`, `device` or `file` residency.
The runtime stages requested packed rows for external sources, separately for normal and apart execution.

Tests:
- `tests/arch_qwen4exp.cpp` (CTest `arch-qwen4exp`) exercises the reader with in-memory metadata mutations and exact PLE row vectors, including the HF reference rows.
  `--reference FILE` checks the pinned fixture with its pinned integer constants; `--gguf FILE --reference FILE` instead reads PLE metadata and vocabulary from the generated GGUF and checks the PLE hash rows.
- `tests/qwen4exp_graph.cpp` (CTest `qwen4exp-graph`) compares helper outputs with intermediates captured from the pinned HF forward pass.
- `tests/qwen4exp_mtp.cpp` (CTest `qwen4exp-mtp`) compares two draft steps with a scalar-double assembled oracle, exercises selection-sensitive auxiliary keys, checks unchanged target logits, partial rejection and edited continuations, and loads tiny blocks-only GGUF pairs through `infer::load_model`.
- `tests/qwen4exp_model.cpp`, built by CMake and not registered as a CTest, is the end-to-end fixture harness for full, sliced and cached logits, mark and retract, batching, scratch reuse, generated-token follow-up and two-CPU layer placement.
  A host-history snapshot restores the combined recurrent state, PLE token tail and QSA auxiliary KV after a distinct history overwrites the released slots, and the continuation is compared bit for bit with a fresh run.
  It also exercises `infer::load_model` across load modes and lookup-storage policies, checking progress and that tables read from the file do not count as whole-table payload.
  Its `--tensor-width 2` mode runs a group of two CPU backends, or of the two devices `--device vulkan:0,vulkan:1` names, against a one-member CPU control.
- `tests/qwen4exp_fixture.py` writes the GGUF fixture and, on POSIX, runs the public server with a kept disk cache: it writes a whole history, restarts after a clean shutdown and after a SIGKILL, and checks adoption, reuse and the next token's log-probabilities against independent tiny HF logits.
