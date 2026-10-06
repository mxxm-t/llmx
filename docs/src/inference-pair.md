# `src/inference/pair.hpp` - a drafter file beside a model

The one check that a drafter file pairs with the model it drafts for (docs/SPECULATIVE.md, step 6), in namespace `infer::spec`, made on the two files' headers before any byte of either is read, and the joining of a drafter's MTP blocks to its model.
The loader ([load](inference-load.md)) resolves `--drafter PATH` through it, and `llmx-drafter-pack` checks what it writes and reads with it.

- `DrafterKind { mtp, model }`: what a drafter file is for its target: its MTP blocks, as converters write them after the target's layers, or a draft model, a model of its own drafting by its own forward pass.
- `pair(target, target_path, drafter, drafter_path) -> DrafterKind`: the kind, or a refusal that names both files and what differs, with both values.
  - Every kind shares the target's tokenizer: `tokenizer.ggml.model`, `pre`, `tokens`, `token_type`, `merges` and `eos_token_id` equal or absent in both; the pad and begin ids may differ.
  - A file of the target's architecture whose tensors all sit in blocks past the target's layers (`gguf::block_of`) is its MTP blocks. Its MTP count must be above 0 and the target's 0, its block count the target's plus that count, every tensor within those blocks, and every key under the architecture's prefix the target's, but for the two counts and per-block arrays, whose entries for the target's blocks must be the target's.
  - Any other file is a draft model, of an architecture the registry runs; a file of another architecture, a DFlash drafter among them, is refused by its name.
  - What pairing cannot see, a drafter's weights trained for another model of the same shapes, shows only as low acceptance, which `generate` prints and `/v1/health` counts.
- `join_blocks(target, drafter)`: MTP blocks that pair, joined to the target as if its file carried them: the drafter's keys under the architecture's prefix taken over the target's, so the architecture reads the target's configuration with its MTP block, and its tensors appended as a file after the target's (`gguf::append`), from which the loader maps or streams them.
