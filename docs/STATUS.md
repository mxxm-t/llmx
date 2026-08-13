# llmx — Development Status

Living tracker. This is the disposable file: notes here are only useful while
work is in progress. When a feature ships, delete its block below and mark the
row `Done` in the table. Read it together with `docs/ROADMAP.md` (the stable
plan) and `docs/ARCHITECTURE.md` (the layer rules) — STATUS carries where each
feature currently stands right now.

## Status table

| Feature                                  | Status   |
|------------------------------------------|----------|
| Layered restructure                      | Done     |
| Build config (config.hpp + CMake + build.bat) | Done |
| Test suite (roundtrip / perf / tokenizer)| Done     |
| Perf `bench` command                     | Done     |
| More quant formats (Q4_0, Q4_1, ...)     | Planned  |
| More model architectures (Llama, ...)    | Planned  |
| More formats (safetensors, ...)          | Planned  |
| GPU backends (ROCm / CUDA / Vulkan)      | Planned  |
| Multi-device split                       | Planned  |
| Multi-node / cluster                     | Planned  |
| Multi-user server                        | Planned  |

## Active feature blocks

One block per in-flight feature. A block is what lets a fresh agent pick a
feature back up with a "continue feature X" prompt, so keep it current. When the
feature ships, delete its block and mark the row `Done` above.

### <feature name>

- **Goal:** what the feature must achieve (from `docs/ROADMAP.md`).
- **Done:** what is implemented and verified.
- **Left:** what remains, in dependency order.
- **Gotchas:** non-obvious constraints, decisions, or traps.

Nothing is in flight right now. When you start a feature, open a block above
before writing code — see `AGENTS.md` → "Starting a feature".
