# `src/model/arch/registry.hpp` - the architectures, by the name a file gives

The one place an architecture's name is read and accepted, in namespace
`infer`, and the one step from a file's metadata to its architecture. It
includes every module under `model/arch/`; the loader
([load](inference-load.md)), the CLI's synthetic bench and the tests that
build a model without the loader include it, and nothing else does. A module never compares names: the entry it is reached through says
which variant it reads and under which prefix.

- `ArchEntry`, `kArchitectures`: each `general.architecture` value llmx
  runs and the reader of its files, which reads the configuration under the
  name's prefix: `qwen3` (`qwen3::open_dense`) and `qwen3moe`
  (`qwen3::open_routed`), [qwen3](model-arch-qwen3.md), and `qwen35`
  (`qwen35::open_dense`), [qwen35](model-arch-qwen35.md).
- `architecture_of(file)`: the entry a file names. A file without
  `general.architecture` is read as qwen3, since the tests' fixtures write
  none; an unknown name, or one that is not a string, is refused
  ("inference: unsupported metadata general.architecture").
- `gguf_weights(GGUFModel)`: a GGUF model's weights (`ModelWeights`,
  [weights](model-weights.md)): the architecture its entry reads, whose
  configuration is read once and, before any backend storage exists, a
  view per tensor, refusing a tensor table whose storage count does not
  match its tensors, with a rank above four, or an offset or extent outside
  the payload, which includes a payload its owner released. A view's data
  is null while its tensor's file is not mapped (`gguf::map_payload`). The
  loader, the tests and the synthetic bench call it.
- `SyntheticShape`, `synthetic_model(shape)`: the model `llmx bench` times
  without `--model`, with random weights, Q8_0 matrices and F32 norms,
  written by the first entry's module as a file that names that entry,
  qwen3, so the CLI names no architecture and the file does not lean on
  the default for a missing name.
