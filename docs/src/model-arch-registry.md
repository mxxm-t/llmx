# `src/model/arch/registry.hpp` - the architectures, by the name a file gives

Each entry also declares the GGUF architecture default activation dtype, BF16 for the current Qwen entries. `gguf_weights` carries it into `ModelWeights`; the registry does not infer it from tensor quantization or claim a source-checkpoint dtype absent from the file.

The one place an architecture's name is read and accepted, in namespace
`infer`, and the one step from a file's metadata to its architecture. It
includes every module under `model/arch/`; the loader
([load](inference-load.md)), the drafter pairing ([pair](inference-pair.md)), the CLI's synthetic bench and the tests that
build a model without the loader include it, and nothing else does. A module never compares names: the entry it is reached through says
which variant it reads and under which prefix.

- `ArchEntry`, `kArchitectures`: each `general.architecture` value llmx
  runs and the reader of its files, which reads the configuration under the
  name's prefix: `qwen3` (`qwen3::open_dense`) and `qwen3moe`
  (`qwen3::open_routed`), [qwen3](model-arch-qwen3.md), and `qwen35`
  (`qwen35::open_dense`) and `qwen35moe` (`qwen35::open_routed`),
  [qwen35](model-arch-qwen35.md), and `qwen4exp` (`qwen4exp::open`), [qwen4exp](model-arch-qwen4exp.md).
  An optional `normalize_drafter(target, side)` callback adapts a same-architecture sidecar's headers to the blocks-only pairing contract.
  The pairing owner calls it after tokenizer equality and before classifying the file.
  Only qwen4exp supplies it; its module validates every exception before changing metadata or tensor names and preserves storage.
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
  views point into the model, so a call on a temporary model is deleted and
  does not compile. The loader, the tests and the synthetic bench call it.
- `VisionEntry`, `kVisionEncoders`, `vision_of(file)`: the vision encoders
  llmx runs, each by the `general.architecture` and `clip.projector_type` an
  mmproj file gives, its reader and its default dtype; another architecture
  or projector is refused by name (docs/VISION.md, section 9).
- `vision_weights(GGUFModel)`: an mmproj file's encoder module and a view per
  tensor (`VisionWeights`), with the same refusals as `gguf_weights`. Its views
  point into the model too, so a call on a temporary model is deleted and does
  not compile.
- `SyntheticShape`, `synthetic_model(shape)`: the model `llmx bench` times
  without `--model`, with random weights, Q8_0 matrices and F32 norms,
  written by the first entry's module as a file that names that entry,
  qwen3, so the CLI names no architecture and the file does not lean on
  the default for a missing name.

`gguf_weights` associates NVFP4 `.weight` tensors with optional F32 `.scale` companions through [weight_scales](format-weight_scales.md), storing indices in `TensorView.scale`. It does not apply arithmetic or treat `.input_scale` as a weight scale. Model validation owns companion compatibility; existing `TensorIndex` owns duplicate-name refusal. `vision_weights` asks the same format owner to refuse any associated scale companion before reading the vision module or exposing tensor storage, since the vision encoder does not consume scaled weights yet.
