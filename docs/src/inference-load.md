# `src/inference/load.hpp` - loading a model file

The one sequence that turns a model file into a model ready to run, in namespace `infer`.
The CLI's commands that run a model, `llmx-split-check` and `compare_cpu` all load through it, and none of them repeats its steps.

- `LoadedModel`: what a load gives its caller.
  - `file` is the `gguf::GGUFModel` the model reads while it lives.
  - `tok` is its tokenizer.
  - `chat` is its `chat::ChatFormat` (`chat::chat_format`), which `chat` and the server render conversations with.
  - `plan` is what each device of a split was given (`LayerSplit::describe`), and is empty otherwise.
  - `model` is the placed `Model`. It is declared last, so it is destroyed before the file it reads.
  - A `LoadedModel` is built in place by `load_model` and is never copied or moved, since the model keeps the file's address.
- `load_model(path, backends, request, options = {}, progress = {}) -> unique_ptr<LoadedModel>`:
  1. It reads the file, or the first shard of a set, with `gguf::read_gguf`, which reports the payload to `progress`.
  2. It builds the tokenizer and the chat format.
  3. It places the model over `backends` as `request` asks, with `options`' caches (`infer::place_model`).
  4. It releases the host's copy of the weights (`GGUFModel::release_payload`) when no weight reads it in place (`Model::holds_payload`), which is the case for a model on device backends alone.

The caller makes the backends (`backend::make_backends`) and builds the request and options from its own flags, so a device that cannot be opened, or a malformed device list, fails before the file is read.
The caller also renders the progress: the CLI prints "Reading model metadata..." before the call, and its bar prints "Preparing model..." once, when the payload is complete, while the model is placed.
Worker counts stay the caller's (`Model::set_threads`).

A failure anywhere in the sequence throws and frees what was built.
A model whose construction failed has already drained its backends (`model/arch_qwen.hpp`).
