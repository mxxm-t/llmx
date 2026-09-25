# `src/inference/load.hpp` - loading a model file

The one sequence that turns a model file into a model ready to run, in namespace `infer`.
The CLI's commands that run a model, `llmx-split-check` and `compare_cpu` all load through it, and none of them repeats its steps.

- `LoadedModel`: what a load gives its caller.
  - `file` is the `gguf::GGUFModel` whose bytes a host backend reads in place while the model lives.
  - `tok` is its tokenizer.
  - `chat` is its `chat::ChatFormat` (`chat::chat_format`), which `chat` and the server render conversations with.
  - `plan` is what each device of a split was given (`LayerSplit::describe`), and is empty otherwise.
  - `model` is the placed `Model`. It is declared last, so it is destroyed before the file it reads.
  - A `LoadedModel` is built in place by `load_model` and is never copied or moved, since a host backend keeps addresses in the file's payload.
- `recording_adopt(weights, host_reads) -> AdoptWeight`: the adoption hook `load_model` builds the model with. It adopts each weight inline and sets `host_reads[i]` when a backend that reads in place (`Backend::reads_in_place`) took tensor i. It sizes `host_reads` before the model is built, so recording cannot fail while a buffer is held. `model-validation` checks what it records.
- `load_model(path, backends, request, options = {}, progress = {}) -> unique_ptr<LoadedModel>`:
  1. It reads the file, or the first shard of a set, with `gguf::read_gguf`, maps its payload (`gguf::map_payload`) and reads every tensor's pages in, in file order, reporting the payload to `progress` (`gguf::warm`).
     Pages read in stay resident only while the host can hold them, so a payload larger than the host's available memory (`core::host_memory_available`) is not read in: its pages would be evicted before a device copied them and read from disk twice, so they are left for whoever reads them to read once, and the progress reports 0 and then the whole payload.
  2. It builds the tokenizer and the chat format.
  3. It takes the file's weights (`infer::gguf_weights`), which reads the configuration once and checks the tensor table.
  4. It places the model over `backends` as `request` asks, with `options`' caches (`infer::place_model`), through `recording_adopt`.
  5. When no host reads a weight in place, which is the case for a model on device backends alone, it releases the host's copy of the weights (`GGUFModel::release_payload`): a backend that copies has consumed its weights when `adopt` returns. Otherwise it lets the pages of every tensor no host reads leave the host's working set (`GGUFModel::drop_pages`), as with experts on the CPU beside a device.

The caller makes the backends (`backend::make_backends`) and builds the request and options from its own flags, so a device that cannot be opened, or a malformed device list, fails before the file is read.
The caller also renders the progress: the CLI prints "Reading model metadata..." before the call, and its bar prints "Preparing model..." once, when the payload is complete, while the model is placed.
Worker counts stay the caller's (`Model::set_threads`).

A failure anywhere in the sequence throws and frees what was built.
A model whose construction failed has already drained its backends (`model/arch_qwen.hpp`).
