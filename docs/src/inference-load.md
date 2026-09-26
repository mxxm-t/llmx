# `src/inference/load.hpp` - loading a model file

The one sequence that turns a model file into a model ready to run, in namespace `infer`.
The CLI's commands that run a model, `llmx-split-check` and `compare_cpu` all load through it, and none of them repeats its steps.

- `LoadMode { automatic, mapped }`, `load_mode_of(name)` and `load_mode_name(mode)`: how a load reads the weights a copying backend takes, the CLI's `--load-mode` (`auto` or `mapped`), the same on every backend. An unknown name throws.
- `LoadTimes`: where a load's time went, for the CLI's timing line: the mode, the files the weights were streamed from and the bytes read for them, and the seconds spent building the model (`construct`), in the busiest reader thread's reads (`read`), in the uploads (`upload`) and in the uploads' waits for a read (`wait`).
- `LoadedModel`: what a load gives its caller.
  - `file` is the `gguf::GGUFModel` whose bytes a host backend reads in place while the model lives.
  - `tok` is its tokenizer.
  - `chat` is its `chat::ChatFormat` (`chat::chat_format`), which `chat` and the server render conversations with.
  - `plan` is what each device of a split was given (`LayerSplit::describe`), and is empty otherwise.
  - `times` is the load's `LoadTimes`.
  - `model` is the placed `Model`. It is declared last, so it is destroyed before the file it reads.
  - A `LoadedModel` is built in place by `load_model` and is never copied or moved, since a host backend keeps addresses in the file's payload.
- `WeightPlan { host_reads, uploads }`: what the loader's hook records while the model is built: per tensor whether a backend that reads in place took it, and each `Upload { tensor, backend, buffer }`, a weight a copying backend took.
- `planning_adopt(weights, backends, plan, defer) -> AdoptWeight`: the adoption hook `load_model` builds the model with. A backend that reads in place (`Backend::reads_in_place`) adopts the weight where it lies and `host_reads[i]` is set. A backend that copies adopts it there and then unless `defer`; with `defer`, as the automatic mode builds, it gets storage that is not filled yet (`Backend::alloc_weight`), recorded as an upload. The model hands each tensor to each backend at most once, and `plan` is sized for that before the model is built, so recording cannot fail while a buffer is held. `model-validation` checks what it records in both ways.
- `detail::plan_pieces(tensors, spans, file_of, granules, limit)`: the reads of the automatic mode. Each `Piece` reads whole granules of one file (`format::FileReader::granule`) from a granule boundary, at most `limit` bytes rounded down to whole granules, and carries on over the tensors that follow in the same file while no gap between them is longer than a granule; its parts say which bytes of which tensor it holds and where.
- `detail::stream(pieces, readers, destinations, streamed, times, readers_count = 1)`: `readers_count` reader threads read the pieces into a ring of four `core::HostPages` slots, reader j taking pieces j, j + readers_count and so on, each slot marked with the piece it holds; and the calling thread writes each part to every upload of its tensor (`destinations`) and reports the part's bytes to `streamed`. A read that comes up short of a part stops it with "`<path>` ended at N bytes, before its tensors", since the file was cut after its header was read. Whatever fails, a read, a write or the callback, the reader is stopped and joined and the ring freed before the error goes on.
- `load_model(path, backends, request, options = {}, progress = {}, mode = LoadMode::automatic) -> unique_ptr<LoadedModel>`:
  1. It reads the headers of the file, or of every shard of a set, with `gguf::read_gguf`.
  2. `mapped` maps the payload (`gguf::map_payload`) and reads every tensor's pages in, in file order, reporting the payload to `progress` (`gguf::warm`). Pages read in stay resident only while the host can hold them, so a payload larger than the host's available memory (`core::host_memory_available`) is not read in: its pages would be evicted before a device copied them and read from disk twice, so they are left for whoever reads them to read once, and the progress reports 0 and then the whole payload.
     `auto` maps the payload only when a host will read weights in place, one of `backends` or the CPU backend the placement adds for experts on the CPU (`PlacementRequest::cpu_moe`), and reads nothing yet.
  3. It builds the tokenizer and the chat format.
  4. It takes the file's weights (`infer::gguf_weights`), which reads the configuration once and checks the tensor table; a view's data is null while its file is not mapped.
  5. It places the model over `backends` as `request` asks, with `options`' caches (`infer::place_model`), through `planning_adopt`, deferring the copies in `auto`. When it returns every role has been checked and every weight, window, cache and RoPE table allocated, so in `auto` a missing tensor or a cache that does not fit fails before any weight is uploaded.
     Building first and copying after costs `mapped` the overlap of the copies with the allocations, which measured 3.6 percent on a warm Qwen3-8B load on the Radeon VII, so `mapped` keeps copying as it builds; `auto` overlaps its copies with its reads instead.
  6. `mapped` has copied every weight already, as the model resolved it: construction adopts it through a copying backend straight from the mapping, as the loader always did.
     `auto` reports 0, opens one `format::FileReader` per file the copied weights lie in, plans the reads of those weights (`detail::plan_pieces`, 16 MiB) and streams them on two reader threads (`detail::stream`), which measured fastest of one or two readers and 16 or 32 MiB; it then reads in the pages of the weights a host alone reads, under the same host-memory rule applied to their bytes. The progress counts the bytes of every tensor some backend took, each once, and reaches complete after the last upload. On the CPU alone nothing is copied, so the load reads in the pages of every tensor the model takes, as `mapped` does, after the model is built rather than before.
  7. When no host reads a weight in place, which is the case for a model on device backends alone, it releases the host's copy of the weights (`GGUFModel::release_payload`), which `auto` never mapped. Otherwise it lets the pages of every tensor no host reads leave the host's working set (`GGUFModel::drop_pages`), as with experts on the CPU beside a device.

The caller makes the backends (`backend::make_backends`) and builds the request and options from its own flags, so a device that cannot be opened, or a malformed device list, fails before the file is read.
The caller also renders the progress: the CLI prints "Reading model metadata..." before the call, and its bar prints "Preparing model..." once, when the progress is complete.
Worker counts stay the caller's (`Model::set_threads`).

A failure anywhere in the sequence throws and frees what was built.
A model whose construction failed has already drained its backends (`model/arch_qwen.hpp`), and one whose stream failed drains them as it is destroyed, before any buffer is freed.
