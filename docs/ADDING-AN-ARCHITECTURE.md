# Adding a model architecture

An architecture is everything that depends on a file's `general.architecture`:
- which metadata keys are read and refused;
- which tensors are read, and how each is checked;
- what each layer keeps per position or per sequence;
- how wide the activations are;
- the math of the layers.

In llmx that is one module under `src/model/arch/`, and nothing else in `src/` names it.
This page covers what a module holds, what the shared runtime does for it, the rules both keep, the order the work goes in, and the gates a new architecture passes before it merges.
- The contract a module implements: [model-architecture](src/model-architecture.md).
- The runtime: [model-runtime](src/model-runtime.md).
- What the execution model already allows for, and the assumptions not to make: [EXECUTION](EXECUTION.md), "Beyond dense Qwen".

`src/model/arch/qwen3.hpp`, which runs Qwen3 and its mixture-of-experts form, is the worked example.

## The runtime, the modules and the registry

- **The runtime** (`src/model/runtime.hpp` and `src/model/place.hpp`) runs every architecture. It owns:
  - sequences and their caches, and passes over batches of entries;
  - stages over devices and the pipelined prompt;
  - the activation arena and the crossings between devices;
  - placement and the fit (`src/model/layer_split.hpp`);
  - experts on the host and streamed to a device;
  - forks, rollback and the logits.

  It indexes the file's tensors by name once and finds each role's tensor there, reads what a module declares, and calls the module's code.
  It names no architecture, tensor or metadata key, and a check in the test suite (`tests/arch_boundary.py`) keeps it so.
- **A module** (`src/model/arch/<name>.hpp`) holds everything that depends on the architecture:
  - its configuration, read from the file's metadata, and every refusal;
  - its plan: the layers the runtime runs, each layer's kind and tensor roles, the arena slots, the residual row, the context length, each layer's cache and the position tables;
  - the values of its position tables;
  - its math, written as backend ops: the embedding, each layer's mixer part and feed-forward part, and the head.
- **The registry** (`src/model/arch/registry.hpp`) maps each `general.architecture` value to its module. It is the only place such a name is accepted, and an unknown name is refused there.
  A module never compares architecture names: the registry entry it is reached through says which variant it reads and under which key prefix.
  The registry also names the synthetic model `llmx bench` times without `--model`.
- **Includes.**
  - A module includes the contract (`src/model/architecture.hpp`, `src/model/weights.hpp`), `src/model/arch/metadata.hpp`, the shared graph pieces once a second module needs them (see "One owner"), the backend interface and the format headers its readers read.
  - It may include another module's graph header when it runs that module's blocks, as a drafter of an architecture does.
  - It never includes the runtime or the registry.
- **Not in a module,** because they are keyed by something other than the architecture:
  - the pretokenizer, chosen by `tokenizer.ggml.pre`, in `src/tokenizer/tokenizer.hpp`;
  - the chat template, which is the file's own, rendered by `src/inference/chat.hpp`;
  - the backend ops, named for their math in `src/backends/backend.hpp`, which any module may call;
  - flags, which the CLI reads. A flag that does not apply to a model is refused where it is honoured, as `place_model` refuses `--n-cpu-moe` and `--cpu-moe` for a model with no routed layers, on the CPU as beside a device.

## The files a new architecture adds

| File | What it holds |
|---|---|
| `src/model/arch/<name>.hpp` | the module: one header until it passes about 1,000 lines, then a directory `src/model/arch/<name>/` |
| a line per name in `src/model/arch/registry.hpp` | each `general.architecture` value the module runs, with its reader |
| `docs/src/model-arch-<name>.md` | the file's page, as every source file has one |
| `docs/<NAME>.md` | when the math or the file's conventions need more than the source page: the shapes, the forward pass of each layer kind, the GGUF conventions and the files on hand, as [QWEN35](QWEN35.md) does |
| `tests/arch_<name>.cpp` | CTest `arch-<name>`: every refused key and tensor with its text, the plan of each layer kind on a fixture, and the footprint |
| `tests/fixture_<name>.py` | the tiny model's configuration, its GGUF writer, and the map from its GGUF tensor names to the reference's parameter names, imported by the suites and by `tools/gen_baseline.py` |
| goldens in `tests/data/` | written by `tools/gen_baseline.py` from the reference named in STATUS, on the tiny model and on the smallest released model that fits the host |
| when the family ships HF directories | the module's `config.json` reader and its HF name table, beside its GGUF reader |
| a plan block in `docs/STATUS.md` and a line in ROADMAP #2 | where each piece lives, the steps, their gates and the decisions |

A new architecture adds nothing that names it to `src/model/runtime.hpp`, `src/model/place.hpp`, `src/inference/`, `src/server/` or `src/cli/`.
When the runtime lacks something the architecture needs, see "When the runtime needs more".

## What each part of a module does

### The configuration

- The reader reads keys under the prefix the registry hands it.
- A key the math depends on is either read, or refused unless it holds the value the math assumes. The refusal names the key.
- Defaults are the ones the format documents. An explicit malformed value never selects a default.
- Metadata values are read through `src/model/arch/metadata.hpp`, which every reader shares, so a value's type and range checks and their messages have one owner.
- The rules a configuration keeps whatever file it came from (whole query heads per KV head, even widths, sizes that fit an allocation) are one function, `check_config`, which the reader of every format calls.
- Behaviour that differs between versions of an architecture is keyed on the registry entry the module was reached through, and read once into the configuration. It is never inferred from which tensors a file holds.
  The exception is a rule the format itself states by tensor presence: a qwen3moe GGUF's routed layers are those with a router.

### The plan

The module's `plan` is built once, from the configuration and the runtime's index of the file's tensors (`TensorIndex`). It gives:
- the decoder layers the runtime runs. Blocks a file carries for other uses, such as a multi-token-prediction block, are not among them;
- for each layer:
  - its kind, a value of the module's own that the runtime hands back on every call for that layer;
  - whether its feed-forward part holds routed experts;
  - what its mixer keeps for each sequence between passes: keys and values for every position, a recurrent state, or nothing;
  - its roles;
- the roles of the pass: the embedding and the head;
- for each role:
  - its tensor name;
  - the part it is placed with;
  - its kind: norm, matrix, gathered table, stacked experts, or an F32 table an op reads whole;
  - its expected shape;
  - what streaming does with it: nothing, a copy beside the mixer, or a copy written into a window each pass;
  - an alias taken when the name is absent, as a tied head takes the embedding;
- the arena slots, as the floats one row takes in each, with slot 0 the residual and each slot sized for the widest use any layer kind makes of it;
- the width of the residual row, which is also what a crossing between devices carries;
- the context length: the positions a sequence may reach, which the tables cover and the cache budget defaults to;
- the geometry of K and V, for the layers that keep them;
- the sizes of the position tables.

The plan looks tensors up through `TensorIndex`, which refuses a missing or repeated tensor with the one text the runtime uses.
A layer whose tensors disagree with its kind is refused here, naming the layer, and so is a tensor form the module does not read.

The runtime then:
- holds the plan to slot 0 as wide as the residual and to every role id inside the row of weights, and treats either as the module's error;
- checks every role by its kind: its shape, F32 for a norm, and trailing axes of one;
- checks each assigned backend's weight types through `supports_type`, before adopting any weight;
- decides whether each host layer can stream to its mixer device once: all its streamed roles must have supported types there, or that layer stays on the host;
- adopts each role on the device of its part;
- reads one buffer for a tensor that two roles take on one device;
- counts each layer's roles in the fit, in file order, and the pass's roles through the fit's three fields for them: the embedding's table, the head's matrix and the head's norm.
  A pass role of another part and kind, such as a second head weight, is refused by the fit until a field for it is added to `Footprint`, in the branch of the module that first declares one.

None of that is written again for each architecture.

Until the architecture's math runs, its `plan` refuses the architecture by name as its last check, so every command refuses the file. `tests/arch_<name>.cpp` calls the module's reader and plan builder directly.

### The position tables

`fill_tables` computes them once at load, as values that depend only on the position and the configuration.
The runtime keeps them for the model's life and adopts them on every device that runs a mixer part.

### The math

`embed`, `mixer`, `ffn` and `head` each receive a `Step`, which holds:
- the backend of the device the part runs on, and the arena's slots there;
- the residual at the call's first row, the rows and their runs;
- the weights to read, by role id;
- the layer's kind and cache views;
- the rows' positions and the tables.

The parts issue backend ops and nothing else:
- no loop over rows, heads or elements, no allocation, no read of device memory and no wait;
- the rows' runs go to every op that takes them, so a device chooses its kernels by each prompt's extent;
- a part does not know which device it runs on or whether its weights were streamed there, since the runtime hands it the weights it should read.

The runtime calls each part once per layer per pass, or once per group of entries when a routed layer streams.

### New backend ops

When a layer needs math the op set lacks, the op is added to `src/backends/backend.hpp`, named for what it computes. It comes with:
- a CPU implementation and a Vulkan implementation;
- a `backend-vulkan` case against the CPU. That case is exact where the arithmetic is the same operations in the same order, and holds a stated tolerance where a transcendental or a reduction order differs.

A backend without the op refuses the model at load, naming the op; no backend substitutes other arithmetic.
The runtime makes that check from the ops the plan says each part issues. That field arrives with its first user (Qwen 3.x step 4).
The op's kernels keep the batch-invariance rule below.

### Drafters

None of this exists yet; each piece arrives with its first user, as "When the runtime needs more" says.
- A drafter the file embeds, such as an MTP block, belongs to the module. It will reach the speculative decoding system through a drafter hook of the module, which arrives with Qwen 3.x step 9, and the module will own its part of taking a rejected draft back.
- A drafter in a sidecar file will be reached through drafter entries of the registry, which arrive with speculative decoding step 5, and the target module's graph header will supply the blocks it runs.
- The rows a drafter reads (taps of the residual entering listed layers) will be taken by the runtime in its layer loop, from speculative decoding step 5 on.
- Output with the drafter on equals output with it off.

### When the runtime needs more

New runtime machinery is added to the contract in a commit of its own, with the new architecture as its first user, and leaves every existing architecture byte-identical. Examples: a cache kind, per-sequence state, a capability flag, rows that cross between devices with the residual, token ids in a part, buffers derived at load, a per-pass hook, or a drafter hook.
From then on the runtime serves every module that declares it, and it never tests which architecture it runs.
Nothing is added to the contract before an architecture uses it.

### Future split modes

Today the runtime splits a model by layers, and each part of a layer runs whole on one device, the one whose backend its `Step` holds.
[MULTI-DEVICE](MULTI-DEVICE.md) plans more ways to split a model:
- the layer split, which runs now;
- the head split of layer split phase 3's step 6, an option that divides the output projection's vocabulary rows over the stages;
- tensor groups in phase 6, where every layer runs on several devices at once: q, k, v, gate and up split by output rows, the attention output and down by input columns, attention by heads with each member keeping the KV of its heads, and two sums over the group per layer;
- staged tensor, stages of a layer split each of which is a tensor group;
- expert parallelism, where a layer's experts are divided between devices.

Each reaches a module as fields of the contract, not as a rewrite of it:
- a `Role` says how its tensor shards: by output rows, by input columns, or not at all;
- a `Step` carries the group member it runs on, the group's width and the range of heads that member holds, and a group-sum hook the parts call where a partial result is summed over the group, which is the identity at width 1.

So once the contract gives a module shard information, its math assumes neither a whole matrix nor all heads: it takes a projection's widths from the `Weight` it is handed, runs over the heads its `Step` names, and calls the group sum where a product split by input columns ends.
The head split lands those fields in their general form as their first user, and the tensor group adds the group sum.

## Rules

### Batch invariance

A row's arithmetic depends only on the row, its prompt's extent and its position. It never depends on the width of the pass, the other sequences in it, how a prompt is sliced into passes, or a reused prefix.
- Kernels are chosen by the extent in `backend::RowRun`, never by a call's row count.
- Anything a pass carries to the next, such as a recurrent state or a partial group, is stored exactly, in F32, so where a slice ends cannot round it.
- Kernel layouts (lanes, reduction trees, chunk sizes) are constants of the model and the device, never of the token or sequence count.
- Every multiply-add in a new kernel is an explicit fused multiply-add, and the accumulators whose order matters are `precise`, as [QWEN35](QWEN35.md), Row classes, sets out.

So the same prompt gives the same bytes through `generate`, `chat` and `perplexity`, through the server alone or beside other requests, and on a layer split.

### Exactness classes and reuse

- When rows of one architecture compute in more than one way (decode rows and prompt rows, prompt rows by extent, a per-token and a chunked recurrence), the design page defines the classes.
- A cached row or state is reused only when it was computed in the class the CLI would compute it in.
- A module whose layers keep a state that exists only at the end of what it has read declares so, through a capability flag that arrives with Qwen 3.x step 4. The runtime then:
  - refuses forks, prefix reuse, and truncation to anything but 0 or the current length, until checkpoints serve them;
  - marks the states lost when a pass fails.

  `serve` refuses the model until its scheduler handles the state.

### Refusals

What is not implemented is refused, never approximated. That covers an architecture name, a key that changes the math, a tensor that does not fit its role, a type no kernel of the chosen backend reads, and a flag or placement that does not apply.
- A refusal names what it refused: the key, the tensor, the type or the flag.
- `tests/arch_<name>.cpp` holds each refusal to its text.
- A file refused today stays refused by name until the branch that implements it.

### One owner

| Rule | Owner |
|---|---|
| which module runs a file | the registry |
| a metadata value's type and range checks | `src/model/arch/metadata.hpp` |
| the keys, their defaults and refusals | the module's reader |
| the tensors, their roles and shapes, the layer kinds | the module's plan |
| a missing or repeated tensor | the runtime's `TensorIndex` |
| a role's check by its kind; one buffer per tensor per device | the runtime |
| what a role costs a device, and the fit | the runtime's footprint over the plan, and `src/model/layer_split.hpp` |
| which roles follow a routed block to the host, which are copied beside the mixer and which are streamed | declared by the plan, done by the runtime |
| slot widths, the residual width and the context length | the module |
| the arena, the handoff buffers and the crossings | the runtime |
| the cache of each layer | the module declares it; the storages, pools, blocks, forks and rollback are the runtime's and the backends' |
| table values | the module; their adoption and the position bound are the runtime's |
| the math | the module |
| the kernels | the backends |

When a second module needs a generic graph piece that another already has, the piece moves to a header of shared graph pieces under `src/model/arch/`, which that second module creates. It takes weights and slots as arguments and names no tensor and no key.

### Files as they are stored

- A GGUF file's weights load as the file stores them, with no byte transform, so the loader streams them in file order.
- A format that needs one (an HF directory's stacked experts, widening, a named bit-exact repack) declares it as parts of the view, in the format's reader. The loader executes it, and the module's math never does.
- A value an architecture derives from its weights (a folded constant, a product of two tables) is computed at load, exactly, by a named function in the module, declared in the plan, and counted in the fit.

### Nothing allocated in a pass

A part allocates nothing. Every buffer a pass uses (arena slots, expert windows, states, handoff buffers) is sized by the runtime from the plan before the pass records work: at load, when a context grows, or by `reserve_passes`.

## The order of the work

1. The design page and the STATUS plan block, saying where each piece lives, before any code.
2. The text path when it differs: the pretokenizer, and the chat template's constructs, each with its own gate.
3. The configuration, the plan and the refusals, with `tests/arch_<name>.cpp`. The plan refuses the architecture by name until its math runs.
4. The CPU ops the math needs, each with a unit test against a reference written from the math.
5. The math on the CPU, gated on the tiny fixture, then on the smallest released model that fits the host.
6. The Vulkan ops, then device identity and speed on the platforms where the model fits.
7. Serving, when the architecture needs more of the scheduler than dense models do.

## Gates

1. **Unit tests** (`arch-<name>`): every refused key and tensor with its text, the plan of each layer kind, tied and untied heads, and the footprint.
2. **The tiny fixture.**
   - It is a random-weight model of the real architecture, built by the reference STATUS names for it. That is HF's own modeling code in the pinned reference environment where HF has the architecture. Any other reference is a gate revision the maintainer approves.
   - It covers every layer kind, dense and routed layers, tied and untied heads, and any extra block, whose presence must not change the logits.
   - It holds 2e-5 per logit and 1e-5 NLL on the CPU across ubatches, batch widths, threads, decode after prefill, scoring and `--then-ids`. Where a layer carries state, the goldens come from the reference's token-by-token cached path.
   - It runs in the hosted CI.
3. **Released models.**
   - The smallest released member of the family that fits the host is checked against the reference: tokenizer ids, top-1 and top-5, perplexity windows, the chat render and its ids. It is pinned in `tests/data/fixtures.json` and run by the hosted HF job.
   - Larger members are checked against the reference's own layers run one at a time, where the whole model does not fit the host.
   - When no member fits, llmx's CPU path on the real files is the reference for the device and split gates.
   - A gate moves to a smaller model only with the maintainer's agreement.
4. **Devices.**
   - The same bounds hold on one MI50 of the Linux machine and on the Radeon VII under Windows, wherever the model fits them.
   - Slice invariance holds bitwise on the device.
   - `llmx-split-check` is bit-identical to one card on two and three MI50s. For a model no single card holds, two layer shares give identical bytes, and CPU stages are held to one CPU.
   - The 16k long-context check passes.
   - Once `serve` runs the model, `tools/server_mix_check.py` runs, with every request equal to itself alone and to the CLI.
5. **Speed.**
   - Every gate cell (pp64, pp247, pp512, pp4096, tg32, tg128 and pp16384/tg512) is at or above the reference STATUS names for the architecture, on the same card and file.
   - That reference is llama.cpp's Vulkan backend on the MI50 and the Radeon VII where that build runs the model on the device, and the reference build STATUS names otherwise.
   - The reference's numbers are reported beside llmx's, and serving percentiles against the references STATUS names.
6. **Everything else unchanged.** Every existing architecture's `generate`, `logits`, `perplexity` and `chat` is byte-identical to main on the CPU, one MI50 and the Radeon VII, and level with main in speed.
7. **CI.** CTest, the Python suites, the boundary check, and a green hosted run on the branch's `gate/<name>` push.
