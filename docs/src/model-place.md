# `src/model/place.hpp` - placing a model over its devices

`resolve_dtype` chooses one automatic activation policy from the declared dtype and the common native policies of all participating backends. A declared F32 model takes the preferred common policy rather than pinning F32. The CPU and Vulkan backends prefer F16 then F32, so their common auto choice is F16, including for the current Qwen models that declare BF16. Emulated policies do not enter that choice. `place_model` applies the result to `ModelOptions` before construction and returns its `DtypePlan`; an added expert host is created once and included in the same resolution. The plan describes device implementations, not an executed-path witness. An optional `PlacementRequest::dtype` overrides auto. Each device keeps the requested dtype natively, emulates it when supported, or falls back to F32; one fallback does not change another device. `DtypePlan` records each effective dtype and `ModelOptions::device_dtypes` carries those choices into execution. Emulation and fallback produce a warning in the one CLI record.

Where a model runs, in namespace `infer`: what it asks of each device's
memory, counted from its plan, and the one place a model is placed over the
backends its caller made. It reads the plan
([architecture](model-architecture.md)) and builds the runtime's `Model`
([runtime](model-runtime.md)), and names no architecture.

- `footprint(weights, plan, options)`: what a model asks of memory, for a split fitted to devices (`model/layer_split.hpp`), counted from its plan: each layer's matrices are the tensors its roles take, in the file's order and each once, marked as products where a role reads them as a matrix, whatever their rank, so a tensor no role takes costs nothing; the embedding is the embed part's table, the output the head's matrix and the output norm the head's norm, and the head is tied when its role took the alias, its output then the embedding counted as a product; a pass role of any other part and kind, or a second role for one of those three, throws `std::logic_error`, so a new kind of pass role extends `Footprint` rather than fitting a split it overruns; it reads each role's tensor as `plan_model` set it and looks no name up; each layer's cache is counted by its kind, a KV layer's the budgeted positions at the plan's K and V geometry and the options' cache types, a state layer's the plan's state for each of the options' state, checkpoint and mark slots with each mark's saved inputs (`ModelOptions::mark_rows` rows of `saved_floats`), and another's none, the tables the plan's, a row of the arena the plan's slots and a row of the residual stream handed between devices the plan's residual; with an embedded drafter, its weights but those the head already takes, its embedding table apart, and its KV layer at the budget, a carried row for each state slot and each mark's rows. `placement_for(split)` turns a `LayerSplit` into a `Placement`.
- `place_model(weights, backends, request, options, adopt = {})`: the one place a model is placed over the backends its caller made, returning the model with the request's ubatch set and, for a split, its plan (`LayerSplit::describe`). It plans the weights once (`plan_model`), and the fit, the experts placement and the model it builds read that plan. A request that names `histories` of `history_tokens` each, as `bench --model` does its sequences, has them counted in whole blocks of each backend, each up to the model's context, which no run passes: where the options' budget would leave any storage short, the budget becomes what they take in the largest blocks, which the fit and every storage then use, and otherwise it is unchanged. With several backends or layer shares it fits the split for `request.ubatch` (default `kDefaultUbatch`) plus `request.decode_rows` rows and `request.slots` pass slots, whose handoff buffers the host holds (`budgets_for`, `split_layers`, `placement_for`); with one backend that is not the CPU (`Backend::is_cpu`) and `PlacementRequest::cpu_moe`, the CPU becomes device 0 beside it and the feed-forward blocks of the first `cpu_moe` layers the plan marks routed (every one at -1) run there, with `stream_from` as the placement's; otherwise the model is on the one backend, the CPU with its experts on it included. Experts on the CPU on a model whose plan marks no layer routed are refused whatever the backends, the CPU included, and so are experts on the CPU with several devices, each refusal naming the flag the request stands for (`--cpu-moe` at -1, `--n-cpu-moe` otherwise); so is a nonzero `stream_from` without experts on the CPU, which would have nothing to stream. `adds_host_for_experts(backends, request)` is the rule for when it adds the CPU for experts, which asks whether the backend is the CPU and not whether it reads its weights in place, since a device may read host memory in place, and `host_reads_in_place(backends, request)` says whether any backend of the placement reads weights in place, one of `backends` or that CPU, which the loader asks before it decides what to map. The CLI, `bench --model`, the server and `llmx-split-check` all build their model through it by way of `infer::load_model` ([load](inference-load.md)), as does `compare_cpu`. A routed layer's experts are streamed to the device only where its mixer runs on a backend that copies its weights and the experts on one that reads them in place (`Backend::reads_in_place`). The runtime also requires that destination to support every streamed weight type and to implement every op of the feed-forward part; otherwise the layer stays on its host, with the decision made once before adoption (`Backend::supports_type`, `Backend::implements`, [runtime](model-runtime.md)).
- `PlacementRequest`, `PlacedModel`: how a caller wants a model placed
  (the backends' names, layer shares, experts on the CPU and their stream
  point, the ubatch, the decode rows, pass slots and logits rows a split is
  fitted for,
  and the histories the budget must hold, `fit_kv`, a server's budget
  fitted to the devices, and `fit_checkpoints`, its checkpoint slots
  fitted too, `fit_marks`, its mark slots past the first fitted after both,
  and `drafter`, the file's embedded drafter planned and loaded
  with the model), and the placed model with its split's description and the
  KV tokens its fitted checkpoint slots took from the budget.
- `settle(budgets, backends, names, settled, level = false)`: a fit the devices' free
  memory falls short of, a split's placement or a server's KV budget, tried
  again every `kSettleWait` (250 ms), up to `kSettleReads` (120) reads, each
  time that memory rises, until `kSettleQuiet` (20) reads in a row, five
  seconds, find it no higher, since a device gives an ended process's memory
  back in steps over a few seconds, holding it level for seconds between them.
  Over several backends with a device that reports its free memory, where
  the layers or the KV budget follow that memory, no shares being given or
  the budget fitted (`level_first`), it reads until the memory has risen no
  further for those five seconds, or the thirty have passed, whatever the
  fit says, and asks the fit of that reading alone,
  since there a fit that holds does not show a card has given its memory
  back: the other devices would take the layers it would hold.
- `fitted_kv(weights, plan, backends, request, options, given_up = nullptr, read = nullptr)`: the options with
  the KV budget fitted, which `place_model` takes when `request.fit_kv` is
  set, as `serve` sets it: the options' budget where the fit of
  `split_layers` places the model on the backends given, one included, and
  the host; else, by bisection, the most whole blocks of the largest block
  size that fit, one more not fitting. A budget short of the options' is
  fitted again as `settle` reads the devices. With `fit_checkpoints`, on a
  model that keeps a state, it then takes as checkpoint slots the fewer of
  the options' and the most at which the fit still holds three quarters
  of the budget it holds without them, so they take at most a quarter of
  the KV room, each count tried through the fit itself, so a device that
  cannot hold one more slot gives none, and only once the budget without
  them has settled, so a card still taking back an ended server's memory
  does not leave a restarted server none (`docs/SPECULATIVE.md`, section 2);
  the KV tokens they took go to `given_up` when given, and the devices'
  budgets it settled on to `read`, by which `place_model` then places a
  split's layers, so the budget and the split see one reading. The budget
  and the checkpoints are fitted without drafting: the plan without its
  embedded drafter (`plan_model` without one) and no mark slot; the
  drafter and a mark, or every mark asked for without `fit_marks`, must
  then fit beside that budget, which stays as it is: automatic checkpoint
  slots (`fit_checkpoints`) give way, the most of them that still fit, and
  checkpoints asked for by number do not, so it refuses, naming the
  budget and the checkpoints that left no room (docs/SPECULATIVE.md,
  section 3). With
  `fit_marks`, on a model that keeps a state, it then takes as mark slots
  the most, up to the options', at which that budget and those
  checkpoints still fit, so marks past the first take only the room left
  over. It refuses a model that does not fit even without
  its KV, and one that leaves no room for one block, each by its own text.
  Beside experts on the CPU the plan it fits leaves out the feed-forward
  roles of the layers whose block runs on the CPU (`ffn_on_host`, which
  `place_model` also places them by), since the device holds neither those
  weights nor their experts. It sets `ModelOptions::kv_backed`, so the
  storages back the whole budget as the model is made. Dtype resolution follows the fit and preserves that backing option; an added expert host is created once and used both to resolve its dtype and to construct the model.
