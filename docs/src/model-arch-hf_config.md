# `src/model/arch/hf_config.hpp` - Hugging Face config.json and tensor names to an architecture's GGUF form

`infer::hf::apply(table, arch, config, model)` turns a Hugging Face directory's `config.json` and the tensor names `safetensors::read_directory` read ([safetensors](format-safetensors.md)) into the metadata and names the architecture's GGUF reader takes, so a directory is read by the same `read_config`, roles and loader as the GGUF file the converter writes from the same checkpoint.
The metadata it writes is `general.architecture` and the table's keys under the architecture's prefix, integers as UINT32 and reals as FLOAT32, as the converter stores them; the `general.*` provenance keys a converter adds are not written. Keys the model already holds, such as `quantization.input_activations` from `hf_quant`, follow them.

Each architecture module that reads directories has one `hf::Table` (`qwen3::hf_table`), which the registry entry names ([registry](model-arch-registry.md)):
- `model_type` and the one `architectures` class it reads; anything else is refused.
- `keys`: every `config.json` key the module accepts, each either stored (`Rule::integer`, `Rule::real`, under its GGUF key; one HF key may give several) or `Rule::rope_scaling`, `rope_scaling` as null for the unscaled rope or YaRN's object stored as the converter's `rope.scaling.*` keys (`rope_type` or `type` `yarn`, `factor`, `original_max_position_embeddings`, else `max_position_embeddings` as transformers takes it, `beta_fast`, `beta_slow`), refusing `attention_factor`, `mscale`, `mscale_all_dim` and any other key by name and leaving the rest to the rope owner ([rope](model-rope.md)), or checked and not stored (`Rule::check`) against the JSON values it allows, or any value for keys read elsewhere or with no effect on inference; `Rule::each` checks every entry of a list the same way, as qwen3's `layer_types` must be all `"full_attention"`, which configs from newer transformers releases write.
  A key the table does not list is refused by name, so a checkpoint the module does not run, such as one with scaled RoPE, attention biases or a sliding window, is never read as one it does.
  The qwen3 row for `pad_token_id` only accepts the key and stores nothing, since the tokenizer reader already takes `config.json`'s id as the fallback padding token id where the pad text names no token ([hf_tokenizer](tokenizer-hf_tokenizer.md)).
- `names`: each HF tensor name and the GGUF name it becomes, `{N}` standing for a layer number below the block count; a tensor no row names, or a layer past the count, is refused.
- `embedding` and `head`: the GGUF names of the token embedding and the head, which the tie and `vocab_size` checks below read, so the shared code names no tensor.
- `config` and `outer`: the object the keys sit in, when `config.json` nests them, as qwen35's `text_config`, and the top level's own keys beside it, checked by the same rules; `hf::keys_of` gives that object to the directory reader, which reads `vocab_size` and the token ids there.
  `tie_word_embeddings` given at both levels must agree, and given at the top level alone it ties the head, as HF's outer model reads it.
- `derive`: the keys no single value gives, written by the module (qwen35's `hf_derive`: the block count with the MTP layer and `nextn_predict_layers`, `ssm.inner_size`, `full_attention_interval` checked against `layer_types`, and the rotary keys of `rope_parameters`, whose other entries are refused by name).
- `k_heads`, `v_heads`, `k_width`: the GGUF keys of the linear attention's K heads, V heads and K width, which the tiled order below reads.

A name's `{N}` is a decoder layer, below the block count less `nextn_predict_layers`, and `{L}` in its GGUF name the MTP layer after them; a directory whose `config.json` gives no MTP layer refuses a tensor named for one.
An HF name ending in `*` matches every name it begins, and an empty GGUF name drops the tensor unread, as qwen35 drops the vision tower `model.visual.*`, since images come from an mmproj file; a dropped tensor that is quantized is refused.
A kept tensor of more than 4 dimensions is refused by name.

A name may carry a transform, which `apply` makes a composite (`gguf::GGUFModel::Composite`, kinds 16 to 19 beside hf_quant's) and `hf::transform` makes at load, called by the loader beside `hf_quant::repack` ([load](inference-load.md)):
- `plus_one`: F32 of 1 + w, one F32 addition after an exact widening, the value HF's `1.0 + weight.float()` computes; the qwen35 norms but `ssm_norm`.
- `neg_exp`: F32 of -exp(w), exp taken in double and rounded once, so every platform gives the same bits; qwen35's `ssm_a` from `A_log`.
  HF takes the exponential in float32 at run time and the converter in float32 as it writes, so a value may differ from either's by an ulp; the HF checks bound that.
- `widen`: F32 of w, with a 3-D tensor's two slower dimensions made one; qwen35's `ssm_conv1d`, whose HF form is [C, 1, 4].
- the V heads in tiled order ([QWEN35](../QWEN35.md), V heads in tiled order): `VHeads::rows`, `rows_after_qk` (the rows after the 2 Hk Dk of q and k) or `columns` name the rows or columns indexed by V head, and with Hv above Hk the composite's `order` moves whole heads, as `reorder` (the type kept, whole rows or runs of a head's columns copied at once) or with the transform; a 1-D tensor that only moves is widened as it moves, since the loader widens every other 16-bit vector.
  With Hv = Hk nothing moves and a linear stays a view of the file.

A transform refuses a tensor that is quantized or not F32, BF16, F16 or F8_E4M3, and an F8_E4M3 weight only moves (`reorder`, one byte an element), never takes an arithmetic transform.
An FP8 weight's scale grid (`hf_quant::fp8_grid`) skips the names pass's V-head order; once every weight has its order, `detail::follow_weights` gives each grid of a moved weight the order of its scale units (`unit_order`): a scale a row moves with its row, a block of 128 rows or 128 inputs moves whole, and one scale for the tensor stays, while a channel grid of a weight that moves by column stays.
`unit_order` refuses by name a weight whose V heads split a scale unit (Dv not a multiple of 128 under block scales, or q and k's 2 Hk Dk rows off a block edge) and keeps a partial last block last, so such a checkpoint is refused rather than regridded.

A tied head (`tie_word_embeddings: true`) stored beside the embedding as `lm_head.weight` is dropped, with a file it held alone (`GGUFModel::erase_tensors`), since the HF model ties the head to the embedding and ignores it; Qwen/Qwen3-0.6B stores one. A tied head that is a quantized linear (a composite, [hf_quant](format-hf_quant.md)) is refused by name instead.
An FP8 linear's scale grid, the composite `X.scale` beside `X.weight`, takes the name its weight's row gives, with `.scale` for `.weight`.
An untied head must be present, and `vocab_size` must equal the embedding's rows.
