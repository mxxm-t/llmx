# `src/model/arch/hf_config.hpp` - Hugging Face config.json and tensor names to an architecture's GGUF form

`infer::hf::apply(table, arch, config, model)` turns a Hugging Face directory's `config.json` and the tensor names `safetensors::read_directory` read ([safetensors](format-safetensors.md)) into the metadata and names the architecture's GGUF reader takes, so a directory is read by the same `read_config`, roles and loader as the GGUF file the converter writes from the same checkpoint.
The metadata it writes is `general.architecture` and the table's keys under the architecture's prefix, integers as UINT32 and reals as FLOAT32, as the converter stores them; the `general.*` provenance keys a converter adds are not written. Keys the model already holds, such as `quantization.input_activations` from `hf_quant`, follow them.

Each architecture module that reads directories has one `hf::Table` (`qwen3::hf_table`), which the registry entry names ([registry](model-arch-registry.md)):
- `model_type` and the one `architectures` class it reads; anything else is refused.
- `keys`: every `config.json` key the module accepts, each either stored (`Rule::integer`, `Rule::real`, under its GGUF key; one HF key may give several) or checked and not stored (`Rule::check`) against the JSON values it allows, or any value for keys read elsewhere or with no effect on inference; `Rule::each` checks every entry of a list the same way, as qwen3's `layer_types` must be all `"full_attention"`, which configs from newer transformers releases write.
  A key the table does not list is refused by name, so a checkpoint the module does not run, such as one with scaled RoPE, attention biases or a sliding window, is never read as one it does.
  The qwen3 row for `pad_token_id` only accepts the key and stores nothing, since the tokenizer reader already takes `config.json`'s id as the fallback padding token id where the pad text names no token ([hf_tokenizer](tokenizer-hf_tokenizer.md)).
- `names`: each HF tensor name and the GGUF name it becomes, `{N}` standing for a layer number below the block count; a tensor no row names, or a layer past the count, is refused.
- `embedding` and `head`: the GGUF names of the token embedding and the head, which the tie and `vocab_size` checks below read, so the shared code names no tensor.

A tied head (`tie_word_embeddings: true`) stored beside the embedding as `lm_head.weight` is dropped, with a file it held alone (`GGUFModel::erase_tensors`), since the HF model ties the head to the embedding and ignores it; Qwen/Qwen3-0.6B stores one. A tied head that is a quantized linear (a composite, [hf_quant](format-hf_quant.md)) is refused by name instead.
An untied head must be present, and `vocab_size` must equal the embedding's rows.
