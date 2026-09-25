#pragma once
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "format/gguf.hpp"
#include "tokenizer/tokenizer.hpp"
#include "inference/chat.hpp"
#include "model/arch_qwen.hpp"

// Loading a model file, the one sequence every command and tool opens a model through (docs/src/inference-load.md).

namespace infer {

// A model file loaded for use: the file, its tokenizer and chat format, and the model placed over the caller's backends.
// The model reads the file while it lives, so load_model builds this in place and it is never copied or moved; `model` is declared last, so it is destroyed first.
struct LoadedModel {
    gguf::GGUFModel file;
    std::optional<bpe::Tokenizer> tok;
    chat::ChatFormat chat;
    std::string plan;                // what each device of a split was given (LayerSplit::describe), empty otherwise
    std::unique_ptr<Model> model;

    LoadedModel() = default;
    LoadedModel(const LoadedModel&) = delete;
    LoadedModel& operator=(const LoadedModel&) = delete;
};

// Load the model at `path`, a GGUF file or the first shard of a set, over `backends` as `request` places it, with `options`' caches.
// It reads the file, reporting the payload to `progress`, builds the tokenizer and the chat format, places the model, and releases the host's copy of the weights when no weight reads it in place.
// The caller makes the backends, so a device that cannot be opened fails before the file is read.
inline std::unique_ptr<LoadedModel> load_model(const std::string& path, std::vector<backend::BackendPtr> backends,
                                               const PlacementRequest& request, const ModelOptions& options = {},
                                               const format::LoadProgress& progress = {}) {
    auto loaded = std::make_unique<LoadedModel>();
    loaded->file = gguf::read_gguf(path, progress);
    loaded->tok.emplace(loaded->file);
    loaded->chat = chat::chat_format(loaded->file, *loaded->tok);
    PlacedModel placed = place_model(loaded->file, std::move(backends), request, options);
    loaded->plan = std::move(placed.plan);
    loaded->model = std::move(placed.model);
    // A model on device backends alone copied every weight into device memory, so the host would otherwise hold the weights twice.
    if (!loaded->model->holds_payload()) loaded->file.release_payload();
    return loaded;
}

} // namespace infer
