#pragma once
#include <algorithm>
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
// A host backend's buffers read the file's payload in place while the model lives, so load_model builds this in place and it is never copied or moved; `model` is declared last, so it is destroyed first.
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

// The adoption hook load_model builds the model with: each weight is adopted inline, and host_reads[i] is set when a backend that reads in place took tensor i.
// It sizes host_reads to the tensors before the model is built, so recording cannot fail while a buffer is held.
inline AdoptWeight recording_adopt(const QwenWeights& weights, std::vector<char>& host_reads) {
    host_reads.assign(weights.tensors.size(), 0);
    return [&weights, &host_reads](size_t i, backend::Backend& b) {
        backend::BufferPtr buffer = b.adopt(weights.tensors[i].data, weights.tensors[i].bytes);
        if (b.reads_in_place()) host_reads[i] = 1;
        return buffer;
    };
}

// Load the model at `path`, a GGUF file or the first shard of a set, over `backends` as `request` places it, with `options`' caches.
// It reads the file, reporting the payload to `progress`, builds the tokenizer and the chat format, and places the model, recording which weights a host reads in place.
// It then releases the host's copy of the weights when no host reads one, and otherwise lets the pages of every tensor no host reads leave the host's working set.
// The caller makes the backends, so a device that cannot be opened fails before the file is read.
inline std::unique_ptr<LoadedModel> load_model(const std::string& path, std::vector<backend::BackendPtr> backends,
                                               const PlacementRequest& request, const ModelOptions& options = {},
                                               const format::LoadProgress& progress = {}) {
    auto loaded = std::make_unique<LoadedModel>();
    loaded->file = gguf::read_gguf(path, progress);
    loaded->tok.emplace(loaded->file);
    loaded->chat = chat::chat_format(loaded->file, *loaded->tok);
    const QwenWeights weights = gguf_weights(loaded->file);
    std::vector<char> host_reads;   // per tensor, whether a backend that reads in place took it
    PlacedModel placed = place_model(weights, std::move(backends), request, options, recording_adopt(weights, host_reads));
    loaded->plan = std::move(placed.plan);
    loaded->model = std::move(placed.model);
    // A backend that copies has consumed its weights when adopt returns: with no host reading one the host's copy goes, and otherwise the tensors no host reads leave its working set.
    if (std::find(host_reads.begin(), host_reads.end(), 1) == host_reads.end()) {
        loaded->file.release_payload();
    } else {
        for (size_t i = 0; i < host_reads.size(); ++i)
            if (!host_reads[i]) loaded->file.drop_pages(i);
    }
    return loaded;
}

} // namespace infer
