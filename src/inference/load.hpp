#pragma once
#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/host_memory.hpp"
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

namespace detail {

// Read the payload's pages in before the model is placed, reporting its bytes to `progress`.
// Pages read in stay resident only while the host can hold them: a payload larger than the memory available would be evicted before a device copies it and read from disk twice, so it is left to be read once by whoever reads it, and the progress goes straight to complete.
inline void warm(const gguf::GGUFModel& file, const format::LoadProgress& progress) {
    const size_t bytes = gguf::bytes_of(file);
    const auto available = core::host_memory_available();
    if (!available || bytes <= *available) return gguf::warm(file, progress);
    if (progress) {
        progress(0, bytes);
        progress(bytes, bytes);
    }
}

} // namespace detail

// Load the model at `path`, a GGUF file or the first shard of a set, over the caller's `backends` as `request` places it: read and map the file, reporting its payload to `progress`, build the tokenizer and the chat format, and place the model.
// The host's copy of the weights is then released when no host reads one in place, and otherwise the pages of every tensor no host reads leave its working set.
inline std::unique_ptr<LoadedModel> load_model(const std::string& path, std::vector<backend::BackendPtr> backends,
                                               const PlacementRequest& request, const ModelOptions& options = {},
                                               const format::LoadProgress& progress = {}) {
    auto loaded = std::make_unique<LoadedModel>();
    loaded->file = gguf::read_gguf(path);
    gguf::map_payload(loaded->file);
    detail::warm(loaded->file, progress);
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
