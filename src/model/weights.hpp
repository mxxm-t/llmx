#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "quant/types.hpp"
#include "backends/backend.hpp"

// What a model is built from, whatever file it came from, and the weights it resolves: the part of the architecture contract (model/architecture.hpp) that the loader and a file's reader also use.

namespace infer {

class Architecture;

// One tensor as a reader hands it to the model: its name, its dimensions with the fastest first, its storage type (the GGUF type id), and its bytes, where `data` is null when they are not in memory.
struct TensorView {
    std::string name;
    std::vector<uint64_t> shape;   // fastest dimension first
    uint32_t type = quant::GGML_TYPE_F32;
    const uint8_t* data = nullptr;
    size_t bytes = 0;
};

// What a model is built from, whatever file it came from: the architecture that reads it and one view per tensor in the file's order, so tensor i is the file's tensor i, each under a unique name.
// The model reads the views only while it is built; a streamed load reads the bytes again afterwards to fill what a copying backend took, and the bytes a backend adopted in place are read for as long as its buffer lives.
// Two models built from one set of weights share the architecture and nothing else.
struct ModelWeights {
    std::shared_ptr<const Architecture> arch;
    std::vector<TensorView> tensors;
};

// The tensors a model is built from, by name, which plan_model builds once over the views for the architecture's plan and for setting each role's tensor.
// It refuses a repeated name, and owns the one text for an absent one; it lives no longer than the views it refers to.
class TensorIndex {
public:
    explicit TensorIndex(const std::vector<TensorView>& tensors) : tensors_(&tensors) {
        index_.reserve(tensors.size());
        for (size_t i = 0; i < tensors.size(); ++i)
            if (!index_.emplace(tensors[i].name, i).second)
                throw std::runtime_error("inference: duplicate tensor " + tensors[i].name);
    }
    std::optional<size_t> find(const std::string& name) const {
        const auto it = index_.find(name);
        if (it == index_.end()) return std::nullopt;
        return it->second;
    }
    size_t at(const std::string& name) const {
        const auto it = index_.find(name);
        if (it == index_.end()) throw missing(name);
        return it->second;
    }
    const TensorView& view(size_t i) const { return (*tensors_)[i]; }
    // The refusal of a tensor the views lack, which the model's resolution also gives for a role without one.
    static std::runtime_error missing(const std::string& name) { return std::runtime_error("inference: missing tensor " + name); }

private:
    const std::vector<TensorView>* tensors_;
    std::unordered_map<std::string, size_t> index_;
};

// How the model's builder puts tensor `tensor` of its ModelWeights on backend `b`, returning the buffer the model reads.
// The model calls it once for each backend that hosts a weight's role; without one it calls b.adopt(view.data, view.bytes).
using AdoptWeight = std::function<backend::BufferPtr(size_t tensor, backend::Backend& b)>;

// A weight resolved once at load: type, storage and dimensions, which the forward pass reads from a row of resolved weights by role id rather than looking a tensor up by name.
// A device backend recognizes a weight across calls by it, which residency needs (docs/DEVICE-EXECUTION.md).
struct Weight {
    uint32_t type = 0;
    // A handle, not a pointer: the backend decides where the bytes live.
    // The model never dereferences it: slice() below names a location, and only a backend turns that into an address.
    backend::BufferPtr data;
    size_t nin = 0, nout = 0;
    // Normalization weights are F32 by validation, so this is the whole row.
    backend::CSlice slice() const { return {data.get(), 0}; }
};

} // namespace infer
