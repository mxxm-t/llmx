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
#include "model/weight_rows.hpp"

// What a model is built from, whatever file it came from, and the weights it resolves: the part of the architecture contract (model/architecture.hpp) that the loader and a file's reader also use.

namespace infer {

class Architecture;
namespace shard { struct Held; }

// One tensor as a reader hands it to the model: its name, its dimensions with the fastest first, its storage type (the GGUF type id), and its bytes, where `data` is null when they are not in memory.
struct TensorView {
    std::string name;
    std::vector<uint64_t> shape;   // fastest dimension first
    uint32_t type = quant::GGML_TYPE_F32;
    const uint8_t* data = nullptr;
    size_t bytes = 0;
    std::shared_ptr<const WeightRows> row_source = {};
    std::optional<size_t> scale = std::nullopt;
};

// What a model is built from, whatever file it came from: the architecture that reads it and one view per tensor in the file's order, each under a unique name; two models built from one set share the architecture and nothing else.
// The model reads the views only while it is built; a streamed load reads the bytes again afterwards to fill what a copying backend took, and the bytes a backend adopted in place are read for as long as its buffer lives.
struct ModelWeights {
    std::shared_ptr<const Architecture> arch;
    std::vector<TensorView> tensors;
    backend::Dtype declared_dtype = backend::Dtype::f32;
};

// A companion holds one F32 scale for the weight or one per expert, with the whole vector retained on every tensor shard.
inline size_t weight_scale_count(const ModelWeights& weights, size_t tensor) {
    const TensorView& weight = weights.tensors.at(tensor);
    if (!weight.scale) return 0;
    const auto refuse = [&]() { return std::runtime_error("inference: incompatible weight scale " + weight.name); };
    if (weight.type != quant::GGML_TYPE_NVFP4 || *weight.scale >= weights.tensors.size()) throw refuse();
    const TensorView& scale = weights.tensors[*weight.scale];
    if (scale.type != quant::GGML_TYPE_F32 || scale.shape.empty() || scale.shape.size() > 4 || scale.row_source || scale.scale) throw refuse();
    size_t count = 1;
    for (uint64_t n : scale.shape) {
        if (!n || n > SIZE_MAX / count) throw refuse();
        count *= size_t(n);
    }
    const uint64_t experts = weight.shape.size() == 3 ? weight.shape[2] : 1;
    if ((count != 1 && count != experts) || count > SIZE_MAX / sizeof(float) || scale.bytes != count * sizeof(float)) throw refuse();
    return count;
}

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

// How the model's builder puts tensor `tensor` of its ModelWeights on backend `b`, returning the buffer the model reads: the whole tensor where `shard` is null, or on a tensor group's member the copy `shard` describes (model/shard.hpp).
// The model calls it once for each backend that hosts a weight's role; without one it adopts the view, or a member's packed copy, which the model keeps.
using AdoptWeight = std::function<backend::BufferPtr(size_t tensor, backend::Backend& b, const shard::Held* shard)>;

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
    std::shared_ptr<const WeightRows> row_source = {};
    backend::BufferPtr scales = {};
    size_t scale_count = 0;
    backend::WeightView operand() const { return {type, slice(), {scales.get(), 0}, scale_count}; }
};

} // namespace infer
