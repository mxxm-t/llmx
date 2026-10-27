#pragma once
#include <optional>
#include <unordered_map>
#include <vector>
#include "format/gguf.hpp"

namespace gguf {

// NVFP4's optional F32 weight scales, indexed beside the file's tensors; input scales belong to activation quantization, not weight decoding.
inline std::vector<std::optional<size_t>> weight_scales(const GGUFModel& model, bool supported = true) {
    std::unordered_map<std::string, size_t> names;
    for (size_t i = 0; i < model.tensors.size(); ++i) names.emplace(model.tensors[i].name, i);
    std::vector<std::optional<size_t>> scales(model.tensors.size());
    for (size_t i = 0; i < model.tensors.size(); ++i) {
        const TensorInfo& weight = model.tensors[i];
        if (weight.type != quant::GGML_TYPE_NVFP4 || weight.name.size() < 7 ||
            weight.name.compare(weight.name.size() - 7, 7, ".weight") != 0) continue;
        const auto found = names.find(weight.name.substr(0, weight.name.size() - 7) + ".scale");
        if (found != names.end()) {
            if (!supported) throw std::runtime_error("inference: weight scales are not supported for tensor " + weight.name);
            scales[i] = found->second;
        }
    }
    return scales;
}

} // namespace gguf
