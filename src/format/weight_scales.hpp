#pragma once
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include "core/bf16.hpp"
#include "core/fp16.hpp"
#include "format/gguf.hpp"

namespace gguf {

// The inputs one F8_E4M3 grid scale covers (docs/src/format-weight_scales.md).
constexpr uint64_t kFp8ScaleInputs = quant::F8_E4M3_SCALE_INPUTS;
// The composite kind (GGUFModel::Composite) whose bytes are an F8_E4M3 weight's grid, made by fp8_grid.
constexpr uint32_t kFp8GridComposite = 8;

// The row shift of the F8_E4M3 grid of shape `scale` beside a weight of shape `weight`, 0 for a scale a row and 7 for one per 128 rows, or none where the shapes do not pair.
inline std::optional<unsigned> fp8_scale_shift(const std::vector<uint64_t>& weight, const std::vector<uint64_t>& scale) {
    if (weight.size() < 2 || weight.size() > 3 || scale.size() != weight.size() || !weight[0] || !weight[1] || weight[0] % kFp8ScaleInputs ||
        scale[0] != weight[0] / kFp8ScaleInputs || (weight.size() == 3 && scale[2] != weight[2]))
        return std::nullopt;
    for (unsigned shift : {0u, 7u})
        if (scale[1] == ((weight[1] - 1) >> shift) + 1) return shift;
    return std::nullopt;
}

// The grid shape of an F8_E4M3 weight of shape `weight` at row shift `shift`.
inline std::vector<uint64_t> fp8_grid_shape(const std::vector<uint64_t>& weight, unsigned shift) {
    std::vector<uint64_t> ne = {weight[0] / kFp8ScaleInputs, ((weight[1] - 1) >> shift) + 1};
    if (weight.size() == 3) ne.push_back(weight[2]);
    return ne;
}

// Writes `count` F32 grid scales at `out` from `n` scales at `source` of type F32, F16 or BF16, each widened exactly and repeated over count / n consecutive scales: one a tensor, an expert or a row, or the grid itself.
inline void fp8_grid(const uint8_t* source, uint32_t type, size_t n, uint8_t* out, size_t count) {
    if (!n || count % n) throw std::logic_error("weight scales: a grid of " + std::to_string(count) + " from " + std::to_string(n) + " scales");
    const size_t per = count / n;
    for (size_t j = 0; j < count; ++j) {
        const size_t i = j / per;
        float v;
        if (type == quant::GGML_TYPE_F32) {
            std::memcpy(&v, source + 4 * i, 4);
        } else {
            uint16_t h;
            std::memcpy(&h, source + 2 * i, 2);
            v = type == quant::GGML_TYPE_BF16 ? bf16_to_f32(h) : f16_to_f32(h);
        }
        std::memcpy(out + 4 * j, &v, 4);
    }
}

// The optional F32 weight scales of NVFP4 and F8_E4M3 weights, `<name>.scale` beside `<name>.weight`, indexed beside the file's tensors; input scales belong to activation quantization, not weight decoding.
inline std::vector<std::optional<size_t>> weight_scales(const GGUFModel& model, bool supported = true) {
    std::unordered_map<std::string, size_t> names;
    for (size_t i = 0; i < model.tensors.size(); ++i) names.emplace(model.tensors[i].name, i);
    std::vector<std::optional<size_t>> scales(model.tensors.size());
    for (size_t i = 0; i < model.tensors.size(); ++i) {
        const TensorInfo& weight = model.tensors[i];
        if ((weight.type != quant::GGML_TYPE_NVFP4 && weight.type != quant::GGML_TYPE_F8_E4M3) || weight.name.size() < 7 ||
            weight.name.compare(weight.name.size() - 7, 7, ".weight") != 0) continue;
        const auto found = names.find(weight.name.substr(0, weight.name.size() - 7) + ".scale");
        if (found != names.end()) {
            if (!supported) throw std::runtime_error("inference: weight scales are not supported for tensor " + weight.name);
            scales[i] = found->second;
        } else if (weight.type == quant::GGML_TYPE_F8_E4M3) {
            throw std::runtime_error("inference: F8_E4M3 tensor " + weight.name + " has no .scale beside it");
        }
    }
    return scales;
}

// A GGUF file's F8_E4M3 scales, an F32 scalar or one an expert, each made the composite of its weight's grid at row shift 7 (fp8_grid).
// A scale of any other form is refused, since llmx reads no grid from a file (docs/src/format-weight_scales.md).
inline void fp8_grids(GGUFModel& m) {
    const auto scales = weight_scales(m);
    for (size_t i = 0; i < m.tensors.size(); ++i) {
        const TensorInfo& weight = m.tensors[i];
        if (!scales[i] || weight.type != quant::GGML_TYPE_F8_E4M3) continue;
        const size_t s = *scales[i];
        TensorInfo& scale = m.tensors[s];
        const uint64_t count = scale.n_elements(), experts = weight.ne.size() == 3 ? weight.ne[2] : 1;
        const std::vector<uint64_t> grid = fp8_grid_shape(weight.ne, 7);
        if (scale.type != quant::GGML_TYPE_F32 || scale.ne.empty() || (count != 1 && count != experts) || m.composites.count(s) ||
            !fp8_scale_shift(weight.ne, grid))
            throw std::runtime_error("inference: F8_E4M3 tensor " + weight.name + " needs an F32 .scale of one value or one an expert, and rows of whole 128-input blocks");
        GGUFModel::Composite c;
        c.kind = kFp8GridComposite;
        c.parts = {{scale, m.offsets[s]}};
        scale.ne = grid;
        m.composites[s] = std::move(c);
    }
}

} // namespace gguf
