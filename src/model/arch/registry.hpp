#pragma once
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include "format/gguf.hpp"
#include "format/weight_scales.hpp"
#include "model/weights.hpp"
#include "model/architecture.hpp"
#include "model/arch/qwen3.hpp"
#include "model/arch/qwen35.hpp"
#include "model/arch/qwen4exp.hpp"
#include "model/arch/qwen3vl_vision.hpp"
#include "model/encoder.hpp"

// The architectures llmx runs, by the name a file gives: the one place such a name is read and accepted, and the one step from a file's metadata to its architecture.

namespace infer {

// A general.architecture value and the reader of its files, which reads the configuration under the name's prefix.
struct ArchEntry {
    const char* name;
    std::shared_ptr<const Architecture> (*read)(const gguf::GGUFModel&, const std::string& prefix);
    backend::Dtype dtype;
    // A same-architecture sidecar's header adaptation, called by the pairing owner after tokenizer equality; null preserves the ordinary contract.
    void (*normalize_drafter)(const gguf::GGUFModel& target, gguf::GGUFModel& side) = nullptr;
};

// GGUF has no source dtype; these are architecture defaults, not declarations inferred from weight storage.
inline const ArchEntry kArchitectures[] = {
    {"qwen3", qwen3::open_dense, backend::Dtype::bf16},
    {"qwen3moe", qwen3::open_routed, backend::Dtype::bf16},
    {"qwen35", qwen35::open_dense, backend::Dtype::bf16},
    {"qwen35moe", qwen35::open_routed, backend::Dtype::bf16},
    {"qwen4exp", qwen4exp::open, backend::Dtype::bf16, qwen4exp::normalize_drafter},
};

// The entry a file's general.architecture names; a file without the key is read as qwen3, since the tests' fixtures write none.
inline const ArchEntry& architecture_of(const gguf::GGUFModel& m) {
    const auto* a = m.find("general.architecture");
    const std::string name = a ? (a->vtype == gguf::V_STRING ? a->s : std::string()) : "qwen3";
    for (const ArchEntry& e : kArchitectures)
        if (name == e.name) return e;
    throw std::runtime_error("inference: unsupported metadata general.architecture");
}

// A GGUF model's weights: its architecture, whose configuration is read once, and a view of every tensor, whose data is null while the tensor's file is not mapped (gguf::map_payload).
// A tensor table whose storage count does not match its tensors, with a rank above four, or an offset or extent outside the payload is refused, which includes a payload its owner released; read_gguf has refused duplicate names already.
inline ModelWeights gguf_weights(const gguf::GGUFModel& m) {
    const ArchEntry& entry = architecture_of(m);
    ModelWeights w;
    w.arch = entry.read(m, std::string(entry.name) + ".");
    w.declared_dtype = entry.dtype;
    if (m.offsets.size() != m.tensors.size())
        throw std::runtime_error("inference: tensor storage count mismatch");
    w.tensors.reserve(m.tensors.size());
    const auto scales = gguf::weight_scales(m);
    for (size_t i = 0; i < m.tensors.size(); i++) {
        const auto& t = m.tensors[i];
        if (t.ne.size() > 4)
            throw std::runtime_error("inference: invalid tensor rank " + t.name);
        const uint64_t bytes = t.data_size();
        if (m.offsets[i] % alignof(float) || m.offsets[i] > m.payload_size() ||
            bytes > m.payload_size() - m.offsets[i])
            throw std::runtime_error("inference: invalid tensor storage " + t.name);
        w.tensors.push_back({t.name, t.ne, t.type, m.tensor_data(i), (size_t)bytes});
        w.tensors.back().scale = scales[i];
    }
    return w;
}

// The weights are views into the model, so a temporary model, const or not, would leave them dangling.
ModelWeights gguf_weights(const gguf::GGUFModel&&) = delete;

// A vision encoder llmx runs, by the general.architecture and clip.projector_type an mmproj file gives, and the reader of its files (docs/VISION.md, section 9).
struct VisionEntry {
    const char* architecture;
    const char* projector;
    std::shared_ptr<const VisionArchitecture> (*read)(const gguf::GGUFModel&);
    backend::Dtype dtype;
};

inline const VisionEntry kVisionEncoders[] = {
    {"clip", "qwen3vl_merger", qwen3vl_vision::open, backend::Dtype::bf16},
};

// The entry an mmproj file names; another architecture or projector is refused by name.
inline const VisionEntry& vision_of(const gguf::GGUFModel& m) {
    const auto* a = m.find("general.architecture");
    const auto* p = m.find("clip.projector_type");
    const std::string arch = a && a->vtype == gguf::V_STRING ? a->s : std::string();
    const std::string projector = p && p->vtype == gguf::V_STRING ? p->s : std::string();
    for (const VisionEntry& e : kVisionEncoders)
        if (arch == e.architecture && projector == e.projector) return e;
    if (arch != "clip") throw std::runtime_error("inference: unsupported vision encoder general.architecture");
    throw std::runtime_error("inference: unsupported vision encoder clip.projector_type");
}

// An mmproj file's encoder module and a view of every tensor, whose data is null while the file is not mapped.
struct VisionWeights {
    std::shared_ptr<const VisionArchitecture> arch;
    std::vector<TensorView> tensors;
    backend::Dtype declared_dtype = backend::Dtype::bf16;
};

inline VisionWeights vision_weights(const gguf::GGUFModel& m) {
    const VisionEntry& entry = vision_of(m);
    gguf::weight_scales(m, false);
    VisionWeights w;
    w.arch = entry.read(m);
    w.declared_dtype = entry.dtype;
    if (m.offsets.size() != m.tensors.size()) throw std::runtime_error("inference: tensor storage count mismatch");
    for (size_t i = 0; i < m.tensors.size(); ++i) {
        const auto& t = m.tensors[i];
        if (t.ne.size() > 4) throw std::runtime_error("inference: invalid tensor rank " + t.name);
        const uint64_t bytes = t.data_size();
        if (m.offsets[i] % alignof(float) || m.offsets[i] > m.payload_size() || bytes > m.payload_size() - m.offsets[i])
            throw std::runtime_error("inference: invalid tensor storage " + t.name);
        w.tensors.push_back({t.name, t.ne, t.type, m.tensor_data(i), (size_t)bytes});
    }
    return w;
}

// The weights are views into the model, so a temporary model, const or not, would leave them dangling.
VisionWeights vision_weights(const gguf::GGUFModel&&) = delete;

// The shape of the model llmx bench times without a file.
struct SyntheticShape {
    int n_layer, n_embd, n_ff, n_head, n_head_kv, head_dim, n_vocab;
    uint32_t seed;
};

// That model with random weights, Q8_0 matrices and F32 norms, written as a file of the first entry, qwen3, which names it.
inline gguf::GGUFModel synthetic_model(const SyntheticShape& s) {
    return qwen3::synthetic_model(kArchitectures[0].name, s.n_layer, s.n_embd, s.n_ff, s.n_head, s.n_head_kv, s.head_dim, s.n_vocab, s.seed);
}

} // namespace infer
