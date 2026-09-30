#pragma once
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include "format/gguf.hpp"
#include "model/weights.hpp"
#include "model/architecture.hpp"
#include "model/arch/qwen3.hpp"
#include "model/arch/qwen35.hpp"

// The architectures llmx runs, by the name a file gives: the one place such a name is read and accepted, and the one step from a file's metadata to its architecture.

namespace infer {

// A general.architecture value and the reader of its files, which reads the configuration under the name's prefix.
struct ArchEntry {
    const char* name;
    std::shared_ptr<const Architecture> (*read)(const gguf::GGUFModel&, const std::string& prefix);
};

inline const ArchEntry kArchitectures[] = {
    {"qwen3", qwen3::open_dense},
    {"qwen3moe", qwen3::open_routed},
    {"qwen35", qwen35::open_dense},
    {"qwen35moe", qwen35::open_routed},
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
    if (m.offsets.size() != m.tensors.size())
        throw std::runtime_error("inference: tensor storage count mismatch");
    w.tensors.reserve(m.tensors.size());
    for (size_t i = 0; i < m.tensors.size(); i++) {
        const auto& t = m.tensors[i];
        if (t.ne.size() > 4)
            throw std::runtime_error("inference: invalid tensor rank " + t.name);
        const uint64_t bytes = t.data_size();
        if (m.offsets[i] % alignof(float) || m.offsets[i] > m.payload_size() ||
            bytes > m.payload_size() - m.offsets[i])
            throw std::runtime_error("inference: invalid tensor storage " + t.name);
        w.tensors.push_back({t.name, t.ne, t.type, m.tensor_data(i), (size_t)bytes});
    }
    return w;
}

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
