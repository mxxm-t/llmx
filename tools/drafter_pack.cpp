// A model's MTP blocks moved between its file and a drafter file beside it (docs/SPECULATIVE.md, step 6), every tensor's bytes copied unchanged.
// `split` writes the model without its MTP blocks, its block count and per-block arrays cut to its layers and its MTP count dropped, and the blocks as a drafter file that holds the model's metadata and only their tensors; the two pair (infer::spec::pair).
// `embed` writes a model with a drafter file's MTP blocks joined to it, as `--drafter FILE` loads them (infer::spec::join_blocks).
// Usage: llmx-drafter-pack split <model.gguf> <model-out.gguf> <drafter-out.gguf> | embed <model.gguf> <drafter.gguf> <out.gguf>; it exits 2 on a usage error and 1 on a refusal.
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "format/gguf.hpp"
#include "inference/pair.hpp"

namespace {

void split(const std::string& in, const std::string& model_out, const std::string& drafter_out) {
    gguf::GGUFModel m = gguf::read_gguf(in);
    const std::string prefix = std::string(infer::architecture_of(m).name) + ".";
    const int blocks = infer::metadata::integer(m, prefix + "block_count");
    const int nextn = infer::metadata::count(m, prefix + "nextn_predict_layers", 0);
    if (!nextn) throw std::runtime_error(in + " carries no MTP block (" + prefix + "nextn_predict_layers is 0)");
    const size_t layers = size_t(blocks - nextn);
    gguf::map_payload(m);
    gguf::GGUFModel model, drafter;
    std::vector<size_t> model_of, drafter_of;   // each written tensor's index in `in`
    drafter.kv = m.kv;
    for (const auto& kv : m.kv) {
        if (kv.first == prefix + "nextn_predict_layers") continue;
        gguf::MetaValue v = kv.second;
        if (kv.first == prefix + "block_count") v.u = layers, v.i = int64_t(layers);
        // A per-block array keeps its layers' entries.
        if (!kv.first.compare(0, prefix.size(), prefix) && v.vtype == gguf::V_ARRAY && v.arr.size() == size_t(blocks)) v.arr.resize(layers);
        model.kv.push_back({kv.first, std::move(v)});
    }
    for (size_t i = 0; i < m.tensors.size(); ++i) {
        const auto b = gguf::block_of(m.tensors[i].name);
        const bool draft = b && *b >= layers;
        (draft ? drafter : model).tensors.push_back(m.tensors[i]);
        (draft ? drafter_of : model_of).push_back(i);
    }
    infer::spec::pair(model, model_out, drafter, drafter_out);
    gguf::write_gguf(model, model_out, [&](size_t i) { return m.tensor_data(model_of[i]); });
    gguf::write_gguf(drafter, drafter_out, [&](size_t i) { return m.tensor_data(drafter_of[i]); });
}

void embed(const std::string& model_path, const std::string& drafter_path, const std::string& out) {
    gguf::GGUFModel model = gguf::read_gguf(model_path);
    gguf::GGUFModel drafter = gguf::read_gguf(drafter_path);
    if (infer::spec::pair(model, model_path, drafter, drafter_path) != infer::spec::DrafterKind::mtp)
        throw std::runtime_error(drafter_path + " holds no MTP blocks of " + model_path + " to embed");
    infer::spec::join_blocks(model, std::move(drafter));
    gguf::map_payload(model);
    gguf::write_gguf(model, out);
}

} // namespace

int main(int argc, char** argv) {
    const std::string cmd = argc > 1 ? argv[1] : "";
    if (argc != 5 || (cmd != "split" && cmd != "embed")) {
        std::fprintf(stderr, "usage: llmx-drafter-pack split <model.gguf> <model-out.gguf> <drafter-out.gguf> | embed <model.gguf> <drafter.gguf> <out.gguf>\n");
        return 2;
    }
    try {
        if (cmd == "split") split(argv[2], argv[3], argv[4]);
        else embed(argv[2], argv[3], argv[4]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "llmx-drafter-pack: %s\n", e.what());
        return 1;
    }
    return 0;
}
