#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include "inference/load.hpp"
#include "loading_backend.hpp"

namespace {
void require(bool condition, const char* text) {
    if (!condition) throw std::runtime_error(text);
}

gguf::GGUFModel fixture(bool decoded, bool tied, bool moe = false) {
    const auto base = infer::synthetic_model({2, 128, 256, 4, 4, 32, 16, 593});
    gguf::GGUFModel out;
    out.kv = base.kv;
    if (moe) {
        for (auto& kv : out.kv) {
            if (kv.first.compare(0, 6, "qwen3.") == 0) kv.first.replace(0, 6, "qwen3moe.");
            if (kv.first == "general.architecture") kv.second.s = "qwen3moe";
        }
        for (const auto& entry : std::vector<std::pair<std::string, uint64_t>>{
                {"expert_count", 2}, {"expert_used_count", 2}, {"expert_feed_forward_length", 256}}) {
            gguf::MetaValue value;
            value.vtype = gguf::V_UINT32;
            value.u = entry.second;
            out.kv.push_back({"qwen3moe." + entry.first, value});
        }
    }
    for (size_t t = 0; t < base.tensors.size(); ++t) {
        auto info = base.tensors[t];
        if (tied && info.name == "output.weight") continue;
        if (moe && (info.name.find(".ffn_gate.") != std::string::npos ||
                    info.name.find(".ffn_up.") != std::string::npos || info.name.find(".ffn_down.") != std::string::npos)) {
            info.name.insert(info.name.size() - 7, "_exps");
            info.ne.push_back(2);
        }
        const bool matrix = info.type != quant::GGML_TYPE_F32;
        const float scale = std::ldexp(t % 3 ? 1.0f : -1.0f, -int(4 + t % 3));
        const std::vector<float> scales = info.ne.size() == 3 ? std::vector<float>{scale, -2 * scale} : std::vector<float>{scale};
        std::vector<uint8_t> data;
        if (!matrix) {
            data.assign(base.tensor_data(t), base.tensor_data(t) + base.tensor_bytes(t));
        } else {
            const size_t values = size_t(info.n_elements());
            info.type = decoded ? quant::GGML_TYPE_F32 : quant::GGML_TYPE_NVFP4;
            data.resize(decoded ? values * sizeof(float) : values / 64 * 36);
            // E4M3 0x38 is 1; E2M1 codes 0,2,4,10,12 represent 0,1,2,-1,-2 independently of llmx's decoder.
            constexpr unsigned codes[] = {0, 2, 4, 10, 12};
            constexpr float weights[] = {0, 1, 2, -1, -2};
            for (size_t i = 0; i < values; ++i) {
                const size_t pick = (i * 13 + i / 128 + t * 7) % 5;
                if (decoded) {
                    const float value = weights[pick] * scales[i / (values / scales.size())];
                    std::memcpy(data.data() + i * sizeof(float), &value, sizeof(value));
                } else {
                    const size_t block = i / 64, group = i % 64 / 16, lane = i % 16;
                    data[block * 36 + group] = 0x38;
                    data[block * 36 + 4 + group * 8 + lane % 8] |= uint8_t(codes[pick] << (4 * (lane / 8)));
                }
            }
        }
        out.tensors.push_back(info);
        out.add_tensor_data(data);
        if (matrix && !decoded) {
            out.tensors.push_back({info.name.substr(0, info.name.size() - 7) + ".scale", {scales.size()}, quant::GGML_TYPE_F32, 0});
            std::vector<uint8_t> bytes(scales.size() * sizeof(float));
            std::memcpy(bytes.data(), scales.data(), bytes.size());
            out.add_tensor_data(bytes);
        }
    }
    if (moe) {
        for (size_t layer = 0; layer < 2; ++layer) {
            out.tensors.push_back({"blk." + std::to_string(layer) + ".ffn_gate_inp.weight", {128, 2}, quant::GGML_TYPE_F32, 0});
            std::vector<uint8_t> bytes(256 * sizeof(float));
            for (size_t i = 0; i < 256; ++i) {
                const float value = float(int((i * 7 + layer * 11) % 19) - 9) / 128;
                std::memcpy(bytes.data() + i * sizeof(float), &value, sizeof(value));
            }
            out.add_tensor_data(bytes);
        }
    }
    gguf::MetaValue tokens;
    tokens.vtype = gguf::V_ARRAY;
    tokens.u = gguf::V_STRING;
    for (char c = 'a'; c < 'a' + 16; ++c) {
        gguf::MetaValue value;
        value.vtype = gguf::V_STRING;
        value.s.assign(1, c);
        tokens.arr.push_back(value);
    }
    out.kv.push_back({"tokenizer.ggml.tokens", tokens});
    gguf::MetaValue eos;
    eos.vtype = gguf::V_UINT32;
    eos.u = 15;
    out.kv.push_back({"tokenizer.ggml.eos_token_id", eos});
    return out;
}

struct Copying : backend::CpuBackend {
    std::vector<const backend::Buffer*> weight_storage;
    bool cpu_collective = false;
    size_t scaled_matmul_calls = 0;
    size_t expert_calls = 0;
    Copying() { set_threads(1); }
    bool is_cpu() const override { return cpu_collective; }
    bool reads_in_place() const override { return false; }
    backend::BufferPtr alloc_weight(size_t bytes) override {
        auto buffer = alloc(bytes, backend::Memory::device);
        weight_storage.push_back(buffer.get());
        return buffer;
    }
    backend::BufferPtr adopt(const void* data, size_t bytes) override {
        auto buffer = alloc(bytes, backend::Memory::device);
        write(*buffer, 0, data, bytes);
        weight_storage.push_back(buffer.get());
        return buffer;
    }
    void matmul(backend::WeightView weight, backend::CSlice x, backend::Slice y,
                size_t nin, size_t nout, size_t rows, backend::RowRuns runs = {},
                backend::Dtype dtype = backend::Dtype::f16) override {
        if (weight.type == quant::GGML_TYPE_NVFP4) {
            require(weight.scale_count == 1 && weight.scales.buffer &&
                        std::find(weight_storage.begin(), weight_storage.end(), weight.scales.buffer) != weight_storage.end(),
                    "scaled matrix did not use a full companion-scale copy on its destination");
            ++scaled_matmul_calls;
        }
        backend::CpuBackend::matmul(weight, x, y, nin, nout, rows, runs, dtype);
    }
    void matmul_experts(std::initializer_list<backend::Projection> projections, backend::CSlice x, size_t nin,
                        size_t rows, const Routing& routing, backend::RowRuns runs = {}, backend::Dtype dtype = backend::Dtype::f16) override {
        for (const auto& p : projections)
            require(p.weight.scale_count == 2 && std::find(weight_storage.begin(), weight_storage.end(), p.weight.scales.buffer) != weight_storage.end(),
                    "streamed expert scales have no storage on their destination");
        ++expert_calls;
        backend::CpuBackend::matmul_experts(projections, x, nin, rows, routing, runs, dtype);
    }
};

backend::BackendPtr cpu() {
    auto device = backend::make_cpu_backend();
    device->set_threads(1);
    return device;
}

void compare(const std::vector<float>& actual, const std::vector<float>& expected) {
    require(actual.size() == expected.size(), "scaled model changed logits size");
    for (size_t i = 0; i < actual.size(); ++i)
        require(std::isfinite(actual[i]) && std::abs(actual[i] - expected[i]) < 2e-5f,
                "scaled model differs from independently constructed F32 weights");
}

void check(const std::string& path, bool tied) {
    const auto file = fixture(false, tied), oracle = fixture(true, tied);
    const auto weights = infer::gguf_weights(file);
    infer::ModelOptions options;
    options.dtype = backend::Dtype::f32;
    options.kv_k = options.kv_v = backend::KVType::f32;
    const std::vector<uint32_t> ids = {1, 4, 2, 7, 3};
    infer::Model reference(infer::gguf_weights(oracle), cpu(), options);
    const auto expected = reference.prefill(ids);
    infer::Model local(weights, cpu(), options);
    compare(local.prefill(ids), expected);
    const auto footprint = infer::footprint(weights, infer::plan_model(weights), options);
    require(footprint.embedding.bytes == file.tensor_bytes(0) + sizeof(float), "embedding scale missing from fit");
    require(footprint.tied == tied, "scale changed tied-head identity");
    for (size_t width : {size_t(1), size_t(2)}) {
        infer::Placement placement;
        placement.width = width;
        placement.mixer_device = width == 1 ? std::vector<int>{0, 1} : std::vector<int>{0, 0};
        placement.ffn_device = placement.mixer_device;
        placement.output_device = width == 1 ? 1 : 0;
        infer::Model model(weights, {cpu(), cpu()}, placement, options);
        compare(model.prefill(ids), expected);
    }
    gguf::write_gguf(file, path);
    std::vector<infer::LoadMode> modes{infer::LoadMode::mapped, infer::LoadMode::automatic};
    try {
        format::FileReader probe(path, true);
        modes.push_back(infer::LoadMode::direct);
    } catch (const format::DirectUnavailable&) {
        infer::PlacementRequest request;
        request.names = {"cpu"};
        std::string error;
        try { infer::load_model(path, {cpu()}, request, options, {}, infer::LoadMode::direct); }
        catch (const std::runtime_error& e) { error = e.what(); }
        require(error.find("--load-mode direct: ") == 0 && error.find(path) != std::string::npos,
                "unsupported direct load did not name the refused file");
        std::cout << "model weight scales: direct reads unavailable, refusal checked\n";
    }
    for (auto mode : modes) {
        for (bool copy : {false, true}) {
            backend::BackendPtr device = copy ? std::make_shared<Copying>() : cpu();
            infer::PlacementRequest request;
            request.names = {"cpu"};
            auto model = infer::load_model(path, {device}, request, options, {}, mode);
            compare(model->model->prefill(ids), expected);
        }
        // File-loaded layer and tensor splits must copy companion scales to both compute destinations.
        for (bool tensor_split : {false, true}) {
            auto first = std::make_shared<Copying>();
            auto second = std::make_shared<Copying>();
            first->cpu_collective = second->cpu_collective = tensor_split;
            infer::PlacementRequest request;
            request.names = {"copy-a", "copy-b"};
            if (tensor_split) request.width = 2;
            else request.shares = {1, 1};
            auto model = infer::load_model(path, {first, second}, request, options, {}, mode);
            compare(model->model->prefill(ids), expected);
            require(first->scaled_matmul_calls && second->scaled_matmul_calls,
                    tensor_split ? "tensor-split file load did not use scaled weights on both backends"
                                 : "layer-split file load did not use scaled weights on both backends");
        }
    }
    auto bad = weights;
    bad.tensors[*bad.tensors[0].scale].shape = {2};
    bool refused = false;
    try { infer::Model model(bad, cpu(), options); }
    catch (const std::runtime_error& e) { refused = std::string(e.what()).find("incompatible weight scale") != std::string::npos; }
    require(refused, "wrong scale shape was accepted");
    auto failing = std::make_shared<LoadingBackend>();
    failing->fail_adopt = 2;
    bool failed = false;
    try { infer::Model model(weights, failing, options); }
    catch (const std::runtime_error& e) { failed = std::string(e.what()) == "injected adoption failure"; }
    require(failed && !failing->state->premature, "scale buffer released before failed loading drained");
}

void check_experts(const std::string& path) {
    const auto file = fixture(false, false, true), oracle = fixture(true, false, true);
    const auto weights = infer::gguf_weights(file);
    infer::ModelOptions options;
    options.dtype = backend::Dtype::f32;
    options.kv_k = options.kv_v = backend::KVType::f32;
    infer::Model reference(infer::gguf_weights(oracle), cpu(), options);
    const std::vector<uint32_t> prompt{1, 4, 2, 7, 3}, followup{2, 5, 6};
    const auto expected = reference.prefill(prompt);
    const auto decoded = reference.step(8);
    const auto continued = reference.prefill(followup);
    for (size_t width : {size_t(1), size_t(2)}) {
        infer::Placement placement;
        placement.width = width;
        placement.mixer_device = width == 1 ? std::vector<int>{0, 1} : std::vector<int>{0, 0};
        placement.ffn_device = placement.mixer_device;
        placement.output_device = width == 1 ? 1 : 0;
        infer::Model model(weights, {cpu(), cpu()}, placement, options);
        compare(model.prefill(prompt), expected);
        compare(model.step(8), decoded);
        compare(model.prefill(followup), continued);
    }
    auto device = std::make_shared<Copying>();
    infer::Placement placement;
    placement.mixer_device = {0, 0};
    placement.ffn_device = {1, 1};
    placement.stream_from = 2;
    infer::Model model(weights, {device, cpu()}, placement, options);
    compare(model.prefill(prompt), expected);
    require(device->expert_calls == 2, "prompt did not stream both layers' experts");
    compare(model.step(8), decoded);
    require(device->expert_calls == 2, "decode unexpectedly streamed experts");
    compare(model.prefill(followup), continued);
    require(device->expert_calls == 4, "follow-up did not reuse both expert windows");

    // Keep all companions in the second shard so every scaled weight finds its scale across a file boundary, including the streamed expert stacks.
    std::vector<std::string> paths{path};
    gguf::write_gguf(file, path);
    for (size_t no = 0; no < 2; ++no) {
        gguf::GGUFModel shard;
        if (!no) shard.kv = file.kv;
        for (const auto& entry : std::vector<std::pair<std::string, uint64_t>>{
                {"split.no", no}, {"split.count", 2}, {"split.tensors.count", file.tensors.size()}}) {
            gguf::MetaValue value;
            value.vtype = entry.first == "split.tensors.count" ? gguf::V_INT32 : gguf::V_UINT16;
            if (value.vtype == gguf::V_INT32) value.i = int64_t(entry.second);
            else value.u = entry.second;
            shard.kv.push_back({entry.first, value});
        }
        for (size_t i = 0; i < file.tensors.size(); ++i) {
            const auto& tensor = file.tensors[i];
            const bool scale = tensor.name.size() >= 6 && tensor.name.compare(tensor.name.size() - 6, 6, ".scale") == 0;
            if (scale != bool(no)) continue;
            shard.tensors.push_back(tensor);
            shard.add_tensor_data(std::vector<uint8_t>(file.tensor_data(i), file.tensor_data(i) + file.tensor_bytes(i)));
        }
        paths.push_back(path + "-0000" + std::to_string(no + 1) + "-of-00002.gguf");
        gguf::write_gguf(shard, paths.back());
    }
    for (size_t source : {size_t(0), size_t(1)}) {
        for (auto mode : {infer::LoadMode::mapped, infer::LoadMode::automatic}) {
            auto copying = std::make_shared<Copying>();
            infer::PlacementRequest request;
            request.names = {"cpu"};
            request.dtype = backend::Dtype::f32;
            request.cpu_moe = -1;
            request.stream_from = 2;
            auto loaded = infer::load_model(paths[source], {copying}, request, options, {}, mode);
            loaded->model->set_threads(1);
            if (mode == infer::LoadMode::automatic)
                require(loaded->times.files == (source ? 2u : 1u), "automatic MoE load did not stream every weight/scale file");
            if (source) {
                const auto companions = gguf::weight_scales(loaded->file);
                size_t separate = 0;
                for (size_t i = 0; i < companions.size(); ++i)
                    if (companions[i]) {
                        require(loaded->file.span(i).file != loaded->file.span(*companions[i]).file,
                                "scaled weight and companion did not cross GGUF shards");
                        ++separate;
                    }
                require(separate > 0, "sharded fixture has no scale companions");
            }
            compare(loaded->model->prefill(prompt), expected);
            require(copying->expert_calls == 2, "file-loaded prompt did not stream both layers' experts");
            compare(loaded->model->step(8), decoded);
            require(copying->expert_calls == 2, "file-loaded decode unexpectedly streamed experts");
            compare(loaded->model->prefill(followup), continued);
            require(copying->expert_calls == 4, "file-loaded follow-up did not reuse both expert windows");
            std::cout << "model weight scales: " << (source ? "cross-shard" : "single-file")
                      << (mode == infer::LoadMode::mapped ? " mapped" : " auto")
                      << " streamed MoE prompt/decode/follow-up pass\n";
        }
    }
    for (size_t i = 1; i < paths.size(); ++i) std::filesystem::remove(std::filesystem::u8path(paths[i]));
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("expected fixture path");
        check(argv[1], false);
        check(argv[1], true);
        check_experts(argv[1]);
        std::filesystem::remove(std::filesystem::u8path(argv[1]));
        std::cout << "model weight scales: independent F32, tied head, layer/tensor split and load modes pass\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
