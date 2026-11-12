#pragma once
// A placed model's passes against what its fit priced and kept (Backend::reserve_scratch), shared by the ROCm and Vulkan backend tests (docs/TESTS.md, backend-rocm-scratch).
#include "backends/backend.hpp"
#include "model/arch/registry.hpp"
#include "model/place.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace scratch_growth {

// How far the device's memory in use rose through the passes, and what its backend keeps free for that beside its priced scratch.
struct Growth {
    size_t grew = 0, kept = 0;
};

// The passes a server runs, from a thread that `used` polls about every millisecond: a prompt in ubatches of 64 and eight passes of 64 sequences, the prompt's among them.
// `low` builds the model with its scratch made for one row, the control that shows the poll sees a pass's scratch.
inline Growth passes(const infer::ModelWeights& w, const backend::BackendPtr& b, const std::string& name, bool low, const std::function<size_t()>& used) {
    const infer::ModelPlan plan = infer::plan_model(w);
    const size_t context = plan.context_length, prompt = std::min<size_t>(1024, context / 2) / 64 * 64, vocab = plan.vocab;
    infer::PlacementRequest request;
    request.names = {name};
    request.ubatch = 64;
    request.decode_rows = 64;
    request.logit_rows = 64;
    request.slots = 64;
    infer::ModelOptions options;
    // Blocks for the prompt's sequence and one for each of the other 63, which eight decode steps fill.
    const size_t block = b->kv_layout().block_tokens;
    options.kv_tokens = (backend::blocks_for(prompt + 8, block) + 63) * block;
    options.state_slots = 64;
    std::unique_ptr<infer::Model> model;
    if (low) {
        model = std::make_unique<infer::Model>(w, b, options);
        model->set_ubatch(64);
        backend::PassWidths one = plan.widths;
        one.rows = 1;
        one.context = options.kv_tokens;
        b->reserve_scratch(one);
    } else {
        model = std::move(infer::place_model(w, std::vector<backend::BackendPtr>{b}, request, options).model);
    }
    infer::ExecContext ctx;
    model->reserve_passes(ctx, 1, infer::fitted_rows(request), request.logit_rows);
    std::vector<infer::Sequence> seqs;
    for (int i = 0; i < 64; ++i) seqs.push_back(model->make_sequence());
    std::vector<uint32_t> ids(prompt);
    for (size_t i = 0; i < ids.size(); ++i) ids[i] = uint32_t((i * 131) % vocab);
    b->sync();
    const size_t start = used();
    std::atomic<bool> stop{false};
    std::atomic<size_t> most{start};
    std::thread poll([&] {
        while (!stop.load()) {
            const size_t u = used();
            if (u > most.load()) most = u;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });
    // The poll stops and joins before a pass's exception leaves, so the test reports it rather than terminating.
    struct Joined {
        std::atomic<bool>& stop;
        std::thread& poll;
        ~Joined() {
            stop = true;
            if (poll.joinable()) poll.join();
        }
    } joined{stop, poll};
    // A pass of the reserved context, its stages in a row.
    const auto pass = [&](const infer::BatchEntry* entries, size_t n) {
        model->begin_pass(ctx, 0, entries, n, 0);
        for (size_t st = 0; st < model->stage_count(); ++st) model->run_pass_stage(ctx, 0, st);
        model->end_pass(ctx, 0);
    };
    for (size_t at = 0; at < ids.size(); at += 64) {
        infer::BatchEntry e{&seqs[0], &ids[at], 64, at + 64 == ids.size()};
        e.extent = ids.size();
        pass(&e, 1);
    }
    std::vector<infer::BatchEntry> batch;
    for (int i = 1; i < 64; ++i) {
        infer::BatchEntry e{&seqs[i], &ids[(size_t)i], 1, true};
        e.extent = 1;
        batch.push_back(e);
    }
    infer::BatchEntry long_one{&seqs[0], &ids[7], 1, true};
    long_one.extent = 1;
    batch.push_back(long_one);
    for (int step = 0; step < 8; ++step) pass(batch.data(), batch.size());
    b->sync();
    stop = true;
    poll.join();
    Growth g;
    g.grew = most.load() - start;
    g.kept = b->memory_kept(0);
    std::cout << "scratch " << (low ? "one-row" : "fitted") << " grew " << g.grew << " kept " << g.kept << "\n";
    return g;
}

// The fitted model grows its device by no more than the backend keeps; the one-row control must grow past the fitted model, unless `control` is false for a model too small to show it.
inline bool held(const infer::ModelWeights& w, const std::function<backend::BackendPtr()>& make, const std::string& name, const std::function<size_t()>& used,
                 bool control, std::string& why) {
    const Growth fitted = passes(w, make(), name, false, used);
    if (fitted.grew > fitted.kept) {
        why = "a model's passes grew its device memory by " + std::to_string(fitted.grew) + " bytes, past the " + std::to_string(fitted.kept) + " its backend keeps beside the priced scratch";
        return false;
    }
    if (!control) return true;
    const Growth low = passes(w, make(), name, true, used);
    if (low.grew <= fitted.grew) {
        why = "a model whose scratch was made for one row grew no more through its passes than the fitted one, so the poll does not see a pass's scratch";
        return false;
    }
    return true;
}

}  // namespace scratch_growth
