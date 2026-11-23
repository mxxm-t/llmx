#pragma once
// A pass a tensor group cuts in two parts (Model::cut_row) against the same pass recorded whole, for the model tests of each architecture that runs on a group.
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "backends/cpu/cpu_backend.hpp"
#include "model/runtime.hpp"

// Every byte of a sequence's history on the devices, its KV blocks and its state, through a whole copy to host memory.
inline std::vector<uint8_t> history_bytes(infer::Model& model, infer::Sequence& seq) {
    infer::HostHistory host;
    model.save_host(seq, seq.length(), host, std::numeric_limits<size_t>::max());
    model.wait_host(host);
    const size_t slab = (size_t)64 << 20;
    std::vector<uint8_t> out;
    for (const auto& ranges : {model.host_ranges(host, 0, seq.length()), model.host_state_ranges(host)})
        for (const infer::HostRange& r : ranges)
            for (size_t at = r.offset; at < r.offset + r.bytes; ++at) out.push_back(((const uint8_t*)host.slabs[r.device][at / slab]->host_ptr())[at % slab]);
    model.release_host(host);
    return out;
}

// A CPU backend whose collectives say a lane sum of fewer than `below` floats copies beside the work, so a group of them cuts its prompt passes as a device group does, counting the lane sums the cut parts send.
struct BesideCpu final : backend::CpuBackend {
    std::shared_ptr<size_t> sends = std::make_shared<size_t>(0);
    std::shared_ptr<size_t> below = std::make_shared<size_t>(std::numeric_limits<size_t>::max());

    std::unique_ptr<backend::Collective> join(const std::vector<backend::Backend*>& members, size_t rows, size_t width) override {
        struct Beside final : backend::Collective {
            std::unique_ptr<backend::Collective> inner;
            std::shared_ptr<size_t> sends, below;
            backend::Slice partial(size_t member) override { return inner->partial(member); }
            void sum_into(const std::vector<backend::Slice>& residual, size_t rows, size_t width) override { inner->sum_into(residual, rows, width); }
            void sum_in_place(size_t rows, size_t width) override { inner->sum_in_place(rows, width); }
            void send(size_t lane, size_t base, size_t rows, size_t width) override {
                ++*sends;
                inner->send(lane, base, rows, width);
            }
            void finish(size_t lane, size_t base, size_t rows, size_t width, const std::vector<backend::Slice>* residual) override {
                inner->finish(lane, base, rows, width, residual);
            }
            bool copies_beside(size_t floats) const override { return floats < *below; }
        };
        auto result = std::make_unique<Beside>();
        result->inner = CpuBackend::join(members, rows, width);
        result->sends = sends;
        result->below = below;
        return result;
    }
};

// A pass a group model cuts, ending at token 256 from no history and after 37, against the same rows beside a one-token entry of another sequence: logits, KV and state bytes and the decode after.
// Where `below` is BesideCpu's bound the second arm records whole and two sequences are cut inside the second entry and at their boundary, else both arms cut; a state model needs two state slots.
inline void check_cut_passes(infer::Model& model, const std::shared_ptr<size_t>& below = {}) {
    const size_t bound = below ? *below : 0;
    const auto whole = [&](bool on) {
        if (below) *below = on ? 0 : bound;
    };
    const size_t V = model.n_vocab();
    std::vector<uint32_t> ids(384);
    for (size_t i = 0; i < ids.size(); ++i) ids[i] = uint32_t((i * 2654435761u >> 7) % V);
    for (size_t history : {size_t(0), size_t(37)}) {
        const size_t n = 256 - history;
        const std::string after = " after a history of " + std::to_string(history);
        std::vector<float> logits(n * V), decode(V);
        std::vector<uint8_t> bytes;
        for (bool cut : {true, false}) {
            infer::Sequence a = model.make_sequence(), other = model.make_sequence();
            infer::ExecContext x;
            if (history) {
                const infer::BatchEntry h{&a, ids.data(), history, false};
                model.forward(x, &h, 1);
            }
            const uint32_t lone = ids[299];
            infer::BatchEntry e[2] = {{&a, ids.data() + history, n, true}, {&other, &lone, 1, false}};
            e[0].every_logits = true;
            e[0].keep = model.keeps_state();
            whole(!cut);
            model.forward(x, e, cut ? 1 : 2);
            whole(false);
            for (size_t r = 0; r < n; ++r) {
                if (cut) std::memcpy(logits.data() + r * V, x.logits(r), V * sizeof(float));
                else if (std::memcmp(logits.data() + r * V, x.logits(r), V * sizeof(float))) throw std::runtime_error("a pass cut in two parts gives another logits row " + std::to_string(r) + after);
            }
            if (cut) bytes = history_bytes(model, a);
            else if (bytes != history_bytes(model, a)) throw std::runtime_error("a pass cut in two parts leaves other KV or state bytes" + after);
            const uint32_t next = ids[history + n];
            const infer::BatchEntry d{&a, &next, 1, true};
            model.forward(x, &d, 1);
            if (cut) std::memcpy(decode.data(), x.logits(0), V * sizeof(float));
            else if (std::memcmp(decode.data(), x.logits(0), V * sizeof(float))) throw std::runtime_error("the decode after a pass cut in two parts differs" + after);
            model.reset(a);
            model.reset(other);
        }
    }
    if (!below) return;
    // Both sequences end on whole KV blocks, since the host copy that compares the bytes takes whole blocks, and the pass's cut falls 64 rows into the second (384 rows) or at their boundary (256).
    for (const size_t second : {size_t(256), size_t(128)}) {
        const size_t first = 128, rows = first + second;
        const std::string shape = " of two sequences of " + std::to_string(first) + " and " + std::to_string(second) + " rows";
        std::vector<float> logits(rows * V);
        std::vector<uint8_t> bytes[2];
        for (bool cut : {true, false}) {
            infer::Sequence a = model.make_sequence(), b = model.make_sequence();
            infer::ExecContext x;
            infer::BatchEntry e[2] = {{&a, ids.data(), first, true}, {&b, ids.data() + first, second, true}};
            for (infer::BatchEntry& entry : e) {
                entry.every_logits = true;
                entry.keep = model.keeps_state();
            }
            whole(!cut);
            model.forward(x, e, 2);
            whole(false);
            for (size_t r = 0; r < rows; ++r) {
                if (cut) std::memcpy(logits.data() + r * V, x.logits(r), V * sizeof(float));
                else if (std::memcmp(logits.data() + r * V, x.logits(r), V * sizeof(float))) throw std::runtime_error("a pass cut in two parts gives another logits row " + std::to_string(r) + shape);
            }
            infer::Sequence* seqs[2] = {&a, &b};
            for (size_t k = 0; k < 2; ++k) {
                if (cut) bytes[k] = history_bytes(model, *seqs[k]);
                else if (bytes[k] != history_bytes(model, *seqs[k])) throw std::runtime_error("a pass cut in two parts leaves other KV or state bytes" + shape);
            }
            model.reset(a);
            model.reset(b);
        }
    }
}
