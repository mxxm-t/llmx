#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "backends/backend.hpp"
#include "core/host_memory.hpp"
#include "quant/types.hpp"

// A model split by layers over several devices: which consecutive layers each device runs, fitted against what each device reports free (docs/MULTI-DEVICE.md, phase 1).
// Nothing here knows an architecture: the model describes what it asks of memory as a Footprint, and the split works on those numbers alone.

namespace infer {

// One matrix's storage type, row shape and bytes including companion scales, with `product` for a projection or head rather than a gather, norm or routed stack.
struct Matrix {
    uint32_t type = 0;
    size_t nin = 0, rows = 0, bytes = 0;
    bool product = false;
};

// An externally-backed lookup's bounded per-pass storage. Its table bytes stay in the source; only requested rows and lookup scratch are fitted.
struct LookupMemory {
    size_t rows_per_token = 0, row_bytes = 0, io_bytes = 0;
};

// A media encoder opened beside a model on the device that holds its embedding (docs/VISION.md, section 3): its weights, its buffers at its bound and the widths of its passes, each priced as the device prices a pass.
struct EncoderMemory {
    std::vector<Matrix> weights;
    std::vector<size_t> buffers;
    std::vector<backend::PassWidths> widths;
};

// What a model asks of the memory of the devices it runs on, part by part; an architecture computes it from its file and options.
struct Footprint {
    std::vector<std::vector<Matrix>> layers;   // each layer's weights, a tensor several layers take counted by the first
    std::vector<std::string> joined;           // for each layer, a tensor it and an earlier layer take, so no split point falls before it; empty where none does
    std::vector<LookupMemory> lookups;         // optional external lookup storage, aligned with layers
    Matrix embedding;                          // the embedding table
    Matrix output;                             // the head's matrix, the embedding table itself when tied
    Matrix output_norm;                        // what runs with the head besides its matrix
    std::vector<Matrix> head_aux;               // additional head matrices or norms, counted on the output device
    std::vector<Matrix> embed_aux;              // embedding-part norms beside the table, counted on the first device
    bool tied = false;                         // the head reads the embedding table, adopted once where both sit on one device
    std::vector<size_t> cache;                 // each layer's cache: a KV layer's for every position the budget allows, a state layer's for every slot; none for every layer when empty
    std::vector<std::vector<size_t>> cache_buffers;   // the buffers of each layer's cache a device allocates on its own (K, V and auxiliary planes, the state), which it rounds (DeviceBudget::allocated)
    // Each layer's entry source (LayerPlan::entry_source) or -1, and its pooled rows, which a device running a reader of it and not it keeps again as a replica; empty where no layer reads another's.
    std::vector<int> reads;
    std::vector<size_t> pooled;
    size_t tables = 0;                         // position tables: the host keeps them while the model lives, and every device that copies weights holds its own
    std::vector<size_t> table_buffers;         // each position table, a buffer of its own on such a device; one buffer of tables where empty
    size_t activations_per_row = 0;            // one row of a pass's activations on each device
    size_t activation_slots = 0;               // the slots of a pass's arena, each aligned to 64 bytes (Model::alloc_arena)
    size_t logits_per_row = 0;                 // one row of logits where the head runs
    size_t head_slice_per_row = 0;             // on a tensor group, a member's slice of a logits row, which each member of the head's group keeps beside its layers
    size_t argmax_per_row = 0, argmax_rows = 0;   // off a tensor group, a logits row and its id kept on the head's device for argmax passes, and how many (Model::argmax_buffers)
    size_t handoff_per_row = 0;                // one row of the stream handed from one device to the next through host memory, in each of the handoff buffers every used device but the last keeps
    // An embedded drafter's head part, which runs where the head does (docs/SPECULATIVE.md, section 7): its weights, the embedding table its first part reads, and its carried rows, chain ids, logits and, without stages, KV or state.
    std::vector<Matrix> drafter;
    Matrix drafter_embedding;
    bool drafter_embedding_held = true;   // the first device's embedding buffer is the whole table the drafter gathers, not a member's rows of it
    size_t drafter_cache = 0;
    // Its stages (LayerPlan::stages) in order, which the split places on any device beside its layers, and the KV or state each device holding one keeps.
    std::vector<std::vector<Matrix>> drafter_stages;
    std::vector<size_t> drafter_pairs;   // a paired drafter's target layer a stage (ModelPlan::draft_pairs), each stage on the device running its layer
    size_t drafter_stage_cache = 0;
    size_t host_state = 0;                     // model-owned host state, counted once rather than per device
    size_t context_rows = 0;                   // the model's own context beside a server's passes, which a mark's rerun and a drafter's chain take (Model::reserve_drafting)
    size_t host_scratch_per_row = 0;           // host lookup indices for each used stage's execution arena
    backend::PassWidths widths;                // the widest shapes a device's passes run, a member's heads on a tensor group; rows and logits are the fit's (pass_widths)
    EncoderMemory encoder;                     // the caller's encoder, charged with the embedding before the KV is sized (PlacementRequest::encoder)
};

// The handoff buffers each device the residual leaves keeps for `slots` passes in flight: on a pipelined split one per slot, two at least, so a prompt's chunk goes out through one while the one before waits in the other; otherwise one.
inline size_t handoff_buffers(size_t slots, bool pipelined) { return pipelined ? std::max<size_t>(2, slots) : 1; }

// What a device offers a split: the bytes it reports free, or nothing when it cannot tell; whether weights placed on it read the mapped file in place, as the CPU's do; and what adopting a matrix keeps resident there, its bytes when empty.
// `host_side` is host memory the device's backend holds for itself, such as upload staging, which counts against the host.
struct DeviceBudget {
    std::string name;
    std::optional<size_t> bytes;
    bool host = false;
    std::function<size_t(const Matrix&)> resident;
    size_t host_side = 0;
    size_t scratch = 0;   // what the device keeps free beside its priced scratch (Backend::memory_kept)
    std::function<size_t(size_t members, size_t width)> collective = nullptr;   // what a member of a tensor group keeps a row of a sum (Backend::collective_bytes_per_row)
    size_t collective_per_row = 0;                                    // on a tensor group, the most its members keep a row of a sum (group_budgets)
    bool argmax = false;                                              // the device keeps argmax passes' logits rows (not the CPU, and it implements argmax_rows)
    std::function<size_t(const backend::PassWidths&)> priced = nullptr;   // the scratch passes of those widths take there (Backend::scratch_bytes)
    std::function<size_t(size_t)> allocated = nullptr;                    // what a buffer of so many bytes takes there (Backend::allocation_bytes)
    std::optional<size_t> allocation_limit;                               // the most one buffer may take there (Backend::allocation_limit)
    std::vector<char> serves;   // per device of the list, whether this one's crossings to it are ordered on the devices (Backend::handoff), with no host round trip
};

// The layers a stage round takes at least: where every crossing is served on the devices, a device running k layers takes them in max(1, k / kRunLayers) runs (split_layers, docs/MULTI-DEVICE.md).
// The landing timing round set it, at four rounds a device of 16 layers (docs/STATUS.md, Stage rounds).
inline constexpr int kRunLayers = 4;

// The activation arenas a device makes (Model::ensure): a pass's `rows`, the model's own context's rows and a server's replies' rows, each a buffer as `allocated` gives what it takes there.
inline size_t arena_bytes(const Footprint& fp, size_t rows, size_t reply_rows, const std::function<size_t(size_t)>& allocated) {
    size_t n = 0;
    for (size_t r : {rows, fp.context_rows, reply_rows}) {
        if (!r) continue;
        // Each of an arena's slots starts on 64 bytes (Model::alloc_arena).
        const size_t b = backend::size_add(backend::size_mul(r, fp.activations_per_row), backend::size_mul(fp.activation_slots, 64));
        n = backend::size_add(n, allocated ? allocated(b) : b);
    }
    return n;
}

// The most outputs one product of float weights (F32, F16, BF16) among `weights` writes, 0 without one (PassWidths::float_out).
inline size_t float_product_out(const std::vector<Matrix>& weights) {
    size_t most = 0;
    for (const Matrix& m : weights)
        if (m.product && (m.type == quant::GGML_TYPE_F32 || m.type == quant::GGML_TYPE_F16 || m.type == quant::GGML_TYPE_BF16)) most = std::max(most, m.rows);
    return most;
}

// The widths a device's passes run its scratch at (Backend::scratch_bytes): the footprint's, at a pass's rows or the model's own context rows, whichever is more, and on the head's device its logits rows.
inline backend::PassWidths pass_widths(const Footprint& fp, size_t rows, size_t logit_rows, bool head) {
    backend::PassWidths w = fp.widths;
    for (const std::vector<Matrix>& layer : fp.layers) w.float_out = std::max(w.float_out, float_product_out(layer));
    for (const std::vector<Matrix>& stage : fp.drafter_stages) w.float_out = std::max(w.float_out, float_product_out(stage));
    w.float_out = std::max({w.float_out, float_product_out(fp.drafter), float_product_out(fp.head_aux)});
    w.rows = std::max(rows, fp.context_rows);
    w.logit_rows = head ? logit_rows : 0;
    w.float_head = head && float_product_out({fp.output}) > 0;
    if (!head) w.vocab = 0;
    return w;
}

// A budget for each backend, named as its caller names it: what the backend reports free, whether it reads weights in place, what adopting a matrix keeps on it, and its own host memory.
inline std::vector<DeviceBudget> budgets_for(const std::vector<backend::BackendPtr>& backends, const std::vector<std::string>& names) {
    if (names.size() != backends.size()) throw std::runtime_error("split: a name for every device");
    std::vector<DeviceBudget> budgets;
    for (size_t d = 0; d < backends.size(); ++d) {
        const backend::Backend* b = backends[d].get();
        const std::optional<size_t> free = b->memory_available();
        budgets.push_back(DeviceBudget{names[d], free, b->reads_in_place(),
                                       [b](const Matrix& w) { return b->resident_bytes(w.type, w.nin, w.rows, w.bytes, w.product); },
                                       b->host_resident(), b->memory_kept(free.value_or(0)),
                                       [b](size_t members, size_t width) { return b->collective_bytes_per_row(members, width); }});
        budgets.back().argmax = !b->is_cpu() && b->implements(backend::Op::argmax_rows);
        budgets.back().priced = [b](const backend::PassWidths& w) { return b->scratch_bytes(w); };
        budgets.back().allocated = [b](size_t bytes) { return b->allocation_bytes(bytes); };
        budgets.back().allocation_limit = b->allocation_limit();
    }
    // Asked once at load, as the model asks for its own readers: a handoff from each device to every other, and which of them it serves.
    for (size_t d = 0; d < backends.size(); ++d) {
        std::vector<backend::Backend*> to;
        for (size_t e = 0; e < backends.size(); ++e)
            if (e != d) to.push_back(backends[e].get());
        const std::unique_ptr<backend::Handoff> h = to.empty() ? nullptr : backends[d]->handoff(to);
        budgets[d].serves.assign(backends.size(), 0);
        for (size_t e = 0, r = 0; h && e < backends.size(); ++e)
            if (e != d) budgets[d].serves[e] = h->serves(r++);
    }
    return budgets;
}

// Per device, in the order given, the consecutive layers it runs and what it was fitted to hold; the embedding goes with the first device that runs layers and the head with the last.
// A device's runs of layers as text, "0-3,8-11".
inline std::string runs_text(const std::vector<std::pair<int, int>>& runs) {
    std::string s;
    for (const auto& [f, n] : runs) s += (s.empty() ? "" : ",") + std::to_string(f) + "-" + std::to_string(f + n - 1);
    return s;
}

struct LayerSplit {
    struct Stage {
        int count = 0;   // its share of the layers as fitted, in device order
        std::vector<std::pair<int, int>> runs;   // the layers it runs, a (first, count) a round (round_runs)
        size_t weights = 0, cache = 0, other = 0;
        bool operator==(const Stage& o) const { return count == o.count && runs == o.runs && weights == o.weights && cache == o.cache && other == o.other; }
    };
    size_t rounds = 1;   // the runs of layers each device takes in turn (stage rounds, docs/MULTI-DEVICE.md)
    std::vector<Stage> stages;
    int embed_device = 0, output_device = 0;
    std::vector<int> draft_device;   // each drafter stage's device
    size_t host = 0;   // what the host holds besides mapped weights: logits rows, position tables, the handoff between devices and the used backends' own host memory

    bool operator==(const LayerSplit& o) const {
        return rounds == o.rounds && stages == o.stages && embed_device == o.embed_device && output_device == o.output_device && draft_device == o.draft_device && host == o.host;
    }

    // A line a device, or a tensor group of `width` devices whose figures are then a member's; the weights figure is the fit's estimate of their device memory, each tensor as the device allocates it.
    std::string describe(const std::vector<DeviceBudget>& devices, size_t width = 1) const {
        std::string s;
        char line[384];
        const double gib = 1024.0 * 1024.0 * 1024.0;
        for (size_t d = 0; d < stages.size(); ++d) {
            const Stage& st = stages[d];
            char free[48] = "free memory unknown";
            if (devices[d].bytes) std::snprintf(free, sizeof free, "%.2f GiB free", *devices[d].bytes / gib);
            std::string drafts;
            for (size_t k = 0; k < draft_device.size(); ++k)
                if (draft_device[k] == (int)d) drafts += (drafts.empty() ? " and drafter stages " : ",") + std::to_string(k);
            if (!st.count)
                std::snprintf(line, sizeof line, "%s: no layers\n", devices[d].name.c_str());
            else
                std::snprintf(line, sizeof line, "%s: layers %s%s, %sweight memory %.2f GiB estimated, cache %.2f GiB, activations and scratch %.2f GiB, %s\n",
                              devices[d].name.c_str(), runs_text(st.runs).c_str(), drafts.c_str(), width > 1 ? "per member: " : "", st.weights / gib, st.cache / gib,
                              st.other / gib, free);
            s += line;
        }
        std::snprintf(line, sizeof line, "host: logits, tables and staging %.2f GiB\n", host / gib);
        return s + line;
    }
};

// Each device's runs of layers when the `count` layers of each, in device order, are cut into `rounds` runs as equal as whole layers allow and run r of every device precedes run r + 1 of any.
inline std::vector<std::vector<std::pair<int, int>>> round_runs(const std::vector<int>& count, size_t rounds) {
    std::vector<std::vector<std::pair<int, int>>> runs(count.size());
    int at = 0;
    for (size_t r = 0; r < rounds; ++r)
        for (size_t d = 0; d < count.size(); ++d) {
            const size_t k = (size_t)count[d];
            const int n = (int)(k * (r + 1) / rounds - k * r / rounds);
            if (n) runs[d].push_back({at, n});
            at += n;
        }
    return runs;
}

// The request's part of a split's refusals, which no reading of free memory changes: at most 16 devices, and shares, where given, one a device, none negative and not all zero.
// place_model asks it before fitting the KV budget, so a malformed request is refused as such rather than as a budget that does not fit.
inline void check_split(size_t devices, const std::vector<int>& shares) {
    if (devices > 16) throw std::runtime_error("split: more than 16 devices");
    if (shares.empty()) return;
    if (shares.size() != devices) throw std::runtime_error("split: " + std::to_string(shares.size()) + " layer shares for " + std::to_string(devices) + " devices");
    size_t sum = 0;
    for (int s : shares) {
        if (s < 0) throw std::runtime_error("split: a negative layer share");
        sum += (size_t)s;
    }
    if (!sum) throw std::runtime_error("split: every layer share is zero");
}

// Consecutive layers per device in the order given: the best plan over every set of devices that could run layers, with every device and the host within its budget (docs/src/model-layer_split.md says what each holds).
// `shares`, when given, is each device's proportion of the layers in place of the balance, still checked against the budgets, and `logit_rows` the logits rows kept where the head runs, in place of one for each of a pass's `rows`.
inline LayerSplit split_layers(const Footprint& fp, const std::vector<DeviceBudget>& devices, size_t rows, const std::vector<int>& shares = {},
                               std::optional<size_t> host_free = std::nullopt, size_t slots = 1, std::optional<size_t> logit_rows = std::nullopt, size_t reply_rows = 0,
                               size_t rounds = 0) {
    if (devices.empty()) throw std::runtime_error("split: no devices");
    // Every arena a device keeps at once: a pass's, the model's own context's and a server's replies' (Model::reserve_passes), each with its activations, lookup rows and host indices.
    const size_t arena_rows = backend::size_add(backend::size_add(rows, fp.context_rows), reply_rows);
    const size_t L = fp.layers.size(), N = devices.size();
    if (!L) throw std::runtime_error("split: a model without layers");
    check_split(N, shares);
    if (!fp.cache.empty() && fp.cache.size() != L) throw std::logic_error("split: a cache for some layers and not others");
    if (!fp.lookups.empty() && fp.lookups.size() != L) throw std::logic_error("split: a lookup footprint for some layers and not others");
    auto page_bytes = [](size_t bytes) {
        if (!bytes) return size_t(0);
        const size_t page = core::page_size();
        return backend::size_mul(backend::size_add(bytes, page - 1) / page, page);
    };
    // What device d takes for a buffer of `bytes` (DeviceBudget::allocated).
    auto taken = [&](size_t d, size_t bytes) -> size_t { return bytes && devices[d].allocated ? devices[d].allocated(bytes) : bytes; };
    // Beside each arena a device keeps (arena_bytes), each layer's lookup staging and a group member's sum rows are a buffer of their own (Model::ensure).
    auto arenas = [&](size_t d, size_t per_row) {
        size_t n = 0;
        for (size_t r : {rows, fp.context_rows, reply_rows}) n = backend::size_add(n, taken(d, backend::size_mul(r, per_row)));
        return n;
    };
    std::vector<size_t> lookup_device(L, 0), lookup_host(L, 0), lookup_pass(L, 0);
    for (size_t l = 0; l < L && !fp.lookups.empty(); ++l) {
        const LookupMemory& lookup = fp.lookups[l];
        if (!lookup.rows_per_token) continue;
        const size_t rows_per_pass = backend::size_mul(lookup.rows_per_token, lookup.row_bytes);
        lookup_device[l] = rows_per_pass;
        auto host_capacity = [&](size_t nrows) {
            if (!nrows) return size_t(0);
            const size_t count = backend::size_mul(nrows, lookup.rows_per_token);
            const size_t raw = page_bytes(backend::size_mul(count, lookup.row_bytes));
            const size_t arrays = backend::size_mul(count, 2 * sizeof(size_t) + 2 * sizeof(uint32_t));
            return backend::size_add(raw, arrays);
        };
        // The table's own host bytes, its store and its read windows, are counted once (WeightRows::host_bytes) with the context's ids, and the rows read for a pass and their ids once a pass in flight (Pass::lookups).
        const size_t ids = backend::size_mul(backend::size_mul(rows, lookup.rows_per_token), sizeof(uint32_t));
        lookup_host[l] = backend::size_add(ids, page_bytes(lookup.io_bytes));
        lookup_pass[l] = backend::size_add(host_capacity(rows), ids);
    }
    // cached[d][i]: the cache of layers [0, i) on device d, each of a layer's buffers as the device allocates it.
    std::vector<std::vector<size_t>> cached(N, std::vector<size_t>(L + 1, 0));
    for (size_t d = 0; d < N; ++d)
        for (size_t l = 0; l < L; ++l) {
            size_t c = fp.cache.empty() ? 0 : fp.cache[l];
            if (devices[d].allocated && l < fp.cache_buffers.size())
                for (size_t b : fp.cache_buffers[l]) c = backend::size_add(c, devices[d].allocated(b) - b);
            cached[d][l + 1] = backend::size_add(cached[d][l], c);
        }
    // The replicas a device running layers [i, i + k) keeps: each source before i that a layer there reads, its pooled rows once.
    auto replicas = [&](size_t i, size_t k) {
        size_t n = 0;
        for (size_t s = 0; s < i && s < fp.pooled.size(); ++s)
            for (size_t j = i; j < i + k; ++j)
                if (fp.reads[j] == (int)s) {
                    n = backend::size_add(n, fp.pooled[s]);
                    break;
                }
        return n;
    };

    auto resident = [&](size_t d, const Matrix& m) -> size_t {
        if (devices[d].host) return 0;
        return devices[d].resident ? devices[d].resident(m) : m.bytes;
    };
    // prefix[d][i]: the resident bytes of layers [0, i) on device d.
    std::vector<std::vector<size_t>> prefix(N, std::vector<size_t>(L + 1, 0));
    // lookup_prefix[d][i]: the lookup staging of layers [0, i) on device d, a buffer a layer in each arena.
    std::vector<std::vector<size_t>> lookup_prefix(N, std::vector<size_t>(L + 1, 0));
    for (size_t d = 0; d < N; ++d)
        for (size_t l = 0; l < L; ++l) lookup_prefix[d][l + 1] = backend::size_add(lookup_prefix[d][l], arenas(d, lookup_device[l]));
    for (size_t d = 0; d < N; ++d)
        for (size_t l = 0; l < L; ++l) {
            size_t b = 0;
            for (const Matrix& m : fp.layers[l]) b += resident(d, m);
            prefix[d][l + 1] = backend::size_add(prefix[d][l], b);
        }
    // The drafter's embedding copy goes with its first part on the head's device, which runs the chain beside the head; a stage 0 apart reads the first device's whole table, whose rows cross to it.
    // A group that splits the table holds no whole one, so its first part keeps a copy wherever it runs.
    const size_t S = fp.drafter_stages.size();
    const bool paired = !fp.drafter_pairs.empty();   // its fusion reads the table on the head's device (Model::draft_table_device)
    auto drafted_embedding = [&](size_t d, bool first, bool last) {
        if (!fp.drafter_embedding_held) return last && fp.tied ? size_t(0) : resident(d, fp.drafter_embedding);
        return last && !first && !fp.tied ? resident(d, fp.drafter_embedding) : size_t(0);
    };
    // The embedding and head weights a device keeps, given whether it runs the first and the last layers.
    // A tied pair on one device is one buffer, kept as the head keeps it, since the head reads it through a product.
    auto end_weights = [&](size_t d, bool first, bool last) -> size_t {
        size_t drafter = 0;
        size_t head_aux = 0;
        if (last) {
            for (const Matrix& m : fp.head_aux) head_aux = backend::size_add(head_aux, resident(d, m));
            for (const Matrix& m : fp.drafter) drafter += resident(d, m);
            if (!S || paired) drafter += drafted_embedding(d, first, last);
        }
        size_t embed_aux = 0, encoder = 0;
        if (first) {
                for (const Matrix& m : fp.embed_aux) embed_aux = backend::size_add(embed_aux, resident(d, m));
                for (const Matrix& m : fp.encoder.weights) encoder = backend::size_add(encoder, resident(d, m));
        }
        if (first && last && fp.tied) return resident(d, fp.output) + resident(d, fp.output_norm) + head_aux + embed_aux + drafter + encoder;
        return (first ? resident(d, fp.embedding) + embed_aux : 0) + (last ? resident(d, fp.output) + resident(d, fp.output_norm) + head_aux : 0) + drafter + encoder;
    };

    // What the host holds for a set of devices running layers, and the device that carries it: the set's first host device, or none.
    struct Host {
        size_t need = 0, carrier = SIZE_MAX;
    };
    // `R` is the stage rounds: every used device sends then, the last device's first round too, R sets of buffers each (Model::ensure).
    auto host_for = [&](const std::vector<size_t>& used, size_t R) {
        Host h;
        // A layer split is pipelined, and every stage but the last sends its residual on; a drafter with stages may send from the head too, to stages the program puts apart from it, so its buffers are counted beside them.
        const size_t senders = used.size() < 2 ? 0 : R > 1 ? used.size() * R : used.size() - (fp.drafter_stages.empty() ? 1 : 0);
        h.need = logit_rows.value_or(rows) * fp.logits_per_row + fp.tables + fp.host_state + handoff_buffers(slots, true) * senders * rows * fp.handoff_per_row;
        h.need = backend::size_add(h.need, backend::size_mul(used.size(), backend::size_mul(arena_rows, fp.host_scratch_per_row)));
        // A pipelined prompt keeps a pass in flight a stage, a server its slots.
        const size_t in_flight = std::max(slots, used.size() * R);
        for (size_t l = 0; l < L; ++l) h.need = backend::size_add(h.need, backend::size_add(lookup_host[l], backend::size_mul(in_flight, lookup_pass[l])));
        for (size_t d : used) {
            h.need += devices[d].host_side;
            if (devices[d].host && h.carrier == SIZE_MAX) h.carrier = d;
        }
        return h;
    };
    const size_t first_host = (size_t)(std::find_if(devices.begin(), devices.end(), [](const DeviceBudget& d) { return d.host; }) - devices.begin());
    const std::optional<size_t> host_room = first_host < N ? devices[first_host].bytes : host_free;
    // A carrier's budget checks the host's needs with its own; without one they must fit the host's room.
    auto host_fits = [&](const Host& h) { return h.carrier != SIZE_MAX || !host_room || h.need <= *host_room; };

    // Besides weights and caches: a pass's activations, the scratch at the pass's widths, what the device keeps free and, where weights are copied, its tables; the carrier also holds the host's needs, its budget being the host's.
    // With the embedding, the encoder's buffers and the scratch of its passes, which a device priced per pass holds beside its own (reserve_scratch).
    auto overhead = [&](size_t d, const Host& h, bool first, bool last) {
        size_t copies = 0;
        if (!devices[d].host) {
            if (fp.table_buffers.empty()) copies = taken(d, fp.tables);
            for (size_t t : fp.table_buffers) copies = backend::size_add(copies, taken(d, t));
        }
        size_t priced = devices[d].priced ? devices[d].priced(pass_widths(fp, rows, logit_rows.value_or(rows), last)) : 0;
        if (first) {
            for (size_t b : fp.encoder.buffers) copies = backend::size_add(copies, taken(d, b));
            for (const backend::PassWidths& w : fp.encoder.widths) priced = backend::size_add(priced, devices[d].priced ? devices[d].priced(w) : 0);
        }
        return backend::size_add(arena_bytes(fp, rows, reply_rows, devices[d].allocated), arenas(d, devices[d].collective_per_row)) + copies + devices[d].scratch + priced +
               (d == h.carrier ? h.need : 0);
    };
    // On the head's device: a group member's slice of the logits rows, and the argmax passes' logits rows and their ids, each a buffer of its own (Model::ensure, Model::argmax_buffers).
    auto head_rows = [&](size_t d) {
        const size_t logits = logit_rows.value_or(rows), kept = devices[d].argmax && fp.argmax_per_row ? std::min(logits, fp.argmax_rows) : 0;
        const size_t ids = backend::size_mul(kept, sizeof(float));
        return backend::size_add(taken(d, backend::size_mul(logits, fp.head_slice_per_row)),
                                 backend::size_add(taken(d, backend::size_mul(kept, fp.argmax_per_row) - ids), taken(d, ids)));
    };
    // What device d holds running layers [i, i + k), given whether it is the first and the last device that runs layers.
    auto need = [&](size_t d, size_t i, size_t k, bool first, bool last, const Host& h) {
        return backend::size_add(backend::size_add(backend::size_add(prefix[d][i + k] - prefix[d][i], lookup_prefix[d][i + k] - lookup_prefix[d][i]), cached[d][i + k] - cached[d][i] + replicas(i, k) + (last ? fp.drafter_cache : 0)),
                                 backend::size_add(backend::size_add(end_weights(d, first, last), last ? head_rows(d) : 0), overhead(d, h, first, last)));
    };
    auto fits = [&](size_t d, size_t bytes) { return !devices[d].bytes || bytes <= *devices[d].bytes; };

    // The drafter's stages a device holds, [t, t + n): their weights as it keeps them, the drafter's embedding with stage 0, and the drafter's KV or state once where it holds any.
    std::vector<std::vector<size_t>> staged(N, std::vector<size_t>(S + 1, 0));
    for (size_t d = 0; d < N; ++d)
        for (size_t t = 0; t < S; ++t) {
            size_t b = 0;
            for (const Matrix& m : fp.drafter_stages[t]) b = backend::size_add(b, resident(d, m));
            staged[d][t + 1] = backend::size_add(staged[d][t], b);
        }
    auto stage_weights = [&](size_t d, size_t t, size_t n, bool first, bool last) {
        return n ? backend::size_add(staged[d][t + n] - staged[d][t], t || paired ? size_t(0) : drafted_embedding(d, first, last)) : size_t(0);
    };
    auto stage_bytes = [&](size_t d, size_t t, size_t n, bool first, bool last) { return n ? backend::size_add(stage_weights(d, t, n, first, last), fp.drafter_stage_cache) : size_t(0); };

    std::vector<int> count(N, 0);
    std::vector<int> drafts(N, 0);
    // What a split point before layer i would put on two devices: a tensor layers on both sides take, or none.
    const auto joined = [&](size_t i) -> const std::string* { return i < L && i < fp.joined.size() && !fp.joined[i].empty() ? &fp.joined[i] : nullptr; };
    const auto refuse_cut = [&](size_t i, const char* what) {
        if (const std::string* t = joined(i))
            throw std::runtime_error(std::string("split: ") + what + " put a split point before layer " + std::to_string(i) + ", and layers on both sides of it take " + *t);
    };
    if (!shares.empty()) {
        // Shares are proportions: each device's layers are its share of the total, rounded so the counts add to the model's layers, the remainders going to the largest fractions first.
        size_t sum = 0;
        for (int s : shares) sum += (size_t)s;
        std::vector<std::pair<size_t, size_t>> fraction;   // remainder numerator, device
        size_t given = 0;
        for (size_t d = 0; d < N; ++d) {
            count[d] = (int)(L * (size_t)shares[d] / sum);
            given += (size_t)count[d];
            fraction.push_back({L * (size_t)shares[d] % sum, d});
        }
        std::stable_sort(fraction.begin(), fraction.end(), [](const auto& x, const auto& y) { return x.first > y.first; });
        for (size_t i = 0; given < L; ++i, ++given) ++count[fraction[i].second];
        for (size_t d = 0, at = 0; d < N; ++d) refuse_cut(at += (size_t)count[d], "the layer shares");
    }
    {
        // A plan ranks by its layers on devices that read in place, its busiest device's layers (a pipeline's pace), its devices holding drafter stages apart from the head, then the busiest device's bytes (docs/src/model-layer_split.md).
        // The devices apart are a dimension of the program, each count keeping its best, and the stages go to the devices in order, any number on one, shares fixing the layers alone.
        using Load = std::pair<size_t, size_t>;
        using Score = std::pair<size_t, Load>;
        using Rank = std::pair<std::pair<size_t, size_t>, std::pair<size_t, size_t>>;   // host layers, busiest layers, devices apart, busiest bytes
        const Score none{SIZE_MAX, Load{SIZE_MAX, SIZE_MAX}};
        bool found = false;
        Rank best{};
        std::vector<int> fixed = count;
        for (size_t mask = 1; mask < ((size_t)1 << N); ++mask) {
            std::vector<size_t> used;
            bool ok = true;
            for (size_t d = 0; d < N; ++d) {
                if (mask >> d & 1) used.push_back(d);
                if (!shares.empty() && (fixed[d] > 0) != bool(mask >> d & 1)) ok = false;
            }
            const size_t J = used.size();
            if (!ok || J > L) continue;
            // The search charges one round; the rounds are chosen and checked on the plan it picks.
            const Host h = host_for(used, 1);
            if (!host_fits(h)) continue;
            // score at (j, i, t, c): the set's first j devices running layers [0, i), one at least each, and stages [0, t), c of those devices holding stages but not the head; from: the layers and stages the last of them took.
            const size_t C = S + 1, per = (L + 1) * (S + 1) * C;
            const auto cell = [&](size_t j, size_t i, size_t t, size_t c) { return j * per + (i * (S + 1) + t) * C + c; };
            std::vector<Score> score((J + 1) * per, none);
            std::vector<std::pair<size_t, size_t>> from((J + 1) * per);
            score[cell(0, 0, 0, 0)] = Score{0, Load{0, 0}};
            for (size_t j = 0; j < J; ++j) {
                const size_t d = used[j];
                const bool first = j == 0, last = j + 1 == J;
                for (size_t i = 0; i < L; ++i)
                    for (size_t t = 0; t <= S; ++t)
                        for (size_t c = 0; c <= t; ++c) {
                            const Score now = score[cell(j, i, t, c)];
                            if (now == none) continue;
                            // The last device runs the rest of the layers and the stages.
                            const size_t k0 = !shares.empty() ? (size_t)fixed[d] : last ? L - i : 1, k1 = !shares.empty() ? (size_t)fixed[d] : L - i;
                            for (size_t k = k0; k >= 1 && k <= k1 && i + k <= L; ++k) {
                                if (last && i + k != L) break;
                                if (joined(i + k)) continue;
                                const size_t base = need(d, i, k, first, last, h);
                                if (!fits(d, base)) break;
                                for (size_t n = last ? S - t : 0; t + n <= S; ++n) {
                                    const size_t bytes = backend::size_add(base, stage_bytes(d, t, n, first, last));
                                    if (!fits(d, bytes)) break;
                                    Score next = now;
                                    if (devices[d].host) next.first += k;
                                    else next.second = std::max(next.second, Load{k, bytes});
                                    const size_t to = cell(j + 1, i + k, t + n, c + (n && !last ? 1 : 0));
                                    if (next < score[to]) {
                                        score[to] = next;
                                        from[to] = {k, n};
                                    }
                                }
                            }
                        }
            }
            for (size_t c = 0; c <= S; ++c) {
                const Score& s = score[cell(J, L, S, c)];
                if (s == none) continue;
                const Rank rank{{s.first, s.second.first}, {c, s.second.second}};
                if (found && !(rank < best)) continue;
                found = true;
                best = rank;
                std::fill(count.begin(), count.end(), 0);
                std::fill(drafts.begin(), drafts.end(), 0);
                for (size_t j = J, i = L, t = S, e = c; j > 0; --j) {
                    const auto [k, n] = from[cell(j, i, t, e)];
                    count[used[j - 1]] = (int)k;
                    drafts[used[j - 1]] = (int)n;
                    i -= k;
                    t -= n;
                    if (n && j != J) --e;
                }
            }
        }
        if (!found && !shares.empty()) {
            // Shares the devices do not hold are refused below, by the device that is short, with the stages on the last device.
            count = fixed;
            for (size_t d = N; d-- > 0;)
                if (count[d]) {
                    drafts[d] = (int)S;
                    break;
                }
        } else if (!found) {
            throw std::runtime_error("split: the model does not fit: no placement of its " + std::to_string(L) + " layers" + (S ? " and " + std::to_string(S) + " drafter stages" : std::string()) +
                                     " fits the devices' free memory and the host's (a smaller cache budget or more devices leave more room)");
        }
    }

    LayerSplit out;
    out.stages.resize(N);
    std::vector<size_t> used;
    for (size_t d = 0; d < N; ++d) {
        out.stages[d].count = count[d];
        if (count[d]) used.push_back(d);
    }
    if (used.empty()) throw std::runtime_error("split: no device runs a layer");
    out.embed_device = (int)used.front();
    out.output_device = (int)used.back();
    // Stage rounds (docs/MULTI-DEVICE.md, Placement and balance): R runs of layers a device, `rounds` where given and else the rule, at least a layer a run and fewer where a device or the host does not hold them.
    // The rule takes rounds only where every crossing between the used devices is served on the devices, since a host round trip a crossing is what decode cannot pay; with one device there is nothing to take turns with.
    int least = INT32_MAX;
    for (size_t d : used) least = std::min(least, count[d]);
    bool served = true;
    for (size_t a : used)
        for (size_t b : used)
            served = served && (a == b || (b < devices[a].serves.size() && devices[a].serves[b]));
    size_t R = rounds ? rounds : served ? (size_t)std::max(1, least / kRunLayers) : 1;
    R = used.size() > 1 ? std::min(R, (size_t)least) : 1;
    // What device d holds running `runs`, its parts written into `st`.
    const auto hold = [&](size_t d, const std::vector<std::pair<int, int>>& runs, const Host& h, LayerSplit::Stage& st) {
        size_t before = 0;
        for (size_t e = 0; e < d; ++e) before += (size_t)drafts[e];
        st.weights = end_weights(d, d == used.front(), d == used.back()) + stage_weights(d, before, (size_t)drafts[d], d == used.front(), d == used.back());
        st.cache = (d == used.back() ? fp.drafter_cache : 0) + (drafts[d] ? fp.drafter_stage_cache : 0);
        st.other = overhead(d, h, d == used.front(), d == used.back());
        for (const auto& [f, n] : runs) {
            const size_t a = (size_t)f, b = (size_t)(f + n);
            st.weights += prefix[d][b] - prefix[d][a];
            st.cache += cached[d][b] - cached[d][a] + replicas(a, (size_t)n);
            st.other = backend::size_add(st.other, lookup_prefix[d][b] - lookup_prefix[d][a]);
        }
        return st.weights + st.cache + st.other;
    };
    // A paired drafter's stages run where their target layers do, in device order as draft_device lists them, and the head's device runs the last pair, which the correction reads.
    const auto pair = [&](const std::vector<std::vector<std::pair<int, int>>>& runs) {
        if (!paired) return true;
        std::fill(drafts.begin(), drafts.end(), 0);
        size_t prev = 0, at = N;
        for (size_t l : fp.drafter_pairs) {
            at = N;
            for (size_t d = 0; d < N && at == N; ++d)
                for (const auto& [f, n] : runs[d])
                    if (l >= (size_t)f && l < (size_t)(f + n)) at = d;
            if (at == N || at < prev) return false;
            ++drafts[at];
            prev = at;
        }
        return at == used.back();
    };
    for (; R > 1; --R) {
        const auto runs = round_runs(count, R);
        if (!pair(runs)) continue;
        const Host h = host_for(used, R);
        bool ok = host_fits(h);
        for (const auto& device_runs : runs)
            for (const auto& run : device_runs) ok = ok && !joined((size_t)run.first);
        LayerSplit::Stage scratch;
        for (size_t d : used) ok = ok && fits(d, hold(d, runs[d], h, scratch));
        if (ok) break;
    }
    out.rounds = R;
    const auto runs = round_runs(count, R);
    if (!pair(runs)) throw std::runtime_error("split: a paired drafter's correction reads target layer " + std::to_string(fp.drafter_pairs.back()) + ", which the head's device does not run (--layer-shares can move it there)");
    const Host h = host_for(used, R);
    for (size_t d : used) {
        LayerSplit::Stage& st = out.stages[d];
        st.runs = runs[d];
        const size_t held = hold(d, runs[d], h, st);
        if (!fits(d, held)) {
            char msg[256];
            std::snprintf(msg, sizeof msg, "split: %s needs %.2f GiB for layers %s and has %.2f GiB free", devices[d].name.c_str(),
                          held / 1073741824.0, runs_text(st.runs).c_str(), *devices[d].bytes / 1073741824.0);
            throw std::runtime_error(msg);
        }
    }
    for (size_t d = 0; d < N; ++d)
        for (int n = 0; n < drafts[d]; ++n) out.draft_device.push_back((int)d);
    if (!host_fits(h)) {
        char msg[160];
        std::snprintf(msg, sizeof msg, "split: the host needs %.2f GiB for logits, tables and staging and has %.2f GiB free",
                      h.need / 1073741824.0, *host_room / 1073741824.0);
        throw std::runtime_error(msg);
    }
    out.host = h.need;
    return out;
}

} // namespace infer
