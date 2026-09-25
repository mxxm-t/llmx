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

// A model split by layers over several devices: which consecutive layers each device runs, fitted against what each device reports free (docs/MULTI-DEVICE.md, phase 1).
// Nothing here knows an architecture: the model describes what it asks of memory as a Footprint, and the split works on those numbers alone.

namespace infer {

// One weight matrix as a device would adopt it: its quant type, its row width and row count, its bytes in the file, and whether a matrix product reads it as its weights (a projection or the head) rather than a gather, a norm or a routed expert stack.
struct Matrix {
    uint32_t type = 0;
    size_t nin = 0, rows = 0, bytes = 0;
    bool product = false;
};

// What a model asks of the memory of the devices it runs on, part by part; an architecture computes it from its file and options.
struct Footprint {
    std::vector<std::vector<Matrix>> layers;   // each layer's weights
    Matrix embedding;                          // the embedding table
    Matrix output;                             // the head's matrix, the embedding table itself when tied
    Matrix output_norm;                        // what runs with the head besides its matrix
    bool tied = false;                         // the head reads the embedding table, adopted once where both sit on one device
    size_t cache_per_layer = 0;                // one layer's cache for every position the budget allows
    size_t tables = 0;                         // position tables: the host keeps them while the model lives, and every device that copies weights holds its own
    size_t activations_per_row = 0;            // one row of a pass's activations on each device
    size_t logits_per_row = 0;                 // one row of logits where the head runs
    size_t handoff_per_row = 0;                // one row of the stream handed from one device to the next through host memory
};

// What a device offers a split: the bytes it reports free, or nothing when it cannot tell and is not checked; whether weights placed on it read the mapped file in place, as the CPU's do, rather than being copied into its memory; and what adopting a matrix keeps resident there, its bytes when empty.
// `host_side` is host memory the device's backend holds for itself, such as upload staging, which counts against the host.
struct DeviceBudget {
    std::string name;
    std::optional<size_t> bytes;
    bool host = false;
    std::function<size_t(const Matrix&)> resident;
    size_t host_side = 0;
};

// A budget for each backend, named as its caller names it: what the backend reports free, whether it reads weights in place, what adopting a matrix keeps on it, and its own host memory.
inline std::vector<DeviceBudget> budgets_for(const std::vector<backend::BackendPtr>& backends, const std::vector<std::string>& names) {
    if (names.size() != backends.size()) throw std::runtime_error("split: a name for every device");
    std::vector<DeviceBudget> budgets;
    for (size_t d = 0; d < backends.size(); ++d) {
        const backend::Backend* b = backends[d].get();
        budgets.push_back(DeviceBudget{names[d], b->memory_available(), b->reads_in_place(),
                                       [b](const Matrix& w) { return b->resident_bytes(w.type, w.nin, w.rows, w.bytes, w.product); },
                                       b->host_resident()});
    }
    return budgets;
}

// Per device, in the order given, the consecutive layers it runs and what it was fitted to hold; the embedding goes with the first device that runs layers and the head with the last.
struct LayerSplit {
    struct Stage {
        int first = 0, count = 0;   // layers [first, first + count)
        size_t weights = 0, cache = 0, other = 0;
    };
    std::vector<Stage> stages;
    int embed_device = 0, output_device = 0;
    size_t host = 0;   // what the host holds besides mapped weights: logits rows, position tables, the handoff between devices and the used backends' own host memory

    std::string describe(const std::vector<DeviceBudget>& devices) const {
        std::string s;
        char line[256];
        const double gib = 1024.0 * 1024.0 * 1024.0;
        for (size_t d = 0; d < stages.size(); ++d) {
            const Stage& st = stages[d];
            char free[48] = "free memory unknown";
            if (devices[d].bytes) std::snprintf(free, sizeof free, "%.2f GiB free", *devices[d].bytes / gib);
            if (!st.count)
                std::snprintf(line, sizeof line, "%s: no layers\n", devices[d].name.c_str());
            else
                std::snprintf(line, sizeof line, "%s: layers %d-%d, weights %.2f GiB, cache %.2f GiB, activations and scratch %.2f GiB, %s\n",
                              devices[d].name.c_str(), st.first, st.first + st.count - 1, st.weights / gib, st.cache / gib, st.other / gib, free);
            s += line;
        }
        std::snprintf(line, sizeof line, "host: logits, tables and staging %.2f GiB\n", host / gib);
        return s + line;
    }
};

// Consecutive layers per device in the order given.
// Every set of devices that could run layers is tried, each of its devices running at least one: the first carries the embedding and the last the head, and a small dynamic program assigns the layers by their actual sizes, the fewest layers on devices that read the mapped file in place, then the lightest busiest device, every device within its budget.
// The plan kept is the best of those sets by the same order; devices are a machine's few cards and its CPU, so trying every set stays small.
// `shares`, when given, is each device's proportion of the layers and overrides the balance; the fit is still checked.
// What a device must hold: its layers' weights as it keeps them resident and their caches; on the first and last devices the embedding and the head, once when a device holds both and they are tied; `rows` rows of activations; and where it copies weights its own tables and a reserve for kernel scratch.
// What the host must hold for a set: `rows` rows of logits, the position tables, the handoff when more than one device runs layers, and the host memory of each backend in the set; on the set's first host device, whose budget is the host's memory, else within the first host device listed or, with none listed, `host_free`.
inline LayerSplit split_layers(const Footprint& fp, const std::vector<DeviceBudget>& devices, size_t rows, const std::vector<int>& shares = {},
                               std::optional<size_t> host_free = std::nullopt) {
    if (devices.empty()) throw std::runtime_error("split: no devices");
    const size_t L = fp.layers.size(), N = devices.size();
    if (!L) throw std::runtime_error("split: a model without layers");
    if (N > 16) throw std::runtime_error("split: more than 16 devices");

    auto resident = [&](size_t d, const Matrix& m) -> size_t {
        if (devices[d].host) return 0;
        return devices[d].resident ? devices[d].resident(m) : m.bytes;
    };
    // prefix[d][i]: the resident bytes of layers [0, i) on device d.
    std::vector<std::vector<size_t>> prefix(N, std::vector<size_t>(L + 1, 0));
    for (size_t d = 0; d < N; ++d)
        for (size_t l = 0; l < L; ++l) {
            size_t b = 0;
            for (const Matrix& m : fp.layers[l]) b += resident(d, m);
            prefix[d][l + 1] = prefix[d][l] + b;
        }
    // The embedding and head weights a device keeps, given whether it runs the first and the last layers.
    // A tied pair on one device is one buffer, kept as the head keeps it, since the head reads it through a product.
    auto end_weights = [&](size_t d, bool first, bool last) -> size_t {
        if (first && last && fp.tied) return resident(d, fp.output) + resident(d, fp.output_norm);
        return (first ? resident(d, fp.embedding) : 0) + (last ? resident(d, fp.output) + resident(d, fp.output_norm) : 0);
    };

    // What the host holds for a set of devices running layers, and the device that carries it: the set's first host device, or none.
    struct Host {
        size_t need = 0, carrier = SIZE_MAX;
    };
    auto host_for = [&](const std::vector<size_t>& used) {
        Host h;
        h.need = rows * fp.logits_per_row + fp.tables + (used.size() > 1 ? rows * fp.handoff_per_row : 0);
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

    // Besides weights and caches: a pass's activations and, where weights are copied, the device's own tables and a reserve for tile split partials and attention merge state; the carrier also holds the host's needs, since its budget is the host's memory.
    auto overhead = [&](size_t d, const Host& h) {
        const size_t copies = devices[d].host ? 0 : fp.tables + ((size_t)256 << 20) + devices[d].bytes.value_or(0) / 20;
        return rows * fp.activations_per_row + copies + (d == h.carrier ? h.need : 0);
    };
    // What device d holds running layers [i, i + k), given whether it is the first and the last device that runs layers.
    auto need = [&](size_t d, size_t i, size_t k, bool first, bool last, const Host& h) {
        return prefix[d][i + k] - prefix[d][i] + k * fp.cache_per_layer + end_weights(d, first, last) + overhead(d, h);
    };
    auto fits = [&](size_t d, size_t bytes) { return !devices[d].bytes || bytes <= *devices[d].bytes; };

    std::vector<int> count(N, 0);
    if (!shares.empty()) {
        if (shares.size() != N) throw std::runtime_error("split: " + std::to_string(shares.size()) + " layer shares for " + std::to_string(N) + " devices");
        // Shares are proportions: each device's layers are its share of the total, rounded so the counts add to the model's layers, the remainders going to the largest fractions first.
        size_t sum = 0;
        for (int s : shares) {
            if (s < 0) throw std::runtime_error("split: a negative layer share");
            sum += (size_t)s;
        }
        if (!sum) throw std::runtime_error("split: every layer share is zero");
        std::vector<std::pair<size_t, size_t>> fraction;   // remainder numerator, device
        size_t given = 0;
        for (size_t d = 0; d < N; ++d) {
            count[d] = (int)(L * (size_t)shares[d] / sum);
            given += (size_t)count[d];
            fraction.push_back({L * (size_t)shares[d] % sum, d});
        }
        std::stable_sort(fraction.begin(), fraction.end(), [](const auto& x, const auto& y) { return x.first > y.first; });
        for (size_t i = 0; given < L; ++i, ++given) ++count[fraction[i].second];
    } else {
        // A plan is better with fewer layers on devices that read in place, then a lighter busiest device.
        using Score = std::pair<size_t, size_t>;
        const Score none{SIZE_MAX, SIZE_MAX};
        bool found = false;
        Score best = none;
        for (size_t mask = 1; mask < ((size_t)1 << N); ++mask) {
            std::vector<size_t> used;
            for (size_t d = 0; d < N; ++d)
                if (mask >> d & 1) used.push_back(d);
            const size_t J = used.size();
            if (J > L) continue;
            const Host h = host_for(used);
            if (!host_fits(h)) continue;
            // score[j][i]: the set's first j devices running layers [0, i), one at least each; from[j][i]: how many the last of them took.
            std::vector<std::vector<Score>> score(J + 1, std::vector<Score>(L + 1, none));
            std::vector<std::vector<size_t>> from(J + 1, std::vector<size_t>(L + 1, 0));
            score[0][0] = Score{0, 0};
            for (size_t j = 0; j < J; ++j) {
                const size_t d = used[j];
                const bool first = j == 0, last = j + 1 == J;
                for (size_t i = 0; i < L; ++i) {
                    if (score[j][i] == none) continue;
                    // The last device runs the rest.
                    for (size_t k = last ? L - i : 1; i + k <= L; ++k) {
                        const size_t bytes = need(d, i, k, first, last, h);
                        if (!fits(d, bytes)) break;
                        Score next = score[j][i];
                        if (devices[d].host) next.first += k;
                        else next.second = std::max(next.second, bytes);
                        if (next < score[j + 1][i + k]) {
                            score[j + 1][i + k] = next;
                            from[j + 1][i + k] = k;
                        }
                        if (last) break;
                    }
                }
            }
            if (score[J][L] == none || (found && !(score[J][L] < best))) continue;
            found = true;
            best = score[J][L];
            std::fill(count.begin(), count.end(), 0);
            for (size_t j = J, i = L; j > 0; --j) {
                count[used[j - 1]] = (int)from[j][i];
                i -= from[j][i];
            }
        }
        if (!found)
            throw std::runtime_error("split: the model does not fit: no placement of its " + std::to_string(L) +
                                     " layers fits the devices' free memory and the host's (a smaller cache budget or more devices leave more room)");
    }

    LayerSplit out;
    out.stages.resize(N);
    std::vector<size_t> used;
    int at = 0;
    for (size_t d = 0; d < N; ++d) {
        out.stages[d].first = at;
        out.stages[d].count = count[d];
        if (count[d]) used.push_back(d);
        at += count[d];
    }
    if (used.empty()) throw std::runtime_error("split: no device runs a layer");
    out.embed_device = (int)used.front();
    out.output_device = (int)used.back();
    const Host h = host_for(used);
    for (size_t d : used) {
        LayerSplit::Stage& st = out.stages[d];
        st.weights = prefix[d][(size_t)(st.first + st.count)] - prefix[d][(size_t)st.first] + end_weights(d, d == used.front(), d == used.back());
        st.cache = (size_t)st.count * fp.cache_per_layer;
        st.other = overhead(d, h);
        const size_t held = st.weights + st.cache + st.other;
        if (!fits(d, held)) {
            char msg[256];
            std::snprintf(msg, sizeof msg, "split: %s needs %.2f GiB for layers %d-%d and has %.2f GiB free", devices[d].name.c_str(),
                          held / 1073741824.0, st.first, st.first + st.count - 1, *devices[d].bytes / 1073741824.0);
            throw std::runtime_error(msg);
        }
    }
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
