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

// A model split by layers over several devices: which consecutive layers each device runs, fitted against what each device reports free (docs/MULTI-DEVICE.md, phase 1).
// Nothing here knows an architecture: the model describes what it asks of memory as a Footprint, and the split works on those numbers alone.

namespace infer {

// One weight matrix as a device would adopt it: its quant type, its row width and row count, and its bytes in the file.
struct Matrix {
    uint32_t type = 0;
    size_t nin = 0, rows = 0, bytes = 0;
};

// What a model asks of the memory of the devices it runs on, part by part; an architecture computes it from its file and options.
struct Footprint {
    std::vector<std::vector<Matrix>> layers;   // each layer's weights
    Matrix embedding;                          // the embedding table
    Matrix output;                             // the head's matrix, the embedding table itself when tied
    Matrix output_norm;                        // what runs with the head besides its matrix
    bool tied = false;                         // the head reads the embedding table, adopted once where both sit on one device
    size_t cache_per_layer = 0;                // one layer's cache for every position the budget allows
    size_t tables = 0;                         // what every device that runs layers holds whatever its share, such as position tables
    size_t activations_per_row = 0;            // one row of a pass's activations on each device
};

// What a device offers a split: the bytes it reports free, or nothing when it cannot tell and is not checked; whether weights placed on it read the mapped file in place, as the CPU's do, rather than being copied into its memory; and what adopting a matrix keeps resident there, its bytes when empty.
struct DeviceBudget {
    std::string name;
    std::optional<size_t> bytes;
    bool host = false;
    std::function<size_t(const Matrix&)> resident;
};

// Per device, in the order given, the consecutive layers it runs and what it was fitted to hold; the embedding goes with the first device that runs layers and the head with the last.
struct LayerSplit {
    struct Stage {
        int first = 0, count = 0;   // layers [first, first + count)
        size_t weights = 0, cache = 0, other = 0;
    };
    std::vector<Stage> stages;
    int embed_device = 0, output_device = 0;

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
        return s;
    }
};

// Consecutive layers per device in the order given.
// Devices that copy weights share the layers as evenly as their budgets allow, and devices that read the mapped file in place take only what the others cannot hold.
// Every choice of the first and last device to run layers is tried, those two carrying the embedding and the head and at least one layer each, and the one kept puts the fewest layers on devices that read in place, then uses the most devices that copy, then loads the busiest of those least.
// `shares`, when given, is each device's proportion of the layers and overrides the balance; the fit is still checked.
// What a device must hold: its layers' weights as it keeps them resident and their caches; on the first and last devices the embedding and the head, once when a device holds both and they are tied; the tables; `rows` rows of activations; and on a device that copies weights a reserve for kernel scratch.
inline LayerSplit split_layers(const Footprint& fp, const std::vector<DeviceBudget>& devices, size_t rows, const std::vector<int>& shares = {}) {
    if (devices.empty()) throw std::runtime_error("split: no devices");
    const size_t L = fp.layers.size(), N = devices.size();
    if (!L) throw std::runtime_error("split: a model without layers");

    auto resident = [&](size_t d, const Matrix& m) -> size_t {
        if (devices[d].host) return 0;
        return devices[d].resident ? devices[d].resident(m) : m.bytes;
    };
    std::vector<std::vector<size_t>> layer_bytes(N, std::vector<size_t>(L, 0));
    std::vector<size_t> heaviest(N, 0);
    for (size_t d = 0; d < N; ++d)
        for (size_t l = 0; l < L; ++l) {
            for (const Matrix& m : fp.layers[l]) layer_bytes[d][l] += resident(d, m);
            heaviest[d] = std::max(heaviest[d], layer_bytes[d][l]);
        }
    // The embedding and head weights a device keeps, given whether it runs the first and the last layers.
    auto end_weights = [&](size_t d, bool first, bool last) -> size_t {
        size_t w = first ? resident(d, fp.embedding) : 0;
        if (last) w += resident(d, fp.output_norm) + (first && fp.tied ? 0 : resident(d, fp.output));
        return w;
    };
    // Besides weights and caches: the tables, a pass's activations and, where weights are copied, a reserve for tile split partials and attention merge state.
    auto overhead = [&](size_t d) {
        const size_t reserve = devices[d].host ? 0 : ((size_t)256 << 20) + devices[d].bytes.value_or(0) / 20;
        return fp.tables + rows * fp.activations_per_row + reserve;
    };
    auto per_layer = [&](size_t d) { return heaviest[d] + fp.cache_per_layer; };
    // Layers device d can hold beside what it carries anyway, the model's all when it cannot tell.
    auto capacity = [&](size_t d, bool first, bool last) -> size_t {
        if (!devices[d].bytes) return L;
        const size_t fixed = end_weights(d, first, last) + overhead(d);
        return *devices[d].bytes > fixed ? std::min(L, (*devices[d].bytes - fixed) / per_layer(d)) : 0;
    };

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
        // A plan is better with fewer layers on devices that read in place, then more devices that copy, then a lighter busiest device.
        struct Score { size_t host_layers, copying_used, busiest; };
        auto better = [](const Score& a, const Score& b) {
            if (a.host_layers != b.host_layers) return a.host_layers < b.host_layers;
            if (a.copying_used != b.copying_used) return a.copying_used > b.copying_used;
            return a.busiest < b.busiest;
        };
        bool found = false;
        Score best{};
        for (size_t f = 0; f < N; ++f)
            for (size_t l = f; l < N; ++l) {
                if (f != l && L < 2) continue;
                std::vector<size_t> cap(N, 0);
                for (size_t d = f; d <= l; ++d) cap[d] = capacity(d, d == f, d == l);
                if (!cap[f] || !cap[l]) continue;
                std::vector<int> c(N, 0);
                c[f] = 1;
                c[l] = 1;
                size_t left = L - (f == l ? 1 : 2);
                // Devices that copy weights first, filled level by level so none carries more than it must.
                std::vector<size_t> copying, hosts;
                for (size_t d = f; d <= l; ++d) (devices[d].host ? hosts : copying).push_back(d);
                while (left) {
                    std::vector<size_t> open;
                    for (size_t d : copying) if ((size_t)c[d] < cap[d]) open.push_back(d);
                    if (open.empty()) break;
                    const size_t each = std::max<size_t>(1, left / open.size());
                    for (size_t d : open) {
                        const size_t take = std::min({each, cap[d] - (size_t)c[d], left});
                        c[d] += (int)take;
                        left -= take;
                        if (!left) break;
                    }
                }
                // Then the host devices, evenly, for what the others could not hold.
                for (size_t i = 0; left && i < hosts.size(); ++i) {
                    const size_t d = hosts[i];
                    const size_t want = (left + (hosts.size() - i) - 1) / (hosts.size() - i);
                    const size_t take = std::min({want, cap[d] - (size_t)c[d], left});
                    c[d] += (int)take;
                    left -= take;
                }
                if (left) continue;
                Score s{0, 0, 0};
                for (size_t d = f; d <= l; ++d) {
                    if (devices[d].host) s.host_layers += (size_t)c[d];
                    else if (c[d]) {
                        ++s.copying_used;
                        s.busiest = std::max(s.busiest, (size_t)c[d] * per_layer(d));
                    }
                }
                if (!found || better(s, best)) {
                    found = true;
                    best = s;
                    count = c;
                }
            }
        if (!found) {
            size_t each = 0;
            for (size_t d = 0; d < N; ++d) each = std::max(each, per_layer(d));
            throw std::runtime_error("split: the model does not fit: no placement of its " + std::to_string(L) +
                                     " layers fits the devices (each takes up to " + std::to_string(each >> 20) +
                                     " MiB with its cache; a smaller cache budget or more devices leave more room)");
        }
    }

    LayerSplit out;
    out.stages.resize(N);
    int at = 0, first_used = -1, last_used = -1;
    for (size_t d = 0; d < N; ++d) {
        out.stages[d].first = at;
        out.stages[d].count = count[d];
        if (count[d]) {
            if (first_used < 0) first_used = (int)d;
            last_used = (int)d;
        }
        at += count[d];
    }
    if (first_used < 0) throw std::runtime_error("split: no device runs a layer");
    out.embed_device = first_used;
    out.output_device = last_used;
    for (size_t d = 0; d < N; ++d) {
        LayerSplit::Stage& st = out.stages[d];
        if (!st.count) continue;
        for (int i = 0; i < st.count; ++i) st.weights += layer_bytes[d][(size_t)(st.first + i)];
        st.weights += end_weights(d, (int)d == first_used, (int)d == last_used);
        st.cache = (size_t)st.count * fp.cache_per_layer;
        st.other = overhead(d);
        const size_t need = st.weights + st.cache + st.other;
        if (devices[d].bytes && need > *devices[d].bytes) {
            char msg[256];
            std::snprintf(msg, sizeof msg, "split: %s needs %.2f GiB for layers %d-%d and has %.2f GiB free", devices[d].name.c_str(),
                          need / 1073741824.0, st.first, st.first + st.count - 1, *devices[d].bytes / 1073741824.0);
            throw std::runtime_error(msg);
        }
    }
    return out;
}

} // namespace infer
