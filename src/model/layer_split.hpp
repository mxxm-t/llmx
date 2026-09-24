#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// A model split by layers over several devices: which consecutive layers each device runs, fitted against what each device reports free (docs/MULTI-DEVICE.md, phase 1).
// Nothing here knows an architecture: the model describes what it asks of memory as a Footprint, and the split works on those numbers alone.

namespace infer {

// What a model asks of the memory of the devices it runs on, part by part; an architecture computes it from its file and options.
struct Footprint {
    std::vector<size_t> layer_weights;   // bytes of each layer's weights
    size_t embedding = 0;                // the embedding table
    size_t head = 0;                     // the head and what runs with it
    bool tied = false;                   // the head reads the embedding table, so a device holding both holds it once
    size_t cache_per_layer = 0;          // one layer's cache for every position the budget allows
    size_t tables = 0;                   // what every device that runs layers holds whatever its share, such as position tables
    size_t activations_per_row = 0;      // one row of a pass's activations on each device
};

// What a device offers a split: the bytes it reports free, and whether weights placed on it read the mapped file in place, as the CPU's do, rather than being copied into its memory.
// Zero bytes means the device could not tell, and it is not checked.
struct DeviceBudget {
    std::string name;
    size_t bytes = 0;
    bool host = false;
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
            if (!st.count)
                std::snprintf(line, sizeof line, "%s: no layers\n", devices[d].name.c_str());
            else
                std::snprintf(line, sizeof line, "%s: layers %d-%d, weights %.2f GiB, cache %.2f GiB, activations and scratch %.2f GiB, %.2f GiB free\n",
                              devices[d].name.c_str(), st.first, st.first + st.count - 1, st.weights / gib, st.cache / gib, st.other / gib,
                              devices[d].bytes / gib);
            s += line;
        }
        return s;
    }
};

// Consecutive layers per device in the order given: devices whose weights are copied share the layers as evenly as their budgets allow, and devices that read the mapped file in place take only what the others cannot hold.
// `shares`, when given, is each device's proportion of the layers and overrides the balance; the fit is still checked.
// What a device must hold: its layers' weights where it copies them and their caches; on the first and last devices that run layers the embedding and the head, once when a device holds both and they are tied; the tables; one pass of `rows` rows of activations; and on a device that copies weights a reserve for kernel scratch.
inline LayerSplit split_layers(const Footprint& fp, const std::vector<DeviceBudget>& devices, size_t rows, const std::vector<int>& shares = {}) {
    if (devices.empty()) throw std::runtime_error("split: no devices");
    const size_t L = fp.layer_weights.size(), N = devices.size();
    if (!L) throw std::runtime_error("split: a model without layers");
    size_t heaviest = 0;
    for (size_t w : fp.layer_weights) heaviest = std::max(heaviest, w);

    // The embedding and head weights a device copies, given whether it is the first and the last that runs layers.
    auto end_weights = [&](size_t d, bool first, bool last) -> size_t {
        if (devices[d].host) return 0;
        return (first ? fp.embedding : 0) + (last && !(first && fp.tied) ? fp.head : 0);
    };
    // Besides weights and caches: the tables, a pass's activations and, where weights are copied, a reserve for tile split partials, attention merge state and padded F32 copies, which stay well under it on the models here.
    auto overhead = [&](size_t d) {
        return fp.tables + rows * fp.activations_per_row + (devices[d].host ? 0 : ((size_t)256 << 20) + devices[d].bytes / 20);
    };
    auto per_layer = [&](size_t d) { return (devices[d].host ? 0 : heaviest) + fp.cache_per_layer; };
    auto unbounded = [&](size_t d) { return devices[d].bytes == 0; };

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
        // Which devices run layers decides which carry the embedding and the head, so the fit repeats until that set stops changing.
        std::vector<bool> active(N, true);
        for (size_t round = 0; round <= N; ++round) {
            size_t first = N, last = 0;
            for (size_t d = 0; d < N; ++d)
                if (active[d]) { first = std::min(first, d); last = d; }
            if (first == N) break;
            std::vector<size_t> cap(N, 0);
            for (size_t d = 0; d < N; ++d) {
                if (!active[d]) continue;
                if (unbounded(d)) { cap[d] = L; continue; }
                const size_t f = end_weights(d, d == first, d == last) + overhead(d);
                cap[d] = devices[d].bytes > f ? std::min(L, (devices[d].bytes - f) / per_layer(d)) : 0;
            }
            std::fill(count.begin(), count.end(), 0);
            // Devices that copy weights first, filled level by level so none carries more than it must.
            size_t left = L;
            std::vector<size_t> copying, hosts;
            for (size_t d = 0; d < N; ++d)
                if (active[d]) (devices[d].host ? hosts : copying).push_back(d);
            while (left && !copying.empty()) {
                std::vector<size_t> open;
                for (size_t d : copying) if ((size_t)count[d] < cap[d]) open.push_back(d);
                if (open.empty()) break;
                const size_t each = std::max<size_t>(1, left / open.size());
                for (size_t d : open) {
                    const size_t take = std::min({each, cap[d] - (size_t)count[d], left});
                    count[d] += (int)take;
                    left -= take;
                    if (!left) break;
                }
            }
            // Then the host devices, evenly, for what the others could not hold.
            for (size_t i = 0; left && i < hosts.size(); ++i) {
                const size_t d = hosts[i];
                const size_t want = (left + (hosts.size() - i) - 1) / (hosts.size() - i);
                const size_t take = std::min({want, cap[d], left});
                count[d] += (int)take;
                left -= take;
            }
            if (left)
                throw std::runtime_error("split: the model does not fit: " + std::to_string(left) + " of " + std::to_string(L) +
                                         " layers have no room (each takes " + std::to_string((heaviest + fp.cache_per_layer) >> 20) +
                                         " MiB with its cache; a smaller cache budget or more devices leave more room)");
            std::vector<bool> next(N);
            for (size_t d = 0; d < N; ++d) next[d] = count[d] > 0;
            if (next == active) break;
            active = next;
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
        for (int i = 0; i < st.count; ++i)
            if (!devices[d].host) st.weights += fp.layer_weights[(size_t)(st.first + i)];
        st.weights += end_weights(d, (int)d == first_used, (int)d == last_used);
        st.cache = (size_t)st.count * fp.cache_per_layer;
        st.other = overhead(d);
        const size_t need = st.weights + st.cache + st.other;
        if (!unbounded(d) && need > devices[d].bytes) {
            char msg[256];
            std::snprintf(msg, sizeof msg, "split: %s needs %.2f GiB for layers %d-%d and has %.2f GiB free", devices[d].name.c_str(),
                          need / 1073741824.0, st.first, st.first + st.count - 1, devices[d].bytes / 1073741824.0);
            throw std::runtime_error(msg);
        }
    }
    return out;
}

} // namespace infer
