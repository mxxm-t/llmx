// Include the implementation to drive the sparse-attention indexer's kernels on chosen inputs, without a model or a cache.
#include "backends/vulkan/vulkan_backend.cpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>

namespace backend {
namespace {
struct VulkanIndexTest {
    // The ids select_rows writes for `rows` query rows after `hist` rows of history, their scores laid out a row of `groups` apart.
    static std::vector<uint32_t> select(VulkanBackend& b, const std::vector<float>& scores, size_t rows, size_t groups, uint32_t hist, uint32_t compression,
                                        uint32_t budget, bool wide) {
        const uint32_t stride = budget + compression - 1;
        const auto s = b.adopt(scores.data(), scores.size() * sizeof(float));
        std::vector<uint32_t> out(rows * stride, 0x5eed5eedu);
        const auto o = b.adopt(out.data(), out.size() * sizeof(uint32_t));
        const std::vector<uint32_t> table = {1u, 0u, 0u, uint32_t(rows), hist, 0u, 0u};
        b.select_rows(b.bind(CSlice{s.get(), 0}), b.bind(CSlice{o.get(), 0}), table, 0, 0, rows, groups, compression, budget, stride, wide);
        b.read(*o, 0, out.data(), out.size() * sizeof(uint32_t));
        return out;
    }
};
}  // namespace
}  // namespace backend

namespace {
void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
}

uint32_t bits_of(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

float float_of(uint32_t u) {
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

// The selection by its contract (Backend::select_kv): up to budget / compression complete groups by their raw score bits, the earlier group on equal bits, emitted chronologically, then the causal tail, then UINT32_MAX.
std::vector<uint32_t> reference(const std::vector<float>& scores, size_t rows, size_t groups, uint32_t hist, uint32_t compression, uint32_t budget) {
    const uint32_t stride = budget + compression - 1;
    std::vector<uint32_t> out(rows * stride, 0xffffffffu);
    for (size_t r = 0; r < rows; ++r) {
        const size_t visible = hist + r + 1, count = visible / compression, take = std::min<size_t>(count, budget / compression);
        std::vector<size_t> order(count);
        for (size_t i = 0; i < count; ++i) order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t c) { return bits_of(scores[r * groups + a]) > bits_of(scores[r * groups + c]); });
        std::vector<size_t> picked(order.begin(), order.begin() + (std::ptrdiff_t)take);
        std::sort(picked.begin(), picked.end());
        size_t slot = 0;
        for (size_t g : picked)
            for (uint32_t t = 0; t < compression; ++t) out[r * stride + slot++] = uint32_t(g * compression + t);
        for (size_t t = 0; t < visible % compression; ++t) out[r * stride + slot++] = uint32_t(count * compression + t);
    }
    return out;
}

// One case: both selections against the contract, every id of every row.
size_t check(backend::VulkanBackend& b, const std::vector<float>& scores, size_t rows, size_t groups, uint32_t hist, uint32_t compression, uint32_t budget,
             const std::string& what) {
    const auto want = reference(scores, rows, groups, hist, compression, budget);
    for (const bool wide : {false, true}) {
        const auto got = backend::VulkanIndexTest::select(b, scores, rows, groups, hist, compression, budget, wide);
        size_t at = 0;
        while (at < want.size() && got[at] == want[at]) ++at;
        require(at == want.size(), what + (wide ? ", the wide selection" : ", the one-workgroup selection") + ", first differing id at " + std::to_string(at));
    }
    return want.size();
}

int run(backend::VulkanBackend& b) {
    uint32_t seed = 2026;
    const auto next = [&seed] { seed = seed * 1664525u + 1013904223u; return seed >> 8; };
    const float specials[] = {std::numeric_limits<float>::quiet_NaN(), float_of(0xffc00000u), std::numeric_limits<float>::infinity(), -0.0f, 0.0f,
                              float_of(1u), 1.0f};
    size_t ids = 0, cases = 0;
    // Flash's compression of 4 and budget of 2048, 512 groups taken, and a compression of 1, where every row of a batch sees one more group.
    for (const uint32_t compression : {4u, 1u}) {
        for (const size_t count : {size_t(1023), size_t(1024), size_t(1025), size_t(2047), size_t(2048), size_t(2049), size_t(5000), size_t(38912)}) {
            for (const uint32_t budget : {2048u, 4u * compression, uint32_t(count + 3) * compression}) {
                for (const int kind : {0, 1, 2}) {
                    const size_t rows = compression == 1 ? 3 : 2;
                    // The last row sees `count` complete groups; earlier rows see fewer, and their scores past them are left as garbage the selection must not read.
                    const uint32_t hist = uint32_t(count * compression - rows);
                    std::vector<float> scores(rows * count);
                    for (float& s : scores) s = float_of(0x7f7fffffu - (next() % 7u));
                    for (size_t r = 0; r < rows; ++r) {
                        const size_t visible_groups = (hist + r + 1) / compression;
                        for (size_t g = 0; g < visible_groups; ++g) {
                            float& s = scores[r * count + g];
                            if (kind == 0) s = float(next() % 5u);   // five levels, ties everywhere
                            else if (kind == 1) s = specials[next() % 7u];   // NaN of both signs, +inf, -0, +0, the smallest subnormal and 1
                            else s = float(next() % 1000u) / 7.0f;
                        }
                        // Equal top scores either side of the slice boundary, so the earlier one decides where the take ends.
                        if (kind == 0 && visible_groups > 1025) scores[r * count + 1023] = scores[r * count + 1024] = 9.0f;
                    }
                    ids += check(b, scores, rows, count, hist, compression, budget,
                                 "compression " + std::to_string(compression) + ", " + std::to_string(count) + " groups, budget " + std::to_string(budget) +
                                     ", scores of kind " + std::to_string(kind));
                    ++cases;
                }
            }
        }
    }
    std::cout << "vulkan-index: " << cases << " selections, " << ids << " ids equal to the contract's in both the one-workgroup and the wide selection\n";
    return 0;
}

// The wide selection's dispatches, five whatever the groups against the one-workgroup selection's one, and each selection's device time, printed and not asserted, up to 38912 groups, Flash's at a 155648-token history.
int timing() {
    backend::VulkanBackend t(0, true);
    int failures = 0;
    for (const size_t count : {size_t(1025), size_t(2048), size_t(4096), size_t(8192), size_t(16384), size_t(38912)}) {
        std::vector<float> scores(count);
        uint32_t seed = 7;
        for (float& s : scores) { seed = seed * 1664525u + 1013904223u; s = float(seed >> 8) / 16777216.0f; }
        for (const bool wide : {false, true}) {
            backend::VulkanIndexTest::select(t, scores, 1, count, uint32_t(count * 4 - 1), 4, 2048, wide);
            t.kernel_times();
            for (int i = 0; i < 5; ++i) backend::VulkanIndexTest::select(t, scores, 1, count, uint32_t(count * 4 - 1), 4, 2048, wide);
            double ms = 0;
            for (const auto& k : t.kernel_times())
                if (k.first.rfind("kv_index_select", 0) == 0) ms += k.second;
            const size_t dispatches = t.timed_dispatches() / 5;
            std::cout << "vulkan-index: " << (wide ? "wide" : "one-workgroup") << " selection of 512 of " << count << " groups, " << ms / 5
                      << " ms of device time a call over " << dispatches << " dispatches\n";
            if (dispatches != (wide ? 5u : 1u)) {
                std::cerr << "vulkan-index: a " << (wide ? "wide" : "one-workgroup") << " selection took " << dispatches << " dispatches\n";
                ++failures;
            }
        }
    }
    return failures;
}
}  // namespace

int main() {
    try {
        backend::VulkanBackend b(0);
        return run(b) + timing() ? 1 : 0;
    } catch (const backend::VulkanUnavailable& e) {
        std::cerr << e.what() << '\n';
        return 77;
    } catch (const std::exception& e) {
        std::cerr << "vulkan-index: " << e.what() << '\n';
        return 1;
    }
}
