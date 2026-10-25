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
    // The scores score_rows writes for `rows` query rows of `heads` x `dim` floats after `hist` rows of history, against `groups` pooled keys, a row of `groups` apart.
    static std::vector<float> score(VulkanBackend& b, const std::vector<float>& query, const std::vector<float>& pooled, size_t rows, size_t groups, uint32_t hist,
                                    uint32_t heads, uint32_t dim, uint32_t compression) {
        const auto qb = b.adopt(query.data(), query.size() * sizeof(float));
        const auto kb = b.adopt(pooled.data(), pooled.size() * sizeof(float));
        std::vector<float> out(rows * groups, -12345.0f);
        const auto o = b.adopt(out.data(), out.size() * sizeof(float));
        // The pooled keys as a plane of blocks of one group each and no token rows, block g holding group g.
        std::vector<uint32_t> table = {1u, 0u, 0u, uint32_t(rows), hist, 0u, uint32_t(groups)};
        for (size_t g = 0; g < groups; ++g) table.push_back(uint32_t(g));
        b.score_rows(b.bind(CSlice{qb.get(), 0}), b.bind(CSlice{kb.get(), 0}), VulkanBackend::PlaneShape{compression, dim, 0}, b.bind(CSlice{o.get(), 0}), table, 0, 0,
                     rows, groups, heads, dim, compression, 1.0f / std::sqrt(float(dim)));
        b.read(*o, 0, out.data(), out.size() * sizeof(float));
        return out;
    }
    static std::vector<std::pair<std::string, std::string>> representations(VulkanBackend& b) { return b.kernel_representations(); }
    static bool preserves_float32(VulkanBackend& b) { return b.dev_->preserve_float32; }
    static uint32_t width_limit(VulkanBackend& b) { return b.dev_->props.limits.maxComputeWorkGroupCount[0]; }
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

// The scores by the indexer's sum, in F32 and in its order (Backend::select_kv): per head the products on 256 slots, +0.0 past the head's width, halved k = 128 to 1, red[i] + red[i + k];
// the head's dot rectified, the heads summed in order and the sum scaled; a row's groups past what it sees are left as they were. The test is built without contraction, so no product fuses into an add.
std::vector<float> score_reference(const std::vector<float>& query, const std::vector<float>& pooled, size_t rows, size_t groups, uint32_t hist, uint32_t heads,
                                   uint32_t dim, uint32_t compression) {
    std::vector<float> out(rows * groups, -12345.0f);
    const float scale = 1.0f / std::sqrt(float(dim));
    for (size_t r = 0; r < rows; ++r) {
        const size_t visible = (hist + r + 1) / compression;
        for (size_t g = 0; g < groups && g < visible; ++g) {
            float score = 0.0f;
            for (uint32_t h = 0; h < heads; ++h) {
                float red[256];
                for (uint32_t j = 0; j < 256; ++j) red[j] = j < dim ? query[(r * heads + h) * dim + j] * pooled[g * dim + j] : 0.0f;
                for (uint32_t k = 128; k > 0; k >>= 1)
                    for (uint32_t i = 0; i < k; ++i) red[i] = red[i] + red[i + k];
                score += red[0] > 0.0f ? red[0] : 0.0f;
            }
            out[r * groups + g] = score * scale;
        }
    }
    return out;
}

// The scoring against score_reference bit for bit: every score of every row, the scores past a row's groups left unwritten.
// Small values give -0 and equal products, magnitudes of 2^-50 to 2^60 and an infinity large and nonfinite sums, and, where the device keeps F32 subnormals, magnitudes down to 2^-80 subnormal ones.
int scoring(backend::VulkanBackend& b) {
    uint32_t seed = 99;
    const auto next = [&seed] { seed = seed * 1664525u + 1013904223u; return seed >> 8; };
    const bool subnormal = backend::VulkanIndexTest::preserves_float32(b);
    size_t values = 0, cases = 0;
    for (const uint32_t heads : {1u, 3u, 4u, 8u})
        for (const uint32_t dim : {32u, 96u, 128u, 200u, 256u})
            for (const size_t groups : {size_t(1), size_t(7), size_t(64), size_t(65), size_t(1000), size_t(4100)})
                for (const int kind : {0, 1, 2}) {
                    if (kind == 2 && !subnormal) continue;
                    const uint32_t compression = 4;
                    const size_t rows = 3;
                    const uint32_t hist = uint32_t(groups * compression - rows);
                    std::vector<float> query(rows * heads * dim), pooled(groups * dim);
                    const auto value = [&](int k) {
                        if (k == 0) return float(int(next() % 9u) - 4) / 4.0f;
                        if (k == 1) return std::ldexp(float(int(next() % 2001u) - 1000), int(next() % 101u) - 50);
                        return std::ldexp(float(int(next() % 2001u) - 1000), int(next() % 61u) - 80);
                    };
                    for (float& v : query) v = value(kind);
                    for (float& v : pooled) v = value(kind);
                    if (kind == 1 && !pooled.empty()) pooled[next() % pooled.size()] = std::numeric_limits<float>::infinity();
                    const auto want = score_reference(query, pooled, rows, groups, hist, heads, dim, compression);
                    const auto got = backend::VulkanIndexTest::score(b, query, pooled, rows, groups, hist, heads, dim, compression);
                    size_t at = 0;
                    while (at < want.size() && bits_of(want[at]) == bits_of(got[at])) ++at;
                    require(at == want.size(), "scores differ from the reference at " + std::to_string(at) + " of " + std::to_string(heads) + " heads " +
                                                   std::to_string(dim) + " wide over " + std::to_string(groups) + " groups, values of kind " + std::to_string(kind));
                    values += want.size();
                    ++cases;
                }
    std::cout << "vulkan-index: " << cases << " scorings, " << values << " scores bit for bit the reference's" << (subnormal ? ", subnormal ones included" : ", no subnormal ones on this device") << "\n";
    // A row of 65537 tiles of groups, past the 65535 workgroups a device may limit a dispatch's width to: where it does, the scoring takes its tiles in two dispatches.
    const size_t groups = size_t(65537) * 8;
    const uint32_t heads = 2, dim = 32, compression = 4;
    std::vector<float> query(heads * dim), pooled(groups * dim);
    for (float& v : query) v = float(int(next() % 9u) - 4) / 4.0f;
    for (float& v : pooled) v = float(int(next() % 9u) - 4) / 4.0f;
    const uint32_t hist = uint32_t(groups * compression - 1);
    const auto want = score_reference(query, pooled, 1, groups, hist, heads, dim, compression);
    const auto got = backend::VulkanIndexTest::score(b, query, pooled, 1, groups, hist, heads, dim, compression);
    size_t at = 0;
    while (at < want.size() && bits_of(want[at]) == bits_of(got[at])) ++at;
    require(at == want.size(), "scores past 65535 tiles differ from the reference at group " + std::to_string(at));
    std::cout << "vulkan-index: " << groups << " groups in 65537 tiles, every score bit for bit the reference's, the device's dispatch width limit "
              << backend::VulkanIndexTest::width_limit(b) << "\n";
    return 0;
}

// The tiled scoring dispatched as its own kernel, its device time printed, and in its ISA, where the driver gives it, no fused or contracted float multiply-add.
int scoring_build() {
    backend::VulkanBackend t(0, true);
    for (const size_t groups : {size_t(1025), size_t(4096), size_t(16384), size_t(38912)}) {
        std::vector<float> q(4 * 128), k(groups * 128);
        for (size_t i = 0; i < q.size(); ++i) q[i] = float(int(i % 7) - 3) / 8.0f;
        for (size_t i = 0; i < k.size(); ++i) k[i] = float(int(i % 11) - 5) / 16.0f;
        backend::VulkanIndexTest::score(t, q, k, 1, groups, uint32_t(groups * 4 - 1), 4, 128, 4);
        t.kernel_times();
        for (int i = 0; i < 5; ++i) backend::VulkanIndexTest::score(t, q, k, 1, groups, uint32_t(groups * 4 - 1), 4, 128, 4);
        double ms = 0;
        bool tiled = false;
        for (const auto& kt : t.kernel_times()) {
            tiled = tiled || kt.first == "kv_index_scores_tile";
            if (kt.first.rfind("kv_index_scores", 0) == 0) ms += kt.second;
        }
        if (!t.timed_dispatches()) {
            std::cout << "vulkan-index: this device times no dispatches, so the scoring's kernel and time are not checked\n";
            return 0;
        }
        if (!tiled) {
            std::cerr << "vulkan-index: the scoring did not dispatch kv_index_scores_tile\n";
            return 1;
        }
        std::cout << "vulkan-index: scoring of " << groups << " groups, " << ms / 5 << " ms of device time a call\n";
    }
    for (const auto& kr : backend::VulkanIndexTest::representations(t)) {
        if (kr.first != "kv_index_scores_tile") continue;
        size_t fused = 0;
        bool isa = false;
        for (size_t at = 0; at < kr.second.size();) {
            const size_t end = std::min(kr.second.find('\n', at), kr.second.size());
            const std::string line = kr.second.substr(at, end - at);
            at = end + 1;
            const size_t op = line.find_first_not_of(" \t");
            if (op == std::string::npos) continue;
            const auto starts = [&](const char* name) { return line.compare(op, std::strlen(name), name) == 0; };
            isa = isa || starts("v_mul_f32");
            const bool f32 = line.find("f32", op) != std::string::npos;
            if (((starts("v_fma") || starts("v_fmac")) && f32) || starts("v_mad_mix_f32") || starts("v_mad_f32") || starts("v_mac_f32") || starts("v_mad_legacy_f32") ||
                starts("v_mac_legacy_f32"))
                ++fused;
        }
        std::cout << "vulkan-index: kv_index_scores_tile " << (isa ? std::to_string(fused) + " fused float multiply-adds in its ISA" : std::string("ISA not read from this driver's representation")) << "\n";
        if (isa && fused) return 1;
    }
    return 0;
}

// The wide selection's dispatches, five whatever the groups against the one-workgroup selection's one, and each selection's device time, printed and not asserted, up to 38912 groups, Flash's at a 155648-token history.
int timing() {
    backend::VulkanBackend t(0, true);
    int failures = 0;
    for (const size_t count : {size_t(1025), size_t(2048), size_t(3072), size_t(4096), size_t(5120), size_t(6144), size_t(8192), size_t(16384), size_t(38912)}) {
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
            if (!dispatches) {
                std::cout << "vulkan-index: this device times no dispatches, so the selections' dispatches and times are not checked\n";
                return failures;
            }
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
        return run(b) + scoring(b) + scoring_build() + timing() ? 1 : 0;
    } catch (const backend::VulkanUnavailable& e) {
        std::cerr << e.what() << '\n';
        return 77;
    } catch (const std::exception& e) {
        std::cerr << "vulkan-index: " << e.what() << '\n';
        return 1;
    }
}
