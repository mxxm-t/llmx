// Include the implementation to drive a tensor group's sum kernel (sum_rows) on one device, at every source count up to the widest group.
#include "backends/vulkan/vulkan_backend.cpp"
#include "group_sums.hpp"
#include <cstring>
#include <iostream>
#include <memory>

namespace {

void require(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(what);
}

// Every source count from 1 to backend::kMaxTensorWidth, a buffer a source, added into a residual, written over it and kept, each the oracle's bits (tests/group_sums.hpp).
// Written over its first source, as a sum in place is, the sum is read before it is written.
size_t check_sources(backend::VulkanBackend& b) {
    using backend::CSlice;
    using backend::Slice;
    size_t checks = 0;
    uint32_t seed = 7;
    for (size_t W = 1; W <= backend::kMaxTensorWidth; ++W)
        for (size_t n : {size_t(1), size_t(255), size_t(256), size_t(257), size_t(1000)})
            for (size_t offset : {size_t(0), size_t(7)})
                for (int mode = 0; mode < 4; ++mode) {
                    const bool replace = mode & 1, keep = mode & 2;
                    const group_sums::Case k = group_sums::make(W, n, seed++);
                    std::vector<backend::BufferPtr> src;
                    std::vector<CSlice> sources;
                    for (size_t m = 0; m < W; ++m) {
                        src.push_back(b.alloc(n * sizeof(float), backend::Memory::device));
                        b.write(*src.back(), 0, k.partial[m].data(), n * sizeof(float));
                        sources.push_back({src.back().get(), 0});
                    }
                    const backend::BufferPtr x = b.alloc((n + offset) * sizeof(float), backend::Memory::device), kept = b.alloc(n * sizeof(float), backend::Memory::device);
                    b.write(*x, offset * sizeof(float), k.start.data(), n * sizeof(float));
                    const Slice kept_rows{kept.get(), 0};
                    b.sum_rows(Slice{x.get(), offset}, sources.data(), W, n, keep ? &kept_rows : nullptr, replace);
                    b.submit();
                    std::vector<float> got(n);
                    b.read(*x, offset * sizeof(float), got.data(), n * sizeof(float));
                    const std::vector<float> want = group_sums::oracle(k, replace), sum = group_sums::oracle(k, true);
                    require(!std::memcmp(got.data(), want.data(), n * sizeof(float)), "a sum of sources is not the oracle's in member order");
                    if (keep) {
                        b.read(*kept, 0, got.data(), n * sizeof(float));
                        require(!std::memcmp(got.data(), sum.data(), n * sizeof(float)), "a kept sum is not the oracle's in member order");
                    }
                    checks += 1 + keep;
                }
    // In place over source 0: each member's own rows are its residual and its first source.
    for (size_t W : {size_t(2), size_t(5), backend::kMaxTensorWidth}) {
        const size_t n = 1000;
        const group_sums::Case k = group_sums::make(W, n, seed++);
        std::vector<backend::BufferPtr> src;
        std::vector<CSlice> sources;
        for (size_t m = 0; m < W; ++m) {
            src.push_back(b.alloc(n * sizeof(float), backend::Memory::device));
            b.write(*src.back(), 0, k.partial[m].data(), n * sizeof(float));
            sources.push_back({src.back().get(), 0});
        }
        b.sum_rows(Slice{src[0].get(), 0}, sources.data(), W, n, nullptr, true);
        b.submit();
        std::vector<float> got(n);
        b.read(*src[0], 0, got.data(), n * sizeof(float));
        const std::vector<float> want = group_sums::oracle(k, true);
        require(!std::memcmp(got.data(), want.data(), n * sizeof(float)), "a sum written over its first source is not the oracle's");
        ++checks;
    }
    bool refused = false;
    try {
        std::vector<CSlice> many(backend::kMaxTensorWidth + 1);
        const backend::BufferPtr x = b.alloc(sizeof(float), backend::Memory::device);
        b.sum_rows(Slice{x.get(), 0}, many.data(), many.size(), 1, nullptr);
    } catch (const std::logic_error&) {
        refused = true;
    }
    require(refused, "a sum of more sources than the widest group was taken");
    return checks + 1;
}

} // namespace

int main() {
    try {
        auto owner = std::make_unique<backend::VulkanBackend>(0);
        const size_t checks = check_sources(*owner);
        std::cout << "vulkan-sum: " << checks << " sums of 1 to " << backend::kMaxTensorWidth << " sources in member order\n";
        return 0;
    } catch (const backend::VulkanUnavailable& e) {
        std::cerr << e.what() << '\n';
        return 77;
    } catch (const std::exception& e) {
        std::cerr << "vulkan-sum: " << e.what() << '\n';
        return 1;
    }
}
