#include <iostream>
#include "model/host_kv_cache.hpp"

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
float expected(size_t layer, size_t head, size_t pos, size_t lane, int seed) {
    return float(seed * 1000000 + layer * 100000 + pos * 1000 + head * 128 + lane);
}
void check(const infer::HostKVCache& cache, size_t used, size_t width, int seed) {
    const size_t stride = cache.head_stride();
    require(stride % width == 0 && stride / width >= used && stride / width <= 67, "capacity bound");
    for (size_t layer = 0; layer < 3; ++layer)
        for (size_t head = 0; head < 3; ++head)
            for (size_t pos = 0; pos < used; ++pos)
                for (size_t lane = 0; lane < width; ++lane) {
                    const auto offset = head * stride + pos * width + lane;
                    const float value = expected(layer, head, pos, lane, seed);
                    require(cache.keys(layer)[offset] == value, "key history changed");
                    require(cache.values(layer)[offset] == -value, "value history changed");
                }
}
void append(infer::HostKVCache& cache, size_t pos, size_t batch, size_t width, int seed) {
    for (size_t layer = 0; layer < 3; ++layer) {
        std::vector<float> k(batch * 3 * width), v(k.size());
        for (size_t b = 0; b < batch; ++b)
            for (size_t h = 0; h < 3; ++h)
                for (size_t d = 0; d < width; ++d) {
                    const auto index = (b * 3 + h) * width + d;
                    k[index] = expected(layer, h, pos + b, d, seed);
                    v[index] = -k[index];
                }
        cache.write(layer, k.data(), v.data(), pos, batch);
    }
}
template<class F> void rejects(F fn) {
    bool failed = false;
    try { fn(); } catch (const std::exception&) { failed = true; }
    require(failed, "invalid extent accepted");
}
int main() {
    try {
        for (size_t width : {size_t(1), size_t(8), size_t(42), size_t(128)}) {
            infer::HostKVCache cache(3, 3, width, 67);
            require(cache.head_stride() == 0, "cache allocated maximum context at startup");
            size_t used = 0;
            for (size_t want : {1, 2, 3, 7, 8, 16, 17, 31, 32, 33, 66, 67}) {
                cache.reserve(want, used);
                check(cache, used, width, 1);
                append(cache, used, want - used, width, 1);
                used = want;
                check(cache, used, width, 1);
            }
            const size_t stride = cache.head_stride();
            rejects([&] { cache.reserve(68, used); });
            rejects([&] { cache.reserve(67, 68); });
            rejects([&] { cache.reserve(1, used); });
            rejects([&] { cache.write(0, nullptr, nullptr, 66, 2); });
            rejects([&] { cache.write(3, nullptr, nullptr, 0, 1); });
            check(cache, used, width, 1);
            // Reset is a sequence operation: retain storage but write a new,
            // shorter history, with no assumption that the unused tail is zero.
            cache.reserve(3, 0);
            append(cache, 0, 3, width, 2);
            require(cache.head_stride() == stride, "reset discarded retained capacity");
            check(cache, 3, width, 2);
        }
        std::cout << "KV storage: layer/head/lane isolation, growth, retained reset and bounds pass\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
