#pragma once
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace infer {

// Physical CPU storage only. The sequence owns its valid length and reset;
// capacity retained across resets is never itself an attention history.
class HostKVCache {
public:
    HostKVCache() = default;
    HostKVCache(size_t layers, size_t heads, size_t width, size_t limit)
        : heads_(heads), width_(width), limit_(limit), k_(layers), v_(layers) {}

    void reserve(size_t want, size_t used) {
        if (used > capacity_ || want < used || want > limit_)
            throw std::runtime_error("KV cache: invalid sequence extent");
        if (want <= capacity_) return;
        const size_t capacity = std::max(want,
            capacity_ > limit_ / 2 ? limit_ : capacity_ * 2);
        std::vector<std::vector<float>> next_k(k_.size()), next_v(v_.size());
        for (size_t l = 0; l < k_.size(); ++l) {
            next_k[l].resize(heads_ * capacity * width_);
            next_v[l].resize(heads_ * capacity * width_);
            if (used) for (size_t h = 0; h < heads_; ++h) {
                std::copy_n(k_[l].data() + h * capacity_ * width_, used * width_,
                            next_k[l].data() + h * capacity * width_);
                std::copy_n(v_[l].data() + h * capacity_ * width_, used * width_,
                            next_v[l].data() + h * capacity * width_);
            }
        }
        k_.swap(next_k);
        v_.swap(next_v);
        capacity_ = capacity;
    }

    void write(size_t layer, const float* k, const float* v, size_t pos, size_t batch) {
        if (layer >= k_.size() || pos > capacity_ || batch > capacity_ - pos)
            throw std::runtime_error("KV cache: write outside storage");
        for (size_t h = 0; h < heads_; ++h) {
            float* kd = k_[layer].data() + (h * capacity_ + pos) * width_;
            float* vd = v_[layer].data() + (h * capacity_ + pos) * width_;
            for (size_t b = 0; b < batch; ++b) {
                std::copy_n(k + (b * heads_ + h) * width_, width_, kd + b * width_);
                std::copy_n(v + (b * heads_ + h) * width_, width_, vd + b * width_);
            }
        }
    }

    const float* keys(size_t layer) const { return k_.at(layer).data(); }
    const float* values(size_t layer) const { return v_.at(layer).data(); }
    size_t head_stride() const { return capacity_ * width_; }

private:
    size_t heads_ = 0, width_ = 0, limit_ = 0, capacity_ = 0;
    std::vector<std::vector<float>> k_, v_;
};

} // namespace infer
