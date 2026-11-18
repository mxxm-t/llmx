#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "format/gguf.hpp"
#include "model/arch/metadata.hpp"

// The rotary position tables every text architecture reads and the rope keys of its file: one writer of the tables, one reader of the scaling keys and the command line's request, and the policy they resolve to (docs/src/model-rope.md).

namespace infer::rope {

// How an architecture spells a pair's base frequency; the two spellings differ in the last bit for about a quarter of the pairs, so each module keeps its own and its tables stay as they were.
enum class Base {
    negative_power,    // pow(theta, -2i/d)
    reciprocal_power,  // 1 / pow(theta, 2i/d)
};

inline float base_frequency(Base form, float theta, size_t i, size_t dim) {
    if (form == Base::negative_power) return std::pow(theta, -2.0f * (float)i / (float)dim);
    return 1.0f / std::pow(theta, 2.0f * (float)i / (float)dim);
}

// The precision a module's tables are computed in: F32 in HF's order of operations, or each frequency and angle in double, rounded to F32 once.
enum class Arithmetic { f32_hf, f64_once };

// YaRN's attention factor on the tables: HF's 0.1 ln(factor) + 1 on cos and sin, or none where the module's reference applies none.
enum class Attention { hf, none };

// What the command line asks (--yarn, --yarn-original-ctx); empty takes the file's own rope.
struct Request {
    std::optional<float> factor;
    std::optional<size_t> original;
};

// What a module hands over, none of it named for an architecture: its base-frequency spelling, the base, the rotated width, the file's context and its two forms.
struct Inputs {
    Base form = Base::negative_power;
    float theta = 10000.0f;
    size_t dim = 0;
    size_t context = 0;
    Arithmetic arithmetic = Arithmetic::f32_hf;
    Attention attention = Attention::hf;
};

// The rope a model runs: the inputs, YaRN's parameters when it is on, the context it defines, and what the file declared and the command line asked, for the record.
struct Policy {
    Inputs in;
    bool yarn = false;
    float factor = 1, beta_fast = 32, beta_slow = 1, attention = 1;
    size_t original = 0, context = 0;
    bool declared_yarn = false;
    float declared_factor = 1;
    size_t declared_original = 0;
    Request requested;
    bool forced = false;  // the request made the rope other than the file declares

    // The line a load prints on stderr where YaRN is on; empty otherwise.
    std::string describe() const;
    // The text a history copied off the devices is keyed on beside the model file; empty for the unscaled rope, whose tables the file alone gives.
    std::string identity() const;
};

namespace detail {
inline std::string g(double v) {
    char b[32];
    std::snprintf(b, sizeof b, "%.9g", v);
    return b;
}
inline std::string over(float factor, size_t original) { return "yarn factor " + g(factor) + " over " + std::to_string(original); }
}  // namespace detail

inline std::string Policy::describe() const {
    if (!yarn) return {};
    const std::string scale = in.attention == Attention::none ? "attention none" : "attention x" + detail::g(attention);
    std::string s = "rope: " + detail::over(factor, original) + " -> context " + std::to_string(context) + ", " + scale + " (model declares " +
                    (declared_yarn ? detail::over(declared_factor, declared_original) : std::string("none")) + ")";
    if (forced && declared_yarn) s += " (forced by --yarn)";
    if (forced && !declared_yarn) s += " (warning: YaRN forced on a model that does not declare it; its quality is measured, not assumed)";
    return s + "\n";
}

inline std::string Policy::identity() const {
    if (!yarn) return {};
    return "rope yarn factor " + detail::g(factor) + " original " + std::to_string(original) + " beta " + detail::g(beta_fast) + " " + detail::g(beta_slow) + " attention " +
           detail::g(attention) + "\n";
}

// The unscaled rope of `in`, for tables no scaling key reaches.
inline Policy unscaled(const Inputs& in) {
    Policy pol;
    pol.in = in;
    pol.context = in.context;
    return pol;
}

// A file's rope keys under the prefix `p` and the request: the unscaled rope, or YaRN as the file declares it or the request sets it, every refusal made before anything is allocated.
// With --yarn the context is floor(factor x original), the original being --yarn-original-ctx, else the file's declared one, else the file's context.
inline Policy resolve(const gguf::GGUFModel& m, const std::string& p, const Inputs& in, const Request& request) {
    const auto refuse = [](const std::string& what) { return std::runtime_error("inference: " + what); };
    if (!std::isfinite(in.theta) || in.theta <= 0 || in.dim == 0 || in.dim % 2 || in.context == 0) throw refuse("invalid rope base, width or context");
    Policy pol;
    pol.in = in;
    pol.requested = request;
    pol.context = in.context;
    const auto* type = m.find(p + "rope.scaling.type");
    if (type && (type->vtype != gguf::V_STRING || (type->s != "none" && type->s != "yarn"))) throw refuse("unsupported metadata " + p + "rope.scaling.type");
    const double factor = metadata::real(m, p + "rope.scaling.factor", 1);
    if (metadata::real(m, p + "rope.scale_linear", 1) != 1) throw refuse("scaled RoPE is unsupported");
    if (type && type->s == "yarn") {
        if (!(factor > 1)) throw refuse(p + "rope.scaling.factor must be above 1 for YaRN");
        pol.declared_yarn = true;
        pol.declared_factor = float(factor);
        pol.declared_original = (size_t)metadata::integer(m, p + "rope.scaling.original_context_length");
        // A declared original past the context is allowed, as the reference implementations apply the ramp all the same.
        pol.beta_fast = float(metadata::real(m, p + "rope.scaling.yarn_beta_fast", 32));
        pol.beta_slow = float(metadata::real(m, p + "rope.scaling.yarn_beta_slow", 1));
        if (!(pol.beta_fast > pol.beta_slow)) throw refuse(p + "rope.scaling.yarn_beta_fast must be above yarn_beta_slow");
        // Keys whose meaning differs between writers: read only once a file shows what it means.
        for (const char* key : {"rope.scaling.attn_factor", "rope.scaling.yarn_ext_factor", "rope.scaling.yarn_attn_factor", "rope.scaling.yarn_log_multiplier"})
            if (m.find(p + key)) throw refuse("unsupported metadata " + p + key);
        pol.yarn = true, pol.factor = pol.declared_factor, pol.original = pol.declared_original;
    } else if (factor != 1) {
        throw refuse("scaled RoPE is unsupported");
    }
    if (request.factor) {
        if (!std::isfinite(*request.factor) || !(*request.factor > 1)) throw refuse("--yarn must be a finite factor above 1");
        const size_t original = request.original ? *request.original : pol.declared_yarn ? pol.declared_original : in.context;
        // Only the user's --yarn-original-ctx is held to the context; a declared original was accepted as the file gave it.
        if (request.original && (original == 0 || original > in.context))
            throw refuse("--yarn-original-ctx " + std::to_string(original) + " is past the model's context of " + std::to_string(in.context));
        const double context = std::floor(double(*request.factor) * double(original));
        if (context > double(std::numeric_limits<int>::max())) throw refuse("--yarn " + detail::g(*request.factor) + " over " + std::to_string(original) + " is past the context limit");
        pol.yarn = true, pol.factor = *request.factor, pol.original = original;
        pol.context = (size_t)context;
        pol.forced = !pol.declared_yarn || pol.factor != pol.declared_factor || pol.original != pol.declared_original || pol.context != in.context;
    }
    if (pol.yarn && in.attention == Attention::hf) pol.attention = float(0.1 * std::log(double(pol.factor)) + 1.0);
    return pol;
}

// The policy on a module's configuration as it is built: the unscaled rope follows the configuration's base, width and context, and YaRN keeps the file's context it was resolved from.
inline Policy bind(Policy pol, const Inputs& in) {
    const size_t file_context = pol.in.context;
    pol.in = in;
    pol.context = in.context;
    if (pol.yarn) pol.in.context = file_context;
    return pol;
}

// The pairs' frequencies the tables rotate by, each held exactly in double: the base frequency, or under YaRN its blend with the interpolated one along the ramp between the correction dims.
// F32 follows HF's _compute_yarn_parameters operation for operation; double follows model.py's precompute_freqs_cis, rounded once by fill.
inline std::vector<double> frequencies(const Policy& pol) {
    const size_t dim = pol.in.dim, half = dim / 2;
    const bool f64 = pol.in.arithmetic == Arithmetic::f64_once;
    const double theta = pol.in.theta;
    std::vector<double> freq(half);
    for (size_t i = 0; i < half; ++i)
        freq[i] = !f64                                   ? double(base_frequency(pol.in.form, pol.in.theta, i, dim))
                  : pol.in.form == Base::negative_power ? std::pow(theta, -double(2 * i) / double(dim))
                                                         : 1.0 / std::pow(theta, double(2 * i) / double(dim));
    if (!pol.yarn) return freq;
    const auto correction = [&](double rotations) { return double(dim) * std::log(double(pol.original) / (rotations * 2 * 3.14159265358979323846)) / (2 * std::log(theta)); };
    const double low = std::max(std::floor(correction(pol.beta_fast)), 0.0);
    double high = std::min(std::ceil(correction(pol.beta_slow)), double(dim - 1));
    if (f64) {
        for (size_t i = 0; i < half; ++i) {
            const double ramp = std::clamp((double(i) - low) / std::max(high - low, 1e-3), 0.0, 1.0), smooth = 1 - ramp;
            freq[i] = freq[i] / double(pol.factor) * (1 - smooth) + freq[i] * smooth;
        }
        return freq;
    }
    if (low == high) high += 0.001;
    for (size_t i = 0; i < half; ++i) {
        const float power = std::pow(pol.in.theta, (float)(2 * i) / (float)dim);
        const float extra = 1.0f / power, inter = 1.0f / (pol.factor * power);
        const float ramp = std::min(std::max(((float)i - (float)low) / (float)(high - low), 0.0f), 1.0f);
        const float extrapolate = 1.0f - ramp;
        freq[i] = double(inter * (1.0f - extrapolate) + extra * extrapolate);
    }
    return freq;
}

// The cos and sin of every position the tables cover over the dim/2 rotated pairs, indexed as [pos * (dim / 2) + i], the attention factor folded into the one write; the fit sizes the tables to the positions the KV budget reaches.
inline void fill(const Policy& pol, std::vector<float>& cos, std::vector<float>& sin) {
    const size_t half = pol.in.dim / 2, positions = cos.size() / half;
    const std::vector<double> wide = frequencies(pol);
    const float scale = pol.attention;
    if (pol.in.arithmetic == Arithmetic::f64_once) {
        for (size_t pos = 0; pos < positions; ++pos)
            for (size_t i = 0; i < half; ++i) {
                cos[pos * half + i] = float(std::cos(double(pos) * wide[i])) * scale;
                sin[pos * half + i] = float(std::sin(double(pos) * wide[i])) * scale;
            }
        return;
    }
    const std::vector<float> freq(wide.begin(), wide.end());
    if (scale == 1.0f) {
        for (size_t pos = 0; pos < positions; ++pos)
            for (size_t i = 0; i < half; ++i) {
                cos[pos * half + i] = std::cos((float)pos * freq[i]);
                sin[pos * half + i] = std::sin((float)pos * freq[i]);
            }
        return;
    }
    for (size_t pos = 0; pos < positions; ++pos)
        for (size_t i = 0; i < half; ++i) {
            cos[pos * half + i] = std::cos((float)pos * freq[i]) * scale;
            sin[pos * half + i] = std::sin((float)pos * freq[i]) * scale;
        }
}

}  // namespace infer::rope
