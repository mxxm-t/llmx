// The rope owner (src/model/rope.hpp): its unscaled tables against the loops each architecture wrote before, bit for bit, YaRN against HF's formulas written here in double precision, and the file's keys and the request with their refusals.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "format/gguf.hpp"
#include "model/rope.hpp"

namespace {

size_t checks = 0;

// qwen3's loop: the frequency recomputed at every position.
void qwen3_tables(float theta, int dim, int context, std::vector<float>& c, std::vector<float>& s) {
    const int half = dim / 2;
    for (int pos = 0; pos < context; pos++)
        for (int i = 0; i < half; i++) {
            float fre = std::pow(theta, -2.0f * (float)i / (float)dim);
            c[(size_t)pos * half + i] = std::cos((float)pos * fre);
            s[(size_t)pos * half + i] = std::sin((float)pos * fre);
        }
}

// qwen35's loop: the frequencies first.
void qwen35_tables(float theta, int dim, int context, std::vector<float>& c, std::vector<float>& s) {
    const size_t half = (size_t)dim / 2;
    std::vector<float> freq(half);
    for (size_t i = 0; i < half; ++i) freq[i] = std::pow(theta, -2.0f * (float)i / (float)dim);
    for (size_t pos = 0; pos < (size_t)context; ++pos)
        for (size_t i = 0; i < half; ++i) {
            c[pos * half + i] = std::cos((float)pos * freq[i]);
            s[pos * half + i] = std::sin((float)pos * freq[i]);
        }
}

// qwen4exp's loop: the reciprocal of the positive power, pair by pair.
void qwen4exp_tables(float theta, int dim, int context, std::vector<float>& c, std::vector<float>& s) {
    const size_t half = size_t(dim / 2);
    for (size_t i = 0; i < half; ++i) {
        const float freq = 1.0f / std::pow(theta, 2.0f * float(i) / float(dim));
        for (size_t pos = 0; pos < size_t(context); ++pos) {
            c[pos * half + i] = std::cos(float(pos) * freq);
            s[pos * half + i] = std::sin(float(pos) * freq);
        }
    }
}

// deepseek41's loop (model.py's precompute_freqs_cis): the frequencies in double, YaRN's ramp with original above 0, each angle in double rounded once.
std::vector<double> deepseek_frequencies(int dim, double base, int original, double factor, double beta_fast, double beta_slow) {
    std::vector<double> f(size_t(dim / 2));
    for (size_t i = 0; i < f.size(); ++i) f[i] = 1.0 / std::pow(base, double(2 * i) / double(dim));
    if (original <= 0) return f;
    const double pi = 3.14159265358979323846;
    const auto corrected = [&](double turns) { return double(dim) * std::log(double(original) / (turns * 2 * pi)) / (2 * std::log(base)); };
    const double low = std::max(std::floor(corrected(beta_fast)), 0.0), high = std::min(std::ceil(corrected(beta_slow)), double(dim - 1));
    for (size_t i = 0; i < f.size(); ++i) {
        const double ramp = std::clamp((double(i) - low) / std::max(high - low, 1e-3), 0.0, 1.0), smooth = 1 - ramp;
        f[i] = f[i] / factor * (1 - smooth) + f[i] * smooth;
    }
    return f;
}

using Oracle = void (*)(float, int, int, std::vector<float>&, std::vector<float>&);

void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
    ++checks;
}

void identical(const std::string& name, const std::vector<float>& oc, const std::vector<float>& os, const std::vector<float>& nc, const std::vector<float>& ns, int dim) {
    const size_t n = oc.size();
    for (size_t k = 0; k < n; ++k)
        if (std::memcmp(&oc[k], &nc[k], sizeof(float)) || std::memcmp(&os[k], &ns[k], sizeof(float))) {
            char buf[160];
            std::snprintf(buf, sizeof buf, ": entry %zu (position %zu, pair %zu) differs", k, k / (size_t)(dim / 2), k % (size_t)(dim / 2));
            throw std::runtime_error(name + buf);
        }
    checks += 2 * n;
}

void same(const char* name, Oracle old, infer::rope::Base form, float theta, int dim, int context) {
    const size_t n = (size_t)context * (size_t)(dim / 2);
    std::vector<float> oc(n), os(n), nc(n), ns(n);
    old(theta, dim, context, oc, os);
    infer::rope::fill(infer::rope::unscaled({form, theta, (size_t)dim, (size_t)context}), nc, ns);
    identical(std::string(name) + " theta " + std::to_string(theta) + " dim " + std::to_string(dim), oc, os, nc, ns, dim);
}

// The two spellings of the base frequency are different numbers for some pairs, so neither may stand in for the other.
void forms_differ() {
    using infer::rope::Base;
    size_t differ = 0;
    for (size_t i = 0; i < 64; ++i) {
        const float a = infer::rope::base_frequency(Base::negative_power, 1e7f, i, 128);
        const float b = infer::rope::base_frequency(Base::reciprocal_power, 1e7f, i, 128);
        differ += std::memcmp(&a, &b, sizeof(float)) != 0;
    }
    require(differ != 0, "the two base-frequency spellings agree at theta 1e7 width 128");
}

gguf::MetaValue text(const std::string& s) {
    gguf::MetaValue v;
    v.vtype = gguf::V_STRING, v.s = s;
    return v;
}
gguf::MetaValue real(double x) {
    gguf::MetaValue v;
    v.vtype = gguf::V_FLOAT64, v.f64 = x;
    return v;
}
gguf::MetaValue count(uint64_t n) {
    gguf::MetaValue v;
    v.vtype = gguf::V_UINT32, v.u = n;
    return v;
}

// A file declaring YaRN at `factor` over `original` under the prefix "x.".
gguf::GGUFModel declaring(double factor, uint64_t original) {
    gguf::GGUFModel m;
    m.kv = {{"x.rope.scaling.type", text("yarn")}, {"x.rope.scaling.factor", real(factor)}, {"x.rope.scaling.original_context_length", count(original)}};
    return m;
}

void refuses(const std::string& name, const std::string& part, const gguf::GGUFModel& m, const infer::rope::Inputs& in, const infer::rope::Request& r) {
    try {
        (void)infer::rope::resolve(m, "x.", in, r);
    } catch (const std::runtime_error& e) {
        require(std::string(e.what()).find(part) != std::string::npos, name + ": refused with '" + e.what() + "', not '" + part + "'");
        return;
    }
    throw std::runtime_error(name + ": not refused");
}

// HF's YaRN frequencies (_compute_yarn_parameters) in double: the correction range, the ramp, the blend of the extrapolated and interpolated frequencies.
std::vector<double> hf_frequencies(double theta, size_t dim, double factor, double original, double& low, double& high) {
    const auto correction = [&](double r) { return double(dim) * std::log(original / (r * 2 * 3.14159265358979323846)) / (2 * std::log(theta)); };
    low = std::max(std::floor(correction(32)), 0.0);
    high = std::min(std::ceil(correction(1)), double(dim - 1));
    if (low == high) high += 0.001;
    std::vector<double> f(dim / 2);
    for (size_t i = 0; i < dim / 2; ++i) {
        const double ramp = std::min(std::max((double(i) - low) / (high - low), 0.0), 1.0);
        const double extra = std::pow(theta, -2.0 * double(i) / double(dim)), inter = extra / factor;
        f[i] = inter * ramp + extra * (1 - ramp);
    }
    return f;
}

// YaRN's frequencies, attention factor and tables against the double reference written from HF's formulas (docs/TESTS.md).
void yarn_reference() {
    using infer::rope::Base;
    const gguf::GGUFModel none;
    struct Case {
        Base form;
        float theta;
        size_t dim, original;
        float factor;
    };
    for (const Case& c : {Case{Base::negative_power, 1e6f, 128, 32768, 4}, Case{Base::negative_power, 1e7f, 64, 262144, 4}, Case{Base::reciprocal_power, 1e7f, 64, 32768, 8},
                          Case{Base::negative_power, 10000.0f, 16, 16, 4}, Case{Base::negative_power, 1e6f, 128, 40960, 2.5f}}) {
        const infer::rope::Policy pol = infer::rope::resolve(none, "x.", {c.form, c.theta, c.dim, c.original}, {c.factor, std::nullopt});
        require(pol.yarn && pol.forced && pol.original == c.original && pol.context == size_t(std::floor(double(c.factor) * double(c.original))), "forced policy fields");
        require(pol.attention == float(0.1 * std::log(double(c.factor)) + 1.0), "attention factor 0.1 ln(s) + 1");
        double low = 0, high = 0;
        const std::vector<double> ref = hf_frequencies(c.theta, c.dim, c.factor, double(c.original), low, high);
        const std::vector<double> got = infer::rope::frequencies(pol);
        bool below = false, past = false;
        for (size_t i = 0; i < c.dim / 2; ++i) {
            const double unit = std::ldexp(1.0, std::ilogb(ref[i]) - 23);
            require(std::fabs(double(got[i]) - ref[i]) <= 4 * unit, "YaRN frequency of pair " + std::to_string(i) + " off the reference");
            // HF's spelling whatever the module's: 1 / theta^(2i/d) below the ramp, 1 / (factor theta^(2i/d)) past it.
            const float power = std::pow(c.theta, (float)(2 * i) / (float)c.dim);
            if (double(i) <= low) {
                require(got[i] == double(1.0f / power), "a pair below the ramp keeps HF's extrapolated frequency");
                below = true;
            }
            if (double(i) >= high) {
                require(got[i] == double(1.0f / (c.factor * power)), "a pair past the ramp takes HF's interpolated frequency");
                past = true;
            }
        }
        // Each case but the narrow one reaches both ends of the ramp, so both blends are exercised.
        if (c.dim > 16) require(below && past && low > 0 && high < double(c.dim / 2), "the ramp lies inside the pairs");
        const size_t context = 4096, half = c.dim / 2;
        infer::rope::Policy shorter = pol;
        shorter.context = context;
        std::vector<float> cs(context * half), sn(context * half);
        infer::rope::fill(shorter, cs, sn);
        for (size_t pos = 0; pos < context; pos += 37)
            for (size_t i = 0; i < half; ++i) {
                const float ec = std::cos((float)pos * (float)got[i]) * pol.attention, es = std::sin((float)pos * (float)got[i]) * pol.attention;
                require(!std::memcmp(&ec, &cs[pos * half + i], sizeof(float)) && !std::memcmp(&es, &sn[pos * half + i], sizeof(float)), "a YaRN table entry");
            }
    }
}

// The file's keys and the request: declared YaRN taken as written, a request equal to the declaration not forced, and identities that follow the effective rope.
void policies() {
    using infer::rope::Base;
    const infer::rope::Inputs q3{Base::negative_power, 1e6f, 128, 131072};
    const gguf::GGUFModel none, file = declaring(4, 32768);
    const auto unscaled = infer::rope::resolve(none, "x.", q3, {});
    require(!unscaled.yarn && unscaled.context == 131072 && unscaled.identity().empty() && unscaled.describe().empty(), "the unscaled rope records nothing");
    const auto declared = infer::rope::resolve(file, "x.", q3, {});
    require(declared.yarn && !declared.forced && declared.factor == 4 && declared.original == 32768 && declared.context == 131072, "declared YaRN as written");
    require(declared.describe().find("model declares yarn factor 4 over 32768") != std::string::npos, "the record names the declaration");
    const auto same_request = infer::rope::resolve(file, "x.", q3, {4.0f, std::nullopt});
    require(!same_request.forced && same_request.identity() == declared.identity(), "a request equal to the declaration is not forced");
    const auto other = infer::rope::resolve(file, "x.", q3, {2.0f, std::nullopt});
    require(other.forced && other.context == 65536 && other.identity() != declared.identity(), "a request replaces the declared factor");
    const auto forced = infer::rope::resolve(none, "x.", {Base::negative_power, 1e6f, 128, 40960}, {4.0f, size_t(32768)});
    require(forced.forced && forced.context == 131072 && forced.identity() == declared.identity(), "forced and declared equal policies share one identity");
    require(forced.describe().find("warning: YaRN forced") != std::string::npos, "forcing an undeclared model warns");
    const auto own_context = infer::rope::resolve(none, "x.", {Base::negative_power, 1e6f, 128, 40960}, {2.0f, std::nullopt});
    require(own_context.original == 40960 && own_context.context == 81920, "the original defaults to the file's context");
    gguf::GGUFModel betas = declaring(4, 32768);
    betas.kv.push_back({"x.rope.scaling.yarn_beta_fast", real(16)});
    betas.kv.push_back({"x.rope.scaling.yarn_beta_slow", real(2)});
    const auto b = infer::rope::resolve(betas, "x.", q3, {});
    require(b.beta_fast == 16 && b.beta_slow == 2 && b.identity() != declared.identity(), "the betas are read and keyed");
}

// deepseek41's tables, the window's unscaled and the compressed layers' YaRN, against the loop the module wrote before, bit for bit; and --yarn on them, attention still none.
void deepseek() {
    using infer::rope::Arithmetic;
    using infer::rope::Attention;
    using infer::rope::Base;
    struct Case {
        float theta, compressed;
        int dim, original, context;
        float factor;
    };
    // The released file's rope (64 wide, 10000 and 160000, 16 over 65536) and a tiny fixture's.
    for (const Case& c : {Case{10000.0f, 160000.0f, 64, 65536, 1048576, 16}, Case{10000.0f, 40000.0f, 16, 64, 256, 4}}) {
        const size_t positions = 8192, half = size_t(c.dim / 2), n = positions * half;
        gguf::GGUFModel m = declaring(c.factor, uint64_t(c.original));
        m.kv.push_back({"x.rope.scaling.yarn_beta_fast", real(32)});
        m.kv.push_back({"x.rope.scaling.yarn_beta_slow", real(1)});
        const infer::rope::Inputs in{Base::reciprocal_power, c.compressed, size_t(c.dim), size_t(c.context), Arithmetic::f64_once, Attention::none};
        const infer::rope::Policy pol = infer::rope::resolve(m, "x.", in, {});
        require(pol.yarn && !pol.forced && pol.attention == 1.0f && pol.describe().find("attention none") != std::string::npos, "deepseek41's declared YaRN, attention none");
        for (bool yarn : {false, true}) {
            const std::vector<double> f = yarn ? deepseek_frequencies(c.dim, c.compressed, c.original, c.factor, 32, 1) : deepseek_frequencies(c.dim, c.theta, 0, 1, 0, 0);
            std::vector<float> oc(n), os(n), nc(n), ns(n);
            for (size_t pos = 0; pos < positions; ++pos)
                for (size_t i = 0; i < half; ++i) {
                    oc[pos * half + i] = float(std::cos(double(pos) * f[i]));
                    os[pos * half + i] = float(std::sin(double(pos) * f[i]));
                }
            infer::rope::Inputs window = in;
            window.theta = c.theta;
            infer::rope::fill(yarn ? pol : infer::rope::unscaled(window), nc, ns);
            identical(std::string("deepseek41 ") + (yarn ? "compressed" : "window") + " dim " + std::to_string(c.dim), oc, os, nc, ns, c.dim);
        }
        const infer::rope::Policy forced = infer::rope::resolve(m, "x.", in, {2 * c.factor, std::nullopt});
        require(forced.forced && forced.original == size_t(c.original) && forced.context == size_t(2 * c.factor * float(c.original)) && forced.attention == 1.0f,
                "--yarn on deepseek41 replaces the factor, attention still none");
    }
}

void refusals() {
    using infer::rope::Base;
    const infer::rope::Inputs q3{Base::negative_power, 1e6f, 128, 131072};
    const gguf::GGUFModel none;
    gguf::GGUFModel m;
    m.kv = {{"x.rope.scaling.type", text("linear")}};
    refuses("linear", "unsupported metadata x.rope.scaling.type", m, q3, {});
    m.kv = {{"x.rope.scaling.type", text("none")}, {"x.rope.scaling.factor", real(2)}};
    refuses("factor without yarn", "scaled RoPE is unsupported", m, q3, {});
    m.kv = {{"x.rope.scale_linear", real(2)}};
    refuses("linear scale", "scaled RoPE is unsupported", m, q3, {});
    refuses("yarn factor 1", "x.rope.scaling.factor must be above 1 for YaRN", declaring(1, 32768), q3, {});
    m = declaring(4, 32768);
    m.kv.pop_back();
    refuses("yarn without original", "missing metadata x.rope.scaling.original_context_length", m, q3, {});
    refuses("original 0", "integer outside supported range x.rope.scaling.original_context_length", declaring(4, 0), q3, {});
    require(infer::rope::resolve(declaring(4, 262144), "x.", q3, {}).context == 131072, "a declared original past the context keeps the file's context");
    for (const char* key : {"rope.scaling.attn_factor", "rope.scaling.yarn_ext_factor", "rope.scaling.yarn_attn_factor", "rope.scaling.yarn_log_multiplier"}) {
        m = declaring(4, 32768);
        m.kv.push_back({std::string("x.") + key, real(1)});
        refuses(key, std::string("unsupported metadata x.") + key, m, q3, {});
    }
    m = declaring(4, 32768);
    m.kv.push_back({"x.rope.scaling.yarn_beta_fast", real(1)});
    refuses("beta order", "yarn_beta_fast must be above yarn_beta_slow", m, q3, {});
    refuses("factor 1", "--yarn must be a finite factor above 1", none, q3, {1.0f, std::nullopt});
    refuses("factor NaN", "--yarn must be a finite factor above 1", none, q3, {std::numeric_limits<float>::quiet_NaN(), std::nullopt});
    refuses("request original past the context", "is past the model's context of 131072", none, q3, {4.0f, size_t(262144)});
    // The deepseek41 golden's shape: a declared original of 65536 over a context of 256 takes --yarn alone, and only a given --yarn-original-ctx is held to the context.
    const infer::rope::Inputs tiny{Base::negative_power, 1e6f, 16, 256};
    const auto past = infer::rope::resolve(declaring(16, 65536), "x.", tiny, {4.0f, std::nullopt});
    require(past.yarn && past.forced && past.original == 65536 && past.context == 262144, "--yarn alone over a declared original past the context");
    refuses("given original past the context", "--yarn-original-ctx 65536 is past the model's context of 256", declaring(16, 65536), tiny, {4.0f, size_t(65536)});
    refuses("context past the limit", "is past the context limit", none, q3, {1e30f, std::nullopt});
    refuses("odd width", "invalid rope base, width or context", none, {Base::negative_power, 1e6f, 127, 4096}, {});
}

}  // namespace

int main() {
    using infer::rope::Base;
    try {
        // The pinned files' bases and widths (Qwen3 1e6 x 128, qwen35 and qwen4exp 1e7 x 64) and the tiny fixtures' 10000 over narrow widths.
        for (float theta : {10000.0f, 1000000.0f, 10000000.0f})
            for (int dim : {8, 16, 32, 64, 128, 256}) {
                same("qwen3", qwen3_tables, Base::negative_power, theta, dim, 4096);
                same("qwen35", qwen35_tables, Base::negative_power, theta, dim, 4096);
                same("qwen4exp", qwen4exp_tables, Base::reciprocal_power, theta, dim, 4096);
            }
        same("qwen3", qwen3_tables, Base::negative_power, 1000000.0f, 128, 40960);
        same("qwen35", qwen35_tables, Base::negative_power, 10000000.0f, 64, 65536);
        same("qwen4exp", qwen4exp_tables, Base::reciprocal_power, 10000000.0f, 64, 65536);
        forms_differ();
        yarn_reference();
        policies();
        deepseek();
        refusals();
        std::cout << "rope: " << checks << " checks pass\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
