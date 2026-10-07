#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "quant/types.hpp"
#include "backends/backend.hpp"
#include "model/weights.hpp"
#include "model/architecture.hpp"

// How a model splits over a tensor group (docs/TENSOR-SPLIT.md, section 4.2): each member's spans of a role's rows or columns from the role's declaration (Role::shard), the checks a width must pass, the bytes a member holds of a tensor and the heads it keeps; the one owner of these, whatever the backend.

namespace infer::shard {

// A run of a role's axis, in rows or columns.
struct Span {
    uint64_t begin = 0, count = 0;
};

// A run of a tensor's bytes a member holds: `bytes` from offset `from` in the tensor, at offset `to` in the member's packed copy.
struct Run {
    size_t from = 0, bytes = 0, to = 0;
};

// The length of a role's axis: its rows, or its columns, a vector's elements.
inline uint64_t extent(const Role& role) { return role.shard.axis == Axis::columns ? role.in : role.out; }

// The units of `section` member `member` of `width` takes, the first and the count, or nothing where the width neither divides the units nor, where the section allows it, is a multiple of them.
inline std::optional<Span> take(const ShardSection& section, size_t width, size_t member) {
    if (section.units % width == 0) {
        const uint64_t count = section.units / width;
        return Span{member * count, count};
    }
    if (section.replicate && section.units && width % section.units == 0) return Span{member / (width / section.units), 1};
    return std::nullopt;
}

// The refusal of a width that does not split a section of `role`.
inline std::runtime_error indivisible(const Role& role, const ShardSection& section, size_t width) {
    return std::runtime_error("inference: a tensor width of " + std::to_string(width) + " does not divide the " + std::to_string(section.units * section.tiles) + " " +
                              section.what + " of " + role.name + (section.replicate ? ", nor is a multiple of them" : ""));
}

// Member `member`'s spans of `role` at `width`, in order along the axis with adjacent spans joined: the whole axis at width 1 or for a role whose axis is none.
// A declaration whose sections do not cover the axis is the architecture's error; a width that does not split a section is refused by name.
inline std::vector<Span> spans(const Role& role, size_t width, size_t member) {
    if (!width || member >= width) throw std::logic_error("shard: member " + std::to_string(member) + " of a group of " + std::to_string(width));
    if (role.shard.axis == Axis::none || width == 1) return {Span{0, extent(role)}};
    std::vector<Span> out;
    uint64_t at = 0;
    for (const ShardSection& s : role.shard.sections) {
        const std::optional<Span> units = take(s, width, member);
        if (!units) throw indivisible(role, s, width);
        for (uint64_t t = 0; t < s.tiles; ++t) {
            const Span span{at + (t * s.units + units->begin) * s.unit, units->count * s.unit};
            if (!out.empty() && out.back().begin + out.back().count == span.begin) out.back().count += span.count;
            else out.push_back(span);
        }
        at += s.tiles * s.units * s.unit;
    }
    if (at != extent(role)) throw std::logic_error("shard: the sections of " + role.name + " cover " + std::to_string(at) + " of its " + std::to_string(extent(role)));
    return out;
}

// Whether a model of `plan` over `views` splits at `width`: each role's tensor by its declaration, every section by its units and on the column axis every span of every member on whole blocks of its storage type, refused by name otherwise, as is a state layer's saved row the width does not divide, an embedded drafter's block as a layer's; a routed layer is refused, as a group does not split one yet (docs/TENSOR-SPLIT.md, sections 4.10 and 6).
inline void check_plan(const ModelPlan& plan, const std::vector<TensorView>& views, size_t width) {
    if (!width) throw std::logic_error("shard: a group of no members");
    if (width == 1) return;
    auto each = [&](const std::vector<Role>& roles) {
        for (const Role& role : roles) {
            if (!role.tensor || role.shard.axis == Axis::none) continue;
            const quant::StorageType* type = quant::storage_type(views[*role.tensor].type);
            if (!type) throw std::runtime_error("quant: unsupported tensor type " + std::to_string(views[*role.tensor].type));
            for (size_t m = 0; m < width; ++m)
                for (const Span& s : spans(role, width, m))
                    if (role.shard.axis == Axis::columns && (s.begin % type->block_size || s.count % type->block_size))
                        throw std::runtime_error("inference: a tensor width of " + std::to_string(width) + " splits " + role.name + " off whole " + type->name + " blocks");
        }
    };
    each(plan.pass);
    for (size_t l = 0; l < plan.layers.size(); ++l) {
        if (plan.layers[l].routed)
            throw std::runtime_error("inference: a tensor width of " + std::to_string(width) + " does not split layer " + std::to_string(l) + "'s routed experts yet");
        each(plan.layers[l].roles);
        // A state layer's saved rows are rows its split roles write, so a member holds its share of each (saved).
        for (const Saved& v : plan.layers[l].saved)
            if (v.width % width)
                throw std::runtime_error("inference: a tensor width of " + std::to_string(width) + " does not divide the " + std::to_string(v.width) +
                                         " floats layer " + std::to_string(l) + " saves a row for a mark");
    }
    if (plan.drafter) each(plan.drafter->roles);
}

// The bytes of `role`'s tensor `t` member `member` of `width` holds, as runs of the tensor's bytes in the order they are packed: whole rows of each span on the row axis, and each row's spans on the column axis.
// At width 1 or for a role whose axis is none, one run of the whole tensor.
inline std::vector<Run> runs(const Role& role, const TensorView& t, size_t width, size_t member) {
    const std::vector<Span> parts = spans(role, width, member);
    if (role.shard.axis == Axis::none || width == 1) return {Run{0, t.bytes, 0}};
    const size_t nin = t.shape.empty() ? 0 : size_t(t.shape[0]);
    const size_t row = quant::row_bytes(t.type, nin);
    std::vector<Run> out;
    auto add = [&](size_t from, size_t bytes) {
        if (!out.empty() && out.back().from + out.back().bytes == from) {
            out.back().bytes += bytes;
            return;
        }
        out.push_back(Run{from, bytes, out.empty() ? 0 : out.back().to + out.back().bytes});
    };
    if (role.shard.axis == Axis::rows) {
        for (const Span& s : parts) add(backend::size_mul(size_t(s.begin), row), backend::size_mul(size_t(s.count), row));
        return out;
    }
    const size_t rows = row ? t.bytes / row : 0;
    for (size_t r = 0; r < rows; ++r)
        for (const Span& s : parts) add(r * row + quant::row_bytes(t.type, size_t(s.begin)), quant::row_bytes(t.type, size_t(s.count)));
    return out;
}

// The bytes the runs pack.
inline size_t bytes(const std::vector<Run>& runs) { return runs.empty() ? 0 : runs.back().to + runs.back().bytes; }

// A member's packed copy of a tensor whose bytes are at `src`, into `dst`, which holds bytes(runs).
inline void pack(const std::vector<Run>& runs, const uint8_t* src, uint8_t* dst) {
    for (const Run& r : runs) std::memcpy(dst + r.to, src + r.from, r.bytes);
}

// The shape of a member's copy of `role`'s tensor `t`, the fastest dimension first.
inline std::vector<uint64_t> shape(const Role& role, const TensorView& t, size_t width, size_t member) {
    std::vector<uint64_t> out = t.shape;
    if (role.shard.axis == Axis::none || width == 1) return out;
    uint64_t n = 0;
    for (const Span& s : spans(role, width, member)) n += s.count;
    out.at(role.shard.axis == Axis::columns ? 0 : 1) = n;
    return out;
}

// The KV heads a member of `width` keeps: its share, or one where KV heads are replicated over members.
inline size_t kv_heads(const ModelPlan& plan, size_t width) { return kv_share(plan.kv_heads, width); }

// The recurrent state a member of `width` keeps: its share of the K and V heads, each head as wide as on one device.
inline backend::StateShape state(const ModelPlan& plan, size_t width) {
    backend::StateShape s = plan.state;
    s.k_heads /= width;
    s.v_heads /= width;
    return s;
}

// The rows a state layer saves for a mark (LayerPlan::saved) as a member of `width` holds them: its share of each, since the split roles that write those rows give a member that share; check_plan refuses a width that does not divide one.
inline Saved saved(const Saved& v, size_t width) {
    if (v.width % width) throw std::logic_error("shard: a saved row of " + std::to_string(v.width) + " floats over a group of " + std::to_string(width));
    return Saved{v.slot, v.width / width, v.plane};
}

// The floats a row of a state layer saves for a mark on a member of `width`.
inline size_t saved_floats(const LayerPlan& layer, size_t width) {
    size_t n = 0;
    for (const Saved& v : layer.saved) n = backend::size_add(n, saved(v, width).width);
    return n;
}

} // namespace infer::shard
