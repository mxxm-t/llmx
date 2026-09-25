#pragma once
// What a device backend needs to know about its device to shape its kernels, apart from any one vendor's API.
// `DeviceCaps` holds what the hardware reports (subgroup width, compute units, names, the integer dot), filled by each backend from its own API.
// `DeviceProfile` holds numbers found by measuring the kernels (lanes per block, tile crossovers, splits), shared across backends because the reasoning is the same.
// A capability decides which kernel to launch; a profile number shapes the same kernel.
// Bringing up a device: fill `DeviceCaps`, check whether a capability calls for a kernel that does not exist yet, measure, and give `profile_for` a row if the numbers differ.
// Branch on what a device reports, never on its vendor name.
#include <cstddef>
#include <cstdint>
#include <string>

#include "format/gguf.hpp"

namespace backend {

// What the device reports about itself.
//
// The sizes shape a kernel; the flags choose one.
// A flag is here only when a different implementation would be written for it, not merely because the API reports it.
struct DeviceCaps {
    uint32_t subgroup_size = 0;      // lanes that execute in lockstep
    uint32_t compute_units = 0;      // independent units, for deciding when work is too small to split further

    // What the device and the software compiling its shaders call themselves.
    // Not a preference and not a brand: two drivers over the same silicon generate different code and want different numbers below, so a profile that has been measured is keyed by both.
    std::string device;
    std::string driver;

    bool integer_dot = false;        // an integer dot-product instruction class
};

// Numbers found by measuring the kernels on a device, which nothing in DeviceCaps implies.
// Every device takes these defaults except where its row in tuned_devices below sets a number; the defaults of those numbers are a compromise between the devices measured (docs/VULKAN.md).
struct DeviceProfile {
    // Lanes sharing one Q8_0 block pair in the per-row matmul, and one K-quant block there (a 256-value block is eight groups of 32).
    uint32_t q8_lanes_per_pair = 4;
    uint32_t kquant_lanes = 8;
    // Lanes a Q6_K row takes at most on the 8-bit twin, so a subgroup takes several rows and one row's loads hide behind another's (docs/VULKAN.md).
    uint32_t q6k_row_lanes = 32;
    // Lanes a Q4_K or Q5_K row takes at most in the integer-dot row kernels; a short row spread over a whole subgroup leaves each lane a few bytes to read.
    uint32_t k45_row_lanes = 32;
    // Query rows from which attention takes its tiled kernel.
    size_t attention_tile_rows = 32;
    // Batch rows from which a matmul takes the tile kernel rather than the row kernel, for 8-bit and other types, narrow and wide rows; the crossover moves with the row width, and the values are measured (docs/VULKAN.md).
    size_t tile_from_8bit = 32, tile_from_8bit_narrow = 64, tile_from_other = 64;
    size_t tile_from_other_narrow = 64;
    size_t tile_narrow_nin = 4096;
    // Prompt extent from which a routed projection takes the tile kernel over each expert's rows rather than the row kernel per entry, by weight family (moe_tile_from_for): F32, Q8_0 and Q6_K, then Q4_0 and Q4_1, Q4_K and Q5_K.
    size_t moe_tile_from = 32, moe_tile_from_q4 = 96, moe_tile_from_q4k = 64, moe_tile_from_q5k = 48;
    // Splitting a row's attention history across workgroups: parts of this many tokens, the part doubling until at most this many cover the row.
    size_t attention_split_chunk = 32, attention_split_max = 64;
    // Workgroups per compute unit below which the integer-dot tile splits a call's inner dimension, for rows at least tile_narrow_nin wide and narrower, and the fewest quant blocks of 32 a part may sum.
    uint32_t tile_split_per_cu = 8, tile_split_per_cu_narrow = 4, tile_split_min_blocks = 8;
    // The float tile's target when it splits, which only a call of a quarter of a workgroup per compute unit or fewer does, such as a router's.
    uint32_t float_tile_split_per_cu = 4;
    // Workgroups per compute unit the tallest tile must yield before a call takes it, for rows at least tile_narrow_nin wide and narrower: fewer leave too few waves per SIMD to hide the loads.
    uint32_t tile_tall_per_cu = 1, tile_tall_per_cu_narrow = 1;
    // Dispatches recorded before a submission, so the device starts on a pass while the host is still recording it.
    uint32_t dispatch_chunk = 64;
    // Whether the matmuls take their dots through the integer dot product instructions: measured per device and driver, since the same silicon gains under Mesa and loses under the AMD proprietary driver.
    bool prefer_integer_dot = false;
};

// A device and driver the profile was tuned on, with the numbers measured there: the table to extend when bringing up hardware, with the sweeps in docs/VULKAN.md.
struct TunedDevice {
    const char* device;              // a substring of what the device calls itself
    const char* driver;              // a substring of what its driver calls itself
    // Where the tile matmul overtakes the per-row one: 8-bit projections at least 4096 wide and narrower, then the other types the same two ways.
    size_t tile_from_8bit, tile_from_8bit_narrow, tile_from_other, tile_from_other_narrow;
    bool prefer_integer_dot;         // whether the matmuls want the integer dot instructions
    // Where a routed projection's tile overtakes its row kernel, by weight family as DeviceProfile orders them.
    size_t moe_tile_from, moe_tile_from_q4, moe_tile_from_q4k, moe_tile_from_q5k;
    uint32_t tile_tall_per_cu, tile_tall_per_cu_narrow;   // the tallest tile's fill, wide and narrow rows
};

// The tuned devices: the tile crossover moves with the driver and with the tile kernel it runs, and a row measured against the integer-dot tile applies only where the device has the integer dot (docs/VULKAN.md).
inline const TunedDevice* tuned_devices(size_t& count) {
    static const TunedDevice table[] = {
        {"Radeon VII", "AMD proprietary", 32, 48, 64, 64, false, 32, 96, 64, 48, 1, 1},
        {"MI60 / MI50", "radv", 16, 32, 24, 40, true, 32, 96, 64, 48, 4, 8},
    };
    count = sizeof(table) / sizeof(table[0]);
    return table;
}

// The profile for a device: the tuned entry when its device and driver both match one, and the compromise defaults otherwise.
inline DeviceProfile profile_for(const DeviceCaps& caps) {
    DeviceProfile p;
    size_t count = 0;
    const TunedDevice* table = tuned_devices(count);
    for (size_t i = 0; i < count; ++i) {
        if (caps.device.find(table[i].device) == std::string::npos) continue;
        if (caps.driver.find(table[i].driver) == std::string::npos) continue;
        if (table[i].prefer_integer_dot && !caps.integer_dot) break;
        p.tile_from_8bit = table[i].tile_from_8bit;
        p.tile_from_8bit_narrow = table[i].tile_from_8bit_narrow;
        p.tile_from_other = table[i].tile_from_other;
        p.tile_from_other_narrow = table[i].tile_from_other_narrow;
        p.prefer_integer_dot = table[i].prefer_integer_dot;
        p.moe_tile_from = table[i].moe_tile_from;
        p.moe_tile_from_q4 = table[i].moe_tile_from_q4;
        p.moe_tile_from_q4k = table[i].moe_tile_from_q4k;
        p.moe_tile_from_q5k = table[i].moe_tile_from_q5k;
        p.tile_tall_per_cu = table[i].tile_tall_per_cu;
        p.tile_tall_per_cu_narrow = table[i].tile_tall_per_cu_narrow;
        break;
    }
    return p;
}

// Batch rows from which a matmul of this shape should take the tile kernel.
inline size_t tile_from_for(const DeviceProfile& profile, bool every_projection_8bit_or_float, size_t nin) {
    if (!every_projection_8bit_or_float)
        return nin < profile.tile_narrow_nin ? profile.tile_from_other_narrow : profile.tile_from_other;
    return nin < profile.tile_narrow_nin ? profile.tile_from_8bit_narrow : profile.tile_from_8bit;
}

// Prompt extent from which a routed projection of this weight type takes the tile kernel: the 4-bit rows, cheap to unpack per entry, stay ahead of the tile's grouping longer.
inline size_t moe_tile_from_for(const DeviceProfile& profile, uint32_t type) {
    switch (type) {
    case gguf::GGML_TYPE_Q4_0: case gguf::GGML_TYPE_Q4_1: return profile.moe_tile_from_q4;
    case gguf::GGML_TYPE_Q4_K: return profile.moe_tile_from_q4k;
    case gguf::GGML_TYPE_Q5_K: return profile.moe_tile_from_q5k;
    default: return profile.moe_tile_from;
    }
}

// Rows of a matmul tile given the call's shape: the tallest height that still yields the profile's workgroups per compute unit, then the middle one, then the smallest.
// A wide projection drops to the smallest only below half fill, since the small tile does half the arithmetic per barrier (docs/VULKAN.md).
inline uint32_t tile_rows_for(const DeviceCaps& caps, const DeviceProfile& profile, uint32_t rows_small, uint32_t rows_short,
                              uint32_t rows_tall, size_t out_rows, size_t column_groups, size_t nin) {
    auto groups = [&](uint32_t h) { return ((out_rows + h - 1) / h) * column_groups; };
    const size_t tall_per_cu = nin < profile.tile_narrow_nin ? profile.tile_tall_per_cu_narrow : profile.tile_tall_per_cu;
    if (groups(rows_tall) >= tall_per_cu * caps.compute_units) return rows_tall;
    const size_t fill = nin < profile.tile_narrow_nin ? caps.compute_units : (caps.compute_units + 1) / 2;
    return groups(rows_short) >= fill ? rows_short : rows_small;
}

} // namespace backend
