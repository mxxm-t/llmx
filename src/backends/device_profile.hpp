#pragma once
// What a device backend needs to know about its device to shape its kernels, apart from any one vendor's API.
// `DeviceCaps` holds what the hardware reports (subgroup width, compute units, shared memory, instruction classes), filled by each backend from its own API.
// `DeviceProfile` holds numbers found by measuring the kernels (lanes per block, tile crossovers, splits), shared across backends because the reasoning is the same.
// A capability decides which kernel to launch; a profile number shapes the same kernel.
// Bringing up a device: fill `DeviceCaps`, check whether a capability calls for a kernel that does not exist yet, measure, and give `profile_for` a row if the numbers differ.
// Branch on what a device reports, never on its vendor name.
#include <cstddef>
#include <cstdint>
#include <string>

namespace backend {

// What the device reports about itself.
//
// The sizes shape a kernel; the flags choose one.
// A flag is here only when a different implementation would be written for it, not merely because the API reports it.
struct DeviceCaps {
    uint32_t subgroup_size = 0;      // lanes that execute in lockstep
    uint32_t compute_units = 0;      // independent units, for deciding when work is too small to split further
    size_t shared_memory_bytes = 0;  // per workgroup, the limit the API reports

    // What the device and the software compiling its shaders call themselves.
    // Not a preference and not a brand: two drivers over the same silicon generate different code and want different numbers below, so a profile that has been measured is keyed by both.
    std::string device;
    std::string driver;

    bool matrix_units = false;       // a matrix-multiply instruction class; no gfx906 has one
    bool integer_dot = false;        // an integer dot-product instruction class
    bool fp16_arithmetic = false;    // half-precision arithmetic, not merely half-precision storage
    bool int8_arithmetic = false;    // 8-bit integer arithmetic
    bool storage_8bit = false;       // 8-bit values addressable in a buffer
    bool storage_16bit = false;      // 16-bit values addressable in a buffer
};

// Numbers found by measuring the kernels on a device, which nothing in DeviceCaps implies.
// The defaults are gfx906's under the AMD proprietary driver; a device and driver measured to want other values has a row in measured_profiles below, as the MI50 under Mesa does (docs/VULKAN.md).
struct DeviceProfile {
    // Lanes sharing one Q8_0 block pair in the per-row matmul, and one K-quant block there (a 256-value block is eight groups of 32).
    uint32_t q8_lanes_per_pair = 4;
    uint32_t kquant_lanes = 8;
    // Lanes a Q6_K row takes at most on the 8-bit twin, so a subgroup takes several rows and one row's loads hide behind another's (docs/VULKAN.md).
    uint32_t q6k_row_lanes = 32;
    // Query rows from which attention takes its tiled kernel.
    size_t attention_tile_rows = 32;
    // Batch rows from which a matmul takes the tile kernel rather than the row kernel, for 8-bit and other types, narrow and wide rows; the crossover moves with the row width, and the values are measured (docs/VULKAN.md).
    size_t tile_from_8bit = 32, tile_from_8bit_narrow = 64, tile_from_other = 64;
    size_t tile_from_other_narrow = 64;
    size_t tile_narrow_nin = 4096;
    // Prompt extent from which a routed projection takes the tile kernel over each expert's rows rather than the row kernel per entry.
    size_t moe_tile_from = 32;
    // Splitting a row's attention history across workgroups: parts of this many tokens, the part doubling until at most this many cover the row.
    size_t attention_split_chunk = 32, attention_split_max = 64;
    // Workgroups per compute unit below which the integer-dot tile splits a call's inner dimension, and the fewest quant blocks of 32 a part may sum.
    uint32_t tile_split_per_cu = 8, tile_split_min_blocks = 16;
    // Dispatches recorded before a submission, so the device starts on a pass while the host is still recording it.
    uint32_t dispatch_chunk = 64;
    // Whether the matmuls take their dots through the integer dot product instructions: measured per device and driver, since the same silicon gains under Mesa and loses under the AMD proprietary driver.
    bool prefer_integer_dot = false;
};

// A profile measured on one device with one driver: the table to extend when bringing up hardware, with the sweeps in docs/VULKAN.md.
struct MeasuredProfile {
    const char* device;              // a substring of what the device calls itself
    const char* driver;              // a substring of what its driver calls itself
    // Where the tile matmul overtakes the per-row one: 8-bit projections at least 4096 wide and narrower, then the other types the same two ways.
    size_t tile_from_8bit, tile_from_8bit_narrow, tile_from_other, tile_from_other_narrow;
    bool prefer_integer_dot;         // whether the matmuls want the integer dot instructions
};

// The measured rows: the tile crossover moves with the driver and with the tile kernel it runs, and a row measured against the integer-dot tile applies only where the device has the integer dot (docs/VULKAN.md).
inline const MeasuredProfile* measured_profiles(size_t& count) {
    static const MeasuredProfile table[] = {
        {"Radeon VII", "AMD proprietary", 32, 48, 64, 64, false},
        {"MI60 / MI50", "radv", 16, 32, 24, 40, true},
    };
    count = sizeof(table) / sizeof(table[0]);
    return table;
}

// The profile for a device: the measured entry when its device and driver both match one, and the compromise defaults otherwise.
inline DeviceProfile profile_for(const DeviceCaps& caps) {
    DeviceProfile p;
    size_t count = 0;
    const MeasuredProfile* table = measured_profiles(count);
    for (size_t i = 0; i < count; ++i) {
        if (caps.device.find(table[i].device) == std::string::npos) continue;
        if (caps.driver.find(table[i].driver) == std::string::npos) continue;
        if (table[i].prefer_integer_dot && !caps.integer_dot) break;
        p.tile_from_8bit = table[i].tile_from_8bit;
        p.tile_from_8bit_narrow = table[i].tile_from_8bit_narrow;
        p.tile_from_other = table[i].tile_from_other;
        p.tile_from_other_narrow = table[i].tile_from_other_narrow;
        p.prefer_integer_dot = table[i].prefer_integer_dot;
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

// Rows of a matmul tile given the call's shape: the tallest height that still yields a workgroup per compute unit, then the middle one, then the smallest.
// A wide projection drops to the smallest only below half fill, since the small tile does half the arithmetic per barrier (docs/VULKAN.md).
inline uint32_t tile_rows_for(const DeviceCaps& caps, const DeviceProfile& profile, uint32_t rows_small, uint32_t rows_short,
                              uint32_t rows_tall, size_t out_rows, size_t column_groups, size_t nin) {
    auto groups = [&](uint32_t h) { return ((out_rows + h - 1) / h) * column_groups; };
    if (groups(rows_tall) >= caps.compute_units) return rows_tall;
    const size_t fill = nin < profile.tile_narrow_nin ? caps.compute_units : (caps.compute_units + 1) / 2;
    return groups(rows_short) >= fill ? rows_short : rows_small;
}

} // namespace backend
