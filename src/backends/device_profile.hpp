#pragma once
// What a device backend needs to know about its device to shape its kernels, apart from any one vendor's API: `DeviceCaps` holds what the hardware reports, `DeviceProfile` numbers found by measuring the kernels.
// A capability decides which kernel to launch and a profile number shapes the same kernel; branch on what a device reports, never on its vendor name.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "quant/types.hpp"

namespace backend {

// The PCI root complex above the device at this address, as Linux's sysfs names it (pci0000:80), or empty where the system does not say, which a tensor group's startup report gives (docs/TENSOR-SPLIT.md, section 4.5).
inline std::string pci_root_of(uint32_t domain, uint32_t bus, uint32_t device, uint32_t function) {
#if defined(__linux__)
    char address[64];
    std::snprintf(address, sizeof address, "/sys/bus/pci/devices/%04x:%02x:%02x.%x", domain, bus, device, function);
    char resolved[4096];
    if (!realpath(address, resolved)) return {};
    const std::string path(resolved);
    const size_t at = path.find("/pci");
    if (at == std::string::npos) return {};
    return path.substr(at + 1, path.find('/', at + 1) - at - 1);
#else
    (void)domain, (void)bus, (void)device, (void)function;
    return {};
#endif
}

// What the device reports about itself: the sizes shape a kernel and the flags choose one.
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
    // Lanes a Q4_K or Q5_K row takes at most in the integer-dot row kernels; a short row spread over a whole subgroup leaves each lane a few bytes to read.
    uint32_t k45_row_lanes = 32;
    // Columns the widest build of the Q8_0 decode kernel on the integer dot holds, which reads a weight once for that many generated tokens; a subgroup that keeps more columns holds more registers and runs fewer waves (docs/VULKAN.md).
    uint32_t q8_decode_cols = 8;
    // Columns the widest two-row decode build of the Q4 and K-quant row kernels holds (2 to 16), or 0 for the eight-column build; set it with row_decode_tree (docs/VULKAN.md).
    // Set only where those builds' columns were checked bit for bit against the one-column build, since a driver may sum their products in another order.
    uint32_t row_decode_cols = 0;
    // Whether those two-row builds reduce a whole cluster through its own pairs transposed, each level halving the values a lane holds, which gives every column the butterfly's bits in fewer shuffles; set only where they were checked bit for bit.
    bool row_decode_tree = false;
    // Forms of the Q8_0 decode kernel a build may take (the Vulkan backend's kQ8Tree and kQ8Half): the first is set only where the disassembly showed the reduction takes the pairs it writes out (docs/VULKAN.md).
    // The second is the half-block order, which every build takes where it is set; it sums a column's products in another order than the quarter layout, so setting it changes the device's decode results once.
    uint32_t q8_decode_forms = 0;
    // Row width below which the Q8_0 decode kernel's one-column chunk takes its build of two rows and one step, since a lane of a narrow row has too few blocks to load ahead.
    uint32_t q8_decode_narrow_below = 1024;
    // Output rows of a call from which a wider one-column chunk over 8-bit activations takes the build of two rows and two steps ahead rather than one row and four steps, which gives a call of few rows twice the subgroups.
    uint32_t q8_decode_two_rows_from = 2560;
    // Output rows below which such a chunk over 16-bit activations takes the build of one row and four steps, and from which it keeps two rows and one step.
    uint32_t q8_decode_one_row_below_16bit = 1024;
    // Prompt extent from which attention takes its tiled kernel; the query rows a tile holds are the kernel's own.
    size_t attention_tile_rows = 32;
    // Batch rows from which a matmul takes the tile kernel rather than the row kernel, for 8-bit and other types, narrow and wide rows; the crossover moves with the row width, and the values are measured (docs/VULKAN.md).
    size_t tile_from_8bit = 32, tile_from_8bit_narrow = 64, tile_from_other = 64;
    size_t tile_from_other_narrow = 64;
    size_t tile_narrow_nin = 4096;
    // Prompt extent from which a routed projection takes the tile kernel over each expert's rows rather than the row kernel per entry, by weight family (the Vulkan weight descriptor): F32, Q8_0 and Q6_K, then Q4_0 and Q4_1, Q4_K and Q5_K.
    size_t moe_tile_from = 32, moe_tile_from_q4 = 96, moe_tile_from_q4k = 64, moe_tile_from_q5k = 48;
    // Splitting a row's attention history across workgroups: parts of this many tokens, the part doubling until at most this many cover the row, for heads 128 wide and narrower and for 256 wide (docs/STATUS.md), each at most 256.
    size_t attention_split_chunk = 32, attention_split_max = 64, attention_split_max_wide = 64;
    // Workgroups per compute unit below which the integer-dot tile splits a call's inner dimension, for rows at least tile_narrow_nin wide and narrower, and the fewest quant blocks of 32 a part may sum.
    uint32_t tile_split_per_cu = 8, tile_split_per_cu_narrow = 4, tile_split_min_blocks = 8;
    // The float tile's target when it splits, which only a call of a quarter of a workgroup per compute unit or fewer does, such as a router's.
    uint32_t float_tile_split_per_cu = 4;
    // Workgroups per compute unit the tallest tile must yield before a call takes it, for rows at least tile_narrow_nin wide and narrower: fewer leave too few waves per SIMD to hide the loads.
    uint32_t tile_tall_per_cu = 1, tile_tall_per_cu_narrow = 1;
    // Complete groups from which the indexer's selection takes its wide form, where the two forms cross; both give the same ids (docs/VULKAN.md).
    size_t select_wide_from = 4096;
    // Dispatches recorded before a submission, so the device starts on a pass while the host is still recording it.
    uint32_t dispatch_chunk = 64;
    // Whether the matmuls take their dots through the integer dot product instructions: measured per device and driver, since the same silicon gains under Mesa and loses under the AMD proprietary driver.
    bool prefer_integer_dot = false;
    // Signed lookup decoding and dot instructions are tuned independently of other row families.
    bool lookup_integer_dot = false;
};

// A device and driver the profile was tuned on, with the numbers measured there: the table to extend when bringing up hardware, with the sweeps in docs/VULKAN.md.
struct TunedDevice {
    const char* device;              // a substring of what the device calls itself
    const char* driver;              // a substring of what its driver calls itself
    void (*tune)(DeviceProfile&);    // assigns the numbers measured there, each by name
};

// The tuned devices: the tile crossover moves with the driver and with the tile kernel it runs, and a row measured against the integer-dot tile applies only where the device has the integer dot (docs/VULKAN.md).
inline const TunedDevice* tuned_devices(size_t& count) {
    static const TunedDevice table[] = {
        {"Radeon VII", "AMD proprietary", [](DeviceProfile& p) {
             p.tile_from_8bit = 32;
             p.tile_from_8bit_narrow = 48;
             p.tile_from_other = 64;
             p.tile_from_other_narrow = 64;
             p.prefer_integer_dot = false;
             p.lookup_integer_dot = true;
             p.moe_tile_from = 32;
             p.moe_tile_from_q4 = 96;
             p.moe_tile_from_q4k = 64;
             p.moe_tile_from_q5k = 48;
             p.tile_tall_per_cu = 1;
             p.tile_tall_per_cu_narrow = 1;
             p.select_wide_from = 5120;
             p.q8_decode_narrow_below = 1024;
             p.q8_decode_two_rows_from = 2560;
             p.q8_decode_one_row_below_16bit = 1280;
         }},
        {"MI60 / MI50", "radv", [](DeviceProfile& p) {
             p.tile_from_8bit = 16;
             p.tile_from_8bit_narrow = 32;
             p.tile_from_other = 24;
             p.tile_from_other_narrow = 40;
             p.prefer_integer_dot = true;
             p.lookup_integer_dot = true;
             p.moe_tile_from = 32;
             p.moe_tile_from_q4 = 96;
             p.moe_tile_from_q4k = 64;
             p.moe_tile_from_q5k = 48;
             p.tile_tall_per_cu = 4;
             p.tile_tall_per_cu_narrow = 8;
             p.q8_decode_cols = 32;
             p.q8_decode_forms = 3;
             p.row_decode_cols = 16;
             p.row_decode_tree = true;
             p.attention_split_max = 32;
             p.q8_decode_narrow_below = 1024;
             p.q8_decode_two_rows_from = 4096;
             p.q8_decode_one_row_below_16bit = 1024;
         }},
    };
    count = sizeof(table) / sizeof(table[0]);
    return table;
}

// The profile for a device: the compromise defaults, tuned by the first row whose device and driver both match unless that row wants an integer dot the device lacks.
inline DeviceProfile profile_for(const DeviceCaps& caps) {
    const DeviceProfile defaults;
    size_t count = 0;
    const TunedDevice* table = tuned_devices(count);
    for (size_t i = 0; i < count; ++i) {
        if (caps.device.find(table[i].device) == std::string::npos) continue;
        if (caps.driver.find(table[i].driver) == std::string::npos) continue;
        DeviceProfile p = defaults;
        table[i].tune(p);
        p.lookup_integer_dot = p.lookup_integer_dot && caps.integer_dot;
        return p.prefer_integer_dot && !caps.integer_dot ? defaults : p;
    }
    return defaults;
}

// Batch rows from which a matmul of this shape should take the tile kernel.
inline size_t tile_from_for(const DeviceProfile& profile, bool every_projection_8bit_or_float, size_t nin) {
    if (!every_projection_8bit_or_float)
        return nin < profile.tile_narrow_nin ? profile.tile_from_other_narrow : profile.tile_from_other;
    return nin < profile.tile_narrow_nin ? profile.tile_from_8bit_narrow : profile.tile_from_8bit;
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
