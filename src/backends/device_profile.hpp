#pragma once
// What a device backend needs to know about its device in order to shape its
// kernels, kept apart from any one vendor's API.
//
// Two kinds of number decide how a GPU kernel is launched. The first kind the
// hardware reports: how wide a subgroup is, how many compute units there are,
// how much shared memory a workgroup may have. Every vendor API answers those,
// under its own names, so `DeviceCaps` holds them once and each backend fills
// it from whatever it has (Vulkan properties, HIP or CUDA device properties,
// Level Zero descriptors).
//
// The second kind is chosen by measurement: how many lanes should share a
// quantized block, from how many rows a tiled kernel beats a per-row one, how
// far a short attention history should be split. Those are not derivable from
// the first kind, they are found by running the kernels, and they are what
// `DeviceProfile` holds. They live here rather than inside a backend because
// the reasoning is the same whichever API launches the kernel, so a second
// backend fills the same struct rather than growing its own copy of the
// constants. The kernels themselves stay per-backend; only the shaping is
// shared.
//
// Those two kinds of number answer two different questions, and it is worth
// keeping them apart, because hardware differs in two ways.
//
// Some hardware wants a different kernel. A device with a matrix-multiply
// instruction class wants a matmul written around it, not the same one with
// other constants; a device without 16-bit arithmetic wants a path that does
// not use it. That is a capability question, answered by the flags in
// `DeviceCaps`, and its answer is which implementation to launch. A backend
// already selects among implementations for reasons that have nothing to do
// with the device, one per quantization family and one per cache element type;
// capability selection is the same mechanism with a different input.
//
// Other hardware wants the same kernel shaped differently: more rows per
// thread, fewer lanes to a block, a different row count before tiling starts.
// That is a configuration question, answered by `DeviceProfile` and by the
// shape of the call, and no new kernel is written for it.
//
// Keeping the two apart is what stops a backend from either writing a new
// kernel where a constant would do, or forcing one kernel to cover hardware it
// has no instructions for. Bringing up a device is therefore: fill
// `DeviceCaps` from its API, see whether any capability calls for an
// implementation that does not exist yet, run the backend's measurements, and
// give `profile_for` a branch if the numbers it wants differ. Branch on what a
// device reports, never on its vendor name.
#include <cstddef>
#include <cstdint>
#include <string>

namespace backend {

// What the device reports about itself.
//
// The sizes shape a kernel; the flags choose one. A flag is here only when a
// different implementation would be written for it, not merely because the API
// reports it.
struct DeviceCaps {
    uint32_t subgroup_size = 0;      // lanes that execute in lockstep
    uint32_t compute_units = 0;      // independent units, for deciding when work is too small to split further
    size_t shared_memory_bytes = 0;  // per workgroup, the limit the API reports

    // What the device and the software compiling its shaders call themselves.
    // Not a preference and not a brand: two drivers over the same silicon
    // generate different code and want different numbers below, so a profile
    // that has been measured is keyed by both.
    std::string device;
    std::string driver;

    bool matrix_units = false;       // a matrix-multiply instruction class; no gfx906 has one
    bool integer_dot = false;        // an integer dot-product instruction class
    bool fp16_arithmetic = false;    // half-precision arithmetic, not merely half-precision storage
    bool int8_arithmetic = false;    // 8-bit integer arithmetic
    bool storage_8bit = false;       // 8-bit values addressable in a buffer
    bool storage_16bit = false;      // 16-bit values addressable in a buffer
};

// Numbers found by measuring the kernels on a device, which nothing in
// DeviceCaps implies. The defaults are gfx906's, measured as a Radeon VII
// under the AMD proprietary driver and an MI50 under Mesa, which wanted the
// same values (docs/VULKAN.md).
struct DeviceProfile {
    // Lanes sharing one Q8_0 block pair in the per-row matmul. Four was the
    // optimum of 1, 2, 4, 8 and 16 at the 8B shapes: 200, 190, 336, 295, 185 GB/s.
    uint32_t q8_lanes_per_pair = 4;
    // Lanes sharing one K-quant block there. Eight follows the block layout
    // more than the card, a 256-value block being eight groups of 32, and
    // sixteen measured worse.
    uint32_t kquant_lanes = 8;
    // Query rows from which attention takes its tiled kernel.
    size_t attention_tile_rows = 32;
    // Batch rows from which a matmul takes the tile kernel rather than the row
    // kernel. The row kernel costs a weight pass per eight columns whatever a
    // row carries, while the tile kernel costs a whole tile however little of
    // it is filled, so the crossover moves with the width of a row as well as
    // with the number of rows. Measured by forcing each kernel and sweeping:
    // on a 1024-wide 8-bit projection it sits near 40 rows under the AMD
    // proprietary driver and near 96 under Mesa, and on a 4096-wide one near
    // 26 and 30. A narrow projection therefore wants a higher threshold on
    // both, and 64 is the value that wins most across the band on both, giving
    // up a little between 48 and 64 rows on the wider driver to avoid losing
    // a quarter to a half below 48 on either.
    size_t tile_from_8bit = 32, tile_from_8bit_narrow = 64, tile_from_other = 64;
    // The same split for the other quantized types, which the integer-dot tile moved: on the MI50 a 1024-wide K-quant projection crosses near 40 rows and a 4096-wide one near 20.
    size_t tile_from_other_narrow = 64;
    size_t tile_narrow_nin = 4096;
    // Splitting a short attention history across workgroups: below this many
    // (row, head) pairs, cut the history into chunks of this many tokens, at
    // most this many ways.
    size_t attention_split_below_pairs = 256, attention_split_chunk = 32, attention_split_max = 64;
    // Dispatches recorded before a submission, so the device starts on a pass
    // while the host is still recording it.
    uint32_t dispatch_chunk = 64;
    // Whether the per-row matmul should take its dots through the integer dot
    // product instructions rather than plain multiplies, where the device has
    // them at all. Having them does not settle it: the same gfx906 silicon
    // gains 15 percent of 8B decode under Mesa, which lowers them to the
    // chip's native 16-bit dot, and loses 2 percent under the AMD proprietary
    // driver, which lowers them to the multiplies anyway with the operands
    // widened first. So it is measured per device, not asked of the hardware.
    bool prefer_integer_dot = false;
};

// A profile that has been measured on one device with one driver.
//
// This is the table to extend when bringing up hardware. The defaults above
// are a compromise across what has been measured; an entry here is what a
// combination actually wanted, so a device in the table runs better than the
// defaults and a device that is not runs no worse than before it was tried.
// Measure with the sweeps in docs/VULKAN.md, add a row, and say in the note
// what was measured rather than what was assumed.
struct MeasuredProfile {
    const char* device;              // a substring of what the device calls itself
    const char* driver;              // a substring of what its driver calls itself
    // Where the tile matmul overtakes the per-row one: 8-bit projections at least 4096 wide and narrower, then the other types the same two ways.
    size_t tile_from_8bit, tile_from_8bit_narrow, tile_from_other, tile_from_other_narrow;
    bool prefer_integer_dot;         // whether the matmuls want the integer dot instructions
};

// The crossover between the per-row and the tile matmul is the number that has been found to move with the driver rather than the hardware, and with the tile kernel the driver runs.
// The same Vega20 wants 40 rows on a narrow 8-bit projection under the AMD proprietary driver, with the float tile.
// Under Mesa it wanted 96 against the float tile, and against the integer-dot tile the crossings fell to 24 to 32 rows on Qwen3-0.6B-Q8_0, 8 to 16 on 8B-Q8_0, 32 to 48 on 0.6B-Q5_K_M and 16 to 24 on 8B-Q4_K_M; at 64 rows on the 0.6B the tile reads 2160 tok/s where the row kernel reads 1098 (docs/VULKAN.md).
// A row measured with the integer-dot tile applies only when the device has the integer dot, since without it the float tile would run with thresholds measured for the other.
inline const MeasuredProfile* measured_profiles(size_t& count) {
    static const MeasuredProfile table[] = {
        {"Radeon VII", "AMD proprietary", 32, 48, 64, 64, false},
        {"MI60 / MI50", "radv", 16, 32, 24, 40, true},
    };
    count = sizeof(table) / sizeof(table[0]);
    return table;
}

// The profile for a device: the measured entry when its device and driver both
// match one, and the compromise defaults otherwise.
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

// Rows of a matmul tile, given the shape of the call. A taller tile reads less
// shared memory per product but yields fewer workgroups, so it is taken only
// while the device still has one per compute unit. Both heights must exist as
// launchable kernels; how a backend produces them is its own business, a
// specialization constant under Vulkan and a template parameter elsewhere.
inline uint32_t tile_rows_for(const DeviceCaps& caps, uint32_t rows_short, uint32_t rows_tall,
                              size_t out_rows, size_t column_groups) {
    const size_t tall_groups = ((out_rows + rows_tall - 1) / rows_tall) * column_groups;
    return tall_groups >= caps.compute_units ? rows_tall : rows_short;
}

} // namespace backend
