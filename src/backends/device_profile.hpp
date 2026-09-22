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

    bool matrix_units = false;       // a matrix-multiply instruction class; no gfx906 has one
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
    // kernel, by whether every projection of the call is 8-bit or float.
    size_t tile_from_8bit = 32, tile_from_other = 64;
    // Splitting a short attention history across workgroups: below this many
    // (row, head) pairs, cut the history into chunks of this many tokens, at
    // most this many ways.
    size_t attention_split_below_pairs = 256, attention_split_chunk = 32, attention_split_max = 64;
    // Dispatches recorded before a submission, so the device starts on a pass
    // while the host is still recording it.
    uint32_t dispatch_chunk = 64;
};

// The profile for a device. One device family has been measured, so this
// returns those numbers; a device wanting others gets a branch here on what it
// reports rather than on who made it.
inline DeviceProfile profile_for(const DeviceCaps&) { return DeviceProfile{}; }

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
