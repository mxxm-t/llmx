#pragma once
// What the GPU backends' shared core (backend_core.hpp) shows a vendor's public header, so its tests can name it.

namespace backend {

// The forms of a tensor group's fast sum (docs/src/backends-gpu.md, the collective), each adding every element in member order, so each gives the one-shot's bits.
// shares gives each member a share of the units to add and send back to every peer, copies does so through host memory by copy engines, and pulls is the copies behind events.
enum class GpuSumForm { one_shot, chain, pulls, shares, copies, wrap };

// A form's name as stderr and the tests print it.
inline const char* sum_form_name(GpuSumForm f) {
    static const char* const names[] = {"one-shot", "chain", "pulls", "shares", "copies", "wrap"};
    return names[(int)f];
}

// How a tensor group's members reach each other's rows (docs/src/backends-gpu.md, the collective): stores into each other's device memory, stores into pinned host memory every member maps, or copies the host orders behind events.
// The first two run the same kernels and forms, and a group keeps each whose probe passes; pulls is the conservative path where neither does.
enum class GroupTransport { peer_writes, host_writes, pulls };

// The transport, form and block cap a fast sum of some size takes, as a group's timing picks them; a cap of 0 takes the form's own (collective_core.hpp, launched_blocks), and for copies it counts chunks.
struct GpuSumPick {
    GroupTransport transport;
    GpuSumForm form;
    unsigned blocks = 0;
};

} // namespace backend
