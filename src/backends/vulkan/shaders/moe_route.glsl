// Expert routing for one row (backend.hpp route_experts) by 64 lanes, quarter q of the workgroup: one subgroup of 64 or two of 32, the whole workgroup calling together, a row past the call's with `live` false.
// The including shader declares route_scores, route_ids and route_weights and sets ROUTE_QUARTERS, one in moe_route.comp and four in the fused routed step's first kernel.
#ifndef LLMX_MOE_ROUTE_GLSL
#define LLMX_MOE_ROUTE_GLSL

// Invocation t holds experts t + 64 q, q < kHeld, in registers; the backend takes the build of the fewest slots that hold the row.
#if defined(LLMX_SLOTS1)
const uint kHeld = 1u;
#elif defined(LLMX_SLOTS2)
const uint kHeld = 2u;
#elif defined(LLMX_SLOTS4)
const uint kHeld = 4u;
#elif defined(LLMX_SLOTS8)
const uint kHeld = 8u;
#else
const uint kHeld = 16u;
#endif
shared float part_v[2u * ROUTE_QUARTERS];
shared uint part_i[2u * ROUTE_QUARTERS];
shared float chosen[ROUTE_QUARTERS][256];
shared uint chosen_id[ROUTE_QUARTERS][256];

// A quarter's reductions: its subgroup's, or its two subgroups' joined through shared memory.
float group_max(float v, uint q) {
    v = subgroupMax(v);
    if (gl_SubgroupSize >= 64u) return v;
    if (subgroupElect()) part_v[gl_SubgroupID] = v;
    barrier();
    float r = max(part_v[2u * q], part_v[2u * q + 1u]);
    barrier();
    return r;
}
float group_sum(float v, uint q) {
    v = subgroupAdd(v);
    if (gl_SubgroupSize >= 64u) return v;
    if (subgroupElect()) part_v[gl_SubgroupID] = v;
    barrier();
    float r = part_v[2u * q] + part_v[2u * q + 1u];
    barrier();
    return r;
}
uint group_min(uint v, uint q) {
    v = subgroupMin(v);
    if (gl_SubgroupSize >= 64u) return v;
    if (subgroupElect()) part_i[gl_SubgroupID] = v;
    barrier();
    uint r = min(part_i[2u * q], part_i[2u * q + 1u]);
    barrier();
    return r;
}

void route_row(uint r, uint t, uint q, bool live, uint n_expert, uint k, bool normalize) {
    uint base = r * n_expert;
    // Every score is loaded at once, a slot past the experts reading the last one, which it then leaves out.
    float pr[kHeld];
    for (uint s = 0u; s < kHeld; ++s) pr[s] = route_scores[base + min(t + 64u * s, n_expert - 1u)];
    float top = -1.0 / 0.0;
    for (uint s = 0u; s < kHeld; ++s)
        if (t + 64u * s < n_expert) top = max(top, pr[s]);
    top = group_max(top, q);
    float part = 0.0;
    for (uint s = 0u; s < kHeld; ++s)
        if (t + 64u * s < n_expert) {
            float v = exp(pr[s] - top);
            pr[s] = v;
            part += v;
        }
    float sum = group_sum(part, q);
    // A probability below FLT_MIN counts as 0 under any float mode, and a slot past the experts holds -2, which no pick takes.
    for (uint s = 0u; s < kHeld; ++s) {
        float v = pr[s] / sum;
        pr[s] = t + 64u * s >= n_expert ? -2.0 : v < uintBitsToFloat(0x00800000u) ? 0.0 : v;
    }
    // Every invocation adds the picks in j order, the sum the weights are divided by.
    float total = 0.0;
    for (uint j = 0u; j < k; ++j) {
        // The largest untaken probability, then the lowest id holding it; a taken expert holds -1, below every probability.
        float bv = -2.0;
        uint bi = 0xffffffffu;
        for (uint s = 0u; s < kHeld; ++s)
            if (pr[s] > bv) { bv = pr[s]; bi = t + 64u * s; }
        float m = group_max(bv, q);
        uint id = group_min(bv == m ? bi : 0xffffffffu, q);
        // Where none is left the pick is the invalid id, which marks nothing and adds 0.
        for (uint s = 0u; s < kHeld; ++s)
            if (t + 64u * s == id) pr[s] = -1.0;
        float c = id < n_expert ? m : 0.0;
        total += c;
        if (t == 0u) {
            chosen[q][j] = c;
            chosen_id[q][j] = id;
        }
    }
    barrier();
    if (live)
        for (uint j = t; j < k; j += 64u) {
            route_ids[r * k + j] = chosen_id[q][j];
            route_weights[r * k + j] = chosen_id[q][j] >= n_expert ? 0.0 : normalize ? chosen[q][j] / total : chosen[q][j];
        }
}
#endif
