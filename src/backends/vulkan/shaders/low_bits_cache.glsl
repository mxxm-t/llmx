shared uvec2 q2_code_pairs[256];
void lookup_q2_cache_init(bool signed_pairs) {
    for (uint raw = gl_LocalInvocationIndex; raw < 256u; raw += gl_WorkGroupSize.x) {
        uvec2 entry = lookup_q2_unsigned_pairs(raw);
        q2_code_pairs[raw] = signed_pairs ? lookup_q2_center_pairs(entry) : entry;
    }
    barrier();
}
uvec2 lookup_q2_cached_entry(uint raw) {
    return q2_code_pairs[raw & 255u];
}
uvec2 lookup_q2_cached_pairs(uint raw) {
    return lookup_q2_center_pairs(lookup_q2_cached_entry(raw));
}
