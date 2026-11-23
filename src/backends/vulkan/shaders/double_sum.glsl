// The sum of every lane's v over a 256-lane workgroup as the GPU core's block_sum_double adds it, red2[i] + red2[i + k] for i below k at each k from 128 down, returned to every lane.
// Additions alone, so no contraction can change it; the caller's shared red2 holds 256 doubles.
double double_sum(double v) {
    uint lid = gl_LocalInvocationID.x;
    red2[lid] = v;
    barrier();
    for (uint k = 128u; k > 0u; k >>= 1u) {
        if (lid < k) red2[lid] = red2[lid] + red2[lid + k];
        barrier();
    }
    double s = red2[0];
    barrier();
    return s;
}
