// The sum of red[0, 256) as the tree of halving levels adds it, red[i] + red[i + k] for i below k at each k from 128 down, returned to every lane of every subgroup.
// Each subgroup adds the levels at or above its width in registers and the rest by shuffles, so it takes no barrier of its own; u holds the 32-lane minimum the device check keeps (docs/src/backends-vulkan.md).
float tree_sum() {
    uint lane = gl_SubgroupInvocationID, m = 256u / gl_SubgroupSize;
    float u[8];
    for (uint j = 0u; j < m; ++j) u[j] = red[lane + gl_SubgroupSize * j];
    for (uint h = m / 2u; h > 0u; h >>= 1u)
        for (uint j = 0u; j < h; ++j) u[j] = u[j] + u[j + h];
    float t = u[0];
    for (uint k = gl_SubgroupSize / 2u; k > 0u; k >>= 1u) t += subgroupShuffleXor(t, k);
    return subgroupShuffle(t, 0u);
}
