// IQ3_S row and tile workgroups cache the immutable dictionary once from IQ3_GRID_SOURCE.
shared uint iq3_s_grid[512];
void iq3_s_init() {
    for (uint i = gl_LocalInvocationIndex; i < 512u; i += gl_WorkGroupSize.x)
        iq3_s_grid[i] = IQ3_GRID_SOURCE(i);
    barrier();
}
#define IQ3_GRID_WORD(i) iq3_s_grid[i]
