// The narrow build of the MXFP4 float tile (matmul_tile.comp, LLMX_TILE_NARROW): a routed tile of up to 16 entries of one expert, a thread holding one of 256 rows and each of the run's columns.
// Each output is one ascending fma over the same exact values as in the 64-column build, and TILE_ROWS is that build's height, whose aligned rows share the input bounds, so the bits are that build's.
#define NARROW_COLS 16u
#define NARROW_ROWS 256u
shared vec4 nx[MXFP4_BLOCK][NARROW_COLS / 4u];
shared uint nbound[2u * (NARROW_ROWS / 32u)];
shared uint ncol[2u * NARROW_COLS];

// The 32 values of one block of a row, codes j and j + 16 sharing byte j + 1, from the five words that hold the block starting at byte `shift / 8` of the first.
void narrow_decode(uint w[5], uint shift, out uint e, out float v[32]) {
    uint b[5];
    [[unroll]] for (uint t = 0u; t < 4u; ++t) b[t] = funnel(w[t], w[t + 1u], shift);
    b[4] = w[4] >> shift;
    e = b[0] & 255u;
    float scale = mxfp4_scale(e);
    [[unroll]] for (uint j = 0u; j < 16u; ++j) {
        uint byte = (b[(j + 1u) / 4u] >> (8u * ((j + 1u) & 3u))) & 255u;
        v[j] = nvfp4_value(scale, byte & 15u);
        v[j + 16u] = nvfp4_value(scale, byte >> 4u);
    }
}

void main() {
    uint tid = gl_LocalInvocationID.x, y = gl_WorkGroupID.y;
    uint expert = tab[4u * y], first = tab[4u * y + 1u], ncols = tab[4u * y + 2u];
    if (ncols == 0u) return;
    uint row = gl_WorkGroupID.x * NARROW_ROWS + tid;
    bool rv = row < p.nout;
    uint nblocks = p.nin / MXFP4_BLOCK, last = uint(w32.length()) - 1u;
    uint o = (expert * p.nout + (rv ? row : 0u)) * nblocks * MXFP4_BYTES;
    // The input loader: 16 threads a column, two values each.
    uint xcol = tid / 16u, xk = 2u * (tid & 15u);
    bool xin = xcol < ncols;
    uint xc = xin ? tab[p.order0 + first + xcol] / p.per : 0u;
#ifdef LLMX_PRESERVE_F32
    precise float acc[NARROW_COLS];
#else
    float acc[NARROW_COLS];
#endif
    [[unroll]] for (uint c = 0u; c < NARROW_COLS; ++c) acc[c] = 0.0;
    // A thread's record over the blocks: its row's wide-scale flag and least and greatest other scale, and its inputs' least and greatest exponent.
    uint wflag = 0u, wlo = 256u, whi = 0u, xelo = 256u, xehi = 0u;
    uint pw[5];
    vec2 px;
    for (uint t = 0u; t < 5u; ++t) pw[t] = w32[min((o >> 2u) + t, last)];
    px = xin ? vec2(x[xc * p.nin + xk], x[xc * p.nin + xk + 1u]) : vec2(0.0);
    for (uint blk = 0u; blk < nblocks; ++blk) {
        vec2 xv = vec2(matrix_input(px.x), matrix_input(px.y));
        nx[xk][xcol / 4u][xcol & 3u] = xv.x;
        nx[xk + 1u][xcol / 4u][xcol & 3u] = xv.y;
        for (uint i = 0u; i < 2u; ++i) {
            uint va = floatBitsToUint(xv[i]) & 0x7fffffffu;
            if (va != 0u) { xelo = min(xelo, va >> 23u); xehi = max(xehi, va >> 23u); }
        }
        uint w[5];
        for (uint t = 0u; t < 5u; ++t) w[t] = pw[t];
        uint shift = (o & 3u) * 8u;
        barrier();
        // The next block's words and inputs are read while this one computes.
        if (blk + 1u < nblocks) {
            uint next = o + MXFP4_BYTES;
            for (uint t = 0u; t < 5u; ++t) pw[t] = w32[min((next >> 2u) + t, last)];
            uint k = (blk + 1u) * MXFP4_BLOCK + xk;
            px = xin ? vec2(x[xc * p.nin + k], x[xc * p.nin + k + 1u]) : vec2(0.0);
        }
        uint e;
        float v[32];
        narrow_decode(w, shift, e, v);
        if (rv) {
            if (wide_row(e)) wflag = 1u;
            else { wlo = min(wlo, e); whi = max(whi, e); }
        }
        // Columns four at a time, a group past the run skipped; the run's last group computes up to three columns nobody stores.
        [[unroll]] for (uint g = 0u; g < NARROW_COLS / 4u; ++g) {
            if (4u * g < ncols)
            [[unroll]] for (uint k = 0u; k < MXFP4_BLOCK; ++k) {
                vec4 b = nx[k][g];
                [[unroll]] for (uint j = 0u; j < 4u; ++j) {
#ifdef LLMX_PRESERVE_F32
                    acc[4u * g + j] = fma(v[k], b[j], acc[4u * g + j]);
#else
                    acc[4u * g + j] += v[k] * b[j];
#endif
                }
            }
        }
        o += MXFP4_BYTES;
        barrier();
    }
    // The input bounds come from the least and greatest scale summed in floats over each aligned TILE_ROWS rows, the 64-column build's tile.
    if (tid < 2u * (NARROW_ROWS / 32u)) nbound[tid] = (tid & 1u) == 0u ? 256u : 0u;
    if (tid < 2u * NARROW_COLS) ncol[tid] = (tid & 1u) == 0u ? 256u : 0u;
    barrier();
    uint group = tid / TILE_ROWS;
    if (rv && wlo <= whi) { atomicMin(nbound[2u * group], wlo); atomicMax(nbound[2u * group + 1u], whi); }
    if (xin && xelo <= xehi) { atomicMin(ncol[2u * xcol], xelo); atomicMax(ncol[2u * xcol + 1u], xehi); }
    barrier();
    if (!rv) return;
    uint lo = nbound[2u * group], hi = nbound[2u * group + 1u];
    uint xlo = lo >= 129u ? 0u : 129u - lo;
    uint xhi = hi == 0u ? 255u : 378u - hi;
    [[unroll]] for (uint c = 0u; c < NARROW_COLS; ++c) {
        if (c < ncols) {
        uint clo = ncol[2u * c], chi = ncol[2u * c + 1u];
        bool wide_col = clo <= chi && (clo < xlo || chi > xhi);
        uint col = tab[p.order0 + first + c];
        bool recompute = wflag != 0u || wide_col || isnan(acc[c]) || isinf(acc[c]);
        uint index = col * p.nout + row;
        float s = recompute ? float(mxfp4_wide(expert * p.nout + row, col / p.per, 0u, p.nin)) : acc[c];
        if (p.accum != 0u) s += uintBitsToFloat(ybits[index]);
        ybits[index] = floatBitsToUint(s);
        }
    }
}
