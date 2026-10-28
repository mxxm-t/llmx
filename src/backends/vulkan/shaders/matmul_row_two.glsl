// Two-row builds (ROWS > 1): a cluster computes adjacent rows and hands each sum to ROW_STORE2; this body returns from the including function.
// The including function supplies row, nout, l, wrow, type, wide, ncols and the XCOL and OCOL column maps, as for matmul_row_one.glsl.
    if (ROWS > 1u) {
        // The two-row builds: cluster `row` takes rows r0 .. r0 + ROWS - 1, those past the matrix's last read as its last and written by no lane.
        uint r0 = row * ROWS, w0 = wrow - row + r0;
        if (r0 >= nout) return;
        uint rlast = nout - 1u - r0;
        float accr[ROWS][MAX_COLS];
        [[unroll]] for (uint r = 0u; r < ROWS; ++r) [[unroll]] for (uint c = 0u; c < MAX_COLS; ++c) accr[r][c] = 0.0;
#if defined(LLMX_Q1_0) && defined(LLMX_LOW_BITS_DOT)
        uint repair[ROWS][MAX_COLS];
        [[unroll]] for (uint r = 0u; r < ROWS; ++r) [[unroll]] for (uint c = 0u; c < MAX_COLS; ++c) repair[r][c] = 0u;
        uint nblocks = p.nin / 32u;
        uint wbr[ROWS];
        [[unroll]] for (uint r = 0u; r < ROWS; ++r) wbr[r] = w0 + min(r, rlast);
        for (uint b = l; b < nblocks; b += p.cluster) {
            for (uint c = 0u; c < MAX_COLS; ++c) {
                if (c >= ncols) break;
                uint col = XCOL(c), a2 = xa_col(col) / 2u + b * 8u;
                uvec2 activations[8];
                [[unroll]] for (uint j = 0u; j < 8u; ++j) activations[j] = xq2[a2 + j];
                float dx = xq_block(col, b).x;
                [[unroll]] for (uint r = 0u; r < ROWS; ++r) {
                    float dw;
                    uint codes = lookup_group(wbr[r], p.nin, b, dw);
                    uvec2 weights[8];
                    [[unroll]] for (uint j = 0u; j < 8u; ++j) weights[j] = lookup_pairs(codes, j);
                    int sum = 0;
                    for (uint j = 0u; j < 8u; ++j) sum = dot_pairs(weights[j], activations[j], sum);
                    float scale = dw * dx;
                    if (lookup_unsafe_scale(dw, dx, scale)) repair[r][c] = 1u;
                    accr[r][c] = fma(scale, float(sum), accr[r][c]);
                }
            }
        }
#endif
#ifdef LLMX_Q4
        if (type == TYPE_Q4_0 && wide != 0u) {
            uint nblocks = p.nin / Q4_0_BLOCK;
            uint npairs = nblocks / 2u;
            uint group = l / 2u, t = l - group * 2u;
            uint groups = p.cluster / 2u;
            uint wbr[ROWS];
            [[unroll]] for (uint r = 0u; r < ROWS; ++r) wbr[r] = (w0 + min(r, rlast)) * nblocks * 9u / 2u;
            for (uint j = group; j < npairs; j += groups) {
                float dr[ROWS];
                uint n0r[ROWS], n1r[ROWS], n2r[ROWS], n3r[ROWS];
                [[unroll]] for (uint r = 0u; r < ROWS; ++r) {
                    uint o = wbr[r] + j * 9u + 4u * t;
                    Q40W_WEIGHTS(o)
                    dr[r] = d; n0r[r] = n0; n1r[r] = n1; n2r[r] = n2; n3r[r] = n3;
                }
                uint b = 2u * j + t;
                [[unroll]] for (uint c = 0u; c < MAX_COLS; ++c) {
                    uint col = XCOL(min(c, ncols - 1u));
                    uint a4 = xa_col(col) / 4u + b * 4u;
                    XQ4T lo0 = xq4[a4], lo1 = xq4[a4 + 1u], hi0 = xq4[a4 + 2u], hi1 = xq4[a4 + 3u];
                    vec2 xb = xq_block(col, b);
                    [[unroll]] for (uint r = 0u; r < ROWS; ++r) {
                        QDOT pl = QDOT(0);
                        QDOT ph = QDOT(0);
                        Q4_DOTS(n0r[r], n1r[r], n2r[r], n3r[r], lo0, lo1, hi0, hi1)
                        accr[r][c] += Q40_TERM(dr[r], xb);
                    }
                }
            }
        } else if (type == TYPE_Q4_0) {
            uint nblocks = p.nin / Q4_0_BLOCK;
            uint wbr[ROWS];
            [[unroll]] for (uint r = 0u; r < ROWS; ++r) wbr[r] = (w0 + min(r, rlast)) * nblocks * Q4_0_BYTES;
            for (uint b = l; b < nblocks; b += p.cluster) {
                float dr[ROWS];
                uint n0r[ROWS], n1r[ROWS], n2r[ROWS], n3r[ROWS];
                [[unroll]] for (uint r = 0u; r < ROWS; ++r) {
                    uint o = wbr[r] + b * Q4_0_BYTES;
                    Q40_WEIGHTS(o)
                    dr[r] = d; n0r[r] = n0; n1r[r] = n1; n2r[r] = n2; n3r[r] = n3;
                }
                [[unroll]] for (uint c = 0u; c < MAX_COLS; ++c) {
                    uint col = XCOL(min(c, ncols - 1u));
                    uint a4 = xa_col(col) / 4u + b * 4u;
                    XQ4T lo0 = xq4[a4], lo1 = xq4[a4 + 1u], hi0 = xq4[a4 + 2u], hi1 = xq4[a4 + 3u];
                    vec2 xb = xq_block(col, b);
                    [[unroll]] for (uint r = 0u; r < ROWS; ++r) {
                        QDOT pl = QDOT(0);
                        QDOT ph = QDOT(0);
                        Q4_DOTS(n0r[r], n1r[r], n2r[r], n3r[r], lo0, lo1, hi0, hi1)
                        accr[r][c] += Q40_TERM(dr[r], xb);
                    }
                }
            }
        } else {
            uint nblocks = p.nin / 32u;
#ifdef LLMX_Q50
            const uint words = Q5_0_BYTES;
#else
            uint words = AFFINE_Q5 ? 6u : 5u;
#endif
            uint wbr[ROWS];
            [[unroll]] for (uint r = 0u; r < ROWS; ++r) wbr[r] = (w0 + min(r, rlast)) * nblocks * words;
            for (uint b = l; b < nblocks; b += p.cluster) {
                vec2 dmr[ROWS];
                uint w1r[ROWS], w2r[ROWS], w3r[ROWS], w4r[ROWS], hbr[ROWS];
                [[unroll]] for (uint r = 0u; r < ROWS; ++r) {
                    uint o = wbr[r] + b * words;
                    AFFINE_WEIGHTS(o)
                    dmr[r] = dm; w1r[r] = w1; w2r[r] = w2; w3r[r] = w3; w4r[r] = w4;
                    hbr[r] = hb;
                }
                [[unroll]] for (uint c = 0u; c < MAX_COLS; ++c) {
                    uint col = XCOL(min(c, ncols - 1u));
                    uint a4 = xa_col(col) / 4u + b * 4u;
                    XQ4T lo0 = xq4[a4], lo1 = xq4[a4 + 1u], hi0 = xq4[a4 + 2u], hi1 = xq4[a4 + 3u];
                    vec2 xb = xq_block(col, b);
                    [[unroll]] for (uint r = 0u; r < ROWS; ++r) {
                        QDOT pl = QDOT(0);
                        QDOT ph = QDOT(0);
                        AFFINE_DOTS(w1r[r], w2r[r], w3r[r], w4r[r], hbr[r], lo0, lo1, hi0, hi1)
                        Q41_ADD(accr[r][c], dmr[r], xb);
                    }
                }
            }
        }
#endif
#if defined(LLMX_K4) || defined(LLMX_K5)
        {
#ifdef LLMX_K5
            const uint q5 = 1u;
#else
            const uint q5 = 0u;
#endif
            uint bw = (q5 != 0u ? Q5_K_BYTES : Q4_K_BYTES) / 4u;
            uint nblocks = p.nin / Q6_K_BLOCK;
            uint group = l / 8u, t = l - group * 8u;
            uint groups = p.cluster / 8u;
            uint c = t / 2u, h = t - c * 2u;
            uint sl = 2u * c, sh = 2u * c + 1u;
            uint wbr[ROWS];
            [[unroll]] for (uint r = 0u; r < ROWS; ++r) wbr[r] = (w0 + min(r, rlast)) * nblocks * bw;
            for (uint b = group; b < nblocks; b += groups) {
                uvec4 nqr[ROWS], hbr[ROWS];
                float dlr[ROWS], dhr[ROWS], mlr[ROWS], mhr[ROWS];
                [[unroll]] for (uint r = 0u; r < ROWS; ++r) {
                    uint o = wbr[r] + b * bw;
                    K45_WEIGHTS(o)
                    nqr[r] = nq; hbr[r] = hbv; dlr[r] = dl; dhr[r] = dh; mlr[r] = ml; mhr[r] = mh;
                }
                uint bl = b * 8u + 2u * c;   // the low group's 32-block, the high group's is next
                [[unroll]] for (uint cc = 0u; cc < MAX_COLS; ++cc) {
                    uint col = XCOL(min(cc, ncols - 1u));
                    uint a4 = xa_col(col) / 4u + (b * 256u + c * 64u + h * 16u) / 8u;
                    XQ4T lo0 = xq4[a4], hi0 = xq4[a4 + 4u], lo1 = xq4[a4 + 1u], hi1 = xq4[a4 + 5u];
                    vec4 xt = xq_pair(col, bl / 2u);   // d, ds of the low group, then the high
                    [[unroll]] for (uint r = 0u; r < ROWS; ++r) {
                        QDOT pl = QDOT(0);
                        QDOT ph = QDOT(0);
                        uvec4 q = nqr[r], hb = hbr[r];
                        nibble_dots(q.x, fifth(hb.x, sl), fifth(hb.x, sl + 8u), fifth(hb.x, sh), fifth(hb.x, sh + 8u), XA(lo0), XA(hi0), pl, ph);
                        nibble_dots(q.y, fifth(hb.y, sl), fifth(hb.y, sl + 8u), fifth(hb.y, sh), fifth(hb.y, sh + 8u), XB(lo0), XB(hi0), pl, ph);
                        nibble_dots(q.z, fifth(hb.z, sl), fifth(hb.z, sl + 8u), fifth(hb.z, sh), fifth(hb.z, sh + 8u), XA(lo1), XA(hi1), pl, ph);
                        nibble_dots(q.w, fifth(hb.w, sl), fifth(hb.w, sl + 8u), fifth(hb.w, sh), fifth(hb.w, sh + 8u), XB(lo1), XB(hi1), pl, ph);
                        accr[r][cc] += K45_TERM(dlr[r], dhr[r], mlr[r], mhr[r], xt);
                    }
                }
            }
        }
#endif
#ifdef LLMX_KQUANT
        {
            uint nblocks = p.nin / Q6_K_BLOCK;
            uint group = l / 8u, t = l - group * 8u;
            uint groups = p.cluster / 8u;
            uint n = t / 4u, l0 = (t - n * 4u) * 8u;
            uint wbr[ROWS];
            [[unroll]] for (uint r = 0u; r < ROWS; ++r) wbr[r] = (w0 + min(r, rlast)) * nblocks * Q6_K_BYTES;   // bytes
            for (uint b = group; b < nblocks; b += groups) {
                float dr[ROWS], s0r[ROWS], s1r[ROWS], s2r[ROWS], s3r[ROWS];
                uint q0ar[ROWS], q0br[ROWS], q1ar[ROWS], q1br[ROWS], q2ar[ROWS], q2br[ROWS], q3ar[ROWS], q3br[ROWS];
                [[unroll]] for (uint r = 0u; r < ROWS; ++r) {
                    uint o = wbr[r] + b * Q6_K_BYTES;
                    Q6K_WEIGHTS(o)
                    dr[r] = d; s0r[r] = s0; s1r[r] = s1; s2r[r] = s2; s3r[r] = s3;
                    q0ar[r] = q0a; q0br[r] = q0b; q1ar[r] = q1a; q1br[r] = q1b; q2ar[r] = q2a; q2br[r] = q2b; q3ar[r] = q3a; q3br[r] = q3b;
                }
                uint bg = b * 8u + n * 4u;   // the 32-block of group 0
                [[unroll]] for (uint c = 0u; c < MAX_COLS; ++c) {
                    uint col = XCOL(min(c, ncols - 1u));
                    uint a4 = xa_col(col) / 4u + (b * Q6_K_BLOCK + n * 128u + l0) / 8u;
                    XQ4T x0v = xq4[a4], x1v = xq4[a4 + 4u], x2v = xq4[a4 + 8u], x3v = xq4[a4 + 12u];
                    vec4 xa = xq_pair(col, bg / 2u), xb = xq_pair(col, bg / 2u + 1u);   // d, ds of groups 0..3
                    [[unroll]] for (uint r = 0u; r < ROWS; ++r) {
#ifdef LLMX_I8
                        int p0 = q6_dot(q0ar[r], q0br[r], x0v), p1 = q6_dot(q1ar[r], q1br[r], x1v);
                        int p2 = q6_dot(q2ar[r], q2br[r], x2v), p3 = q6_dot(q3ar[r], q3br[r], x3v);
#else
                        QDOT p0 = pair_dot(q0ar[r] & 0x00FF00FFu, (q0ar[r] >> 8u) & 0x00FF00FFu, x0v.xy) + pair_dot(q0br[r] & 0x00FF00FFu, (q0br[r] >> 8u) & 0x00FF00FFu, x0v.zw);
                        QDOT p1 = pair_dot(q1ar[r] & 0x00FF00FFu, (q1ar[r] >> 8u) & 0x00FF00FFu, x1v.xy) + pair_dot(q1br[r] & 0x00FF00FFu, (q1br[r] >> 8u) & 0x00FF00FFu, x1v.zw);
                        QDOT p2 = pair_dot(q2ar[r] & 0x00FF00FFu, (q2ar[r] >> 8u) & 0x00FF00FFu, x2v.xy) + pair_dot(q2br[r] & 0x00FF00FFu, (q2br[r] >> 8u) & 0x00FF00FFu, x2v.zw);
                        QDOT p3 = pair_dot(q3ar[r] & 0x00FF00FFu, (q3ar[r] >> 8u) & 0x00FF00FFu, x3v.xy) + pair_dot(q3br[r] & 0x00FF00FFu, (q3br[r] >> 8u) & 0x00FF00FFu, x3v.zw);
#endif
                        accr[r][c] += Q6K_TERM(dr[r], s0r[r], s1r[r], s2r[r], s3r[r], xa, xb);
                    }
                }
            }
        }
#endif
        // Each row and column reduced as the one-row builds reduce one.
        [[unroll]] for (uint c = 0u; c < MAX_COLS; ++c) {
            if (c >= ncols) break;
            [[unroll]] for (uint r = 0u; r < ROWS; ++r) {
                float s = accr[r][c];
                for (uint off = p.cluster / 2u; off > 0u; off >>= 1u) s += subgroupShuffleXor(s, off);
#if defined(LLMX_Q1_0) && defined(LLMX_LOW_BITS_DOT)
                uint unsafe = repair[r][c];
                for (uint off = p.cluster / 2u; off > 0u; off >>= 1u) unsafe |= subgroupShuffleXor(unsafe, off);
                if (l == 0u && r <= rlast && (unsafe != 0u || isnan(s) || isinf(s)))
                    s = lookup_float_twin(w0 + r, XCOL(c), p.nin, 0u, p.nin / 32u);
#endif
                ROW_STORE2(c, r, s)
            }
        }
        return;
    }
