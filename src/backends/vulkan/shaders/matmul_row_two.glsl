// Two-row builds (ROWS > 1): a cluster computes adjacent rows and hands each sum to ROW_STORE2; this body returns from the including function.
// The including function supplies row, nout, l, wrow, type, wide, ncols and the XCOL and OCOL column maps, as for matmul_row_one.glsl.
    if (ROWS > 1u) {
        // The two-row builds: cluster `row` takes rows r0 .. r0 + ROWS - 1, those past the matrix's last read as its last and written by no lane.
        uint r0 = row * ROWS, w0 = wrow - row + r0;
        if (r0 >= nout) return;
        uint rlast = nout - 1u - r0;
        // A plain chunk computes every column its build holds, so its column loop has no branch; a grouped run, often narrower than its build, stops at its entries.
#if defined(LLMX_PRESERVE_F32) && !(defined(LLMX_Q4) || defined(LLMX_K4) || defined(LLMX_K5) || defined(LLMX_KQUANT))
        precise float accr[ROWS][MAX_COLS];
#else
        float accr[ROWS][MAX_COLS];
#endif
        [[unroll]] for (uint r = 0u; r < ROWS; ++r) [[unroll]] for (uint c = 0u; c < MAX_COLS; ++c) accr[r][c] = 0.0;
#if defined(LLMX_LOOKUP_DOT) || defined(LLMX_K23) || defined(LLMX_MXFP4)
        // A product whose scale or sum leaves the fast path's range, which the one-column build's row recomputes (ROW_REPAIR).
        uint repair[ROWS][MAX_COLS];
        [[unroll]] for (uint r = 0u; r < ROWS; ++r) [[unroll]] for (uint c = 0u; c < MAX_COLS; ++c) repair[r][c] = 0u;
#endif
#ifdef LLMX_LOOKUP_DOT
        // Every 32-value group a lane's in the one-column build's order, whatever block holds it (lookup_group), so a type plugs in by its group decode alone.
        {
            uint nblocks = p.nin / 32u;
            uint wbr[ROWS];
            [[unroll]] for (uint r = 0u; r < ROWS; ++r) wbr[r] = w0 + min(r, rlast);
            for (uint b = l; b < nblocks; b += p.cluster) {
                float dwr[ROWS];
                uvec2 wr[ROWS][8];
                [[unroll]] for (uint r = 0u; r < ROWS; ++r) {
                    uint codes = lookup_group(wbr[r], p.nin, b, dwr[r]);
                    [[unroll]] for (uint j = 0u; j < 8u; ++j) wr[r][j] = lookup_pairs(codes, j);
                }
                [[unroll]] for (uint c = 0u; c < MAX_COLS; ++c) {
                    if (GROUPED && c >= ncols) break;
                    uint col = XCOL(min(c, ncols - 1u)), a2 = xa_col(col) / 2u + b * 8u;
                    uvec2 activations[8];
                    [[unroll]] for (uint j = 0u; j < 8u; ++j) activations[j] = xq2[a2 + j];
                    float dx = xq_block(col, b).x;
                    [[unroll]] for (uint r = 0u; r < ROWS; ++r) {
                        int sum = 0;
                        for (uint j = 0u; j < 8u; ++j) sum = dot_pairs(wr[r][j], activations[j], sum);
                        float scale = dwr[r] * dx;
                        if (lookup_unsafe_scale(dwr[r], dx, scale)) repair[r][c] = 1u;
                        accr[r][c] = fma(scale, float(sum), accr[r][c]);
                    }
                }
            }
        }
#endif
#ifdef LLMX_K23
        {
            uint nblocks = p.nin / 256u;
            uint first = l / 8u, part = l % 8u, stride = p.cluster / 8u;
            uint wbr[ROWS];
            [[unroll]] for (uint r = 0u; r < ROWS; ++r) wbr[r] = w0 + min(r, rlast);
            for (uint b = first; b < nblocks; b += stride) {
                uint j = part * 32u, block = b * 8u + part;
                uvec2 codesr[ROWS][8];
                vec4 weightsr[ROWS];
                [[unroll]] for (uint r = 0u; r < ROWS; ++r) {
                    uint o = (wbr[r] * nblocks + b) * K23_BYTES;
                    [[unroll]] for (uint w = 0u; w < 8u; ++w) codesr[r][w] = weight16(k23_word(o, j + 4u * w));
#ifdef LLMX_Q2
                    Q2KScale low = q2_k_scale(o, j), high = q2_k_scale(o, j + 16u);
                    weightsr[r] = vec4(low.scale, high.scale, low.min, high.min);
#else
                    weightsr[r] = vec4(q3_k_scale(o, j).scale, q3_k_scale(o, j + 16u).scale, 0.0, 0.0);
#endif
                }
                [[unroll]] for (uint c = 0u; c < MAX_COLS; ++c) {
                    if (GROUPED && c >= ncols) break;
                    uint col = XCOL(min(c, ncols - 1u)), a = col * p.nin / 4u + block * 8u;
                    uvec2 xl[4], xh[4];
                    [[unroll]] for (uint w = 0u; w < 4u; ++w) { xl[w] = xq2[a + w]; xh[w] = xq2[a + w + 4u]; }
                    float scale = xq_block(col, block).x;
                    bool unsafe = k23_repair_scale(scale);
                    int ls = 0, hs = 0;
#ifdef LLMX_Q2
                    [[unroll]] for (uint w = 0u; w < 4u; ++w) {
                        ls = dot_pairs(uvec2(0x00010001u), xl[w], ls);
                        hs = dot_pairs(uvec2(0x00010001u), xh[w], hs);
                    }
#endif
                    [[unroll]] for (uint r = 0u; r < ROWS; ++r) {
                        int lo = 0, hi = 0;
                        [[unroll]] for (uint w = 0u; w < 4u; ++w) {
                            lo = dot_pairs(codesr[r][w], xl[w], lo);
                            hi = dot_pairs(codesr[r][w + 4u], xh[w], hi);
                        }
                        if (unsafe) repair[r][c] = 1u;
                        accr[r][c] += k23_term(weightsr[r], scale, lo, hi, ls, hs);
                    }
                }
            }
        }
#endif
#if defined(LLMX_MXFP4) && defined(LLMX_DOT)
        {
            uint nblocks = p.nin / MXFP4_BLOCK;
            uint wbr[ROWS];
            [[unroll]] for (uint r = 0u; r < ROWS; ++r) wbr[r] = (w0 + min(r, rlast)) * nblocks * MXFP4_BYTES;
            for (uint block = l; block < nblocks; block += p.cluster) {
                uint er[ROWS], pairsr[ROWS][16];
                float dwr[ROWS];
                [[unroll]] for (uint r = 0u; r < ROWS; ++r) {
                    uvec4 quants;
                    er[r] = mxfp4_block(wbr[r] + block * MXFP4_BYTES, quants);
                    dwr[r] = mxfp4_scale(er[r]);
                    uint pairs[16];
                    mxfp4_pairs_of(quants, pairs);
                    [[unroll]] for (uint j = 0u; j < 16u; ++j) pairsr[r][j] = pairs[j];
                }
                [[unroll]] for (uint c = 0u; c < MAX_COLS; ++c) {
                    if (GROUPED && c >= ncols) break;
                    uint col = XCOL(min(c, ncols - 1u)), a2 = xa_col(col) / 2u + block * 8u;
                    float dx = xq_block(col, block).x;
                    [[unroll]] for (uint r = 0u; r < ROWS; ++r) {
                        uint pairs[16];
                        [[unroll]] for (uint j = 0u; j < 16u; ++j) pairs[j] = pairsr[r][j];
                        float sum = float(mxfp4_dot(pairs, a2));
                        float scale = dwr[r] * dx;
                        uint exponent = (floatBitsToUint(scale) >> 23u) & 255u;
                        // At 253 the largest decoded weight overflows f32; a combined scale must be normal.
                        if (er[r] >= MXFP4_FIRST_OVERFLOW_EXPONENT || exponent == 0u || exponent == 255u) repair[r][c] = 1u;
                        else accr[r][c] = fma(scale, sum, accr[r][c]);
                    }
                }
            }
        }
#endif
#ifdef LLMX_VALUES
        // F32, F16 and BF16 rows, each lane's values in the one-row build's order and form; Q8_0 rows of the plain module take its one-row builds alone.
        if (p.nin % 4u == 0u) {
            uint n4 = p.nin / 4u;
            uint wbr[ROWS];
            [[unroll]] for (uint r = 0u; r < ROWS; ++r) wbr[r] = (w0 + min(r, rlast)) * n4;
            for (uint i = l; i < n4; i += p.cluster) {
                vec4 wr[ROWS];
                [[unroll]] for (uint r = 0u; r < ROWS; ++r) wr[r] = WEIGHT4(wbr[r] + i);
                [[unroll]] for (uint c = 0u; c < MAX_COLS; ++c) {
                    if (GROUPED && c >= ncols) break;
                    vec4 xv = xt4[XCOL(min(c, ncols - 1u)) * n4 + i];
                    [[unroll]] for (uint r = 0u; r < ROWS; ++r) accr[r][c] += dot(wr[r], xv);
                }
            }
        } else {
            uint wbr[ROWS];
            [[unroll]] for (uint r = 0u; r < ROWS; ++r) wbr[r] = (w0 + min(r, rlast)) * p.nin;
            for (uint i = l; i < p.nin; i += p.cluster) {
                float wr[ROWS];
                [[unroll]] for (uint r = 0u; r < ROWS; ++r) wr[r] = WEIGHT1(wbr[r] + i);
                [[unroll]] for (uint c = 0u; c < MAX_COLS; ++c) {
                    if (GROUPED && c >= ncols) break;
                    float xv = x[XCOL(min(c, ncols - 1u)) * p.nin + i];
                    [[unroll]] for (uint r = 0u; r < ROWS; ++r) accr[r][c] += wr[r] * xv;
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
                    if (GROUPED && c >= ncols) break;
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
                    if (GROUPED && c >= ncols) break;
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
                    if (GROUPED && c >= ncols) break;
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
                    if (GROUPED && cc >= ncols) break;
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
                    if (GROUPED && c >= ncols) break;
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
#if defined(LLMX_LOOKUP_DOT) || defined(LLMX_K23) || defined(LLMX_MXFP4)
#ifdef LLMX_LOOKUP_DOT
#define ROW_REPAIR(wr, col) lookup_float_twin(wr, col, p.nin, 0u, p.nin / 32u)
#elif defined(LLMX_K23)
#define ROW_REPAIR(wr, col) k23_float_twin(wr, col, p.nin, 0u, p.nin / 32u)
#else
#define ROW_REPAIR(wr, col) mxfp4_wide(wr, col)
#endif
        // A cluster holding a product to repair reduces by the butterfly below, which carries the flags; otherwise a sum that is not finite is the only one to recompute.
        uint flagged = 0u;
        [[unroll]] for (uint r = 0u; r < ROWS; ++r) [[unroll]] for (uint c = 0u; c < MAX_COLS; ++c) flagged |= repair[r][c];
        for (uint off = p.cluster / 2u; off > 0u; off >>= 1u) flagged |= subgroupShuffleXor(flagged, off);
        const bool flags = flagged != 0u;
#define ROW_TREE (TREE_CLUSTER > 1u && !flags)
#else
#define ROW_TREE (TREE_CLUSTER > 1u)
#endif
        // The butterfly's pairs l ^ TREE_CLUSTER / 2 .. l ^ 1 in order, transposed so each level halves the values a lane holds; float adds commute, so the sums keep the butterfly's bits.
        if (ROW_TREE) {
            const uint NV = ROWS * MAX_COLS;
            const uint NLEFT = NV >= TREE_CLUSTER ? NV / (TREE_CLUSTER > 0u ? TREE_CLUSTER : 1u) : 1u;
            float t[ROWS * MAX_COLS];
            [[unroll]] for (uint r = 0u; r < ROWS; ++r) [[unroll]] for (uint c = 0u; c < MAX_COLS; ++c) t[r * MAX_COLS + c] = accr[r][c];
            uint idx = 0u;
            bool writer = true;
            [[unroll]] for (uint k = 0u; k < 8u; ++k) {
                if ((TREE_CLUSTER >> (k + 1u)) == 0u) break;
                const uint off = TREE_CLUSTER >> (k + 1u), h = NV >> (k + 1u);
                bool up = (l & off) != 0u;
                if (h > 0u) {
                    [[unroll]] for (uint j = 0u; j < h; ++j) {
                        float lo = t[j], hi = t[j + h];
                        t[j] = (up ? hi : lo) + subgroupShuffleXor(up ? lo : hi, off);
                    }
                    if (up) idx += h;
                } else {
                    t[0] = t[0] + subgroupShuffleXor(t[0], off);
                    if (up) writer = false;
                }
            }
            [[unroll]] for (uint j = 0u; j < NLEFT; ++j) {
                uint v = idx + j, r = v / MAX_COLS, c = v - r * MAX_COLS;
#ifdef ROW_REPAIR
                if (writer && c < ncols && r <= rlast && (isnan(t[j]) || isinf(t[j]))) t[j] = ROW_REPAIR(w0 + r, XCOL(c));
#endif
                if (writer && c < ncols && r <= rlast) {
                    uint o = OCOL(c) * nout + r0 + r;
                    if (p.accum != 0u) ys[pi].y[o] += t[j];
                    else ys[pi].y[o] = t[j];
                }
            }
            return;
        }
        // Each row and column reduced as the one-row builds reduce one.
        [[unroll]] for (uint c = 0u; c < MAX_COLS; ++c) {
            if (c >= ncols) break;
            [[unroll]] for (uint r = 0u; r < ROWS; ++r) {
                float s = accr[r][c];
                for (uint off = p.cluster / 2u; off > 0u; off >>= 1u) s += subgroupShuffleXor(s, off);
#ifdef ROW_REPAIR
                uint unsafe = repair[r][c];
                if (flags)
                    for (uint off = p.cluster / 2u; off > 0u; off >>= 1u) unsafe |= subgroupShuffleXor(unsafe, off);
                if (l == 0u && r <= rlast && (unsafe != 0u || isnan(s) || isinf(s))) s = ROW_REPAIR(w0 + r, XCOL(c));
#endif
                ROW_STORE2(c, r, s)
            }
        }
        return;
    }
#undef ROW_REPAIR
#undef ROW_TREE
