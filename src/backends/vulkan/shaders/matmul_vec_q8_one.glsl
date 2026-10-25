// The Q8_0 decode kernel's body (matmul_vec_q8.comp): a subgroup's dots of ROWS weight rows from wrow0 + row0 with up to MAX_COLS columns, reduced, each row and column's sum handed to VEC_STORE or, with TREE, VEC_TREE_STORE.
// The including function supplies pi, nout, lane, S, row0, wrow0, ncols and the XCOL and OCOL column maps; a kernel fusing a Q8_0 row product with other work includes the same text, so a row has the same bits in both.
    uint nblk = p.nin / 32u;
#ifdef LLMX_I8
    uint xtab = p.nbatch * p.nin / 8u;
#else
    uint xtab = p.nbatch * p.nin / 4u;
#endif
#ifdef LLMX_PRESERVE_F32
    // Keep normal-input accumulation identical to the ordinary module when float preservation changes contraction.
    precise float acc[ROWS][MAX_COLS];
#else
    float acc[ROWS][MAX_COLS];
#endif
    [[unroll]] for (uint r = 0u; r < ROWS; ++r) [[unroll]] for (uint c = 0u; c < MAX_COLS; ++c) acc[r][c] = 0.0;
    if (HALF && (nblk & 1u) == 0u && S == 64u) {
        // Each row's first word, a multiple of 17 words since the row's first block is even; per lane its four quant words, a fifth where the bytes straddle, its scale word and its activation offsets, which a step moves by 272 words.
        const uint bh = lane >> 1u;
        uint rw[ROWS];
        [[unroll]] for (uint r = 0u; r < ROWS; ++r) rw[r] = ((wrow0 + min(row0 + r, nout - 1u)) * nblk) * 17u / 2u;
        const bool odd = (bh & 1u) == 0u;
        uint lq = (bh * 34u + 2u + (lane & 1u) * 16u) >> 2u;
        uint ls = (bh * 34u) >> 2u;
        uint l5 = lq + (odd ? 4u : 3u);
        uint lx = lane;
        uint lb = bh;
        const uint nin16 = p.nin / 16u;
// Column g + J's products, one for each row, added to its accumulators.
#ifdef LLMX_I8
#define HX0(J) xa[J].x
#define HX1(J) xa[J].y
#define HX2(J) xa[J].z
#define HX3(J) xa[J].w
#else
#define HX0(J) xa[J].xy
#define HX1(J) xa[J].zw
#define HX2(J) xb[J].xy
#define HX3(J) xb[J].zw
#endif
#define HALF_PRODUCTS(J) [[unroll]] for (uint r = 0u; r < ROWS; ++r) { \
            int s = dot_word(wq[r][3], HX3(J), dot_word(wq[r][2], HX2(J), dot_word(wq[r][1], HX1(J), dot_word(wq[r][0], HX0(J), 0)))); \
            precise float product = dw[r] * dx[J] * float(s); \
            acc[r][g + (J)] += product; \
        }
        for (uint bb = bh; bb < nblk; bb += 32u) {
            WQT wq[ROWS][4];
            float dw[ROWS];
            [[unroll]] for (uint r = 0u; r < ROWS; ++r) {
                uint t0 = w32s[pi].w32[lq + rw[r]], t1 = w32s[pi].w32[lq + rw[r] + 1u], t2 = w32s[pi].w32[lq + rw[r] + 2u], t3 = w32s[pi].w32[lq + rw[r] + 3u];
                uint t4 = w32s[pi].w32[l5 + rw[r]];
                wq[r][0] = WIDEN(odd ? (t0 >> 16u) | (t1 << 16u) : t0);
                wq[r][1] = WIDEN(odd ? (t1 >> 16u) | (t2 << 16u) : t1);
                wq[r][2] = WIDEN(odd ? (t2 >> 16u) | (t3 << 16u) : t2);
                wq[r][3] = WIDEN(odd ? (t3 >> 16u) | (t4 << 16u) : t3);
                uint sw = w32s[pi].w32[ls + rw[r]];
                dw[r] = unpackHalf2x16(odd ? sw : sw >> 16u).x;
            }
            [[unroll]] for (uint g = 0u; g < MAX_COLS; g += GROUP) {
                if (g > 0u && (CHECK_EVERY_GROUP || 2u * g > MAX_COLS) && g >= ncols) break;
#ifdef LLMX_I8
                uvec4 xa[GROUP];
#else
                uvec4 xa[GROUP], xb[GROUP];
#endif
                float dx[GROUP];
                [[unroll]] for (uint j = 0u; j < GROUP; ++j) {
                    uint col = XCOL(min(g + j, ncols - 1u));
#ifdef LLMX_I8
                    xa[j] = XQ_AT(col * nin16 + lx);
#else
                    xa[j] = XQ_AT(2u * (col * nin16 + lx));
                    xb[j] = XQ_AT(2u * (col * nin16 + lx) + 1u);
#endif
                    dx[j] = XT2_AT(xtab + col * nblk + lb).x;
                }
                if (!PARTIAL || 2u * g < MAX_COLS || g + GROUP <= ncols) {
                    [[unroll]] for (uint j = 0u; j < GROUP; ++j) HALF_PRODUCTS(j)
                } else {
                    [[unroll]] for (uint j = 0u; j < GROUP; ++j) if (g + j < ncols) HALF_PRODUCTS(j)
                }
            }
            lq += 272u; ls += 272u; l5 += 272u; lx += 64u; lb += 32u;
        }
    } else {
        uint q = lane & 3u;
        // The third word a lane reads is clamped to the buffer, since an aligned last quarter of the matrix ends at its end; it is used only where the bytes straddle.
        uint wlast = w32s[pi].w32.length() - 1u;
        const uint SB = S >> 2u;
        // A lane's steps are blocks b, b + SB and on: STEPS at a time while that many remain, then one at a time.
        uint b = lane >> 2u;
// Column g + J's products of step d, one for each row, added to its accumulators.
#ifdef LLMX_I8
#define QXT uvec2
#define QX(i) XQ8_AT(i)
#define QX0(J) xa[J].x
#define QX1(J) xa[J].y
#else
#define QXT uvec4
#define QX(i) XQ_AT(i)
#define QX0(J) xa[J].xy
#define QX1(J) xa[J].zw
#endif
#define COLUMN_PRODUCTS(J) [[unroll]] for (uint r = 0u; r < ROWS; ++r) { \
        int s = dot_word(w1[d][r], QX1(J), dot_word(w0[d][r], QX0(J), 0)); \
        precise float product = dw[d][r] * dx[J] * float(s); \
            acc[r][g + (J)] += product; \
    }
// NS steps of a lane from block b: every row's words for all NS steps first, then per group of columns the NS steps in order, so each column still adds its blocks in order.
// No branch separates the loads, so every weight load of the batch is issued before the first product.
#define QUARTER_STEPS(NS) { \
        WQT w0[NS][ROWS], w1[NS][ROWS]; \
        float dw[NS][ROWS]; \
        [[unroll]] for (uint d = 0u; d < NS; ++d) \
            [[unroll]] for (uint r = 0u; r < ROWS; ++r) { \
                uint ob = ((wrow0 + min(row0 + r, nout - 1u)) * nblk + b + d * SB) * 34u; \
                uint qb = ob + 2u + q * 8u; \
                uint i = qb >> 2u; \
                bool odd = (qb & 2u) != 0u; \
                uint t0 = w32s[pi].w32[i], t1 = w32s[pi].w32[i + 1u], t2 = w32s[pi].w32[min(i + 2u, wlast)]; \
                w0[d][r] = WIDEN(odd ? (t0 >> 16u) | (t1 << 16u) : t0); \
                w1[d][r] = WIDEN(odd ? (t1 >> 16u) | (t2 << 16u) : t1); \
                uint sw = w32s[pi].w32[ob >> 2u]; \
                dw[d][r] = unpackHalf2x16((ob & 2u) != 0u ? sw >> 16u : sw).x; \
            } \
        [[unroll]] for (uint g = 0u; g < MAX_COLS; g += GROUP) { \
            if (g > 0u && (CHECK_EVERY_GROUP || 2u * g > MAX_COLS) && g >= ncols) break; \
            [[unroll]] for (uint d = 0u; d < NS; ++d) { \
                uint bd = b + d * SB; \
                QXT xa[GROUP]; \
                float dx[GROUP]; \
                [[unroll]] for (uint j = 0u; j < GROUP; ++j) { \
                    uint col = XCOL(min(g + j, ncols - 1u)); \
                    xa[j] = QX((col * p.nin + bd * 32u + q * 8u) / 8u); \
                    dx[j] = XT2_AT(xtab + col * nblk + bd).x; \
                } \
                if (!PARTIAL || 2u * g < MAX_COLS || g + GROUP <= ncols) { \
                    [[unroll]] for (uint j = 0u; j < GROUP; ++j) COLUMN_PRODUCTS(j) \
                } else { \
                    [[unroll]] for (uint j = 0u; j < GROUP; ++j) if (g + j < ncols) COLUMN_PRODUCTS(j) \
                } \
            } \
        } \
    }
        for (; b + (STEPS - 1u) * SB < nblk; b += STEPS * SB) QUARTER_STEPS(STEPS)
        if (STEPS > 1u)
            for (; b < nblk; b += SB) QUARTER_STEPS(1u)
    }
    if (TREE && S == 64u) {
        // Level l pairs a lane's values i and i + h, keeps one and sends the other to its partner, and adds what the partner sends; where fewer values than levels remain, a level adds the one value both hold.
        // Each lane then holds whole sums of values idx to idx + NLEFT - 1, and the one lane that holds a value alone writes it.
        float t[NV];
        [[unroll]] for (uint r = 0u; r < ROWS; ++r) [[unroll]] for (uint c = 0u; c < MAX_COLS; ++c) t[r * MAX_COLS + c] = acc[r][c];
        uint idx = 0u;
        bool writer = true;
        [[unroll]] for (uint l = 0u; l < 6u; ++l) {
            bool up = keeps_upper(l, lane);
            if (l < LOGNV) {
                const uint h = NV >> (l + 1u);
                [[unroll]] for (uint j = 0u; j < h; ++j) {
                    float lo = t[j], hi = t[j + h];
                    t[j] = (up ? hi : lo) + partner(up ? lo : hi, l);
                }
                if (up) idx += h;
            } else {
                t[0] = t[0] + partner(t[0], l);
                if (up) writer = false;
            }
        }
        [[unroll]] for (uint j = 0u; j < NLEFT; ++j) {
            uint v = idx + j;
            uint r = v / MAX_COLS, c = v % MAX_COLS;
            uint row = row0 + r;
            VEC_TREE_STORE(c, row, t[j])
        }
        return;
    }
    [[unroll]] for (uint r = 0u; r < ROWS; ++r) {
        uint row = row0 + r;
        [[unroll]] for (uint c = 0u; c < MAX_COLS; ++c) {
            if (c >= ncols) break;
            float s = subgroupAdd(acc[r][c]);
            VEC_STORE(c, row, s)
        }
    }
