// The rows of one workgroup of attention_tile.comp, included once for each ROWS, the query rows a lane holds, as the function TILE_ROWS.
// The workgroup's rows start at row `base` of view v; with ROWS 2 they are two 32-row tiles, a lane holding rows i and i + 16 of one.
void TILE_ROWS(uint v, uint base, uint h, uint nq) {
    const uint PER = TQ / ROWS;
    uint hist = view_hist(v), row0 = view_row0(v);
    uint kvh = h / (p.n_head / p.n_head_kv);
    uint lid = gl_LocalInvocationID.x;
    uint rr = lid >> 3u, lane = lid & 7u;            // row group of the workgroup, lane within it
    uint sub = rr / PER;                             // the lane's 32-row tile within the workgroup
    uint r[ROWS], b[ROWS], qbase[ROWS], my_end[ROWS]; // the lane's rows of the workgroup and of the view
    bool live[ROWS];
    uint qstride = p.n_head * DIM;
    for (uint j = 0u; j < ROWS; ++j) {
        r[j] = sub * TQ + (rr - sub * PER) + j * PER;
        b[j] = base + r[j];
        live[j] = b[j] < nq;
        qbase[j] = (row0 + min(b[j], nq - 1u)) * qstride + h * DIM;
        my_end[j] = hist + b[j] + 1u;
    }
    vec4 qv[ROWS][RUNS], acc[ROWS][RUNS];
    float m[ROWS], l[ROWS];
    for (uint j = 0u; j < ROWS; ++j) {
        for (uint k = 0u; k < RUNS; ++k) {
            uint d = 32u * k + 4u * lane;
            qv[j][k] = vec4(q[qbase[j] + d], q[qbase[j] + d + 1u], q[qbase[j] + d + 2u], q[qbase[j] + d + 3u]) * p.scale;
            acc[j][k] = vec4(0.0);
        }
        m[j] = -1.0 / 0.0;
        l[j] = 0.0;
    }
    // The last token any row of the workgroup attends, exclusive, and the same for the lane's 32-row tile, which attends nothing when it holds no live row.
    uint tile_end = min(hist + base + ROWS * TQ, hist + nq);
#if ROWS > 1
    uint sub_first = base + sub * TQ;
    uint sub_end = sub_first < nq ? min(hist + sub_first + TQ, hist + nq) : 0u;
#else
    uint sub_end = tile_end;
#endif
    uint bf = p.n_head_kv * p.bt * DIM;
    for (uint t0 = 0u; t0 < tile_end; t0 += TK) {
        // Stage the tile eight values a load, zero past the workgroup's last token so a masked score meets a finite value.
        for (uint i = lid; i < TK * DIM / 8u; i += 256u) {
            uint j = i / (DIM / 8u), d = (i - j * (DIM / 8u)) * 8u;
            uint t = t0 + j;
            vec4 k0 = vec4(0.0), k1 = vec4(0.0), v0 = vec4(0.0), v1 = vec4(0.0);
            if (t < tile_end) {
                uint base = view_block(v, t / p.bt) * bf + (kvh * p.bt + t - (t / p.bt) * p.bt) * DIM + d;
                k_load8(base, k0, k1);
                v_load8(base, v0, v1);
            }
            uint s = (j * DIM + d) / 4u;
            ks[s] = k0; ks[s + 1u] = k1;
            vs[s] = v0; vs[s + 1u] = v1;
        }
        barrier();
        // A 32-row tile's lanes, whole subgroups, skip the key tiles past its own last token.
        if (t0 < sub_end) {
            // Phase one: partial dots eight tokens at a time, then the butterfly, which leaves token 8g + l's score in lane l.
            float sc[ROWS][OWN];
            float mt[ROWS];
            for (uint j = 0u; j < ROWS; ++j) mt[j] = -1.0 / 0.0;
            for (uint g = 0u; g < OWN; ++g) {
                float part[ROWS][8];
                for (uint t = 0u; t < 8u; ++t) {
                    float s[ROWS];
                    for (uint j = 0u; j < ROWS; ++j) s[j] = 0.0;
                    for (uint k = 0u; k < RUNS; ++k) {
                        vec4 kk = ks[(8u * g + t) * (DIM / 4u) + 8u * k + lane];
                        for (uint j = 0u; j < ROWS; ++j) s[j] += dot(qv[j][k], kk);
                    }
                    for (uint j = 0u; j < ROWS; ++j) part[j][t] = s[j];
                }
                // Every index a constant, so no compiler keeps part in scratch memory for a loop it leaves rolled.
                for (uint j = 0u; j < ROWS; ++j) {
                    ATTN_BFLY(j, 0u, 4u) ATTN_BFLY(j, 1u, 4u) ATTN_BFLY(j, 2u, 4u) ATTN_BFLY(j, 3u, 4u)
                    ATTN_BFLY(j, 0u, 2u) ATTN_BFLY(j, 1u, 2u)
                    ATTN_BFLY(j, 0u, 1u)
                }
                for (uint j = 0u; j < ROWS; ++j) {
                    sc[j][g] = t0 + 8u * g + lane < my_end[j] ? part[j][0] : -1.0 / 0.0;
                    mt[j] = max(mt[j], sc[j][g]);
                }
            }
            for (uint j = 0u; j < ROWS; ++j) {
                mt[j] = max(mt[j], subgroupShuffleXor(mt[j], 1u));
                mt[j] = max(mt[j], subgroupShuffleXor(mt[j], 2u));
                mt[j] = max(mt[j], subgroupShuffleXor(mt[j], 4u));
                float mn = max(m[j], mt[j]);
                // A row with nothing unmasked yet keeps m at -inf; its weights are zero rather than the NaN of inf - inf.
                float corr = mn == -1.0 / 0.0 ? 1.0 : exp(m[j] - mn);
                float sum = 0.0;
                for (uint g = 0u; g < OWN; ++g) {
                    float w = sc[j][g] == -1.0 / 0.0 ? 0.0 : exp(sc[j][g] - mn);
                    ps[r[j] * TK + 8u * g + lane] = w;
                    sum += w;
                }
                sum += subgroupShuffleXor(sum, 1u);
                sum += subgroupShuffleXor(sum, 2u);
                sum += subgroupShuffleXor(sum, 4u);
                l[j] = l[j] * corr + sum;
                m[j] = mn;
                for (uint k = 0u; k < RUNS; ++k) acc[j][k] *= corr;
            }
        }
        // A row's eight lanes are one subgroup, so their probabilities are visible to each other past this point.
        // Every subgroup takes this barrier, skipping tile or not, since a compiler may implement it as a workgroup barrier.
        subgroupMemoryBarrierShared();
        subgroupBarrier();
        if (t0 < sub_end) {
            // Phase two: the weighted sum of the tile's values.
#if ROWS > 1
            // Values past the 32-row tile's last token read as the zeros a workgroup of that tile alone would have staged.
            if (t0 + TK > sub_end) {
                for (uint t = 0u; t < TK; ++t) {
                    float w[ROWS];
                    for (uint j = 0u; j < ROWS; ++j) w[j] = ps[r[j] * TK + t];
                    bool in_tile = t0 + t < sub_end;
                    for (uint k = 0u; k < RUNS; ++k) {
                        vec4 vv = in_tile ? vs[t * (DIM / 4u) + 8u * k + lane] : vec4(0.0);
                        for (uint j = 0u; j < ROWS; ++j) acc[j][k] += w[j] * vv;
                    }
                }
            } else
#endif
            for (uint t = 0u; t < TK; ++t) {
                float w[ROWS];
                for (uint j = 0u; j < ROWS; ++j) w[j] = ps[r[j] * TK + t];
                for (uint k = 0u; k < RUNS; ++k) {
                    vec4 vv = vs[t * (DIM / 4u) + 8u * k + lane];
                    for (uint j = 0u; j < ROWS; ++j) acc[j][k] += w[j] * vv;
                }
            }
        }
        barrier();
    }
    for (uint j = 0u; j < ROWS; ++j)
        for (uint k = 0u; k < RUNS; ++k) {
            uint d = 32u * k + 4u * lane;
            vec4 a = acc[j][k] / l[j];
            if (live[j]) { o[qbase[j] + d] = a.x; o[qbase[j] + d + 1u] = a.y; o[qbase[j] + d + 2u] = a.z; o[qbase[j] + d + 3u] = a.w; }
            // Dimensions 32k .. 32k + 31 of head h are block RUNS h + k of the row, held by its eight lanes in order.
            if (p.quant != 0u) xquant_word(live[j] ? a : vec4(0.0), live[j], lane, (row0 + b[j]) * (qstride / 32u) + RUNS * h + k, p.rows * qstride, p.major);
        }
}
