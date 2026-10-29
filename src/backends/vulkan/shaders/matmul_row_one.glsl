// The one-row build's body (matmul_row.comp): a cluster's dot of weight row `wrow` with up to MAX_COLS columns, reduced across the cluster, each column's sum handed to ROW_STORE(c, s).
// The including function supplies row, nout, l, lane, wrow, type, wide, ncols and the XCOL and OCOL column maps; a kernel fusing a row product with other work includes the same text, so a row has the same bits in both.
#ifdef LLMX_MXFP4
    uint wide_range[MAX_COLS];
    for (uint c = 0u; c < MAX_COLS; ++c) wide_range[c] = 0u;
#endif
#if defined(LLMX_PRESERVE_F32) && !(defined(LLMX_Q4) || defined(LLMX_K4) || defined(LLMX_K5) || defined(LLMX_KQUANT))
    precise float acc[MAX_COLS];
#else
    float acc[MAX_COLS];
#endif
#if defined(LLMX_LOOKUP_DOT) || defined(LLMX_K23)
    uint repair[MAX_COLS];
    for (uint c = 0u; c < MAX_COLS; ++c) repair[c] = 0u;
#endif
    for (uint c = 0u; c < MAX_COLS; ++c) acc[c] = 0.0;
    if (row < nout) {
#ifdef LLMX_NVFP4
        const float factor = wfs[pi].wf[min(wrow / nout, uint(wfs[pi].wf.length()) - 1u)];
        for (uint first = 8u * l; first < p.nin; first += 8u * p.cluster) {
            uint o = (wrow * (p.nin / 64u) + first / 64u) * NVFP4_BYTES;
            uint group = (first & 63u) / 16u;
            float scale = nvfp4_scale(QBYTE(o + group));
            uint word = (o + 4u + group * 8u) / 4u;
            uvec2 packed = uvec2(w32s[pi].w32[word], w32s[pi].w32[word + 1u]);
            uint shift = (first & 8u) >> 1u;
            [[unroll]] for (uint k = 0u; k < 8u; ++k) {
                uint code = (packed[k / 4u] >> (8u * (k & 3u) + shift)) & 15u;
                precise float weight = nvfp4_value(scale, code) * factor;
                for (uint c = 0u; c < ncols; ++c)
                    acc[c] = fma(weight, x[XCOL(c) * p.nin + first + k], acc[c]);
            }
        }
#elif defined(LLMX_FLOAT_X) && defined(LLMX_Q1_0)
        const uint block = Q1_0_BLOCK, packed_type = TYPE_Q1_0;
        // Two vectors share one packed byte and scale.
        const uint vectors = 2u;
        uint n4 = p.nin / 4u;
        for (uint i = vectors * l; i < n4; i += vectors * p.cluster) {
            uint j = 4u * i, o = (wrow * (p.nin / block) + j / block) * block_bytes(packed_type);
            float d = half_at(QBYTE(o), QBYTE(o + 1u));
            [[unroll]] for (uint v = 0u; v < vectors; ++v) {
                vec4 weights = low_bits_four(o, j % block + 4u * v, d);
                for (uint c = 0u; c < ncols; ++c) {
                    // Round each product before cancellation; a contracted dot can leave a large residual.
                    precise vec4 products = weights * xt4[XCOL(c) * n4 + i + v];
                    acc[c] += (products.x + products.y) + (products.z + products.w);
                }
            }
        }
#elif defined(LLMX_FLOAT_X)
#if defined(LLMX_Q2) || defined(LLMX_Q3)
        // These 256-value rows use a whole subgroup; unrolling within a block reuses its metadata without changing any lane's FMA order.
        for (uint b = 0u; b < p.nin / 256u; ++b) {
        [[unroll]] for (uint k = 0u; k < 256u / gl_SubgroupSize; ++k) {
            uint local = l + k * gl_SubgroupSize;
            uint j = b * 256u + local;
#else
        for (uint j = l; j < p.nin; j += p.cluster) {
#endif
#ifdef LLMX_Q2
            uint o = (wrow * (p.nin / 256u) + b) * Q2_K_BYTES;
            float weight = q2_k_at(o, local);
#elif defined(LLMX_Q3)
            uint o = (wrow * (p.nin / 256u) + b) * Q3_K_BYTES;
            float weight = q3_k_at(o, local);
#elif defined(LLMX_TQ)
            uint o = (wrow * (p.nin / 256u) + j / 256u) * TQ_BYTES;
            float weight = tq_at(TQ_TYPE, o, j % 256u, tq_scale(TQ_TYPE, o));
#elif defined(LLMX_IQ3_XXS)
            uint o = (wrow * (p.nin / IQ3_XXS_BLOCK) + j / IQ3_XXS_BLOCK) * IQ3_XXS_BYTES;
            float weight = iq3_xxs_at(o, j % IQ3_XXS_BLOCK, half_at(QBYTE(o), QBYTE(o + 1u)));
#elif defined(LLMX_IQ2_SMALL)
            uint o = (wrow * (p.nin / 256u) + j / 256u) * IQ2_SMALL_BYTES;
            float weight = iq2_small_at(IQ2_SMALL_TYPE, o, j % 256u, half_at(QBYTE(o), QBYTE(o + 1u)));
#elif defined(LLMX_IQ1)
            uint o = (wrow * (p.nin / 256u) + j / 256u) * IQ1_BYTES;
            float weight = iq1_at(IQ1_TYPE, o, j % 256u, iq1_scale(IQ1_TYPE, o));
#elif defined(LLMX_IQ2_S)
            uint o = (wrow * (p.nin / IQ2_S_BLOCK) + j / IQ2_S_BLOCK) * IQ2_S_BYTES;
            float weight = iq2_s_at(o, j % IQ2_S_BLOCK, half_at(QBYTE(o), QBYTE(o + 1u)));
#elif defined(LLMX_IQ3_S)
            uint o = (wrow * (p.nin / IQ3_S_BLOCK) + j / IQ3_S_BLOCK) * IQ3_S_BYTES;
            float weight = iq3_s_at(o, j % IQ3_S_BLOCK, half_at(QBYTE(o), QBYTE(o + 1u)));
#else
            float weight = quantized_at(type, wrow, p.nin, j);
#endif
            for (uint c = 0u; c < ncols; ++c) {
                float value = x[XCOL(c) * p.nin + j];
#ifdef LLMX_MXFP4
                uint wbits = floatBitsToUint(weight) & 0x7fffffffu, xbits = floatBitsToUint(value) & 0x7fffffffu;
                uint exponent = (wbits >> 23u) + (xbits >> 23u);
                if (wbits != 0u && xbits != 0u && (exponent < 129u || exponent > 380u)) wide_range[c] = 1u;
#endif
                acc[c] = fma(weight, value, acc[c]);
            }
        }
#if defined(LLMX_Q2) || defined(LLMX_Q3)
        }
#endif
#else
#ifdef LLMX_K23
        uint nblocks = p.nin / 256u;
        uint first = l / 8u, part = l % 8u, stride = p.cluster / 8u;
        for (uint b = first; b < nblocks; b += stride) {
            uint o = (wrow * nblocks + b) * K23_BYTES, j = part * 32u;
            uvec2 codes[8];
            [[unroll]] for (uint w = 0u; w < 8u; ++w) codes[w] = weight16(k23_word(o, j + 4u * w));
#ifdef LLMX_Q2
            Q2KScale low = q2_k_scale(o, j), high = q2_k_scale(o, j + 16u);
            vec4 weights = vec4(low.scale, high.scale, low.min, high.min);
#else
            vec4 weights = vec4(q3_k_scale(o, j).scale, q3_k_scale(o, j + 16u).scale, 0.0, 0.0);
#endif
            uint block = b * 8u + part;
            for (uint c = 0u; c < ncols; ++c) {
                uint col = XCOL(c), a = col * p.nin / 4u + block * 8u;
                int lo = 0, hi = 0, ls = 0, hs = 0;
                [[unroll]] for (uint w = 0u; w < 4u; ++w) {
                    uvec2 xl = xq2[a + w], xh = xq2[a + w + 4u];
                    lo = dot_pairs(codes[w], xl, lo);
                    hi = dot_pairs(codes[w + 4u], xh, hi);
#ifdef LLMX_Q2
                    ls = dot_pairs(uvec2(0x00010001u), xl, ls);
                    hs = dot_pairs(uvec2(0x00010001u), xh, hs);
#endif
                }
                float scale = xq_block(col, block).x;
                if (k23_repair_scale(scale)) repair[c] = 1u;
                acc[c] += k23_term(weights, scale, lo, hi, ls, hs);
            }
        }
#endif
#ifdef LLMX_LOOKUP_DOT
        uint nblocks = p.nin / 32u;
        for (uint b = l; b < nblocks; b += p.cluster) {
            float dw;
            uint codes = lookup_group(wrow, p.nin, b, dw);
            uvec2 weights[8];
            for (uint j = 0u; j < 8u; ++j) weights[j] = lookup_pairs(codes, j);
            for (uint c = 0u; c < ncols; ++c) {
                uint col = XCOL(c), a2 = xa_col(col) / 2u + b * 8u;
                // IQ3_XXS is largest: 32 * 1922 * 32767 = 2015301568, within int32.
                int sum = 0;
                for (uint j = 0u; j < 8u; ++j) sum = dot_pairs(weights[j], xq2[a2 + j], sum);
                float dx = xq_block(col, b).x, scale = dw * dx;
                if (lookup_unsafe_scale(dw, dx, scale)) repair[c] = 1u;
                acc[c] = fma(scale, float(sum), acc[c]);
            }
        }
#endif
#ifdef LLMX_MXFP4
        // One lane per 32-value block, signed E2M1 codes against the 16-bit twin.
        // A block's sum is an integer below 2^24 in magnitude (32 * 12 * 32767), so the float sum and the integer dot both give it exactly.
        uint nblocks = p.nin / MXFP4_BLOCK;
        uint base = wrow * nblocks * MXFP4_BYTES;
        for (uint block = l; block < nblocks; block += p.cluster) {
            uvec4 quants;
            uint e = mxfp4_block(base + block * MXFP4_BYTES, quants);
            float dw = mxfp4_scale(e);
#ifdef LLMX_DOT
            uint pairs[16];
            mxfp4_pairs_of(quants, pairs);
#else
            int weights[32];
            for (uint j = 0u; j < 16u; ++j) {
                uint packed = (quants[j / 4u] >> ((j & 3u) * 8u)) & 255u;
                weights[j] = fp4_value(packed & 15u);
                weights[j + 16u] = fp4_value(packed >> 4u);
            }
#endif
            for (uint c = 0u; c < ncols; ++c) {
                uint col = XCOL(c), a2 = xa_col(col) / 2u + block * 8u;
#ifdef LLMX_DOT
                float sum = float(mxfp4_dot(pairs, a2));
#else
                precise float sum = 0.0;
                for (uint j = 0u; j < 32u; j += 4u) {
                    uvec2 q = xq2[a2 + j / 4u];
                    sum += float(weights[j]) * float(lo16(q.x));
                    sum += float(weights[j + 1u]) * float(lo16(q.y));
                    sum += float(weights[j + 2u]) * float(hi16(q.x));
                    sum += float(weights[j + 3u]) * float(hi16(q.y));
                }
#endif
                float scale = dw * xq_block(col, block).x;
                uint exponent = (floatBitsToUint(scale) >> 23u) & 255u;
                // At 253 the largest decoded weight overflows f32; a combined scale must be normal.
                if (e >= MXFP4_FIRST_OVERFLOW_EXPONENT || exponent == 0u || exponent == 255u) wide_range[c] = 1u;
                else acc[c] = fma(scale, sum, acc[c]);
            }
        }
#endif
#ifdef LLMX_Q8W
        {
            uint nblocks = p.nin / Q8_0_BLOCK;
            uint npairs = nblocks / 2u;
            uint wbase = wrow * nblocks * 17u / 2u;   // words: a pair is 17
            uint group = l / 4u, t = l - group * 4u;   // four lanes per pair
            uint groups = p.cluster / 4u;
            for (uint j = group; j < npairs; j += groups) {
                uint o = wbase + j * 17u;
                uint wa = w32s[pi].w32[o + t];
                uint wb = w32s[pi].w32[o + t + 4u];
                uint wc = w32s[pi].w32[o + t + 8u];
                uint wd = w32s[pi].w32[o + t + 12u];
                uint we = (t == 0u) ? w32s[pi].w32[o + 16u] : 0u;
                float s0 = unpackHalf2x16(w32s[pi].w32[o]).x;
                float s1 = unpackHalf2x16(w32s[pi].w32[o + 8u]).y;
                // Block 0's words shifted by a half word with the next word, held by the next lane of the pair; word 8's remaining half is the second scale, so it drops out.
                uint src = (lane & ~3u) | ((t + 1u) & 3u);
                uint sa = subgroupShuffle(wa, src), sb = subgroupShuffle(wb, src), sc = subgroupShuffle(wc, src);
                wa = (wa >> 16u) | ((t == 3u ? sb : sa) << 16u);
                wb = (wb >> 16u) | ((t == 3u ? sc : sb) << 16u);
                if (t == 0u) wc = 0u;
                for (uint c = 0u; c < ncols; ++c) {
                    uint col = XCOL(c);
                    uint a2 = xa_col(col) / 2u + j * 16u;
                    int p0 = 0, p1 = 0;
                    q8_word(wa, t, a2, p0, p1);
                    q8_word(wb, t + 4u, a2, p0, p1);
                    q8_word(wc, t + 8u, a2, p0, p1);
                    q8_word(wd, t + 12u, a2, p0, p1);
                    if (t == 0u) q8_word(we, 16u, a2, p0, p1);
                    vec4 dd = xq_pair(col, j);
#ifdef LLMX_PRESERVE_F32
                    acc[c] += fma(s0, dd.x * float(p0), s1 * (dd.z * float(p1)));
#else
                    acc[c] += s0 * (dd.x * float(p0)) + s1 * (dd.z * float(p1));
#endif
                }
            }
        }
#endif
#if !defined(LLMX_K23) && !defined(LLMX_LOOKUP_DOT) && !defined(LLMX_MXFP4) && !defined(LLMX_Q8W) && !defined(LLMX_Q4) && !defined(LLMX_KQUANT) && !(defined(LLMX_K4) || defined(LLMX_K5))
#if !defined(LLMX_WEIGHT_F16) && !defined(LLMX_WEIGHT_BF16)
        if (type == TYPE_Q8_0) {
            uint nblocks = p.nin / Q8_0_BLOCK;
            uint bbase = wrow * nblocks * Q8_0_BYTES;   // bytes
            for (uint b = l; b < nblocks; b += p.cluster) {
                uint o = bbase + b * Q8_0_BYTES;
                float scale = unpackHalf2x16(uint(w16s[pi].w16[o >> 1u])).x;
                uint q0 = hword_at(o + 2u), q1 = hword_at(o + 6u), q2 = hword_at(o + 10u), q3 = hword_at(o + 14u);
                uint q4 = hword_at(o + 18u), q5 = hword_at(o + 22u), q6 = hword_at(o + 26u), q7 = hword_at(o + 30u);
                for (uint c = 0u; c < ncols; ++c) {
                    uint col = XCOL(c);
                    uint a2 = xa_col(col) / 2u + b * 8u;
                    int s = byte_dot(q0, xq2[a2]) + byte_dot(q1, xq2[a2 + 1u]) + byte_dot(q2, xq2[a2 + 2u]) + byte_dot(q3, xq2[a2 + 3u]) +
                            byte_dot(q4, xq2[a2 + 4u]) + byte_dot(q5, xq2[a2 + 5u]) + byte_dot(q6, xq2[a2 + 6u]) + byte_dot(q7, xq2[a2 + 7u]);
#ifdef LLMX_PRESERVE_F32
                    acc[c] = fma(scale, xq_block(col, b).x * float(s), acc[c]);
#else
                    acc[c] += scale * (xq_block(col, b).x * float(s));
#endif
                }
            }
        } else
#endif
        if (p.nin % 4u == 0u) {
            // Float rows read four weights at a time and bind the original X through the twin's vec4 view.
            uint n4 = p.nin / 4u, wbase4 = wrow * n4;
            uint i = l, cl = p.cluster;
            // Four steps' weights loaded before any is used, so a small call's few waves wait on one latency rather than four; the sums keep their order.
            for (; i + 3u * cl < n4; i += 4u * cl) {
                vec4 w0 = WEIGHT4(wbase4 + i);
                vec4 w1 = WEIGHT4(wbase4 + i + cl);
                vec4 w2 = WEIGHT4(wbase4 + i + 2u * cl);
                vec4 w3 = WEIGHT4(wbase4 + i + 3u * cl);
                for (uint c = 0u; c < ncols; ++c) {
                    uint xb = XCOL(c) * n4 + i;
                    acc[c] += dot(w0, xt4[xb]);
                    acc[c] += dot(w1, xt4[xb + cl]);
                    acc[c] += dot(w2, xt4[xb + 2u * cl]);
                    acc[c] += dot(w3, xt4[xb + 3u * cl]);
                }
            }
            for (; i < n4; i += cl) {
                vec4 w = WEIGHT4(wbase4 + i);
                for (uint c = 0u; c < ncols; ++c) acc[c] += dot(w, xt4[XCOL(c) * n4 + i]);
            }
        } else {
            uint wbase = wrow * p.nin;
            for (uint i = l; i < p.nin; i += p.cluster) {
                float w = WEIGHT1(wbase + i);
                for (uint c = 0u; c < ncols; ++c) acc[c] += w * x[XCOL(c) * p.nin + i];
            }
        }
#endif
#ifdef LLMX_Q4
        if (type == TYPE_Q4_0 && wide != 0u) {
            // Two lanes per 9-word pair, a block each: lane 0 words 0..4 (the scale, then nibble bytes starting two bytes in), lane 1 words 4..8 (the scale in word 4's high half, then aligned nibble words).
            uint nblocks = p.nin / Q4_0_BLOCK;
            uint npairs = nblocks / 2u;
            uint wbase = wrow * nblocks * 9u / 2u;   // words: a pair is 9
            uint group = l / 2u, t = l - group * 2u;
            uint groups = p.cluster / 2u;
            for (uint j = group; j < npairs; j += groups) {
                uint o = wbase + j * 9u + 4u * t;
                Q40W_WEIGHTS(o)
                uint b = 2u * j + t;
                for (uint c = 0u; c < ncols; ++c) {
                    uint col = XCOL(c);
                    uint a4 = xa_col(col) / 4u + b * 4u;
                    XQ4T lo0 = xq4[a4], lo1 = xq4[a4 + 1u], hi0 = xq4[a4 + 2u], hi1 = xq4[a4 + 3u];
                    QDOT pl = QDOT(0);
                    QDOT ph = QDOT(0);
                    Q4_DOTS(n0, n1, n2, n3, lo0, lo1, hi0, hi1)
                    vec2 xb = xq_block(col, b);
                    acc[c] += Q40_TERM(d, xb);
                }
            }
        } else if (type == TYPE_Q4_0) {
            // 16-bit loads over 18-byte blocks for an odd block count.
            uint nblocks = p.nin / Q4_0_BLOCK;
            uint bbase = wrow * nblocks * Q4_0_BYTES;
            for (uint b = l; b < nblocks; b += p.cluster) {
                uint o = bbase + b * Q4_0_BYTES;
                Q40_WEIGHTS(o)
                for (uint c = 0u; c < ncols; ++c) {
                    uint col = XCOL(c);
                    uint a4 = xa_col(col) / 4u + b * 4u;
                    XQ4T lo0 = xq4[a4], lo1 = xq4[a4 + 1u], hi0 = xq4[a4 + 2u], hi1 = xq4[a4 + 3u];
                    QDOT pl = QDOT(0);
                    QDOT ph = QDOT(0);
                    Q4_DOTS(n0, n1, n2, n3, lo0, lo1, hi0, hi1)
                    vec2 xb = xq_block(col, b);
                    acc[c] += Q40_TERM(d, xb);
                }
            }
        } else {
            // One lane per affine block, d and m in word 0.
            uint nblocks = p.nin / 32u;
#ifdef LLMX_Q50
            const uint words = Q5_0_BYTES;
#else
            uint words = AFFINE_Q5 ? 6u : 5u;
#endif
            uint wbase = wrow * nblocks * words;
            for (uint b = l; b < nblocks; b += p.cluster) {
                uint o = wbase + b * words;
                AFFINE_WEIGHTS(o)
                for (uint c = 0u; c < ncols; ++c) {
                    uint col = XCOL(c);
                    uint a4 = xa_col(col) / 4u + b * 4u;
                    XQ4T lo0 = xq4[a4], lo1 = xq4[a4 + 1u], hi0 = xq4[a4 + 2u], hi1 = xq4[a4 + 3u];
                    QDOT pl = QDOT(0);
                    QDOT ph = QDOT(0);
                    AFFINE_DOTS(w1, w2, w3, w4, hb, lo0, lo1, hi0, hi1)
                    vec2 xb = xq_block(col, b);
                    Q41_ADD(acc[c], dm, xb);
                }
            }
        }
#endif
#if defined(LLMX_K4) || defined(LLMX_K5)
        {
            // Q4_K and Q5_K: eight lanes a block, lane t the 16 nibble bytes from h * 16 of 64-value chunk c = t / 2, h = t % 2: low nibbles are group 2c at c * 64 + h * 16 + i, high nibbles group 2c + 1 at c * 64 + 32 + h * 16 + i.
            // Q5_K's fifth bits are bits 2c and 2c + 1 of each byte of the matching high-bit word.
#ifdef LLMX_K5
            const uint q5 = 1u;
#else
            const uint q5 = 0u;
#endif
            uint bw = (q5 != 0u ? Q5_K_BYTES : Q4_K_BYTES) / 4u;
            uint nblocks = p.nin / Q6_K_BLOCK;
            uint wbase = wrow * nblocks * bw;
            uint group = l / 8u, t = l - group * 8u;
            uint groups = p.cluster / 8u;
            uint c = t / 2u, h = t - c * 2u;
            uint sl = 2u * c, sh = 2u * c + 1u;
            for (uint b = group; b < nblocks; b += groups) {
                uint o = wbase + b * bw;
                K45_WEIGHTS(o)
                uint bl = b * 8u + 2u * c;   // the low group's 32-block, the high group's is next
                for (uint cc = 0u; cc < ncols; ++cc) {
                    uint col = XCOL(cc);
                    uint a4 = xa_col(col) / 4u + (b * 256u + c * 64u + h * 16u) / 8u;
                    QDOT pl = QDOT(0);
                    QDOT ph = QDOT(0);
                    {
                        XQ4T lo = xq4[a4], hi = xq4[a4 + 4u];
                        nibble_dots(nq.x, fifth(hbv.x, sl), fifth(hbv.x, sl + 8u), fifth(hbv.x, sh), fifth(hbv.x, sh + 8u), XA(lo), XA(hi), pl, ph);
                        nibble_dots(nq.y, fifth(hbv.y, sl), fifth(hbv.y, sl + 8u), fifth(hbv.y, sh), fifth(hbv.y, sh + 8u), XB(lo), XB(hi), pl, ph);
                    }
                    {
                        XQ4T lo = xq4[a4 + 1u], hi = xq4[a4 + 5u];
                        nibble_dots(nq.z, fifth(hbv.z, sl), fifth(hbv.z, sl + 8u), fifth(hbv.z, sh), fifth(hbv.z, sh + 8u), XA(lo), XA(hi), pl, ph);
                        nibble_dots(nq.w, fifth(hbv.w, sl), fifth(hbv.w, sl + 8u), fifth(hbv.w, sh), fifth(hbv.w, sh + 8u), XB(lo), XB(hi), pl, ph);
                    }
                    vec4 xt = xq_pair(col, bl / 2u);   // d, ds of the low group, then the high
                    acc[cc] += K45_TERM(dl, dh, ml, mh, xt);
                }
            }
        }
#endif
#ifdef LLMX_KQUANT
        {
            // Q6_K: eight lanes per block, lane t taking positions l0..l0+7 of half n, each with its own sub-scale, index l0 / 16 for all eight.
            // Position l0 + i has one value per 32-group: the low nibble of ql[l0 + i] with qh bits 0..1, of ql[l0 + 32 + i] with bits 2..3, the high nibbles with bits 4..5 and 6..7, as unsigned bytes 0..63 centered by pair_dot.
            uint nblocks = p.nin / Q6_K_BLOCK;
            uint bbase = wrow * nblocks * Q6_K_BYTES;   // bytes
            uint group = l / 8u, t = l - group * 8u;
            uint groups = p.cluster / 8u;
            uint n = t / 4u, l0 = (t - n * 4u) * 8u;
            for (uint b = group; b < nblocks; b += groups) {
                uint o = bbase + b * Q6_K_BYTES;
                Q6K_WEIGHTS(o)
                uint bg = b * 8u + n * 4u;   // the 32-block of group 0
                for (uint c = 0u; c < ncols; ++c) {
                    uint col = XCOL(c);
                    uint a4 = xa_col(col) / 4u + (b * Q6_K_BLOCK + n * 128u + l0) / 8u;
                    XQ4T x0v = xq4[a4], x1v = xq4[a4 + 4u], x2v = xq4[a4 + 8u], x3v = xq4[a4 + 12u];
#ifdef LLMX_I8
                    int p0 = q6_dot(q0a, q0b, x0v), p1 = q6_dot(q1a, q1b, x1v), p2 = q6_dot(q2a, q2b, x2v), p3 = q6_dot(q3a, q3b, x3v);
#else
                    QDOT p0 = pair_dot(q0a & 0x00FF00FFu, (q0a >> 8u) & 0x00FF00FFu, x0v.xy) + pair_dot(q0b & 0x00FF00FFu, (q0b >> 8u) & 0x00FF00FFu, x0v.zw);
                    QDOT p1 = pair_dot(q1a & 0x00FF00FFu, (q1a >> 8u) & 0x00FF00FFu, x1v.xy) + pair_dot(q1b & 0x00FF00FFu, (q1b >> 8u) & 0x00FF00FFu, x1v.zw);
                    QDOT p2 = pair_dot(q2a & 0x00FF00FFu, (q2a >> 8u) & 0x00FF00FFu, x2v.xy) + pair_dot(q2b & 0x00FF00FFu, (q2b >> 8u) & 0x00FF00FFu, x2v.zw);
                    QDOT p3 = pair_dot(q3a & 0x00FF00FFu, (q3a >> 8u) & 0x00FF00FFu, x3v.xy) + pair_dot(q3b & 0x00FF00FFu, (q3b >> 8u) & 0x00FF00FFu, x3v.zw);
#endif
                    vec4 xa = xq_pair(col, bg / 2u), xb = xq_pair(col, bg / 2u + 1u);   // d, ds of groups 0..3
                    acc[c] += Q6K_TERM(d, s0, s1, s2, s3, xa, xb);
                }
            }
        }
#endif
#endif
    }
    // Reduce within each cluster; every lane joins every shuffle and lanes past the last row add zero.
    for (uint c = 0u; c < ncols; ++c) {
        float s = acc[c];
#ifdef LLMX_MXFP4
        // The same butterfly with constant offsets for a 64-lane cluster, which every MXFP4 row from 2048 values takes.
        if (p.cluster == 64u) {
            s += subgroupShuffleXor(s, 32u);
            s += subgroupShuffleXor(s, 16u);
            s += subgroupShuffleXor(s, 8u);
            s += subgroupShuffleXor(s, 4u);
            s += subgroupShuffleXor(s, 2u);
            s += subgroupShuffleXor(s, 1u);
        } else {
            for (uint off = p.cluster / 2u; off > 0u; off >>= 1u) s += subgroupShuffleXor(s, off);
        }
        // Clusters reduce their range flags only where some lane of the subgroup holds one; otherwise every cluster's is zero.
        uint wide = wide_range[c];
        if (subgroupOr(wide) != 0u)
            for (uint off = p.cluster / 2u; off > 0u; off >>= 1u) wide |= subgroupShuffleXor(wide, off);
        if (l == 0u && row < nout && (wide != 0u || isnan(s) || isinf(s))) s = mxfp4_wide(wrow, XCOL(c));
#elif defined(LLMX_FLOAT_X) && (defined(LLMX_Q2) || defined(LLMX_Q3))
        [[unroll]] for (uint off = gl_SubgroupSize / 2u; off > 0u; off >>= 1u) s += subgroupShuffleXor(s, off);
#else
        for (uint off = p.cluster / 2u; off > 0u; off >>= 1u) s += subgroupShuffleXor(s, off);
#endif
#ifdef LLMX_LOOKUP_DOT
        uint unsafe = repair[c];
        for (uint off = p.cluster / 2u; off > 0u; off >>= 1u) unsafe |= subgroupShuffleXor(unsafe, off);
        if (l == 0u && row < nout && (unsafe != 0u || isnan(s) || isinf(s))) s = lookup_float_twin(wrow, XCOL(c), p.nin, 0u, p.nin / 32u);
#endif
#ifdef LLMX_K23
        uint unsafe = repair[c];
        for (uint off = p.cluster / 2u; off > 0u; off >>= 1u) unsafe |= subgroupShuffleXor(unsafe, off);
        if (l == 0u && row < nout && (unsafe != 0u || isnan(s) || isinf(s))) s = k23_float_twin(wrow, XCOL(c), p.nin, 0u, p.nin / 32u);
#endif
        ROW_STORE(c, s)
    }
