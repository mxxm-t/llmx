// The row kernels' declarations and helpers (matmul_row.comp), which a kernel fusing a row product with other work shares: it may widen the projection arrays, qualify the outputs, add push constants and move the activation loads.
#ifndef LLMX_ROW_PROJ
#define LLMX_ROW_PROJ 3
#endif
#ifndef LLMX_ROW_Y
#define LLMX_ROW_Y
#endif
#ifndef LLMX_ROW_PUSH
#define LLMX_ROW_PUSH
#endif
#if (defined(LLMX_Q2) || defined(LLMX_Q3)) && !defined(LLMX_FLOAT_X)
#define LLMX_K23
#endif
// Y[b*nout + o] = dot(row o, X column b), the decode kernel, for up to three projections of one X per dispatch; a subgroup takes `rows_per_sg` clusters of `cluster` lanes, each a row, reducing with xor shuffles.
// Quantized rows meet the activations as a signed 16-bit twin (xquant.glsl), or the 8-bit twin with LLMX_I8 (docs/PRECISION.md); block layouts and the module per type family are in docs/src/backends-vulkan.md, Shader notes.
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_shuffle : require
#ifdef LLMX_MXFP4
#extension GL_KHR_shader_subgroup_arithmetic : require
#endif
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_control_flow_attributes : require
#ifdef LLMX_DOT
#extension GL_EXT_integer_dot_product : require
#endif
#if defined(LLMX_I8) && !(defined(LLMX_DOT) && (defined(LLMX_Q4) || defined(LLMX_K4) || defined(LLMX_K5) || defined(LLMX_KQUANT)))
#error "the 8-bit twin is read by the Q4 and K-quant families through the integer dot"
#endif
#if defined(LLMX_DOT) && !(defined(LLMX_Q4) || defined(LLMX_K4) || defined(LLMX_K5) || defined(LLMX_KQUANT) || defined(LLMX_MXFP4) || defined(LLMX_IQ4) || defined(LLMX_IQ3_S) || defined(LLMX_IQ2_S) || defined(LLMX_IQ3_XXS) || defined(LLMX_IQ2_SMALL) || defined(LLMX_IQ1) || defined(LLMX_K23) || defined(LLMX_TQ))
#error "Q8_0 and F32 rows take matmul_vec_q8.comp and the plain build on integer-dot devices"
#endif
#if defined(LLMX_TQ) && defined(LLMX_DOT)
#define LLMX_LOOKUP_DOT
#endif
#if defined(LLMX_IQ4) || ((defined(LLMX_IQ3_S) || defined(LLMX_IQ2_S) || defined(LLMX_IQ3_XXS) || defined(LLMX_IQ2_SMALL) || defined(LLMX_IQ1)) && defined(LLMX_DOT))
#define LLMX_LOOKUP_DOT
#endif
#include "q.glsl"
#include "iq2_layout.glsl"
#include "tq_layout.glsl"
#ifdef LLMX_IQ1
#include "iq1_layout.glsl"
#endif
// The fifth-bit plane is a compile-time weight loader choice. Existing Q4
// products must not carry its branches or live high-bit registers.
#ifdef LLMX_Q5
const bool AFFINE_Q5 = true;
#else
const bool AFFINE_Q5 = false;
#endif
#ifdef LLMX_MXFP4
#extension GL_ARB_gpu_shader_fp64 : require
#endif
#include "float_controls.glsl"
layout(local_size_x = 256) in;
// Batch columns a lane keeps in registers, specialization constant 0; the backend takes the one-column build for a one-column chunk, which frees the registers seven unused accumulators would hold.
layout(constant_id = 0) const uint MAX_COLS = 8;
layout(constant_id = 15) const uint WEIGHT_TYPE = 0xffffffffu;
layout(push_constant) uniform P {
    uint nin; uint nbatch; uint col0; uint ncols; uint cluster; uint rows_per_sg; uint nproj;
    uint nout0; uint type0; uint wide0; uint start0;
    uint nout1; uint type1; uint wide1; uint start1;
    uint nout2; uint type2; uint wide2; uint start2;
    uint accum;   // Y += rather than Y =, the residual add folded in
    uint per;
    uint order0;   // where a grouped call's entries start in tab
    LLMX_ROW_PUSH
} p;
layout(set = 0, binding = 0) LLMX_ROW_Y buffer Y { float y[]; } ys[LLMX_ROW_PROJ];
layout(set = 0, binding = 1) readonly buffer W16 { uint16_t w16[]; } w16s[LLMX_ROW_PROJ];
layout(set = 0, binding = 2) readonly buffer W32 { uint w32[]; } w32s[LLMX_ROW_PROJ];
layout(set = 0, binding = 3) readonly buffer WF { float wf[]; } wfs[LLMX_ROW_PROJ];
layout(set = 0, binding = 4) readonly buffer X { float x[]; };
// The weights as 16-byte vectors, for the K-quant blocks whose sizes are multiples of 16.
layout(set = 0, binding = 5) readonly buffer W128 { uvec4 w128[]; } w128s[LLMX_ROW_PROJ];
// The quantized activations (xquant.glsl), one buffer bound four ways: four values, eight, a table entry or a pair of entries per load, four being one xq2 element and eight one xq4 element in both twins.
// The 16-bit twin: per column nin / 2 words of pairs, then per block the scale and scaled sum; the 8-bit twin: per column nin / 4 words of four bytes in position order, then the same per block.
#ifdef LLMX_I8
layout(set = 0, binding = 6) readonly buffer XQ2 { uint xq2[]; };
layout(set = 0, binding = 7) readonly buffer XQ4 { uvec2 xq4[]; };
#define XQ4T uvec2
#define XWT uint
#define XA(v) (v).x
#define XB(v) (v).y
#else
layout(set = 0, binding = 6) readonly buffer XQ2 { uvec2 xq2[]; };
layout(set = 0, binding = 7) readonly buffer XQ4 { uvec4 xq4[]; };
#define XQ4T uvec4
#define XWT uvec2
#define XA(v) (v).xy
#define XB(v) (v).zw
#endif
layout(set = 0, binding = 8) readonly buffer XT2 { vec2 xt2[]; };
layout(set = 0, binding = 9) readonly buffer XT4 { vec4 xt4[]; };
// A routed call (per != 0, backend.hpp matmul_experts) runs one entry per workgroup row y: entry e reads X column e / per through the rows of expert ids[e] and writes output row e.
layout(set = 0, binding = 10) readonly buffer IDS { uint ids[]; };
// A grouped routed call, the wide build's third pipeline (GROUPED, specialization constant 8), runs one chunk of the grouping (moe_group.comp) per workgroup row y: up to MAX_COLS entries of one expert.
// As a specialization it costs the plain and routed pipelines nothing.
layout(constant_id = 8) const bool GROUPED = false;
layout(set = 0, binding = 11) readonly buffer TAB { uint tab[]; };
// Rows a cluster takes, specialization constant 9: the Q4 and K-quant families' two-row decode builds (kRowBuilds) give a lane two rows' weights, so each column's activation loads serve both.
// A row's lanes, blocks, products and reduction are the one-row builds', and each block's weights and float terms come from the same macros below, so a column gets the one-column build's bits (docs/VULKAN.md, batch invariance).
layout(constant_id = 9) const uint ROWS = 1;

uint pi = 0u;
#if defined(LLMX_WEIGHT_F16) || defined(LLMX_WEIGHT_BF16)
#include "f16.glsl"
#ifdef LLMX_WEIGHT_F16
#define WEIGHT_FLOAT(h) f16_to_f32(h)
#else
#define WEIGHT_FLOAT(h) bf16_to_f32(h)
#endif
vec4 weight4(uint i) {
    uint lo = w32s[pi].w32[2u * i], hi = w32s[pi].w32[2u * i + 1u];
    return vec4(WEIGHT_FLOAT(lo & 65535u), WEIGHT_FLOAT(lo >> 16u), WEIGHT_FLOAT(hi & 65535u), WEIGHT_FLOAT(hi >> 16u));
}
#define WEIGHT4(i) weight4(i)
#define WEIGHT1(i) WEIGHT_FLOAT(uint(w16s[pi].w16[i]))
#else
#define WEIGHT4(i) uintBitsToFloat(w128s[pi].w128[i])
#define WEIGHT1(i) wfs[pi].wf[i]
#endif
#if defined(LLMX_FLOAT_X) || defined(LLMX_MXFP4) || defined(LLMX_LOOKUP_DOT) || defined(LLMX_Q50) || defined(LLMX_K23)
#define QBYTE(i) ((w32s[pi].w32[(i) >> 2u] >> (8u * ((i) & 3u))) & 255u)
#ifdef LLMX_TQ
#define TQ_WORD(i) w32s[pi].w32[i]
#endif
#ifdef LLMX_IQ3_S
// Float rows use the spare twin view; int16 rows use the spare F32 weight view.
#ifdef LLMX_DOT
#define LOOKUP_GRID_SOURCE(i) floatBitsToUint(wfs[0].wf[i])
#else
#define LOOKUP_GRID_SOURCE(i) xq2[(i) >> 1u][(i) & 1u]
#endif
#define LOOKUP_GRID_SIZE 512u
#include "lookup_cache.glsl"
#define IQ3_GRID_WORD(i) lookup_words[i]
#include "iq3_s.glsl"
#elif defined(LLMX_IQ3_XXS)
#ifdef LLMX_DOT
#define IQ3_XXS_GRID_WORD(i) floatBitsToUint(wfs[0].wf[i])
#else
#define LOOKUP_GRID_SOURCE(i) xq2[(i) >> 1u][(i) & 1u]
#define LOOKUP_GRID_SIZE 256u
#include "lookup_cache.glsl"
#define IQ3_XXS_GRID_WORD(i) lookup_words[i]
#endif
#include "iq3_xxs.glsl"
#elif defined(LLMX_IQ2_SMALL)
#define IQ2_SMALL_HALF(i) uint(w16s[pi].w16[(i) / 2u])
#ifdef LLMX_DOT
#define IQ2_GRID_WORD(i) floatBitsToUint(wfs[0].wf[i])
#else
#define LOOKUP_GRID_SOURCE(i) xq2[(i) >> 1u][(i) & 1u]
#define LOOKUP_GRID_SIZE IQ2_SMALL_GRID_WORDS
#include "lookup_cache.glsl"
#define IQ2_GRID_WORD(i) lookup_words[i]
#endif
#include "iq2_small.glsl"
#elif defined(LLMX_IQ1)
#ifdef LLMX_DOT
#define LOOKUP_GRID_SOURCE(i) floatBitsToUint(wfs[0].wf[i])
#else
#define LOOKUP_GRID_SOURCE(i) xq2[(i) >> 1u][(i) & 1u]
#endif
#define LOOKUP_GRID_SIZE 1024u
#include "lookup_cache.glsl"
#define IQ1_GRID_WORD(i) lookup_words[i]
#include "iq1.glsl"
#elif defined(LLMX_IQ2_S)
#ifdef LLMX_DOT
#define IQ2_GRID_WORD(i) floatBitsToUint(wfs[0].wf[i])
#else
#define LOOKUP_GRID_SOURCE(i) xq2[(i) >> 1u][(i) & 1u]
#define LOOKUP_GRID_SIZE 2048u
#include "lookup_cache.glsl"
#define IQ2_GRID_WORD(i) lookup_words[i]
#endif
#include "iq2_s.glsl"
#endif
#include "qdecode.glsl"
#if defined(LLMX_MXFP4) && defined(LLMX_DOT)
shared uint mxfp4_pairs[256];
#endif
#endif

// Offset in pair words into the quantized activations of column c.
uint xa_col(uint c) { return c * p.nin / 2u; }
// A block's scale d and d times its integer sum.
#ifdef LLMX_I8
#define XTAB2 (p.nbatch * p.nin / 8u)
#else
#define XTAB2 (p.nbatch * p.nin / 4u)
#endif
vec2 xq_block(uint c, uint b) { return xt2[XTAB2 + c * (p.nin / 32u) + b]; }
// The scale entries of blocks 2j and 2j + 1 in one load.
vec4 xq_pair(uint c, uint j) { return xt4[XTAB2 / 2u + c * (p.nin / 64u) + j]; }
#ifdef LLMX_LOOKUP_DOT
#include "dot16.glsl"
float lookup_input(uint col, uint b, uint j) {
    uvec2 pair = xq2[col * p.nin / 4u + b * 8u + j / 4u];
    uint word = (j & 1u) == 0u ? pair.x : pair.y;
    int q = int(int16_t(word >> ((j & 2u) * 8u)));
    return float(q) * xq_block(col, b).x;
}
#ifdef LLMX_TQ
#include "tq_dot.glsl"
#elif defined(LLMX_IQ3_XXS)
#include "iq3_xxs_dot.glsl"
#elif defined(LLMX_IQ1)
#include "iq1_dot.glsl"
#elif defined(LLMX_IQ2_S) || defined(LLMX_IQ2_SMALL)
#include "iq2_dot.glsl"
#elif defined(LLMX_IQ3_S)
#include "iq3_dot.glsl"
#else
#include "iq4_dot.glsl"
#endif
#endif
#ifdef LLMX_K23
#define K23_WORD(i) w32s[pi].w32[i]
#include "dot16.glsl"
float k23_input(uint col, uint b, uint j) {
    uvec2 pair = xq2[col * p.nin / 4u + b * 8u + j / 4u];
    uint word = (j & 1u) == 0u ? pair.x : pair.y;
    int q = int(int16_t(word >> ((j & 2u) * 8u)));
    return float(q) * xq_block(col, b).x;
}
#include "k23_dot.glsl"
#endif

// The signed halves of a word.
int lo16(uint w) { return int(int16_t(w)); }
int hi16(uint w) { return int(int16_t(w >> 16u)); }
// The dots in two forms, picked per device (backends/device_profile.hpp): LLMX_DOT through the integer dot product extension, otherwise plain multiplies.
#ifdef LLMX_I8
#define QDOT int
// Four signed bytes against four 8-bit activations, the four-wide dot.
int dot4(uint a, uint b) { return dotPacked4x8EXT(int(a), int(b)); }
#elif defined(LLMX_DOT)
#define QDOT int
// Two pairs of 16-bit values against two activation pair words.
int pair_dot(uint a, uint b, uvec2 xw) {
#if defined(LLMX_KQUANT) || defined(LLMX_Q50)
#ifdef LLMX_Q50
    const int16_t center = int16_t(16);
#else
    const int16_t center = int16_t(32);
#endif
    // A 32-bit offset would widen the dot operands and lose the native instruction.
    return dotEXT(unpack16(int(a)) - i16vec2(center), unpack16(int(xw.x))) + dotEXT(unpack16(int(b)) - i16vec2(center), unpack16(int(xw.y)));
#else
    return dotEXT(unpack16(int(a)), unpack16(int(xw.x))) + dotEXT(unpack16(int(b)), unpack16(int(xw.y)));
#endif
}
#else
#define QDOT float
// One signed byte of a word.
int byte_at(uint w, uint s) { return int(int8_t(w >> s)); }
// The nibble and K-quant dots multiply as floats, because the chip's 32-bit integer multiply is quarter rate.
// Callers pass at most six-bit quants against 16-bit activations, Q6_K centered before the dot, so the sums stay below 2^24 and are exact (the bound per type is in docs/VULKAN.md).
float pair_dot(uint a, uint b, uvec2 xw) {
#ifdef LLMX_Q50
    const float offset = 16.0;
#elif defined(LLMX_KQUANT)
    const float offset = 32.0;
#else
    const float offset = 0.0;
#endif
    return (float(a & 0xFFFFu) - offset) * float(lo16(xw.x)) + (float(a >> 16u) - offset) * float(hi16(xw.x))
         + (float(b & 0xFFFFu) - offset) * float(lo16(xw.y)) + (float(b >> 16u) - offset) * float(hi16(xw.y));
}
// Q8_0's weights are signed and its sums can exceed 2^24, so it keeps the integer multiply.
int byte_dot(uint w, uvec2 xw) {
    return byte_at(w, 0u) * lo16(xw.x) + byte_at(w, 16u) * hi16(xw.x) + byte_at(w, 8u) * lo16(xw.y) + byte_at(w, 24u) * hi16(xw.y);
}
#endif
// A word at any even byte offset, two 16-bit loads.
uint hword_at(uint byte_off) {
    uint h = byte_off >> 1u;
    return uint(w16s[pi].w16[h]) | (uint(w16s[pi].w16[h + 1u]) << 16u);
}

#ifdef LLMX_Q8W
// Word d of a Q8_0 pair shifted so block 0's words 0..7 hold values 4d..4d+3 and block 1's words 9..16 hold 32+4(d-9)..; word 8 is zero. `a2` is the pair's first pair-word couple.
void q8_word(uint w, uint d, uint a2, inout int p0, inout int p1) {
    int r = byte_dot(w, xq2[d < 8u ? a2 + d : a2 + 8u + d - 9u]);
    if (d < 8u) p0 += r; else p1 += r;
}
#endif

#if defined(LLMX_Q4) || defined(LLMX_K4) || defined(LLMX_K5)
// The eight nibbles of a word: byte i's low nibble is position 4k + i of a group, its high nibble the same position 16 (Q4_0, Q4_1) or 32 (Q4_K, Q5_K) on.
// Low and high go to separate dots since the K-quants scale them apart; `hb` carries Q5_K's fifth bit in bit 4 of each pair word, zero otherwise.
void nibble_dots(uint w, uint hl0, uint hl1, uint hh0, uint hh1, XWT xl, XWT xh, inout QDOT pl, inout QDOT ph) {
#ifdef LLMX_I8
#ifdef LLMX_Q50
    pl += dot4(q5_signed_bytes((w & 0x0F0F0F0Fu) | hl0 | (hl1 << 8u)), xl);
    ph += dot4(q5_signed_bytes(((w >> 4u) & 0x0F0F0F0Fu) | hh0 | (hh1 << 8u)), xh);
#else
    pl += dot4((w & 0x0F0F0F0Fu) | hl0 | (hl1 << 8u), xl);
    ph += dot4(((w >> 4u) & 0x0F0F0F0Fu) | hh0 | (hh1 << 8u), xh);
#endif
#else
    pl += pair_dot((w & 0x000F000Fu) | hl0, ((w >> 8u) & 0x000F000Fu) | hl1, xl);
    ph += pair_dot(((w >> 4u) & 0x000F000Fu) | hh0, ((w >> 12u) & 0x000F000Fu) | hh1, xh);
#endif
}
// Q5_K's fifth bits for one nibble word: bit s of each byte of `hb`, placed in bit 4 of each 16-bit lane.
uint fifth(uint hb, uint s) { return ((hb >> s) & 0x00010001u) << 4u; }
#endif


#if defined(LLMX_KQUANT) && defined(LLMX_I8)
// Four 6-bit values, a byte each, as the signed bytes value - 32: flipping bit 5 gives value - 32 or value + 32, and a set bit 5 then means negative, so it is copied into bits 6 and 7.
uint signed6(uint q) {
    uint t = q ^ 0x20202020u, neg = t & 0x20202020u;
    return t | (neg << 1u) | (neg << 2u);
}
// A Q6_K group's dot: two words of four values against the column's eight activations.
int q6_dot(uint a, uint b, XQ4T x) { return dot4(signed6(a), x.x) + dot4(signed6(b), x.y); }
#endif

#if defined(LLMX_KQUANT) || defined(LLMX_K4) || defined(LLMX_K5)
// A word at any even byte offset: aligned, or the high half of one word and the low half of the next.
uint word_at(uint byte_off) {
    uint i = byte_off >> 2u;
    if ((byte_off & 3u) == 0u) return w32s[pi].w32[i];
    return (w32s[pi].w32[i] >> 16u) | (w32s[pi].w32[i + 1u] << 16u);
}
// Two consecutive words at any even byte offset, three loads when unaligned.
void words_at(uint byte_off, out uint a, out uint b) {
    uint i = byte_off >> 2u;
    if ((byte_off & 3u) == 0u) { a = w32s[pi].w32[i]; b = w32s[pi].w32[i + 1u]; return; }
    uint w0 = w32s[pi].w32[i], w1 = w32s[pi].w32[i + 1u], w2 = w32s[pi].w32[i + 2u];
    a = (w0 >> 16u) | (w1 << 16u);
    b = (w1 >> 16u) | (w2 << 16u);
}
float half_word_at(uint byte_off) {
    uint w = w32s[pi].w32[byte_off >> 2u];
    return (byte_off & 3u) == 0u ? unpackHalf2x16(w).x : unpackHalf2x16(w).y;
}
// The sub-scales and sub-mins of groups 2c and 2c + 1 from the twelve packed bytes in three words (layout in quant/k_quants.hpp).
// Branch-free, since the lanes of a block hold different c.
void scale_min_pair(uint s0, uint s1, uint s2, uint c, out float scl, out float mnl, out float sch, out float mnh) {
    uint sh = 16u * (c & 1u);   // the two groups' bytes within their words
    uint a0 = s0 >> sh, a1 = s1 >> sh, a2 = s2 >> sh;
    bool low = c < 2u;
    scl = float(low ? a0 & 63u : (a2 & 15u) | (((a0 >> 6u) & 3u) << 4u));
    mnl = float(low ? a1 & 63u : ((a2 >> 4u) & 15u) | (((a1 >> 6u) & 3u) << 4u));
    sch = float(low ? (a0 >> 8u) & 63u : ((a2 >> 8u) & 15u) | (((a0 >> 14u) & 3u) << 4u));
    mnh = float(low ? (a1 >> 8u) & 63u : ((a2 >> 12u) & 15u) | (((a1 >> 14u) & 3u) << 4u));
}
#endif

#ifdef LLMX_Q4
float symmetric_term(float d, vec2 xb, float dot) {
#ifdef LLMX_Q50
    // Centered before the dot:32*16*32767 <2^24, avoiding offset cancellation.
    precise float term = d * (xb.x * dot);
#else
    precise float term = d * (xb.x * dot - 8.0 * xb.y);
#endif
    return term;
}
#endif
#if defined(LLMX_K4) || defined(LLMX_K5)
float k45_term(float dl, float dh, float ml, float mh, vec4 xt, float pl, float ph, uint part) {
    ml = part == 0u ? ml : 0.0;
    mh = part == 0u ? mh : 0.0;
    return dl * (xt.x * pl) + dh * (xt.z * ph) - (ml * xt.y + mh * xt.w);
}
#endif
#ifdef LLMX_MXFP4
// The same row/column recomputed from decoded f32 weights, in block/value order.
// Double keeps each product in range before the final conversion to f32.
float mxfp4_wide(uint row, uint col) {
    uint nblocks = p.nin / MXFP4_BLOCK;
    uint base = row * nblocks * MXFP4_BYTES;
    precise double sum = 0.0;
    for (uint block = 0u; block < nblocks; ++block) {
        uint o = base + block * MXFP4_BYTES;
#ifndef LLMX_FLOAT_X
        uint a2 = xa_col(col) / 2u + block * 8u;
        double dx = double(xq_block(col, block).x);
#endif
        for (uint j = 0u; j < 32u; ++j) {
#ifdef LLMX_FLOAT_X
            sum += double(mxfp4_at(o, j)) * double(x[col * p.nin + block * 32u + j]);
#else
            uvec2 pair = xq2[a2 + j / 4u];
            uint word = (j & 1u) == 0u ? pair.x : pair.y;
            int q = (j & 2u) == 0u ? lo16(word) : hi16(word);
            sum += double(mxfp4_at(o, j)) * (double(q) * dx);
#endif
        }
    }
    return float(sum);
}

// The 17-byte block at byte o, read as five words: returns its scale byte and sets its 16 code bytes; the last block's fifth word lies in whole-word padded storage.
uint mxfp4_block(uint o, out uvec4 quants) {
    uint word = o >> 2u, shift = (o & 3u) * 8u;
    uvec4 lo = uvec4(w32s[pi].w32[word], w32s[pi].w32[word + 1u], w32s[pi].w32[word + 2u], w32s[pi].w32[word + 3u]);
    uvec4 hi = uvec4(lo.yzw, w32s[pi].w32[word + 4u]);
    quants = shift == 24u ? hi : (lo >> ((shift + 8u) & 31u)) | (hi << ((24u - shift) & 31u));
    return (lo.x >> shift) & 255u;
}

#ifdef LLMX_DOT
// Codes j and j + 2 as one pair from the table, in the twin's pair order; byte b of word k holds code 4k + b in its low nibble and code 4k + b + 16 in its high one.
void mxfp4_pairs_of(uvec4 quants, out uint pairs[16]) {
    for (uint k = 0u; k < 4u; ++k) {
        uint q = quants[k];
        pairs[4u * k] = mxfp4_pairs[(q & 15u) | ((q >> 12u) & 240u)];
        pairs[4u * k + 1u] = mxfp4_pairs[((q >> 8u) & 15u) | ((q >> 20u) & 240u)];
        pairs[4u * k + 2u] = mxfp4_pairs[((q >> 4u) & 15u) | ((q >> 16u) & 240u)];
        pairs[4u * k + 3u] = mxfp4_pairs[((q >> 12u) & 15u) | ((q >> 24u) & 240u)];
    }
}

// A block's integer sum against the twin's words from a2, the same exact value as the float sum.
int mxfp4_dot(uint pairs[16], uint a2) {
    int isum = 0;
    for (uint k = 0u; k < 4u; ++k) {
        isum += pair_dot(pairs[4u * k], pairs[4u * k + 1u], xq2[a2 + k]);
        isum += pair_dot(pairs[4u * k + 2u], pairs[4u * k + 3u], xq2[a2 + 4u + k]);
    }
    return isum;
}
#endif
#endif

// Each family's block pieces, which the one-row and two-row builds share: a lane's weights at offset o, and a block's float terms over its dots pl and ph (or p0 to p3).
#ifdef LLMX_Q4
// Wide Q4_0, lane t's block of a 9-word pair at word o: lane 0 words 0..4 (the scale, then nibble bytes starting two bytes in), lane 1 words 4..8 (the scale in word 4's high half, then aligned nibble words).
#define Q40W_WEIGHTS(o) \
    uint wa = w32s[pi].w32[o], wb = w32s[pi].w32[(o) + 1u], wc = w32s[pi].w32[(o) + 2u]; \
    uint wd = w32s[pi].w32[(o) + 3u], we = w32s[pi].w32[(o) + 4u]; \
    float d; \
    uint n0, n1, n2, n3; \
    if (t == 0u) { \
        d = unpackHalf2x16(wa).x; \
        n0 = (wa >> 16u) | (wb << 16u); \
        n1 = (wb >> 16u) | (wc << 16u); \
        n2 = (wc >> 16u) | (wd << 16u); \
        n3 = (wd >> 16u) | (we << 16u); \
    } else { \
        d = unpackHalf2x16(wa).y; \
        n0 = wb; n1 = wc; n2 = wd; n3 = we; \
    }
// Q4_0 of an odd block count, the 18-byte block at byte o through 16-bit loads.
#define Q40_WEIGHTS(o) \
    float d = unpackHalf2x16(uint(w16s[pi].w16[(o) >> 1u])).x; \
    uint n0 = hword_at((o) + 2u), n1 = hword_at((o) + 6u), n2 = hword_at((o) + 10u), n3 = hword_at((o) + 14u);
// Affine blocks carry d and m in word 0; Q5_1 inserts a high-bit word before the nibbles.
#ifdef LLMX_Q50
// Q5_0 offsets are bytes, not words: its 22-byte blocks alternate alignment.
#define AFFINE_WEIGHTS(o) \
    vec2 dm = vec2(unpackHalf2x16(uint(w16s[pi].w16[(o) >> 1u])).x, 0.0); \
    uint hb = hword_at((o) + 2u); \
    uint w1 = hword_at((o) + 6u), w2 = hword_at((o) + 10u); \
    uint w3 = hword_at((o) + 14u), w4 = hword_at((o) + 18u);
#else
#define AFFINE_WEIGHTS(o) \
    uint w0 = w32s[pi].w32[o]; \
    uint qoff = AFFINE_Q5 ? 2u : 1u; \
    uint hb = AFFINE_Q5 ? w32s[pi].w32[(o) + 1u] : 0u; \
    uint w1 = w32s[pi].w32[(o) + qoff], w2 = w32s[pi].w32[(o) + qoff + 1u]; \
    uint w3 = w32s[pi].w32[(o) + qoff + 2u], w4 = w32s[pi].w32[(o) + qoff + 3u]; \
    vec2 dm = unpackHalf2x16(w0);
#endif
// A Q4 block's four nibble words against the column's activation pairs, into pl and ph.
#define Q4_DOTS(n0, n1, n2, n3, lo0, lo1, hi0, hi1) \
    nibble_dots(n0, 0u, 0u, 0u, 0u, XA(lo0), XA(hi0), pl, ph); \
    nibble_dots(n1, 0u, 0u, 0u, 0u, XB(lo0), XB(hi0), pl, ph); \
    nibble_dots(n2, 0u, 0u, 0u, 0u, XA(lo1), XA(hi1), pl, ph); \
    nibble_dots(n3, 0u, 0u, 0u, 0u, XB(lo1), XB(hi1), pl, ph);
// Two fifth bits from the contiguous plane, in bit 4 of each 16-bit lane.
uint affine_fifth(uint bits, uint at) {
    return ((bits >> at) & 1u) << 4u | ((bits >> (at + 2u)) & 1u) << 20u;
}
void affine_nibble_dots(uint word, uint high, uint at, XWT lo, XWT hi, inout QDOT pl, inout QDOT ph) {
    nibble_dots(word, affine_fifth(high, at), affine_fifth(high, at + 1u),
                affine_fifth(high, at + 16u), affine_fifth(high, at + 17u), lo, hi, pl, ph);
}
#define AFFINE_DOTS(n0, n1, n2, n3, hb, lo0, lo1, hi0, hi1) \
    if (AFFINE_Q5) { \
        affine_nibble_dots(n0, hb, 0u, XA(lo0), XA(hi0), pl, ph); \
        affine_nibble_dots(n1, hb, 4u, XB(lo0), XB(hi0), pl, ph); \
        affine_nibble_dots(n2, hb, 8u, XA(lo1), XA(hi1), pl, ph); \
        affine_nibble_dots(n3, hb, 12u, XB(lo1), XB(hi1), pl, ph); \
    } else { Q4_DOTS(n0, n1, n2, n3, lo0, lo1, hi0, hi1) }
// Q4_0's term: the scaled dot less the offset of 8 through the block sum.
#define Q40_TERM(d, xb) symmetric_term(d, xb, float(pl + ph))
#ifdef LLMX_Q50
#define Q41_ADD(acc, dm, xb) acc += symmetric_term((dm).x, xb, float(pl + ph))
#elif defined(LLMX_DOT)
#define Q41_ADD(acc, dm, xb) acc += (dm).x * ((xb).x * float(pl + ph)) + (dm).y * (xb).y
#else
// The minimum's term joins the sum before the scaled dot, as two additions: a driver that reassociates float sums then has one order to contract in every build (docs/VULKAN.md, batch invariance).
#define Q41_ADD(acc, dm, xb) acc = (acc + (dm).y * (xb).y) + (dm).x * ((xb).x * float(pl + ph))
#endif
#endif
#if defined(LLMX_K4) || defined(LLMX_K5)
// A Q4_K or Q5_K lane's words of the block at word o: the scale and three sub-scale words, its 16 nibble bytes and Q5_K's fifth bits, and its two groups' scales folded once; each min uses the whole group sum in one lane.
#define K45_WEIGHTS(o) \
    uvec4 hd = w128s[pi].w128[(o) / 4u]; \
    vec2 dd = unpackHalf2x16(hd.x); \
    uint s0 = hd.y, s1 = hd.z, s2 = hd.w; \
    uvec4 nq = w128s[pi].w128[((o) + (q5 != 0u ? 12u : 4u) + c * 8u + h * 4u) / 4u]; \
    uvec4 hbv = uvec4(0u); \
    if (q5 != 0u) hbv = w128s[pi].w128[((o) + 4u + h * 4u) / 4u]; \
    float scl, mnl, sch, mnh; \
    scale_min_pair(s0, s1, s2, c, scl, mnl, sch, mnh); \
    float dl = dd.x * scl, dh = dd.x * sch; \
    float ml = dd.y * mnl, mh = dd.y * mnh;
// The low and high groups' dots, each scaled once, less the mins through the group sums.
#define K45_TERM(dl, dh, ml, mh, xt) k45_term(dl, dh, ml, mh, xt, float(pl), float(ph), h)
#endif
#ifdef LLMX_KQUANT
// A Q6_K lane's words of the block at byte o: its six quant words unpacked to a byte a value, groups 0 to 3's sub-scales and the block's scale.
#define Q6K_WEIGHTS(o) \
    uint lo0, lo1, hi0, hi1, qh0, qh1, sc_a, sc_b; \
    words_at((o) + n * 64u + l0, lo0, lo1); \
    words_at((o) + n * 64u + 32u + l0, hi0, hi1); \
    words_at((o) + 128u + n * 32u + l0, qh0, qh1); \
    words_at((o) + 192u + n * 8u, sc_a, sc_b); \
    float d = half_word_at((o) + 208u); \
    uint sh = (l0 / 16u) * 8u; \
    float s0 = float((int(sc_a << (24u - sh))) >> 24); \
    float s1 = float((int(sc_a << (8u - sh))) >> 24); \
    float s2 = float((int(sc_b << (24u - sh))) >> 24); \
    float s3 = float((int(sc_b << (8u - sh))) >> 24); \
    uint q0a = (lo0 & 0x0F0F0F0Fu) | ((qh0 & 0x03030303u) << 4u), q0b = (lo1 & 0x0F0F0F0Fu) | ((qh1 & 0x03030303u) << 4u); \
    uint q1a = (hi0 & 0x0F0F0F0Fu) | (((qh0 >> 2u) & 0x03030303u) << 4u), q1b = (hi1 & 0x0F0F0F0Fu) | (((qh1 >> 2u) & 0x03030303u) << 4u); \
    uint q2a = ((lo0 >> 4u) & 0x0F0F0F0Fu) | (((qh0 >> 4u) & 0x03030303u) << 4u), q2b = ((lo1 >> 4u) & 0x0F0F0F0Fu) | (((qh1 >> 4u) & 0x03030303u) << 4u); \
    uint q3a = ((hi0 >> 4u) & 0x0F0F0F0Fu) | (((qh0 >> 6u) & 0x03030303u) << 4u), q3b = ((hi1 >> 4u) & 0x0F0F0F0Fu) | (((qh1 >> 6u) & 0x03030303u) << 4u); \
// The four groups' dots, each under its sub-scale and the activation block's scale, under the block's scale.
#define Q6K_TERM(d, s0, s1, s2, s3, xa, xb) ((d) * ((s0) * ((xa).x * float(p0)) + (s1) * ((xa).z * float(p1)) + \
                                                        (s2) * ((xb).x * float(p2)) + (s3) * ((xb).z * float(p3))))
#endif

