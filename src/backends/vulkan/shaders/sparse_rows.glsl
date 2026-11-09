// Value d of a kv row or a pooled row from its first word, per format with the format a constant, read through the kv and aux bindings' float and uint views.
// Shared by the two sparse attention kernels, so the softmax's dots and the columns' sums decode a row the same way (model_formats.glsl).
#ifndef LLMX_SPARSE_ROWS_GLSL
#define LLMX_SPARSE_ROWS_GLSL
#include "model_formats.glsl"
#define SOURCE_VALUE(NAME, FLOATS, FMT)      \
    float NAME(uint w, uint d) {             \
        if (FMT == 3u) return FLOATS[w + d]; \
        precise float v;                     \
        FORMAT_VALUE(w, d, FMT, v)           \
        return v;                            \
    }
#define FORMAT_WORD(i) kv_words[i]
SOURCE_VALUE(kv0, kv, 0u)
SOURCE_VALUE(kv1, kv, 1u)
SOURCE_VALUE(kv2, kv, 2u)
SOURCE_VALUE(kv3, kv, 3u)
#undef FORMAT_WORD
#define FORMAT_WORD(i) aux_words[i]
SOURCE_VALUE(ax0, aux, 0u)
SOURCE_VALUE(ax1, aux, 1u)
SOURCE_VALUE(ax2, aux, 2u)
SOURCE_VALUE(ax3, aux, 3u)
#undef FORMAT_WORD
float row_value(uint w, uint d, bool pooled, uint f) {
    if (pooled) return f == 0u ? ax0(w, d) : f == 1u ? ax1(w, d) : f == 2u ? ax2(w, d) : ax3(w, d);
    return f == 0u ? kv0(w, d) : f == 1u ? kv1(w, d) : f == 2u ? kv2(w, d) : kv3(w, d);
}
#endif
