// tanh(v) in float: an odd series below 1/8, where (1 - e) / (1 + e) would cancel, and through e = exp(-2|v|) above, so a large |v| gives +-1 rather than inf / inf.
#ifndef LLMX_TANH_GLSL
#define LLMX_TANH_GLSL
float tanh_of(float v) {
    float a = abs(v);
    if (a < 0.125) { float s = v * v; return v * (1.0 + s * (-1.0 / 3.0 + s * (2.0 / 15.0 + s * (-17.0 / 315.0)))); }
    float e = exp(-2.0 * a);
    return sign(v) * ((1.0 - e) / (1.0 + e));
}
#endif
