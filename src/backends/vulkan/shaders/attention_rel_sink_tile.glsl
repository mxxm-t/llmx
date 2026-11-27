// The prompt build (LLMX_TILE): a workgroup a (query head, 64 rows of view p.row0, whose p.rows rows are a prompt's), four lanes a row each holding 32 of its dims, keys staged 32 at a time.
// Scores add the relative bias from a staged slice of the table and start from the row's sink, as the per-row build does; dots and sums run in another order than it.
#define TR 64u
#define TKEYS 32u
#define SPANMAX (TR + TKEYS)
shared vec4 tk[TKEYS * 32u];
shared vec4 tv[TKEYS * 32u];
shared float tb[16u * SPANMAX];
void main() {
    const uint h = gl_WorkGroupID.x, lid = gl_LocalInvocationID.x, r = lid / 4u, quarter = lid % 4u;
    const uint D = p.dim, KD = p.kv_heads * D, HD = p.heads * D, kh = h / (p.heads / p.kv_heads);
    const uint v = p.row0;
    uint len, src = 0u, first;
    if (p.ring != 0u) {
        first = tab[1u + 5u * v];
        len = tab[3u + 5u * v];
        src = tab[4u + 5u * v];
    } else {
        first = view_local0(v);
        len = view_hist(v);
    }
    const uint g0 = first + gl_WorkGroupID.y * TR, rows = first + p.rows;
    const uint gl = min(g0 + TR, rows) - 1u, g = min(g0 + r, gl);
    const bool live = g0 + r < rows;
    const uint pos = len + g - first, pos_hi = len + gl - first, pos_lo = len + g0 - first;
    const uint lo = p.window != 0u && pos + 1u > p.window ? pos + 1u - p.window : 0u;
    // Key chunks start on a TKEYS grid, so a row's chunks and rescales are the same wherever its tile starts.
    const uint lo_tile = p.window != 0u && pos_lo + 1u > p.window ? (pos_lo + 1u - p.window) / TKEYS * TKEYS : 0u;
    vec4 qv[8], acc[8];
    for (uint j = 0u; j < 8u; ++j) {
        const uint at = g * HD + h * D + quarter * 32u + 4u * j;
        qv[j] = vec4(q[at], q[at + 1u], q[at + 2u], q[at + 3u]);
        acc[j] = vec4(0.0);
    }
    float rv[16];
    for (uint i = 0u; i < 16u; ++i) rv[i] = i < p.R ? rel[(g * p.heads + h) * p.R + i] : 0.0;
    float top = sink[g * p.heads + h];
    precise float total = 1.0;
    for (uint k0 = lo_tile; k0 <= pos_hi; k0 += TKEYS) {
        barrier();
        // Stage the keys' K and V rows of head kh and the table columns of the distances this tile meets.
        for (uint i = lid; i < TKEYS * D; i += 256u) {
            const uint j = i / D, e = i - j * D, kp = k0 + j;
            float kx = 0.0, vx = 0.0;
            if (kp <= pos_hi) {
                if (kp >= len) {
                    const uint at = (first + kp - len) * KD + kh * D + e;
                    kx = kn[at], vx = vn[at];
                } else {
                    const uint at = stored(v, src, kp) + kh * D + e;
                    kx = st[at], vx = st[at + KD];
                }
            }
            tk[i / 4u][i % 4u] = kx;
            tv[i / 4u][i % 4u] = vx;
        }
        // Distance d = pos - kp runs over [pos_lo - k0 - TKEYS + 1, pos_hi - k0], staged from dmin.
        const int dmin = int(pos_lo) - int(k0) - int(TKEYS) + 1;
        for (uint i = lid; i < p.R * SPANMAX; i += 256u) {
            const uint rr = i / SPANMAX, c = i - rr * SPANMAX;
            const int d = dmin + int(c);
            tb[rr * SPANMAX + c] = d >= 0 && uint(d) < p.extent ? table[rr * p.extent + uint(d)] : 0.0;
        }
        barrier();
        float s[TKEYS];
        float peak = uintBitsToFloat(0xff800000u);
        for (uint j = 0u; j < TKEYS; ++j) {
            precise float part = 0.0;
            for (uint x = 0u; x < 8u; ++x) {
                const vec4 kx = tk[j * 32u + quarter * 8u + x];
                part = fma(qv[x].w, kx.w, fma(qv[x].z, kx.z, fma(qv[x].y, kx.y, fma(qv[x].x, kx.x, part))));
            }
            part += subgroupShuffleXor(part, 1u);
            part += subgroupShuffleXor(part, 2u);
            const uint kp = k0 + j;
            precise float bias = 0.0;
            const uint c = uint(int(pos) - int(kp) - dmin);
            if (kp <= pos && pos - kp < p.extent)
                for (uint i = 0u; i < p.R; ++i) bias = fma(rv[i], tb[i * SPANMAX + c], bias);
            const precise float score = part * p.scale + bias;
            s[j] = kp <= pos && kp >= lo ? score : uintBitsToFloat(0xff800000u);
            peak = max(peak, s[j]);
        }
        const float now = max(top, peak), f = exp(top - now);
        total *= f;
        for (uint x = 0u; x < 8u; ++x) acc[x] *= f;
        for (uint j = 0u; j < TKEYS; ++j) {
            const float w = exp(s[j] - now);
            total += w;
            for (uint x = 0u; x < 8u; ++x) acc[x] = fma(vec4(w), tv[j * 32u + quarter * 8u + x], acc[x]);
        }
        top = now;
    }
    if (live)
        for (uint x = 0u; x < 8u; ++x) {
            const uint at = g * HD + h * D + quarter * 32u + 4u * x;
            const vec4 a = acc[x] / total;
            o[at] = a.x, o[at + 1u] = a.y, o[at + 2u] = a.z, o[at + 3u] = a.w;
        }
    // A pooled layer's new rows go to their positions: the first query head of each KV head writes them.
    if (p.ring == 0u && h % (p.heads / p.kv_heads) == 0u)
        for (uint i = lid; i < TR * D; i += 256u) {
            const uint rr = i / D, e = i - rr * D, gg = g0 + rr;
            if (gg >= rows) break;
            const uint at = stored(v, 0u, len + gg - first) + kh * D + e;
            st[at] = kn[gg * KD + kh * D + e];
            st[at + KD] = vn[gg * KD + kh * D + e];
        }
}
