// The view table a batched cache kernel reads (vulkan_backend.cpp,
// view_table): tab[0] is the view count, then six words per view, the
// batch row its rows start at, the dispatch-local row they start at (a
// dispatch may cover a subset of the batch's views), its row count, its
// history length, the offset of its block table and that table's length;
// after the entries, every view's block ids in order. A dispatch covers
// several views, so a workgroup or thread finds its view by walking the
// entries, which are few.
#ifndef LLMX_VIEWS_GLSL
#define LLMX_VIEWS_GLSL
#define VIEW_WORDS 6u
uint view_count() { return tab[0]; }
uint view_row0(uint v) { return tab[1u + VIEW_WORDS * v]; }
uint view_local0(uint v) { return tab[2u + VIEW_WORDS * v]; }
uint view_nq(uint v) { return tab[3u + VIEW_WORDS * v]; }
uint view_hist(uint v) { return tab[4u + VIEW_WORDS * v]; }
uint view_block(uint v, uint i) { return tab[1u + VIEW_WORDS * view_count() + tab[5u + VIEW_WORDS * v] + i]; }
// The view holding dispatch-local row g, and g's row within it; the
// batch row is then view_row0(v) + b.
void view_of_row(uint g, out uint v, out uint b) {
    uint n = view_count();
    v = 0u;
    for (uint i = 1u; i < n; ++i) if (g >= view_local0(i)) v = i;
    b = g - view_local0(v);
}
#endif
