// A sparse row's selection slots (Op::sparse_attention_blocks): slot t is position t mod 2^id_shift of the block its id names, the ids group-major by KV head, and a padding id names none.
// At id_shift 0 and one group a slot is its id, the chain every other caller runs.
uint slot_base(uint row, uint kvh) {
    return (row * p.id_groups + kvh / (p.n_head_kv / p.id_groups)) * p.selected_stride;
}
// An id past the visible blocks, padding included, names none before the shift, which would wrap it into the history.
uint slot_token(uint base, uint t, uint visible) {
    uint id = ids[base + (t >> p.id_shift)];
    return id > (visible >> p.id_shift) ? visible : (id << p.id_shift) + (t & ((1u << p.id_shift) - 1u));
}
