// The shared n-gram hash (blocks::ngram_rows) in DeepSeek-V4.1's form against the deepseek41 tiny reference's hash ids: the token map, the mapped pad fill that cuts nothing in text, and every slicing of the sequence.
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/json.hpp"
#include "model/arch/blocks.hpp"

namespace {

int checked = 0;

void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
    ++checked;
}

void refuses(const std::string& what, const std::string& text, const std::function<void()>& run) {
    try {
        run();
    } catch (const std::exception& e) {
        require(std::string(e.what()).find(text) != std::string::npos, what + ": refused with '" + e.what() + "'");
        return;
    }
    throw std::runtime_error(what + ": not refused");
}

const jmini::Value& field(const jmini::Value& v, const char* name) {
    const jmini::Value* f = v.get(name);
    if (!f) throw std::runtime_error(std::string("reference: no field ") + name);
    return *f;
}

uint64_t whole(const jmini::Value& v) {
    if (!v.isNumber() || !v.whole || v.num < 0 || v.num >= 9007199254740992.0) throw std::runtime_error("reference: not an exact whole number");
    return uint64_t(v.num);
}

std::vector<uint64_t> wholes(const jmini::Value& v) {
    std::vector<uint64_t> out;
    for (const jmini::Value& x : v.asArray()) out.push_back(whole(x));
    return out;
}

struct Reference {
    uint32_t vocab = 0;
    std::vector<infer::blocks::NgramHash> layers;
    std::vector<uint32_t> ids;
    std::vector<std::vector<int32_t>> expected;  // [layer][position * heads + head]
    size_t heads = 0, pad_position = 0, mapped_position = 0;
    uint32_t mapped_raw = 0, mapped_id = 0;
};

Reference read_reference(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open " + path);
    std::stringstream text;
    text << file.rdbuf();
    const jmini::Value root = jmini::parse(text.str());
    const jmini::Value& args = field(root, "args");
    const jmini::Value& e = field(root, "engram");
    Reference r;
    r.vocab = uint32_t(whole(field(args, "vocab_size")));
    const int ngram = int(whole(field(args, "engram_max_ngram_size"))), per = int(whole(field(args, "engram_n_heads")));
    std::vector<uint32_t> map;
    for (uint64_t id : wholes(field(e, "token_map"))) map.push_back(uint32_t(id));
    const size_t n_layers = field(e, "layer_ids").asArray().size();
    for (size_t l = 0; l < n_layers; ++l) {
        infer::blocks::NgramHash h;
        h.ngram_size = ngram;
        h.heads_per_ngram = per;
        h.fill_id = int(whole(field(e, "pad_id")));
        h.multipliers = wholes(field(e, "multipliers").asArray()[l]);
        h.head_offsets = wholes(field(e, "offsets").asArray()[l]);
        h.head_vocab_sizes = wholes(field(e, "primes").asArray()[l]);
        h.token_map = map;
        h.mapped_vocab = uint32_t(whole(field(args, "engram_compressed_vocab_size")));
        h.fill_cuts = false;
        r.layers.push_back(h);
    }
    r.heads = size_t(ngram - 1) * size_t(per);
    for (uint64_t id : wholes(field(root, "sequence"))) r.ids.push_back(uint32_t(id));
    r.expected.assign(n_layers, {});
    const auto& positions = field(e, "hash_ids").asArray();
    require(positions.size() == r.ids.size(), "reference: a hash row for every position");
    for (const jmini::Value& at : positions)
        for (size_t l = 0; l < n_layers; ++l)
            for (uint64_t id : wholes(at.asArray()[l])) r.expected[l].push_back(int32_t(id));
    r.pad_position = size_t(whole(field(field(e, "pad_in_text"), "position")));
    const jmini::Value& differs = field(e, "mapped_differs");
    r.mapped_position = size_t(whole(field(differs, "position")));
    r.mapped_raw = uint32_t(whole(field(differs, "raw")));
    r.mapped_id = uint32_t(whole(field(differs, "mapped")));
    return r;
}

// Rows of ids[first, last) after the ngram_size - 1 ids before first, padded with the fill as the runtime pads its token history.
std::vector<int32_t> rows(const infer::blocks::NgramHash& h, const Reference& r, size_t first, size_t last) {
    std::vector<uint32_t> tail;
    for (size_t k = size_t(h.ngram_size - 1); k > 0; --k) tail.push_back(first >= k ? r.ids[first - k] : uint32_t(h.fill_id));
    std::vector<int32_t> out((last - first) * r.heads, -1);
    const size_t n = infer::blocks::ngram_rows(h, r.vocab, tail.data(), tail.size(), r.ids.data() + first, last - first, out.data(), out.size());
    require(n == out.size(), "ngram_rows returns the rows it wrote");
    return out;
}

bool same_span(const std::vector<int32_t>& got, const std::vector<int32_t>& want, size_t from) {
    for (size_t i = 0; i < got.size(); ++i)
        if (got[i] != want[from + i]) return false;
    return true;
}

void check(const Reference& r) {
    const size_t n = r.ids.size();
    require(r.layers[0].token_map[r.mapped_raw] == r.mapped_id && r.mapped_raw != r.mapped_id, "reference: a token the map moves");
    require(r.ids[r.pad_position] == uint32_t(r.layers[0].fill_id), "reference: the pad id in text");
    for (size_t l = 0; l < r.layers.size(); ++l) {
        const auto& h = r.layers[l];
        const std::string at = " on Engram layer " + std::to_string(l);
        require(rows(h, r, 0, n) == r.expected[l], "the reference's hash ids for the whole sequence" + at);
        std::vector<int32_t> fresh(n * r.heads, -1);
        infer::blocks::ngram_rows(h, r.vocab, nullptr, 0, r.ids.data(), n, fresh.data(), fresh.size());
        require(fresh == r.expected[l], "the reference's hash ids with no tail, the start's fill taken" + at);
        for (size_t cut = 1; cut < n; ++cut)
            require(same_span(rows(h, r, 0, cut), r.expected[l], 0) && same_span(rows(h, r, cut, n), r.expected[l], cut * r.heads),
                    "the hash ids cut at " + std::to_string(cut) + at);
        for (size_t p = 0; p < n; ++p)
            require(same_span(rows(h, r, p, p + 1), r.expected[l], p * r.heads), "the hash ids a token at a time at " + std::to_string(p) + at);

        auto cuts = h;
        cuts.fill_cuts = true;
        require(!same_span(rows(cuts, r, r.pad_position + 1, r.pad_position + 2), r.expected[l], (r.pad_position + 1) * r.heads),
                "a fill that cuts in text changes the row after the pad id" + at);
        auto unmapped = h;
        unmapped.token_map.clear();
        unmapped.mapped_vocab = 0;
        require(!same_span(rows(unmapped, r, r.mapped_position, r.mapped_position + 1), r.expected[l], r.mapped_position * r.heads),
                "the map changes the row of a token it moves" + at);
    }

    const auto& h = r.layers[0];
    int32_t out[64] = {};
    const uint32_t one[] = {r.ids[0]};
    auto short_map = h;
    short_map.token_map.pop_back();
    refuses("a map shorter than the vocabulary", "token map does not cover the vocabulary",
            [&] { infer::blocks::ngram_rows(short_map, r.vocab, nullptr, 0, one, 1, out, 64); });
    auto no_range = h;
    no_range.mapped_vocab = 0;
    refuses("a map without its range", "token map and its range come together",
            [&] { infer::blocks::ngram_rows(no_range, r.vocab, nullptr, 0, one, 1, out, 64); });
    auto no_map = h;
    no_map.token_map.clear();
    refuses("a range without its map", "token map and its range come together",
            [&] { infer::blocks::ngram_rows(no_map, r.vocab, nullptr, 0, one, 1, out, 64); });
    auto narrow = h;
    narrow.mapped_vocab = r.mapped_id;
    const uint32_t moved[] = {r.mapped_raw};
    refuses("a mapped id past the map's range", "token id is outside the vocabulary",
            [&] { infer::blocks::ngram_rows(narrow, r.vocab, nullptr, 0, moved, 1, out, 64); });
    auto wide = h;
    wide.mapped_vocab = 0xffffffffu;
    refuses("products past int64 over the map's range", "hash product exceeds int64 range",
            [&] { infer::blocks::ngram_rows(wide, r.vocab, nullptr, 0, one, 1, out, 64); });
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: llmx-ngram-hash-test tests/data/deepseek41_reference.json");
        check(read_reference(argv[1]));
        std::printf("ngram-hash: %d checks passed\n", checked);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ngram-hash: FAIL: %s\n", e.what());
        return 1;
    }
}
