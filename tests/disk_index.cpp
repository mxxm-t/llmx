// The disk tier's index alone (server::DiskIndex, docs/DISK-TIER.md, Entries written as what changed): histories as paths of segments and states, with no model, no store and no file.
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "server/disk_index.hpp"

namespace {

using server::ClassRun;
using server::DiskIndex;

constexpr size_t kBlock = 64;
int checks = 0;

void require(bool ok, const std::string& what) {
    ++checks;
    if (!ok) throw std::runtime_error(what);
}

// A history's tokens: `n` of them from `seed`, the same below `shared` for every seed.
std::vector<uint32_t> tokens(uint32_t seed, size_t n, size_t shared = 0) {
    std::vector<uint32_t> t(n);
    for (size_t i = 0; i < n; ++i) t[i] = (uint32_t)((i < shared ? 7 : seed) * 2654435761u + i * 40503u);
    return t;
}
std::vector<ClassRun> classed(size_t n, size_t key = 1) { return {ClassRun{n, {key, 0}}}; }
std::vector<std::string> digests(const std::vector<uint32_t>& t, const std::vector<ClassRun>& c) { return DiskIndex::digests(t.data(), t.size(), c, kBlock); }

DiskIndex::Time at(int seconds) { return DiskIndex::Time{} + std::chrono::seconds(seconds); }

// The file of `key`, which must be there.
DiskIndex::File file(const DiskIndex& index, uint64_t key) {
    for (const DiskIndex::File& f : index.files())
        if (f.key == key) return f;
    throw std::runtime_error("no file " + std::to_string(key) + " in the index");
}

// The files a history of `n` tokens is written as, on top of what `index` holds of it: its missing segments and, with `state`, the state at its end.
uint64_t next_key = 0;
std::vector<uint64_t> write(DiskIndex& index, const std::vector<std::string>& d, size_t n, bool state, int when, bool back = false) {
    std::vector<uint64_t> keys;
    for (const auto& range : index.missing(d, kBlock, n)) {
        DiskIndex::File f;
        f.key = ++next_key;
        f.first = range.first;
        f.end = range.second;
        f.below = d[range.first / kBlock];
        for (size_t k = range.first / kBlock + 1; k <= range.second / kBlock; ++k) f.ends.push_back(d[k]);
        f.bytes = (range.second - range.first) * 100;
        f.used = at(when);
        f.back = back;
        index.add(f);
        keys.push_back(f.key);
    }
    if (state && !index.has_state(d, kBlock, n)) {
        DiskIndex::File f;
        f.key = ++next_key;
        f.state = true;
        f.first = f.end = n;
        f.below = d[n / kBlock];
        f.bytes = 5000;
        f.used = at(when);
        f.back = back;
        index.add(f);
        keys.push_back(f.key);
    }
    return keys;
}

}  // namespace

int main() {
    try {
        // A digest names the rows below it: the same tokens and classes give the same digests however long the history, other tokens or another class another from the block that differs on.
        const std::vector<uint32_t> a = tokens(1, 40 * kBlock);
        const auto da = digests(a, classed(a.size()));
        require(da.size() == 41 && da[0] == DiskIndex::root(), "a history of 40 blocks has other than 41 digests from the root");
        const std::vector<uint32_t> shorter(a.begin(), a.begin() + 10 * kBlock + 5);
        const auto ds = digests(shorter, classed(shorter.size()));
        require(ds.size() == 11 && std::equal(ds.begin(), ds.end(), da.begin()), "a history's first blocks have other digests than the same rows of a longer one");
        std::vector<uint32_t> other = a;
        other[5 * kBlock + 3] ^= 1;
        const auto dother = digests(other, classed(other.size()));
        require(std::equal(da.begin(), da.begin() + 6, dother.begin()) && da[6] != dother[6] && da[40] != dother[40], "a token changed in block 5 did not part the digests there");
        std::vector<ClassRun> two{ClassRun{3 * kBlock + 10, {1, 0}}, ClassRun{a.size(), {2, 0}}};
        const auto dclass = digests(a, two);
        require(std::equal(da.begin(), da.begin() + 4, dclass.begin()) && da[4] != dclass[4], "rows of another class from block 3 on did not part the digests there");
        const std::vector<ClassRun> split{ClassRun{2 * kBlock, {1, 0}}, ClassRun{a.size(), {1, 0}}};
        require(digests(a, split) == da, "one class recorded as two stretches gave other digests");

        // A history is written as segments cut at every 1024 tokens and a state, and found again whole; a longer one of the same conversation writes only what it adds.
        DiskIndex index;
        const size_t first_turn = 20 * kBlock, second_turn = 21 * kBlock, third_turn = 40 * kBlock;
        require(index.path(da, kBlock, first_turn, true).length == 0 && index.missing(da, kBlock, first_turn) == std::vector<std::pair<size_t, size_t>>{{0, 1024}, {1024, first_turn}},
                "an empty index holds a path, or a history of 1280 tokens is not two segments cut at 1024");
        const std::vector<uint64_t> w1 = write(index, da, first_turn, true, 10);
        require(w1.size() == 3, "a first turn was not written as two segments and a state");
        DiskIndex::Path p = index.path(da, kBlock, a.size(), true);
        require(p.length == first_turn && p.whole == first_turn && p.pieces.size() == 2 && p.pieces[1].use == first_turn && p.state == w1[2], "a history's path is not its two segments and its state");
        const std::vector<uint64_t> w2 = write(index, da, second_turn, true, 20);
        require(w2.size() == 2 && file(index, w2[0]).first == first_turn && file(index, w2[0]).end == second_turn, "a turn of one block wrote other than that block and a state");
        const std::vector<uint64_t> w3 = write(index, da, third_turn, true, 30);
        require(w3.size() == 3 && file(index, w3[0]).end == 2048 && file(index, w3[1]).first == 2048, "a longer turn was not cut at 2048");
        p = index.path(da, kBlock, a.size(), true);
        require(p.length == third_turn && p.pieces.size() == 5 && p.state == w3[2], "the whole conversation's path is not its five segments and its newest state");
        require(index.path(da, kBlock, second_turn + 5 * kBlock, true).state == w2[1] && index.path(da, kBlock, second_turn + 5 * kBlock, true).length == second_turn,
                "a request that reaches into the third turn does not fork the second's state");
        require(index.path(da, kBlock, second_turn + 5 * kBlock, false).length == second_turn + 5 * kBlock, "a model without a state does not take the blocks a segment holds below its end");
        require(index.missing(da, kBlock, third_turn).empty() && index.has_state(da, kBlock, third_turn), "a history wholly on disk has something missing");

        // An edit that forks inside a segment shares the segments wholly below it and writes its own from that segment's start, cut as any history is.
        std::vector<uint32_t> edit = a;
        for (size_t i = 18 * kBlock; i < edit.size(); ++i) edit[i] ^= 0x55;
        const auto de = digests(edit, classed(edit.size()));
        p = index.path(de, kBlock, edit.size(), false);
        require(p.length == 18 * kBlock && p.whole == 1024 && p.pieces.size() == 2 && p.pieces[1].use == 18 * kBlock, "an edit at block 18 does not share the first segment whole and the second to its fork");
        require(index.path(de, kBlock, edit.size(), true).length == 0, "an edit forked a state its history does not reach");
        const std::vector<uint64_t> we = write(index, de, 22 * kBlock, true, 40);
        require(we.size() == 2 && file(index, we[0]).first == 1024 && file(index, we[0]).end == 22 * kBlock, "an edit did not write its own segment from 1024");
        require(index.path(de, kBlock, edit.size(), true).length == 22 * kBlock && index.path(da, kBlock, a.size(), true).length == third_turn, "the edit or the conversation it left is not whole after it");

        // Room takes leaves only, a conversation's middle states first, the oldest first, then the conversation that was not used longest from its end down, and its base last.
        const auto taken = [&] {
            const std::optional<DiskIndex::Victim> v = index.victim();
            require(v.has_value(), "nothing to take from an index that holds files");
            const DiskIndex::File f = file(index, v->key);
            index.remove(f.key);
            require(index.unreachable().empty(), "taking file " + std::to_string(f.key) + " left files nothing reaches");
            return f;
        };
        DiskIndex::File f = taken();
        require(f.key == w2[1], "room's first file is not the conversation's middle state, the state at the second turn");
        f = taken();
        require(f.key == w3[2], "after the middle state, room did not take the older conversation's newest state");
        f = taken();
        require(f.key == w3[1] && !f.state, "then its last segment");
        f = taken();
        require(f.key == w3[0], "then the segment under it");
        f = taken();
        require(f.key == w2[0], "then the second turn's block");
        f = taken();
        require(f.key == w1[2] && f.state, "then its first state, which nothing stands on any more");
        f = taken();
        require(f.key == w1[1], "then the first turn's second segment, which the edit does not share");
        f = taken();
        require(f.key == we[1] && f.state, "with the older conversation gone but for the shared base, room turns to the edit's state");
        f = taken();
        require(f.key == we[0], "then the edit's own segment");
        f = taken();
        require(f.key == w1[0] && index.files().empty(), "the shared base did not go last");

        // A conversation that came back is spared before one that did not, whatever their ages; a use renews a whole path through its leaf; the age limit takes only what was not used since.
        const std::vector<uint32_t> b = tokens(2, 4 * kBlock), c = tokens(3, 4 * kBlock);
        const auto db = digests(b, classed(b.size())), dc = digests(c, classed(c.size()));
        const std::vector<uint64_t> wb = write(index, db, b.size(), true, 100, true), wc = write(index, dc, c.size(), true, 200);
        require(index.victim()->key == wc[1], "room took a conversation that came back before a newer one that did not");
        require(index.victim(at(150)) && index.victim(at(150))->key == wb[1] && index.victim(at(150))->back, "the age limit at 150 does not take the conversation last used at 100");
        index.touch(wb[1], at(300), true);
        require(!index.victim(at(150)).has_value(), "a use at 300 did not keep a conversation from the age limit at 150");
        require(index.victim(at(250))->key == wc[1] && !index.victim(at(250))->back && !index.victim(at(250), {wc[1]}), "the age limit at 250 does not take the conversation last used at 200");

        // A file lost from the middle of a path leaves everything above it unreachable and the path below whole.
        const std::vector<uint64_t> wa = write(index, da, third_turn, true, 400);
        require(wa.size() == 4, "a conversation written again is not three segments and a state");
        index.remove(wa[1]);
        const std::vector<uint64_t> gone = index.unreachable();
        require(gone == std::vector<uint64_t>{wa[2], wa[3]}, "a segment lost from a path did not leave the one above it and the state unreachable");
        for (uint64_t key : gone) index.remove(key);
        require(index.path(da, kBlock, a.size(), false).length == 1024 && index.path(da, kBlock, a.size(), true).length == 0, "the path below a lost segment is not whole, or a state is found on it");
        require(index.missing(da, kBlock, third_turn).front() == std::make_pair<size_t, size_t>(1024, 2048), "what is written again does not start where the path ends");

        // A file's description reads back as the file, and a damaged one is refused.
        {
            const DiskIndex::File seg = file(index, wa[0]);
            DiskIndex::File back;
            require(DiskIndex::parse(DiskIndex::describe(seg), back) && back.first == seg.first && back.end == seg.end && back.below == seg.below && back.ends == seg.ends && !back.state,
                    "a segment's description does not read back");
            DiskIndex::File st;
            st.state = true;
            st.first = st.end = 128;
            st.below = da[2];
            st.back = true;
            require(DiskIndex::parse(DiskIndex::describe(st), back) && back.state && back.back && back.end == 128 && back.below == da[2] && back.ends.empty(), "a state's description does not read back");
            std::string cut = DiskIndex::describe(seg);
            cut.pop_back();
            require(!DiskIndex::parse(cut, back) && !DiskIndex::parse("", back), "a damaged description was read");
        }

        std::cout << "disk index: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "disk index: " << e.what() << '\n';
        return 1;
    }
}
