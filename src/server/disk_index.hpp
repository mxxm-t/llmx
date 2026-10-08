#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include "core/sha.hpp"

// What the disk tier holds, as a tree (docs/DISK-TIER.md, Entries written as what changed).
// A history on disk is segments, each the blocks of a token range, and states, each the recurrent state at one position; this index says which files make a history, what of one is missing, and which file room, the cap or the age limit may take next.
// It holds no model and no store: it is given tokens, row classes, sizes and times, and answers with keys and ranges.

namespace server {

// A stretch of a history's rows that one class computed: the rows before `end`, from the stretch before it on, with the class's key (Model::row_class of the extent they took).
struct ClassRun {
    size_t end = 0;
    std::vector<size_t> key;
};

class DiskIndex {
public:
    // A segment never crosses a multiple of this many tokens: it bounds what a fork copies, 68 MiB on a 27B model over two cards, and leaves 75 files for 76k tokens.
    static constexpr size_t kSegmentTokens = 1024;

    using Time = std::filesystem::file_time_type;

    // A file: a segment, the blocks of tokens `first` to `end`, or a state at `end` (`first` equal to it).
    // `below` is the digest of the history under `first`, and `ends` the digest at each whole block from there to `end`, the last being the file's own; a state has none of its own and stands at `below`.
    struct File {
        uint64_t key = 0;
        bool state = false;
        size_t first = 0, end = 0;
        std::string below;
        std::vector<std::string> ends;
        uint64_t bytes = 0;
        Time used{};
        bool back = false;   // its conversation came back, which room spares longest
    };

    // The digest of a history at each whole block: d[k] names its first k blocks of `block` tokens, their tokens and the classes that computed them, whatever files hold them, so two histories share a digest exactly where they share every row below it.
    static std::vector<std::string> digests(const uint32_t* tokens, size_t n, const std::vector<ClassRun>& classes, size_t block) {
        std::vector<std::string> d{root()};
        size_t run = 0;
        for (size_t at = 0; at + block <= n; at += block) {
            core::Sha sha(true);
            sha.update(d.back().data(), d.back().size());
            sha.update(tokens + at, block * sizeof(uint32_t));
            // The classes of the block's rows, as stretches: where each ends within the block and its key, a class recorded in two stretches counting as one.
            for (size_t row = at; row < at + block;) {
                while (run < classes.size() && classes[run].end <= row) ++run;
                if (run == classes.size()) throw std::logic_error("disk index: a history's classes end before its tokens");
                size_t last = run;
                while (last + 1 < classes.size() && classes[last].end < at + block && classes[last + 1].key == classes[run].key) ++last;
                const uint64_t stop = std::min(classes[last].end, at + block) - at, size = classes[run].key.size();
                sha.update(&stop, sizeof stop);
                sha.update(&size, sizeof size);
                for (const size_t k : classes[run].key) {
                    const uint64_t v = k;
                    sha.update(&v, sizeof v);
                }
                row = at + (size_t)stop;
            }
            d.push_back(sha.hex());
        }
        return d;
    }
    static std::string root() { return std::string(64, '0'); }

    // What a file's header says of it, from which a start rebuilds the index: its kind, whether its conversation came back, its range and its digests; no token is written.
    static std::string describe(const File& f) {
        std::string b;
        const auto put = [&](uint64_t v) { b.append(reinterpret_cast<const char*>(&v), sizeof v); };
        put(f.state);
        put(f.back);
        put(f.first);
        put(f.end);
        put(f.ends.size());
        b += f.below;
        for (const std::string& e : f.ends) b += e;
        return b;
    }
    static bool parse(const std::string& b, File& f) {
        uint64_t v[5];
        if (b.size() < sizeof v + 64) return false;
        std::memcpy(v, b.data(), sizeof v);
        if (v[0] > 1 || v[1] > 1 || v[2] > v[3] || b.size() != sizeof v + 64 * (v[4] + 1) || (v[0] ? v[4] != 0 || v[2] != v[3] : v[4] == 0 || (v[3] - v[2]) % v[4] != 0)) return false;
        f.state = v[0];
        f.back = v[1];
        f.first = (size_t)v[2];
        f.end = (size_t)v[3];
        f.below = b.substr(sizeof v, 64);
        f.ends.clear();
        for (uint64_t k = 0; k < v[4]; ++k) f.ends.push_back(b.substr(sizeof v + 64 * (k + 1), 64));
        return true;
    }

    void add(File f) {
        files_.push_back(std::move(f));
        linked_ = false;
    }
    bool remove(uint64_t key) {
        const auto it = std::find_if(files_.begin(), files_.end(), [&](const File& f) { return f.key == key; });
        if (it == files_.end()) return false;
        files_.erase(it);
        linked_ = false;
        return true;
    }
    const std::vector<File>& files() const { return files_; }
    size_t states() const { return (size_t)std::count_if(files_.begin(), files_.end(), [](const File& f) { return f.state; }); }
    // A use of file `key`, which makes its whole path as young: a node is as old as its newest descendant, so only the leaf is touched.
    void touch(uint64_t key, Time used, bool back) {
        for (File& f : files_)
            if (f.key == key) {
                f.used = std::max(f.used, used);
                f.back = f.back || back;
                aged_ = false;
            }
    }
    uint64_t bytes() const {
        uint64_t n = 0;
        for (const File& f : files_) n += f.bytes;
        return n;
    }

    // One file of a path, and how far it is used: all of a segment, or its blocks below `use` where the history parts from it inside.
    struct Piece {
        uint64_t key = 0;
        size_t first = 0, end = 0, use = 0;
    };
    // What of a history is on disk: the segments of its path in order, the tokens they cover, how far whole segments reach (`whole`, where a write of the rest starts) and, with `state`, the state the path ends at.
    struct Path {
        std::vector<Piece> pieces;
        size_t length = 0, whole = 0;
        std::optional<uint64_t> state;
    };

    // The path of the history whose digests are `d`, up to `n` tokens: from the root, the segment that continues it, as far as its blocks are the history's.
    // With `state`, as a model that keeps one needs, it ends at the highest state on it, or is empty where there is none.
    Path path(const std::vector<std::string>& d, size_t block, size_t n, bool state) const {
        link();
        Path p;
        const size_t blocks = std::min(n / block, d.size() - 1);
        for (size_t at = 0; at < blocks;) {
            const File* best = nullptr;
            size_t shared = 0;
            const auto starts = starts_.find(d[at]);
            for (size_t i : starts == starts_.end() ? std::vector<size_t>{} : starts->second) {
                const File& f = files_[i];
                if (f.state || f.first != at * block) continue;
                size_t k = 0;
                while (k < f.ends.size() && at + k < blocks && f.ends[k] == d[at + k + 1]) ++k;
                if (k > shared) {
                    shared = k;
                    best = &f;
                }
            }
            if (!best) break;
            p.pieces.push_back(Piece{best->key, best->first, best->end, (at + shared) * block});
            at += shared;
            p.length = at * block;
            if (shared < best->ends.size()) break;
            p.whole = p.length;
        }
        if (!state) return p;
        for (size_t at = p.length / block + 1; at-- > 0 && !p.state;) {
            const auto starts = starts_.find(d[at]);
            for (size_t i : starts == starts_.end() ? std::vector<size_t>{} : starts->second)
                if (files_[i].state && files_[i].end == at * block) p.state = files_[i].key;
            if (p.state) p.length = at * block;
        }
        if (!p.state) p.length = 0;
        while (!p.pieces.empty() && p.pieces.back().first >= p.length) p.pieces.pop_back();
        if (!p.pieces.empty()) p.pieces.back().use = std::min(p.pieces.back().use, p.length);
        p.whole = std::min(p.whole, p.length);
        return p;
    }

    // The token ranges of a history of `n` tokens that are not on disk yet, each one segment to write: from where whole segments of its path end, cut at every multiple of kSegmentTokens.
    // A history that goes on past where it parts from a segment inside it writes its own from that segment's start, so no file is ever cut.
    std::vector<std::pair<size_t, size_t>> missing(const std::vector<std::string>& d, size_t block, size_t n) const {
        std::vector<std::pair<size_t, size_t>> out;
        const size_t end = std::min(n / block, d.size() - 1) * block;
        const Path p = path(d, block, n, false);
        // A history another's longer segment covers, parting from it inside, lacks nothing.
        for (size_t at = p.length >= end ? end : p.whole; at < end;) {
            const size_t stop = std::min(end, (at / kSegmentTokens + 1) * kSegmentTokens);
            out.push_back({at, stop});
            at = stop;
        }
        return out;
    }
    // Whether the state at `at` of that history is on disk.
    bool has_state(const std::vector<std::string>& d, size_t block, size_t at) const {
        link();
        if (at / block >= d.size()) return false;
        const auto starts = starts_.find(d[at / block]);
        for (size_t i : starts == starts_.end() ? std::vector<size_t>{} : starts->second)
            if (files_[i].state && files_[i].end == at) return true;
        return false;
    }

    // The highest state of that history at or below `n` tokens, whatever of its blocks is on disk: its position and its file. A state stands alone where its history's blocks are held in memory and not yet written.
    std::optional<std::pair<size_t, uint64_t>> state_below(const std::vector<std::string>& d, size_t block, size_t n) const {
        link();
        for (size_t at = std::min(n / block, d.size() - 1) + 1; at-- > 1;) {
            const auto starts = starts_.find(d[at]);
            for (size_t i : starts == starts_.end() ? std::vector<size_t>{} : starts->second)
                if (files_[i].state && files_[i].end == at * block) return std::make_pair(at * block, files_[i].key);
        }
        return std::nullopt;
    }

    // The files no path from the root reaches: segments whose parent is not there, and, with `states`, states whose path is not whole, which only a history still in memory could use.
    // What a start drops of what it adopts, and what a lost file leaves above it.
    std::vector<uint64_t> unreachable(bool states = true) const {
        link();
        std::vector<bool> reached(files_.size(), false);
        std::vector<const std::string*> open{&root_};
        while (!open.empty()) {
            const std::string* digest = open.back();
            open.pop_back();
            const auto starts = starts_.find(*digest);
            if (starts == starts_.end()) continue;
            for (size_t i : starts->second) {
                if (reached[i]) continue;
                reached[i] = true;
                for (const std::string& e : files_[i].ends) open.push_back(&e);
            }
        }
        std::vector<uint64_t> out;
        for (size_t i = 0; i < files_.size(); ++i)
            if (!reached[i] && (states || !files_[i].state)) out.push_back(files_[i].key);
        return out;
    }

    // The file room, the cap or the age limit takes next: only ever a leaf, a state or a segment nothing on disk stands on, so a path loses its end first and never its base.
    // First a conversation's earlier states, the oldest first, never its first nor its newest; then whole conversations, those that did not come back before those that did, the least recently used first, each from its end down.
    // With `before`, only a file with nothing beyond it used since (the age limit); never one of `pinned`, the files being read or written on; none where nothing may go.
    struct Victim {
        uint64_t key = 0;
        bool back = false;   // its conversation came back
    };
    std::optional<Victim> victim(std::optional<Time> before = std::nullopt, const std::vector<uint64_t>& pinned = {}) const {
        link();
        std::optional<size_t> best;
        int best_rank = 0;
        for (size_t i = 0; i < files_.size(); ++i) {
            const File& f = files_[i];
            if (children_[i] || (before && fresh_[i] >= *before) || std::find(pinned.begin(), pinned.end(), f.key) != pinned.end()) continue;
            const int rank = f.state && between_[i] ? 0 : back_[i] ? 2 : 1;
            const auto older = [&](size_t a, size_t b) {
                const Time fa = rank ? fresh_[a] : files_[a].used, fb = rank ? fresh_[b] : files_[b].used;
                return fa != fb ? fa < fb : files_[a].end > files_[b].end;
            };
            if (!best || rank < best_rank || (rank == best_rank && older(i, *best))) {
                best = i;
                best_rank = rank;
            }
        }
        if (!best) return std::nullopt;
        return Victim{files_[*best].key, back_[*best] != 0};
    }

private:
    static constexpr size_t kNone = ~size_t(0);

    // The tree's links, rebuilt after a file came or went: for each digest the files that stand at it, each file's parent (the segment holding the history under it, the one that ends there where there is a choice), the files that stand on each, and whether a state has states of its history both below and above it.
    // What follows from the uses, rebuilt after a use too, which changes no link: for each file the newest use of anything beyond it and whether any of that came back.
    void link() const {
        const size_t n = files_.size();
        if (!linked_) {
            starts_.clear();
            std::unordered_map<std::string, std::vector<size_t>> holds;
            for (size_t i = 0; i < n; ++i) {
                starts_[files_[i].below].push_back(i);
                for (const std::string& e : files_[i].ends) holds[e].push_back(i);
            }
            parent_.assign(n, kNone);
            children_.assign(n, 0);
            for (size_t i = 0; i < n; ++i) {
                const auto held = holds.find(files_[i].below);
                if (held == holds.end()) continue;
                for (size_t s : held->second)
                    if (s != i && (parent_[i] == kNone || files_[s].ends.back() == files_[i].below)) parent_[i] = s;
                if (parent_[i] != kNone) ++children_[parent_[i]];
            }
            // Deepest first, the order a file's use goes up to its parent in.
            std::vector<size_t> depth(n, 0);
            order_.resize(n);
            for (size_t i = 0; i < n; ++i) {
                order_[i] = i;
                for (size_t p = parent_[i]; p != kNone && depth[i] <= n; p = parent_[p]) ++depth[i];
            }
            std::sort(order_.begin(), order_.end(), [&](size_t a, size_t b) { return depth[a] > depth[b]; });
            on_.assign(n, {});
            for (size_t i = 0; i < n; ++i)
                if (parent_[i] != kNone) on_[parent_[i]].push_back(i);
            // For each state, the states under it on its way to the root lie below it, and it above them.
            std::vector<char> below(n, 0), above(n, 0);
            for (size_t g = 0; g < n; ++g) {
                if (!files_[g].state) continue;
                size_t pos = files_[g].end, steps = 0;
                bool own = true;
                for (size_t s = parent_[g]; s != kNone && steps <= n; s = parent_[s], ++steps) {
                    for (size_t f : on_[s])
                        if (f != g && files_[f].state && (own ? files_[f].end < pos : files_[f].end <= pos)) {
                            above[f] = 1;
                            below[g] = 1;
                        }
                    pos = files_[s].first;
                    own = false;
                }
            }
            between_.assign(n, false);
            for (size_t i = 0; i < n; ++i) between_[i] = below[i] && above[i];
            linked_ = true;
            aged_ = false;
        }
        if (aged_) return;
        // Each file's own use and what stands on it go up to its parent.
        std::vector<Time> sub(n);
        std::vector<char> sub_back(n);
        for (size_t i = 0; i < n; ++i) {
            sub[i] = files_[i].used;
            sub_back[i] = files_[i].back;
        }
        for (size_t i : order_)
            if (parent_[i] != kNone) {
                sub[parent_[i]] = std::max(sub[parent_[i]], sub[i]);
                sub_back[parent_[i]] = sub_back[parent_[i]] || sub_back[i];
            }
        // What lies beyond a segment is what stands on it, its states included; beyond a state, what stands on its segment at or above its position.
        fresh_.assign(n, Time{});
        back_.assign(n, false);
        for (size_t i = 0; i < n; ++i) {
            fresh_[i] = sub[i];
            back_[i] = sub_back[i];
            const size_t from = files_[i].state ? parent_[i] : i;
            if (from == kNone) continue;
            for (size_t g : on_[from]) {
                if (g == i || (files_[i].state && files_[g].first < files_[i].end)) continue;
                fresh_[i] = std::max(fresh_[i], sub[g]);
                back_[i] = back_[i] || sub_back[g];
            }
        }
        aged_ = true;
    }

    std::vector<File> files_;
    const std::string root_ = root();
    mutable bool linked_ = false, aged_ = false;
    mutable std::unordered_map<std::string, std::vector<size_t>> starts_;
    mutable std::vector<size_t> parent_, children_, order_;
    mutable std::vector<std::vector<size_t>> on_;
    mutable std::vector<Time> fresh_;
    mutable std::vector<char> back_, between_;
};

} // namespace server
