// A model on one device against the same model split by layers over several, as raw float logits bit for bit: a scored text through the prompt path, a prefill in ubatch chunks, and greedy decode steps (docs/MULTI-DEVICE.md, phases 1 and 2).
// It then replays the prompt and steps by class, verifies marked and retracted drafts, and runs passes in flight, each of which must give what the single path gives (usage in the tool's message, docs/TENSOR-SPLIT.md).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>
#include "backends/devices.hpp"
#include "inference/load.hpp"

// A device given as `cpu` or a Vulkan index, in the spelling a device list takes.
static std::string name(const std::string& spec) { return spec == "cpu" ? spec : "vulkan:" + spec; }

// A decoding sequence with an established history beside a fresh prompt, then both decoding, in the same row order on one device and the split.
static size_t mixed(infer::Model& one, infer::Model& two, const std::vector<uint32_t>& ids) {
    one.reset();
    two.reset();
    infer::Sequence a = one.make_sequence(), b = one.make_sequence();
    infer::Sequence c = two.make_sequence(), d = two.make_sequence();
    infer::ExecContext x, y;
    const size_t warm = std::min(size_t(4), ids.size()), prompt = std::min(size_t(64), ids.size());
    const infer::BatchEntry warm_one{&a, ids.data(), warm, false};
    const infer::BatchEntry warm_two{&c, ids.data(), warm, false};
    one.forward(x, &warm_one, 1);
    two.forward(y, &warm_two, 1);
    size_t differ = 0, checked = 0;
    for (size_t pass = 0; pass < 3; ++pass) {
        const uint32_t next_a = ids[pass % ids.size()], next_b = ids[(pass + 1) % ids.size()];
        const size_t fresh = pass == 0 ? prompt : 1;
        const uint32_t* input = pass == 0 ? ids.data() : &next_b;
        const infer::BatchEntry first[] = {{&a, &next_a, 1, true}, {&b, input, fresh, true, true}};
        const infer::BatchEntry second[] = {{&c, &next_a, 1, true}, {&d, input, fresh, true, true}};
        one.forward(x, first, 2);
        two.forward(y, second, 2);
        for (size_t row = 0; row < 1 + fresh; ++row) {
            differ += std::memcmp(x.logits(row), y.logits(row), one.n_vocab() * sizeof(float)) != 0;
            ++checked;
        }
        if (a.length() != warm + pass + 1 || c.length() != a.length() ||
            b.length() != prompt + pass || d.length() != b.length())
            throw std::runtime_error("mixed pass did not preserve both histories");
    }
    std::printf("mixed path: 3 passes, %zu logits rows, %zu differ\n", checked, differ);
    one.reset(a);
    one.reset(b);
    two.reset(c);
    two.reset(d);
    return differ;
}

// The prompt and the decode's tokens recomputed as a paused request's resume recomputes them (docs/SERVER.md, pausing): the prompt in ubatch slices, the tokens as entries of extent 1 of up to 64 rows, whole and from a fork.
// Each must give the logits the decode gave after its last token, bit for bit; the count of those that differ, each fork counted in `forked`.
static size_t replay(infer::Model& m, const std::vector<uint32_t>& ids, size_t prompt, const std::vector<float>& want, size_t& forked) {
    m.reset();
    infer::ExecContext ctx;
    const size_t vocab = m.n_vocab(), ubatch = m.prefill_batch();
    auto rest = [&](infer::Sequence& seq) {
        std::vector<float> got;
        while (seq.length() < ids.size()) {
            const size_t at = seq.length();
            const bool generated = at >= prompt;
            const size_t n = generated ? std::min<size_t>(64, ids.size() - at) : std::min(ubatch, prompt - at);
            infer::BatchEntry e{&seq, ids.data() + at, n, at + n == ids.size()};
            e.extent = generated ? 1 : prompt;
            m.forward(ctx, &e, 1);
            if (e.want_logits) got.assign(ctx.logits(0), ctx.logits(0) + vocab);
        }
        return std::memcmp(got.data(), want.data(), vocab * sizeof(float)) != 0;
    };
    infer::Sequence whole = m.make_sequence();
    size_t differ = rest(whole);
    const size_t fork = (ids.size() - 1) / m.kv_block_tokens() * m.kv_block_tokens();
    if (fork && !m.keeps_state()) {
        infer::Sequence from = m.fork(whole, fork);
        m.reset(whole);
        differ += rest(from);
        m.reset(from);
        ++forked;
    }
    m.reset(whole);
    return differ;
}

// One entry of a pass: a sequence, where in the text its tokens start, how many, whether it wants the logits after its last, and its extent.
struct Planned {
    size_t seq, at, n;
    bool want;
    size_t extent;
};

// Passes in flight through the pass API (docs/MULTI-DEVICE.md): 2P sequences of a prompt cut from `ids` in chunks of up to 32 tokens and 8 generated tokens, four entries to a pass, P in flight, each stage recorded after a random delay.
// Every logits row must equal the same passes run in turn on the split and on the single device, each device running its passes in formation order and a pass retiring the round after its last stage as the server's round has them.
static size_t in_flight(infer::Model& one, infer::Model& two, const std::vector<uint32_t>& ids, size_t P, uint32_t seed, size_t& split_differ, size_t& one_differ) {
    one.reset();
    two.reset();
    const size_t S = two.stage_count(), K = 2 * P, vocab = two.n_vocab(), kChunk = 32, kSteps = 8, kPer = 4;
    uint32_t state = seed;
    const auto rng = [&state] { state = state * 1664525u + 1013904223u; return state >> 8; };
    // Each sequence's work, in order: its prompt's chunks, then its generated tokens one a pass.
    std::vector<std::vector<Planned>> work(K);
    for (size_t q = 0; q < K; ++q) {
        const size_t len = 1 + rng() % std::min<size_t>(80, ids.size() - kSteps - 1), start = rng() % (ids.size() - len - kSteps);
        for (size_t at = 0; at < len; at += kChunk) {
            const size_t n = std::min(kChunk, len - at);
            work[q].push_back({q, start + at, n, at + n == len, len});
        }
        for (size_t g = 0; g < kSteps; ++g) work[q].push_back({q, start + len + g, 1, true, 1});
    }
    std::vector<size_t> next(K, 0);
    std::vector<char> flying(K, 0);
    infer::ExecContext ctx;
    two.reserve_passes(ctx, P, kPer * kChunk, P * kPer);
    std::vector<infer::Sequence> seqs;
    for (size_t q = 0; q < K; ++q) seqs.push_back(two.make_sequence());
    struct Slot {
        bool live = false;
        uint64_t formed = 0;
        size_t ran = 0;
        std::vector<Planned> entries;
    };
    std::vector<Slot> slots(P);
    std::vector<std::vector<Planned>> formed;         // every pass, in formation order
    std::vector<std::vector<std::vector<float>>> rows; // its logits rows, as the pass API gave them
    const auto pause = [&] { std::this_thread::sleep_for(std::chrono::microseconds(rng() % 3000)); };
    for (bool more = true; more;) {
        // From the last stage down, each stage takes its oldest waiting pass; passes whose last stage ran in an earlier round retire.
        std::vector<std::pair<size_t, size_t>> advance;
        for (size_t s = S; s-- > 1;) {
            size_t pick = P;
            for (size_t k = 0; k < P; ++k)
                if (slots[k].live && slots[k].ran == s && (pick == P || slots[k].formed < slots[pick].formed)) pick = k;
            if (pick < P) advance.push_back({pick, s});
        }
        std::vector<size_t> retire;
        for (size_t k = 0; k < P; ++k)
            if (slots[k].live && slots[k].ran == S) retire.push_back(k);
        std::sort(retire.begin(), retire.end(), [&](size_t a, size_t b) { return slots[a].formed < slots[b].formed; });
        for (const auto& a : advance) {
            pause();
            two.run_pass_stage(ctx, a.first, a.second);
            ++slots[a.first].ran;
        }
        for (size_t k : retire) {
            Slot& sl = slots[k];
            std::vector<std::vector<float>>& got = rows[sl.formed - 1];
            size_t w = 0;
            for (const Planned& e : sl.entries)
                if (e.want) {
                    const float* row = two.pass_logits(ctx, k, w++);
                    got.emplace_back(row, row + vocab);
                }
            two.end_pass(ctx, k);
            for (const Planned& e : sl.entries) flying[e.seq] = 0;
            sl.live = false;
        }
        // New passes in the free slots, each of up to four sequences not in flight, their next work in turn.
        for (size_t k = 0; k < P; ++k) {
            if (slots[k].live) continue;
            std::vector<Planned> entries;
            for (size_t q0 = rng() % K, i = 0; i < K && entries.size() < kPer; ++i) {
                const size_t q = (q0 + i) % K;
                if (!flying[q] && next[q] < work[q].size()) entries.push_back(work[q][next[q]]);
            }
            if (entries.empty()) break;
            std::vector<infer::BatchEntry> batch;
            for (const Planned& e : entries) {
                infer::BatchEntry b{&seqs[e.seq], ids.data() + e.at, e.n, e.want};
                b.extent = e.extent;
                batch.push_back(b);
                flying[e.seq] = 1;
                ++next[e.seq];
            }
            // Each slot's pass writes its own four logits rows.
            Slot& sl = slots[k];
            sl.live = true;
            sl.formed = formed.size() + 1;
            sl.ran = 0;
            sl.entries = entries;
            formed.push_back(entries);
            rows.emplace_back();
            two.begin_pass(ctx, k, batch.data(), batch.size(), k * kPer);
            pause();
            two.run_pass_stage(ctx, k, 0);
            ++sl.ran;
        }
        more = false;
        for (size_t k = 0; k < P; ++k) more = more || slots[k].live;
        for (size_t q = 0; q < K; ++q) more = more || next[q] < work[q].size();
    }
    for (auto& s : seqs) two.reset(s);
    // The same passes one after another through forward, on the split and on the single device.
    const auto serial = [&](infer::Model& m, size_t& differ) {
        infer::ExecContext x;
        std::vector<infer::Sequence> own;
        for (size_t q = 0; q < K; ++q) own.push_back(m.make_sequence());
        for (size_t p = 0; p < formed.size(); ++p) {
            std::vector<infer::BatchEntry> batch;
            for (const Planned& e : formed[p]) {
                infer::BatchEntry b{&own[e.seq], ids.data() + e.at, e.n, e.want};
                b.extent = e.extent;
                batch.push_back(b);
            }
            m.forward(x, batch.data(), batch.size());
            for (size_t w = 0; w < rows[p].size(); ++w) differ += std::memcmp(x.logits(w), rows[p][w].data(), vocab * sizeof(float)) != 0;
        }
        for (auto& s : own) m.reset(s);
    };
    serial(two, split_differ);
    serial(one, one_differ);
    return formed.size();
}

// Verifies of drafts (docs/SPECULATIVE.md, section 3): after the prompt, rounds that mark the history, feed k + 1 of the decode's tokens in one pass, k from 1 to 16, and retract to keep some of them, none and all included, on both models.
// Every verify row and every step after a retract is compared, and it returns the rows that differ and counts the rounds.
static size_t verify(infer::Model& one, infer::Model& two, const std::vector<uint32_t>& history, size_t prompt, size_t& rounds) {
    const size_t vocab = one.n_vocab();
    one.reset();
    two.reset();
    const std::vector<uint32_t> head(history.begin(), history.begin() + (std::ptrdiff_t)prompt);
    one.prefill(head);
    two.prefill(head);
    size_t at = prompt, differ = 0;
    std::vector<float> rows;
    auto step_differs = [&](uint32_t id) {
        const std::vector<float> x = one.step((int)id), y = two.step((int)id);
        return std::memcmp(x.data(), y.data(), vocab * sizeof(float)) != 0;
    };
    for (rounds = 0; at + 17 <= history.size(); ++rounds) {
        const size_t k = 1 + rounds % 16, keep = rounds % 5 == 4 ? k + 1 : (rounds * 7) % (k + 1);
        if (!one.mark() || !two.mark()) throw std::runtime_error("a mark was refused");
        const float* a = one.step(history.data() + at, k + 1);
        rows.assign(a, a + (k + 1) * vocab);
        const float* b = two.step(history.data() + at, k + 1);
        for (size_t i = 0; i <= k; ++i) differ += std::memcmp(&rows[i * vocab], b + i * vocab, vocab * sizeof(float)) != 0;
        if (one.retract(at + keep) != at + keep || two.retract(at + keep) != at + keep) throw std::runtime_error("a retract inside a mark fell short");
        at += keep;
        if (!keep) {
            differ += step_differs(history[at]);
            ++at;
        }
    }
    if (at < history.size()) differ += step_differs(history[at]);
    return differ;
}

int main(int argc, char** argv) {
    if (argc < 3 || argc > 10) {
        std::fprintf(stderr, "usage: llmx-split-check <model.gguf> <text file> [single] [split, e.g. 0,1,2] [steps] [ubatch] [f16|f32] [auto|f16|bf16|f32|int8] [tensor width]\n");
        return 2;
    }
    try {
        const std::string single = argc > 3 ? argv[3] : "0", split = argc > 4 ? argv[4] : "0,1";
        const int steps = argc > 5 ? std::atoi(argv[5]) : 32, ubatch = argc > 6 ? std::atoi(argv[6]) : 0;
        // The cache type is checked before the model file is read, so a wrong name costs no load.
        infer::ModelOptions options;
        if (argc > 7) options.kv_k = options.kv_v = backend::kv_type_of(argv[7]);
        options.kv_tokens = 4096;
        // A model that keeps a recurrent state holds a slot for each sequence at once: the mixed passes' two, and the 2P of passes in flight for P up to twice the stages, which are at most the split's devices.
        options.state_slots = std::max<size_t>(2, 4 * core::comma_list(split).size());
        // A mark for the verifies, of up to 17 rows.
        options.mark_slots = 1;
        options.mark_rows = 17;
        const size_t width = argc > 9 ? (size_t)std::atoi(argv[9]) : 1;
        if (!width) throw std::runtime_error("the tensor width is a whole number of at least 1");
        infer::PlacementRequest alone;
        for (const std::string& d : core::comma_list(single)) alone.names.push_back(name(d));
        alone.ubatch = ubatch;
        alone.width = width;
        const std::string dtype = argc > 8 ? argv[8] : "auto";
        if (dtype != "auto") {
            for (auto d : {backend::Dtype::f32, backend::Dtype::f16, backend::Dtype::bf16, backend::Dtype::int8})
                if (dtype == backend::dtype_name(d)) alone.dtype = d;
            if (!alone.dtype) throw std::runtime_error("dtype must be auto, f16, bf16, f32 or int8");
        }
        const auto first = infer::load_model(argv[1], backend::make_backends(alone.names), alone, options);
        infer::Model& one = *first->model;
        std::fputs(first->dtype.describe().c_str(), stderr);
        std::ifstream in(argv[2], std::ios::binary);
        const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        const std::vector<uint32_t> ids = first->tok->encode(text);
        if (ids.size() < 2) throw std::runtime_error("the text holds fewer than two tokens");

        // The split takes equal shares of the layers, as a device list with --layer-shares 1,1 and so on places them.
        // Each entry is a backend of its own, without the CLI's listed-once rule (backend::device_specs), so `cpu,cpu` splits over two CPU backends.
        infer::PlacementRequest request;
        for (const std::string& d : core::comma_list(split)) request.names.push_back(name(d));
        request.shares.assign(request.names.size() / width, 1);
        request.width = width;
        request.ubatch = ubatch;
        request.dtype = alone.dtype;
        const auto second = infer::load_model(argv[1], backend::make_backends(request.names), request, options);
        infer::Model& two = *second->model;
        std::fputs(second->dtype.describe().c_str(), stderr);
        std::printf("%s: %zu tokens, %s caches; single %s, split:\n%s", argv[1], ids.size(), backend::kv_type_name(options.kv_k), single.c_str(), second->plan.c_str());

        const size_t vocab = one.n_vocab();
        std::vector<float> scored(ids.size() * vocab);
        one.score(ids, [&](size_t pos, const float* logits) { std::memcpy(&scored[pos * vocab], logits, vocab * sizeof(float)); });
        size_t differ = 0;
        two.score(ids, [&](size_t pos, const float* logits) { differ += std::memcmp(&scored[pos * vocab], logits, vocab * sizeof(float)) != 0; });
        std::printf("prompt path: %zu positions x %zu logits, %zu positions differ\n", ids.size(), vocab, differ);

        // Decode from the prompt, both models fed the same greedy token each step.
        one.reset();
        two.reset();
        std::vector<float> a = one.prefill(ids), b = two.prefill(ids);
        size_t steps_differ = std::memcmp(a.data(), b.data(), vocab * sizeof(float)) != 0;
        std::vector<uint32_t> history = ids;
        for (int s = 0; s < steps; ++s) {
            uint32_t next = 0;
            for (size_t v = 1; v < vocab; ++v)
                if (a[v] > a[next]) next = (uint32_t)v;
            a = one.step((int)next);
            b = two.step((int)next);
            history.push_back(next);
            steps_differ += std::memcmp(a.data(), b.data(), vocab * sizeof(float)) != 0;
        }
        std::printf("decode path: prefill and %d greedy steps, %zu differ\n", steps, steps_differ);
        size_t forked = 0;
        const size_t replay_differ = steps ? replay(one, history, ids.size(), a, forked) + replay(two, history, ids.size(), b, forked) : 0;
        std::printf("replay by class: the prompt and %d steps on one device and the split, whole and %zu from a fork at a block, %zu differ from the decode\n", steps, forked, replay_differ);
        size_t rounds = 0;
        const size_t verify_differ = verify(one, two, history, ids.size(), rounds);
        std::printf("verifies: %zu rounds of 2 to 17 rows after a mark, each retracted, %zu rows differ\n", rounds, verify_differ);
        const size_t mixed_differ = mixed(one, two, ids);
        // Passes in flight at P = S, S + 1 and 2S, on a split that takes them and a text long enough for their prompts and steps.
        size_t flight_differ = 0;
        const size_t S = two.stage_count();
        if (two.pipelined() && ids.size() >= 12) {
            for (size_t P : {S, S + 1, 2 * S}) {
                size_t split_differ = 0, one_differ = 0;
                const size_t passes = in_flight(one, two, ids, P, 20260927u + (uint32_t)P, split_differ, one_differ);
                std::printf("passes in flight: P = %zu, %zu passes with random host delays, %zu logits rows differ from forward on the split and %zu from the single device\n",
                            P, passes, split_differ, one_differ);
                flight_differ += split_differ + one_differ;
            }
        } else {
            std::printf("passes in flight: not run, the split %s\n", two.pipelined() ? "text is too short" : "takes one pass at a time");
        }
        const auto paths = [](const char* label, infer::Model& model) {
            const auto devices = model.take_matrix_paths();
            for (size_t i = 0; i < devices.size(); ++i) {
                std::printf("%s device %zu matrix paths:", label, i);
                for (const auto& path : devices[i]) std::printf(" %s", path.c_str());
                std::printf("\n");
            }
        };
        paths("single", one);
        paths("split", two);
        const bool same = !differ && !steps_differ && !replay_differ && !verify_differ && !mixed_differ && !flight_differ;
        std::printf("%s\n", same ? "bit-identical" : "DIFFERENT");
        return same ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "llmx-split-check: %s\n", e.what());
        return 2;
    }
}
