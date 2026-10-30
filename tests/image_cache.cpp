// An image's identity in every tier of the server's cache (docs/VISION.md, section 5), and its encode in pieces (section 10), through the scheduler over a tiny hybrid model (AGENTS.md, Tests).
// Each reply is held to the same request's reply on a fresh model, bit for bit.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "server_harness.hpp"

namespace {

// The hybrid model with rope sections [2, 1, 1]: its four rotated pairs take an image row's start, row, column and start; with `mtp`, an MTP block after its layers.
gguf::GGUFModel image_hybrid(bool mtp = false) {
    gguf::GGUFModel m = served_hybrid(kHybrid, mtp);
    for (auto& kv : m.kv)
        if (kv.first == "qwen35.rope.dimension_sections")
            for (size_t i = 0; i < kv.second.arr.size(); ++i) kv.second.arr[i].i = i == 0 ? 2 : i < 3 ? 1 : 0;
    return m;
}

// An image of `h` by `w` rows from row `first`, its rows and digest made from `seed`, so two seeds are two images of one grid.
infer::MediaRun image(size_t first, uint32_t h, uint32_t w, uint8_t seed) {
    infer::MediaRun run;
    run.first = first;
    run.rows = size_t(h) * w;
    run.grid_h = h;
    run.grid_w = w;
    std::vector<float> rows(run.rows * (size_t)kHybrid.embd);
    for (size_t i = 0; i < rows.size(); ++i) rows[i] = float(int((i * 13 + seed * 31) % 23) - 11) / 16.0f;
    run.encoded = std::make_shared<const std::vector<float>>(std::move(rows));
    for (size_t i = 0; i < run.digest.size(); ++i) run.digest[i] = uint8_t(seed * 7 + i);
    return run;
}

const uint32_t kVocab = (uint32_t)kHybrid.vocab;

// A prompt of `before` text ids, an image, then `after` text ids, the image's rows holding id 0.
Req with_image(uint32_t first, size_t before, uint32_t h, uint32_t w, size_t after, uint8_t seed, int cap) {
    Req r;
    r.prompt = prompt_of(first, before, kVocab);
    r.images = {image(before, h, w, seed)};
    r.prompt.insert(r.prompt.end(), size_t(h) * w, 0u);
    const std::vector<uint32_t> tail = prompt_of(first + 1, after, kVocab);
    r.prompt.insert(r.prompt.end(), tail.begin(), tail.end());
    r.cap = cap;
    return r;
}

// The follow-up turn of `first`: its prompt, its reply and 30 more ids, with its images.
Req follow(const Make& make, const bpe::Tokenizer& tok, const Req& first, uint32_t seed) {
    auto model = make(1024, 0);
    Req f = first;
    const Reply reply = serve(*model, tok, 3, {{first}})[0];
    for (const auto& t : reply) f.prompt.push_back(t.id);
    const std::vector<uint32_t> more = prompt_of(seed, 30, kVocab);
    f.prompt.insert(f.prompt.end(), more.begin(), more.end());
    f.cap = 32;
    return f;
}

// `r` after the `setup` requests forks `forked` tokens and gives its reply on a fresh model.
void as_on_fresh(const Make& make, const bpe::Tokenizer& tok, const std::vector<Req>& setup, const Req& r, size_t forked, const std::string& what) {
    auto fresh = make(1024, 0);
    const Reply alone = serve(*fresh, tok, 3, {{r}})[0];
    std::vector<std::vector<Req>> waves;
    for (const Req& s : setup) waves.push_back({s});
    waves.push_back({r});
    auto model = make(1024, 0);
    server::Scheduler::Stats stats;
    same(alone, serve(*model, tok, 3, waves, &stats).back(), what);
    require(stats.prefix_tokens == forked, what + ": " + std::to_string(stats.prefix_tokens) + " tokens reused, against " + std::to_string(forked));
}

// A follow-up turn of the same image forks the first turn's checkpoint at 256 tokens, past its image and inside one, and the same turn with another image of the same grid and pad ids at that place forks nothing.
void follow_ups(const gguf::GGUFModel& weights, const bpe::Tokenizer& tok) {
    const Make kept = on(weights, [] { return cpus(1); }, 3, 3);
    const Req past = with_image(5, 100, 6, 5, 170, 1, 40), inside = with_image(7, 240, 6, 5, 30, 2, 40);
    as_on_fresh(kept, tok, {past}, follow(kept, tok, past, 6), 2 * kBlock, "a follow-up past an image");
    const Req again = follow(kept, tok, inside, 8);
    as_on_fresh(kept, tok, {inside}, again, 2 * kBlock, "a follow-up forking inside an image");
    Req other = again;
    other.images = {image(240, 6, 5, 3)};
    as_on_fresh(kept, tok, {inside}, other, 0, "a follow-up with another image of the same grid at the same place");
}

// Three uncapped requests with images on a pool too small for them: paused and resumed, each recomputing its image's rows from what gives them again, each with its reply alone.
void paused(const gguf::GGUFModel& weights, const bpe::Tokenizer& tok) {
    const Make make = on(weights, [] { return cpus(1); }, 3, 0);
    const std::vector<Req> reqs = {with_image(1, 20, 4, 4, 10, 4, 0), with_image(2, 5, 3, 5, 4, 5, 0), with_image(3, 12, 5, 3, 6, 6, 0)};
    std::vector<Reply> alone;
    for (const Req& r : reqs) {
        auto model = make(1024, 0);
        alone.push_back(serve(*model, tok, 3, {{r}})[0]);
    }
    auto model = make(1024, 0);
    server::Scheduler::Stats stats;
    const std::vector<Reply> together = serve(*model, tok, 3, {reqs}, &stats);
    require(stats.pauses >= 1 && stats.recomputed > 0, "requests with images on a small pool: " + std::to_string(stats.pauses) + " pauses, " + std::to_string(stats.recomputed) + " rows recomputed");
    for (size_t i = 0; i < reqs.size(); ++i) same(alone[i], together[i], "a paused request with an image, " + std::to_string(i));
}

// Two conversations with images and their follow-ups, the second evicting the first's donor to host memory: each follow-up is promoted from the host tier, forks 256 tokens and gives its reply alone.
void host_tier(const gguf::GGUFModel& weights, const bpe::Tokenizer& tok) {
    const Make make = on(weights, [] { return cpus(1); }, 3, 3);
    const Req a = with_image(5, 240, 6, 5, 30, 7, 40), b = with_image(9, 100, 4, 6, 176, 8, 40);
    const Req fa = follow(make, tok, a, 6), fb = follow(make, tok, b, 10);
    std::vector<Reply> alone;
    for (const Req& r : {a, b, fa, fb}) {
        auto model = make(512, 0);
        alone.push_back(serve(*model, tok, 3, {{r}})[0]);
    }
    auto model = make(512, 0);
    server::Scheduler::Stats stats;
    const std::vector<Reply> got = serve(*model, tok, 3, {{a}, {b}, {fa}, {fb}}, &stats, 0, (size_t)1 << 30);
    for (size_t i = 0; i < got.size(); ++i) same(alone[i], got[i], "the host tier with images, request " + std::to_string(i));
    require(stats.host_hits == 2 && stats.prefix_tokens == 4 * kBlock,
            "the host tier with images: " + std::to_string(stats.host_hits) + " promotions and " + std::to_string(stats.prefix_tokens) + " tokens reused, against 2 and " + std::to_string(4 * kBlock));
}

// Six conversations with images demoted to disk, the first's asked for again: its history is read back from disk, given its images by the request, forked at 256 tokens and gives the reply it gives on a fresh model.
void disk_tier(const gguf::GGUFModel& weights, const bpe::Tokenizer& tok) {
    const Make make = on(weights, [] { return cpus(1); }, 3, 3);
    DiskRoot disk("image");
    auto model = make(2048, 16);
    std::vector<Req> turns;
    std::vector<Reply> replies;
    Req again;
    Reply got;
    size_t reused = 0;
    server::Scheduler::Stats stats;
    {
        // Four copies' slabs and a MiB for the images' last rows the copies keep (ImageTail).
        server::Scheduler sched(*model, tok, 1, 64, 0, false, 4 * ((size_t)64 << 20) + ((size_t)1 << 20), nullptr, 0, false, disk.options(uint64_t(1) << 30));
        std::thread runner([&] { sched.run(); });
        const auto submit = [&](const Req& r) {
            return sched.submit(r.prompt, params_of(r), std::numeric_limits<size_t>::max(), 0, server::Request::Clock::now(), {}, images_of(r));
        };
        try {
            for (uint32_t k = 0; k < 6; ++k) turns.push_back(with_image(30 + k, 240, 6, 5, 30, uint8_t(20 + k), 20));
            for (size_t k = 0; k < turns.size(); ++k) {
                replies.push_back(drain(*submit(turns[k])));
                const size_t written = (k >= 4 ? k - 3 : 0) * 2;
                within_a_minute([&] { return sched.stats().disk_entries == written; }, "images on disk: " + std::to_string(written) + " entries after turn " + std::to_string(k));
            }
            again = turns[0];
            for (const auto& t : replies[0]) again.prompt.push_back(t.id);
            const std::vector<uint32_t> more = prompt_of(50, 30, kVocab);
            again.prompt.insert(again.prompt.end(), more.begin(), more.end());
            again.cap = 32;
            const auto h = submit(again);
            got = drain(*h);
            reused = h->reused();
            stats = sched.stats();
        } catch (...) {
            sched.stop();
            runner.join();
            throw;
        }
        sched.stop();
        runner.join();
    }
    require(reused == 2 * kBlock && stats.disk_hits == 1, "a history with an image from disk: " + std::to_string(reused) + " tokens reused and " + std::to_string(stats.disk_hits) +
                                                              " entries read, against " + std::to_string(2 * kBlock) + " and 1");
    auto fresh = make(2048, 16);
    same(serve(*fresh, tok, 1, {{again}})[0], got, "a history with an image from disk, its follow-up");
}

// A CPU backend of 16 decode columns, so verifies of drafts fit a pass.
struct Wide : backend::CpuBackend {
    Wide() { set_threads(1); }
    size_t decode_columns() const override { return 16; }
};

// Requests with images drafting with lookup and with the embedded drafter give each reply its reply without drafts, drafts fed: lookup drafts no image row, and the drafter's rows take their positions after the image.
void drafts(const gguf::GGUFModel& weights, const bpe::Tokenizer& tok) {
    const std::vector<Req> reqs = {with_image(1, 40, 4, 4, 20, 9, 48), with_image(2, 10, 3, 5, 30, 10, 48)};
    const Make plain = on(weights, [] { return cpus(1); }, 3, 0);
    const Make drafting = on(weights, [] { return std::vector<backend::BackendPtr>{std::make_shared<Wide>()}; }, 3, 0, 3, 9, true);
    std::vector<Reply> alone;
    for (const Req& r : reqs) {
        auto model = plain(1024, 0);
        alone.push_back(serve(*model, tok, 2, {{r}})[0]);
    }
    for (const bool embedded : {false, true}) {
        auto model = drafting(1024, 0);
        std::unique_ptr<infer::spec::Proposer> proposer;
        if (embedded) proposer = std::make_unique<infer::spec::Embedded>(*model);
        else proposer = std::make_unique<infer::spec::Lookup>();
        server::Scheduler::Stats stats;
        const std::vector<Reply> got = serve(*model, tok, 2, {reqs}, &stats, 0, 0, proposer.get(), 8);
        const std::string what = std::string(embedded ? "the embedded drafter" : "lookup") + " beside images";
        size_t fed = 0;
        for (size_t d : stats.drafted) fed += d;
        require(!embedded || fed > 0, what + ": no draft fed");
        for (size_t i = 0; i < reqs.size(); ++i) same(alone[i], got[i], what + ", request " + std::to_string(i));
    }
}


// An image's encode in `steps` pieces, each counted with the passes retired before it, its steps' life watched: the rows those of r's image, which the request does not hold.
struct Encodes {
    std::atomic<size_t> steps{0}, passes{0};
    std::atomic<int> alive{0};
    std::chrono::milliseconds pause{0};
    std::mutex m;
    std::vector<size_t> at;  // the passes retired as each piece began
};
server::Scheduler::Images stepped(const Req& r, size_t steps, const std::shared_ptr<Encodes>& seen) {
    server::Scheduler::Images images;
    images.runs = r.images;
    for (auto& run : images.runs) run.encoded.reset();
    const auto rows = r.images.at(0).encoded;
    images.encode = [rows, steps, seen](size_t) {
        struct Life {
            std::shared_ptr<Encodes> seen;
            ~Life() {
                if (seen) --seen->alive;
            }
        };
        ++seen->alive;
        auto life = std::make_shared<Life>();
        life->seen = seen;
        auto left = std::make_shared<std::atomic<size_t>>(steps);
        return server::ImageSteps{[seen, life, left] {
                                      ++seen->steps;
                                      {
                                          std::lock_guard<std::mutex> lk(seen->m);
                                          seen->at.push_back(seen->passes.load());
                                      }
                                      std::this_thread::sleep_for(seen->pause);
                                      --*left;
                                  },
                                  [rows, left]() -> std::shared_ptr<const std::vector<float>> { return left->load() ? nullptr : rows; }, steps};
    };
    return images;
}

// A follow-up turn that forks the first turn's history inside its image encodes nothing: the rows past the fork, those of the image's last block, are kept from the first turn's encode.
// Its reply is its reply on a fresh model; a follow-up with another image of the same grid at that place forks nothing and encodes its image.
void reused_inside(const gguf::GGUFModel& weights, const bpe::Tokenizer& tok) {
    const Make kept = on(weights, [] { return cpus(1); }, 3, 3);
    const Req inside = with_image(7, 240, 6, 5, 30, 2, 40);
    const Req again = follow(kept, tok, inside, 8);
    Req other = again;
    other.images = {image(240, 6, 5, 3)};
    for (const Req* r : std::vector<const Req*>{&again, &other}) {
        auto fresh = kept(1024, 0);
        const Reply alone = serve(*fresh, tok, 3, {{*r}})[0];
        auto model = kept(1024, 0);
        server::Scheduler sched(*model, tok, 3, 64);
        std::thread runner([&] { sched.run(); });
        const auto submit = [&](const Req& q, server::Scheduler::Images images) {
            return sched.submit(q.prompt, params_of(q), std::numeric_limits<size_t>::max(), 0, server::Request::Clock::now(), {}, std::move(images));
        };
        const bool same_image = r == &again;
        const std::string what = same_image ? "a follow-up forking inside its image" : "a follow-up with another image where the first turn's was";
        try {
            auto first = std::make_shared<Encodes>(), next = std::make_shared<Encodes>();
            drain(*submit(inside, stepped(inside, 2, first)));
            const auto h = submit(*r, stepped(*r, 2, next));
            same(alone, drain(*h), what);
            require(first->steps == 2, what + ": the first turn's image ran " + std::to_string(first->steps.load()) + " pieces, against 2");
            require(h->reused() == (same_image ? 2 * kBlock : 0), what + ": " + std::to_string(h->reused()) + " tokens reused");
            require(next->steps == (same_image ? 0u : 2u), what + ": its image ran " + std::to_string(next->steps.load()) + " pieces, against " + (same_image ? "none" : "2"));
        } catch (...) {
            sched.stop();
            runner.join();
            throw;
        }
        sched.stop();
        runner.join();
    }
}

// A follow-up whose first turn's history went to host memory forks the copy inside its image and encodes nothing, the image's last rows going with the copy.
// With no host tier the history leaves with nothing kept and the follow-up encodes its image again; each reply is its reply on a fresh model.
void reused_evicted(const gguf::GGUFModel& weights, const bpe::Tokenizer& tok) {
    const Make make = on(weights, [] { return cpus(1); }, 3, 3);
    const Req a = with_image(5, 240, 6, 5, 30, 7, 40), b = with_image(9, 100, 4, 6, 176, 8, 40);
    const Req fa = follow(make, tok, a, 6);
    auto fresh = make(512, 0);
    const Reply alone = serve(*fresh, tok, 3, {{fa}})[0];
    for (const size_t host : {size_t(1) << 30, size_t(0)}) {
        const std::string what = host ? "a follow-up of a history in host memory" : "a follow-up of a history evicted with no copy";
        auto model = make(512, 0);
        server::Scheduler sched(*model, tok, 3, 64, 0, false, host);
        std::thread runner([&] { sched.run(); });
        const auto submit = [&](const Req& q, server::Scheduler::Images images) {
            return sched.submit(q.prompt, params_of(q), std::numeric_limits<size_t>::max(), 0, server::Request::Clock::now(), {}, std::move(images));
        };
        try {
            auto first = std::make_shared<Encodes>(), other = std::make_shared<Encodes>(), next = std::make_shared<Encodes>();
            drain(*submit(a, stepped(a, 2, first)));
            drain(*submit(b, stepped(b, 2, other)));
            const auto h = submit(fa, stepped(fa, 2, next));
            same(alone, drain(*h), what);
            const server::Scheduler::Stats stats = sched.stats();
            require(stats.host_hits == (host ? 1u : 0u), what + ": " + std::to_string(stats.host_hits) + " promotions");
            require(h->reused() == (host ? 2 * kBlock : 0), what + ": " + std::to_string(h->reused()) + " tokens reused");
            require(next->steps == (host ? 0u : 2u), what + ": its image ran " + std::to_string(next->steps.load()) + " pieces, against " + (host ? "none" : "2"));
        } catch (...) {
            sched.stop();
            runner.join();
            throw;
        }
        sched.stop();
        runner.join();
    }
}

// Beside a reply an image's six pieces, each longer than the chunk budget, meet one a replies' pass (encode_pieces), so with one pass in flight no more than two begin between two retirements while the reply has passes to come.
// A request cancelled mid-encode ends it at a piece boundary, freeing the encoder for the next image.
void pieces(const gguf::GGUFModel& weights, const bpe::Tokenizer& tok) {
    const Make make = on(weights, [] { return cpus(1); }, 3, 0);
    Req text;
    text.prompt = prompt_of(40, 30, kVocab);
    text.cap = 200;
    const Req img = with_image(3, 20, 4, 4, 10, 11, 20), next = with_image(4, 8, 3, 3, 5, 12, 12);
    std::vector<Reply> alone;
    for (const Req& r : {text, img, next}) {
        auto model = make(1024, 0);
        alone.push_back(serve(*model, tok, 3, {{r}})[0]);
    }
    auto model = make(1024, 0);
    server::Scheduler sched(*model, tok, 3, 64);
    auto seen = std::make_shared<Encodes>();
    seen->pause = std::chrono::milliseconds((long long)(1000 / server::kReplyTokensPerSecond) + 75);
    std::atomic<const server::Request*> replying{nullptr};
    std::atomic<size_t> reply_last{0};
    sched.on_retire = [&](const server::Scheduler::Retired& done) {
        const size_t n = ++seen->passes;
        if (std::find(done.requests.begin(), done.requests.end(), replying.load()) != done.requests.end()) reply_last = n;
    };
    std::thread runner;
    const auto submit = [&](const Req& r, server::Scheduler::Images images) {
        return sched.submit(r.prompt, params_of(r), std::numeric_limits<size_t>::max(), 0, server::Request::Clock::now(), {}, std::move(images));
    };
    try {
        const auto t = submit(text, images_of(text)), i = submit(img, stepped(img, 6, seen));
        replying = t.get();
        runner = std::thread([&] { sched.run(); });
        same(alone[0], drain(*t), "a reply beside an image's encode");
        same(alone[1], drain(*i), "a request whose image encodes in pieces beside a reply");
        require(seen->steps == 6 && seen->alive == 0, "an image's six pieces: " + std::to_string(seen->steps.load()) + " run, " + std::to_string(seen->alive.load()) + " encodes alive");
        std::map<size_t, size_t> between;
        for (const size_t n : seen->at)
            if (n < reply_last) ++between[n];
        for (const auto& [n, k] : between) require(k <= 2, std::to_string(k) + " of an image's pieces began beside a reply after " + std::to_string(n) + " passes retired, against two at most");

        auto slow = std::make_shared<Encodes>();
        slow->pause = std::chrono::milliseconds(2);
        const auto c = submit(img, stepped(img, 100000, slow));
        within_a_minute([&] { return slow->steps.load() >= 3; }, "a long encode's first pieces");
        c->cancel();
        within_a_minute([&] { return slow->alive.load() == 0; }, "a cancelled request's encode ended");
        collect(*c);
        require(c->finish() == "cancel", "a request cancelled mid-encode ended with " + c->finish());
        require(slow->steps < 100000, "a cancelled request's encode ran to its end");
        auto after = std::make_shared<Encodes>();
        same(alone[2], drain(*submit(next, stepped(next, 4, after))), "an image after a cancelled encode");
        require(after->steps == 4 && after->alive == 0, "the image after a cancelled encode: " + std::to_string(after->steps.load()) + " pieces");
    } catch (...) {
        sched.stop();
        if (runner.joinable()) runner.join();
        throw;
    }
    sched.stop();
    runner.join();
}

// A reply beside another request's image encode: on two devices whose second stage records on its own thread, the encode's pieces take no hold that stops the stages.
// Both replies are their replies alone; holds are counted, not times, and how many pieces a replies' pass meets is the price rule's (server-passes, encode_pieces).
void no_hold(const gguf::GGUFModel& weights, const bpe::Tokenizer& tok) {
    const Make make = on(weights, [] {
        std::vector<backend::BackendPtr> v;
        for (const auto& h : hooked(2, true)) v.push_back(h);
        return v;
    }, 3, 0);
    Req text;
    text.prompt = prompt_of(40, 30, kVocab);
    text.cap = 200;
    const Req img = with_image(3, 20, 4, 4, 10, 11, 12);
    std::vector<Reply> alone;
    for (const Req& r : {text, img}) {
        auto model = make(1024, 0);
        alone.push_back(serve(*model, tok, 3, {{r}}, nullptr, 2)[0]);
    }
    auto model = make(1024, 0);
    server::Scheduler sched(*model, tok, 3, 64, 2);
    auto seen = std::make_shared<Encodes>();
    std::thread runner([&] { sched.run(); });
    const auto submit = [&](const Req& r, server::Scheduler::Images images) {
        return sched.submit(r.prompt, params_of(r), std::numeric_limits<size_t>::max(), 0, server::Request::Clock::now(), {}, std::move(images));
    };
    try {
        const auto t = submit(text, images_of(text));
        server::Request::Token first;
        require(t->next(first, server::Request::Clock::now() + std::chrono::seconds(60)) == server::Request::Next::id, "a reply gave no token within a minute");
        const uint64_t holds = sched.stats().holds;
        const auto i = submit(img, stepped(img, 24, seen));
        Reply got{first};
        for (const auto& u : collect(*t)) got.push_back(u);
        same(alone[0], got, "a reply beside an image's encode on two devices");
        same(alone[1], drain(*i), "a request whose image encodes beside a reply on two devices");
        require(seen->steps == 24, "an image's 24 pieces: " + std::to_string(seen->steps.load()) + " run");
        const uint64_t held = sched.stats().holds - holds;
        require(held == 0, "an image's encode beside a reply stopped the stages " + std::to_string(held) + " times");
    } catch (...) {
        sched.stop();
        runner.join();
        throw;
    }
    sched.stop();
    runner.join();
}

// A request alone at its image runs its pieces in one round, which a health poll ends, so the live items show it in phase "encode": its image's rows and its pieces done rising to their total.
void encode_live(const gguf::GGUFModel& weights, const bpe::Tokenizer& tok) {
    const Make make = on(weights, [] { return cpus(1); }, 3, 0);
    const Req img = with_image(3, 20, 4, 4, 10, 11, 8);
    auto fresh = make(1024, 0);
    const Reply alone = serve(*fresh, tok, 3, {{img}})[0];
    auto model = make(1024, 0);
    server::Scheduler sched(*model, tok, 3, 64);
    auto seen = std::make_shared<Encodes>();
    seen->pause = std::chrono::milliseconds(5);
    std::thread runner([&] { sched.run(); });
    try {
        const auto h = sched.submit(img.prompt, params_of(img), std::numeric_limits<size_t>::max(), 0, server::Request::Clock::now(), {}, stepped(img, 400, seen));
        size_t first = 0, last = 0;
        uint64_t first_ms = 0, last_ms = 0;
        within_a_minute([&] {
            const server::Scheduler::Stats s = sched.stats();
            for (size_t i = 0; i < s.live.count; ++i) {
                const auto& v = s.live.items[i];
                if (std::string(v.phase) != "encode") continue;
                require(v.image_index == 1 && v.image_count == 1 && v.image_rows == 16 &&
                            (v.pieces_total ? v.pieces_total == 400 && v.pieces_done <= 400 : !v.pieces_done && !v.image_elapsed_ms),
                        "an encoding request's live item: image " + std::to_string(v.image_index) + " of " + std::to_string(v.image_count) + ", " + std::to_string(v.image_rows) +
                            " rows, " + std::to_string(v.pieces_done) + " of " + std::to_string(v.pieces_total) + " pieces in " + std::to_string(v.image_elapsed_ms) + " ms");
                if (!first) first = v.pieces_done, first_ms = v.image_elapsed_ms;
                last = v.pieces_done;
                last_ms = v.image_elapsed_ms;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            return first && last > first && last_ms > first_ms;
        }, "a lone request's encode in the live items, its pieces and its time rising");
        same(alone, drain(*h), "a request whose encode the live items showed");
    } catch (...) {
        sched.stop();
        runner.join();
        throw;
    }
    sched.stop();
    runner.join();
}

// A job reading its reply again on a fork inside its image without a digest, so no rows of it are kept (Cache::keep_tail), encodes it again in background turns.
// A request whose image arrives meanwhile takes the encoder from the job and gives its reply alone, and the job starts its image again and completes after it.
void job_gives_way(const gguf::GGUFModel& weights, const bpe::Tokenizer& tok) {
    const Make make = on(weights, [] { return cpus(1); }, 3, 3);
    Req first = with_image(5, 100, 6, 5, 40, 13, 20);
    first.images[0].digest = {};
    const Req next = with_image(4, 8, 3, 3, 5, 12, 12);
    auto fresh = make(1024, 0);
    const Reply alone = serve(*fresh, tok, 3, {{next}})[0];
    auto model = make(1024, 0);
    server::Scheduler sched(*model, tok, 3, 64);
    // The conversation's first encode is one piece, and each later one, the job's, 3000 pieces of a millisecond.
    auto job = std::make_shared<Encodes>();
    job->pause = std::chrono::milliseconds(1);
    server::Scheduler::Images images = stepped(first, 3000, job);
    auto calls = std::make_shared<std::atomic<size_t>>(0);
    images.encode = [whole = images_of(first).encode, slow = images.encode, calls](size_t i) { return (*calls)++ ? slow(i) : whole(i); };
    std::thread runner([&] { sched.run(); });
    const auto submit = [&](const Req& r, server::Scheduler::Images given) {
        return sched.submit(r.prompt, params_of(r), std::numeric_limits<size_t>::max(), 0, server::Request::Clock::now(), {}, std::move(given));
    };
    try {
        const auto h = submit(first, std::move(images));
        std::vector<uint32_t> ids = first.prompt;
        for (const auto& t : drain(*h)) ids.push_back(t.id);
        const std::vector<uint32_t> more = prompt_of(9, 80, kVocab);
        ids.insert(ids.end(), more.begin(), more.end());
        sched.follow(h, ids, true);
        within_a_minute([&] { return job->steps.load() >= 3; }, "a job's encode of its conversation's image");
        auto own = std::make_shared<Encodes>();
        const auto n = submit(next, stepped(next, 4, own));
        server::Request::Token t;
        require(n->next(t, server::Request::Clock::now() + std::chrono::seconds(60)) == server::Request::Next::id, "a request whose image arrived while a job encoded gave no token within a minute");
        Reply got{t};
        for (const auto& u : collect(*n)) got.push_back(u);
        require(n->finish() == "length", "a request whose image arrived while a job encoded ended with " + n->finish());
        same(alone, got, "a request whose image arrived while a job encoded");
        require(own->steps == 4, "the request's image beside a job's encode: " + std::to_string(own->steps.load()) + " pieces, against 4");
        within_a_minute([&] { return sched.stats().reprefills >= 1; }, "the job's read after it gave way");
        require(job->steps > 3000, "a job that gave way ran " + std::to_string(job->steps.load()) + " pieces, against more than one encode's 3000");
    } catch (...) {
        sched.stop();
        runner.join();
        throw;
    }
    sched.stop();
    runner.join();
}

}  // namespace

int main() {
    try {
        const gguf::GGUFModel weights = image_hybrid();
        const bpe::Tokenizer tok(weights);
        pieces(weights, tok);
        no_hold(weights, tok);
        job_gives_way(weights, tok);
        encode_live(weights, tok);
        follow_ups(weights, tok);
        reused_inside(weights, tok);
        reused_evicted(weights, tok);
        paused(weights, tok);
        host_tier(weights, tok);
        disk_tier(weights, tok);
        const gguf::GGUFModel with_mtp = image_hybrid(true);
        drafts(with_mtp, bpe::Tokenizer(with_mtp));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "image-cache: FAIL %s\n", e.what());
        return 1;
    }
    std::printf("image-cache: %zu checks passed\n", checks);
    return 0;
}
