// An image's rows in a prompt (docs/VISION.md, section 4) on a tiny qwen35 model built in memory: every position the rope reads against HF's get_rope_index written out, and the rows the same bits however the prompt is sliced or batched.
// The encoder's rows overwrite their pad rows whatever the pad id embeds, and a fork, a host copy and a retract of a history with images continue with its bits; the logits against HF are the suite's image component.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "inference/chat.hpp"
#include "model/arch/registry.hpp"
#include "model/runtime.hpp"
#include "tiny_qwen.hpp"
#include "tiny_qwen35.hpp"

namespace {

int checked = 0;

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
    ++checked;
}

void refuses(const std::string& what, const std::string& text, const std::function<void()>& run) {
    try {
        run();
    } catch (const std::exception& e) {
        require(std::string(e.what()).find(text) != std::string::npos, what + ": refused with \"" + e.what() + "\", not \"" + text + "\"");
        return;
    }
    throw std::runtime_error(what + ": not refused");
}

bool same(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && !std::memcmp(a.data(), b.data(), a.size() * sizeof(float));
}

// The tiny model with a rotary width of 6, its three pairs taking an image row's start, row and column in turn, as sections [1, 1, 1] interleave them, and a context of `context` rows.
gguf::GGUFModel model_file(uint64_t context = tiny35::CTX) {
    return tiny35::tiny([context](gguf::GGUFModel& m) {
        for (auto& kv : m.kv) {
            if (kv.first == "qwen35.context_length") kv.second.u = context;
            if (kv.first == "qwen35.rope.dimension_count") kv.second.u = 6;
            if (kv.first == "qwen35.rope.dimension_sections")
                for (size_t i = 0; i < kv.second.arr.size(); ++i) kv.second.arr[i].i = i < 3 ? 1 : 0;
        }
    });
}

// Every call of the rope: its rows, their positions three a row, a row of one stream written out thrice.
struct RopeProbe : backend::CpuBackend {
    std::vector<std::pair<size_t, std::vector<uint32_t>>> calls;
    void norm_rope_partial(backend::Slice dst, backend::CSlice src, size_t rows, size_t src_stride, size_t src_head_stride, size_t heads, size_t head_dim,
                           size_t rope_dim, backend::CSlice w, float eps, backend::CSlice cos, backend::CSlice sin, const uint32_t* pos,
                           backend::RopeStreams streams = {}) override {
        std::vector<uint32_t> three;
        for (size_t r = 0; r < rows; ++r)
            for (size_t k = 0; k < 3; ++k) three.push_back(pos[r * streams.count + (streams.count == 3 ? k : 0)]);
        calls.push_back({rows, std::move(three)});
        CpuBackend::norm_rope_partial(dst, src, rows, src_stride, src_head_stride, heads, head_dim, rope_dim, w, eps, cos, sin, pos, streams);
    }
};

// A prompt of text and images: a text of `n` tokens, or an image of a merged grid `h` by `w`.
struct Segment {
    size_t n = 0;
    uint32_t h = 0, w = 0;
};

// The positions HF's get_rope_index gives, written out: a text run counts on from the start, an image's row at (start, start + row, start + column), and the start then moves past the larger side of its grid.
std::vector<uint32_t> hf_positions(const std::vector<Segment>& segments, size_t generated) {
    std::vector<uint32_t> out;
    uint32_t st = 0;
    for (const Segment& s : segments) {
        if (s.n) {
            for (size_t k = 0; k < s.n; ++k) out.insert(out.end(), 3, uint32_t(st + k));
            st += uint32_t(s.n);
        } else {
            for (uint32_t r = 0; r < s.h; ++r)
                for (uint32_t c = 0; c < s.w; ++c) out.insert(out.end(), {st, st + r, st + c});
            st += std::max(s.h, s.w);
        }
    }
    for (size_t k = 0; k < generated; ++k) out.insert(out.end(), 3, uint32_t(st + k));
    return out;
}

// The prompt's ids with `pad` at every image row, and its images' runs, each row of an image a pattern of its index.
std::pair<std::vector<uint32_t>, std::vector<infer::MediaRun>> prompt(const std::vector<Segment>& segments, uint32_t pad) {
    std::vector<uint32_t> ids;
    std::vector<infer::MediaRun> runs;
    for (const Segment& s : segments) {
        if (s.n) {
            for (size_t k = 0; k < s.n; ++k) ids.push_back(uint32_t((ids.size() * 7 + 3) % tiny35::VOCAB));
            continue;
        }
        infer::MediaRun run;
        run.first = ids.size();
        run.rows = size_t(s.h) * s.w;
        run.grid_h = s.h;
        run.grid_w = s.w;
        std::vector<float> rows(run.rows * tiny35::E);
        for (size_t i = 0; i < rows.size(); ++i) rows[i] = float(int((i * 13 + runs.size() * 5) % 23) - 11) / 16.0f;
        run.encoded = std::make_shared<const std::vector<float>>(std::move(rows));
        ids.insert(ids.end(), run.rows, pad);
        runs.push_back(std::move(run));
    }
    return {ids, runs};
}

const std::vector<Segment> kPrompt = {{3, 0, 0}, {0, 2, 3}, {2, 0, 0}, {0, 3, 2}, {2, 0, 0}};
const uint32_t kSteps[] = {9, 14, 27};

// The prompt and three decode steps on a fresh model: every logits row in order, and the rope's calls.
std::vector<float> run(const gguf::GGUFModel& m, int ubatch, int threads, uint32_t pad, std::vector<std::pair<size_t, std::vector<uint32_t>>>* calls = nullptr) {
    auto probe = std::make_shared<RopeProbe>();
    probe->set_threads(threads);
    infer::ModelOptions options;
    options.dtype = backend::Dtype::f32;
    infer::Model model(infer::gguf_weights(m), probe, options);
    model.set_ubatch(ubatch);
    const auto [ids, runs] = prompt(kPrompt, pad);
    std::vector<float> out = model.prefill(ids, 0, runs);
    for (uint32_t t : kSteps) {
        const std::vector<float> next = step_row(model, int(t));
        out.insert(out.end(), next.begin(), next.end());
    }
    if (calls) *calls = probe->calls;
    return out;
}

// Every row's positions as HF places them, through passes of 16, 1 and 3 rows: the q and k calls of both full-attention layers of each pass read the same positions, and the passes' rows in order are the prompt's and the steps'.
// The logits are the same bits however the prompt is sliced and at 1, 4 and 16 threads, and whatever id the image rows hold, since their rows are written over.
void positions_and_slices() {
    const gguf::GGUFModel m = model_file();
    const std::vector<uint32_t> expected = hf_positions(kPrompt, std::size(kSteps));
    std::vector<float> first;
    for (int ubatch : {16, 1, 3}) {
        std::vector<std::pair<size_t, std::vector<uint32_t>>> calls;
        const std::vector<float> out = run(m, ubatch, 1, 5, &calls);
        require(calls.size() % 4 == 0, "the rope's calls are not four a pass");
        std::vector<uint32_t> got;
        for (size_t c = 0; c < calls.size(); c += 4) {
            for (size_t k = 1; k < 4; ++k) require(calls[c + k].second == calls[c].second, "a pass's rope calls read different positions");
            got.insert(got.end(), calls[c].second.begin(), calls[c].second.end());
        }
        require(got == expected, "positions at ubatch " + std::to_string(ubatch) + " are not HF's get_rope_index");
        if (first.empty()) first = out;
        require(same(out, first), "logits at ubatch " + std::to_string(ubatch) + " differ from one pass");
    }
    for (int threads : {4, 16}) require(same(run(m, 16, threads, 5), first), "logits at " + std::to_string(threads) + " threads differ");
    require(same(run(m, 16, 1, 9), first), "an image's rows depend on the pad id they are written over");
}

// A sequence with images beside a text-only one in one pass gives each alone's bits, and a history without images keeps its indices as positions.
void beside_another() {
    const gguf::GGUFModel m = model_file();
    const auto [ids, runs] = prompt(kPrompt, 5);
    const std::vector<uint32_t> text = {7, 3, 11, 30, 2, 19, 5, 8};
    auto probe = std::make_shared<RopeProbe>();
    infer::ModelOptions options;
    options.dtype = backend::Dtype::f32;
    options.state_slots = 2;
    options.kv_tokens = 2 * 128;
    infer::Model model(infer::gguf_weights(m), probe, options);
    infer::ExecContext ctx;
    std::vector<float> alone_a, alone_b;
    {
        infer::Sequence a = model.make_sequence();
        for (const infer::MediaRun& r : runs) model.add_media(a, r);
        const infer::BatchEntry e{&a, ids.data(), ids.size(), true};
        model.forward(ctx, &e, 1);
        alone_a.assign(ctx.logits(0), ctx.logits(0) + tiny35::VOCAB);
        model.reset(a);
    }
    probe->calls.clear();
    {
        infer::Sequence b = model.make_sequence();
        const infer::BatchEntry e{&b, text.data(), text.size(), true};
        model.forward(ctx, &e, 1);
        alone_b.assign(ctx.logits(0), ctx.logits(0) + tiny35::VOCAB);
        model.reset(b);
    }
    std::vector<uint32_t> indices;
    for (uint32_t i = 0; i < text.size(); ++i) indices.insert(indices.end(), 3, i);
    require(!probe->calls.empty() && probe->calls[0].second == indices, "a history without images is not at its indices");
    infer::Sequence a = model.make_sequence(), b = model.make_sequence();
    for (const infer::MediaRun& r : runs) model.add_media(a, r);
    const infer::BatchEntry both[] = {{&a, ids.data(), ids.size(), true}, {&b, text.data(), text.size(), true}};
    model.forward(ctx, both, 2);
    require(!std::memcmp(ctx.logits(0), alone_a.data(), alone_a.size() * sizeof(float)) &&
                !std::memcmp(ctx.logits(1), alone_b.data(), alone_b.size() * sizeof(float)),
            "a sequence with images beside a text-only one differs from each alone");
    model.reset(a);
    model.reset(b);
}

// An image the model or the history cannot take is refused by name, nothing left behind; a reset drops a sequence's images, so the same image may be given again.
void refusals() {
    const gguf::GGUFModel m = model_file();
    const auto [ids, runs] = prompt(kPrompt, 5);
    infer::ModelOptions options;
    options.dtype = backend::Dtype::f32;
    infer::Model model(infer::gguf_weights(m), backend::make_cpu_backend(), options);
    infer::MediaRun bad = runs[0];
    bad.grid_w = 2;
    refuses("a grid its rows do not fill", "do not fill its grid", [&] { model.prefill(ids, 0, {bad}); });
    bad = runs[0];
    bad.encoded = std::make_shared<const std::vector<float>>(bad.rows * tiny35::E - 1, 0.0f);
    refuses("rows of another width", "rows of the embedding's", [&] { model.prefill(ids, 0, {bad}); });
    bad = runs[0];
    bad.encoded.reset();
    refuses("an image to read given without rows", "given without its rows", [&] { model.prefill(ids, 0, {bad}); });
    refuses("images out of order", "an image before or over an image", [&] { model.prefill(ids, 0, {runs[1], runs[0]}); });
    bad = runs[0];
    bad.first = tiny35::CTX - 2;
    refuses("an image past the context", "context length exceeded", [&] { model.prefill(ids, 0, {bad}); });
    require(model.n_tokens() == 0, "a refused image left a history");
    infer::Sequence q = model.make_sequence();
    for (const infer::MediaRun& r : runs) model.add_media(q, r);
    bad = runs[1];
    bad.digest[0] = 1;
    refuses("another image given where the history holds one", "is not the image the history holds", [&] { model.add_media(q, bad); });
    model.reset(q);
    model.add_media(q, runs[0]);
    model.reset(q);
    // The file outlives the model, whose weights are views into it.
    const gguf::GGUFModel dense = tiny_qwen(2, 64, true);
    infer::Model text(infer::gguf_weights(dense));
    refuses("a model whose architecture places no images", "takes no image rows", [&] { text.prefill(ids, 0, runs); });
}

// A history with images through every way a history continues (docs/VISION.md, section 5), its checkpoint inside the first image of a context of 512 rows.
// Each continuation, given its images again, has the bits of the history never forked; rows dropped once read and not given again are refused.
const std::vector<Segment> kLong = {{100, 0, 0}, {0, 6, 5}, {10, 0, 0}, {0, 4, 4}, {20, 0, 0}};

std::vector<float> rows_after(infer::Model& model, infer::Sequence& s, const std::vector<uint32_t>& ids, size_t from) {
    infer::ExecContext ctx;
    const infer::BatchEntry e{&s, ids.data() + from, ids.size() - from, true};
    model.forward(ctx, &e, 1);
    std::vector<float> out(ctx.logits(0), ctx.logits(0) + tiny35::VOCAB);
    for (uint32_t t : kSteps) {
        const infer::BatchEntry step{&s, &t, 1, true};
        model.forward(ctx, &step, 1);
        out.insert(out.end(), ctx.logits(0), ctx.logits(0) + tiny35::VOCAB);
    }
    return out;
}

void histories() {
    const gguf::GGUFModel m = model_file(512);
    const auto [ids, runs] = prompt(kLong, 5);
    infer::ModelOptions options;
    options.dtype = backend::Dtype::f32;
    options.state_slots = 3;
    options.checkpoint_slots = 2;
    options.kv_tokens = 4 * 512;
    infer::Model model(infer::gguf_weights(m), backend::make_cpu_backend(), options);
    const size_t at = 128;
    require(runs[0].first < at && at < runs[0].first + runs[0].rows, "the checkpoint is not inside the first image");
    // The history never forked: the prompt to the checkpoint, kept there, then the rest and three steps.
    infer::Sequence a = model.make_sequence();
    for (const infer::MediaRun& r : runs) model.add_media(a, r);
    {
        infer::ExecContext ctx;
        infer::BatchEntry e{&a, ids.data(), at, false};
        e.keep = true;
        model.forward(ctx, &e, 1);
    }
    const std::vector<float> whole = rows_after(model, a, ids, at);

    // A fork at the checkpoint carries the images below it, without rows: the image it ends inside needs its rows given again, and another image there is refused.
    infer::Sequence f = model.fork(a, at);
    refuses("a fork reading an image's rows not given again", "were not given again", [&] { rows_after(model, f, ids, at); });
    require(f.length() == at, "a refused pass changed the fork's history");
    infer::MediaRun other = runs[0];
    other.digest[5] = 9;
    refuses("another image given to a fork", "is not the image the history holds", [&] { model.add_media(f, other); });
    for (const infer::MediaRun& r : runs) model.add_media(f, r);
    require(same(rows_after(model, f, ids, at), whole), "a fork inside an image differs from the history never forked");
    model.reset(f);

    // A host copy at the checkpoint and its restore.
    infer::HostHistory host;
    model.save_host(a, at, host, SIZE_MAX);
    infer::Sequence h = model.restore_host(host);
    model.wait_host(host);
    for (const infer::MediaRun& r : runs) model.add_media(h, r);
    require(same(rows_after(model, h, ids, at), whole), "a host copy restored inside an image differs from the history never forked");
    model.reset(h);
    model.release_host(host);

    // A state alone at the checkpoint and a fork with it, as a message boundary forks: it carries the images below it from the history it shares blocks with.
    infer::HostHistory state;
    model.save_host(a, at, state, SIZE_MAX, false);
    infer::Sequence b = model.fork(a, at, state);
    model.wait_host(state);
    for (const infer::MediaRun& r : runs) model.add_media(b, r);
    require(same(rows_after(model, b, ids, at), whole), "a fork with a state inside an image differs from the history never forked");
    model.reset(b);
    model.release_host(state);

    // A retract below the second image's last row reaches the checkpoint inside the first; its images given again, the history recomputes the same bits.
    const size_t reached = model.retract(a, runs[1].first + 3);
    require(reached == at, "a retract below the second image did not reach the checkpoint (" + std::to_string(reached) + ")");
    refuses("a recompute of an image read whole, its rows dropped", "were not given again", [&] { rows_after(model, a, ids, at); });
    for (const infer::MediaRun& r : runs) model.add_media(a, r);
    require(same(rows_after(model, a, ids, at), whole), "a retract below an image recomputed differs from the history first computed");
    model.reset(a);
}

// The chat renderer hands a message's parts to the template as HF passes them, and a message without parts renders as its content.
void parts() {
    const chat::ChatFormat format = [] {
        chat::ChatFormat f;
        f.program = std::make_shared<const chat::jj::Template>(
            "{% for m in messages %}{% if m.content is string %}[{{ m.content }}]{% else %}{% for p in m.content %}"
            "{% if p.type == 'image' %}<img>{% else %}({{ p.text }}){% endif %}{% endfor %}{% endif %}{% endfor %}");
        return f;
    }();
    chat::Message m{"user", "hello", std::nullopt};
    require(format.render({m}, false) == "[hello]", "a message without parts");
    m.parts = {{true, ""}, {true, ""}, {false, "hello"}};
    require(format.render({m}, false) == "<img><img>(hello)", "a message's parts");
}

}  // namespace

int main() {
    try {
        positions_and_slices();
        beside_another();
        refusals();
        histories();
        parts();
    } catch (const std::exception& e) {
        std::cerr << "image-prompt: FAIL " << e.what() << "\n";
        return 1;
    }
    std::cout << "image-prompt: " << checked << " checks passed\n";
    return 0;
}
