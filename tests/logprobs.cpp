// infer::log_sum_exp, logprob and top_logprobs against a double-precision reference computed another way, and the server's token channel carrying them.
// The reference shifts by the row's maximum and sums with compensation, so a float log-probability must be within half a float step of it, plus the reference's own error.
// The channel's values must be the log-softmax of the very logits row the scheduler sampled from, which a second model run through the same passes gives, whether the reader keeps up or falls behind.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "inference/logprobs.hpp"
#include "inference/perplexity.hpp"
#include "server/scheduler.hpp"
#include "tiny_qwen.hpp"

namespace {

size_t checks = 0;

void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
    ++checks;
}

// Every entry's log-softmax in double: the row's maximum subtracted, exp summed with Neumaier's compensation, the log of the sum added back.
std::vector<double> reference(const std::vector<float>& row, double& lse) {
    double m = -std::numeric_limits<double>::infinity();
    for (float x : row) m = std::max(m, (double)x);
    double sum = 0.0, c = 0.0;
    for (float x : row) {
        const double t = std::exp((double)x - m), y = sum + t;
        c += std::fabs(sum) >= std::fabs(t) ? (sum - y) + t : (t - y) + sum;
        sum = y;
    }
    lse = m + std::log(sum + c);
    std::vector<double> out(row.size());
    for (size_t i = 0; i < row.size(); ++i) out[i] = (double)row[i] - lse;
    return out;
}

// A float log-probability is the reference rounded once: within half a float step of it, beside the reference's own error, which grows with the row's log-sum-exp.
void rounded(float got, double want, double lse, const std::string& what) {
    const float a = std::fabs(got);
    const double half = 0.5 * ((double)std::nextafter(a, std::numeric_limits<float>::infinity()) - a);
    require(std::fabs((double)got - want) <= half + 1e-12 * (1.0 + std::fabs(want) + std::fabs(lse)),
            what + ": " + std::to_string(got) + " against " + std::to_string(want));
}

// The ids of a row by logit, largest first and the lower id first on a tie, which is greedy's order.
std::vector<uint32_t> ranked(const std::vector<float>& row) {
    std::vector<uint32_t> ids(row.size());
    for (uint32_t i = 0; i < (uint32_t)ids.size(); ++i) ids[i] = i;
    std::stable_sort(ids.begin(), ids.end(), [&](uint32_t a, uint32_t b) { return row[a] > row[b]; });
    return ids;
}

// A row checked whole: every log-probability against the reference, token_nll as the same quantity in double, the probabilities summing to one, and the top k for several k.
void row_matches(const std::vector<float>& row, const std::string& what) {
    double ref_lse = 0.0;
    const std::vector<double> want = reference(row, ref_lse);
    const double lse = infer::log_sum_exp(row.data(), row.size());
    require(std::fabs(lse - ref_lse) <= 1e-12 * (1.0 + std::fabs(ref_lse)), what + ": log-sum-exp");
    double total = 0.0;
    for (uint32_t i = 0; i < (uint32_t)row.size(); ++i) {
        const float lp = infer::logprob(row.data(), lse, i);
        rounded(lp, want[i], ref_lse, what + " id " + std::to_string(i));
        total += std::exp((double)lp);
    }
    require(std::fabs(total - 1.0) <= 1e-5, what + ": the probabilities sum to " + std::to_string(total));
    const uint32_t target = (uint32_t)(row.size() / 3);
    require(infer::token_nll(row.data(), row.size(), target) == -((double)row[target] - lse), what + ": token_nll is not the logprob in double");
    const std::vector<uint32_t> order = ranked(row);
    for (size_t k : {size_t(0), size_t(1), size_t(5), size_t(20), row.size(), row.size() + 3}) {
        const std::vector<infer::TokenLogprob> top = infer::top_logprobs(row.data(), row.size(), lse, k);
        require(top.size() == std::min(k, row.size()), what + ": top " + std::to_string(k) + " has " + std::to_string(top.size()));
        for (size_t j = 0; j < top.size(); ++j) {
            require(top[j].id == order[j], what + ": top " + std::to_string(k) + " entry " + std::to_string(j) + " is id " + std::to_string(top[j].id));
            require(top[j].logprob == infer::logprob(row.data(), lse, top[j].id), what + ": a top entry's value");
        }
    }
}

void rows() {
    std::mt19937 rng(20260925);
    // The vocabulary of the Qwen3 models, with logits spread as a model's are.
    for (float spread : {1.0f, 3.0f, 12.0f}) {
        std::normal_distribution<float> logit(0.0f, spread);
        std::vector<float> row(151936);
        for (float& x : row) x = logit(rng);
        row_matches(row, "a 151936-token row spread " + std::to_string(spread));
    }
    // One token certain to within float rounding, and the rest a hundred logits below.
    std::vector<float> sure(1000, -100.0f);
    sure[7] = 0.0f;
    row_matches(sure, "one likely token");
    // Logits too large for exp without the shift.
    std::vector<float> large = {5000.0f, 4999.0f, 4990.0f, -5000.0f, 4999.5f};
    row_matches(large, "large logits");
    // Equal logits, where every value is -log n and the top lists ids in order.
    std::vector<float> flat(10, 2.5f);
    row_matches(flat, "equal logits");
    require(infer::logprob(flat.data(), infer::log_sum_exp(flat.data(), flat.size()), 3) == (float)-std::log(10.0), "equal logits: -log 10");
    // Ties at the top and in the middle, broken by id.
    row_matches({1.0f, 3.0f, 2.0f, 3.0f, 2.0f, 3.0f, -1.0f}, "tied logits");
    // One token, certain.
    const std::vector<float> one = {-7.25f};
    require(infer::logprob(one.data(), infer::log_sum_exp(one.data(), 1), 0) == 0.0f, "a one-token row is certain");
    // A row wholly below -1e30, which still shifts by its own maximum.
    row_matches({-2.0e38f, -3.0e38f, -2.5e38f, -3.3e38f}, "logits below -1e30");
    const double none = infer::log_sum_exp(nullptr, 0);
    require(std::isinf(none) && none < 0, "an empty row's log-sum-exp is minus infinity");
}

// tiny_qwen's 16 tokens named for the tokenizer, and no end token, so a reply runs to its cap.
gguf::GGUFModel served() {
    gguf::GGUFModel m = tiny_qwen(2, 4 * 128, false);
    gguf::MetaValue tokens;
    tokens.vtype = gguf::V_ARRAY;
    tokens.u = gguf::V_STRING;
    for (int i = 0; i < 16; ++i) {
        gguf::MetaValue t;
        t.vtype = gguf::V_STRING;
        t.s = std::string(1, char('a' + i));
        tokens.arr.push_back(t);
    }
    m.kv.push_back({"tokenizer.ggml.tokens", tokens});
    return m;
}

// Waits for a request to end with its channel unread.
void await_end(const server::Request& r) {
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (r.finish().empty()) {
        require(std::chrono::steady_clock::now() < until, "a request left unread did not end in 30 seconds");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

std::vector<server::Request::Token> drain(server::Request& r) {
    std::vector<server::Request::Token> out;
    server::Request::Token t;
    for (;;) {
        const auto got = r.next(t, server::Request::Clock::now() + std::chrono::seconds(30));
        if (got == server::Request::Next::end) break;
        require(got == server::Request::Next::id, "the channel gave nothing for 30 seconds");
        out.push_back(t);
    }
    require(r.finish() == "length", "the request ended with " + r.finish());
    return out;
}

// A greedy request with logprobs through the scheduler, read from its channel, against a second model run through the passes the scheduler runs for a request alone: its prompt as one entry at the prompt's extent, then each token as a decode entry.
// Every id must be the row's greedy choice, its logprob and top five that row's log-softmax; the same request without logprobs gets the same ids and no values.
// A request left unread until it ends holds kRowsWaiting rows and gets the same values, the scheduler computing those past them; cancelled, its waiting rows are dropped without values.
void channel() {
    const gguf::GGUFModel weights = served();
    const bpe::Tokenizer tok(weights);
    auto cpu = std::make_shared<backend::CpuBackend>();
    cpu->set_threads(1);
    const std::vector<uint32_t> prompt = {3, 1, 4, 1, 5, 9, 2, 6};
    server::SampleParams asked;
    asked.max_tokens = 24;
    asked.temp = 0.0f;
    asked.logprobs = true;
    asked.top_logprobs = 5;
    server::SampleParams plain = asked;
    plain.logprobs = false;
    plain.top_logprobs = 0;

    std::vector<server::Request::Token> with, without, behind, cancelled;
    constexpr size_t waiting = server::Request::kRowsWaiting;
    require((size_t)asked.max_tokens > waiting, "the reply must outrun the rows a channel holds");
    {
        infer::Model model(weights, cpu);
        server::Scheduler sched(model, tok, 2, 4);
        std::thread runner([&] { sched.run(); });
        try {
            with = drain(*sched.submit(prompt, asked));
            without = drain(*sched.submit(prompt, plain));
            auto late = sched.submit(prompt, asked);
            await_end(*late);
            require(late->rows_waiting() == waiting, "a reader behind holds " + std::to_string(late->rows_waiting()) + " rows");
            behind = drain(*late);
            require(late->rows_waiting() == 0, "rows left once the channel is read");
            auto gone = sched.submit(prompt, asked);
            await_end(*gone);
            gone->cancel();
            cancelled = drain(*gone);
        } catch (...) {
            sched.stop();
            runner.join();
            throw;
        }
        sched.stop();
        runner.join();
    }
    require(with.size() == (size_t)asked.max_tokens && without.size() == with.size() && behind.size() == with.size() && cancelled.size() == with.size(),
            "a reply's length");
    for (size_t i = 0; i < with.size(); ++i) {
        const std::string at = "token " + std::to_string(i);
        require(behind[i].id == with[i].id && behind[i].logprob == with[i].logprob && behind[i].top.size() == with[i].top.size(),
                at + ": a reader behind gets other values");
        for (size_t j = 0; j < with[i].top.size(); ++j)
            require(behind[i].top[j].id == with[i].top[j].id && behind[i].top[j].logprob == with[i].top[j].logprob, at + ": a reader behind gets another top entry");
        // The rows that waited are dropped, and the tokens past them came with the scheduler's values.
        require(cancelled[i].id == with[i].id, at + ": a cancelled request's id");
        if (i < waiting) require(cancelled[i].top.empty() && cancelled[i].logprob == 0.0f, at + ": a cancelled request's waiting row was read");
        else require(cancelled[i].logprob == with[i].logprob && cancelled[i].top.size() == with[i].top.size(), at + ": values the scheduler computed");
    }

    infer::Model control(weights, cpu);
    infer::Sequence seq = control.make_sequence();
    infer::ExecContext ctx;
    infer::BatchEntry first{&seq, prompt.data(), prompt.size(), true};
    first.extent = prompt.size();
    control.forward(ctx, &first, 1);
    for (size_t i = 0; i < with.size(); ++i) {
        const std::string at = "token " + std::to_string(i);
        const std::vector<float> row(ctx.logits(0), ctx.logits(0) + ctx.width);
        const uint32_t id = ranked(row)[0];
        require(with[i].id == id && without[i].id == id, at + ": not the row's greedy id");
        double ref_lse = 0.0;
        const std::vector<double> want = reference(row, ref_lse);
        const double lse = infer::log_sum_exp(row.data(), row.size());
        require(with[i].logprob == infer::logprob(row.data(), lse, id), at + ": the channel's logprob is not its row's");
        rounded(with[i].logprob, want[id], ref_lse, at);
        const std::vector<infer::TokenLogprob> top = infer::top_logprobs(row.data(), row.size(), lse, 5);
        require(with[i].top.size() == 5, at + ": top 5");
        for (size_t j = 0; j < top.size(); ++j)
            require(with[i].top[j].id == top[j].id && with[i].top[j].logprob == top[j].logprob, at + ": a top entry is not its row's");
        require(with[i].top[0].logprob == with[i].logprob, at + ": greedy's logprob is not the largest");
        require(without[i].top.empty() && without[i].logprob == 0.0f, at + ": values for a request that did not ask");
        infer::BatchEntry next{&seq, &id, 1, true};
        control.forward(ctx, &next, 1);
    }
}

} // namespace

int main() {
    try {
        rows();
        channel();
        std::cout << "logprobs: " << checks << " checks pass\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "logprobs: " << error.what() << '\n';
        return 1;
    }
}
