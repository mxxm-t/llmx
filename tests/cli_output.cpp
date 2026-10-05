// Include the real CLI so its glue is tested as it runs: removing the emitter's flush is observable without process-timing assumptions, and --device lists and cache types are read as the commands read them.
#include <atomic>
#define main llmx_cli_main
#include "../src/cli/main.cpp"
#undef main

class OutputBuffer : public std::stringbuf {
public:
    std::string delivered;
    int sync() override {
        delivered = str();
        return 0;
    }
};

// A --device list is canonical and each device appears once, however it is spelled; malformed entries are refused.
bool device_lists() {
    auto refused = [](const std::string& value) {
        try { backend::device_specs(value); } catch (const std::runtime_error&) { return true; }
        return false;
    };
    return backend::device_specs("vulkan,cpu,vulkan:01") == std::vector<std::string>{"vulkan:0", "cpu", "vulkan:1"} &&
           refused("vulkan,vulkan:0") && refused("vulkan:00,vulkan:0") && refused("cpu,cpu") && refused("vulkan:0,") &&
           refused("vulkan:x") && refused("gpu:0") && layer_shares("3,1") == std::vector<int>{3, 1} && refused("");
}

// A cache type is kept in its one spelling, and an empty or unknown name is a usage error as the flag is read, before any model file is.
bool cache_types() {
    auto read = [](std::string flag, std::string value, ExecOptions& exec) {
        char* argv[] = {flag.data(), value.data()};
        int i = 0;
        return exec_flag(2, argv, i, exec, false) && i == 1;
    };
    auto refused = [&](const std::string& flag, const std::string& value) {
        ExecOptions exec;
        try { read(flag, value, exec); } catch (const UsageError&) { return exec.cache_type_k.empty() && exec.cache_type_v.empty(); }
        return false;
    };
    ExecOptions exec;
    return exec.cache_type_k.empty() && exec.cache_type_v.empty() && read("-ctk", "f32", exec) && read("--cache-type-v", "f16", exec) &&
           exec.cache_type_k == "f32" && exec.cache_type_v == "f16" && refused("-ctk", "") && refused("-ctv", "") &&
           refused("--cache-type-k", "q8_0") && refused("--cache-type-v", "F16");
}

// A load mode is read by its name, and an empty or unknown one is a usage error as the flag is read, before any model file is.
bool load_modes() {
    auto read = [](std::string value, ExecOptions& exec) {
        std::string flag = "--load-mode";
        char* argv[] = {flag.data(), value.data()};
        int i = 0;
        return exec_flag(2, argv, i, exec, false) && i == 1;
    };
    auto refused = [&](const std::string& value) {
        ExecOptions exec;
        try { read(value, exec); } catch (const UsageError&) { return exec.load_mode == infer::LoadMode{}; }
        return false;
    };
    ExecOptions exec;
    return exec.load_mode == infer::LoadMode::automatic && read("mapped", exec) && exec.load_mode == infer::LoadMode::mapped &&
           read("direct", exec) && exec.load_mode == infer::LoadMode::direct && read("auto", exec) && exec.load_mode == infer::LoadMode::automatic &&
           refused("") && refused("mmap") && refused("Auto");
}

// An empty --layer-shares would read as no shares given, so it is refused as the flag is read, as a malformed list is.
bool empty_layer_shares() {
    std::string flag = "--layer-shares", empty;
    char* argv[] = {flag.data(), empty.data()};
    ExecOptions exec;
    int i = 0;
    try { exec_flag(2, argv, i, exec, false); } catch (const UsageError&) { return true; }
    return false;
}

// A second value for a flag is refused in either spelling, and so are --cpu-moe and --n-cpu-moe together, while a switch given again is taken; an argument without a dash is no flag, and spelling() gives the spelling a line used.
// long_spelling gives the flag each short form stands for, and any other argument as it is.
bool given_flags() {
    // Each flag of a line is taken as its branch would take it, with a value (true) or as a switch (false).
    auto refused = [](const std::vector<std::pair<std::string, bool>>& line) {
        GivenFlags given;
        try {
            for (const auto& [a, took_value] : line) given.take(a, took_value);
        } catch (const UsageError&) {
            return true;
        }
        return false;
    };
    GivenFlags given;
    for (const char* a : {"-tb", "text", "text", "--threads"}) given.take(a, a[0] == '-');
    return refused({{"--temp", true}, {"--temp", true}}) && refused({{"-n", true}, {"--max-tokens", true}}) &&
           refused({{"--threads-batch", true}, {"-tb", true}}) && refused({{"-c", true}, {"--ctx-size", true}}) &&
           refused({{"-ctk", true}, {"--cache-type-k", true}}) && refused({{"--cache-type-v", true}, {"-ctv", true}}) &&
           refused({{"--cpu-moe", false}, {"--n-cpu-moe", true}}) && refused({{"--n-cpu-moe", true}, {"--cpu-moe", false}}) &&
           !refused({{"--cpu-moe", false}, {"--cpu-moe", false}}) && !refused({{"--verbose", false}, {"--verbose", false}}) &&
           !refused({{"-n", true}, {"--n", true}}) && !refused({{"--cache-type-k", true}, {"--cache-type-v", true}}) &&
           given.spelling("--threads-batch") == "-tb" && given.spelling("--threads") == "--threads" && given.spelling("--max-tokens").empty() &&
           long_spelling("-n") == "--max-tokens" && long_spelling("-tb") == "--threads-batch" && long_spelling("-c") == "--ctx-size" &&
           long_spelling("-ctk") == "--cache-type-k" && long_spelling("-ctv") == "--cache-type-v" && long_spelling("--n") == "--n" &&
           long_spelling("text") == "text";
}

// pull's page prints --cache-dir's default from hub, where the cache root inside the home directory is kept.
bool pull_cache_default() {
    std::ostringstream page;
    return print_usage("pull", page) && page.str().find(std::string("Cache root (default: <home>/") + hub::cache_in_home + ")\n") != std::string::npos;
}

// --threads-batch and -tb are execution flags only where the command asks for them, as generate, chat and perplexity do; elsewhere they are not read, so the command refuses them as unknown.
bool batch_threads() {
    auto read = [](std::string flag, bool batch_threads, ExecOptions& exec) {
        std::string value = "3";
        char* argv[] = {flag.data(), value.data()};
        int i = 0;
        return exec_flag(2, argv, i, exec, batch_threads) && i == 1;
    };
    ExecOptions taken, left;
    return read("-tb", true, taken) && taken.threads_batch == 3 && read("--threads-batch", true, taken) &&
           !read("-tb", false, left) && !read("--threads-batch", false, left) && left.threads_batch == 0 &&
           read("--threads", false, left) && left.threads == 3 && left.threads_batch == 0;
}

// Runs `read` over the command line `flag value...` as a command's flag loop does, at the flag; the value it read, or nullopt when it refused the line as a usage error.
template <class Read>
auto read_flag(std::vector<std::string> line, Read read) -> std::optional<decltype(read(0, nullptr, std::declval<int&>()))> {
    std::vector<char*> argv;
    for (auto& s : line) argv.push_back(s.data());
    int i = 0;
    try {
        const auto v = read((int)argv.size(), argv.data(), i);
        if (i != 1) return std::nullopt;   // the reader must step past the value it read
        return v;
    } catch (const UsageError&) {
        return std::nullopt;
    }
}

// Every numeric flag reads a decimal number in its range and nothing else: a missing value, a sign, space, base prefix, fraction or trailing character where the form has none, and a value past the range or the type are refused.
bool number_readers() {
    const auto threads = [](int argc, char** argv, int& i) { return int_arg(argc, argv, i, "--threads", 0); };
    const auto port = [](int argc, char** argv, int& i) { return int_arg(argc, argv, i, "--port", 0, 65535); };
    const auto seed = [](int argc, char** argv, int& i) { return int_arg<uint64_t>(argc, argv, i, "--seed", 0); };
    // The sampling flags read the ranges of infer::Sampling, as generate and chat do.
    using infer::Sampling;
    const auto temp = [](int argc, char** argv, int& i) { return float_arg(argc, argv, i, "--temp", Sampling::temp_range.lo, Sampling::temp_range.hi); };
    const auto topk = [](int argc, char** argv, int& i) { return int_arg(argc, argv, i, "--topk", Sampling::top_k_range.lo, Sampling::top_k_range.hi); };
    const auto topp = [](int argc, char** argv, int& i) { return float_arg(argc, argv, i, "--topp", Sampling::top_p_range.lo, Sampling::top_p_range.hi); };
    const auto penalty = [](int argc, char** argv, int& i) {
        return float_arg(argc, argv, i, "--penalty", Sampling::penalty_range.lo, Sampling::penalty_range.hi);
    };
    bool ok = read_flag({"--threads", "0"}, threads) == 0 && read_flag({"--threads", "12"}, threads) == 12 &&
              read_flag({"--threads", "2147483647"}, threads) == 2147483647 &&
              read_flag({"--port", "0"}, port) == 0 && read_flag({"--port", "65535"}, port) == 65535 &&
              read_flag({"--seed", "18446744073709551615"}, seed) == UINT64_MAX && read_flag({"--seed", "010"}, seed) == 10 &&
              read_flag({"--temp", "0"}, temp) == 0.0f && read_flag({"--temp", "0.7"}, temp) == 0.7f &&
              read_flag({"--temp", "2.5e-1"}, temp) == 0.25f && read_flag({"--topk", "0"}, topk) == 0 &&
              read_flag({"--topp", "0"}, topp) == 0.0f && read_flag({"--topp", "1"}, topp) == 1.0f &&
              read_flag({"--penalty", "1"}, penalty) == 1.0f && read_flag({"--penalty", "1.3"}, penalty) == 1.3f;
    for (const char* bad : {"", "x", "4x", "4 ", " 4", "+4", "-1", "-0", "0x10", "1.5", "1e3", "2147483648", "99999999999999999999"})
        ok = ok && !read_flag({"--threads", bad}, threads);
    for (const char* bad : {"65536", "-1", "-0"}) ok = ok && !read_flag({"--port", bad}, port);
    for (const char* bad : {"-1", "0x10", "+1", "18446744073709551616"}) ok = ok && !read_flag({"--seed", bad}, seed);
    for (const char* bad : {"", "x", "0.5x", " 0.5", "+0.5", "-0.1", "nan", "inf", "-inf", "1e39", "0x1p-1", ".", "1e", "1,5"})
        ok = ok && !read_flag({"--temp", bad}, temp);
    ok = ok && !read_flag({"--topk", "-1"}, topk) && !read_flag({"--penalty", "0.99"}, penalty) && !read_flag({"--penalty", "0"}, penalty);
    for (const char* bad : {"1.5", "1.0001", "-0.5"}) ok = ok && !read_flag({"--topp", bad}, topp);
    // A length of time is seconds or a whole number with one unit, refused past what 64 bits of seconds hold.
    const auto age = [](int argc, char** argv, int& i) { return seconds_arg(argc, argv, i, "--disk-cache-max-age"); };
    ok = ok && read_flag({"--disk-cache-max-age", "0"}, age) == 0 && read_flag({"--disk-cache-max-age", "90"}, age) == 90 &&
         read_flag({"--disk-cache-max-age", "90s"}, age) == 90 && read_flag({"--disk-cache-max-age", "30m"}, age) == 1800 &&
         read_flag({"--disk-cache-max-age", "24h"}, age) == 86400 && read_flag({"--disk-cache-max-age", "7d"}, age) == 604800;
    for (const char* bad : {"", "h", "-1h", "+1h", "1.5h", "1hh", "1 h", "1w", "1H", "213503982334602d", "18446744073709551616"})
        ok = ok && !read_flag({"--disk-cache-max-age", bad}, age);
    // A flag at the end of the line has no value, for every reader.
    return ok && !read_flag({"--threads"}, threads) && !read_flag({"--temp"}, temp) && !read_flag({"--seed"}, seed) &&
           !read_flag({"--device"}, [](int argc, char** argv, int& i) { return flag_value(argc, argv, i, "--device"); });
}

// Token ids are separated by commas or any whitespace and are refused past the vocabulary before they are narrowed, so 2^32 does not become id 0.
bool token_id_lists() {
    const auto refused = [](const std::string& text) {
        try { token_ids(text, 10); } catch (const std::runtime_error&) { return true; }
        return false;
    };
    bool ok = token_ids("1,2 3\t4\n5\r\n6", 10) == std::vector<uint32_t>{1, 2, 3, 4, 5, 6} &&
              token_ids(" ,9,,0, ", 10) == std::vector<uint32_t>{9, 0} && token_ids("", 10).empty();
    for (const char* bad : {"10", "4294967296", "4294967301", "18446744073709551616", "1;2", "-1", "+1", "1.0", "0x1", "1a", "a"})
        ok = ok && refused(bad);
    return ok;
}

// The synthetic bench times the hot path alone: over a CPU backend that counts its allocations, nothing is allocated while the clock runs, so the history's storage, grown on a model's first steps, is not in the timed prefill.
struct CountingCpu : backend::CpuBackend {
    std::atomic<size_t> allocs{0};
    backend::BufferPtr alloc(size_t bytes, backend::Memory where) override {
        ++allocs;
        return backend::CpuBackend::alloc(bytes, where);
    }
};
bool bench_times_hot_path() {
    auto b = std::make_shared<CountingCpu>();
    b->set_threads(1);
    const gguf::GGUFModel sm = infer::synthetic_model({2, 256, 1024, 8, 2, 32, 512, 12345u});
    infer::Model model(infer::gguf_weights(sm), b);
    size_t before = 0;
    time_steps(model, 64, 64, 512, [&] { before = b->allocs.load(); });
    return b->allocs.load() == before;
}

int main() {
    OutputBuffer output;
    auto* saved = std::cout.rdbuf(&output);
    emit_text("A\xc3");
    const bool first = output.delivered == "A\xc3";
    emit_text("\xa9Z");
    const bool second = output.delivered == "A\xc3\xa9Z";
    std::cout.rdbuf(saved);
    if (!first || !second) {
        std::cerr << "CLI did not flush each byte chunk\n";
        return 1;
    }
    if (!device_lists()) {
        std::cerr << "CLI device lists not read canonically\n";
        return 1;
    }
    if (!cache_types()) {
        std::cerr << "CLI cache types not read as given or not refused as the flag is read\n";
        return 1;
    }
    if (!load_modes()) {
        std::cerr << "CLI load modes not read by name or not refused as the flag is read\n";
        return 1;
    }
    if (!empty_layer_shares()) {
        std::cerr << "CLI --layer-shares with an empty list read as no shares given\n";
        return 1;
    }
    if (!given_flags()) {
        std::cerr << "CLI second value for a flag not refused in one spelling or two, a switch given again refused, a line's spelling of a flag not kept, or a short form not read as its long one\n";
        return 1;
    }
    if (!pull_cache_default()) {
        std::cerr << "CLI pull help does not print hub's cache root as --cache-dir's default\n";
        return 1;
    }
    if (!batch_threads()) {
        std::cerr << "CLI --threads-batch read where the command does not ask for it, or not read where it does\n";
        return 1;
    }
    if (!number_readers()) {
        std::cerr << "CLI number flags accept a value outside their form or range\n";
        return 1;
    }
    if (!bench_times_hot_path()) {
        std::cerr << "the synthetic bench's timed steps allocate, so they time the model's first-run setup\n";
        return 1;
    }
    if (!token_id_lists()) {
        std::cerr << "CLI token id lists not read as comma or whitespace separated ids within the vocabulary\n";
        return 1;
    }
    std::cout << "CLI output: each byte chunk flushed immediately; device lists canonical; cache types, load modes and an empty share list refused as read; a second value for a flag refused and a switch given again taken; pull's cache default from hub; -tb read only where asked; numbers and token ids read strictly; the bench's timed steps allocate nothing\n";
    return 0;
}
