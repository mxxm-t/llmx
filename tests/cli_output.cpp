// Include the real CLI so its glue is tested as it runs: removing the emitter's flush is observable without process-timing assumptions, and --device lists and cache types are read as the commands read them.
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
    auto read = [](std::string flag, std::string value, infer::GenParams& gp) {
        char* argv[] = {flag.data(), value.data()};
        int i = 0;
        return exec_flag(2, argv, i, gp) && i == 1;
    };
    auto refused = [&](const std::string& flag, const std::string& value) {
        infer::GenParams gp;
        try { read(flag, value, gp); } catch (const UsageError&) { return gp.cache_type_k.empty() && gp.cache_type_v.empty(); }
        return false;
    };
    infer::GenParams gp;
    return gp.cache_type_k.empty() && gp.cache_type_v.empty() && read("-ctk", "f32", gp) && read("--cache-type-v", "f16", gp) &&
           gp.cache_type_k == "f32" && gp.cache_type_v == "f16" && refused("-ctk", "") && refused("-ctv", "") &&
           refused("--cache-type-k", "q8_0") && refused("--cache-type-v", "F16");
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
    // The sampling flags read the sampler's ranges, as generate and chat do.
    const auto temp = [](int argc, char** argv, int& i) { return float_arg(argc, argv, i, "--temp", infer::kTempRange.lo, infer::kTempRange.hi); };
    const auto topk = [](int argc, char** argv, int& i) { return int_arg(argc, argv, i, "--topk", infer::kTopKRange.lo, infer::kTopKRange.hi); };
    const auto topp = [](int argc, char** argv, int& i) { return float_arg(argc, argv, i, "--topp", infer::kTopPRange.lo, infer::kTopPRange.hi); };
    const auto penalty = [](int argc, char** argv, int& i) {
        return float_arg(argc, argv, i, "--penalty", infer::kPenaltyRange.lo, infer::kPenaltyRange.hi);
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
    if (!number_readers()) {
        std::cerr << "CLI number flags accept a value outside their form or range\n";
        return 1;
    }
    if (!token_id_lists()) {
        std::cerr << "CLI token id lists not read as comma or whitespace separated ids within the vocabulary\n";
        return 1;
    }
    std::cout << "CLI output: each byte chunk flushed immediately; device lists canonical; cache types refused as read; numbers and token ids read strictly\n";
    return 0;
}
