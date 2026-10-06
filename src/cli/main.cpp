#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <charconv>
#include <functional>
#include <limits>
#include <optional>

#if defined(_WIN32)
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>
#include <io.h>
#pragma comment(lib, "shell32.lib")
#else
#include <unistd.h>
#endif

#include "config.hpp"
#if LLMX_HAS_BACKEND_VULKAN
#include "backends/vulkan/vulkan_backend.hpp"
#endif
#include "core/list.hpp"
#include "backends/devices.hpp"
#include "hub/pull.hpp"
#include "format/gguf.hpp"
#include "format/raw_convert.hpp"
#include "quant/quant.hpp"
#include "tokenizer/tokenizer.hpp"
#include "inference/sampler.hpp"
#include "inference/generate.hpp"
#include "inference/logprobs.hpp"
#include "inference/perplexity.hpp"
#include "inference/chat.hpp"
#include "inference/load.hpp"
#include "model/runtime.hpp"
#include "model/arch/registry.hpp"
#include "server/api.hpp"

// CLI argument parsing and dispatch; format, quantization, inference and model logic stay in their own layers.

namespace {

// Defaults only the CLI owns, which each command's parser starts from and its help prints.
constexpr int kLogitsTop = 10;                                      // logits --top
constexpr const char* kChatSystem = "You are a helpful assistant.";  // chat --system
constexpr const char* kQuantType = "q8_0";                          // quantize without a type

// bench's numbers as its parser starts them, which its help prints as the defaults.
struct BenchNumbers {
    int size = 1024;    // the synthetic matrix width
    int iters = 5;      // the synthetic kernels' repetitions
    int prompt = 64;    // prompt tokens
    int decode = 64;    // decode tokens
    int repeats = 3;    // a model run's repetitions
    int seqs = 1;       // sequences decoding together
    int depth = 0;      // tokens of history each test runs after
};

// The execution flags, the "Execution options" of a model command's help, and --verbose: where the model runs, its workers, its prompt batch and its caches.
// exec_flag fills them, and a flag the command line does not give keeps the default here, which the help prints.
struct ExecOptions {
    std::string device = "cpu";   // cpu, or vulkan:N when built with it; several, comma separated, split the model by layers over them
    std::optional<backend::Dtype> dtype; // empty selects auto
    std::string layer_shares;     // with several devices, their proportions of the layers, comma separated; empty fits them to the devices' free memory
    int tensor_width = 1;         // devices each layer is split across: the listed devices form groups of this many, the stages of the layer split (docs/TENSOR-SPLIT.md)
    int threads = 0;              // CPU workers, decode's where a command tells the phases apart; 0 selects automatically
    int threads_batch = 0;        // CPU workers for a prompt's batched passes; 0 takes the decode count
    int ubatch = 0;               // prompt tokens a pass takes; 0 is infer::kDefaultUbatch
    std::string cache_type_k;     // each cache side's type as its flag gives it; empty keeps the model's default (ModelOptions)
    std::string cache_type_v;
    int kv_tokens = 0;            // the KV pool's total token budget, serve's --ctx-size; 0 is the model context
    int cpu_moe = 0;              // routed layers whose experts run on the CPU beside a device: the first N, -1 all
    int moe_stream_from = 0;      // prompt length from which those experts are copied to the device for a pass; 0 never
    infer::LoadMode load_mode{};  // how weights are read; the default is the loader's first mode
    bool verbose = false;         // the prompt token count, the thread counts, a split's plan, progress, generate's KV line and perplexity's per-window lines
};

// A command line the command cannot take: main prints the command's page on stderr and exits with status 2.
struct UsageError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// The value after `flag`, which a flag at the end of the line lacks.
std::string flag_value(int argc, char** argv, int& i, const std::string& flag) {
    if (i + 1 >= argc) throw UsageError(flag + " needs a value");
    return argv[++i];
}

// The value after a flag whose empty value would read as the flag not given; `what` names what the flag needs.
std::string nonempty_value(int argc, char** argv, int& i, const std::string& flag, const std::string& what) {
    std::string value = flag_value(argc, argv, i, flag);
    if (value.empty()) throw UsageError(flag + " needs " + what);
    return value;
}

// Each short spelling with the long one it stands for; the parsers compare long_spelling(arg), so a pair is written only here.
constexpr std::pair<std::string_view, std::string_view> kShortForms[] = {
    {"-n", "--max-tokens"}, {"-tb", "--threads-batch"}, {"-c", "--ctx-size"}, {"-ctk", "--cache-type-k"}, {"-ctv", "--cache-type-v"}};

// The long spelling of the flag `arg` gives: the one a short form stands for, or `arg` itself.
std::string_view long_spelling(std::string_view arg) {
    for (const auto& [short_form, long_form] : kShortForms)
        if (arg == short_form) return long_form;
    return arg;
}

// The flags one command line gives, so a second value for a setting, in one spelling or two, is refused rather than overwriting the first.
class GivenFlags {
public:
    // Records `arg` once its branch has read it, `took_value` when the branch read a value after it; an argument without a dash is no flag.
    // A setting the line gave before is a usage error, unless this is the same switch again, which changes nothing.
    void take(const std::string& arg, bool took_value) {
        if (arg.empty() || arg[0] != '-') return;
        const std::string_view name = setting(arg);
        for (const auto& [earlier_name, earlier] : given_) {
            if (earlier_name != name) continue;
            if (!took_value && earlier == arg) return;
            throw UsageError(earlier == arg ? arg + " is given twice" : earlier + " and " + arg + " set the same thing; give one of them");
        }
        given_.emplace_back(std::string(name), arg);
    }
    // The spelling the line gave `flag`'s setting in, or empty when it gave none; `flag` is the long spelling.
    std::string spelling(std::string_view flag) const {
        for (const auto& [name, arg] : given_)
            if (name == flag) return arg;
        return {};
    }

private:
    // --cpu-moe is --n-cpu-moe for every routed layer, so the two set one setting.
    static std::string_view setting(std::string_view arg) {
        const std::string_view flag = long_spelling(arg);
        return flag == "--cpu-moe" ? "--n-cpu-moe" : flag;
    }
    std::vector<std::pair<std::string, std::string>> given_;   // each setting's flag and the spelling that gave it
};

template <class T>
std::string range_text(T lo, T hi) {
    std::ostringstream s;
    if (hi == std::numeric_limits<T>::max()) s << "of at least " << lo;
    else s << "from " << lo << " to " << hi;
    return s.str();
}

// A decimal whole number from `lo` to `hi`, digits and nothing else: no space, sign, base prefix or fraction.
template <class T>
T whole_number(const std::string& text, const std::string& what, T lo, T hi) {
    T v{};
    const char* end = text.data() + text.size();
    const bool digits = !text.empty() && text.find_first_not_of("0123456789") == std::string::npos;
    const auto parsed = std::from_chars(text.data(), end, v);
    if (!digits || parsed.ec != std::errc() || parsed.ptr != end || v < lo || v > hi)
        throw UsageError(what + ": '" + text + "' is not a whole number " + range_text(lo, hi));
    return v;
}

// The whole number after `flag`, from `lo` to `hi`.
template <class T = int>
T int_arg(int argc, char** argv, int& i, const std::string& flag, T lo, T hi = std::numeric_limits<T>::max()) {
    return whole_number<T>(flag_value(argc, argv, i, flag), flag, lo, hi);
}

// The length of time after `flag`, in seconds: a whole number alone or followed by s, m, h or d.
uint64_t seconds_arg(int argc, char** argv, int& i, const std::string& flag) {
    const std::string text = flag_value(argc, argv, i, flag);
    const char last = text.empty() ? '\0' : text.back();
    const uint64_t unit = last == 'm' ? 60 : last == 'h' ? 3600 : last == 'd' ? 86400 : 1;
    const bool suffix = last == 's' || unit > 1;
    try {
        return whole_number<uint64_t>(suffix ? text.substr(0, text.size() - 1) : text, flag, 0, std::numeric_limits<uint64_t>::max() / unit) * unit;
    } catch (const UsageError&) {
        throw UsageError(flag + ": '" + text + "' is not a length of time, a whole number of seconds or one followed by s, m, h or d");
    }
}

// The decimal number after `flag`, from `lo` to `hi`: digits, a point and an exponent, so no infinity, NaN or hexadecimal form.
float float_arg(int argc, char** argv, int& i, const std::string& flag, float lo, float hi = std::numeric_limits<float>::max()) {
    const std::string text = flag_value(argc, argv, i, flag);
    const bool decimal = !text.empty() && text[0] != '+' && text.find_first_not_of("0123456789.eE+-") == std::string::npos;
    char* end = nullptr;
    const float v = decimal ? std::strtof(text.c_str(), &end) : 0.0f;
    if (!decimal || end != text.c_str() + text.size() || !(v >= lo && v <= hi))
        throw UsageError(flag + ": '" + text + "' is not a number " + range_text(lo, hi));
    return v;
}

// The cache type after `flag` in its one spelling, checked as it is read, so an empty or unknown name is refused before any model file is read.
std::string cache_type_arg(int argc, char** argv, int& i, const std::string& flag) {
    const std::string name = flag_value(argc, argv, i, flag);
    try {
        return backend::kv_type_name(backend::kv_type_of(name));
    } catch (const std::runtime_error& e) {
        throw UsageError(flag + ": " + e.what());
    }
}

// Token ids separated by commas or whitespace, each checked against the vocabulary while it is still 64 bits wide, so an id past 2^32 is not narrowed into it.
std::vector<uint32_t> token_ids(const std::string& text, size_t vocab_size) {
    static const char separators[] = ", \t\n\v\f\r";
    std::vector<uint32_t> ids;
    for (size_t i = text.find_first_not_of(separators); i != std::string::npos; i = text.find_first_not_of(separators, i)) {
        const size_t end = std::min(text.find_first_of(separators, i), text.size());
        const std::string item = text.substr(i, end - i);
        if (item.find_first_not_of("0123456789") != std::string::npos)
            throw std::runtime_error("'" + item + "' is not a token id");
        uint64_t id = 0;
        if (std::from_chars(item.data(), item.data() + item.size(), id).ec != std::errc() || id >= vocab_size)
            throw std::runtime_error("token id " + item + " is outside the vocabulary of " + std::to_string(vocab_size) + " tokens");
        ids.push_back((uint32_t)id);
        i = end;
    }
    return ids;
}

// The bytes of a text file as they are; `command` names the refusal.
std::string read_text_file(const std::string& path, const std::string& command) {
    std::ifstream input(std::filesystem::u8path(path), std::ios::binary);
    if (!input) throw std::runtime_error(command + ": cannot open file: " + path);
    std::string text;
    char buffer[8192];
    while (input.read(buffer, sizeof(buffer)) || input.gcount())
        text.append(buffer, (size_t)input.gcount());
    if (!input.eof()) throw std::runtime_error(command + ": cannot read file: " + path);
    return text;
}

// logits and perplexity take their text right after the model, inline or from the file `--file` or `-f` names; returns the first flag's index, 5 for a file.
int text_arg(int argc, char** argv) {
    if (argc < 4) throw UsageError("missing the model or the text");
    const std::string a = argv[3];
    if (a != "--file" && a != "-f") return 4;
    if (argc < 5) throw UsageError(a + " needs a path");
    return 5;
}

// A file among the flags comes after the text, so it would be a second one.
void no_second_text(const std::string& a) {
    if (a == "--file" || a == "-f") throw UsageError(a + " goes right after the model, in place of the text");
}

bool show_progress(const ExecOptions& exec) {
#if defined(_WIN32)
    return exec.verbose || _isatty(_fileno(stderr));
#else
    return exec.verbose || isatty(fileno(stderr));
#endif
}

infer::ModelOptions model_options(const ExecOptions& exec) {
    infer::ModelOptions o;
    if (!exec.cache_type_k.empty()) o.kv_k = backend::kv_type_of(exec.cache_type_k);
    if (!exec.cache_type_v.empty()) o.kv_v = backend::kv_type_of(exec.cache_type_v);
    o.kv_tokens = exec.kv_tokens > 0 ? (size_t)exec.kv_tokens : 0;
    return o;
}

// The loading progress on stderr: the share of the payload read, then "Preparing model..." once it is complete, while the rest of the load runs (the placement too in mapped mode, which reads first).
// A second report of completion prints nothing.
format::LoadProgress progress_bar() {
    return [previous = -1, finished = false](size_t completed, size_t total) mutable {
        const int percent = total ? int(100.0 * double(completed) / double(total)) : 100;
        if (percent != previous) {
            previous = percent;
            std::cerr << "\rLoading tensor data: " << percent << "%" << std::flush;
        }
        if (completed == total && !finished) {
            finished = true;
            std::cerr << "\nPreparing model...\n";
        }
    };
}

void emit_text(const std::string& text) {
    std::cout << text << std::flush;
}

// ---------------------------------------------------------------------------
// commands
// ---------------------------------------------------------------------------

// An unknown type name is a usage error, refused before any file is opened.
int cmd_quantize(const std::string& json_path, const std::string& bin_path,
                 const std::string& out_path, const std::string& type_arg) {
    const std::optional<uint32_t> type = format::quant_type_of(type_arg);
    if (!type) throw UsageError("unknown quant type: " + type_arg + " (expected q8_0 or q4_0)");
    const size_t tensors = format::quantize_raw(json_path, bin_path, out_path, *type);
    std::cout << "wrote " << out_path << " (" << tensors << " tensors, " << type_arg << ")\n";
    return 0;
}

int cmd_dequantize(const std::string& in_path, const std::string& out_json,
                   const std::string& out_bin) {
    format::dequantize_to_raw(in_path, out_json, out_bin);
    std::cout << "wrote " << out_json << " and " << out_bin << "\n";
    return 0;
}

// File type names come from storage metadata, including types whose kernels are not implemented.
const char* type_name(uint32_t t) {
    const quant::StorageType* qt = quant::storage_type(t);
    return qt ? qt->name : "?";
}

void dump_value(const gguf::MetaValue& v) {
    switch (v.vtype) {
        case gguf::V_UINT8:  std::cout << v.u; break;
        case gguf::V_INT8:   std::cout << v.i; break;
        case gguf::V_UINT16: std::cout << v.u; break;
        case gguf::V_INT16:  std::cout << v.i; break;
        case gguf::V_UINT32: std::cout << v.u; break;
        case gguf::V_INT32:  std::cout << v.i; break;
        case gguf::V_FLOAT32:{ float x; std::memcpy(&x, &v.fb, 4); std::cout << x; break; }
        case gguf::V_BOOL:   std::cout << (v.b ? "true" : "false"); break;
        case gguf::V_STRING: std::cout << '"' << v.s << '"'; break;
        case gguf::V_ARRAY: {
            std::cout << "[";
            for (size_t i = 0; i < v.arr.size(); i++) {
                if (i) std::cout << ", ";
                dump_value(v.arr[i]);
            }
            std::cout << "]";
            break;
        }
        case gguf::V_UINT64:  std::cout << v.u; break;
        case gguf::V_INT64:   std::cout << v.i; break;
        case gguf::V_FLOAT64: std::cout << v.f64; break;
    }
}

int cmd_info(const std::string& in_path) {
    gguf::GGUFModel m = gguf::read_gguf(in_path);
    std::cout << "File: " << in_path << "\n";
    std::cout << "Tensors: " << m.tensors.size() << "\n";
    std::cout << "Metadata: " << m.kv.size() << " entries\n";
    for (const auto& kv : m.kv) {
        std::cout << "  " << kv.first << " = ";
        dump_value(kv.second);
        std::cout << "\n";
    }
    std::cout << "Tensor list:\n";
    for (const auto& t : m.tensors) {
        std::cout << "  " << type_name(t.type) << " " << t.name << " shape=[";
        for (size_t d = 0; d < t.ne.size(); d++) {
            if (d) std::cout << ", ";
            std::cout << t.ne[d];
        }
        std::cout << "] elements=" << t.n_elements()
                  << " bytes=" << t.data_size() << "\n";
    }
    return 0;
}

int cmd_tokenize(const std::string& model_path, const std::string& text) {
    gguf::GGUFModel m = gguf::read_gguf(model_path);
    bpe::Tokenizer t(m);
    std::vector<uint32_t> ids = t.encode(text);
    for (size_t i = 0; i < ids.size(); i++) {
        if (i) std::cout << ", ";
        std::cout << ids[i];
    }
    std::cout << "\n";
    return 0;
}

int cmd_detokenize(const std::string& model_path, const std::string& ids_arg) {
    gguf::GGUFModel m = gguf::read_gguf(model_path);
    bpe::Tokenizer t(m);
    std::cout << t.decode(token_ids(ids_arg, t.vocab.size())) << "\n";
    return 0;
}

// --layer-shares: one whole-number proportion per listed device.
std::vector<int> layer_shares(const std::string& value) {
    std::vector<int> shares;
    if (value.empty()) return shares;
    for (const auto& item : core::comma_list(value)) shares.push_back(whole_number(item, "--layer-shares", 0, 999999));
    return shares;
}

// A device spec's kind: cpu, or the backend before its index.
std::string kind_of(const std::string& spec) { return spec.substr(0, spec.find(':')); }

// --tensor-width over the listed devices (docs/TENSOR-SPLIT.md, section 4.6), refused as the command line gives it: whole groups of at most 4 devices of one kind, and one layer share a group.
// Each group's devices with their PCI roots on stderr, and, where infer::one_root_order finds one, the order of the listed devices that keeps every group under one root, the order given being kept (docs/TENSOR-SPLIT.md, section 4.5).
void report_groups(const std::vector<std::string>& specs, const std::vector<backend::BackendPtr>& backends, size_t w) {
    std::vector<std::string> roots;
    for (const auto& b : backends) roots.push_back(b->pci_root());
    for (size_t g = 0; g < backends.size(); g += w) {
        std::string names, under;
        for (size_t m = g; m < g + w; ++m) {
            names += (m > g ? "+" : "") + specs[m];
            const std::string root = roots[m].empty() ? "an unknown root" : roots[m];
            if (under.find(root) == std::string::npos) under += (under.empty() ? "" : " and ") + root;
        }
        std::fprintf(stderr, "tensor group %s under %s\n", names.c_str(), under.c_str());
    }
    std::string order;
    for (size_t d : infer::one_root_order(roots, w)) order += (order.empty() ? "" : ",") + specs[d];
    if (!order.empty()) std::fprintf(stderr, "tensor groups: --device %s puts every group under one root\n", order.c_str());
}

void tensor_groups(const ExecOptions& exec, const std::vector<std::string>& specs) {
    const size_t w = (size_t)exec.tensor_width;
    if (w == 1) return;
    const std::string flag = "--tensor-width " + std::to_string(w);
    if (w > 4) throw UsageError(flag + ": at most 4 devices a group; list more devices to form stages");
    if (specs.size() % w) throw UsageError(flag + " needs a device list of whole groups: " + std::to_string(specs.size()) + " devices listed");
    const size_t shares = layer_shares(exec.layer_shares).size();
    if (shares && shares != specs.size() / w)
        throw UsageError("--layer-shares gives one share a group: " + std::to_string(specs.size() / w) + " groups, " + std::to_string(shares) + " shares");
    for (size_t g = 0; g < specs.size(); g += w)
        for (size_t m = g + 1; m < g + w; ++m)
            if (kind_of(specs[m]) != kind_of(specs[g])) throw UsageError(flag + ": " + specs[g] + " and " + specs[m] + " cannot form a group");
}

// The load mode after `flag`, checked as it is read, so an unknown name is refused before any model file is read.
infer::LoadMode load_mode_arg(int argc, char** argv, int& i, const std::string& flag) {
    const std::string name = flag_value(argc, argv, i, flag);
    try {
        return infer::load_mode_of(name);
    } catch (const std::runtime_error& e) {
        throw UsageError(flag + ": " + e.what());
    }
}

// The execution flags every model command takes, the "Execution options" of its help: where it runs, its workers, its prompt batch, its caches and how its weights are read.
// Reads argv[i] (and its value) into `exec` and returns true when it is one of them.
// `batch_threads` adds --threads-batch (-tb), which only the commands that give a prompt's batched passes their own worker count read: generate, chat and perplexity.
bool exec_flag(int argc, char** argv, int& i, ExecOptions& exec, bool batch_threads) {
    const std::string a = argv[i];
    const std::string_view f = long_spelling(a);
    if (f == "--device") exec.device = flag_value(argc, argv, i, a);
    else if (f == "--layer-shares") { exec.layer_shares = nonempty_value(argc, argv, i, a, "a share for each device"); layer_shares(exec.layer_shares); }
    else if (f == "--tensor-width") exec.tensor_width = int_arg(argc, argv, i, a, 1);
    else if (f == "--n-cpu-moe") exec.cpu_moe = int_arg(argc, argv, i, a, 0);
    else if (f == "--cpu-moe") exec.cpu_moe = -1;
    else if (f == "--moe-stream-from") exec.moe_stream_from = int_arg(argc, argv, i, a, 0);
    else if (f == "--threads") exec.threads = int_arg(argc, argv, i, a, 0);
    else if (batch_threads && f == "--threads-batch") exec.threads_batch = int_arg(argc, argv, i, a, 0);
    else if (f == "--ubatch") exec.ubatch = int_arg(argc, argv, i, a, 1);
    else if (f == "--cache-type-k") exec.cache_type_k = cache_type_arg(argc, argv, i, a);
    else if (f == "--cache-type-v") exec.cache_type_v = cache_type_arg(argc, argv, i, a);
    else if (f == "--load-mode") exec.load_mode = load_mode_arg(argc, argv, i, a);
    else if (f == "--dtype") {
        const std::string value = flag_value(argc, argv, i, a);
        if (value == "auto") exec.dtype.reset();
        else if (value == "f32") exec.dtype = backend::Dtype::f32;
        else if (value == "f16") exec.dtype = backend::Dtype::f16;
        else if (value == "bf16") exec.dtype = backend::Dtype::bf16;
        else if (value == "int8") exec.dtype = backend::Dtype::int8;
        else throw UsageError("--dtype: expected auto, f16, bf16, f32 or int8");
    }
    else return false;
    return true;
}

// A load's timing line, printed with a split's plan: the mode, where a streamed load read from, and where its time went.
std::string load_timing(const infer::LoadTimes& t) {
    char line[400];
    if (t.files || t.direct_files) {
        const std::string from = (t.files ? std::to_string(t.files) + (t.files == 1 ? " file" : " files") + " buffered" : std::string()) +
                                 (t.files && t.direct_files ? " and " : "") +
                                 (t.direct_files ? std::to_string(t.direct_files) + (t.direct_files == 1 ? " file" : " files") + " direct" : std::string());
        char copied[64] = "";
        if (t.copied) std::snprintf(copied, sizeof copied, " (%.2f GiB copied from the reads)", double(t.copied) / double(size_t(1) << 30));
        std::snprintf(line, sizeof line, "load: %s, %.2f GiB read from %s; construct %.2f s, read %.2f s, upload %.2f s%s, waiting for reads %.2f s\n",
                      infer::load_mode_name(t.mode), double(t.streamed) / double(size_t(1) << 30), from.c_str(), t.construct, t.read, t.upload, copied, t.wait);
    } else {
        std::snprintf(line, sizeof line, "load: %s; construct %.2f s\n", infer::load_mode_name(t.mode), t.construct);
    }
    return line;
}

// Open a model file as the flags ask, through infer::load_model: the devices --device lists, made first so a bad flag fails before the file is read, the model placed over them for its ubatch plus `decode_rows` generated tokens a pass (a server's sequences), progress on stderr when `progress`, and a split's plan when `show_plan`.
// `threads` is the worker count to set, 0 to keep the backend's own; with `profiled`, the one device times its kernels and its address is written there (bench --profile).
// `history_tokens`, when given, is what each of the `decode_rows` sequences holds, and the cache grows to hold them all at once where its budget would not (infer::PlacementRequest::histories).
// `slots` is the passes a server keeps in flight, whose handoff buffers and logits rows a split's fit counts, and a server fits its KV budget to the devices and backs it at load; with `timed` every device times its work (serve --timing).
// `checkpoints` is the states a model that keeps one holds at a position (infer::ModelOptions::checkpoint_slots), -1 for the most a server's fit gives up to `decode_rows`.
// With `drafter` the file's embedded drafter is loaded with the model, and a `drafter_file` beside it is paired with it (infer::spec::pair) and, holding its MTP blocks, loaded as that drafter.
// `shared` are the backends of a model already open, which a draft model beside it is placed over rather than devices of its own.
std::unique_ptr<infer::LoadedModel> open_model(const std::string& path, const ExecOptions& exec, bool progress, int threads, size_t decode_rows = 0,
                                               bool show_plan = false, backend::Backend** profiled = nullptr, size_t history_tokens = 0, size_t slots = 0,
                                               bool timed = false, int checkpoints = 0, size_t mark_rows = 0, bool drafter = false,
                                               const std::string& drafter_file = {}, const std::vector<backend::BackendPtr>& shared = {}) {
    // Only experts on the CPU are streamed, and the flags alone say whether there are any, so a stream without them is refused before the file is read.
    if (exec.moe_stream_from && !exec.cpu_moe) throw UsageError("--moe-stream-from streams the experts on the CPU; give --n-cpu-moe or --cpu-moe");
    const auto specs = backend::device_specs(exec.device);
    tensor_groups(exec, specs);
    auto backends = shared.empty() ? backend::make_backends(specs, profiled != nullptr || timed) : shared;
    // Beside the placement a command prints, so a quiet command stays quiet.
    if (exec.tensor_width > 1 && (progress || show_plan)) report_groups(specs, backends, (size_t)exec.tensor_width);
    if (profiled) *profiled = backends.front().get();
    infer::PlacementRequest request;
    request.names = specs;
    request.dtype = exec.dtype;
    request.shares = layer_shares(exec.layer_shares);
    request.width = (size_t)exec.tensor_width;
    request.cpu_moe = exec.cpu_moe;
    request.stream_from = (size_t)exec.moe_stream_from;
    request.ubatch = exec.ubatch;
    request.decode_rows = decode_rows;
    request.slots = slots;
    // A server keeps logits rows for its passes in flight by its own rule, not one for every row of a pass.
    if (slots) request.logit_rows = server::logit_rows(slots, decode_rows).size;
    request.fit_kv = slots > 0;
    if (history_tokens) {
        request.histories = decode_rows;
        request.history_tokens = history_tokens;
    }
    infer::ModelOptions options = model_options(exec);
    // A recurrent state is held by each sequence decoding at once: the command's own one, or the sequences a pass carries.
    if (decode_rows) options.state_slots = decode_rows;
    options.checkpoint_slots = checkpoints < 0 ? decode_rows : (size_t)checkpoints;
    request.fit_checkpoints = checkpoints < 0;
    request.drafter = drafter;
    // A command that verifies drafts marks a sequence before each verify, of up to `mark_rows` rows: its one sequence, or as many of the `decode_rows` a server decodes at once as can draft in one pass, past the first only in the room its budget and checkpoints leave.
    if (mark_rows) {
        size_t columns = SIZE_MAX;
        for (const auto& b : backends) columns = std::min(columns, b->decode_columns());
        options.mark_slots = decode_rows ? server::draft_marks(decode_rows, columns) : 1;
        options.mark_rows = mark_rows;
        request.fit_marks = decode_rows > 0;
    }
    format::LoadProgress shown;
    if (progress) {
        std::cerr << "Reading model metadata...\n";
        shown = progress_bar();
    }
    auto loaded = infer::load_model(path, std::move(backends), request, options, shown, exec.load_mode, drafter_file);
    std::cerr << loaded->dtype.describe();
    if (show_plan) std::cerr << loaded->plan << load_timing(loaded->times);
    if (threads > 0) loaded->model->set_threads(threads);
    return loaded;
}

// One turn's prompt, before its reply is generated: prefill `ids` on the prompt's worker count (--threads-batch, else `decode_threads`), then set `decode_threads` back.
// `prefilled`, called before the decode lines are shown, gets the prompt's time in milliseconds, which includes setting the decode count back, since a changed count stops the CPU workers the prompt ran on.
// Returns the logits after the last prompt token.
// A `keep_at` inside the turn keeps the model's state there as its checkpoint (infer::Model::prefill).
std::vector<float> prefill_turn(infer::Model& model, const ExecOptions& exec, const std::vector<uint32_t>& ids, int decode_threads, bool progress,
                                const std::function<void(double)>& prefilled = {}, size_t keep_at = 0) {
    model.set_threads(exec.threads_batch > 0 ? exec.threads_batch : decode_threads);
    if (exec.verbose) std::cerr << "threads: prefill " << model.threads_available() << "\n";
    if (progress) std::cerr << "Processing " << ids.size() << " prompt tokens...\n";
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<float> logits = model.prefill(ids, keep_at);
    model.set_threads(decode_threads);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (prefilled) prefilled(ms);
    if (exec.verbose) std::cerr << "threads: decode " << model.threads_available() << "\n";
    if (progress) std::cerr << "Generating...\n";
    return logits;
}

// The text `--chat` makes of a prompt: one user message through the model's chat template with the assistant's header after it, as the server's chat routes render a conversation of that one message.
std::string chat_prompt(const chat::ChatFormat& format, const std::string& text) {
    return format.render({ { "user", text, std::nullopt } }, true);
}

// What `--drafter` and `--draft-max` ask of generate, chat and serve: the proposer, none for off, whether it is the file's embedded drafter, which the model is loaded with, and the most drafts a verify feeds after the last pick.
struct Drafts {
    std::unique_ptr<infer::spec::Proposer> proposer;
    bool embedded = false;
    std::string file;   // a drafter file beside the model, `--drafter PATH`
    size_t draft_max = 3;
    std::unique_ptr<infer::LoadedModel> beside;   // a draft model the file was, loaded as a model of its own
    // The rows of a verify a mark must save, none without a proposer.
    size_t mark_rows() const { return proposer || embedded || !file.empty() ? draft_max + 1 : 0; }
    // The proposer over what the model was loaded with: its embedded drafter or the MTP blocks the drafter file held, or the draft model the file was, opened on the same devices once the model has taken its room.
    // A `server` takes no draft model, which drafts on its own history.
    void attach(const infer::LoadedModel& loaded, const ExecOptions& exec, bool progress, bool server = false);
    // A request's drafting over the history it continues, or nothing without a proposer.
    std::unique_ptr<infer::spec::Drafting> drafting(const std::vector<uint32_t>& history) const {
        if (!proposer) return nullptr;
        auto d = std::make_unique<infer::spec::Drafting>();
        d->proposer = proposer.get();
        d->draft_max = draft_max;
        d->history = history;
        return d;
    }
};

// --drafter and --draft-max, which generate, chat and serve take; true when `f` is one of them.
bool drafts_flag(int argc, char** argv, int& i, const std::string& a, std::string_view f, Drafts& drafts) {
    if (f == "--drafter") {
        const std::string d = flag_value(argc, argv, i, a);
        drafts.proposer.reset();
        drafts.embedded = d == "embedded";
        drafts.file.clear();
        if (d == "lookup") drafts.proposer = std::make_unique<infer::spec::Lookup>();
        else if (d != "off" && d != "embedded") drafts.file = d;
        return true;
    }
    if (f == "--draft-max") {
        drafts.draft_max = (size_t)int_arg(argc, argv, i, a, 1, infer::spec::kMaxDrafts);
        return true;
    }
    return false;
}

void Drafts::attach(const infer::LoadedModel& loaded, const ExecOptions& exec, bool progress, bool server) {
    if (loaded.drafter == infer::spec::DrafterKind::mtp) embedded = true;
    if (embedded) {
        proposer = std::make_unique<infer::spec::Embedded>(*loaded.model);
        return;
    }
    if (loaded.drafter != infer::spec::DrafterKind::model) return;
    if (server)
        throw std::runtime_error("serve: " + file + " is a draft model, which drafts in generate and chat; the server drafts with lookup, embedded, or MTP blocks in a file beside the model");
    // A draft model that keeps a recurrent state keeps one where its drafts begin, so each draft goes back there (infer::spec::DraftModel); it shares the model's devices, so the two take turns on one queue a device.
    beside = open_model(file, exec, progress, exec.threads, 0, false, nullptr, 0, 0, false, 1, 0, false, {}, loaded.backends);
    proposer = std::make_unique<infer::spec::DraftModel>(*beside->model);
}

int cmd_generate(const std::string& model_path, const std::string& prompt, bool as_chat, const infer::GenParams& gp, const ExecOptions& exec,
                 Drafts& drafts) {
    const bool progress = show_progress(exec);
    const auto loaded = open_model(model_path, exec, progress, exec.threads, 0, exec.verbose, nullptr, 0, 0, false, 0, drafts.mark_rows(), drafts.embedded, drafts.file);
    bpe::Tokenizer& tok = *loaded->tok;
    infer::Model& model = *loaded->model;
    drafts.attach(*loaded, exec, progress);
    const int decode_threads = model.threads_available();
    infer::RNG rng;
    if (gp.seed) rng.seed(gp.seed);

    std::vector<uint32_t> ids = tok.encode(as_chat ? chat_prompt(loaded->chat, prompt) : prompt);
    if (ids.empty()) throw std::runtime_error("generate: empty prompt");

    const std::vector<float> logits = prefill_turn(model, exec, ids, decode_threads, progress, [&](double pp_ms) {
        if (exec.verbose) std::cout << "prompt tokens: " << ids.size() << "\n";
        // Preserve fractional throughput for slow models.
        printf("pp: %zu tok, %.0f ms, %.2f tok/s\n", ids.size(), pp_ms, (double)ids.size() / (pp_ms / 1e3));
    });
    const auto t0 = std::chrono::steady_clock::now();
    const auto drafting = drafts.drafting(ids);
    std::vector<uint32_t> gen = infer::generate(model, tok, gp, rng, logits, emit_text, drafting.get());
    std::cout << "\n";
    double tg_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    printf("tg: %zu tok, %.0f ms, %.2f tok/s\n", gen.size(), tg_ms,
           (double)gen.size() / (tg_ms / 1e3));
    if (exec.verbose) {
        printf("kv: allocated %zu bytes, peak %zu bytes, used %zu bytes\n",
               model.kv_allocated_bytes(), model.kv_peak_bytes(), model.kv_used_bytes());
        // The generated ids, which `logits --then-ids` reads back as the tokens they were, where their text could tokenize otherwise (tools/long_context_check.py).
        printf("ids:");
        for (size_t i = 0; i < gen.size(); ++i) printf("%s%u", i ? "," : " ", gen[i]);
        printf("\n");
        // By draft position, the drafts the verifies kept of those they fed.
        if (drafting) {
            printf("drafts kept:");
            const infer::spec::Acceptance& a = drafting->acceptance;
            for (size_t i = 0; i < a.drafted().size(); ++i) printf(" %zu/%zu", a.kept()[i], a.drafted()[i]);
            printf("\n");
        }
    }
    return 0;
}

// Completed diagnostic work, for the HF checker: the selected policy and paths actually dispatched, never the capability catalog.
void matrix_record(infer::LoadedModel& loaded) {
    const auto devices = loaded.model->take_matrix_paths();
    std::cerr << "matrix-paths: {\"dtype\":\"" << backend::dtype_name(loaded.dtype.effective) << "\",\"devices\":[";
    for (size_t i = 0; i < devices.size(); ++i) {
        std::cerr << (i ? "," : "") << "[";
        for (size_t j = 0; j < devices[i].size(); ++j) std::cerr << (j ? "," : "") << jmini::quote(devices[i][j]);
        std::cerr << "]";
    }
    std::cerr << "]}\n";
}

// `then_ids` appends exact generated IDs without re-tokenizing their text; `last` reports the final positions.
// Positions go through the batched passes a prompt takes, or with `per_token` one at a time through step, the decode path a generated token takes.
// These logits support the external correctness gate in docs/ROADMAP.md #8.
int cmd_logits(const std::string& model_path, const std::string& text, bool as_chat,
               int topn, const ExecOptions& exec, const std::string& then_ids = "", size_t last = 0, bool per_token = false) {
    const auto loaded = open_model(model_path, exec, false, exec.threads);
    bpe::Tokenizer& tok = *loaded->tok;
    infer::Model& model = *loaded->model;

    std::vector<uint32_t> ids = tok.encode(as_chat ? chat_prompt(loaded->chat, text) : text);
    if (!then_ids.empty()) {
        const std::vector<uint32_t> more = token_ids(read_text_file(then_ids, "logits"), model.n_vocab());
        ids.insert(ids.end(), more.begin(), more.end());
    }
    if (ids.empty()) throw std::runtime_error("logits: empty prompt");

    // Ranked by the top list the server's log-probabilities take, a tie to the lower id; the values printed are the raw logits.
    auto top = [&](const float* logits) { return infer::top_logprobs(logits, model.n_vocab(), 0.0, (size_t)topn); };
    const size_t from = ids.size() - std::min(last, ids.size());
    auto row = [&](size_t pos, const float* logits) {
        if (pos < from) return;
        printf("%zu", pos);
        for (const auto& t : top(logits)) printf(" %u %.6f", t.id, logits[t.id]);
        printf("\n");
    };
    printf("tokens: %zu\n", ids.size());
    std::vector<float> logits;
    if (per_token) {
        for (size_t pos = 0; pos < ids.size(); ++pos) {
            logits = model.step((int)ids[pos]);
            if (last) row(pos, logits.data());
        }
    } else if (last) {
        model.score(ids, row);
    } else {
        logits = model.prefill(ids);
    }
    if (!last)
        for (const auto& t : top(logits.data())) printf("%u %.6f\n", t.id, logits[t.id]);
    matrix_record(*loaded);
    return 0;
}

int cmd_perplexity(const std::string& model_path, const std::string& text,
                   const ExecOptions& exec, int context_size, int chunks, bool per_token) {
    const int threads = exec.threads_batch > 0 ? exec.threads_batch : exec.threads;   // main refuses --threads-batch with --per-token
    const auto loaded = open_model(model_path, exec, false, threads, 0, exec.verbose);
    bpe::Tokenizer& tok = *loaded->tok;
    infer::Model& model = *loaded->model;
    if (exec.verbose)
        std::cerr << "threads: " << (per_token ? "decode " : "prefill ") << model.threads_available() << "\n";

    std::vector<uint32_t> ids = tok.encode(text);
    const auto result = infer::perplexity(model, ids, context_size, chunks, per_token);
    if (exec.verbose)
        for (size_t w = 0; w < result.windows.size(); ++w)
            printf("chunk %zu: scored %zu, mean NLL %.9f\n", w + 1, result.windows[w].scored, result.windows[w].nll / (double)result.windows[w].scored);
    double mean_nll = result.mean_nll();
    double ppl = std::exp(mean_nll);
    std::cout << "tokens: " << ids.size() << "\n";
    std::cout << "used tokens: " << result.used_tokens << "\n";
    std::cout << "scored tokens: " << result.scored_tokens << "\n";
    std::cout << "chunks: " << result.windows.size() << "\n";
    std::cout << "context size: " << result.context << "\n";
    std::cout << "mean NLL: " << mean_nll << "\n";
    std::cout << "perplexity: " << ppl << "\n";
    matrix_record(*loaded);
    return 0;
}

int cmd_chat(const std::string& model_path, const std::string& system, const infer::GenParams& gp, const ExecOptions& exec, Drafts& drafts) {
    const bool progress = show_progress(exec);
    // A model that keeps a state keeps it where the next turn will begin, and two slots let a turn's checkpoint take over from the last one's.
    const auto loaded = open_model(model_path, exec, progress, exec.threads, 0, exec.verbose, nullptr, 0, 0, false, 2, drafts.mark_rows(), drafts.embedded, drafts.file);
    bpe::Tokenizer& tok = *loaded->tok;
    infer::Model& model = *loaded->model;
    drafts.attach(*loaded, exec, progress);
    const chat::ChatFormat& format = loaded->chat;
    format.require();
    const int decode_threads = model.threads_available();
    infer::RNG rng;
    if (gp.seed) rng.seed(gp.seed);

    std::vector<chat::Message> messages;
    messages.push_back({ "system", system, std::nullopt });
    std::vector<uint32_t> cached_ids;
    size_t kept_extent = 0;   // the length of the prompt whose rows the model's checkpoint holds

    std::cout << "Chat ready (type your message; Ctrl+C to quit)\n" << std::flush;
    std::string line;
    while (std::getline(std::cin, line)) {
        messages.push_back({ "user", line, std::nullopt });

        std::string gen = format.render(messages, true);
        std::vector<uint32_t> gen_ids = tok.encode(gen);
        if (gen_ids.empty()) throw std::runtime_error("chat: template produced an empty prompt");
        // Templates can rewrite previous turns or change token boundaries.
        // Reuse only an exact prefix; an unchanged prompt also needs fresh logits because generate() does not retain its final distribution.
        if (cached_ids.size() >= gen_ids.size() ||
            !std::equal(cached_ids.begin(), cached_ids.end(), gen_ids.begin())) {
            // A model that keeps a state goes back to its checkpoint within the shared prefix, whose rows the last prompt computed as this one computes them.
            size_t shared = 0;
            while (shared < cached_ids.size() && shared + 1 < gen_ids.size() && cached_ids[shared] == gen_ids[shared]) ++shared;
            const bool alike = model.keeps_state() && model.row_class(kept_extent) == model.row_class(gen_ids.size());
            const size_t reached = alike ? model.retract(shared) : 0;
            if (!reached) model.reset();
            cached_ids.resize(reached);
        }
        // The state is kept where the next turn begins: whole blocks of the conversation rendered without the generation prompt.
        size_t keep_at = 0;
        if (model.keeps_state()) {
            const size_t n = std::min(chat::stable_prefix(format, tok, messages, gen_ids), gen_ids.size() - 1);
            keep_at = n / model.kv_block_tokens() * model.kv_block_tokens();
            if (keep_at <= cached_ids.size()) keep_at = 0;
            else kept_extent = gen_ids.size();
        }
        const std::vector<float> logits = prefill_turn(model, exec, std::vector<uint32_t>(gen_ids.begin() + cached_ids.size(), gen_ids.end()),
                                                       decode_threads, progress, {}, keep_at);
        cached_ids = std::move(gen_ids);
        const auto drafting = drafts.drafting(cached_ids);
        std::vector<uint32_t> reply = infer::generate(model, tok, gp, rng, logits, emit_text, drafting.get());
        std::cout << "\n" << std::flush;
        // A stop match may return its final token without feeding it.
        // EOS is excluded; the next rendered turn supplies its own closing tokens.
        const size_t fed = (size_t)model.n_tokens() - cached_ids.size();
        cached_ids.insert(cached_ids.end(), reply.begin(), reply.begin() + fed);

        messages.push_back(format.assistant(tok.decode(reply)));
    }
    return 0;
}


// The synthetic bench's prefill of `prefill` tokens and decode of `decode` more, one token a step from a reset history, each timed in milliseconds; `timing` runs as the clock starts.
// The whole run goes once untimed first, so the timed one leaves out the model's first-run setup on its backend: the history's storage growing, kernels made on first use, a device leaving its idle clocks.
struct StepTimes {
    double prefill_ms = 0, decode_ms = 0;
};
StepTimes time_steps(infer::Model& model, int prefill, int decode, int vocab, const std::function<void()>& timing = {}) {
    using clock = std::chrono::steady_clock;
    StepTimes t;
    model.reset();
    for (int i = 0; i < prefill + decode; i++) model.step((uint32_t)(i % vocab));
    model.reset();
    if (timing) timing();
    auto t0 = clock::now();
    for (int i = 0; i < prefill; i++) model.step((uint32_t)(i % vocab));
    t.prefill_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    t0 = clock::now();
    for (int i = 0; i < decode; i++) model.step((uint32_t)((prefill + i) % vocab));
    t.decode_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    return t;
}

// Micro-benchmark of the backend hot paths (matmul, RMSNorm, norm+RoPE) plus end-to-end prefill/decode TPS on the synthetic model (infer::synthetic_model).
// Used by tests/perf.py as the perf-regression gate for hot-path changes.
int cmd_bench(int size, int iters, int threads, int prefill, int decode,
              const std::string& device) {
    // The hot paths are one backend's; with several devices listed, the first one's.
    auto b = backend::make_backend(backend::device_specs(device).front());
    if (threads > 0) b->set_threads(threads);
    std::cout << "bench: threads " << b->threads_available() << "\n";

    // Square matmul: mat is [nin, nout] = [size, size]. x is the input (length nin), out the result (length nout). nout rows, each nin/32 blocks.
    const size_t nblocks = (size_t)size / quant::Q8_0_BLOCK;
    std::vector<float> x(size, 0.5f);
    std::vector<uint8_t> mat((size_t)size * nblocks * quant::Q8_0_TYPESIZE);
    std::vector<float> src(size), w(size), dst(size);
    for (int i = 0; i < size; i++) { src[i] = std::sin((float)i * 0.01f); w[i] = 0.1f; }
    std::vector<float> cos(size / 2), sin(size / 2);
    for (int i = 0; i < size / 2; i++) { cos[i] = std::cos(0.1f); sin[i] = std::sin(0.1f); }

    const auto weights = b->adopt(mat.data(), mat.size());
    const auto x_buf = b->adopt(x.data(), x.size() * sizeof(float));
    const auto dst_buf = b->adopt(dst.data(), dst.size() * sizeof(float));

    // One untimed pass, drained, so the timed loop leaves out one-time setup such as the CPU backend starting its workers on its first parallel dispatch.
    b->matmul(quant::GGML_TYPE_Q8_0, {weights.get(), 0}, {x_buf.get(), 0},
              {dst_buf.get(), 0}, (size_t)size, (size_t)size, 1);
    b->sync();
    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();
    for (int it = 0; it < iters; it++)
        b->matmul(quant::GGML_TYPE_Q8_0, {weights.get(), 0}, {x_buf.get(), 0},
                  {dst_buf.get(), 0}, (size_t)size, (size_t)size, 1);
    double mm_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count() / iters;

    const auto src_buf = b->adopt(src.data(), src.size() * sizeof(float));
    const auto w_buf = b->adopt(w.data(), w.size() * sizeof(float));

    t0 = clock::now();
    for (int it = 0; it < iters; it++)
        b->rms_norm({dst_buf.get(), 0}, {src_buf.get(), 0}, {w_buf.get(), 0},
                    (size_t)size, 1e-6f);
    double rn_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count() / iters;

    // The op the model runs: one row of one head of `size` floats, at position 0 of a one-entry table.
    const auto cos_buf = b->adopt(cos.data(), cos.size() * sizeof(float));
    const auto sin_buf = b->adopt(sin.data(), sin.size() * sizeof(float));
    const uint32_t pos0 = 0;
    t0 = clock::now();
    for (int it = 0; it < iters; it++)
        b->norm_rope_partial({dst_buf.get(), 0}, {dst_buf.get(), 0}, 1, (size_t)size, (size_t)size, 1, (size_t)size, (size_t)size,
                             {w_buf.get(), 0}, 1e-6f, {cos_buf.get(), 0}, {sin_buf.get(), 0}, &pos0);
    double rp_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count() / iters;

    double mm_gflops = 2.0 * (double)size * (double)size / (mm_ms * 1e6);
    printf("bench: matmul %dx%d  %8.3f ms  %8.2f GFLOPS\n", size, size, mm_ms, mm_gflops);
    printf("bench: rms_norm n=%d  %8.3f ms\n", size, rn_ms);
    printf("bench: norm_rope n=%d  %8.3f ms\n", size, rp_ms);

    // End-to-end TPS on a small synthetic model (2 layers, 256 embd).
    {
        const int nl = 2, ne = 256, nf = 1024, nh = 8, nk = 2, hd = 32, nv = 512;
        gguf::GGUFModel sm = infer::synthetic_model({nl, ne, nf, nh, nk, hd, nv, 12345u});
        infer::Model model(infer::gguf_weights(sm), b);

        const int P = prefill, G = decode;
        const StepTimes st = time_steps(model, P, G, nv);
        const double pre_ms = st.prefill_ms, dec_ms = st.decode_ms;
        const double pre_tps = (double)P / (pre_ms / 1e3), dec_tps = (double)G / (dec_ms / 1e3);

        printf("bench: prefill %3d tok  %8.3f ms  %8.1f tok/s\n", P, pre_ms, pre_tps);
        printf("bench: decode  %3d tok  %8.3f ms  %8.1f tok/s\n", G, dec_ms, dec_tps);
    }
    return 0;
}

// Time model execution over fixed IDs after warm-up; history setup and sampling are outside the timer.
// Multi-sequence decode follows each sequence's prompt, while single-sequence runs may use the requested depth; see docs/USAGE.md.
// What bench's `--drafter` value asks for: the file's embedded drafter, the MTP blocks of a file beside it, or neither for off; it drafts nothing either way.
struct BenchDrafter {
    bool embedded = false;
    std::string file;
    explicit operator bool() const { return embedded || !file.empty(); }
};
BenchDrafter bench_drafter(const std::string& d) {
    if (d == "lookup") throw UsageError("bench --model takes --drafter off, embedded or a drafter file, not 'lookup'");
    return {d == "embedded", d == "embedded" || d == "off" ? std::string() : d};
}

// With `drafts` the file's embedded drafter, or the MTP blocks its file holds, is loaded and its context rows run in every pass, with no draft taken (docs/SPECULATIVE.md, section 7, the k = 0 gate).
int cmd_bench_model(const std::string& path, const ExecOptions& exec, int P, int G, int R, bool profile, int seqs = 1, int D = 0, const BenchDrafter& drafts = {}) {
    const bool drafter = bool(drafts);
    // The drafter's rollback is timed after a verify of kRollbackDrafts drafts, which its mark holds, and a step after it: the prompt after the depth and those rows and one more.
    const size_t kRollbackDrafts = 3, rollback_reach = (size_t)D + (size_t)P + kRollbackDrafts + 2;
    // What each sequence holds at most: a batched one its prompt and its generated tokens, the one sequence its depth and the longer of its two tests, or the rollback's reach.
    const size_t reach = seqs > 1 ? (size_t)P + (size_t)G : std::max((size_t)D + (size_t)std::max(P, G), drafter ? rollback_reach : 0);
    backend::Backend* b = nullptr;   // the device --profile times
    const auto loaded = open_model(path, exec, false, exec.threads, (size_t)seqs, true, profile ? &b : nullptr, reach, 0, false, 0,
                                   drafter ? kRollbackDrafts + 1 : 0, drafts.embedded, drafts.file);
    if (loaded->drafter == infer::spec::DrafterKind::model)
        throw std::runtime_error("bench --model times a drafter's rows in the model's own passes, and " + drafts.file + " is a draft model");
    infer::Model& model = *loaded->model;
    // Ids below 1000, or below a smaller vocabulary's size, such as the test fixtures'.
    const uint32_t vocab = (uint32_t)std::min<size_t>(1000, model.n_vocab());
    auto ids_from = [vocab](uint32_t seed, size_t n) {
        std::vector<uint32_t> ids(n);
        for (auto& t : ids) { seed = seed * 1664525u + 1013904223u; t = (seed >> 8) % vocab; }
        return ids;
    };
    const std::vector<uint32_t> prompt = ids_from(12345u, (size_t)P), gen = ids_from(777u, (size_t)G), history = ids_from(4242u, (size_t)D);
    // A cleared history, then the depth's tokens if any, before a timed test starts.
    auto fresh = [&] {
        model.reset();
        if (D > 0) model.prefill(history);
    };
    using clock = std::chrono::steady_clock;
    auto ms_since = [](clock::time_point t0) {
        return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    };
    auto pp = [&] {
        fresh();
        const auto t0 = clock::now();
        model.prefill(prompt);
        return (double)P / (ms_since(t0) / 1e3);
    };
    infer::ExecContext ctx;
    // `start` runs once the decode is set up and before its first pass, so a profile of it leaves out the sequences' prompts.
    // Batched decode clears the one sequence too, so the last prompt's blocks are back in the pool before its sequences take theirs.
    auto tg = [&](const std::function<void()>& start) {
        fresh();
        if (seqs <= 1) {
            if (start) start();
            const auto t0 = clock::now();
            for (uint32_t t : gen) model.step((int)t);
            return (double)G / (ms_since(t0) / 1e3);
        }
        std::vector<infer::Sequence> s;
        for (int i = 0; i < seqs; ++i) s.push_back(model.make_sequence());
        for (auto& q : s) {
            infer::BatchEntry e{&q, prompt.data(), prompt.size(), true};
            model.forward(ctx, &e, 1);
        }
        ctx.logits(0);
        if (start) start();
        std::vector<infer::BatchEntry> batch;
        const auto t0 = clock::now();
        for (int g = 0; g < G; ++g) {
            batch.clear();
            for (auto& q : s) batch.push_back(infer::BatchEntry{&q, &gen[(size_t)g], 1, true});
            model.forward(ctx, batch.data(), batch.size());
            ctx.logits(0);   // a server reads each pass's logits before it samples the next tokens
        }
        const double rate = (double)G * seqs / (ms_since(t0) / 1e3);
        for (auto& q : s) model.reset(q);
        return rate;
    };
    auto report = [&](const char* what, int n, const std::vector<double>& v) {
        double mean = 0, var = 0;
        for (double x : v) mean += x;
        mean /= (double)v.size();
        for (double x : v) var += (x - mean) * (x - mean);
        const double sd = v.size() > 1 ? std::sqrt(var / (double)(v.size() - 1)) : 0.0;
        const std::string at = D > 0 ? " @ d" + std::to_string(D) : "";
        printf("bench: %s%d%s  %8.2f +- %.2f tok/s  (%zu runs)\n", what, n, at.c_str(), mean, sd, v.size());
    };
    pp();
    tg({});
    std::vector<double> ppv, tgv;
    for (int r = 0; r < R; r++) ppv.push_back(pp());
    for (int r = 0; r < R; r++) tgv.push_back(tg({}));
    report("pp", P, ppv);
    report(seqs > 1 ? ("x" + std::to_string(seqs) + " tg").c_str() : "tg", G, tgv);
    // With the drafter, a round's rollback (docs/SPECULATIVE.md, section 7): after a mark and a verify of kRollbackDrafts drafts, the retract keeping each number of its rows and the decode step after it, timed as completed work up to the step's logits.
    // Each repeat takes every rejection position in turn after the same history, so a short rerun cannot stand for a long one, and the cost of keeping fewer rows is its difference from keeping all, whose retract runs nothing, in the same repeat, so a card whose clock drifts with its heat moves both; the median of those differences and their middle half, after one round that is not counted.
    if (drafter && seqs <= 1 && G > (int)kRollbackDrafts + 1 && rollback_reach > (size_t)model.context_length())
        printf("bench: rollback skipped: the depth, the prompt, the verify's %zu rows and the step after them reach %zu tokens, past the context of %d\n",
               kRollbackDrafts + 1, rollback_reach, model.context_length());
    else if (drafter && seqs <= 1 && G > (int)kRollbackDrafts + 1) {
        const size_t L = (size_t)D + (size_t)P, rows = kRollbackDrafts + 1;
        const int next = (int)gen[rows];
        std::vector<std::vector<double>> after(rows + 1), recorded(rows + 1);
        std::vector<double> marking;
        for (int r = 0; r <= R; r++)
            for (size_t kept = 1; kept <= rows; ++kept) {
                fresh();
                model.prefill(prompt);
                auto t0 = clock::now();
                if (!model.mark()) throw std::runtime_error("bench: no mark for the rollback");
                const double mark_ms = ms_since(t0);
                model.step(gen.data(), rows);
                t0 = clock::now();
                model.retract(L + kept);
                const double record_ms = ms_since(t0);
                model.step(next);
                if (!r) continue;
                after[kept].push_back(ms_since(t0));
                recorded[kept].push_back(record_ms);
                marking.push_back(mark_ms);
            }
        auto quantile = [](std::vector<double> v, double q) {
            std::sort(v.begin(), v.end());
            const double pos = q * (double)(v.size() - 1);
            const size_t i = (size_t)pos;
            return i + 1 < v.size() ? v[i] + (pos - (double)i) * (v[i + 1] - v[i]) : v[i];
        };
        const double all = quantile(after[rows], 0.5);
        const std::string at = D > 0 ? " @ d" + std::to_string(D) : "";
        printf("bench: rollback%s, a step after keeping all %zu rows %.3f ms, the mark %.3f ms  (%d runs)\n", at.c_str(), rows, all, quantile(marking, 0.5), R);
        for (size_t kept = 1; kept < rows; ++kept) {
            std::vector<double> extra(after[kept].size());
            for (size_t i = 0; i < extra.size(); ++i) extra[i] = after[kept][i] - after[rows][i];
            const double ms = quantile(extra, 0.5);
            printf("bench: rollback keeping %zu of %zu rows%s  %+8.3f ms, %+.2f%% of the step, middle half %+.3f to %+.3f ms, the retract's own call %.3f ms\n", kept, rows,
                   at.c_str(), ms, 100.0 * ms / all, quantile(extra, 0.25), quantile(extra, 0.75), quantile(recorded[kept], 0.5));
        }
    }
    if (profile) {
#if LLMX_HAS_BACKEND_VULKAN
        // Device time per kernel over one more prompt and one more decode run, each read on its own, so a pass is attributed to its kernels rather than inferred from kernels timed alone.
        // A run calls its argument where its interval starts: a batched decode's after its sequences' prompts.
        auto section = [&](const char* what, const std::function<double(const std::function<void()>&)>& run) {
            run([&] { backend::vulkan_kernel_times(*b); });
            auto times = backend::vulkan_kernel_times(*b);
            std::sort(times.begin(), times.end(),
                      [](const auto& x, const auto& y) { return x.second > y.second; });
            double total = 0.0;
            for (const auto& t : times) total += t.second;
            std::cout << "profile " << what << ": " << total << " ms of device time over "
                      << backend::vulkan_timed_dispatches(*b) << " dispatches sampled\n";
            for (const auto& t : times)
                std::cout << "profile:   " << t.first << " " << t.second << " ms ("
                          << (total > 0.0 ? 100.0 * t.second / total : 0.0) << "%)\n";
        };
        section("pp", [&](const std::function<void()>& start) { start(); return pp(); });
        section(seqs > 1 ? "batched tg" : "tg", tg);
        // And what the driver made of each kernel that ran: registers, shared memory and waves per SIMD, where it reports them.
        std::istringstream stats(backend::vulkan_kernel_statistics(*b));
        for (std::string line; std::getline(stats, line);) std::cout << "profile: kernel " << line << "\n";
#else
        (void)b;   // main refuses --profile unless the device is a Vulkan one, which this build cannot open
#endif
    }
    matrix_record(*loaded);
    return 0;
}

// llmx serve: the multi-user server of docs/SERVER.md over one model.
int cmd_serve(const std::string& model_path, const server::Config& cfg, const ExecOptions& exec, Drafts& drafts) {
    // Without --passes a pipelined split keeps a pass in flight per stage, and its stages are at most the devices listed.
    const size_t slots = cfg.passes ? cfg.passes : backend::device_specs(exec.device).size();
    const auto loaded = open_model(model_path, exec, true, exec.threads, cfg.max_seqs, false, nullptr, 0, slots, cfg.timing, cfg.state_checkpoints,
                                   drafts.mark_rows(), drafts.embedded, drafts.file);
    bpe::Tokenizer& tok = *loaded->tok;
    infer::Model& model = *loaded->model;
    drafts.attach(*loaded, exec, true, true);
    // A template the renderer refuses stops the server before it listens, as it stops chat before a turn.
    loaded->chat.require();
    server::Config c = cfg;
    // The path is UTF-8, as the loader reads it, so the name is read back as UTF-8 rather than in the system code page.
    c.model_name = std::filesystem::u8path(model_path).filename().u8string();
    c.dtype = loaded->dtype;
    // Read once the model and its caches are in memory, so the default takes what they leave.
    if (!c.host_cache_bytes) c.host_cache_bytes = server::default_host_cache(model, c.max_seqs);
    if (c.disk.bytes) {
        c.disk.model_path = model_path;
        if (c.disk.dir.empty()) c.disk.dir = (hub::default_cache("--disk-cache-dir") / "kv").u8string();
        server::check_disk_cache(c.disk, *c.host_cache_bytes);
    }
    c.proposer = drafts.proposer.get();
    c.draft_max = drafts.draft_max;
    http::Listener listener(c.host, c.port);
    // A split's plan, what each device was given, so a lopsided placement shows in the log.
    std::cerr << loaded->plan;
    std::cerr << "serving " << c.model_name << " on http://" << c.host << ":" << listener.port()
              << " (device " << exec.device << ", up to " << c.max_seqs << " sequences over "
              << model.kv_tokens_total() << " KV tokens" << (model.keeps_state() ? ", " + std::to_string(model.checkpoint_slots()) + " state checkpoints" : std::string())
              << (loaded->checkpoint_kv_tokens ? " taking " + std::to_string(loaded->checkpoint_kv_tokens) + " KV tokens" : std::string())
              << ", " << (*c.host_cache_bytes >> 20) << " MiB of host memory for evicted prefixes"
              << ", queue of " << c.max_queue << ")\n";
    server::serve(model, tok, loaded->chat, c, listener);
    return 0;
}

// Writes the overview, or `command`'s page, to `out`; false when there is no such command.
bool print_usage(const std::string& command, std::ostream& out) {
    if (!command.empty() && command != "chat" && command != "generate" &&
        command != "serve" && command != "pull" && command != "info" &&
        command != "quantize" && command != "dequantize" && command != "tokenize" &&
        command != "detokenize" && command != "logits" && command != "perplexity" &&
        command != "bench") return false;
    out << "llmx " << LLMX_VERSION_STRING << " - ground-up LLM runtime\n\n";
    if (command.empty()) {
        out
            << "Usage: llmx <command> [arguments] [options]\n\n"
            << "Run a model:\n"
            << "  chat        <model>                  Interactive chat with follow-up turns\n"
            << "  generate    <model> \"<prompt>\"       Generate from a raw text prompt\n"
            << "  serve       <model>                  Start the HTTP inference server\n\n"
            << "Manage models:\n"
            << "  pull        <owner/repo>:<quant>     Download a GGUF from Hugging Face\n"
            << "  info        <model>                  Show metadata and tensor layouts\n"
            << "  quantize    <json> <bin> <out.gguf>   Convert F32 tensors to Q8_0/Q4_0\n"
            << "  dequantize  <gguf> <json> <bin>       Export tensors as F32\n\n"
            << "Inspect and measure:\n"
            << "  tokenize    <model> \"<text>\"         Encode text to token IDs\n"
            << "  detokenize  <model> <ids>            Decode comma/whitespace-separated IDs\n"
            << "  logits      <model> \"<text>\"         Inspect next-token scores\n"
            << "  perplexity  <model> \"<text>\"         Score text or a file\n"
            << "  bench                               Measure kernels or a real model\n\n"
            << "A model is a GGUF file, or the first shard of a GGUF set.\n"
            << "Downloads require curl 8.4+; no Python runtime is needed.\n\n"
            << "Help:     llmx <command> --help  (or -h)\n"
            << "Version:  llmx --version\n"
            << "Example:  llmx chat model.gguf --threads 6 --temp 0 -n 256\n";
        return true;
    }
    const infer::Sampling sampling;
    const ExecOptions defaults;
    const infer::ModelOptions caches;
    // A cache side's two types, the model's default first and marked.
    const auto cache_types = [](backend::KVType d) {
        const backend::KVType other = d == backend::KVType::f16 ? backend::KVType::f32 : backend::KVType::f16;
        return std::string(backend::kv_type_name(d)) + " (default) or " + backend::kv_type_name(other);
    };
    const auto model_options = [&](bool batch_threads) {
        out << "\nExecution options:\n"
            << "  --device D              " << defaults.device << " (default), or vulkan:N when built with Vulkan;\n"
            << "                          several, comma separated, split the model by layers\n"
            << "                          over them in that order, fitted to their free memory\n"
            << "  --dtype T               Matmul inputs: auto (default), f16, bf16, f32 or int8;\n"
            << "                          emulation or a wider fallback is reported per device\n"
            << "  --layer-shares A,B      With several devices, their proportions of the layers\n"
            << "  --tensor-width N        Devices each layer is split across: the listed devices\n"
            << "                          form groups of N, the stages of the layer split (default: " << defaults.tensor_width << ")\n"
            << "  --threads N             CPU workers; 0 takes the fewest of the hardware threads\n"
            << "                          and the CPUs the affinity and the CPU quota allow,\n"
            << "                          at most 64 (default: " << defaults.threads << ")\n";
        if (batch_threads) out
            << "  --threads-batch N, -tb  CPU prefill workers; default follows --threads\n";
        out
            << "  --ubatch N              Prompt tokens per pass (default: " << infer::kDefaultUbatch << ")\n"
            << "  --cache-type-k T, -ctk  Key cache: " << cache_types(caches.kv_k) << "\n"
            << "  --cache-type-v T, -ctv  Value cache: " << cache_types(caches.kv_v) << "\n"
            << "  --n-cpu-moe N           First N routed layers' experts on CPU (default: " << defaults.cpu_moe << ")\n"
            << "  --cpu-moe               All routed layers' experts on CPU\n"
            << "  --moe-stream-from N     Copy those experts to the device for a prompt of\n"
            << "                          at least N tokens; 0 disables this (default: " << defaults.moe_stream_from << ").\n"
            << "                          Generated tokens and one-token prompts stay on CPU.\n"
            << "  --load-mode M           How weights are read: auto, mapped or direct (default: " << infer::load_mode_name(defaults.load_mode) << ")\n";
    };
    if (command == "chat" || command == "generate") {
        const bool chat = command == "chat";
        out << (chat ? "Interactive chat with retained conversation history.\n\n"
                     : "Generate from raw text, or with --chat from one user message in the chat template.\n\n")
            << "Usage: llmx " << command << " <model.gguf>"
            << (chat ? " [options]\n" : " \"<prompt>\" [options]\n       llmx generate <model.gguf> --file <path> [options]\n")
            << "\nGeneration options:\n";
        if (!chat) out
            << "  --file PATH, -f         UTF-8 prompt file, immediately after the model\n"
            << "  --chat                  Send the prompt as one user message through the model's chat template\n";
        out
            << "  -n N, --max-tokens N    Maximum generated tokens per turn (default: " << sampling.max_tokens << ")\n"
            << "  --temp F                Temperature; 0 is greedy (default: " << sampling.temp << ")\n"
            << "  --topk N                Top-k sampling (default: " << sampling.top_k << ")\n"
            << "  --topp F                Nucleus sampling (default: " << sampling.top_p << ")\n"
            << "  --penalty F             Repetition penalty (default: " << sampling.penalty << ")\n"
            << "  --seed N                RNG seed; 0 keeps the fixed default state\n"
            << "  --stop TEXT             Stop when generated text contains TEXT\n"
            << "  --ignore-eos            Never end at the end-of-text token; run to -n or --stop\n"
            << "  --drafter D             Draft tokens to verify in one pass: off, lookup (the\n"
            << "                          tokens that followed the last ones earlier), embedded\n"
            << "                          (the MTP block the file carries), or a drafter file beside\n"
            << "                          the model: its MTP blocks or a draft model; the output is\n"
            << "                          the same either way (default: off)\n"
            << "  --draft-max N           Most drafts a verify takes, 1 to " << infer::spec::kMaxDrafts << " (default: " << Drafts{}.draft_max << ")\n"
            << "  --verbose               Show the prompt token count, progress and execution details" << (chat ? "" : ", and the generated ids") << "\n";
        if (chat) out
            << "  --system TEXT           System message (default: " << kChatSystem << ")\n";
        model_options(true);
        if (chat) out << "\nEnter one message per line; Ctrl+C or end of input exits.\n";
        out << "\nExample: llmx " << command << " model.gguf"
            << (chat ? "" : " \"The capital of France is\"") << " --temp 0 -n 256\n";
    } else if (command == "serve") {
        const server::Config cfg;
        out << "Serve concurrent requests with streaming and prefix reuse.\n\n"
            << "Usage: llmx serve <model.gguf> [options]\n\n"
            << "Server options:\n"
            << "  --host H                Listen address (default: " << cfg.host << ")\n"
            << "  --port N                Listen port; 0 picks a free one (default: " << cfg.port << ")\n"
            << "  --max-seqs N            Active request limit (default: " << cfg.max_seqs << ")\n"
            << "  --max-queue N           Queued request limit, paused requests not counted (default: " << cfg.max_queue << ")\n"
            << "  --passes N              Passes in flight; above 1 needs a layer split (default: its stages, else 1)\n"
            << "  --state-checkpoints N   States a recurrent model keeps for prefix reuse (default: fitted, up to --max-seqs)\n"
            << "  --host-cache-bytes N    Host memory for prefixes the devices evict; 0 keeps none (default: --max-seqs histories as long as a request may hold, within half of free host memory once the model is loaded, none with every cache on the CPU)\n"
            << "  --disk-cache-bytes N    Disk for what the host cache drops; needs a host cache (default: 0, none)\n"
            << "  --disk-cache-dir PATH   Where the disk cache lives (default: <home>/" << hub::cache_in_home << "/kv)\n"
            << "  --disk-cache-floor N    Free space the disk keeps after every write (default: the larger of 16 GiB and a twentieth of the disk)\n"
            << "  --disk-cache-keep       At a clean exit, write what memory holds to disk within " << server::kDiskFlush.count() << " s and keep it for the next server\n"
            << "  --disk-cache-max-age TIME  Delete entries unused for longer than TIME: seconds, or a number followed by s, m, h or d; 0 for no limit (default: " << (server::DiskOptions{}.max_age / 3600) << "h)\n"
            << "  --timing                Time the rounds and each device's work for /v1/health; slows serving\n"
            << "  --ctx-size N, -c        Most KV tokens in total, fitted to the devices at load (default: model context)\n"
            << "  --drafter D             Draft tokens to verify beside other requests: off, lookup, embedded,\n"
            << "                          or a file of MTP blocks beside the model, where the passes' measured\n"
            << "                          cost finds a gain; each reply the same either way (default: off)\n"
            << "  --draft-max N           Most drafts a verify takes, 1 to " << infer::spec::kMaxDrafts << " (default: " << Drafts{}.draft_max << ")\n";
        model_options(false);
        out << "\nRoutes:\n"
            << "  POST /v1/generate             POST /v1/chat\n"
            << "  POST /v1/completions          POST /v1/chat/completions\n"
            << "  POST /v1/tokenize             POST /v1/detokenize\n"
            << "  GET  /v1/health               GET  /v1/models\n\n"
            << "Sampling settings belong in each request's JSON body.\n"
            << "Example: llmx serve model.gguf --device vulkan:0 --port 8080\n";
    } else if (command == "pull") {
        const hub::PullOptions pull;
        out << "Download and verify a GGUF model or complete shard set.\n\n"
            << "Usage: llmx pull <owner/repo>:<quant> [options]\n\n"
            << "Options:\n"
            << "  --revision REF          Branch, tag or commit SHA (default: " << pull.revision << ")\n"
            << "  --file NAME             Choose a file when several match the quant\n"
            << "  --cache-dir PATH        Cache root (default: <home>/" << hub::cache_in_home << ")\n"
            << "  --parallel N            Streams per file, 1.." << hub::max_parallel_streams << " (default: " << pull.parallel << ")\n\n"
            << "Requires curl 8.4+. HF_TOKEN supplies gated-repo credentials.\n"
            << "The verified local path goes to stdout; progress goes to stderr.\n\n"
            << "Example: llmx pull Qwen/Qwen3-0.6B-GGUF:Q8_0 --parallel 4\n";
    } else if (command == "logits" || command == "perplexity") {
        const bool ppl = command == "perplexity";
        out << (ppl ? "Score next-token likelihoods over text or bounded windows.\n\n"
                    : "Print the highest next-token logits after a prompt.\n\n")
            << "Usage: llmx " << command << " <model.gguf> \"<text>\" [options]\n"
            << "       llmx " << command << " <model.gguf> --file <path> [options]\n"
            << (ppl ? "\nScoring options:\n" : "\nOptions:\n")
            << "  --file PATH, -f         UTF-8 input file, immediately after the model\n";
        if (ppl) out
            << "  --ctx-size N, -c        Window tokens (default: model context)\n"
            << "  --chunks N              Maximum windows (default: all)\n"
            << "  --per-token             Score through decode; default uses batched passes\n"
            << "  --verbose               Show scoring phase, actual worker count and each window's NLL\n";
        else out
            << "  --chat                  Read the text as one user message through the model's chat template\n"
            << "  --top N                 Number of logits to print (default: " << kLogitsTop << ")\n"
            << "  --then-ids PATH         Append these token IDs, comma or whitespace separated\n"
            << "  --last N                Print each of the last N positions, one per line\n"
            << "  --per-token             Read every token through decode steps; default uses batched passes\n";
        model_options(ppl);
        out << "\nExample: llmx " << command << " model.gguf "
            << (ppl ? "--file corpus.txt --ctx-size 512 --chunks 4\n"
                    : "\"The capital of France is\" --top 10\n");
    } else if (command == "bench") {
        const BenchNumbers bench;
        out << "Measure synthetic kernels or a real model after warm-up.\n\n"
            << "Usage: llmx bench [options]\n"
            << "       llmx bench --model <model.gguf> [options]\n\n"
            << "Benchmark options:\n"
            << "  --p N                   Prompt tokens (default: " << bench.prompt << ")\n"
            << "  --n N                   Decode tokens (default: " << bench.decode << ")\n"
            << "  --size N                Synthetic matrix width, multiple of 32 (default: " << bench.size << ")\n"
            << "  --iters N               Synthetic kernel repetitions (default: " << bench.iters << ")\n"
            << "  --model PATH            Benchmark this model instead of synthetic weights\n"
            << "  --r N                   Real-model repetitions (default: " << bench.repeats << ")\n"
            << "  --seqs N                Sequences decoding together, a pass one token of each (default: " << bench.seqs << ")\n"
            << "  --depth N               History of N tokens, filled untimed, that each test runs after (default: " << bench.depth << ")\n"
            << "                          It counts toward --moe-stream-from's prompt length.\n"
            << "  --profile               Real-model kernel timing and statistics on one Vulkan device\n"
            << "  --drafter D             off, embedded (the MTP block the file carries) or a file\n"
            << "                          of MTP blocks beside it: load it, run its rows in every\n"
            << "                          pass, drafting nothing, then time a round's rollback at\n"
            << "                          each kept row (default: off)\n";
        model_options(false);
        out << "\nExecution options other than --device and --threads apply only with --model.\n"
            << "Example: llmx bench --model model.gguf --p 512 --n 128 --r 3\n";
    } else if (command == "quantize") {
        out << "Convert raw F32 tensors into a quantized GGUF file.\n\n"
            << "Usage: llmx quantize <model.json> <model.bin> <out.gguf> [q8_0|q4_0]\n\n"
            << "Default quant: " << kQuantType << ". Input row widths must be divisible by 32.\n"
            << "The JSON describes tensor names/shapes; the binary contains F32 values.\n"
            << "Example: llmx quantize model.json model.bin model.gguf q8_0\n";
    } else if (command == "dequantize") {
        out << "Export supported GGUF tensors as JSON metadata and F32 values.\n\n"
            << "Usage: llmx dequantize <in.gguf> <out.json> <out.bin>\n\n"
            << "Example: llmx dequantize model.gguf model.json model.bin\n";
    } else if (command == "info") {
        out << "Show GGUF metadata, tensor types and dimensions.\n\n"
            << "Usage: llmx info <model.gguf>\n\n"
            << "For a sharded model, pass its first shard.\n"
            << "Example: llmx info model.gguf\n";
    } else if (command == "tokenize") {
        out << "Encode text with the model's tokenizer.\n\n"
            << "Usage: llmx tokenize <model.gguf> \"<text>\"\n\n"
            << "Example: llmx tokenize model.gguf \"hello world\"\n";
    } else {
        out << "Decode comma- or whitespace-separated token IDs.\n\n"
            << "Usage: llmx detokenize <model.gguf> <ids>\n\n"
            << "Example: llmx detokenize model.gguf \"1,2,3\"\n";
    }
    out << "\n-h, --help shows this page. Full reference: docs/USAGE.md\n";
    return true;
}

} // namespace

#if defined(_WIN32)
// On Windows argv arrives in the system ANSI codepage, which cannot represent most non-ASCII text -- a Japanese or Cyrillic prompt is mangled before it reaches us.
// Re-read the command line as UTF-16 and convert to UTF-8 so text arguments survive.
// Storage is owned by the caller and must outlive argv.
static bool utf8_argv(int& argc, char**& argv,
                      std::vector<std::string>& store, std::vector<char*>& ptrs) {
    int wargc = 0;
    LPWSTR* wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
    if (!wargv) return false;
    store.reserve((size_t)wargc);
    for (int i = 0; i < wargc; i++) {
        int n = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, nullptr, 0, nullptr, nullptr);
        std::string s(n > 0 ? (size_t)(n - 1) : 0, '\0');
        if (n > 1) WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, &s[0], n, nullptr, nullptr);
        store.push_back(std::move(s));
    }
    LocalFree(wargv);
    ptrs.reserve(store.size() + 1);
    for (auto& s : store) ptrs.push_back(&s[0]);
    ptrs.push_back(nullptr);
    argc = wargc;
    argv = ptrs.data();
    return true;
}
#endif

int main(int argc, char** argv) {
#if defined(_WIN32)
    std::vector<std::string> argv_store;
    std::vector<char*> argv_ptrs;
    utf8_argv(argc, argv, argv_store, argv_ptrs);
    SetConsoleOutputCP(CP_UTF8);
#endif
    const std::string cmd = argc > 1 ? argv[1] : "";
    try {
        if (argc < 2) { print_usage({}, std::cout); return 1; }
        if (cmd == "--help" || cmd == "-h") {
            if (argc == 2) { print_usage({}, std::cout); return 0; }
            throw UsageError(cmd + " takes nothing after it; a command's page is llmx <command> --help");
        }
        if (argc == 3 && (std::string(argv[2]) == "--help" || std::string(argv[2]) == "-h") && print_usage(cmd, std::cout)) return 0;
        if (cmd == "--version") {
            if (argc > 2) throw UsageError("--version takes nothing after it");
            std::cout << "llmx " << LLMX_VERSION_STRING << "\n";
            return 0;
        }

        if (cmd == "pull") {
            if (argc < 3) throw UsageError("missing the <owner/repo>:<quant> to download");
            const std::string target = argv[2];
            const auto colon = target.find(':');
            if (colon == std::string::npos) throw UsageError("expected <owner/repo>:<quant>, not '" + target + "'");
            hub::PullOptions options;
            options.repo = target.substr(0, colon);
            options.quant = target.substr(colon + 1);
#ifdef _WIN32
            char* token = nullptr;
            size_t token_size = 0;
            if (_dupenv_s(&token, &token_size, "HF_TOKEN")) throw std::runtime_error("pull: cannot read HF_TOKEN");
            std::unique_ptr<char, decltype(&std::free)> token_owner(token, &std::free);
            if (token) options.token = token;
#else
            if (const char* token = std::getenv("HF_TOKEN")) options.token = token;
#endif
            GivenFlags given;
            for (int i = 3; i < argc; ++i) {
                const int at = i;
                const std::string flag = argv[i];
                const std::string_view f = long_spelling(flag);
                if (f == "--revision") options.revision = flag_value(argc, argv, i, flag);
                else if (f == "--file") options.filename = nonempty_value(argc, argv, i, flag, "a file name");
                else if (f == "--cache-dir") options.cache = std::filesystem::u8path(nonempty_value(argc, argv, i, flag, "a path"));
                else if (f == "--parallel") options.parallel = int_arg(argc, argv, i, flag, 1u, hub::max_parallel_streams);
                else throw UsageError("unknown flag: " + flag);
                given.take(flag, i > at);
            }
            const auto path = hub::pull(options, [](const std::string& message) { std::cerr << message << '\n'; });
            std::cout << path.u8string() << '\n';
            return 0;
        }


        if (cmd == "generate" || cmd == "chat") {
            const bool chat = cmd == "chat";
            if (argc < 3) throw UsageError("missing the model");
            infer::GenParams gp;
            ExecOptions exec;
            std::string system = kChatSystem;
            std::string prompt;
            bool have_prompt = false, as_chat = false;
            GivenFlags given;
            Drafts drafts;
            // generate takes its prompt from the file `--file` or `-f` names right after the model, as logits and perplexity take their text, for a prompt longer than a command line holds.
            const std::string third = argc > 3 ? argv[3] : "";
            const int first = !chat && (third == "--file" || third == "-f") ? text_arg(argc, argv) : 3;
            have_prompt = first == 5;
            for (int i = first; i < argc; i++) {
                const int at = i;
                const std::string a = argv[i];
                const std::string_view f = long_spelling(a);
                if (!chat) no_second_text(a);
                if (f == "--max-tokens") gp.max_tokens = int_arg(argc, argv, i, a, 1);
                else if (f == "--temp") gp.temp = float_arg(argc, argv, i, a, infer::Sampling::temp_range.lo, infer::Sampling::temp_range.hi);
                else if (f == "--topk") gp.top_k = int_arg(argc, argv, i, a, infer::Sampling::top_k_range.lo, infer::Sampling::top_k_range.hi);
                else if (f == "--topp") gp.top_p = float_arg(argc, argv, i, a, infer::Sampling::top_p_range.lo, infer::Sampling::top_p_range.hi);
                else if (f == "--penalty") gp.penalty = float_arg(argc, argv, i, a, infer::Sampling::penalty_range.lo, infer::Sampling::penalty_range.hi);
                else if (f == "--seed") gp.seed = int_arg<uint64_t>(argc, argv, i, a, 0);
                else if (f == "--stop") gp.stop = nonempty_value(argc, argv, i, a, "a text");
                else if (f == "--ignore-eos") gp.ignore_eos = true;
                else if (drafts_flag(argc, argv, i, a, f, drafts)) {}
                else if (exec_flag(argc, argv, i, exec, true)) {}
                else if (f == "--system" && chat) system = flag_value(argc, argv, i, a);
                else if (f == "--chat" && !chat) as_chat = true;
                else if (f == "--verbose") exec.verbose = true;
                else if (!a.empty() && a[0] == '-') throw UsageError("unknown flag: " + a);
                else if (chat) throw UsageError("chat reads its messages from standard input, not '" + a + "'");
                else if (have_prompt) throw UsageError("a second prompt, '" + a + "'; quote the prompt to keep its spaces");
                else { prompt = a; have_prompt = true; }
                given.take(a, i > at);
            }
            if (chat) return cmd_chat(argv[2], system, gp, exec, drafts);
            if (!have_prompt) throw UsageError("missing the prompt");
            return cmd_generate(argv[2], first == 5 ? read_text_file(argv[4], cmd) : prompt, as_chat, gp, exec, drafts);
        }

        if (cmd == "perplexity") {
            ExecOptions exec;
            int context_size = 0, chunks = 0;
            bool per_token = false;
            GivenFlags given;
            const int first = text_arg(argc, argv);
            for (int i = first; i < argc; i++) {
                const int at = i;
                const std::string a = argv[i];
                const std::string_view f = long_spelling(a);
                no_second_text(a);
                if (f == "--ctx-size") context_size = int_arg(argc, argv, i, a, infer::kMinPerplexityWindow);
                else if (f == "--chunks") chunks = int_arg(argc, argv, i, a, 1);
                else if (f == "--per-token") per_token = true;
                else if (f == "--verbose") exec.verbose = true;
                else if (exec_flag(argc, argv, i, exec, true)) {}
                else throw UsageError("unknown flag: " + a);
                given.take(a, i > at);
            }
            if (const std::string batch = given.spelling("--threads-batch"); per_token && !batch.empty())
                throw UsageError(batch + " sets the workers of batched passes, which --per-token does not run");
            const std::string text = first == 5 ? read_text_file(argv[4], cmd) : argv[3];
            return cmd_perplexity(argv[2], text, exec, context_size, chunks, per_token);
        }

        if (cmd == "logits") {
            ExecOptions exec;
            int topn = kLogitsTop;
            std::string then_ids;
            size_t last = 0;
            bool per_token = false;
            bool as_chat = false;
            GivenFlags given;
            const int first = text_arg(argc, argv);
            for (int i = first; i < argc; i++) {
                const int at = i;
                const std::string a = argv[i];
                const std::string_view f = long_spelling(a);
                no_second_text(a);
                if (f == "--top") topn = int_arg(argc, argv, i, a, 1);
                else if (f == "--then-ids") then_ids = nonempty_value(argc, argv, i, a, "a path");
                else if (f == "--last") last = (size_t)int_arg(argc, argv, i, a, 1);
                else if (f == "--per-token") per_token = true;
                else if (f == "--chat") as_chat = true;
                else if (exec_flag(argc, argv, i, exec, false)) {}
                else throw UsageError("unknown flag: " + a);
                given.take(a, i > at);
            }
            const std::string text = first == 5 ? read_text_file(argv[4], cmd) : argv[3];
            return cmd_logits(argv[2], text, as_chat, topn, exec, then_ids, last, per_token);
        }

        if (cmd == "tokenize") {
            if (argc != 4) throw UsageError("takes a model and a text");
            return cmd_tokenize(argv[2], argv[3]);
        }
        if (cmd == "detokenize") {
            if (argc != 4) throw UsageError("takes a model and a list of token ids");
            return cmd_detokenize(argv[2], argv[3]);
        }

        if (cmd == "quantize") {
            if (argc < 5 || argc > 6) throw UsageError("takes the JSON file, the binary file, the output file and optionally the type");
            return cmd_quantize(argv[2], argv[3], argv[4], argc == 6 ? argv[5] : kQuantType);
        }
        if (cmd == "dequantize") {
            if (argc != 5) throw UsageError("takes the model and the two output files");
            return cmd_dequantize(argv[2], argv[3], argv[4]);
        }
        if (cmd == "info") {
            if (argc != 3) throw UsageError("takes one model file");
            return cmd_info(argv[2]);
        }
        if (cmd == "serve") {
            if (argc < 3) throw UsageError("missing the model");
            server::Config cfg;
            ExecOptions exec;
            GivenFlags given;
            Drafts drafts;
            std::optional<uint64_t> host_bytes;
            for (int i = 3; i < argc; i++) {
                const int at = i;
                const std::string a = argv[i];
                const std::string_view f = long_spelling(a);
                if (f == "--host") cfg.host = flag_value(argc, argv, i, a);
                else if (f == "--port") cfg.port = (uint16_t)int_arg(argc, argv, i, a, 0, 65535);   // 0 asks the system for a free port
                else if (f == "--max-seqs") cfg.max_seqs = (size_t)int_arg(argc, argv, i, a, 1);
                else if (f == "--max-queue") cfg.max_queue = (size_t)int_arg(argc, argv, i, a, 1);
                else if (f == "--passes") cfg.passes = (size_t)int_arg(argc, argv, i, a, 1);
                else if (f == "--state-checkpoints") cfg.state_checkpoints = int_arg(argc, argv, i, a, 0);
                else if (f == "--host-cache-bytes") host_bytes = int_arg<uint64_t>(argc, argv, i, a, 0);
                else if (f == "--disk-cache-bytes") cfg.disk.bytes = int_arg<uint64_t>(argc, argv, i, a, 0);
                else if (f == "--disk-cache-dir") cfg.disk.dir = nonempty_value(argc, argv, i, a, "a path");
                else if (f == "--disk-cache-floor") cfg.disk.floor = int_arg<uint64_t>(argc, argv, i, a, 0);
                else if (f == "--disk-cache-keep") cfg.disk.keep = true;
                else if (f == "--disk-cache-max-age") cfg.disk.max_age = seconds_arg(argc, argv, i, a);
                else if (f == "--timing") cfg.timing = true;
                else if (f == "--ctx-size") exec.kv_tokens = int_arg(argc, argv, i, a, 1);
                else if (drafts_flag(argc, argv, i, a, f, drafts)) {}
                else if (exec_flag(argc, argv, i, exec, false)) {}
                else throw UsageError("unknown flag: " + a);
                given.take(a, i > at);
            }
            if (host_bytes) cfg.host_cache_bytes = (size_t)*host_bytes;
            return cmd_serve(argv[2], cfg, exec, drafts);
        }
        if (cmd == "bench") {
            BenchNumbers n;
            bool profile = false;
            BenchDrafter drafter;
            std::string model_path, model_only, synthetic_only;   // the first flag given that only a model run reads, and the first only the synthetic bench reads
            ExecOptions exec;
            GivenFlags given;
            for (int i = 2; i < argc; i++) {
                const int at = i;
                const std::string a = argv[i];
                const std::string_view f = long_spelling(a);
                if (f == "--size") { n.size = int_arg(argc, argv, i, a, 32); if (synthetic_only.empty()) synthetic_only = a; }
                else if (f == "--iters") { n.iters = int_arg(argc, argv, i, a, 1); if (synthetic_only.empty()) synthetic_only = a; }
                else if (f == "--p") n.prompt = int_arg(argc, argv, i, a, 1);
                else if (f == "--n") n.decode = int_arg(argc, argv, i, a, 1);
                else if (f == "--model") model_path = nonempty_value(argc, argv, i, a, "a path");
                else if (exec_flag(argc, argv, i, exec, false)) { if (f != "--device" && f != "--threads" && model_only.empty()) model_only = a; }
                else if (f == "--r") { n.repeats = int_arg(argc, argv, i, a, 1); if (model_only.empty()) model_only = a; }
                else if (f == "--seqs") { n.seqs = int_arg(argc, argv, i, a, 1); if (model_only.empty()) model_only = a; }
                else if (f == "--depth") { n.depth = int_arg(argc, argv, i, a, 0); if (model_only.empty()) model_only = a; }
                else if (f == "--profile") { profile = true; if (model_only.empty()) model_only = a; }
                else if (f == "--drafter") { drafter = bench_drafter(flag_value(argc, argv, i, a)); if (model_only.empty()) model_only = a; }
                else throw UsageError("unknown flag: " + a);
                given.take(a, i > at);
            }
            // The synthetic bench times one backend's kernels and reads only --device, --threads, --size, --iters, --p and --n; a model run reads neither --size nor --iters.
            if (model_path.empty() && !model_only.empty()) throw UsageError(model_only + " takes --model");
            if (!model_path.empty() && !synthetic_only.empty()) throw UsageError(synthetic_only + " is for the synthetic bench, not --model");
            if (n.size % 32 != 0) throw UsageError("--size must be a multiple of 32");
            // Batched decode already starts after each sequence's prompt.
            if (n.depth > 0 && n.seqs > 1) throw UsageError("--depth takes one sequence");
            if (profile) {
                const auto specs = backend::device_specs(exec.device);
                if (specs.size() != 1 || specs[0].rfind("vulkan:", 0) != 0 || !exec.layer_shares.empty())
                    throw UsageError("--profile times the kernels of one Vulkan device");
            }
            if (!model_path.empty()) return cmd_bench_model(model_path, exec, n.prompt, n.decode, n.repeats, profile, n.seqs, n.depth, drafter);
            return cmd_bench(n.size, n.iters, exec.threads, n.prompt, n.decode, exec.device);
        }
        std::cerr << "unknown command: " << cmd << "\n";
        return 2;
    } catch (const UsageError& e) {
        if (!print_usage(cmd, std::cerr)) print_usage({}, std::cerr);
        std::cerr << "\nerror: " << e.what() << "\n";
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
