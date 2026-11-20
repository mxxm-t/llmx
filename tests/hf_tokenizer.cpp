#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include "tokenizer/hf_tokenizer.hpp"
#include "tokenizer/tokenizer.hpp"

namespace fs = std::filesystem;

static size_t cases = 0;
static fs::path root;

static void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

// A replacement of one exact piece of a file's text, or an addition before its closing brace when `from` is empty.
struct Edit {
    std::string from, to;
};

static std::string edited(std::string text, const Edit& e) {
    if (e.from.empty() && e.to.empty()) return text;
    if (e.from.empty()) {
        text.insert(text.size() - 1, "," + e.to);
        return text;
    }
    const size_t at = text.find(e.from);
    require(at != std::string::npos, "test file has no " + e.from);
    return text.replace(at, e.from.size(), e.to);
}

static const std::string pattern_json =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+)";

// A byte-level BPE of 12 tokens and three added ones in 16 ids: ids 11 and 15 are unused.
static std::string tokenizer_json() {
    return R"({"version":"1.0","truncation":null,"padding":null,"added_tokens":[)"
        R"({"id":12,"content":"<|endoftext|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},)"
        R"({"id":13,"content":"<|im_end|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":false},)"
        R"({"id":14,"content":"<think>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":false}],)"
        R"("normalizer":{"type":"NFC"},"pre_tokenizer":{"type":"Sequence","pretokenizers":[)"
        R"({"type":"Split","pattern":{"Regex":")" + pattern_json + R"("},"behavior":"Isolated","invert":false},)"
        R"({"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}]},)"
        R"("post_processor":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},)"
        R"("decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},)"
        R"("model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":"","end_of_word_suffix":"",)"
        R"("fuse_unk":false,"byte_fallback":false,"ignore_merges":false,)"
        "\"vocab\":{\"h\":0,\"e\":1,\"l\":2,\"o\":3,\"\xC4\xA0\":4,\"w\":5,\"r\":6,\"d\":7,\"he\":8,\"ll\":9,\"hell\":10},"
        R"("merges":["h e",["l","l"],["he","ll"]]}})";
}

static std::string tokenizer_config() {
    return R"({"add_bos_token":false,"add_prefix_space":false,"added_tokens_decoder":{"12":{"content":"<|endoftext|>","special":true}},)"
        R"("bos_token":null,"chat_template":"{{ messages }}","clean_up_tokenization_spaces":false,"eos_token":"<|im_end|>",)"
        R"("errors":"replace","model_max_length":64,"pad_token":{"content":"<|endoftext|>"},"split_special_tokens":false,)"
        R"("tokenizer_class":"Qwen2Tokenizer","unk_token":null})";
}

static const std::string config_json = R"({"bos_token_id":12,"eos_token_id":12})";

static fs::path directory(const Edit& tj = {}, const Edit& tc = {}, const std::string& jinja = "") {
    const fs::path dir = root / std::to_string(cases);
    fs::create_directories(dir);
    std::ofstream(dir / "tokenizer.json", std::ios::binary) << edited(tokenizer_json(), tj);
    std::ofstream(dir / "tokenizer_config.json", std::ios::binary) << edited(tokenizer_config(), tc);
    if (!jinja.empty()) std::ofstream(dir / "chat_template.jinja", std::ios::binary) << jinja;
    return dir;
}

static bpe::hf::KeyValues read(const fs::path& dir) {
    return bpe::hf::read_keys(dir.u8string(), jmini::parse(config_json), 16);
}

static void refused(const Edit& tj, const Edit& tc, const std::string& expected) {
    ++cases;
    try {
        read(directory(tj, tc));
    } catch (const std::exception& e) {
        if (std::string(e.what()).find(expected) == std::string::npos)
            throw std::runtime_error("refused with \"" + std::string(e.what()) + "\", expected \"" + expected + "\"");
        return;
    }
    throw std::runtime_error("accepted, expected a refusal with \"" + expected + "\"");
}

static const gguf::MetaValue& key(const gguf::GGUFModel& m, const char* name) {
    const gguf::MetaValue* v = m.find(name);
    require(v != nullptr, std::string("no key ") + name);
    return *v;
}

int main() {
    root = fs::temp_directory_path() / ("llmx-hf-tokenizer-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        {
            ++cases;
            gguf::GGUFModel m;
            m.kv = read(directory());
            require(m.kv.size() == 10, "ten keys");
            require(key(m, "tokenizer.ggml.model").s == "gpt2" && key(m, "tokenizer.ggml.pre").s == "qwen2", "model and pre");
            const auto& tokens = key(m, "tokenizer.ggml.tokens");
            const auto& types = key(m, "tokenizer.ggml.token_type");
            require(tokens.u == gguf::V_STRING && tokens.arr.size() == 16 && types.u == gguf::V_INT32 && types.arr.size() == 16, "array shapes");
            require(tokens.arr[4].s == "\xC4\xA0" && tokens.arr[10].s == "hell" && tokens.arr[11].s == "[PAD11]" && tokens.arr[15].s == "[PAD15]", "tokens");
            const int64_t want[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 3, 3, 4, 5};
            for (size_t i = 0; i < 16; ++i) require(types.arr[i].vtype == gguf::V_INT32 && types.arr[i].i == want[i], "type of " + std::to_string(i));
            const auto& merges = key(m, "tokenizer.ggml.merges");
            require(merges.arr.size() == 3 && merges.arr[0].s == "h e" && merges.arr[1].s == "l l" && merges.arr[2].s == "he ll", "merges");
            // eos and pad from tokenizer_config.json's text, bos from config.json, whose eos does not override the text's.
            require(key(m, "tokenizer.ggml.eos_token_id").u == 13 && key(m, "tokenizer.ggml.padding_token_id").u == 12 &&
                    key(m, "tokenizer.ggml.bos_token_id").u == 12, "special ids");
            require(key(m, "tokenizer.ggml.add_bos_token").vtype == gguf::V_BOOL && !key(m, "tokenizer.ggml.add_bos_token").b, "add_bos_token");
            require(key(m, "tokenizer.chat_template").s == "{{ messages }}", "chat template");
            bpe::Tokenizer tok(m);
            const std::vector<uint32_t> ids = tok.encode("hello<think> world<|im_end|>");
            require(ids == std::vector<uint32_t>({10, 3, 14, 4, 5, 3, 6, 2, 7, 13}), "encode");
            require(tok.eos_id == 13 && tok.bos_id == 12, "tokenizer ids");
        }
        {
            ++cases;
            gguf::GGUFModel m;
            m.kv = read(directory({}, {"\"add_bos_token\":false,", ""}, "{{ jinja }}"));
            require(key(m, "tokenizer.chat_template").s == "{{ jinja }}" && !m.find("tokenizer.ggml.add_bos_token"), "chat_template.jinja");
        }
        {
            ++cases;
            gguf::GGUFModel m;
            m.kv = read(directory({"\"post_processor\":{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"trim_offsets\":false,\"use_regex\":false}",
                                   "\"post_processor\":null"}, {"\"chat_template\":\"{{ messages }}\",", ""}));
            require(!m.find("tokenizer.chat_template"), "no template");
        }
        {
            // gptqmodel writes a pad text naming no token, which gives no padding id as HF's tokenizer has none, beside two keys of no effect on encoding.
            ++cases;
            gguf::GGUFModel m;
            m.kv = read(directory({}, {"\"pad_token\":{\"content\":\"<|endoftext|>\"}", "\"pad_token\":\"<unk>\",\"extra_special_tokens\":{},\"_commit_hash\":null"}));
            require(!m.find("tokenizer.ggml.padding_token_id") && key(m, "tokenizer.ggml.eos_token_id").u == 13, "pad naming no token");
        }
        {
            // A Qwen 3.5 file: the qwen35 Split pattern, repeated as tokenizer_config.json's pretokenize_regex, and a token only added_tokens_decoder adds, taken as an added token.
            // Qwen 3.6 adds response_template, how a reply is parsed, which changes no encoding.
            ++cases;
            gguf::GGUFModel m;
            const std::string qwen35 =
                R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+)";
            const std::string decoder = R"("special":true},"15":{"content":"<|audio|>","special":true}},"pretokenize_regex":")" + qwen35 +
                R"(","response_template":{"defaults":{"role":"assistant"},"start_anchor":"<|im_start|>assistant\n"})";
            m.kv = read(directory({pattern_json, qwen35}, {"\"special\":true}}", decoder}));
            const gguf::MetaValue& tokens = key(m, "tokenizer.ggml.tokens");
            const gguf::MetaValue& types = key(m, "tokenizer.ggml.token_type");
            require(key(m, "tokenizer.ggml.pre").s == "qwen35" && tokens.arr[15].s == "<|audio|>" && types.arr[15].i == 3, "qwen35 pattern and a decoder-only token");
            refused({pattern_json, qwen35}, {"", "\"pretokenize_regex\":\"" + pattern_json + "\""}, "pretokenize_regex differs");
        }
        {
            // extra_special_tokens naming added tokens, as a map or a list, changes no encoding.
            ++cases;
            gguf::GGUFModel m;
            m.kv = read(directory({}, {"", "\"extra_special_tokens\":{\"think\":\"<think>\"}"}));
            require(key(m, "tokenizer.ggml.eos_token_id").u == 13, "extra_special_tokens map");
            ++cases;
            m.kv = read(directory({}, {"", "\"extra_special_tokens\":[\"<think>\",\"<|im_end|>\"]"}));
            require(key(m, "tokenizer.ggml.eos_token_id").u == 13, "extra_special_tokens list");
        }
        {
            // llm-compressor saves tokenizer.json with the truncation and padding of its calibration; transformers clears both on every encode it is not asked to cut or pad, so they change no id.
            ++cases;
            gguf::GGUFModel m;
            m.kv = read(directory({"\"truncation\":null,\"padding\":null", "\"truncation\":{\"direction\":\"Right\",\"max_length\":4,\"strategy\":\"LongestFirst\",\"stride\":0},"
                                   "\"padding\":{\"strategy\":{\"Fixed\":4},\"direction\":\"Right\",\"pad_to_multiple_of\":null,\"pad_id\":0,\"pad_type_id\":0,\"pad_token\":\"a\"}"}, {}));
            require(key(m, "tokenizer.ggml.eos_token_id").u == 13, "truncation and padding");
        }
        {
            // transformers saves the same settings in tokenizer_config.json and keeps them as init kwargs that no encode reads unless asked to cut or pad.
            ++cases;
            gguf::GGUFModel m;
            m.kv = read(directory({}, {"", "\"max_length\":2048,\"stride\":0,\"truncation_side\":\"right\",\"truncation_strategy\":\"longest_first\",\"padding_side\":\"left\""}));
            require(key(m, "tokenizer.ggml.eos_token_id").u == 13, "tokenizer_config.json truncation and padding");
        }
        refused({}, {"", "\"extra_special_tokens\":{\"x\":\"<nope>\"}"}, "extra_special_tokens <nope> is not an added token");
        refused({}, {"", "\"extra_special_tokens\":[\"<think>\",\"<nope>\"]"}, "extra_special_tokens <nope> is not an added token");
        refused({}, {"", "\"extra_special_tokens\":\"<think>\""}, "extra_special_tokens is not a map or a list");
        refused({"\"normalizer\":{\"type\":\"NFC\"}", "\"normalizer\":null"}, {}, "normalizer is not an object");
        refused({"\"normalizer\":{\"type\":\"NFC\"}", "\"normalizer\":{\"type\":\"NFKC\"}"}, {}, "normalizer type");
        refused({"\\\\s+(?!\\\\S)|", ""}, {}, "not the qwen2 or qwen35 pattern");
        refused({"\"Isolated\"", "\"Removed\""}, {}, "behavior");
        refused({"\"invert\":false", "\"invert\":true"}, {}, "invert");
        refused({"{\"type\":\"ByteLevel\",\"add_prefix_space\":false", "{\"type\":\"ByteLevel\",\"add_prefix_space\":true"}, {}, "add_prefix_space");
        refused({"\"post_processor\":{\"type\":\"ByteLevel\"", "\"post_processor\":{\"type\":\"TemplateProcessing\""}, {}, "post_processor type");
        refused({"\"byte_fallback\":false", "\"byte_fallback\":true"}, {}, "byte_fallback");
        refused({"", "\"extra\":1"}, {}, "key extra is not one");
        refused({"", "\"model\":1"}, {}, "repeats key model");
        refused({"\"hell\":10", "\"hell\":16"}, {}, "below the embedding's 16 rows");
        refused({"\"hell\":10", "\"hell\":9"}, {}, "gives id 9 twice");
        refused({"\"id\":14", "\"id\":10"}, {}, "takes an id the vocab gives");
        refused({"\"id\":14,\"content\":\"<think>\",\"single_word\":false,\"lstrip\":false", "\"id\":14,\"content\":\"<think>\",\"single_word\":false,\"lstrip\":true"},
                {}, "lstrip");
        refused({"\"id\":14,\"content\":\"<think>\",\"single_word\":false,\"lstrip\":false,\"rstrip\":false,\"normalized\":false",
                 "\"id\":14,\"content\":\"<think>\",\"single_word\":false,\"lstrip\":false,\"rstrip\":false,\"normalized\":true"}, {}, "normalized");
        refused({"\"h e\"", "\"he\""}, {}, "merge is not two parts");
        refused({"[\"l\",\"l\"]", "[\"l\",\"l l\"]"}, {}, "holds a space");
        refused({"[\"l\",\"l\"]", "[\"l\"]"}, {}, "neither a string nor a pair");
        refused({}, {"\"add_bos_token\":false", "\"add_bos_token\":true"}, "add_bos_token");
        refused({}, {"\"clean_up_tokenization_spaces\":false", "\"clean_up_tokenization_spaces\":true"}, "clean_up_tokenization_spaces");
        refused({}, {"\"split_special_tokens\":false", "\"split_special_tokens\":true"}, "split_special_tokens");
        refused({}, {"", "\"made_up_key\":\"left\""}, "key made_up_key is not one");
        refused({}, {"\"eos_token\":\"<|im_end|>\"", "\"eos_token\":\"<|end|>\""}, "is not an added token");
        refused({}, {"\"chat_template\":\"{{ messages }}\"", "\"chat_template\":[{\"name\":\"default\",\"template\":\"x\"}]"}, "not one template");
        refused({}, {"\"special\":true}}", "\"special\":false}}"}, "added_tokens_decoder 12 differs");
        ++cases;
        try {
            read(root / "absent");
            throw std::logic_error("a missing directory accepted");
        } catch (const std::runtime_error& e) {
            require(std::string(e.what()).find("cannot open file") != std::string::npos, "missing tokenizer.json");
        }
    } catch (const std::exception& e) {
        std::cerr << "hf-tokenizer: FAIL after " << cases << " cases: " << e.what() << "\n";
        std::error_code ec;
        fs::remove_all(root, ec);
        return 1;
    }
    std::error_code ec;
    fs::remove_all(root, ec);
    std::cout << "hf-tokenizer: " << cases << " cases passed\n";
    return 0;
}
