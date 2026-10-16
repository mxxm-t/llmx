#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "core/json.hpp"
#include "core/sha.hpp"
#include "format/gguf.hpp"
#include "inference/chat.hpp"
#include "tokenizer/tokenizer.hpp"

// Checks chat templates against the HF reference renderer's output in a fixture tools/gen_chat_baseline.py writes (tests/data/baseline_chat_template.json or a --scan of GGUF files), and the renderer's own limits.
// A model's chat goldens given after it are rendered too, each under its fixture's template, and every failure is printed before the exit status says whether there was one (docs/src/inference-chat.md).

using chat::jj::Value;

const jmini::Value& field(const jmini::Value& value, const char* name) {
    const auto* item = value.get(name);
    if (!item) throw std::runtime_error(std::string("missing fixture field: ") + name);
    return *item;
}

std::vector<chat::Message> messages_of(const jmini::Value& list) {
    std::vector<chat::Message> messages;
    for (const auto& message : list.arr) {
        chat::Message m{ field(message, "role").str, field(message, "content").str, std::nullopt };
        if (const auto* reasoning = message.get("reasoning_content")) m.reasoning_content = reasoning->str;
        messages.push_back(std::move(m));
    }
    return messages;
}

// Whether every message is one chat::Message holds: a string role and content, and a string reasoning_content where there is one.
bool plain(const jmini::Value& list) {
    for (const auto& message : list.arr)
        for (const auto& kv : message.obj)
            if (!kv.second.isString() || (kv.first != "role" && kv.first != "content" && kv.first != "reasoning_content")) return false;
    return true;
}

// A case's render: through chat::Message where it holds the messages, and otherwise, with tools or tool calls or content parts, from the fixture's JSON into the variables chat::context sets.
std::string render_case(const chat::ChatFormat& format, const jmini::Value& c) {
    const jmini::Value& messages = field(c, "messages");
    const bool generate = field(c, "generate").b;
    const auto* tools = c.get("tools");
    if (!tools && plain(messages)) return format.render(messages_of(messages), generate);
    format.require();
    Value context = chat::context({}, generate, format.bos, format.eos);
    context.obj->set("messages", chat::template_value(messages));
    if (tools) context.obj->set("tools", chat::template_value(*tools));
    return format.program->render(context);
}

struct Tally {
    size_t cases = 0, failures = 0;
    void fail(const std::string& what) { ++failures; std::cerr << "FAIL " << what << "\n"; }
};

// The reference's failures whose message the renderer gives too: the template's own raise, undefined values, and Python's type and division errors.
// A recursion limit or a failure Python raises beyond what the renderer holds only has to fail.
bool same_message(const std::string& type) {
    return type == "TemplateError" || type == "TemplateRuntimeError" || type == "UndefinedError" || type == "TypeError" || type == "ZeroDivisionError";
}

// One render against the reference's: the text it gave, or the failure it raised.
void check_render(const std::string& where, const std::function<std::string()>& render, const jmini::Value& reference, Tally& tally) {
    ++tally.cases;
    const auto* expected = reference.get("expected");
    try {
        const std::string actual = render();
        if (!expected) tally.fail(where + ": renders where the reference raises " + field(reference, "error_type").str + ": " + field(reference, "error").str);
        else if (actual != expected->str) tally.fail(where + ": renders\n" + actual + "\nwhere the reference renders\n" + expected->str);
    } catch (const chat::TemplateError& e) {
        if (expected) { tally.fail(where + ": fails with \"" + e.what() + "\" where the reference renders"); return; }
        const std::string& type = field(reference, "error_type").str;
        if (same_message(type) && e.what() != field(reference, "error").str)
            tally.fail(where + ": fails with \"" + e.what() + "\" where the reference raises " + type + ": " + field(reference, "error").str);
    }
}

void check_cases(const std::string& name, const chat::ChatFormat& format, const jmini::Value& cases, Tally& tally) {
    for (const auto& c : cases.arr)
        check_render(name + " / " + field(c, "name").str, [&] { return render_case(format, c); }, c, tally);
}

// A conversation as a client sends it to the server's chat routes, read into chat::Message as the server reads it: content null beside calls as "", each call's arguments as given, and `tool_call_id` and `name`.
std::vector<chat::Message> client_messages(const chat::ChatFormat& format, const jmini::Value& list) {
    std::vector<chat::Message> messages;
    for (const auto& message : list.arr) {
        const auto* content = message.get("content");
        const std::string text = content && content->isString() ? content->str : std::string();
        const auto* reasoning = message.get("reasoning_content");
        if (reasoning) messages.push_back({ field(message, "role").str, text, reasoning->str });
        else if (field(message, "role").str == "assistant") messages.push_back(format.assistant(text));
        else messages.push_back({ field(message, "role").str, text, std::nullopt });
        if (const auto* calls = message.get("tool_calls"))
            for (const auto& c : calls->arr) {
                const auto& function = field(c, "function");
                messages.back().tool_calls.push_back({ field(c, "id").str, field(function, "name").str, field(function, "arguments") });
            }
        if (const auto* id = message.get("tool_call_id")) messages.back().tool_call_id = id->str;
        if (const auto* name = message.get("name")) messages.back().name = name->str;
    }
    return messages;
}

// The conversations a client sends with tool calls, through chat::context's own reading of the calls, against the reference's render of them as the server hands them on.
void check_server_cases(const std::string& name, const chat::ChatFormat& format, const jmini::Value& cases, Tally& tally) {
    for (const auto& c : cases.arr)
        check_render(name + " / " + field(c, "name").str,
                     [&] { return format.render(client_messages(format, field(c, "messages")), field(c, "generate").b); }, c, tally);
}

// The two-turn conversation of the fixture with each reply kept as chat and the server keep it, through ChatFormat::assistant.
void check_turns(const std::string& name, const chat::ChatFormat& format, const jmini::Value& fixture, const jmini::Value& t, Tally& tally) {
    const auto keeps = [](bool split) { return std::string(split ? "split" : "whole"); };
    if (format.split_turns != field(t, "split_turns").b)
        tally.fail(name + ": keeps an assistant turn " + keeps(format.split_turns) + " where Jinja's parse of the template keeps it " + keeps(!format.split_turns));
    const auto& reference = field(t, "reference_keeps");
    if (reference.isString() && reference.str != keeps(format.split_turns))
        tally.fail(name + ": keeps an assistant turn " + keeps(format.split_turns) + " where the reference's renders keep it " + reference.str);
    const auto before = messages_of(field(fixture, "turn_before"));
    const auto after = messages_of(field(fixture, "turn_after"));
    check_render(name + " / first turn", [&] { return format.render(before, true); }, field(t, "first"), tally);
    const auto& texts = field(fixture, "turn_texts").arr;
    const auto& turns = field(t, "turns").arr;
    if (texts.size() != turns.size()) throw std::runtime_error(name + ": the fixture's turns do not match its texts");
    for (size_t k = 0; k < texts.size(); ++k) {
        std::vector<chat::Message> messages = before;
        messages.push_back(format.assistant(texts[k].str));
        messages.insert(messages.end(), after.begin(), after.end());
        check_render(name + " / turn \"" + texts[k].str + "\"", [&] { return format.render(messages, true); }, turns[k], tally);
    }
}

// A template that must render or fail as the renderer's own limits say, never end the process: `renders` whether it renders, else it must fail with a message holding `failure`.
void check_limit(const std::string& what, const std::string& source, bool renders, const std::string& failure, Tally& tally) {
    ++tally.cases;
    const chat::ChatFormat format = chat::chat_format(source, "", "");
    if (!format.program) {
        if (renders || format.refusal.find(failure) == std::string::npos) tally.fail(what + ": refused: " + format.refusal);
        return;
    }
    try {
        format.render({ { "user", "x", std::nullopt } }, false);
        if (!renders) tally.fail(what + ": renders where it must fail with " + failure);
    } catch (const chat::TemplateError& e) {
        if (renders || std::string(e.what()).find(failure) == std::string::npos) tally.fail(what + ": fails with \"" + e.what() + "\"");
    }
}

// A template that must render `expected`, as Jinja 3.1.6 renders it.
void check_output(const std::string& what, const std::string& source, const std::string& expected, Tally& tally) {
    ++tally.cases;
    const chat::ChatFormat format = chat::chat_format(source, "", "");
    if (!format.program) return tally.fail(what + ": refused: " + format.refusal);
    try {
        const std::string got = format.render({ { "user", "x", std::nullopt } }, false);
        if (got != expected) tally.fail(what + ": renders \"" + got + "\" where Jinja renders \"" + expected + "\"");
    } catch (const chat::TemplateError& e) {
        tally.fail(what + ": fails with \"" + e.what() + "\"");
    }
}

// A reply split into its reasoning and content, inside a <think> the template opened (`opened`) or the reply opens: the whole text at once and cut at every byte and every pair of bytes, fed piece by piece, give the same parts.
void check_reply_split(const std::string& what, bool opened, const std::string& text, const std::string& reasoning, const std::string& content, Tally& tally) {
    auto split = [opened](const std::vector<std::string>& pieces) {
        chat::ReplySplit splitter(opened);
        chat::ReplySplit::Parts all;
        for (const std::string& piece : pieces) {
            const chat::ReplySplit::Parts p = splitter.feed(piece);
            all.reasoning += p.reasoning;
            all.content += p.content;
        }
        const chat::ReplySplit::Parts p = splitter.finish();
        all.reasoning += p.reasoning;
        all.content += p.content;
        return all;
    };
    std::vector<std::vector<std::string>> cuts = { { text } };
    for (size_t i = 0; i <= text.size(); ++i) {
        cuts.push_back({ text.substr(0, i), text.substr(i) });
        for (size_t j = i; j <= text.size(); ++j) cuts.push_back({ text.substr(0, i), text.substr(i, j - i), text.substr(j) });
    }
    std::vector<std::string> bytes;
    for (char c : text) bytes.push_back(std::string(1, c));
    cuts.push_back(bytes);
    for (const auto& pieces : cuts) {
        ++tally.cases;
        const chat::ReplySplit::Parts got = split(pieces);
        if (got.reasoning != reasoning || got.content != content)
            return tally.fail(what + ": split into reasoning \"" + got.reasoning + "\" and content \"" + got.content + "\" from " + std::to_string(pieces.size()) + " pieces");
    }
}

// A call as the test expects it: its function's name and its arguments.
struct Call {
    std::string name, arguments;
    bool operator==(const Call& o) const { return name == o.name && arguments == o.arguments; }
};

std::string calls_text(const std::vector<Call>& calls) {
    std::string out;
    for (const Call& c : calls) out += "[" + c.name + " " + c.arguments + "]";
    return out;
}

// A reply's content split into its text and its tool calls in `format`: the whole text at once and cut at every byte and every pair of bytes give the same content and calls, and each call's name comes with its first piece.
void check_tool_split(const std::string& what, chat::ToolFormat format, const chat::ToolTypes& typed, const std::string& text, const std::string& content,
                      const std::vector<Call>& calls, Tally& tally, bool single = false) {
    auto split = [&](const std::vector<std::string>& pieces, std::string& got, std::vector<Call>& found) {
        chat::ToolCallSplit splitter(format, typed, single);
        std::vector<chat::ToolCallDelta> deltas;
        for (const std::string& piece : pieces) {
            const chat::ToolCallSplit::Parts p = splitter.feed(piece);
            got += p.content;
            deltas.insert(deltas.end(), p.calls.begin(), p.calls.end());
        }
        const chat::ToolCallSplit::Parts p = splitter.finish();
        got += p.content;
        deltas.insert(deltas.end(), p.calls.begin(), p.calls.end());
        bool named = splitter.calls() == calls.size() && !splitter.open() && splitter.done() == (single && !calls.empty());
        for (const chat::ToolCallDelta& d : deltas) {
            if (d.index == found.size()) {
                named = named && !d.name.empty();
                found.push_back({ d.name, std::string() });
            } else {
                named = named && d.index + 1 == found.size() && d.name.empty();
            }
            if (d.index < found.size()) found[d.index].arguments += d.arguments;
        }
        return named;
    };
    std::vector<std::vector<std::string>> cuts = { { text } };
    for (size_t i = 0; i <= text.size(); ++i) {
        cuts.push_back({ text.substr(0, i), text.substr(i) });
        for (size_t j = i; j <= text.size(); j += 3) cuts.push_back({ text.substr(0, i), text.substr(i, j - i), text.substr(j) });
    }
    std::vector<std::string> bytes;
    for (char c : text) bytes.push_back(std::string(1, c));
    cuts.push_back(bytes);
    for (const auto& pieces : cuts) {
        ++tally.cases;
        std::string got;
        std::vector<Call> found;
        const bool named = split(pieces, got, found);
        if (got != content || !(found == calls) || !named)
            return tally.fail(what + ": split into content \"" + got + "\" and calls " + calls_text(found) + " from " + std::to_string(pieces.size()) +
                              " pieces, where the content is \"" + content + "\" and the calls " + calls_text(calls) + (named ? "" : ", or a call's name came late"));
    }
}

void check_tool_splits(Tally& tally) {
    using chat::ToolFormat;
    const chat::ToolTypes typed = { { "get_weather", { "days" } } };
    const std::string paris = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";
    check_tool_split("a JSON call alone", ToolFormat::json, typed, paris, "", { { "get_weather", "{\"city\": \"Paris\"}" } }, tally);
    check_tool_split("text and two JSON calls", ToolFormat::json, typed, "Let me check.\n" + paris + "\n<tool_call>\n{\"name\": \"noop\", \"arguments\": {}}\n</tool_call>",
                     "Let me check.", { { "get_weather", "{\"city\": \"Paris\"}" }, { "noop", "{}" } }, tally);
    check_tool_split("text after a JSON call", ToolFormat::json, typed, paris + "\n\nDone.", "Done.", { { "get_weather", "{\"city\": \"Paris\"}" } }, tally);
    check_tool_split("nested arguments", ToolFormat::json, typed, "<tool_call>{\"name\":\"f\",\"arguments\":{\"a\": {\"b\": [1, {\"c\": \"}\"}]}}}</tool_call>", "",
                     { { "f", "{\"a\": {\"b\": [1, {\"c\": \"}\"}]}}" } }, tally);
    check_tool_split("an escaped name", ToolFormat::json, typed, "<tool_call>\n{\"name\": \"a\\\"b\", \"arguments\": {}}\n</tool_call>", "", { { "a\"b", "{}" } }, tally);
    check_tool_split("an opening that is no call", ToolFormat::json, typed, "Look: <tool_call>\nnot json\n</tool_call> end", "Look: <tool_call>\nnot json\n</tool_call> end", {}, tally);
    check_tool_split("arguments first", ToolFormat::json, typed, "<tool_call>{\"arguments\": {}, \"name\": \"f\"}</tool_call>", "<tool_call>{\"arguments\": {}, \"name\": \"f\"}</tool_call>", {}, tally);
    check_tool_split("a call with arguments that are not JSON", ToolFormat::json, typed, "<tool_call>\n{\"name\": \"f\", \"arguments\": {\"a\": }\n</tool_call>", "", { { "f", "{\"a\":" } }, tally);
    check_tool_split("a JSON call cut short", ToolFormat::json, typed, "<tool_call>\n{\"name\": \"f\", \"arguments\": {\"a\": 1", "", { { "f", "{\"a\": 1" } }, tally);
    check_tool_split("an opening cut short", ToolFormat::json, typed, "x <tool_call>\n{\"na", "x <tool_call>\n{\"na", {}, tally);
    check_tool_split("text that only looks like an opening", ToolFormat::json, typed, "a < b <tool_c\n", "a < b <tool_c\n", {}, tally);
    check_tool_split("a reply without calls", ToolFormat::json, typed, "Plain reply.\n\n", "Plain reply.\n\n", {}, tally);
    // With one call a reply (parallel_tool_calls false), the reply ends at the first call's close: what follows it is not read.
    check_tool_split("two JSON calls, one a reply", ToolFormat::json, typed, "Let me check.\n" + paris + "\n<tool_call>\n{\"name\": \"noop\", \"arguments\": {}}\n</tool_call>\nDone.",
                     "Let me check.", { { "get_weather", "{\"city\": \"Paris\"}" } }, tally, true);
    check_tool_split("a reply without calls, one a reply", ToolFormat::json, typed, "Plain <tool_c reply.", "Plain <tool_c reply.", {}, tally, true);

    const std::string fn = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>";
    check_tool_split("a function call", ToolFormat::xml, typed, fn, "", { { "get_weather", "{\"city\": \"Paris\", \"days\": 3}" } }, tally);
    check_tool_split("text and two function calls", ToolFormat::xml, typed, "Checking.\n\n" + fn + "\n<tool_call>\n<function=noop>\n</function>\n</tool_call>", "Checking.",
                     { { "get_weather", "{\"city\": \"Paris\", \"days\": 3}" }, { "noop", "{}" } }, tally);
    check_tool_split("a value of several lines", ToolFormat::xml, typed, "<tool_call>\n<function=f>\n<parameter=text>\na \"b\"\n\nc\\d\n</parameter>\n</function>\n</tool_call>", "",
                     { { "f", "{\"text\": \"a \\\"b\\\"\\u000a\\u000ac\\\\d\"}" } }, tally);
    check_tool_split("a typed value that is not JSON", ToolFormat::xml, typed, "<tool_call>\n<function=get_weather>\n<parameter=days>\nthree\n</parameter>\n</function>\n</tool_call>", "",
                     { { "get_weather", "{\"days\": \"three\"}" } }, tally);
    check_tool_split("an untyped value that looks like JSON", ToolFormat::xml, typed, "<tool_call>\n<function=f>\n<parameter=n>\n3\n</parameter>\n</function>\n</tool_call>", "",
                     { { "f", "{\"n\": \"3\"}" } }, tally);
    check_tool_split("a function opening that is no call", ToolFormat::xml, typed, "<tool_call>\nhello\n</tool_call>", "<tool_call>\nhello\n</tool_call>", {}, tally);
    check_tool_split("a function call cut inside a parameter", ToolFormat::xml, typed, "<tool_call>\n<function=f>\n<parameter=a>\n1\n</parameter>\n<parameter=b>\nx", "",
                     { { "f", "{\"a\": \"1\"}" } }, tally);
    check_tool_split("a function call without its closing tag", ToolFormat::xml, typed, "<tool_call>\n<function=f>\n<parameter=a>\n1\n</parameter>\n</tool_call>", "",
                     { { "f", "{\"a\": \"1\"}" } }, tally);
    check_tool_split("two function calls, one a reply", ToolFormat::xml, typed, fn + "\n<tool_call>\n<function=noop>\n</function>\n</tool_call>", "",
                     { { "get_weather", "{\"city\": \"Paris\", \"days\": 3}" } }, tally, true);
    check_tool_split("text inside a call", ToolFormat::xml, typed, "<tool_call>\n<function=f>\njunk\n<parameter=a>\n1\n</parameter>\n</function>\n</tool_call>", "",
                     { { "f", "{\"a\": \"1\"}" } }, tally);
}

// A tokenizer of one token a byte and the Qwen templates' special tokens, <|im_end|> its end of text.
bpe::Tokenizer qwen_byte_tokenizer() {
    gguf::GGUFModel m;
    gguf::MetaValue tokens, types, eos;
    tokens.vtype = types.vtype = gguf::V_ARRAY;
    tokens.u = gguf::V_STRING;
    types.u = gguf::V_UINT32;
    const auto bytes = bpe::build_byte_encoder();
    const auto add = [&](const std::string& t, uint64_t type) {
        gguf::MetaValue v, k;
        v.vtype = gguf::V_STRING;
        v.s = t;
        k.vtype = gguf::V_UINT32;
        k.u = type;
        tokens.arr.push_back(v);
        types.arr.push_back(k);
    };
    for (int b = 0; b < 256; ++b) add(bytes.at((uint8_t)b), 1);
    add("<|im_start|>", 3);
    add("<|im_end|>", 3);
    eos.vtype = gguf::V_UINT32;
    eos.u = 257;
    m.kv.push_back({ "tokenizer.ggml.tokens", tokens });
    m.kv.push_back({ "tokenizer.ggml.token_type", types });
    m.kv.push_back({ "tokenizer.ggml.eos_token_id", eos });
    return bpe::Tokenizer(m);
}

// Under a pinned template: chat_format reads the tool-call format Jinja's parse gives, a turn of calls read back as a reply and sent back renders the same bytes, and the next turn begins a tool result's ids.
// The end tokens place each message's end where its render ends, trailing whitespace aside, wherever that render begins the prompt.
void check_tool_template(const std::string& name, const chat::ChatFormat& format, const jmini::Value& t, const bpe::Tokenizer& tok, Tally& tally) {
    static const char* const names[] = { "none", "json", "xml" };
    ++tally.cases;
    if (names[(int)format.tool_format] != field(t, "tool_format").str)
        tally.fail(name + ": reads the tool-call format " + names[(int)format.tool_format] + " where the fixture records " + field(t, "tool_format").str);
    const auto args = [](const std::string& text) {
        jmini::Value v;
        v.t = jmini::Value::T::String;
        v.str = text;
        return v;
    };
    const chat::ToolTypes typed = { { "get_weather", { "days" } } };
    if (format.tool_format != chat::ToolFormat::none)
        for (const std::string& content : { std::string(), std::string("Let me check.") }) {
            ++tally.cases;
            const std::vector<chat::Message> ask = { { "user", "Weather in Paris?", std::nullopt } };
            chat::Message turn{ "assistant", content, std::nullopt };
            turn.tool_calls = { { "call_1", "get_weather", args("{\"city\": \"Paris\", \"days\": 3}") }, { "call_2", "noop", args("{}") } };
            const std::string where = name + " / a turn of calls" + (content.empty() ? "" : " after text");
            try {
                const std::string prompt = format.render(ask, true), whole = format.render({ ask[0], turn }, false);
                const size_t end = whole.rfind("<|im_end|>");
                if (whole.compare(0, prompt.size(), prompt) != 0 || end == std::string::npos || end < prompt.size()) {
                    tally.fail(where + ": the turn's render does not begin with the prompt");
                    continue;
                }
                bool open = true;
                const chat::Message back = chat::reply_message(format, chat::opens_reasoning(prompt), format.tool_format, typed, whole.substr(prompt.size(), end - prompt.size()), &open);
                const std::string again = format.render({ ask[0], back }, false);
                if (open || back.tool_calls.size() != 2 || again != whole)
                    tally.fail(where + ": read back as " + std::to_string(back.tool_calls.size()) + " calls and content \"" + back.content + "\", it renders\n" + again + "\nwhere the turn renders\n" + whole);
                ++tally.cases;
                std::vector<chat::Message> after = { ask[0], back, { "tool", "21", std::nullopt } };
                const std::vector<uint32_t> next = chat::stable_prefix(format, tok, { ask[0], back }, false), full = tok.encode(format.render(after, true));
                if (next.empty() || next.size() > full.size() || !std::equal(next.begin(), next.end(), full.begin()))
                    tally.fail(where + ": the next turn's " + std::to_string(next.size()) + " ids do not begin a tool result's render");
            } catch (const chat::TemplateError& e) {
                tally.fail(where + ": fails with \"" + e.what() + "\"");
            }
        }
    for (const bool system : { false, true }) {
        std::vector<chat::Message> c = { { "user", "a", std::nullopt }, { "assistant", "b", std::nullopt }, { "user", "c", std::nullopt } };
        if (system) c.insert(c.begin(), { "system", "s", std::nullopt });
        const std::string where = name + " / the end tokens of a conversation " + (system ? "with" : "without") + " a system message";
        ++tally.cases;
        const std::string prompt = format.render(c, true);
        const std::vector<uint32_t> ids = tok.encode(prompt);
        std::vector<size_t> points, bytes;
        size_t at = 0;
        for (size_t i = 0; i < ids.size(); ++i) {
            if (tok.is_eos(ids[i])) points.push_back(i + 1);
            bytes.push_back(at += tok.decode({ ids[i] }).size());
        }
        const size_t own = chat::own_end_tokens(format, tok, c, {});
        if (points.size() != c.size() + own) {
            tally.fail(where + ": " + std::to_string(points.size()) + " end tokens for " + std::to_string(c.size()) + " messages and " + std::to_string(own) + " of the template's own");
            continue;
        }
        size_t compared = 0;
        for (size_t k = 1; k <= c.size(); ++k) {
            std::string part;
            try { part = format.render(std::vector<chat::Message>(c.begin(), c.begin() + (std::ptrdiff_t)k), false); }
            catch (const chat::TemplateError&) { continue; }
            if (prompt.compare(0, part.size(), part) != 0) continue;
            ++compared;
            size_t len = part.size();
            while (len > 0 && (part[len - 1] == '\n' || part[len - 1] == ' ')) --len;
            const size_t rows = (size_t)(std::upper_bound(bytes.begin(), bytes.end(), len) - bytes.begin());
            if (rows != points[own + k - 1]) tally.fail(where + ": message " + std::to_string(k) + " ends at row " + std::to_string(points[own + k - 1]) + " by the end tokens and " + std::to_string(rows) + " by its render");
        }
        if (!compared) tally.fail(where + ": no render of the conversation up to a message begins its prompt");
    }
}

std::string repeated(const std::string& s, size_t n) {
    std::string out;
    for (size_t k = 0; k < n; ++k) out += s;
    return out;
}

// A conversation ending in an assistant turn under the Qwen3 template of the official repositories (the tokenizer_config.json of Qwen/Qwen3-0.6B, 8B and 14B), which finds the last user turn through `messages[::-1]`.
// The assistant turn after that user turn keeps its reasoning, as transformers 5.17.0 renders it; the old renderer dropped it, and the fixture holds the same case under the template's SHA-256 pin.
const char* const qwen3_repositories =
    "{%- if tools %}\n"
    "    {{- '<|im_start|>system\\n' }}\n"
    "    {%- if messages[0].role == 'system' %}\n"
    "        {{- messages[0].content + '\\n\\n' }}\n"
    "    {%- endif %}\n"
    "    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n"
    "    {%- for tool in tools %}\n"
    "        {{- \"\\n\" }}\n"
    "        {{- tool | tojson }}\n"
    "    {%- endfor %}\n"
    "    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n"
    "{%- else %}\n"
    "    {%- if messages[0].role == 'system' %}\n"
    "        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n"
    "    {%- endif %}\n"
    "{%- endif %}\n"
    "{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n"
    "{%- for message in messages[::-1] %}\n"
    "    {%- set index = (messages|length - 1) - loop.index0 %}\n"
    "    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n"
    "        {%- set ns.multi_step_tool = false %}\n"
    "        {%- set ns.last_query_index = index %}\n"
    "    {%- endif %}\n"
    "{%- endfor %}\n"
    "{%- for message in messages %}\n"
    "    {%- if message.content is string %}\n"
    "        {%- set content = message.content %}\n"
    "    {%- else %}\n"
    "        {%- set content = '' %}\n"
    "    {%- endif %}\n"
    "    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n"
    "        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n"
    "    {%- elif message.role == \"assistant\" %}\n"
    "        {%- set reasoning_content = '' %}\n"
    "        {%- if message.reasoning_content is string %}\n"
    "            {%- set reasoning_content = message.reasoning_content %}\n"
    "        {%- else %}\n"
    "            {%- if '</think>' in content %}\n"
    "                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n"
    "                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n"
    "            {%- endif %}\n"
    "        {%- endif %}\n"
    "        {%- if loop.index0 > ns.last_query_index %}\n"
    "            {%- if loop.last or (not loop.last and reasoning_content) %}\n"
    "                {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n"
    "            {%- else %}\n"
    "                {{- '<|im_start|>' + message.role + '\\n' + content }}\n"
    "            {%- endif %}\n"
    "        {%- else %}\n"
    "            {{- '<|im_start|>' + message.role + '\\n' + content }}\n"
    "        {%- endif %}\n"
    "        {%- if message.tool_calls %}\n"
    "            {%- for tool_call in message.tool_calls %}\n"
    "                {%- if (loop.first and content) or (not loop.first) %}\n"
    "                    {{- '\\n' }}\n"
    "                {%- endif %}\n"
    "                {%- if tool_call.function %}\n"
    "                    {%- set tool_call = tool_call.function %}\n"
    "                {%- endif %}\n"
    "                {{- '<tool_call>\\n{\"name\": \"' }}\n"
    "                {{- tool_call.name }}\n"
    "                {{- '\", \"arguments\": ' }}\n"
    "                {%- if tool_call.arguments is string %}\n"
    "                    {{- tool_call.arguments }}\n"
    "                {%- else %}\n"
    "                    {{- tool_call.arguments | tojson }}\n"
    "                {%- endif %}\n"
    "                {{- '}\\n</tool_call>' }}\n"
    "            {%- endfor %}\n"
    "        {%- endif %}\n"
    "        {{- '<|im_end|>\\n' }}\n"
    "    {%- elif message.role == \"tool\" %}\n"
    "        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n"
    "            {{- '<|im_start|>user' }}\n"
    "        {%- endif %}\n"
    "        {{- '\\n<tool_response>\\n' }}\n"
    "        {{- content }}\n"
    "        {{- '\\n</tool_response>' }}\n"
    "        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n"
    "            {{- '<|im_end|>\\n' }}\n"
    "        {%- endif %}\n"
    "    {%- endif %}\n"
    "{%- endfor %}\n"
    "{%- if add_generation_prompt %}\n"
    "    {{- '<|im_start|>assistant\\n' }}\n"
    "    {%- if enable_thinking is defined and enable_thinking is false %}\n"
    "        {{- '<think>\\n\\n</think>\\n\\n' }}\n"
    "    {%- endif %}\n"
    "{%- endif %}";
const char* const qwen3_repositories_expected =
    "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nRemember violet.<|im_end|>\n<|im_start|>assistant\n<think>\nReason.\n</think>\n\nAnswer.<|im_end|>\n";

// How much of a prompt a follow-up turn begins with (chat::stable_prefix), over a tokenizer of one token a byte below 128: the ids of the conversation rendered without the generation prompt where they prefix the prompt's, else 0.
void check_stable_prefix(Tally& tally) {
    gguf::GGUFModel m;
    gguf::MetaValue tokens;
    tokens.vtype = gguf::V_ARRAY;
    tokens.u = gguf::V_STRING;
    const auto bytes = bpe::build_byte_encoder();
    for (int b = 0; b < 128; ++b) {
        gguf::MetaValue t;
        t.vtype = gguf::V_STRING;
        t.s = bytes.at((uint8_t)b);
        tokens.arr.push_back(t);
    }
    m.kv.push_back({ "tokenizer.ggml.tokens", tokens });
    const bpe::Tokenizer tok(m);
    const std::vector<chat::Message> messages = { { "user", "Hello there.", std::nullopt }, { "assistant", "Hi.", std::nullopt }, { "user", "Go on.", std::nullopt } };
    const std::string turns = "{% for m in messages %}<{{ m.role }}>{{ m.content }}\n{% endfor %}";
    const chat::ChatFormat plain = chat::chat_format(turns + "{% if add_generation_prompt %}<assistant>\n{% endif %}", "", "");
    const std::vector<uint32_t> prompt = tok.encode(plain.render(messages, true));
    const size_t stable = tok.encode(plain.render(messages, false)).size();
    ++tally.cases;
    if (stable >= prompt.size() || chat::stable_prefix(plain, tok, messages, prompt) != stable)
        tally.fail("the stable prefix of a conversation is not its render without the generation prompt");
    ++tally.cases;
    const std::vector<uint32_t> other(prompt.begin(), prompt.begin() + 5);
    if (chat::stable_prefix(plain, tok, messages, other) != 5) tally.fail("the stable prefix runs past the prompt it prefixes");
    const chat::ChatFormat refusing =
        chat::chat_format("{% if not add_generation_prompt %}{{ raise_exception('a generation prompt is required') }}{% endif %}" + turns + "<assistant>\n", "", "");
    ++tally.cases;
    if (chat::stable_prefix(refusing, tok, messages, prompt) != 0) tally.fail("a template refusing the render without the generation prompt gives a stable prefix");
    ++tally.cases;
    if (chat::stable_prefix(plain, tok, { { "user", "caf\xc3\xa9", std::nullopt } }, prompt) != 0) tally.fail("a text the tokenizer refuses gives a stable prefix");

    // After a reply, the ids a next turn begins with: up to the next user turn's text, or up to the reply's end while it is written, and without the reasoning where the template renders it only for the last turn.
    std::vector<chat::Message> replied = messages;
    replied.push_back({ "assistant", "Sure thing.", std::string("Thinking.") });
    const std::string before = "<user>Hello there.\n<assistant>Hi.\n<user>Go on.\n<assistant>Sure thing.";
    ++tally.cases;
    if (chat::stable_prefix(plain, tok, replied, false) != tok.encode(before + "\n<user>")) tally.fail("a next turn after a whole reply does not begin where its user turn does");
    ++tally.cases;
    if (chat::stable_prefix(plain, tok, replied, true) != tok.encode(before)) tally.fail("a next turn after a reply being written runs past what is written");
    const chat::ChatFormat reasoning = chat::chat_format(
        "{% for m in messages %}<{{ m.role }}>{% if loop.last and m.reasoning_content %}[{{ m.reasoning_content }}]{% endif %}{{ m.content }}\n{% endfor %}"
        "{% if add_generation_prompt %}<assistant>\n{% endif %}", "", "");
    ++tally.cases;
    if (chat::stable_prefix(reasoning, tok, replied, false) != tok.encode(before + "\n<user>"))
        tally.fail("a next turn keeps the reasoning a template renders only for the last turn");
    replied.back().content = "caf\xc3\xa9";
    ++tally.cases;
    if (!chat::stable_prefix(plain, tok, replied, false).empty()) tally.fail("a reply the tokenizer refuses gives a next turn");
}

int main(int argc, char** argv) {
    try {
        if (argc < 2) throw std::runtime_error("usage: llmx-chat-template-test FIXTURE.json...");
        Tally tally;
        const bpe::Tokenizer qwen_bytes = qwen_byte_tokenizer();
        size_t templates = 0, features = 0, refused = 0, splits = 0, goldens = 0;
        std::map<std::string, std::string> sources;   // the templates read so far, by SHA-256
        for (int arg = 1; arg < argc; ++arg) {
            std::ifstream input(argv[arg], std::ios::binary);
            if (!input) throw std::runtime_error(std::string("cannot open ") + argv[arg]);
            const std::string text{ std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>() };
            const auto fixture = jmini::Parser(text).parse();
            const std::string& bos = field(fixture, "bos_token").str;
            const std::string& eos = field(fixture, "eos_token").str;

            // A model's chat goldens: each case's messages rendered under the file's template, against the reference's render.
            if (const auto* pinned = fixture.get("template_sha256")) {
                const auto source = sources.find(pinned->str);
                if (source == sources.end()) throw std::runtime_error(std::string(argv[arg]) + ": its template is in no fixture read before it");
                const chat::ChatFormat format = chat::chat_format(source->second, bos, eos);
                for (const auto& c : field(fixture, "cases").arr) {
                    ++goldens;
                    ++tally.cases;
                    const std::string where = std::string(argv[arg]) + " / " + field(c, "name").str, &expected = field(c, "text").str;
                    try {
                        const std::string actual = render_case(format, c);
                        if (actual != expected) tally.fail(where + ": renders\n" + actual + "\nwhere the reference renders\n" + expected);
                    } catch (const chat::TemplateError& e) {
                        tally.fail(where + ": fails with \"" + e.what() + "\" where the reference renders");
                    }
                }
                continue;
            }
            for (const auto& t : field(fixture, "templates").arr) {
                ++templates;
                const std::string& name = field(t, "name").str;
                const std::string& source = field(t, "template").str;
                core::Sha sha(true);
                sha.update(source.data(), source.size());
                if (sha.hex() != field(t, "sha256").str) { tally.fail(name + ": the template does not match its SHA-256 pin"); continue; }
                sources[sha.hex()] = source;
                const chat::ChatFormat format = chat::chat_format(source, bos, eos);
                if (!format.program) { tally.fail(name + ": refused: " + format.refusal); continue; }
                check_cases(name, format, field(t, "cases"), tally);
                check_server_cases(name, format, field(t, "server_cases"), tally);
                check_tool_template(name, format, t, qwen_bytes, tally);
                check_turns(name, format, fixture, t, tally);
            }
            if (const auto* list = fixture.get("features"))
                for (const auto& f : list->arr) {
                    ++features;
                    const std::string name = "feature " + field(f, "name").str;
                    const chat::ChatFormat format = chat::chat_format(field(f, "template").str, bos, eos);
                    if (!format.program) { tally.fail(name + ": refused: " + format.refusal); continue; }
                    check_cases(name, format, field(f, "cases"), tally);
                }
            if (const auto* list = fixture.get("refused"))
                for (const auto& r : list->arr) {
                    ++refused;
                    const chat::ChatFormat format = chat::chat_format(field(r, "template").str, bos, eos);
                    if (format.program) tally.fail("refusal " + field(r, "name").str + ": parsed, though it holds " + field(r, "why").str);
                }
            if (const auto* list = fixture.get("splits"))
                for (const auto& s : list->arr) {
                    ++splits;
                    const chat::Message turn = chat::assistant_turn(field(s, "text").str);
                    const auto& reasoning = field(s, "reasoning_content");
                    const bool none = reasoning.t == jmini::Value::T::Null;
                    if (turn.role != "assistant" || turn.content != field(s, "content").str || turn.reasoning_content.has_value() == none ||
                        (!none && *turn.reasoning_content != reasoning.str))
                        tally.fail("split of \"" + field(s, "text").str + "\": reply \"" + turn.content + "\", reasoning " +
                                   (turn.reasoning_content ? "\"" + *turn.reasoning_content + "\"" : std::string("none")));
                }
        }
        // A date format the C runtime does not take renders as that runtime writes it or fails the render, as Python's strftime does on the same platform, and never ends the process, as the MSVC runtime does without a handler of the thread's own.
        {
            const chat::ChatFormat format = chat::chat_format("{{ strftime_now('%Q %-d %Ez') }}", "", "");
            try { format.render({ { "user", "x", std::nullopt } }, false); } catch (const chat::TemplateError&) {}
        }
        check_stable_prefix(tally);
        check_tool_splits(tally);
        {
            ++tally.cases;
            const chat::ChatFormat format = chat::chat_format(qwen3_repositories, "", "<|im_end|>");
            const std::vector<chat::Message> ending = { { "system", "You are helpful.", std::nullopt }, { "user", "Remember violet.", std::nullopt },
                                                        { "assistant", "<think>Reason.</think>\n\nAnswer.", std::nullopt } };
            const std::string shipped = format.render(ending, false);
            if (shipped != qwen3_repositories_expected)
                tally.fail("the Qwen3 template of the official repositories renders a conversation ending in an assistant turn as\n" + shipped +
                           "\nwhere transformers renders\n" + qwen3_repositories_expected);
        }
        // Templates that would take the host stack past its end, or build values no render can free or print, are refused or fail where Python has no such limit.
        check_limit("a sum of 200000 terms", "{{ 1" + repeated(" + 1", 199999) + " }}", false, "nested too deeply", tally);
        check_limit("200000 attributes", "{{ x" + repeated(".y", 200000) + " }}", false, "nested too deeply", tally);
        check_limit("200000 filters", "{{ x" + repeated("|trim", 200000) + " }}", false, "nested too deeply", tally);
        check_limit("a map naming map 5000 times", "{{ 'a'|map(" + repeated("'map', ", 5000) + "'upper') }}", false,
                    "maximum recursion depth exceeded", tally);
        check_limit("a macro recursing 99 deep through ten statements a level",
                    "{% macro f(n) %}" + repeated("{% for i in [1] %}{% if true %}", 5) + "{% if n > 0 %}{{ f(n - 1) }}{% endif %}" +
                    repeated("{% endif %}{% endfor %}", 5) + "{% endmacro %}{{ f(99) }}", false, "maximum recursion depth exceeded", tally);
        check_limit("a macro recursing 20 deep", "{% macro f(n) %}{% if n > 0 %}{{ f(n - 1) }}{% endif %}{% endmacro %}{{ f(20) }}", true, "", tally);
        check_limit("a list nested 1000 deep", "{% set ns = namespace(v=[]) %}{% for i in range(1000) %}{% set ns.v = [ns.v] %}{% endfor %}", false,
                    "a value nested more than 100 deep is not supported", tally);
        check_limit("a namespace holding itself", "{% set ns = namespace(a=1) %}{% set ns.me = ns %}{{ ns }}", false,
                    "a namespace holding a namespace is not supported", tally);
        check_limit("a chain of namespaces", "{% set ns = namespace(a=1) %}{% set outer = namespace(inner=ns) %}", false,
                    "a namespace holding a namespace is not supported", tally);
        check_limit("a namespace in a list in a namespace", "{% set ns = namespace(a=1) %}{% set ns.all = [ns] %}", false,
                    "a namespace holding a namespace is not supported", tally);
        // A filter or test given more positional arguments than it reads fails, as Jinja fails it, rather than drop them; join's second, Jinja's attribute, fails where Jinja would render other text.
        check_limit("upper given an argument", "{{ 'a'|upper(1) }}", false, "upper() takes at most 0 arguments", tally);
        check_limit("first given an argument", "{{ [1]|first(1) }}", false, "first() takes at most 0 arguments", tally);
        check_limit("eq given two arguments", "{{ [1, 2]|select('eq', 1, 2)|list }}", false, "eq() takes at most 1 arguments", tally);
        check_limit("odd given an argument", "{{ [1, 2]|select('odd', 1)|list }}", false, "odd() takes at most 0 arguments", tally);
        check_limit("join given an attribute", "{{ ['a', 'b']|join(', ', 'x') }}", false, "join's attribute argument is not supported", tally);
        // The tests Jinja names by operator, reached by string through the select filters.
        check_output("the operator-named tests",
                     "{{ [1, 2, 1, 3]|select('==', 1)|list }}{{ [1, 2, 3]|reject('<', 2)|list }}{{ [1, 2, 3]|select('>=', 2)|list }}"
                     "{{ [1, 2]|select('!=', 1)|list }}{{ [1, 2, 3]|select('<=', 2)|list }}{{ [1, 2, 3]|select('>', 2)|list }}",
                     "[1, 1][2, 3][2, 3][2][1, 2][3]", tally);
        // A reply the template opened inside <think>: the reasoning before the first </think> and the reply after it, the newlines around the reasoning and those opening the reply dropped, however the text arrives.
        check_reply_split("a reasoned reply", true, "Plan it.\n</think>\n\nHello!", "Plan it.", "Hello!", tally);
        check_reply_split("newlines around the reasoning", true, "\n\nline one\n\nline two\n\n</think>\n\nA\nB\n", "line one\n\nline two", "A\nB\n", tally);
        check_reply_split("a reply cut off while it reasons", true, "still thinking <\n", "still thinking <", "", tally);
        check_reply_split("a partial closing tag that is not one", true, "a </thin> b </think>c", "a </thin> b ", "c", tally);
        check_reply_split("a second </think> in the reply", true, "r</think>x</think>y", "r", "x</think>y", tally);
        check_reply_split("no reasoning", true, "</think>\n\nonly the reply", "", "only the reply", tally);
        // A reply that opens its own <think>, as Qwen3's do, splits the same way; one that does not is all content, byte for byte.
        check_reply_split("a reply that opens <think>", false, "<think>\nPlan it.\n</think>\n\nHello!", "Plan it.", "Hello!", tally);
        check_reply_split("<think> after newlines", false, "\n<think>x</think>y", "x", "y", tally);
        check_reply_split("a reply without <think>", false, "\nHello </think> <think>!", "", "\nHello </think> <think>!", tally);
        check_reply_split("a partial <think> that is not one", false, "<thin", "", "<thin", tally);
        check_reply_split("<think> not at the start", false, "a<think>b</think>c", "", "a<think>b</think>c", tally);
        // A prompt leaves the reply inside <think> when its last <think> is not closed.
        for (const auto& [prompt, open] : std::vector<std::pair<std::string, bool>>{
                 { "<|im_start|>assistant\n<think>\n", true }, { "<|im_start|>assistant\n<think>\n\n</think>\n\n", false },
                 { "<|im_start|>assistant\n", false }, { "<think>a</think><|im_start|>assistant\n<think>\n", true } }) {
            ++tally.cases;
            if (chat::opens_reasoning(prompt) != open) tally.fail("opens_reasoning of \"" + prompt + "\"");
        }
        std::cout << templates << " templates and " << features << " feature templates over " << tally.cases << " cases, "
                  << refused << " refusals, " << splits << " splits and " << goldens << " model chat goldens against the HF reference renderer: " << tally.failures << " failed\n";
        if (!templates && !features) throw std::runtime_error("no templates in the fixture");
        return tally.failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
