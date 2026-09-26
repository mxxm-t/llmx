#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include "core/json.hpp"
#include "core/sha.hpp"
#include "inference/chat.hpp"

// Checks chat templates against what the HF reference renderer gives for them, from a fixture tools/gen_chat_baseline.py writes: tests/data/baseline_chat_template.json, or a --scan of GGUF files.
// Each pinned template must match its SHA-256, parse, and render every case byte for byte, or fail where the reference fails, with the same message where Python's message is one the renderer gives.
// Each also keeps an assistant turn split or whole as the rule read from Jinja's parse of it says, and as the reference's renders say where they decide it, and renders a two-turn conversation with the turn kept that way as the reference renders it.
// The feature templates are checked the same way, the refused ones must be refused, and chat::assistant_turn must split each text as the Qwen templates split it.
// Every failure is printed before the exit status says whether there was one.

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

// A fixture's JSON as the renderer's values, as Python's json module reads it; a whole number is an int, so the fixtures write no float with a whole value.
Value value_of(const jmini::Value& v) {
    switch (v.t) {
        case jmini::Value::T::Null: return Value::none();
        case jmini::Value::T::Bool: return Value::boolean(v.b);
        case jmini::Value::T::Number:
            if (v.num == double(int64_t(v.num)) && std::abs(v.num) < 9e15) return Value::integer(int64_t(v.num));
            return Value::number(v.num);
        case jmini::Value::T::String: return Value::str(v.str);
        case jmini::Value::T::Array: {
            std::vector<Value> items;
            for (const auto& x : v.arr) items.push_back(value_of(x));
            return Value::list(std::move(items));
        }
        case jmini::Value::T::Object: {
            auto o = std::make_shared<chat::jj::Object>();
            for (const auto& kv : v.obj) o->set(kv.first, value_of(kv.second));
            return Value::dict(o);
        }
    }
    return Value::none();
}

// A case's render: through chat::Message where it holds the messages, and otherwise, with tools or tool calls or content parts, from the fixture's JSON into the variables chat::context sets.
std::string render_case(const chat::ChatFormat& format, const jmini::Value& c) {
    const jmini::Value& messages = field(c, "messages");
    const bool generate = field(c, "generate").b;
    const auto* tools = c.get("tools");
    if (!tools && plain(messages)) return format.render(messages_of(messages), generate);
    format.require();
    Value context = chat::context({}, generate, format.bos, format.eos);
    context.obj->set("messages", value_of(messages));
    if (tools) context.obj->set("tools", value_of(*tools));
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

int main(int argc, char** argv) {
    try {
        if (argc < 2) throw std::runtime_error("usage: llmx-chat-template-test FIXTURE.json...");
        Tally tally;
        size_t templates = 0, features = 0, refused = 0, splits = 0;
        for (int arg = 1; arg < argc; ++arg) {
            std::ifstream input(argv[arg], std::ios::binary);
            if (!input) throw std::runtime_error(std::string("cannot open ") + argv[arg]);
            const std::string text{ std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>() };
            const auto fixture = jmini::Parser(text).parse();
            const std::string& bos = field(fixture, "bos_token").str;
            const std::string& eos = field(fixture, "eos_token").str;

            for (const auto& t : field(fixture, "templates").arr) {
                ++templates;
                const std::string& name = field(t, "name").str;
                const std::string& source = field(t, "template").str;
                core::Sha sha(true);
                sha.update(source.data(), source.size());
                if (sha.hex() != field(t, "sha256").str) { tally.fail(name + ": the template does not match its SHA-256 pin"); continue; }
                const chat::ChatFormat format = chat::chat_format(source, bos, eos);
                if (!format.program) { tally.fail(name + ": refused: " + format.refusal); continue; }
                check_cases(name, format, field(t, "cases"), tally);
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
        std::cout << templates << " templates and " << features << " feature templates over " << tally.cases << " cases, "
                  << refused << " refusals and " << splits << " splits against the HF reference renderer: " << tally.failures << " failed\n";
        if (!templates && !features) throw std::runtime_error("no templates in the fixture");
        return tally.failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
