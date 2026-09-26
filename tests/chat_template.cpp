#include <fstream>
#include <iostream>
#include <iterator>
#include "core/json.hpp"
#include "inference/chat.hpp"

const jmini::Value& field(const jmini::Value& value, const char* name) {
    const auto* item = value.get(name);
    if (!item) throw std::runtime_error(std::string("missing fixture field: ") + name);
    return *item;
}

// Each expected string is what Jinja2 renders from the same messages when `tools` is passed as none, as it is for a request without tools.
struct DefinedCase { const char* tpl; const char* expected; };
const DefinedCase defined_cases[] = {
    // A date default of the kind other families' templates give a variable the caller did not pass.
    {R"({%- if not date_string is defined %}{%- set date_string = "26 Jul 2024" %}{%- endif %}Today Date: {{ date_string }})",
     "Today Date: 26 Jul 2024"},
    {R"({% if tools is defined and tools is none %}T{% endif %}{% if tool_calls is not defined and system is not defined %}N{% endif %}{% if messages is defined and eos_token is defined %}M{% endif %})",
     "TNM"},
    {R"({% for m in messages %}{{ m.role[0] }}{% if m.content is defined %}c{% endif %}{% if m.tool_calls is not defined %}-{% endif %}{% endfor %})",
     "sc-uc-"},
    {R"({% if messages[0]['content'] is defined %}K{% endif %}{% if messages[0]['name'] is not defined %}U{% endif %}{% if messages[1] is defined and messages[2] is not defined %}R{% endif %})",
     "KUR"},
    {R"({% set x = none %}{% set ns = namespace(a=none) %}{% if x is defined and ns.a is defined and ns.b is not defined %}D{% endif %})",
     "D"},
    {R"({% for m in messages %}{% if loop is defined %}L{% endif %}{% endfor %}{% if loop is not defined %}N{% endif %})",
     "LLN"},
};

// A conversation ending in an assistant turn under the Qwen3 template of the official repositories (the tokenizer_config.json of Qwen/Qwen3-0.6B, 8B and 14B), which finds the last user turn through `messages[::-1]`.
// The assistant turn after that user turn keeps its reasoning, as transformers 5.17.0 renders it.
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
        if (argc != 2) throw std::runtime_error("expected chat-template fixture path");
        std::ifstream input(argv[1], std::ios::binary);
        if (!input) throw std::runtime_error("cannot open chat-template fixture");
        const std::string text{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
        const auto fixture = jmini::Parser(text).parse();
        const auto& tpl = field(fixture, "template").str;
        size_t count = 0;
        for (const auto& test : field(fixture, "cases").arr) {
            std::vector<chat::Message> messages;
            for (const auto& message : field(test, "messages").arr)
                messages.push_back({field(message, "role").str, field(message, "content").str});
            const auto actual = chat::render(tpl, messages, field(test, "generate").b, "", "<|im_end|>");
            if (actual != field(test, "expected").str)
                throw std::runtime_error("Qwen template differs from Jinja2 at case " + std::to_string(count));
            ++count;
        }
        if (!count) throw std::runtime_error("empty chat-template fixture");
        std::cout << count << " Qwen chat-template cases match Jinja2\n";
        const std::vector<chat::Message> messages = {{"system", "S"}, {"user", "U"}};
        size_t differ = 0;
        for (size_t i = 0; i < std::size(defined_cases); ++i) {
            const auto actual = chat::render(defined_cases[i].tpl, messages, true, "", "<|im_end|>");
            if (actual == defined_cases[i].expected) continue;
            std::cerr << "defined case " << i << " renders \"" << actual << "\", Jinja2 renders \"" << defined_cases[i].expected << "\"\n";
            ++differ;
        }
        if (differ) throw std::runtime_error(std::to_string(differ) + " defined-test cases differ from Jinja2");
        std::cout << std::size(defined_cases) << " defined-test cases match Jinja2\n";
        const std::vector<chat::Message> ending = {{"system", "You are helpful."}, {"user", "Remember violet."}, {"assistant", "<think>Reason.</think>\n\nAnswer."}};
        const auto shipped = chat::render(qwen3_repositories, ending, false, "", "<|im_end|>");
        if (shipped != qwen3_repositories_expected)
            throw std::runtime_error("the Qwen3 template of the official repositories renders a conversation ending in an assistant turn as\n" + shipped +
                                     "\nwhere transformers renders\n" + qwen3_repositories_expected);
        std::cout << "the Qwen3 template of the official repositories renders a conversation ending in an assistant turn as transformers does\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
