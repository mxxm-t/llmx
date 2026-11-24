"""Suite component `flags`: every flag of a model command takes effect, refuses with an error naming it, or says why it does nothing, on every architecture's fixture (docs/TESTS.md, flags)."""
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

import common

COMMANDS = ("generate", "chat", "logits", "perplexity", "bench", "serve")
PROMPT = "abcabcabcab"
TEXT = "abc" * 14
G = ["-n", "8", "--temp", "0", "--verbose"]
S = ["-n", "8", "--temp", "0.8", "--seed", "1", "--verbose"]
B = ["--p", "8", "--n", "4", "--r", "1"]
TEMPLATE = "{% for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}"


# Outcomes: what a row expects of a fixture with `tags` on the CPU (`cpu` true) or a device.
def works(observable):
    """Exit 0 and a line matching `observable` that the base run does not print, or the other way round."""
    return ("works", observable)


def refuses(text):
    """A non-zero exit with an `error:` line holding `text`, the flag's name."""
    return ("refuses", text)


def note(text):
    """Exit 0 with a line holding `text`, the reason the flag does nothing here."""
    return ("note", text)


def equal(case):
    """Exit 0 and the output of `case`: an alias, or a value that is the default."""
    return ("equal", case)


def elsewhere(component, flag):
    """The flag is proved by the component `component`, whose source names it."""
    return ("elsewhere", component, flag)


SKIP = ("skip",)


def by(**cases):
    """An outcome chosen by the fixture: the first of the keyword cases whose tag the fixture holds (`cpu` and `device` for where it runs), else `other`."""
    def pick(tags, cpu):
        held = set(tags) | {"cpu" if cpu else "device"}
        for tag, outcome in cases.items():
            if tag != "other" and set(tag.split("_")) <= held:
                return outcome
        return cases["other"]
    return pick


# One row a case: (name, command, args, outcome or by(...), base case, only on the first fixture).
# A row marked `one` checks a flag whose owner reads no weight (the sampler, the template, the scheduler's limits), so it runs on the qwen3 fixture alone.
ROWS = [
    ("base", "generate", G, None, None, False),
    ("ctk-f32", "generate", G + ["--cache-type-k", "f32", "--cache-type-v", "f32"], works(r"^kv: allocated"), "base", False),
    ("ctk-alias", "generate", G + ["-ctk", "f32", "-ctv", "f32"], equal("ctk-f32"), "base", False),
    ("ctk-f16", "generate", G + ["--cache-type-k", "f16", "--cache-type-v", "f16"], by(native=refuses("--cache-type-k"), other=equal("base")), "base", False),
    ("ctk-native", "generate", G + ["--cache-type-k", "native", "--cache-type-v", "native"], by(native=equal("base"), other=refuses("--cache-type-k")), "base", False),
    ("dtype-f32", "generate", G + ["--dtype", "f32"], works(r"^dtype:"), "base", False),
    ("dtype-bf16", "generate", G + ["--dtype", "bf16"], works(r"^dtype:"), "base", False),
    ("load-mapped", "generate", G + ["--load-mode", "mapped"], works(r"^load:"), "base", False),
    ("threads-2", "generate", G + ["--threads", "2"], None, None, False),
    ("threads-1", "generate", G + ["--threads", "1"], by(cpu=works(r"^threads:"), other=note("runs nothing on the CPU, so they change nothing")), "threads-2", False),
    ("threads-batch-1", "generate", G + ["--threads", "2", "--threads-batch", "1"], by(cpu=works(r"^threads: prefill"), other=note("runs nothing on the CPU, so they change nothing")), "threads-2", False),
    ("threads-batch-alias", "generate", G + ["--threads", "2", "-tb", "1"], equal("threads-batch-1"), "base", True),
    ("ubatch-2", "generate", G + ["--ubatch", "2"], works(r"^prefill: \d+ slices"), "base", False),
    ("kv-tokens-short", "generate", G + ["--kv-tokens", "16"], refuses("16 KV tokens"), "base", False),
    ("yarn-2", "generate", G + ["--yarn", "2"], works(r"^rope:"), "base", False),
    ("yarn-original", "generate", G + ["--yarn", "2", "--yarn-original-ctx", "64"], works(r"^rope:"), "yarn-2", False),
    ("yarn-original-alone", "generate", G + ["--yarn-original-ctx", "64"], refuses("--yarn-original-ctx"), "base", False),
    ("layer-shares-one", "generate", G + ["--layer-shares", "1,3"], refuses("layer shares"), "base", True),
    ("tensor-width-one", "generate", G + ["--tensor-width", "2"], refuses("--tensor-width"), "base", True),
    ("cpu-moe", "generate", G + ["--cpu-moe"], by(routed_cpu=refuses("--cpu-moe: the experts already run on the CPU"), routed=works(r"layer|expert"), other=refuses("--cpu-moe")), "base", False),
    ("n-cpu-moe-1", "generate", G + ["--n-cpu-moe", "1"], by(routed_cpu=refuses("--n-cpu-moe: the experts already run on the CPU"), routed=works(r"layer|expert"), other=refuses("--n-cpu-moe")), "base", False),
    ("n-cpu-moe-0", "generate", G + ["--n-cpu-moe", "0"], equal("base"), "base", False),
    ("moe-stream-from", "generate", G + ["--cpu-moe", "--moe-stream-from", "1"], by(routed_cpu=refuses("--cpu-moe"), routed=works(r"stream"), other=refuses("--cpu-moe")), "cpu-moe", False),
    ("moe-stream-alone", "generate", G + ["--moe-stream-from", "1"], refuses("--moe-stream-from"), "base", True),
    ("expert-cache", "generate", G + ["--cpu-moe", "--expert-cache", "1mb"], by(cpu=refuses("--expert-cache"), other=elsewhere("moe", "--expert-cache")), "base", False),
    ("lookup-file", "generate", G + ["--lookup-storage", "file"], by(lookup=works(r"^lookup reads|^load:"), other=refuses("--lookup-storage")), "base", False),
    ("lookup-device", "generate", G + ["--lookup-storage", "device"], by(lookup=equal("base"), other=refuses("--lookup-storage")), "base", False),
    ("drafter-lookup", "generate", G + ["--drafter", "lookup"], works(r"^drafts kept"), "base", False),
    ("drafter-off", "generate", G + ["--drafter", "off"], equal("base"), "base", False),
    ("drafter-embedded", "generate", G + ["--drafter", "embedded"], by(mtp=works(r"^drafts kept"), other=refuses("embedded drafter")), "base", False),
    ("draft-max-alone", "generate", G + ["--draft-max", "2"], refuses("--draft-max"), "base", False),
    ("draft-max-off", "generate", G + ["--drafter", "off", "--draft-max", "2"], refuses("--draft-max"), "base", True),
    ("drafts-long", "generate", ["-n", "32", "--temp", "0", "--verbose", "--drafter", "lookup"], None, None, True),
    ("draft-max-1", "generate", ["-n", "32", "--temp", "0", "--verbose", "--drafter", "lookup", "--draft-max", "1"], works(r"^drafts kept"), "drafts-long", True),
    ("max-tokens", "generate", ["-n", "4", "--temp", "0", "--verbose"], works(r"^ids:"), "base", True),
    ("max-tokens-long", "generate", ["--max-tokens", "4", "--temp", "0", "--verbose"], equal("max-tokens"), "base", True),
    ("sampled", "generate", S, works(r"^ids:"), "base", True),
    ("seed", "generate", ["-n", "8", "--temp", "0.8", "--seed", "2", "--verbose"], works(r"^ids:"), "sampled", True),
    ("topk", "generate", S + ["--topk", "1"], works(r"^ids:"), "sampled", True),
    ("topp", "generate", S + ["--topp", "0.3"], works(r"^ids:"), "sampled", True),
    ("min-p", "generate", S + ["--min-p", "0.99"], works(r"^ids:"), "sampled", True),
    ("penalty", "generate", G + ["--penalty", "1.8"], works(r"^ids:"), "base", True),
    ("presence-penalty", "generate", G + ["--presence-penalty", "2"], works(r"^ids:"), "base", True),
    ("frequency-penalty", "generate", G + ["--frequency-penalty", "2"], works(r"^ids:"), "base", True),
    ("ignore-eos", "generate", G + ["--ignore-eos"], elsewhere("server", "ignore_eos"), "base", True),
    ("stop", "generate", G + ["--stop", "zz"], elsewhere("cli", "--stop"), "base", True),
    ("chat", "generate", G + ["--chat"], works(r"^prompt tokens"), "base", True),
    ("chat-template", "generate", G + ["--chat", "--chat-template-file", "{template}"], works(r"^prompt tokens"), "chat", True),
    ("template-alone", "generate", G + ["--chat-template-file", "{template}"], refuses("--chat-template-file"), "base", True),
    ("file", "generate", ["--file", "{prompt}"] + G, equal("base"), "base", True),
    ("verbose", "generate", ["-n", "8", "--temp", "0"], works(r"^ids:"), "base", True),
    ("image-alone", "generate", G + ["--chat", "--image", "{template}"], refuses("--mmproj"), "base", True),
    ("mmproj", "generate", G, elsewhere("image", "--mmproj"), "base", True),
    ("image-max-tokens", "generate", G, elsewhere("image", "--image-max-tokens"), "base", True),
    ("device", "generate", G, elsewhere("cli", "--device"), "base", True),

    ("chat-base", "chat", ["-n", "4", "--temp", "0", "--verbose"], None, None, True),
    ("chat-system", "chat", ["-n", "4", "--temp", "0", "--verbose", "--system", "Answer in one word, always."], works(r"^Processing \d+ prompt tokens"), "chat-base", True),
    ("chat-draft-max", "chat", ["-n", "4", "--temp", "0", "--draft-max", "2"], refuses("--draft-max"), "chat-base", True),

    ("logits-base", "logits", ["--top", "5"], None, None, False),
    ("logits-top", "logits", ["--top", "2"], works(r"."), "logits-base", False),
    ("logits-last", "logits", ["--top", "2", "--last", "2"], works(r"."), "logits-top", True),
    ("logits-then", "logits", ["--top", "5", "--then-ids", "{ids}"], works(r"."), "logits-base", True),
    ("logits-per-token", "logits", ["--top", "5", "--per-token"], works(r"."), "logits-base", True),
    ("logits-chat", "logits", ["--top", "5", "--chat"], works(r"."), "logits-base", True),
    ("logits-template", "logits", ["--top", "5", "--chat", "--chat-template-file", "{template}"], works(r"."), "logits-chat", True),
    ("logits-file", "logits", ["--file", "{prompt}", "--top", "5"], equal("logits-base"), "logits-base", True),

    ("ppl-base", "perplexity", ["--verbose"], None, None, False),
    ("ppl-ctx", "perplexity", ["--verbose", "--ctx-size", "8"], works(r"^context size"), "ppl-base", False),
    ("ppl-chunks", "perplexity", ["--verbose", "--ctx-size", "8", "--chunks", "1"], works(r"^chunks"), "ppl-ctx", True),
    ("ppl-per-token", "perplexity", ["--verbose", "--per-token"], works(r"^threads: decode|^threads: prefill"), "ppl-base", True),
    ("ppl-quiet", "perplexity", [], works(r"^threads:"), "ppl-base", True),
    ("ppl-batch-per-token", "perplexity", ["--per-token", "--threads-batch", "1"], refuses("--threads-batch"), "ppl-base", True),
    ("ppl-file", "perplexity", ["--file", "{text}", "--verbose"], equal("ppl-base"), "ppl-base", True),

    ("bench-base", "bench", B, None, None, False),
    ("bench-p", "bench", ["--p", "4", "--n", "4", "--r", "1"], works(r"^bench: pp4"), "bench-base", True),
    ("bench-n", "bench", ["--p", "8", "--n", "2", "--r", "1"], works(r"^bench: tg2"), "bench-base", True),
    ("bench-r", "bench", ["--p", "8", "--n", "4", "--r", "2"], works(r"\(2 runs\)"), "bench-base", True),
    ("bench-seqs", "bench", B + ["--seqs", "2"], works(r"^bench: x2 tg4"), "bench-base", True),
    ("bench-depth", "bench", B + ["--depth", "8"], works(r"depth|d8|@"), "bench-base", True),
    ("bench-profile", "bench", B + ["--profile"], by(cpu=refuses("--profile"), other=SKIP), "bench-base", True),
    ("bench-drafter", "bench", B + ["--drafter", "embedded"], by(mtp=works(r"draft"), other=refuses("embedded drafter")), "bench-base", False),
    ("bench-draft-max", "bench", B + ["--draft-max", "2"], refuses("--draft-max"), "bench-base", True),
    ("bench-size", "bench", B + ["--size", "64"], refuses("--size"), "bench-base", True),
    ("bench-iters", "bench", B + ["--iters", "2"], refuses("--iters"), "bench-base", True),

    ("s-base", "serve", [], None, None, False),
    ("s-max-seqs", "serve", ["--max-seqs", "1"], works(r"limits\.active"), "s-base", True),
    ("s-max-queue", "serve", ["--max-queue", "1"], works(r"limits\.queued"), "s-base", True),
    ("s-ctx-size", "serve", ["--ctx-size", "32"], works(r"server\.request_tokens"), "s-base", True),
    ("s-ctx-alias", "serve", ["-c", "32"], equal("s-ctx-size"), "s-base", True),
    ("s-kv-tokens", "serve", ["--kv-tokens", "320"], works(r"server.kv_tokens="), "s-base", False),
    ("s-cache-type", "serve", ["--cache-type-k", "f32", "--cache-type-v", "f32"], works(r"server\.cache_type_k=f32"), "s-base", False),
    ("s-threads-2", "serve", ["--threads", "2"], None, None, False),
    ("s-threads", "serve", ["--threads", "1"], by(cpu=works(r"server\.threads=1"), other=note("runs nothing on the CPU, so they change nothing")), "s-threads-2", False),
    ("s-load-mode", "serve", ["--load-mode", "mapped"], works(r"server\.load_mode=mapped"), "s-base", False),
    ("s-dtype", "serve", ["--dtype", "f32"], works(r"precision\.requested"), "s-base", True),
    ("s-yarn", "serve", ["--yarn", "2"], works(r"rope\."), "s-base", False),
    ("s-ubatch", "serve", ["--ubatch", "2"], elsewhere("server", "--ubatch"), "s-base", True),
    ("s-state-checkpoints", "serve", ["--state-checkpoints", "1"], by(recurrent=works(r"state checkpoints"), other=refuses("--state-checkpoints")), "s-base", False),
    ("s-state-checkpoints-0", "serve", ["--state-checkpoints", "0"], by(recurrent=works(r"state checkpoints"), other=equal("s-base")), "s-base", False),
    ("s-host-cache", "serve", ["--host-cache-bytes", "1m"], works(r"reuse\.host\.now\.limit_bytes"), "s-base", True),
    ("s-disk", "serve", ["--host-cache-bytes", "8m", "--disk-cache-bytes", "64m", "--disk-cache-dir", "{disk}", "--disk-cache-floor", "0"], works(r"reuse\.disk\.now\.limit_bytes"), "s-base", True),
    ("s-disk-floor", "serve", ["--host-cache-bytes", "8m", "--disk-cache-bytes", "64m", "--disk-cache-dir", "{disk}", "--disk-cache-floor", "1m"], works(r"keeping 1 MiB free"), "s-disk", True),
    ("s-disk-keep", "serve", ["--host-cache-bytes", "8m", "--disk-cache-bytes", "64m", "--disk-cache-dir", "{disk}", "--disk-cache-floor", "0", "--disk-cache-keep"], works(r"kept for the next server"), "s-disk", True),
    ("s-disk-max-age", "serve", ["--host-cache-bytes", "8m", "--disk-cache-bytes", "64m", "--disk-cache-dir", "{disk}", "--disk-cache-floor", "0", "--disk-cache-max-age", "1h"], works(r"unused for 3600 s"), "s-disk", True),
    ("s-disk-adopt", "serve", ["--host-cache-bytes", "8m", "--disk-cache-bytes", "64m", "--disk-cache-dir", "{disk}", "--disk-cache-floor", "0", "--disk-cache-keep", "--disk-cache-adopt", "proved"],
     works(r"adopting proved"), "s-disk-keep", True),
    ("s-disk-dir-alone", "serve", ["--disk-cache-dir", "{disk}"], refuses("--disk-cache-dir"), "s-base", True),
    ("s-disk-floor-alone", "serve", ["--disk-cache-floor", "1m"], refuses("--disk-cache-floor"), "s-base", True),
    ("s-disk-keep-alone", "serve", ["--disk-cache-keep"], refuses("--disk-cache-keep"), "s-base", True),
    ("s-disk-max-age-alone", "serve", ["--disk-cache-max-age", "1h"], refuses("--disk-cache-max-age"), "s-base", True),
    ("s-disk-adopt-alone", "serve", ["--disk-cache-adopt", "exact"], refuses("--disk-cache-adopt"), "s-base", True),
    ("s-timing", "serve", ["--timing"], works(r"^timing"), "s-base", True),
    ("s-passes", "serve", ["--passes", "2"], refuses("passes"), "s-base", True),
    ("s-overflow", "serve", ["--ctx-size", "32", "--context-overflow", "shift"], works(r"^long reply"), "s-ctx-size", True),
    ("s-template", "serve", ["--chat-template-file", "{template}"], works(r"^chat reply"), "s-base", True),
    ("s-drafter", "serve", ["--drafter", "lookup"], works(r"drafting\.sources"), "s-base", False),
    ("s-draft-max", "serve", ["--drafter", "lookup"], elsewhere("server", "--draft-max"), "s-base", True),
    ("s-draft-max-alone", "serve", ["--draft-max", "2"], refuses("--draft-max"), "s-base", False),
    ("s-cpu-moe", "serve", ["--cpu-moe"], by(routed_cpu=refuses("--cpu-moe"), routed=works(r"topology"), other=refuses("--cpu-moe")), "s-base", False),
    ("s-lookup", "serve", ["--lookup-storage", "file"], by(lookup=works(r"topology\.lookups"), other=refuses("--lookup-storage")), "s-base", False),
    ("s-image-max-tokens", "serve", ["--image-max-tokens", "64"], refuses("--image-max-tokens"), "s-base", True),
    ("s-mmproj", "serve", [], elsewhere("image", "--mmproj"), "s-base", True),
    ("s-host", "serve", [], elsewhere("server", "--host"), "s-base", True),
    ("s-port", "serve", [], elsewhere("server", "--port"), "s-base", True),
    ("s-api-key-file", "serve", [], elsewhere("server", "--api-key-file"), "s-base", True),
    ("s-no-auth", "serve", [], elsewhere("server", "--no-auth"), "s-base", True),
]

# Health paths that change from run to run whatever the flags; they are left out of a comparison.
VOLATILE = re.compile(r"uptime|started|elapsed|_ms\b|_s\b|seconds|bytes_per_s|bytes_done|\.free|available|since_start|live\.|digest|adopt\.state|\.pass\b|in_flight|recent\[|disk\.now\.(ready|writing|entries|bytes|segments|states)|\.ms\b")


def flags_of(text):
    """The long spellings of the flags a help page lists as options."""
    return set(re.findall(r"^\s+(?:-[a-z]+(?: \S+)?, )?(--[a-z][a-z0-9-]*)", text, re.M)) | set(re.findall(r"^\s+(-[a-z]+)\b", text, re.M))


def short_forms():
    """kShortForms of src/cli/main.cpp: each short spelling with the long one it stands for."""
    with open(os.path.join(common.ROOT, "src", "cli", "main.cpp"), encoding="utf-8") as f:
        source = f.read()
    table = re.search(r"kShortForms\[\] = \{(.*?)\};", source, re.S)[1]
    return dict(re.findall(r'\{"(-[a-z]+)", "(--[a-z-]+)"\}', table)), source


def completeness():
    """Each flag a model command's help lists has a row of that command, or is one exec_flag parses for every model command and has a row of some command; each flag the parser reads is on a help page; each row's flag is on its command's page."""
    shorts, source = short_forms()
    longs = lambda flags: {shorts.get(f, f) for f in flags}
    pages = {}
    for command in COMMANDS:
        p = subprocess.run([common.exe_path(), command, "--help"], capture_output=True, encoding="utf-8", errors="replace")
        assert p.returncode == 0, "flags: %s --help failed" % command
        pages[command] = longs(flags_of(p.stdout)) - {"--help", "-h"}
    # The flags one parser reads for every command that calls it: exec_flag's and images_flag's.
    parsers = "".join(re.search(r"bool %s\(.*?\n}\n" % name, source, re.S)[0] for name in ("exec_flag", "images_flag"))
    shared = longs(re.findall(r'f == "(--[a-z-]+)"', parsers))
    rowed = {c: set() for c in COMMANDS}
    for name, command, args, outcome, _, _ in ROWS:
        named = longs(a for a in args if a.startswith("-") and not re.match(r"-\d", a))
        if isinstance(outcome, tuple) and outcome[0] == "elsewhere":
            named.add(outcome[2] if outcome[2].startswith("-") else "")
        rowed[command] |= named - {""}
    rowed["bench"].add("--model")  # every bench row runs a model
    # generate and chat read their flags in one loop, so generate's rows cover chat's flags but the ones only generate takes.
    rowed["chat"] |= rowed["generate"] - {"--file", "--chat", "--image", "--mmproj", "--image-max-tokens"}
    problems = []
    every = set().union(*rowed.values())
    for command in COMMANDS:
        for flag in sorted(pages[command] - rowed[command]):
            if not (flag in shared and flag in every):
                problems.append("%s %s has no row" % (command, flag))
        for flag in sorted(rowed[command] - pages[command]):
            problems.append("a row of %s names %s, which its help does not list" % (command, flag))
    parsed = longs(re.findall(r'f == "(--?[a-z][a-z-]*)"', source))
    other = {"--revision", "--cache-dir", "--parallel"}  # pull's, which reads no model
    for flag in sorted(parsed - set().union(*pages.values()) - other):
        problems.append("the parser reads %s, which no model command's help lists" % flag)
    for name, command, args, outcome, _, _ in ROWS:
        if isinstance(outcome, tuple) and outcome[0] == "elsewhere":
            with open(os.path.join(common.ROOT, "tests", outcome[1] + ".py"), encoding="utf-8") as f:
                if outcome[2] not in f.read():
                    problems.append("%s: tests/%s.py does not name %s" % (name, outcome[1], outcome[2]))
    return problems


def registry_names():
    with open(os.path.join(common.ROOT, "src", "model", "arch", "registry.hpp"), encoding="utf-8") as f:
        table = re.search(r"kArchitectures\[\] = \{(.*?)\n\};", f.read(), re.S)[1]
    return set(re.findall(r'^\s*\{"([a-z0-9-]+)"', table, re.M))


def norm(text):
    """The lines of a run with its times, rates, progress and the probe lines of a device's queue left out."""
    out = []
    for line in text.splitlines():
        line = re.sub(r"(Loading tensor data: [0-9% ]*)+", "", line).strip()
        if not line or line.startswith("queue:"):
            continue
        line = re.sub(r"\d+(\.\d+)?( \+- \d+(\.\d+)?)? tok/s", "T tok/s", line)
        # The disk cache's start line gives its age limit in seconds, a setting and not a time.
        if not line.startswith("server: disk cache of"):
            line = re.sub(r"\b\d+(\.\d+)? ?(s|ms|us)\b", "T", line)
        line = re.sub(r"\b\d+ MiB free, keeping", "N free, keeping", line)
        line = re.sub(r"127\.0\.0\.1:\d+|server-\d+-\d+", "ADDR", line)
        out.append(line)
    return out


def flat(value, path=""):
    if isinstance(value, dict):
        for k, v in value.items():
            yield from flat(v, path + "." + k if path else k)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from flat(v, "%s[%d]" % (path, i))
    else:
        yield path, value


def device():
    return os.environ.get("LLMX_DEVICE", "cpu")


def run_cli(command, model, args, files):
    argv = [common.exe_path(), command] + ([] if command == "bench" else [model])
    args = [a.format(**files) for a in args]
    if command in ("generate", "logits", "perplexity") and "--file" not in args:
        argv.append(TEXT if command == "perplexity" else PROMPT)
    if command == "bench":
        argv += ["--model", model]
    argv += args + ([] if "--device" in args else ["--device", device()])
    p = subprocess.run(argv, input="abc\n" if command == "chat" else "", capture_output=True, encoding="utf-8", errors="replace", timeout=300)
    return p.returncode, p.stdout + p.stderr, " ".join(argv)


def get(url, body=None):
    request = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=120) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def run_serve(model, args, files, name):
    """Start a server with `args`, read /v1/health and send a completion, a chat and a prompt longer than 32 tokens; the exit status (0 for a server that came up), its log with the replies and health's stable paths.
    A pool at the model's context and 8 MiB of host cache unless the row names them, so a device's free memory changes no comparison."""
    port = common.free_port()
    argv = [common.exe_path(), "serve", model, "--no-auth", "--host", "127.0.0.1", "--port", str(port), "--device", device()] + [a.format(**files) for a in args]
    argv += ([] if "--kv-tokens" in args else common.pool_args(model)) + ([] if "--host-cache-bytes" in args else ["--host-cache-bytes", "8m"])
    log = tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace")
    p = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
    lines = []
    try:
        deadline = time.time() + 120
        up = False
        while time.time() < deadline and p.poll() is None:
            try:
                status, body = get("http://127.0.0.1:%d/v1/health" % port)
                if status == 200 and json.loads(body).get("status") == "ok":
                    up = True
                    break
            except OSError:
                pass
            time.sleep(0.1)
        if up:
            base = "http://127.0.0.1:%d/v1/" % port
            _, reply = get(base + "completions", {"prompt": PROMPT, "max_tokens": 8, "temperature": 0})
            lines.append("reply " + reply.split('"usage"')[0].split('"text":')[-1][:200])
            _, reply = get(base + "chat/completions", {"messages": [{"role": "user", "content": "abcab"}], "max_tokens": 8, "temperature": 0})
            lines.append("chat reply " + reply.split('"usage"')[0].split('"content":')[-1][:200])
            status, _ = get(base + "completions", {"prompt": "abc" * 16, "max_tokens": 4, "temperature": 0})
            lines.append("long reply %d" % status)
            _, body = get(base + "health")
            health = json.loads(body)
            lines += ["%s=%s" % (k, v) for k, v in flat(health) if not VOLATILE.search(k) and not k.startswith("timing")]
            lines += ["timing"] if "timing" in health else []
        else:
            p.wait(60)
        rc = p.returncode
        log.seek(0)
        text = log.read()
    finally:
        if p.poll() is None:
            p.kill()
            p.wait()
        log.close()
    return (rc if not up else 0), text + "\n" + "\n".join(lines), " ".join(argv)


def judge(outcome, rc, out, runs, base):
    """None when the run is what `outcome` declares, else the failure's class and its evidence."""
    if rc is None or (isinstance(rc, int) and (rc < 0 or rc > 128)) or re.search(r"terminate called|Segmentation|Assertion|Aborted", out):
        return "CRASH"
    kind = outcome[0]
    errors = [l for l in out.splitlines() if l.startswith("error:")]
    if kind == "refuses":
        if rc == 0:
            return "IGNORED-ACCEPTED (a refusal naming %s expected)" % outcome[1]
        return None if any(outcome[1] in l for l in errors) else "UNCLEAR (%s)" % (errors[-1:] or out.splitlines()[-1:])
    if rc != 0:
        return "FAILED (%s)" % (errors[-1:] or out.splitlines()[-1:])
    if kind == "note":
        return None if outcome[1] in out else "IGNORED (no line saying %r)" % outcome[1]
    if kind == "equal":
        return None if norm(out) == norm(runs[outcome[1]][1]) else "WRONG (differs from %s)" % outcome[1]
    a, b = norm(out), norm(runs[base][1])
    changed = [l for l in a if l not in b] + [l for l in b if l not in a]
    if not changed:
        return "IGNORED (output identical to %s)" % base
    return None if any(re.search(outcome[1], l) for l in changed) else "CHANGED-OTHER (%s)" % changed[:3]


def run():
    if common.tensor_width() > 1 or any(os.environ.get(v) for v in ("LLMX_LAYER_SHARES", "LLMX_CACHE_TYPE", "LLMX_DTYPE", "LLMX_LOAD_MODE")) or device() != "cpu":
        print("flags: skipped, each flag is compared against a base run on the CPU without the suite's configured flags")
        return common.SKIPPED
    problems = completeness()
    cpu = device() == "cpu"
    with tempfile.TemporaryDirectory(prefix="llmx_flags_") as directory:
        fixtures = common.arch_fixtures(directory)
        missing = registry_names() - {arch for _, arch, _ in fixtures.values()}
        problems += ["the architecture %s has no fixture in common.arch_fixtures" % a for a in sorted(missing)]
        files = {"template": os.path.join(directory, "template.jinja"), "ids": os.path.join(directory, "ids.txt"),
                 "prompt": os.path.join(directory, "prompt.txt"), "text": os.path.join(directory, "text.txt"), "disk": os.path.join(directory, "disk")}
        for key, content in (("template", TEMPLATE), ("ids", "97,98"), ("prompt", PROMPT), ("text", TEXT)):
            with open(files[key], "w", encoding="utf-8", newline="") as f:
                f.write(content)
        checked = 0
        for index, (fixture, (model, arch, tags)) in enumerate(fixtures.items()):
            runs = {}
            for name, command, args, outcome, base, one in ROWS:
                if one and index:
                    continue
                if outcome is not None and callable(outcome):
                    outcome = outcome(tags, cpu)
                if outcome is not None and outcome[0] in ("elsewhere", "skip"):
                    continue
                if command == "serve":
                    shutil.rmtree(files["disk"], ignore_errors=True)
                    rc, out, line = run_serve(model, args, files, name)
                else:
                    rc, out, line = run_cli(command, model, args, files)
                runs[name] = (rc, out)
                if outcome is None:
                    if rc != 0:
                        problems.append("%s %s: the base run failed: %s\n  %s" % (fixture, name, out.splitlines()[-1:], line))
                    continue
                failure = judge(outcome, rc, out, runs, base)
                checked += 1
                if failure:
                    problems.append("%s %s: %s\n  %s" % (fixture, name, failure, line))
    for p in problems:
        print("  flags:", p)
    print("flags: %d flag cases over %d fixtures on %s, %d problems" % (checked, len(fixtures), device(), len(problems)))
    return not problems


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        common.EXE = os.path.abspath(sys.argv[1])
    raise SystemExit(0 if run() is not False else 1)
