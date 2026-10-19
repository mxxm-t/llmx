"""The runtime names no architecture: outside the modules under src/model/arch/, no file of the runtime, the loader, the passes, the server or the CLI names a registered architecture or a tensor, and neither does a header the modules share (AGENTS.md, Tests, the Arch boundary entry).

The registered names are read from src/model/arch/registry.hpp. Modules are the headers it includes or headers under src/model/arch/ that implement Architecture; the rest are shared.
A name counts wherever it stands as a word: in its own case anywhere, comments and strings included, and in any case outside a string, so a comment's "Qwen3" counts and a model id in a help text does not.
A tensor is a string literal that begins with "blk." or ends in ".weight" or ".bias".
The tokenizer is left out, since its pretokenizer names are keyed by the file's tokenizer and some match architecture names.
Standard library only; it reads the source tree and runs no binary.
"""
import glob
import os
import re

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
REGISTRY = "src/model/arch/registry.hpp"
LAYERS = ("src/model/*.hpp", "src/inference/*.hpp", "src/server/*.hpp", "src/cli/*.cpp")
TOKENS = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'', re.S)


def table(text):
    """The body of the registry's table, kArchitectures."""
    m = re.search(r"kArchitectures\[\]\s*=\s*\{(.*?)\};", text, re.S)
    return m.group(1) if m else ""


def registered(text):
    """The names of the registry's table."""
    return re.findall(r'\{\s*"([^"]+)"\s*,', table(text))


def implements_architecture(text):
    """A concrete architecture declaration, ignoring comments and literals."""
    return bool(re.search(r'\bclass\s+\w+\s+(?:final\s+)?[:]\s*public\s+Architecture\b',
                          TOKENS.sub('', text)))


def shared(text):
    """Headers that are neither registered modules nor an Architecture implementation awaiting registration."""
    modules = set(re.findall(r'#include\s+"model/arch/([^"/]+\.hpp)"', text))
    paths = []
    for p in glob.glob(os.path.join(ROOT, "src/model/arch/*.hpp")):
        if os.path.basename(p) in modules | {os.path.basename(REGISTRY)}:
            continue
        with open(p, encoding="utf-8") as f:
            if not implements_architecture(f.read()):
                paths.append(p)
    return sorted(paths)


def findings(path, text, names):
    """Each registered name and tensor literal in `text`, as (path, line, what), in the order they stand."""
    out = []
    line = lambda pos: text.count("\n", 0, pos) + 1
    unquoted = TOKENS.sub(lambda m: " " * len(m.group(0)) if m.group(0)[0] in "\"'" else m.group(0), text)
    for name in names:
        word = r"(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])" % re.escape(name)
        at = {m.start() for m in re.finditer(word, text)} | {m.start() for m in re.finditer(word, unquoted, re.I)}
        out += [(pos, "the architecture " + name) for pos in at]
    for m in TOKENS.finditer(text):
        token = m.group(0)
        if token.startswith('"'):
            value = token[1:-1]
            if value.startswith("blk.") or value.endswith(".weight") or value.endswith(".bias"):
                out.append((m.start(), "the tensor " + token))
    return [(path, line(pos), what) for pos, what in sorted(out)]


def self_test(text, names):
    """The check finds what it is for, and not a name inside a longer word, a tensor name in a comment or a model id in a string; and it read every row of the table."""
    assert implements_architecture('class NewModel final : public Architecture {};')
    assert implements_architecture('class NewModel : public Architecture {};')
    assert not implements_architecture('// class NewModel : public Architecture {};\nvoid helper();')
    assert not implements_architecture('const char* x = "class NewModel : public Architecture {};";')
    assert not implements_architecture('class Helper : public Other {};')
    planted = ('int x; // qwen3 in a comment\nf("blk.0.attn_q.weight"); g("output.weight"); h("a.bias");\n// "blk.1."\n'
               'int qwen3x = qwen3moe_count;\n// Qwen3\'s slots\nhelp("Qwen/Qwen3-0.6B");\n')
    got = [what for _, _, what in findings("planted", planted, ["qwen3", "qwen3moe"])]
    want = ["the architecture qwen3", 'the tensor "blk.0.attn_q.weight"', 'the tensor "output.weight"', 'the tensor "a.bias"',
            "the architecture qwen3"]
    assert got == want, "the boundary check's self-test found %s" % got
    rows = len(re.findall(r"^\s*\{", table(text), re.M))
    assert names and len(names) == rows, "%d architecture names read from the %d rows of %s" % (len(names), rows, REGISTRY)


def run():
    with open(os.path.join(ROOT, REGISTRY), encoding="utf-8") as f:
        text = f.read()
    names = registered(text)
    self_test(text, names)
    found = []
    paths = sorted({p for pattern in LAYERS for p in glob.glob(os.path.join(ROOT, pattern))} | set(shared(text)))
    for p in paths:
        with open(p, encoding="utf-8") as f:
            found += findings(os.path.relpath(p, ROOT).replace(os.sep, "/"), f.read(), names)
    for path, n, what in found:
        print("  %s:%d names %s" % (path, n, what))
    print("arch-boundary: %d files, %d architecture names, %d findings" % (len(paths), len(names), len(found)))
    return not found


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
