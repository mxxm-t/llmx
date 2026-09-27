"""Stale docs: references in the Markdown that the tree no longer has.

Only what a reference names is checked, never whether prose about the code is still true; that is the merge sweep's (AGENTS.md, Dead code and stale docs).
Each finding is keyed by its check, file and name, and held to tests/data/known_findings.txt, which a new finding and a listed one that no longer occurs both fail.
Standard library only; the command and usage checks ask the built binary for its help pages.
"""
import ast
import collections
import io
import json
import os
import re
import shlex
import sys
import tempfile
import time
import tokenize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common
import cli

CHECKS = ("link", "path", "line-pin", "name", "command", "usage", "test-name", "src-page", "section")


def doc_files(texts):
    return sorted(p for p in texts if p in ("README.md", "AGENTS.md") or re.fullmatch(r"docs/(src/)?[^/]+\.md", p))


# ----------------------------------------------------------------------------------------------------------------------
# Regions: live text is held to the tree; dated records and plans keep the names they had or will have.

DATED = re.compile(r"20\d\d-\d\d-\d\d")
PLAN_HEADINGS = ("Order of work", "Open questions", "Not chosen", "Out of scope", "Non-goals", "Migration order")
PLAN_DOCS = ("docs/QWEN35.md", "docs/ROADMAP.md")
STATUS_LIVE = ("Status table", "Active feature blocks", "Working rules and ownership")


def heading_region(doc, title, level):
    """The region a heading opens, or None when it inherits its parent's."""
    if doc == "docs/STATUS.md" and level == 2:
        return "live" if title.startswith(STATUS_LIVE) else "plan" if "planned" in title else "record"
    if "planned" in title:
        return "plan"
    if DATED.search(title):
        return "record"
    if level == 2 and title.startswith(PLAN_HEADINGS):
        return "plan"
    return None


def doc_lines(doc, text):
    """(line number, line, region, fenced) for every line of `doc` but a fence's own."""
    fence, stack = False, {}
    base = "plan" if doc in PLAN_DOCS else "live"
    for n, line in enumerate(text.split("\n"), 1):
        if line.lstrip().startswith("```"):
            fence = not fence
            continue
        if not fence:
            m = re.match(r"(#{1,6})\s+(.*?)\s*$", line)
            if m:
                level = len(m.group(1))
                stack = {k: v for k, v in stack.items() if k < level}
                stack[level] = heading_region(doc, m.group(2), level)
        region = next((stack[k] for k in sorted(stack, reverse=True) if stack[k]), base)
        yield n, line, region, fence


def headings(text):
    out, fence = [], False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            fence = not fence
        elif not fence:
            m = re.match(r"#{1,6}\s+(.*?)\s*#*\s*$", line)
            if m:
                out.append(m.group(1))
    return out


def slugs(text):
    """The anchors GitHub gives a document's headings: lower case, punctuation but - and _ dropped, spaces as -, a repeat numbered."""
    seen, out = collections.Counter(), set()
    for h in headings(text):
        s = re.sub(r"[^\w\- ]", "", h.strip().lower()).replace(" ", "-")
        out.add(s if not seen[s] else "%s-%d" % (s, seen[s]))
        seen[s] += 1
    return out


# ----------------------------------------------------------------------------------------------------------------------
# What the tree has: files, the names its code uses, class members, tests, and every command's flags.

ID = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
CODE_TOPS = ("src/", "tests/", "tools/", "cmake/", "docker/", ".github/")


def python_code(text):
    """Python source without its comments, strings kept."""
    try:
        return " ".join(t.string for t in tokenize.generate_tokens(io.StringIO(text).readline) if t.type != tokenize.COMMENT)
    except (tokenize.TokenError, SyntaxError):
        return text


class Tree:
    def __init__(self, texts, paths, root=common.ROOT, help_pages=None):
        self.texts, self.paths = texts, paths
        self.dirs = {"/".join(p.split("/")[:i]) for p in paths for i in range(1, p.count("/") + 1)}
        self.by_base = collections.defaultdict(list)
        for p in sorted(paths):
            self.by_base[p.rsplit("/", 1)[-1]].append(p)
        self.stems = {os.path.splitext(p.rsplit("/", 1)[-1])[0] for p in paths}
        code = {}
        for p, t in texts.items():
            if not (p.startswith(CODE_TOPS) or p in ("CMakeLists.txt", "build.bat")) or p.endswith(".md"):
                continue
            if p.endswith((".hpp", ".cpp", ".h", ".comp", ".glsl", ".in")):
                code[p] = common.cpp_code(t, keep_strings=True)
            elif p.endswith(".py"):
                code[p] = python_code(t)
            else:
                code[p] = "\n".join(line for line in t.split("\n") if not line.lstrip().startswith("#"))
        self.code = code
        self.code_text = "\n".join(code.values())
        self.tokens = set(ID.findall(self.code_text))
        self.json_keys = set()
        for p in sorted(paths):
            if p.startswith("tests/data/") and p.endswith(".json") and os.path.getsize(os.path.join(root, p)) < 8000000:
                with open(os.path.join(root, p), encoding="utf-8") as f:
                    self.keys(json.load(f))
        self.members, self.bases = collections.defaultdict(set), collections.defaultdict(set)
        head = re.compile(r"\b(?:struct|class|union|enum\s+class|enum|namespace)\s+(?:alignas\s*\([^)]*\)\s*)?([A-Za-z_]\w*(?:\s*::\s*[A-Za-z_]\w*)*)([^;{()]*)\{")
        for p, c in code.items():
            if not p.endswith((".hpp", ".cpp", ".h")):
                continue
            for m in head.finditer(c):
                depth, j = 0, m.end() - 1
                while j < len(c):
                    depth += (c[j] == "{") - (c[j] == "}")
                    if not depth:
                        break
                    j += 1
                body = set(ID.findall(c[m.end():j]))
                for name in re.split(r"\s*::\s*", m.group(1)):
                    self.members[name] |= body
                if m.group(2).strip().startswith(":"):
                    self.bases[m.group(1)] |= set(re.findall(r"(?:public|private|protected)?\s*(?:\w+::)*(\w+)\s*(?:<[^>]*>)?\s*(?:,|$)", m.group(2).strip()[1:]))
        for p, t in texts.items():
            if p.endswith(".py") and p.startswith(("tests/", "tools/")):
                self.members[os.path.splitext(p.rsplit("/", 1)[-1])[0]] |= set(ID.findall(code[p]))
        cmake = texts["CMakeLists.txt"]
        self.ctests = dict(re.findall(r"add_test\(NAME ([\w-]+) COMMAND ([\w-]+)", cmake))
        self.targets = dict(re.findall(r"add_executable\(([\w-]+)\s+([\w/.]+)", cmake))
        runner = texts["tests/run_tests.py"]
        self.components = dict(re.findall(r'\("([\w-]+)",\s*(?:lambda:\s*)?(\w+)\.run', runner))
        self.build_text = "\n".join(t for p, t in texts.items() if p in ("CMakeLists.txt", "build.bat") or p.startswith(("cmake/", ".github/", "docker/")))
        self.tool_flags = {}
        for p, t in texts.items():
            if p.startswith(("tools/", "tests/")) and p.endswith(".py"):
                flags = set()
                for m in re.finditer(r"add_argument\(([^)]*)", t):
                    flags |= set(re.findall(r"""["'](-{1,2}[A-Za-z][\w-]*)["']""", m.group(1)))
                self.tool_flags[p] = flags or set(re.findall(r"""["'](--?[A-Za-z][\w-]*)["']""", t))
            elif p.startswith(("tools/", "tests/")) and p.endswith(".cpp"):
                self.tool_flags[p] = set(re.findall(r'"(--?[A-Za-z][\w-]*)"', t))
        self.help = help_pages if help_pages is not None else read_help_pages()
        self.commands = {c: set().union(*(g["spellings"] for g in groups)) for c, groups in self.help.items() if c}

    def keys(self, o, depth=0):
        if isinstance(o, dict):
            for k, v in o.items():
                self.json_keys.add(k)
                self.keys(v, depth + 1)
        elif isinstance(o, list):
            for v in o[:200]:
                self.keys(v, depth + 1)

    def known(self, name):
        """Whether an identifier is one the tree has: in code or a string, a test data key, a file stem or a test's name."""
        return name in self.tokens or name in self.json_keys or name in self.stems or name in self.ctests or name in self.components \
            or name in self.targets

    def member(self, scope, name, seen=None):
        if name in self.members.get(scope, ()) or (scope + "::" + name) in self.code_text:
            return True
        seen = seen if seen is not None else set()
        for b in self.bases.get(scope, ()):
            if b not in seen:
                seen.add(b)
                if self.member(b, name, seen):
                    return True
        return False

    def resolve(self, path, doc):
        """Whether a repository path as a doc writes it exists: from the root, under src/, beside the doc, by basename or by suffix; a glob when any file matches."""
        q = path.replace("\\", "/").lstrip("./") if not path.startswith("../") else path.replace("\\", "/")
        q = q.rstrip("/")
        if not q:
            return True
        near = os.path.normpath(os.path.join(os.path.dirname(doc), q)).replace(os.sep, "/")
        cands = (q, "src/" + q, near)
        everything = self.paths | self.dirs
        if "*" in q:
            pats = [re.compile(re.escape(c).replace(r"\*", "[^/]*") + "$") for c in cands]
            return any(p.match(f) for p in pats for f in everything)
        if any(c in everything for c in cands):
            return True
        if "/" not in q:
            return q in self.by_base
        return any(f.endswith("/" + q) for f in everything)


def read_help_pages():
    """Each command's option groups from the binary's help pages: [{spellings, default}], by command; the overview's commands under None."""
    pages = {}
    with tempfile.TemporaryDirectory(prefix="llmx_docs_") as directory:
        overview = cli.help_page(None, directory)
        commands = re.findall(r"^  ([a-z]+) ", overview, re.M)
        pages[None] = [{"spellings": {c}, "default": None} for c in commands]
        for c in commands:
            groups = []
            for line in cli.help_page(c, directory).splitlines():
                m = re.match(r"  (-\S.*?)(?:  +(.*)|$)", line)
                if m:
                    groups.append({"spellings": {s.split(" ", 1)[0] for s in m.group(1).split(", ")}, "text": m.group(2) or ""})
                elif groups and re.match(r"\s{4,}\S", line):
                    groups[-1]["text"] += " " + line.strip()
                elif not line.strip():
                    groups.append({"spellings": set(), "text": ""})
            groups = [g for g in groups if g["spellings"]]
            for g in groups:
                d = re.search(r"\(default: ([^)]*)\)", g.pop("text"))
                g["default"] = d.group(1).strip() if d else None
            pages[c] = groups
    return pages


# ----------------------------------------------------------------------------------------------------------------------
# The references a line makes.

MD_LINK = re.compile(r"!?\[((?:[^\[\]]|\[[^\]]*\])*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
SPAN = re.compile(r"(`+)(.+?)\1")
FILE_LINE = re.compile(r"(?<![\w/.-])((?:[\w.-]+/)*[\w.-]+\.(?:hpp|cpp|h|py|md|comp|glsl|yml|txt|cmake|in|bat|json|sh)):(\d+)")
PIN = re.compile(r"\bat\s+`?[0-9a-f]{7,40}\b")
BARE_PATH = re.compile(r"(?<![\w/`.<-])((?:docs|src|tests|tools|cmake|docker|\.github)/[\w./*-]*[\w/*])")
ROOT_FILES = ("CMakeLists.txt", "build.bat", "README.md", "AGENTS.md", ".gitignore")
RUNTIME = ("build/", "build\\", "generated/", "~", "<", "%", "$", "/", "C:", "c:", ".cache/", "models--")
SECTION = re.compile(r"\(`((?:docs/)?[A-Z][\w-]*\.md)`,\s+([A-Z][^)`]*)\)")
ROADMAP_ITEM = re.compile(r"ROADMAP(?:\.md)?`?\s*\(?#(\d+[a-z]?)\b")


def is_path(t):
    return " " not in t and ("/" in t or re.search(r"\.(?:hpp|cpp|h|py|md|comp|glsl|json|yml|bat|in|cmake|txt|sh)$", t)) \
        and not re.search(r"[(){}\[\]=,;\"'|]", t) and not re.match(r"^(?:https?:|-)", t)


def rooted(t, src_dirs):
    q = t.replace("\\", "/").lstrip("./")
    return q.startswith(("src/", "docs/", "tests/", "tools/", "cmake/", "docker/", ".github/") + tuple(d + "/" for d in src_dirs)) \
        or q in ROOT_FILES or ("/" not in q and re.search(r"\.(?:hpp|cpp|comp|glsl|py)$", q) is not None) \
        or re.fullmatch(r"[A-Z][A-Z0-9-]*\.md", q) is not None


def branch_like(p):
    """Whether a path that is not in the tree may be a branch, such as `tools/server-load-v2`: no extension and no closing `/`, as a file or a directory would have."""
    return not p.endswith("/") and "." not in p.rsplit("/", 1)[-1]


def sentence_at(block, pos):
    """The sentence of `block` around `pos`: from the last full stop before it to the next."""
    start = max(block.rfind(". ", 0, pos), block.rfind(".\n", 0, pos))
    end = min([i for i in (block.find(". ", pos), block.find(".\n", pos)) if i >= 0] or [len(block)])
    return block[start + 1:end + 1]


def blocks(lines):
    """The paragraph or list item each (number, line) belongs to, as its joined text and the line's offset in it."""
    out, cur = {}, []

    def close():
        text, off = "\n".join(l for _, l in cur), 0
        for k, l in cur:
            out[k] = (text, off)
            off += len(l) + 1

    for n, line in lines:
        s = line.strip()
        if not s or re.match(r"(?:[-*+]|\d+[.)])\s|#|\|", s):
            close()
            cur = [(n, line)] if s else []
        else:
            cur.append((n, line))
    close()
    return out


def command_words(text):
    try:
        words = shlex.split(text.replace("\\", "/"), posix=True)
    except ValueError:
        words = text.replace("\\", "/").split()
    return words


def flags_in(words):
    return [w.split("=")[0].rstrip(",.;:)") for w in words if re.match(r"-{1,2}[A-Za-z]", w)]


def check_command(tree, text, prefixed=True):
    """The problems of one command line: an llmx command it lacks or a flag the command does not take, or a flag a script or a built tool does not take.
    Returns None for a line that is not one of the tree's commands, else a list of problems."""
    words = command_words(text.strip().lstrip("$> ").strip())
    if not words:
        return None
    first = words[0]
    if re.search(r"(^|/)llmx(\.exe)?$", first) and prefixed or not prefixed and first in tree.commands:
        rest = words[1:] if prefixed else words
        if not rest or rest[0].startswith(("-", "<")):
            bad = [f for f in flags_in(rest) if f not in ("--version", "--help", "-h")]
            return ["llmx %s" % f for f in bad]
        if rest[0] not in tree.commands:
            return ["llmx " + rest[0]] if re.fullmatch(r"[a-z]+", rest[0]) else []
        return ["%s %s" % (rest[0], f) for f in flags_in(rest[1:]) if f not in tree.commands[rest[0]] | {"--help", "-h"}]
    m = re.search(r"(?:^|/)(llmx-[\w-]+?)(?:\.exe)?$", first)
    if m and prefixed:
        source = tree.targets.get(m.group(1))
        if not source:
            return [m.group(1)]
        return ["%s %s" % (m.group(1), f) for f in flags_in(words[1:]) if f not in tree.tool_flags.get(source, set()) | {"--help", "-h"}]
    if not prefixed and first in tree.ctests:
        source = tree.targets.get(tree.ctests[first])
        return ["%s %s" % (first, f) for f in flags_in(words[1:]) if f not in tree.tool_flags.get(source, set())]
    if first in ("python", "python3", "py") or first.endswith(".py"):
        i = 0 if first.endswith(".py") else 1
        while i < len(words) and words[i] in ("-X", "-u", "-B", "utf8"):
            i += 1
        if i >= len(words) or words[i] == "-m" or words[i] == "-" or not words[i].startswith(("tests/", "tools/")):
            return None
        script = words[i]
        if script not in tree.texts:
            return [script]
        return ["%s %s" % (script, f) for f in flags_in(words[i + 1:]) if f not in tree.tool_flags.get(script, set()) | {"--help", "-h"}]
    return None


def reference_findings(tree):
    """`link`, `path`, `line-pin`, `name`, `command` and `section`: what each doc's lines name that the tree lacks."""
    found = {}
    src_dirs = sorted({p.split("/")[1] for p in tree.paths if p.startswith("src/") and p.count("/") >= 2})
    slug_cache = {}

    def add(check, doc, name, n, why):
        found.setdefault((check, doc, name), "%s:%d, %s" % (doc, n, why))

    for doc in doc_files(tree.texts):
        lines = list(doc_lines(doc, tree.texts[doc]))
        paragraph = blocks((n, l) for n, l, _, fence in lines if not fence)
        for n, line, region, fence in lines:
            live = region == "live"
            if fence:
                if live:
                    problems = check_command(tree, line)
                    for p in problems or ():
                        add("command", doc, p, n, "a command line the tree does not take")
                continue
            # Links, in every region.
            for m in MD_LINK.finditer(line):
                target = m.group(2)
                if re.match(r"(?:https?|mailto):", target):
                    continue
                path, _, anchor = target.partition("#")
                dest = os.path.normpath(os.path.join(os.path.dirname(doc), path)).replace(os.sep, "/") if path else doc
                if path and dest not in tree.paths and dest.rstrip("/") not in tree.dirs:
                    add("link", doc, target, n, "no such file")
                elif anchor and dest.endswith(".md"):
                    if dest not in slug_cache:
                        slug_cache[dest] = slugs(tree.texts.get(dest, ""))
                    if anchor not in slug_cache[dest]:
                        add("link", doc, target, n, "no such heading")
            # Line references carry the commit they point into, in every region.
            text, off = paragraph.get(n, (line, 0))
            for m in FILE_LINE.finditer(line):
                if not PIN.search(sentence_at(text, off + m.start())):
                    add("line-pin", doc, m.group(0), n, "a line number without the commit it points into")
            if not live:
                continue
            prose = MD_LINK.sub(" ", SPAN.sub(" ", line))
            for m in BARE_PATH.finditer(prose):
                p = m.group(1).rstrip(".,;:)")
                if not tree.resolve(p, doc) and not branch_like(p):
                    add("path", doc, p, n, "no such path")
            for m in SECTION.finditer(line):
                target, title = m.group(1), m.group(2).strip().split(",")[0].strip()
                target = target if target.startswith("docs/") or target in ROOT_FILES else "docs/" + target
                hs = [re.sub(r"[`*\"]", "", h).lower() for h in headings(tree.texts.get(target, ""))]
                if not any(h.startswith(re.sub(r"[`*\"]", "", title).lower()) for h in hs):
                    add("section", doc, "%s, %s" % (m.group(1), title), n, "no such heading")
            for m in ROADMAP_ITEM.finditer(line):
                if not any(h.startswith(m.group(1) + ".") for h in headings(tree.texts["docs/ROADMAP.md"])):
                    add("section", doc, "ROADMAP #" + m.group(1), n, "no such item")
            for d in re.findall(r"(?<![\w-])-DLLMX_\w+", line):
                if not re.search(r"\b" + d[2:] + r"\b", tree.build_text + "\n" + tree.code_text):
                    add("command", doc, d, n, "no such build option")
            for m in SPAN.finditer(line):
                span_findings(tree, doc, n, m.group(2).strip(), src_dirs, add)
    return found


def span_findings(tree, doc, n, t, src_dirs, add):
    """What one inline code span of live text names: a path, a command, a qualified name or a call."""
    if not t:
        return
    words = t.split()
    # A program with arguments is a command line; a bare `llmx-vulkan` is a target's name.
    if re.match(r"^(\.?[/\\])?(build[/\\](Release[/\\])?)?llmx(-[\w-]+)?(\.exe)?\s", t) or re.match(r"^(python3?|py)\s", t):
        for p in check_command(tree, t) or ():
            add("command", doc, p, n, "a command line the tree does not take")
        return
    if len(words) > 1 and (words[0] in tree.commands or words[0] in tree.ctests) and re.search(r"\s-{1,2}[A-Za-z]", t):
        for p in check_command(tree, t, prefixed=False) or ():
            add("command", doc, p, n, "a command line the tree does not take")
        return
    m = re.fullmatch(r"((?:[\w.-]+/)*[\w.-]+\.(?:hpp|cpp|h|py|comp|glsl))(?::|\s+)([A-Za-z_~][\w:~]*)", t)
    if m:
        if tree.resolve(m.group(1), doc):
            path = next((p for p in (m.group(1), "src/" + m.group(1)) if p in tree.texts), None) or \
                next((p for p in tree.texts if p.endswith("/" + m.group(1))), None)
            symbol = m.group(2).split("::")[-1].lstrip("~")
            if path and not re.search(r"\b" + re.escape(symbol) + r"\b", tree.texts[path]):
                add("path", doc, t, n, "the file has no " + symbol)
        else:
            add("path", doc, t, n, "no such path")
        return
    if is_path(t):
        if t.startswith(RUNTIME) or re.search(r"[<>%$~]", t) or not rooted(t, src_dirs) or re.fullmatch(r"[0-9a-f]{40,64}", t):
            return
        if not tree.resolve(t, doc) and not branch_like(t):
            add("path", doc, t, n, "no such path")
        return
    # A qualified name, with or without arguments, and a call with arguments; `name()` alone is a plain name, which may be another program's.
    qualified = re.fullmatch(r"~?[A-Za-z_]\w*(?:<[^<>]*>)?(?:::~?[A-Za-z_]\w*(?:<[^<>]*>)?)+(\(.*\))?", t)
    call = re.fullmatch(r"([A-Za-z_][\w.]*)\((.+)\)", t)
    if qualified:
        head = re.sub(r"<[^<>]*>", "", t.split("(")[0])
        if not head.startswith("std::"):
            parts = [p.lstrip("~") for p in head.split("::")]
            bad = [p for p in parts if not tree.known(p)]
            if bad:
                add("name", doc, t, n, "the tree has no " + bad[0])
            elif head not in tree.code_text and not all(tree.member(a, b) for a, b in zip(parts, parts[1:])):
                add("name", doc, t, n, "no such member")
    elif call:
        callee = call.group(1)
        bad = [p for p in re.split(r"\.|->", callee) if p and not tree.known(p)]
        if bad and callee not in tree.code_text:
            add("name", doc, t, n, "the tree has no " + bad[0])
    args = re.fullmatch(r"[\w:~.<>-]+\((.*)\)", t)
    if args and (qualified or call):
        for a in sorted(set(ID.findall(re.sub(r'"[^"]*"', "", args.group(1))))):
            if len(a) >= 3 and not a.isupper() and not tree.known(a):
                add("name", doc, t, n, "an argument the tree has no name for: " + a)


# ----------------------------------------------------------------------------------------------------------------------
# docs/USAGE.md against the help pages, and the load tool's table against its options.

USAGE = "docs/USAGE.md"
NUMBER = r"-?\d+(?:\.\d+)?"


def usage_sections(text):
    """USAGE.md's level-2 sections: [(command or None, heading, [(line number, line)])]."""
    out = []
    for n, line in enumerate(text.split("\n"), 1):
        if line.startswith("## "):
            m = re.match(r"## `llmx (\w+)", line)
            out.append((m.group(1) if m and not line.startswith("## `llmx --") else None, line, []))
        elif out:
            out[-1][2].append((n, line))
    return out


def number(v):
    """The number a default is, written alone or before a parenthesis, or None."""
    m = re.fullmatch(r"`?(" + NUMBER + r")`?(?:\s*\(.*\))?", (v or "").strip())
    return float(m.group(1)) if m else None


def row_default(cells, header):
    """A table row's numeric default: its Default column, or the one 'default N' of its text."""
    if "default" in header and header.index("default") < len(cells):
        return number(cells[header.index("default")])
    said = re.findall(r"\bdefault(?:s to| is)?:?\s+`?(" + NUMBER + r")`?(?![\w.])", " ".join(cells[1:]))
    return float(said[0]) if len(said) == 1 else None


def table_rows(lines):
    """(line number, first cell's flags, cells, header) of every table row of `lines`."""
    header = None
    for n, line in lines:
        if not line.startswith("|"):
            header = None
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if header is None:
            header = [c.lower() for c in cells]
            continue
        if set(line.replace("|", "").strip()) <= set("-: "):
            continue
        yield n, re.findall(r"`(-{1,2}[A-Za-z][\w-]*)", cells[0]), cells, header


def usage_findings(tree):
    """`usage`: each command's flags as its help page lists them against its USAGE.md section, synopsis, table and numeric defaults, and the load tool's options against its table."""
    found = {}

    def add(name, n, why):
        found.setdefault(("usage", USAGE, name), "%s:%d, %s" % (USAGE, n, why))

    text = tree.texts[USAGE]
    sections = usage_sections(text)
    commands = sorted(c for c in tree.commands)
    if not commands or not any(tree.help[c] for c in commands):
        raise RuntimeError("the binary's help pages list no commands or no flags, so docs/USAGE.md would be held to nothing")
    flag = re.compile(r"(?<![\w-])(-{1,2}[A-Za-z][\w-]*)")
    for c in commands:
        own = [s for s in sections if s[0] == c]
        if not own:
            add(c, 1, "a command with no section")
            continue
        read = own + ([s for s in sections if s[0] == "generate"] if c == "chat" else [])
        mentioned = {f for _, _, lines in read for _, line in lines for f in flag.findall(line)}
        mentioned |= {f for _, h, _ in read for f in flag.findall(h)}
        heading_flags = {f for _, h, _ in own for f in flag.findall(h)}
        exhaustive = heading_flags and not any(re.search(r"\[(?:flags|options)\.\.\.\]", h) for _, h, _ in own)
        taken = tree.commands[c]
        for g in tree.help[c]:
            name = max(g["spellings"], key=len)
            if not g["spellings"] & mentioned:
                add("%s %s" % (c, name), own[0][2][0][0] - 1, "a flag the command takes that its section never names")
            if exhaustive and not g["spellings"] & heading_flags:
                add("%s %s synopsis" % (c, name), own[0][2][0][0] - 1, "a flag the command takes that its synopsis leaves out")
        for f in sorted(heading_flags - taken):
            add("%s %s" % (c, f), own[0][2][0][0] - 1, "a synopsis flag the command does not take")
        defaults = {max(g["spellings"], key=len): number(g["default"]) for g in tree.help[c]}
        for _, _, lines in own:
            for n, flags, cells, header in table_rows(lines):
                for f in flags:
                    if f not in taken:
                        add("%s %s" % (c, f), n, "a table row for a flag the command does not take")
                    group = next((g for g in tree.help[c] if f in g["spellings"]), None)
                    ours = row_default(cells, header)
                    theirs = defaults.get(max(group["spellings"], key=len)) if group else None
                    if ours is not None and theirs is not None and ours != theirs:
                        add("%s %s default" % (c, f), n, "default %g here, %g in the help" % (ours, theirs))
    for c, h, lines in sections:
        if c and c not in tree.commands:
            add(c, lines[0][0] - 1 if lines else 1, "a section for a command the binary lacks")
    # The load tool's table against its options.
    tool = "tools/server_load.py"
    options = {}
    for node in ast.walk(ast.parse(tree.texts[tool])):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument":
            names = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value.startswith("-")]
            default = next((k.value.value for k in node.keywords if k.arg == "default" and isinstance(k.value, ast.Constant)), None)
            for name in names:
                options[name] = default if isinstance(default, (int, float)) and not isinstance(default, bool) else None
    section = next((s for s in sections if "`%s`" % tool in s[1]), None)
    if not section:
        add(tool, 1, "no section documents the load tool")
        return found
    rows = set()
    for n, flags, cells, header in table_rows(section[2]):
        for f in flags:
            rows.add(f)
            if f not in options:
                add("%s %s" % (tool, f), n, "a table row for an option the tool does not take")
            ours = row_default(cells, header)
            if ours is not None and options.get(f) is not None and ours != options[f]:
                add("%s %s default" % (tool, f), n, "default %g here, %g in the tool" % (ours, options[f]))
    for f in sorted(set(options) - rows - {"-h", "--help"}):
        add("%s %s" % (tool, f), section[2][0][0] - 1, "an option of the tool its table leaves out")
    return found


# ----------------------------------------------------------------------------------------------------------------------
# Test names, and the docs/src page of every source file.

def test_name_findings(tree):
    """`test-name`: a test's name AGENTS.md's Tests or docs/CI.md gives that the tree lacks, and a CTest or suite component AGENTS.md's Tests never describes."""
    found = {}
    agents = tree.texts["AGENTS.md"]
    start = agents.index("\n## Tests")
    tests = agents[start:agents.index("\n## ", start + 1)]
    for doc, text in (("AGENTS.md", tests), ("docs/CI.md", tree.texts["docs/CI.md"])):
        for m in SPAN.finditer(text):
            t = m.group(2).strip()
            if re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)+", t) and not tree.known(t) and t not in tree.code_text and t not in tree.build_text:
                found[("test-name", doc, t)] = "%s, a test name the tree lacks" % doc
    named = set(re.findall(r"`([^`\n]+)`", tests))
    for name, target in sorted(tree.ctests.items()):
        source = tree.targets.get(target, "")
        if name not in named and source not in named:
            found[("test-name", "AGENTS.md", name)] = "a CTest AGENTS.md's Tests never describes"
    for name, module in sorted(tree.components.items()):
        if name not in named and "tests/%s.py" % module not in named:
            found[("test-name", "AGENTS.md", name)] = "a suite component AGENTS.md's Tests never describes"
    return found


def src_page_findings(tree):
    """`src-page`: a src/ file no docs/src page's title names, as the file or a directory holding it, or one two titles name, and a title naming a path that does not exist."""
    found, titles = {}, {}
    for p in sorted(tree.texts):
        if re.fullmatch(r"docs/src/[^/]+\.md", p):
            first = tree.texts[p].split("\n", 1)[0]
            m = re.match(r"#\s+`?([^`\s]+)`?", first)
            if m:
                t = m.group(1)
                titles[p] = t if t.startswith("src/") else "src/" + t
    for page, t in titles.items():
        if t.rstrip("/") not in tree.paths and t.rstrip("/") not in tree.dirs:
            found[("src-page", page, t)] = "%s names %s, which does not exist" % (page, t)
    for f in sorted(p for p in tree.paths if p.startswith("src/")):
        pages = [page for page, t in titles.items() if t == f or t.endswith("/") and f.startswith(t)]
        if not pages:
            found[("src-page", f, "no page")] = "no docs/src page's title names %s or a directory holding it" % f
        elif len(pages) > 1:
            found[("src-page", f, "pages " + ", ".join(pages))] = "several pages name %s" % f
    return found


def all_findings(tree):
    found = {}
    for check in (reference_findings, usage_findings, test_name_findings, src_page_findings):
        found.update(check(tree))
    return found


# ----------------------------------------------------------------------------------------------------------------------
# Planted faults, then the tree.

def planted(tree, path, before, add):
    """`tree` with `add` written into the doc `path` before the first `before`; a doc is no part of the code's index, so the index is shared."""
    texts = dict(tree.texts)
    t = texts.get(path, "")
    i = t.index(before) if before is not None else len(t)
    texts[path] = t[:i] + add + t[i:]
    copy = object.__new__(Tree)
    copy.__dict__.update(tree.__dict__, texts=texts, paths=tree.paths | {path})
    return copy


def self_test(tree, found, listed):
    """The planted faults: a dangling path, a broken link, a wrong flag in USAGE.md, an unpinned line number, a command line with a flag it does not take, a page for a file that does not exist, and a stale list entry."""
    wrong_row = "| `--planted-flag N` | a flag generate does not take | 1 |\n"
    cases = [
        ("a dangling path", reference_findings, planted(tree, "docs/ARCHITECTURE.md", "\n## ", "\nSee `src/planted/nothing.hpp`.\n"),
         ("path", "docs/ARCHITECTURE.md", "src/planted/nothing.hpp")),
        ("a broken link", reference_findings, planted(tree, "docs/BUILD.md", "\n## ", "\nSee [the plan](PLANTED.md#nothing).\n"),
         ("link", "docs/BUILD.md", "PLANTED.md#nothing")),
        ("a wrong flag in USAGE.md", usage_findings, planted(tree, USAGE, "| `--seed N`", wrong_row),
         ("usage", USAGE, "generate --planted-flag")),
        ("a wrong default in USAGE.md", usage_findings, planted(tree, USAGE, "| `--iters N`", "| `--size N` | matrix width | 2048 |\n"),
         ("usage", USAGE, "bench --size default")),
        ("an unpinned line number", reference_findings, planted(tree, "docs/SERVER.md", "\n## ", "\nThe route is at `server/api.hpp:12`.\n"),
         ("line-pin", "docs/SERVER.md", "server/api.hpp:12")),
        ("a command line with a flag it does not take", reference_findings,
         planted(tree, "docs/BUILD.md", "\n## ", "\n```\nllmx generate model.gguf hi --planted-flag 3\n```\n"),
         ("command", "docs/BUILD.md", "generate --planted-flag")),
        ("a page for a file that does not exist", src_page_findings,
         planted(tree, "docs/src/core-planted.md", None, "# `src/core/planted.hpp` - nothing\n"),
         ("src-page", "docs/src/core-planted.md", "src/core/planted.hpp")),
    ]
    ok = True
    for what, check, planted_tree, key in cases:
        got = check(planted_tree)
        if key not in got or key in found:
            print("  self-test: %s was not reported as %s" % (what, " | ".join(key)))
            ok = False
    stale = dict(listed)
    stale[("path", "docs/ARCHITECTURE.md", "src/planted/gone.hpp")] = "a planted entry"
    if common.settle_findings(found, CHECKS, stale, say=lambda *a: None):
        print("  self-test: a listed finding that does not occur was accepted")
        ok = False
    return ok, len(cases) + 1


def run():
    t0 = time.time()
    texts, paths = common.read_tree()
    tree = Tree(texts, paths)
    with open(common.KNOWN_FINDINGS, encoding="utf-8") as f:
        listed = common.known_findings(f.read())
    found = all_findings(tree)
    planted_ok, count = self_test(tree, found, listed)
    ok = common.settle_findings(found, CHECKS, listed)
    counts = collections.Counter(k[0] for k in found)
    print("docs: %d findings (%s) against the list; %d planted faults; %.1f s  [%s]" % (
        len(found), ", ".join("%s %d" % (c, counts[c]) for c in CHECKS if counts[c]) or "none", count, time.time() - t0,
        "ok" if ok and planted_ok else "FAIL"))
    return ok and planted_ok


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stale-doc checks against the tree and the built binary's help pages.")
    parser.add_argument("--exe", default=common.EXE, help="path to the built llmx executable")
    common.EXE = os.path.abspath(parser.parse_args().exe)
    common.exe_path()
    sys.exit(0 if run() else 1)
