"""Dead code: what no product path reaches, found from the source in every job and from the linked binaries in the Vulkan job.

The product is `llmx` and the tools in tools/; code only tests reach is a finding too, since it is kept for nothing the product does.
Each finding is keyed by its check, file and name, and held to tests/data/known_findings.txt, which a new finding, a count that differs and a listed one that no longer occurs all fail (AGENTS.md, Dead code and stale docs).
Standard library only; the linked check also needs Linux, GCC, GNU ld, binutils, CMake, the Vulkan headers and glslc.
"""
import argparse
import ast
import collections
import functools
import http.client
import http.server
import io
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common
import docs_check

# Several checks, and each planted copy of the tree, strip and read the same files, so both are remembered by the text.
code_of = functools.lru_cache(maxsize=None)(common.cpp_code)

# The checks the source reads, run by the suite in every job.
SOURCE_CHECKS = ("unused", "test-only", "override", "shader", "flag", "python", "file", "macro")
# The checks of the linked binaries, run on their own build (`--linked`).
LINKED_CHECKS = ("linked-unreached", "linked-test-only")

CPP_SUFFIXES = (".hpp", ".cpp", ".h")


def cpp_files(texts, tops=("src", "tests", "tools")):
    return sorted(p for p in texts if p.endswith(CPP_SUFFIXES) and p.split("/")[0] in tops)


# ----------------------------------------------------------------------------------------------------------------------
# C++ tokens and the declarations at namespace, class and enum scope.

TOKEN = re.compile(r"""
 (?P<nl>\n)
|(?P<ws>[ \t\r\f\v]+|\\\r?\n)
|(?P<lit>""|'\ ')
|(?P<num>\.?[0-9](?:[0-9a-zA-Z_.]|[eEpP][+-])*)
|(?P<id>[A-Za-z_][A-Za-z0-9_]*)
|(?P<op>::|->|\+\+|--|<=|>=|==|!=|&&|\|\||<<=|>>=|[-+*/%&|^]=|<<|\.\.\.|[{}()\[\];,<>=+\-*/%&|^!~?:.\#@$])
""", re.X)

Tok = collections.namedtuple("Tok", "kind text line pp")

KEYWORDS = set("""alignas alignof and asm auto bool break case catch char char8_t char16_t char32_t class const consteval constexpr
constinit const_cast continue decltype default delete do double dynamic_cast else enum explicit export extern false float for friend
goto if inline int long mutable namespace new noexcept not nullptr operator or private protected public register reinterpret_cast
requires return short signed sizeof static static_assert static_cast struct switch template this thread_local throw true try typedef
typeid typename union unsigned using virtual void volatile wchar_t while override final __attribute__ __declspec __restrict
__forceinline""".split())


@functools.lru_cache(maxsize=None)
def lex(code):
    """Tokens of C++ `code` whose comments and literals `common.cpp_code` blanked; a preprocessor line's tokens carry pp=True."""
    toks, line, pos, start, in_pp = [], 1, 0, True, False
    n = len(code)
    while pos < n:
        m = TOKEN.match(code, pos)
        if not m:
            pos += 1
            continue
        kind, s = m.lastgroup, m.group(0)
        pos = m.end()
        if kind == "nl":
            line += 1
            in_pp, start = False, True
            continue
        if kind == "ws":
            line += s.count("\n")
            continue
        if s == "#" and start:
            in_pp = True
        start = False
        toks.append(Tok("id" if kind == "id" else "lit" if kind == "lit" else "op", s, line, in_pp))
    return toks


def skip_template(texts, k):
    """The index after a `template <...>` prefix starting at `k`, or `k` when there is none."""
    while k < len(texts) and texts[k] == "template" and k + 1 < len(texts) and texts[k + 1] == "<":
        depth = 0
        for m in range(k + 1, len(texts)):
            if texts[m] == "<":
                depth += 1
            elif texts[m] == ">":
                depth -= 1
                if depth == 0:
                    k = m + 1
                    break
        else:
            return len(texts)
    return k


def match_forward(texts, j, open_, close):
    depth = 0
    for k in range(j, len(texts)):
        if texts[k] == open_:
            depth += 1
        elif texts[k] == close:
            depth -= 1
            if depth == 0:
                return k
    return len(texts) - 1


class Declarations:
    """Every declaration at namespace, class and enum scope of a set of C++ files, and every identifier token that is not one."""

    def __init__(self, texts, files):
        self.decls = []          # dicts: name kind file line owner flags
        self.positions = set()   # (file, token index) of declared names
        self.macros = collections.defaultdict(list)   # name -> [(file, line, token index)]
        self.tested = collections.defaultdict(list)   # macro name -> [(file, line)] where a conditional tests it
        self.tokens = {}
        for f in files:
            self.tokens[f] = lex(code_of(texts[f]))
        for f in files:
            self.preprocessor(f)
        for f in files:
            self.scan(f)
        self.uses = collections.defaultdict(collections.Counter)   # name -> top directory -> count
        for f, toks in self.tokens.items():
            top = f.split("/")[0]
            include = False
            for i, t in enumerate(toks):
                if t.pp and t.text == "#":
                    include = i + 1 < len(toks) and toks[i + 1].text == "include"
                if t.kind != "id" or (f, i) in self.positions or (t.pp and include):
                    continue
                self.uses[t.text][top] += 1

    def preprocessor(self, f):
        toks = self.tokens[f]
        for i, t in enumerate(toks):
            if not (t.pp and t.text == "#") or i + 1 >= len(toks):
                continue
            d = toks[i + 1].text
            if d == "define" and i + 2 < len(toks) and toks[i + 2].kind == "id":
                self.macros[toks[i + 2].text].append((f, toks[i + 2].line, i + 2))
                self.positions.add((f, i + 2))
            elif d in ("if", "ifdef", "ifndef", "elif", "undef"):
                k = i + 2
                while k < len(toks) and toks[k].pp and toks[k].text != "#":
                    if toks[k].kind == "id" and toks[k].text != "defined":
                        self.tested[toks[k].text].append((f, toks[k].line))
                    k += 1

    def add(self, f, pair, kind, owner, flags=""):
        idx, tok = pair
        if (f, idx) in self.positions:
            return
        self.positions.add((f, idx))
        self.decls.append({"name": tok.text, "kind": kind, "file": f, "line": tok.line, "owner": owner or "", "flags": flags})

    def head(self, f, head, owner, ends_with):
        """One declaration statement at namespace or class scope, as (index, token) pairs."""
        texts = [t.text for _, t in head]
        if not texts or texts[0] in ("static_assert", "friend", "namespace"):
            return
        if texts[0] == "using":
            if len(texts) > 2 and texts[2] == "=" and head[1][1].kind == "id":
                self.add(f, head[1], "type", owner)
            return
        if texts[0] == "typedef":
            ids = [x for x in head if x[1].kind == "id"]
            if ids:
                self.add(f, ids[-1], "type", owner)
            return
        k = skip_template(texts, 0)
        body, bt = head[k:], texts[k:]
        if not body:
            return
        if bt[0] in ("class", "struct", "union", "enum"):
            m = 1
            if m < len(bt) and bt[m] in ("class", "struct"):
                m += 1
            while m < len(bt) and bt[m] in ("[", "alignas", "__declspec"):
                m = match_forward(bt, m, "[", "]") + 1 if bt[m] == "[" else match_forward(bt, m + 1, "(", ")") + 1
            if m < len(bt) and body[m][1].kind == "id" and bt[m] not in KEYWORDS and ("(" not in bt[m:] or ends_with == "{"):
                self.add(f, body[m], "enum" if bt[0] == "enum" else "type", owner)
                return
        if "operator" in bt:
            return
        # A function: the first top-level ( after a name that is no keyword or macro, before any top-level =.
        paren = brack = angle = 0
        func = eq_at = None
        for m, s in enumerate(bt):
            if s == "(":
                if paren == brack == angle == 0 and func is None and m > 0:
                    prev = body[m - 1][1]
                    if prev.kind == "id" and prev.text not in KEYWORDS and prev.text not in self.macros \
                            and (m < 2 or bt[m - 2] not in (".", "->", "~")):
                        func = m - 1
                paren += 1
            elif s == ")":
                paren -= 1
                if func is not None and paren == 0:
                    break
            elif s == "[":
                brack += 1
            elif s == "]":
                brack -= 1
            elif s == "<" and paren == brack == 0 and m > 0 and (body[m - 1][1].kind == "id" or bt[m - 1] == "template"):
                angle += 1
            elif s == ">" and angle > 0 and paren == 0:
                angle -= 1
            elif s == "=" and paren == brack == angle == 0:
                eq_at = m
                break
        if func is not None:
            name = bt[func]
            if name == owner or (func > 0 and bt[func - 1] == "~"):
                return
            qualified = func > 1 and bt[func - 1] == "::"
            cls = bt[func - 2] if qualified else owner
            if qualified and name == cls:
                return
            flags = " ".join(x for x in ("override", "final", "virtual", "constexpr") if x in bt)
            self.add(f, body[func], "function", cls, flags)
            return
        # Variables, fields and constants: each declarator's name is the last identifier before = { [ : or the end.
        end = eq_at if eq_at is not None else len(bt)
        parts, cur = [], []
        paren = brack = angle = brace = 0
        for m in range(end):
            s = bt[m]
            paren += (s == "(") - (s == ")")
            brack += (s == "[") - (s == "]")
            brace += (s == "{") - (s == "}")
            if s == "<" and m > 0 and body[m - 1][1].kind == "id":
                angle += 1
            elif s == ">" and angle > 0:
                angle -= 1
            if s == "," and paren == brack == angle == brace == 0:
                parts.append(cur)
                cur = []
                continue
            cur.append(m)
        parts.append(cur)
        flags = set(bt) & {"const", "constexpr", "static"}
        kind = "constant" if flags & {"const", "constexpr"} else "field" if owner and "static" not in flags and self.scope_kind == "class" else "variable"
        for part in parts:
            name = None
            for m in part:
                s = bt[m]
                if s in ("{", "[", "=", ":"):
                    break
                if body[m][1].kind == "id" and s not in KEYWORDS:
                    name = m
            if name is not None and name > 0 and bt[name] not in self.macros:
                self.add(f, body[name], kind, owner)

    def scan(self, f):
        code = [(i, t) for i, t in enumerate(self.tokens[f]) if not t.pp]
        stack = [("namespace", "")]
        head = []
        enum_expect, enum_paren = [], []
        for idx, t in code:
            s = t.text
            kind, name = stack[-1]
            if kind in ("function", "block", "init"):
                if s == "{":
                    stack.append(("block", ""))
                elif s == "}":
                    stack.pop()
                    if stack[-1][0] in ("namespace", "class"):
                        if kind == "init":
                            head.append((idx, t))
                        elif kind == "function":
                            # A lambda's body inside an initializer continues its statement; a function's body ends it.
                            if head and head[-1][1].text == "\x00lambda":
                                head.pop()
                            else:
                                head = []
                continue
            if kind == "enum":
                if s == "}":
                    stack.pop()
                    enum_expect.pop()
                    enum_paren.pop()
                    head = []
                elif t.kind == "id" and enum_expect[-1]:
                    self.add(f, (idx, t), "enumerator", name)
                    enum_expect[-1] = False
                elif s in "()":
                    enum_paren[-1] += 1 if s == "(" else -1
                elif s == "," and enum_paren[-1] == 0:
                    enum_expect[-1] = True
                continue
            self.scope_kind = kind
            if s == ";":
                self.head(f, head, name, ";")
                head = []
            elif s == ":" and len(head) == 1 and head[0][1].text in ("public", "private", "protected"):
                head = []
            elif s == "}":
                if head:
                    self.head(f, head, name, ";")
                head = []
                stack.pop()
                if not stack:
                    stack = [("namespace", "")]
            elif s == "{" and sum((x[1].text == "(") - (x[1].text == ")") for x in head) > 0:
                # A braced default argument, `RowRuns runs = {}`, is part of the declaration its parameter list is in.
                stack.append(("init", ""))
                head.append((idx, t))
            elif s == "{":
                texts = [x[1].text for x in head]
                bare = texts[skip_template(texts, 0):]
                while bare and bare[0] == "typedef":
                    bare = bare[1:]
                if "namespace" in texts or (texts and texts[0] == "extern"):
                    stack.append(("namespace", name))
                    head = []
                elif bare[:1] and bare[0] in ("class", "struct", "union") and "(" not in bare and "=" not in texts:
                    cls = next((x for x in bare[1:] if x not in KEYWORDS and re.match(r"[A-Za-z_]", x)), "")
                    self.head(f, head, name, "{")
                    stack.append(("class", cls))
                    head = []
                elif bare[:1] == ["enum"]:
                    # The name after `enum` and `class` or `struct`; an enum without one is keyed by where it is, never by its underlying type.
                    rest = bare[1:] if bare[1:2] not in (["class"], ["struct"]) else bare[2:]
                    enum = rest[0] if rest and re.match(r"[A-Za-z_]\w*$", rest[0]) and rest[0] not in KEYWORDS else "enum@%s:%d" % (f, t.line)
                    self.head(f, head, name, "{")
                    stack.append(("enum", enum))
                    enum_expect.append(True)
                    enum_paren.append(0)
                    head = []
                elif "(" in texts:
                    if lambda_head(texts):
                        head.append((idx, Tok("op", "\x00lambda", t.line, False)))
                    else:
                        self.head(f, head, name, "{")
                    stack.append(("function", ""))
                else:
                    stack.append(("init", ""))
                    head.append((idx, t))
            else:
                head.append((idx, t))


def lambda_head(texts):
    """Whether a `{` after `texts` opens a lambda's body: a capture list after the statement's first top-level `=`."""
    paren = 0
    for m, s in enumerate(texts):
        if s == "=" and paren == 0:
            return "[" in texts[m:] and "]" in texts[m:]
        paren += (s == "(") - (s == ")")
    return False


def pinned_structs(code_by_file):
    """Structs whose layout a static assertion pins with offsetof or sizeof: their fields mirror a layout something else reads."""
    pinned = set()
    for code in code_by_file.values():
        for m in re.finditer(r"static_assert\s*\((.*?)\)\s*;", code, re.S):
            pinned |= set(re.findall(r"\boffsetof\s*\(\s*(\w+)", m.group(1)))
            pinned |= set(re.findall(r"\bsizeof\s*\(\s*(\w+)\s*\)", m.group(1)))
    return pinned


def integer_enums(code_by_file, declared):
    """The enums of `declared`, the named enums, that a value is cast to from an integer: their enumerators are reached by arithmetic, not by name.
    A cast to a name the file takes as a template parameter is no cast to an enum of that name."""
    enums = set()
    for code in code_by_file.values():
        params = set(re.findall(r"\b(?:class|typename)\s+(\w+)\s*(?:=[^,>]*)?[,>]", code))
        casts = re.findall(r"\(\s*(?:\w+\s*::\s*)*(\w+)\s*\)\s*[\w(]", code) + re.findall(r"static_cast\s*<\s*(?:\w+\s*::\s*)*(\w+)\s*>", code)
        enums |= (set(casts) & declared) - params
    return enums


def system_configured(toks, index):
    """A macro defined at token `index` that a later `#include <...>` of the same file reads, as VK_NO_PROTOTYPES configures the Vulkan header."""
    for i in range(index, len(toks) - 2):
        if toks[i].pp and toks[i].text == "#" and toks[i + 1].text == "include" and toks[i + 2].text == "<":
            return True
    return False


def name_findings(texts):
    """`unused`: a declaration whose name nothing else names; `test-only`: one in src/ that only tests/ name; `macro`: a macro defined and never used, or an LLMX_ macro tested and set nowhere."""
    files = cpp_files(texts)
    ds = Declarations(texts, files)
    if len(ds.decls) < 1000:
        raise Layout("the name check found only %d declarations" % len(ds.decls))
    code = {f: code_of(texts[f]) for f in files}
    pinned = pinned_structs(code)
    enums = integer_enums(code, {d["name"] for d in ds.decls if d["kind"] == "enum"})
    found = {}
    seen = set()
    for d in sorted(ds.decls, key=lambda d: (d["file"], d["line"])):
        if d["name"] == "main" or "override" in d["flags"] or "final" in d["flags"]:
            continue
        if d["kind"] == "field" and d["owner"] in pinned or d["kind"] == "enumerator" and d["owner"] in enums:
            continue
        uses = ds.uses.get(d["name"], collections.Counter())
        top = d["file"].split("/")[0]
        if sum(uses.values()) == 0:
            check = "unused"
        elif top == "src" and uses["src"] == 0 and uses["tools"] == 0:
            check = "test-only"
        else:
            continue
        name = (d["owner"] + "::" if d["owner"] and not d["owner"].startswith("enum@") else "") + d["name"]
        if (check, name) in seen:   # a declaration and its definition
            continue
        seen.add((check, name))
        found[(check, d["file"], name)] = ["%s:%d, a %s" % (d["file"], d["line"], d["kind"])]
    # Macros: one defined and never named again, unless a system header after it reads it; an LLMX_ one tested but set nowhere.
    for name, where in sorted(ds.macros.items()):
        if sum(ds.uses.get(name, {}).values()) or name in ds.tested:
            continue
        f, line, index = where[0]
        if not system_configured(ds.tokens[f], index):
            found[("macro", f, name)] = ["%s:%d, defined and never used" % (f, line)]
    build = "\n".join(t for p, t in texts.items() if p in ("CMakeLists.txt", "build.bat") or p.startswith("cmake/"))
    docs = "\n".join(t for p, t in texts.items() if p.endswith(".md"))
    for name, where in sorted(ds.tested.items()):
        if not name.startswith("LLMX_") or name in ds.macros:
            continue
        if re.search(r"\b" + name + r"\b", build) or re.search(r"[-/]D" + name + r"\b", docs):
            continue
        f, line = where[0]
        found[("macro", f, name)] = ["%s:%d, tested but set nowhere" % (f, line)]
    return found


# ----------------------------------------------------------------------------------------------------------------------
# Virtual overrides, which a link keeps through the vtable whether or not anything calls them.

def body_end(code, start):
    """The index after the brace block that opens at or after `start`."""
    i = code.index("{", start)
    depth = 0
    for j in range(i, len(code)):
        if code[j] == "{":
            depth += 1
        elif code[j] == "}":
            depth -= 1
            if depth == 0:
                return j + 1
    return len(code)


def override_findings(texts):
    """`override`: an override in src/ that neither src/ nor tools/ calls through an object (`->v(`, `.v(`), and no member of its class or a base calls unqualified."""
    code = {f: code_of(texts[f]) for f in cpp_files(texts)}
    classes = {}   # name -> {bases, spans, virtuals, overrides, file}
    for f, c in code.items():
        for m in re.finditer(r"\b(?:class|struct)\s+(\w+)(?:\s+final)?\s*(?::\s*([^{;]+))?\{", c):
            bases = re.findall(r"(?:public|protected|private)?\s*(?:\w+::)*(\w+)\s*(?:<[^>]*>)?\s*(?:,|$)", (m.group(2) or "").strip())
            end = body_end(c, m.end() - 1)
            info = classes.setdefault(m.group(1), {"bases": set(), "spans": [], "virtuals": set(), "overrides": {}, "file": f})
            info["bases"] |= {b for b in bases if b}
            info["spans"].append((f, m.end(), end))
            body = c[m.end():end]
            info["virtuals"] |= set(re.findall(r"\bvirtual\s+[^;{(]*?\b(\w+)\s*\(", body))
            for o in re.finditer(r"\b(\w+)\s*\((?:[^;{}]|=\s*\{[^;{}]*\})*\)\s*(?:const\s*)?(?:noexcept\s*)?(?:override|final)\b", body):
                info["overrides"].setdefault(o.group(1), (f, c.count("\n", 0, m.end() + o.start()) + 1))
    # Out-of-class member definitions are member bodies too.
    for f, c in code.items():
        for m in re.finditer(r"\b(\w+)::~?\w+\s*\([^;{}]*\)\s*(?:const\s*)?(?:noexcept\s*)?(?::[^;{]*)?\{", c):
            if m.group(1) in classes:
                classes[m.group(1)]["spans"].append((f, m.end() - 1, body_end(c, m.end() - 1)))

    def ancestors(name, seen):
        for b in classes.get(name, {}).get("bases", ()):
            if b not in seen:
                seen.add(b)
                ancestors(b, seen)
        return seen

    def owner_at(f, pos):
        best = None
        for name, info in classes.items():
            for ff, a, b in info["spans"]:
                if ff == f and a <= pos < b and (best is None or a > best[1]):
                    best = (name, a)
        return best[0] if best else None

    virtual_names = {v for info in classes.values() for v in info["virtuals"]} - set(classes)
    src = {f: c for f, c in code.items() if f.startswith("src/")}
    product = {f: c for f, c in code.items() if f.startswith(("src/", "tools/"))}
    found = {}
    for cname, info in sorted(classes.items()):
        if not info["file"].startswith("src/"):
            continue
        family = ancestors(cname, set()) | {cname}
        for v, (f, line) in sorted(info["overrides"].items()):
            if v not in virtual_names:
                continue
            called = any(re.search(r"(?:->|\.)\s*" + v + r"\s*\(", c) for c in product.values())
            # A declaration or definition, not a call, its parameters on one line or several.
            declared = re.compile(v + r"\s*\((?:[^()]|\([^()]*\))*\)\s*(?:const\s*)?(?:noexcept\s*)?(?:override|final|=\s*0|\{)")
            if not called:
                for ff, c in src.items():
                    for m in re.finditer(r"(?<![\w.>:~])" + v + r"\s*\(", c):
                        if declared.match(c, m.start()):
                            continue
                        if owner_at(ff, m.start()) in family:
                            called = True
                            break
                    if called:
                        break
            if not called:
                found[("override", f, cname + "::" + v)] = ["%s:%d, an override product code never calls" % (f, line)]
    return found


# ----------------------------------------------------------------------------------------------------------------------
# Shaders: every source built, every module embedded, in the kernel table and named at a dispatch; every define tested and set.

VK = "src/backends/vulkan/vulkan_backend.cpp"
SHADERS = "src/backends/vulkan/shaders/"


class Layout(Exception):
    """A file the checks read no longer has the layout they rely on; the check is updated with the change, since a silent pass would check nothing."""


def find(text, what, path, start=0):
    i = text.find(what, start)
    if i < 0:
        raise Layout("%s no longer holds %r" % (path, what))
    return i


def cmake_list(cmake, name):
    m = re.search(r"set\(" + name + r"\s+(.*?)\)", cmake, re.S)
    if not m:
        raise Layout("CMakeLists.txt no longer sets " + name)
    return re.sub(r"#[^\n]*", "", m.group(1)).split()


def shader_findings(texts):
    """`shader`: a source no CMake entry compiles, an include nothing includes, a module never embedded or tabled, a kernel id no dispatch names, a variant's define its source never tests, a tested define nothing sets, and an #ifndef default nothing overrides."""
    cmake, vk = texts["CMakeLists.txt"], texts[VK]
    found = {}

    def add(path, name, why):
        found[("shader", path, name)] = [why]

    entries = {e: (e, []) for e in cmake_list(cmake, "LLMX_VK_SHADERS")}
    for e in cmake_list(cmake, "LLMX_VK_VARIANTS"):
        name, src, defs = e.split("=")
        entries[name] = (src, defs.split("+"))
    shader_texts = {p[len(SHADERS):]: t for p, t in texts.items() if p.startswith(SHADERS)}
    comps = sorted(p[:-5] for p in shader_texts if p.endswith(".comp"))
    sources = {src for src, _ in entries.values()}
    for c in comps:
        if c not in sources:
            add(SHADERS + c + ".comp", "not built", "no entry of LLMX_VK_SHADERS or LLMX_VK_VARIANTS compiles it")
    for name, (src, _) in entries.items():
        if src not in comps:
            add("CMakeLists.txt", name, "names %s.comp, which does not exist" % src)

    def includes(path):
        return [i for i in re.findall(r'#\s*include\s+"([^"]+)"', shader_texts.get(path, "")) if i in shader_texts]

    reached, work = set(), [c + ".comp" for c in sources if c in comps]
    while work:
        p = work.pop()
        if p not in reached:
            reached.add(p)
            work += includes(p)
    for g in sorted(p for p in shader_texts if p.endswith(".glsl") and p not in reached):
        add(SHADERS + g, "not included", "no built shader includes it")

    code = code_of(vk)
    arrays = dict((m.group(2), m.group(1)) for m in re.finditer(r'const uint32_t (kSpv\w+)\[\] = \{\s*#include "vulkan/(\w+)\.inc"', vk))
    for name in entries:
        if name not in arrays:
            add(VK, name + ".inc", "built but no kSpv array includes it")
    catalog_start = find(code, "#define LLMX_VULKAN_KERNELS(X)", VK)
    catalog_end = find(code, "#define LLMX_KERNEL_ID", VK, catalog_start)
    catalog = code[catalog_start:catalog_end]
    for inc, arr in sorted(arrays.items()):
        if arr not in catalog:
            add(VK, arr, "embedded from %s.inc but not in kKernels" % inc)
    ids = re.findall(r"X\((K_[A-Z0-9_]+),", catalog) + ["K_COUNT"]
    raw_catalog = vk[vk.index("#define LLMX_VULKAN_KERNELS(X)"):vk.index("#define LLMX_KERNEL_ID")]
    names = dict(re.findall(r'X\((K_[A-Z0-9_]+), "([^"]*)",', raw_catalog))
    for k in ids[:-1]:
        if not names.get(k):
            add(VK, k + ".name", "kernel catalog entry has no diagnostic name")
    pos = {k: n for n, k in enumerate(ids)}
    outside = code[:catalog_start] + code[catalog_end:]
    named = set(re.findall(r"\bK_[A-Z0-9_]+\b", outside))
    derived = set()
    # kv_variant(f32, k16, s) returns f32, k16, k16 + 1 or k16 + 2 by the storage's cache types.
    for _, k16 in re.findall(r"kv_variant\(\s*(K_\w+)\s*,\s*(K_\w+)\s*,", outside):
        derived |= {ids[pos[k16] + i] for i in range(3) if pos[k16] + i < len(ids)}
    for k in ids:
        if k != "K_COUNT" and k not in named and k not in derived:
            add(VK, k, "no dispatch names it and no kv_variant call derives it")

    def tested(path, seen):
        if path in seen or path not in shader_texts:
            return set(), set()
        seen.add(path)
        t, d = set(), set()
        for line in code_of(shader_texts[path]).split("\n"):
            s = line.strip()
            if re.match(r"#\s*(if|ifdef|ifndef|elif)\b", s):
                t |= set(re.findall(r"\b[A-Z_][A-Z0-9_]+\b", s)) - {"defined"}
            m = re.match(r"#\s*define\s+(\w+)", s)
            if m:
                d.add(m.group(1))
        for i in includes(path):
            ti, di = tested(i, seen)
            t |= ti
            d |= di
        return t, d

    set_by = collections.defaultdict(list)
    for name, (src, defs) in entries.items():
        t, _ = tested(src + ".comp", set())
        for d in defs:
            set_by[(src, d)].append(name)
            if d not in t:
                add("CMakeLists.txt", name + " " + d, "%s.comp and its includes never test %s, so the variant is the base module" % (src, d))
    all_defs = {d for _, defs in entries.values() for d in defs}
    for src in sorted(sources):
        t, d = tested(src + ".comp", set())
        for x in sorted(t):
            if x.startswith("GL_") or x in d or (src, x) in set_by:
                continue
            add(SHADERS + src + ".comp", x, "tested but nothing sets it")
    for p in sorted(reached):
        lines = code_of(shader_texts[p]).split("\n")
        for n, line in enumerate(lines):
            m = re.match(r"\s*#\s*ifndef\s+(\w+)", line)
            if not m or m.group(1) in all_defs or m.group(1).endswith("_GLSL"):
                continue
            depth = 1
            for later in lines[n + 1:]:
                s = later.strip()
                if re.match(r"#\s*if", s):
                    depth += 1
                elif re.match(r"#\s*endif", s):
                    depth -= 1
                    if not depth:
                        break
                elif re.match(r"#\s*define\s+" + m.group(1) + r"(?!\w)", s):
                    add(SHADERS + p, "#ifndef " + m.group(1), "a default no CMake entry overrides, so the switch changes nothing")
                    break
    return found


# ----------------------------------------------------------------------------------------------------------------------
# CLI flags: each flag src/cli/main.cpp parses stores into something src/ reads.

MAIN = "src/cli/main.cpp"


def usage_span(code):
    """Where print_usage's definition starts and ends in `code`, main.cpp's text."""
    m = re.search(r"\bbool\s+print_usage\s*\(", code)
    if not m:
        raise Layout(MAIN + " no longer defines print_usage")
    return m.start(), body_end(code, m.end())


# What follows a name that is stored into rather than read.
STORE = re.compile(r"\s*(?:[-+*/|&]?=(?!=)|\+\+|--)")


def statement_at(code, pos):
    """The line of `code` that holds `pos`."""
    return code[code.rfind("\n", 0, pos) + 1:code.find("\n", pos)]


def in_if_condition(line, at):
    """Whether position `at` of `line` sits inside the parentheses of an `if` the same line opens."""
    opens = list(re.finditer(r"\bif\s*\(", line[:at]))
    if not opens:
        return False
    span = line[opens[-1].end() - 1:at]
    return span.count("(") > span.count(")")


def flag_findings(texts):
    """`flag`: a flag whose parse stores nothing, or stores into a field or local that nothing uses.
    The help printing the default is no use, nor is a refusal's condition, nor a copy into a field of the same name, which passes the value on.
    A field of a struct main.cpp declares is looked for in main.cpp, and in all of src/ once a copy passes it on; any other field in all of src/."""
    lines = code_of(texts[MAIN], keep_strings=True).split("\n")
    main_code = code_of(texts[MAIN])
    a, b = usage_span(main_code)
    main_code = main_code[:a] + re.sub(r"[^\n]", " ", main_code[a:b]) + main_code[b:]
    main_lines = main_code.split("\n")
    local_structs = set(re.findall(r"\bstruct\s+(\w+)\s*(?::[^{;]*)?\{", main_code))
    src = [code_of(t) for p, t in sorted(texts.items()) if p.startswith("src/") and p.endswith(CPP_SUFFIXES) and p != MAIN]
    found, sites = {}, 0
    for n, line in enumerate(lines, 1):
        for m in re.finditer(r'\b(\w+)\s*==\s*"(-{1,2}[A-Za-z][\w-]*)"', line):
            flag = m.group(2)
            # A parse is a comparison inside an if's condition; one elsewhere, as a spelling mapped to its setting, parses nothing.
            if m.group(1) == "cmd" or not in_if_condition(line, m.start()):
                continue
            sites += 1
            # The statement the comparison guards: from the condition's closing parenthesis to the next ';' or '}'.
            rest = line[m.end():] + "\n" + "\n".join(lines[n:n + 3])
            depth = line[:m.start()].count("(") - line[:m.start()].count(")")
            i = 0
            while i < len(rest) and depth > 0:
                depth += (rest[i] == "(") - (rest[i] == ")")
                i += 1
            stmt = rest[i:].strip()
            if stmt.startswith("throw"):
                continue   # a refusal, not a stored value
            # A statement stores into its target; a block, into the first field it assigns.
            target = re.match(r"(?:\(\w+\)\s*)?([\w.>-]+?)\s*=(?!=)", stmt.lstrip("{").strip()) or \
                re.search(r"(?<![\w.>-])((?:\w+(?:\.|->))+\w+)\s*=(?!=)", stmt.split("}")[0] if stmt.startswith("{") else "")
            if not target:
                found[("flag", MAIN, flag)] = ["%s:%d parses it and stores nothing" % (MAIN, n)]
                continue
            parts = re.split(r"\.|->", target.group(1))
            name, field = parts[-1], len(parts) > 1
            copy = re.compile(r"(?:\.|->)\s*" + re.escape(name) + r"\s*=(?!=)")
            # The parse's own line is no read of it.
            own = "\n".join(main_lines[:n - 1] + [""] + main_lines[n:])
            # The type of the object a field belongs to, from its last declaration before the parse.
            before = "\n".join(main_lines[:n - 1])
            kinds = [d.group(1) for d in re.finditer(r"\b(\w+)\s*[&*]?\s+" + re.escape(parts[0]) + r"\s*[;,)=({]", before)
                     if d.group(1) not in ("return", "else", "case", "throw", "delete", "new", "do", "goto", "sizeof")]
            local = not field or bool(kinds) and kinds[-1] in local_structs
            reads, copied = 0, False
            for c in [own] + ([] if local else src):
                for u in re.finditer(r"\b" + re.escape(name) + r"\b", c):
                    if STORE.match(c, u.end()):
                        continue
                    if not field and re.search(r"\b(?:bool|int|size_t|double|float|auto|uint\d+_t|std::string)\s*$", c[max(0, u.start() - 40):u.start()]):
                        continue   # the local's declaration
                    statement = statement_at(c, u.end())
                    if re.match(r"\s*(?:else\s+)?if\s*\(.*\)\s*throw\b", statement):
                        continue   # a refusal's condition
                    if copy.search(statement):
                        copied = True
                        continue
                    reads += 1
            if not reads and copied and local and field:
                reads = sum(1 for c in src for u in re.finditer(r"\b" + re.escape(name) + r"\b", c)
                            if not STORE.match(c, u.end()) and not copy.search(statement_at(c, u.end())))
            if not reads:
                found[("flag", MAIN, flag)] = ["%s:%d stores it in %s, which nothing uses" % (MAIN, n, target.group(1))]
    if not sites:
        raise Layout(MAIN + " no longer compares an argument with a flag's literal")
    return found


# ----------------------------------------------------------------------------------------------------------------------
# Python: definitions nothing reaches from a module's top level, argparse options never read, test modules nothing runs.

# Methods a standard library caller reaches by name: handler and test hooks, and every method of the objects tests stand in for.
PY_HOOKS = {"setUp", "tearDown", "setUpClass", "tearDownClass", "run", "handle", "log_message", "log_request", "log_error",
            "do_GET", "do_POST", "do_HEAD", "do_PUT", "do_DELETE", "do_OPTIONS", "default", "handle_starttag", "handle_endtag",
            "handle_data", "finish", "setup", "server_bind", "handle_error", "send_error", "address_string", "emit", "format"}
PY_PROTOCOLS = set()
for _cls in (socket.socket, io.RawIOBase, io.BufferedReader, io.TextIOWrapper, http.client.HTTPResponse, subprocess.Popen,
             threading.Thread, http.server.BaseHTTPRequestHandler):
    PY_PROTOCOLS |= {a for a in dir(_cls) if not a.startswith("_")}


# Each planted copy of the tree parses the same modules; the trees are only read, so each text is parsed once.
parse_python = functools.lru_cache(maxsize=None)(ast.parse)


def python_findings(texts):
    """`python`: a function, class, method or module constant in tests/ or tools/ that nothing reaches, an argparse option that neither its module nor a module its parsed namespace goes to reads, and a test module no suite import, workflow step or live doc's command line runs."""
    modules = {}
    for p in sorted(texts):
        if p.endswith(".py") and p.count("/") == 1 and p.split("/")[0] in ("tests", "tools"):
            modules[(p.split("/")[0], p.split("/")[1][:-3])] = (p, parse_python(texts[p], p))

    def module(top, name):
        # Scripts put their own directory on sys.path, and tools reach tests through a path insert.
        for t in (top, "tests" if top == "tools" else "tools"):
            if (t, name) in modules:
                return (t, name)
        return None

    imports = collections.defaultdict(dict)
    for mod, (_, tree) in modules.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    m = module(mod[0], a.name.split(".")[0])
                    if m:
                        imports[mod][a.asname or a.name] = ("module", m)
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                m = module(mod[0], node.module.split(".")[0])
                if m:
                    for a in node.names:
                        imports[mod][a.asname or a.name] = ("name", m, a.name)

    defs, by_name, top_defs, node_def = {}, collections.defaultdict(set), collections.defaultdict(dict), {}
    edges, roots = collections.defaultdict(set), set()

    def add(mod, kind, node, qual):
        i = modules[mod][0] + "::" + qual
        defs[i] = {"kind": kind, "name": node.name, "file": modules[mod][0], "line": node.lineno, "qual": qual}
        by_name[node.name].add(i)
        node_def[id(node)] = i
        return i

    def collect(mod, body, prefix, cls):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if cls is not None:
                    i = add(mod, "method", node, prefix + node.name)
                    if node.name.startswith(("__", "test", "visit_")) or node.name in PY_PROTOCOLS or node.name in PY_HOOKS:
                        roots.add(i)
                else:
                    i = add(mod, "function", node, prefix + node.name)
                    if not prefix:
                        top_defs[mod][node.name] = i
                if node.decorator_list:
                    roots.add(i)
                collect(mod, node.body, prefix + node.name + ".", None)
            elif isinstance(node, ast.ClassDef):
                i = add(mod, "class", node, prefix + node.name)
                if not prefix:
                    top_defs[mod][node.name] = i
                # unittest finds a TestCase by its base.
                if any((b.id if isinstance(b, ast.Name) else getattr(b, "attr", "")) == "TestCase" for b in node.bases):
                    roots.add(i)
                collect(mod, node.body, prefix + node.name + ".", node.name)
            elif isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                subs = []
                for field in ("body", "orelse", "finalbody", "handlers"):
                    for s in getattr(node, field, None) or []:
                        subs += s.body if isinstance(s, ast.ExceptHandler) else [s]
                collect(mod, subs, prefix, cls)

    for mod, (p, tree) in modules.items():
        collect(mod, tree.body, "", None)
        for node in tree.body:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) and node.value else []
            for t in targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name) and n.id not in top_defs[mod]:
                        i = p + "::" + n.id
                        defs[i] = {"kind": "constant", "name": n.id, "file": p, "line": node.lineno, "qual": n.id}
                        by_name[n.id].add(i)
                        top_defs[mod][n.id] = i
                        if n.id.startswith("__"):
                            roots.add(i)

    attrs, attr_strings = collections.defaultdict(collections.Counter), collections.defaultdict(set)

    class Refs(ast.NodeVisitor):
        def __init__(self, mod):
            self.mod, self.stack, self.scopes = mod, ["root:" + modules[mod][0]], [{}]

        def local_defs(self, body):
            d, work = {}, list(body)
            while work:
                s = work.pop()
                if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if id(s) in node_def:
                        d[s.name] = node_def[id(s)]
                    continue
                work.extend(ast.iter_child_nodes(s))
            return d

        def visit_FunctionDef(self, node):
            for x in node.decorator_list + node.args.defaults + [k for k in node.args.kw_defaults if k is not None]:
                self.visit(x)
            self.stack.append(node_def.get(id(node), self.stack[-1]))
            self.scopes.append(self.local_defs(node.body))
            for s in node.body:
                self.visit(s)
            self.scopes.pop()
            self.stack.pop()
        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node):
            for x in node.bases + node.keywords + node.decorator_list:
                self.visit(x)
            self.stack.append(node_def.get(id(node), self.stack[-1]))
            for s in node.body:
                self.visit(s)
            self.stack.pop()

        def resolve(self, name):
            for scope in reversed(self.scopes[1:]):
                if name in scope:
                    return {scope[name]}
            if name in top_defs[self.mod]:
                return {top_defs[self.mod][name]}
            imp = imports[self.mod].get(name)
            if imp and imp[0] == "name" and imp[2] in top_defs[imp[1]]:
                return {top_defs[imp[1]][imp[2]]}
            return set()

        def visit_Name(self, node):
            if not isinstance(node.ctx, ast.Store):
                edges[self.stack[-1]] |= self.resolve(node.id) - {self.stack[-1]}

        def visit_Attribute(self, node):
            self.generic_visit(node)
            attrs[self.mod][node.attr] += 1
            base = node.value
            if isinstance(base, ast.Name) and imports[self.mod].get(base.id, ("",))[0] == "module":
                other = imports[self.mod][base.id][1]
                if node.attr in top_defs[other]:
                    edges[self.stack[-1]].add(top_defs[other][node.attr])
                return
            if isinstance(node.ctx, ast.Store) and not (isinstance(base, ast.Name) and base.id == "self"):
                return
            # Anything else may be an instance, so the attribute reaches every definition of that name but a constant.
            edges[self.stack[-1]] |= {t for t in by_name.get(node.attr, ()) if defs[t]["kind"] != "constant"} - {self.stack[-1]}

        def visit_Call(self, node):
            self.generic_visit(node)
            f = node.func
            if isinstance(f, ast.Name) and f.id in ("getattr", "hasattr", "setattr") and len(node.args) >= 2 \
                    and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
                attr_strings[self.mod].add(node.args[1].value)
                edges[self.stack[-1]] |= by_name.get(node.args[1].value, set())

    for mod, (p, tree) in modules.items():
        Refs(mod).visit(tree)
        roots.add("root:" + p)
    seen, work = set(), list(roots)
    while work:
        n = work.pop()
        if n not in seen:
            seen.add(n)
            work.extend(edges.get(n, ()))
    found = {}
    for i, d in defs.items():
        if i not in seen:
            found[("python", d["file"], d["qual"])] = ["%s:%d, a %s nothing reaches" % (d["file"], d["line"], d["kind"])]
    # An option is read where its parsed namespace goes: its own module, a module a call hands the namespace to, and a module that calls a function returning it.
    readers = collections.defaultdict(set)
    called = collections.defaultdict(set)   # (module, function) -> the modules that call it
    for mod, (p, tree) in modules.items():
        for c in ast.walk(tree):
            if isinstance(c, ast.Call) and callee(imports[mod], c.func):
                called[callee(imports[mod], c.func)].add(mod)
    for mod, (p, tree) in modules.items():
        readers[mod].add(mod)
        if "parse_args" not in texts[p] and "parse_known_args" not in texts[p]:
            continue
        parsed = {t.id for node in ast.walk(tree) if isinstance(node, ast.Assign) and parses(node.value)
                  for t in node.targets if isinstance(t, ast.Name)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and any(isinstance(a, ast.Name) and a.id in parsed for a in node.args):
                other = callee(imports[mod], node.func)
                if other:
                    readers[mod].add(other[0])
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
                    isinstance(r, ast.Return) and r.value is not None and (parses(r.value) or isinstance(r.value, ast.Name) and r.value.id in parsed)
                    for r in ast.walk(node)):
                readers[mod] |= called[(mod, node.name)]
    for mod, (p, tree) in modules.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
                opts = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
                if not opts:
                    continue
                dest = next((k.value.value for k in node.keywords if k.arg == "dest" and isinstance(k.value, ast.Constant)), None)
                if dest is None:
                    longs = [o for o in opts if o.startswith("--")]
                    dest = (longs[0] if longs else opts[0]).lstrip("-").replace("-", "_")
                if not any(attrs[r].get(dest) or dest in attr_strings[r] or "vars(" in texts[modules[r][0]] for r in readers[mod]):
                    found[("python", p, max(opts, key=len))] = ["%s:%d, an option nothing reads" % (p, node.lineno)]
    # A test module is run by the suite, through run_tests.py's imports, by a workflow step, or by a command line in a live section of a doc.
    run, work = set(), [("tests", "run_tests")]
    while work:
        mod = work.pop()
        if mod in run or mod not in modules:
            continue
        run.add(mod)
        work += [v[1] for v in imports[mod].values()]
    commands = script_commands(texts)
    for (top, name), (p, _) in sorted(modules.items()):
        if top == "tests" and (top, name) not in run and p not in commands:
            found[("python", p, "module")] = ["%s: no suite import, workflow step or live doc's command line runs it" % p]
    return found


def parses(node):
    """Whether an expression is a parser's parse_args or parse_known_args call."""
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("parse_args", "parse_known_args")


def callee(imports, f):
    """(module, function) for a call's function from another module of tests/ or tools/, by the calling module's `imports`: `module.function` or a function imported by name; None for any other."""
    if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
        imp = imports.get(f.value.id)
        return (imp[1], f.attr) if imp and imp[0] == "module" else None
    if isinstance(f, ast.Name):
        imp = imports.get(f.id)
        return (imp[1], imp[2]) if imp and imp[0] == "name" else None
    return None


SCRIPT_RUN = re.compile(r"\bpy(?:thon3?)?(?:\s+-[A-Za-z]+(?:\s+utf8)?)*\s+((?:tests|tools)/\w+\.py)\b")


def script_commands(texts):
    """The scripts of tests/ and tools/ that a workflow step, or a command line in a live section of a doc, runs with Python; a record or a plan naming one runs nothing."""
    lines = [line for p, t in texts.items() if p.startswith(".github/") for line in t.split("\n")]
    for doc in docs_check.doc_files(texts):
        for _, line, region, fence in docs_check.doc_lines(doc, texts[doc]):
            if region == "live":
                lines += [line] if fence else [m.group(2) for m in docs_check.SPAN.finditer(line)]
    return {m.group(1) for line in lines for m in SCRIPT_RUN.finditer(line)}


# ----------------------------------------------------------------------------------------------------------------------
# Files: every source file is built, included or run.

def include_target(texts, path, name):
    for c in (os.path.dirname(path) + "/" + name, "src/" + name):
        c = os.path.normpath(c).replace(os.sep, "/")
        if c in texts:
            return c
    return None


# The sources CMake does not build that the linked check builds by hand beside its targets (linked_build), as docs/ASSETS.md builds them, and the executable each makes.
HAND_BUILT = {"tools/compare_cpu.cpp": "llmx-compare-cpu"}


def file_findings(texts):
    """`file`: a src/ file no translation unit CMake builds includes, a test or tool source neither CMake nor the linked check builds, and a test or tool header nothing includes."""
    cmake = texts["CMakeLists.txt"]
    built = set(re.findall(r"\b((?:src|tests|tools)/[\w/]+\.cpp)\b", cmake)) | set(HAND_BUILT)
    reached, work = set(), sorted(built)
    while work:
        p = work.pop()
        if p in reached or p not in texts:
            continue
        reached.add(p)
        work += [t for t in (include_target(texts, p, i) for i in re.findall(r'#\s*include\s+"([^"]+)"', texts[p])) if t]
    found = {}
    for p in sorted(texts):
        if p.endswith(CPP_SUFFIXES) and p.split("/")[0] in ("src", "tests", "tools") and p not in reached:
            why = "neither CMake nor the linked check builds it" if p.endswith(".cpp") else "no built source includes it"
            found[("file", p, "unreached")] = [p + ": " + why]
    return found


def source_findings(texts):
    found = {}
    for check in (name_findings, override_findings, shader_findings, flag_findings, python_findings, file_findings):
        found.update(check(texts))
    return found


# ----------------------------------------------------------------------------------------------------------------------
# The linked binaries: what each executable of an -O0 build keeps after --gc-sections, with every inline and static function emitted.

LINK_FLAGS = ["-O0", "-fno-inline", "-ffunction-sections", "-fdata-sections", "-fkeep-inline-functions", "-fkeep-static-functions"]


def compile_launcher(argv):
    """The compiler launcher of the linked build: the flags go last, so they override the targets' -O3."""
    os.execvp(argv[0], argv + LINK_FLAGS)


def split_args(s):
    """The top-level comma-separated parts inside the outer parentheses of a demangled signature's parameter list."""
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch in "(<[":
            depth += 1
        elif ch in ")>]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    return [p.strip() for p in parts + [cur] if p.strip()]


def strip_templates(s):
    out, depth, i = [], 0, 0
    while i < len(s):
        if s.startswith(("operator<", "operator>"), i):
            j = i + len("operator")
            while j < len(s) and s[j] in "<>=":
                j += 1
            out.append(s[i:j])
            i = j
            continue
        c = s[i]
        if c == "<":
            depth += 1
        elif c == ">" and depth:
            depth -= 1
        elif not depth:
            out.append(c)
        i += 1
    return "".join(out)


def parse_signature(sig):
    """(qualified name parts, parameter list text, trailing qualifiers) of a demangled function signature without template arguments, or None."""
    s = strip_templates(sig.replace("(anonymous namespace)::", "").replace("[abi:cxx11]", ""))
    depth, open_at = 0, None
    for i, c in enumerate(s):
        if c == "(":
            if depth == 0 and open_at is None and not s[:i].endswith("operator"):
                open_at = i
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0 and open_at is not None:
                parts = s[:open_at].split(" ")[-1].split("::")
                return parts, s[open_at + 1:i], s[i + 1:].strip()
    return None


PLATFORM_MACROS = {"_WIN32", "_MSC_VER", "__APPLE__", "__linux__", "__GNUC__", "__clang__", "__unix__", "__x86_64__", "_M_X64"}
LINUX_GCC = {"__linux__", "__GNUC__", "__unix__", "__x86_64__"}


def platform_value(cond):
    """Whether a conditional that names only platform macros holds for Linux GCC, or None when it names anything else."""
    names = set(re.findall(r"[A-Za-z_]\w*", cond)) - {"defined"}
    if not names or not names <= PLATFORM_MACROS:
        return None
    expr = re.sub(r"defined\s*\(?\s*(\w+)\s*\)?", lambda m: str(m.group(1) in LINUX_GCC), cond)
    expr = re.sub(r"\b[A-Za-z_]\w*\b", lambda m: m.group(0) if m.group(0) in ("True", "False") else str(m.group(0) in LINUX_GCC), expr)
    expr = expr.replace("&&", " and ").replace("||", " or ").replace("!", " not ")
    try:
        return bool(eval(expr, {"__builtins__": {}}))
    except SyntaxError:
        return None


def linux_only_names(texts):
    """The names src/ code calls only where a Linux GCC build leaves it out, in the branches of platform conditionals it does not take.
    A name also called in code Linux compiles is left to the linker, so a common name such as size is never skipped because another platform calls it too."""
    names, compiled = set(), set()
    for p, t in texts.items():
        if not p.startswith("src/") or not p.endswith(CPP_SUFFIXES):
            continue
        stack = []   # per conditional: [platform, this branch compiled, a branch taken]
        for line in code_of(t).split("\n"):
            m = re.match(r"\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)", line)
            if m:
                d, cond = m.group(1), m.group(2).strip()
                if d in ("ifdef", "ifndef"):
                    cond = ("!" if d == "ifndef" else "") + "defined(%s)" % cond
                if d in ("if", "ifdef", "ifndef"):
                    v = platform_value(cond)
                    stack.append([v is not None, v is not False, bool(v)])
                elif d == "elif" and stack:
                    top = stack[-1]
                    if top[0]:
                        top[1] = not top[2] and platform_value(cond) is not False
                        top[2] = top[2] or top[1]
                elif d == "else" and stack:
                    stack[-1][1] = not stack[-1][2] if stack[-1][0] else True
                elif d == "endif" and stack:
                    stack.pop()
                continue
            called = calls(line)
            if all(frame[1] for frame in stack):
                compiled |= called
            else:
                names |= called
    return names - compiled


CALL = re.compile(r"((?:[A-Za-z_]\w*\s*::\s*)*)([A-Za-z_]\w*)\s*\(")
STATEMENT_WORDS = {"return", "else", "case", "throw", "new", "delete", "co_return", "co_await", "sizeof", "not", "and", "or", "do"}


def calls(line):
    """The names a line of stripped code calls; a name after a type, as a declaration or definition has it, is no call."""
    out = set()
    for m in CALL.finditer(line):
        before = line[:m.start()].rstrip()
        word = re.search(r"([A-Za-z_]\w*)$", before)
        declared = word is not None and word.group(1) not in STATEMENT_WORDS or before.endswith(("*", "~")) \
            or before.endswith(">") and not before.endswith("->") or before.endswith("&") and not before.endswith("&&")
        if not declared:
            out.add(m.group(2))
    return out


def source_of(src, parts):
    """The src/ file that defines the function `parts` names, from `src`, the stripped code by path: its class's file for a member, else the file with its body."""
    name, owner = parts[-1], parts[-2] if len(parts) > 1 else None
    if owner:
        homes = sorted(p for p, c in src.items() if re.search(r"\b(?:class|struct|union)\s+" + re.escape(owner) + r"\b[^;{]*\{", c))
        if homes:
            return next((p for p in homes if re.search(r"\b" + re.escape(name.lstrip("~")) + r"\s*\(", src[p])), homes[0])
    bodies = sorted(p for p, c in src.items() if re.search(r"\b" + re.escape(name) + r"\s*\([^;{]*\)[^;{()]*\{", c))
    return next((p for p in bodies if p.endswith(".cpp")), bodies[0] if bodies else None)


def special_written(code, parts, params):
    """For a constructor, destructor or assignment: whether its class, in stripped `code`, declares that member with this many parameters and not as = default or = delete."""
    cls, name = parts[-2], parts[-1]
    m = re.search(r"\b(?:class|struct|union)\s+" + re.escape(cls) + r"\b[^;{]*\{", code)
    if not m:
        return False
    body = code[m.end():body_end(code, m.end() - 1)]
    what = re.escape(name) if name != "operator=" else r"operator\s*="
    for d in re.finditer(r"(?<![\w:~])" + what + r"\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)([^;{]*)", body):
        count = len(split_args(d.group(1))) if d.group(1).strip() not in ("", "void") else 0
        if count == params and not re.search(r"=\s*(?:default|delete)", d.group(2)):
            return True
    return False


def readable(sym):
    """A demangled signature as a finding names it: no template arguments, anonymous namespace or ABI tag, and std::string for the library's string."""
    s = strip_templates(sym.replace("(anonymous namespace)::", "").replace("[abi:cxx11]", ""))
    return s.replace("std::__cxx11::basic_string", "std::string")


def nm(path):
    r = subprocess.run(["nm", "-C", "--defined-only", path], capture_output=True, text=True, check=True)
    out = set()
    for line in r.stdout.splitlines():
        parts = line.split(" ", 2)
        if len(parts) == 3 and parts[1] in "TtWwi":
            out.add(parts[2])
    return out


def linked_build(root, build, jobs):
    """Configure and build every target with the Vulkan backend at -O0 through the launcher, and the hand-built comparison tool beside them."""
    here = os.path.abspath(__file__)
    subprocess.run(["cmake", "-S", root, "-B", build, "-DCMAKE_BUILD_TYPE=", "-DLLMX_HAS_BACKEND_VULKAN=ON",
                    "-DCMAKE_CXX_COMPILER_LAUNCHER=%s;%s;--compile" % (sys.executable, here),
                    "-DCMAKE_EXE_LINKER_FLAGS=-Wl,--gc-sections"], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["cmake", "--build", build, "--parallel", str(jobs)], check=True, stdout=subprocess.DEVNULL)
    # The hand-built sources are built against src/ as docs/ASSETS.md builds them.
    cxx = re.search(r"CMAKE_CXX_COMPILER:\w+=(.*)", open(os.path.join(build, "CMakeCache.txt")).read()).group(1).strip()
    for source, exe in HAND_BUILT.items():
        obj = os.path.join(build, exe + ".o")
        subprocess.run([cxx, "-std=c++17", "-mavx2", "-mfma", "-mf16c", "-I", os.path.join(root, "src"), "-c",
                        os.path.join(root, source), "-o", obj] + LINK_FLAGS, check=True)
        subprocess.run([cxx, obj, "-o", os.path.join(build, exe), "-Wl,--gc-sections", "-pthread"], check=True)


def linked_findings(texts, build):
    """`linked-unreached`: a src/ function no executable keeps; `linked-test-only`: one only test executables keep.
    Functions are their demangled signatures without template arguments; lambdas, function-local classes, special members nobody wrote or that are defaulted, constexpr functions and functions whose name only code a Linux build leaves out calls are left out, as are functions the name check already reports."""
    cmake = texts["CMakeLists.txt"]
    kinds = dict({"llmx": "product"}, **{exe: "product" for exe in HAND_BUILT.values()})
    for target, source in re.findall(r"add_executable\(([\w-]+)\s+([\w/.]+)", cmake):
        kinds[target] = "product" if source.startswith(("src/", "tools/")) else "test"
    universe = set()
    for directory, _, files in os.walk(os.path.join(build, "CMakeFiles")):
        for f in files:
            if f.endswith(".o") and "/src/" in os.path.join(directory, f).replace(os.sep, "/"):
                universe |= nm(os.path.join(directory, f))
    kept = collections.defaultdict(set)
    for exe, kind in kinds.items():
        path = os.path.join(build, exe)
        # A test only another platform builds is not here; a product executable always is.
        if not os.path.exists(path) and kind == "test":
            continue
        if not os.path.exists(path):
            raise Layout("%s is missing from the linked build" % exe)
        for s in nm(path):
            kept[strip_templates(s)].add(kind)
    # A build whose objects kept no functions would report nothing and pass.
    if sum(1 for s in universe if "product" in kept.get(strip_templates(s), ())) < 100:
        raise Layout("the linked build's executables keep almost none of the functions its objects define")
    namespaces = set()
    for p, t in texts.items():
        if p.startswith("src/"):
            namespaces |= set(re.findall(r"\bnamespace\s+(\w+)", t))
    src = {p: code_of(t) for p, t in texts.items() if p.startswith("src/") and p.endswith(CPP_SUFFIXES)}
    skip, reported = linux_only_names(texts), name_findings(texts)
    reported = {(k[1], k[2]) for k in reported if k[0] in ("unused", "test-only")}
    found = {}
    for sym in sorted(universe):
        # A lambda, and a class local to a function, live and die with their function.
        if "{lambda" in sym or ")::" in sym.replace("(anonymous namespace)::", "") or sym.startswith(("std::", "__gnu_cxx::", "__cxx")):
            continue
        sig = parse_signature(sym)
        if not sig:
            continue
        parts, params, _ = sig
        if len(parts) > 1 and parts[0] not in namespaces or parts[-1] in skip:
            continue
        where = kept.get(strip_templates(sym), set())
        if "product" in where:
            continue
        path = source_of(src, parts)
        if not path:
            continue
        special = len(parts) > 1 and (parts[-1] in (parts[-2], "~" + parts[-2]) or parts[-1] == "operator=")
        count = len(split_args(params)) if params.strip() not in ("", "void") else 0
        if special and not special_written(src[path], parts, count):
            continue
        # A constexpr function may run only where a constant is evaluated, which leaves no call in any executable.
        if re.search(r"(?:^|[;{}])[^;{}()]*\b(?:constexpr|consteval)\b[^;{}()]*\b" + re.escape(parts[-1]) + r"\s*\(", src[path]):
            continue
        if (path, "::".join(parts[-2:])) in reported or (path, parts[-1]) in reported:
            continue
        check = "linked-test-only" if "test" in where else "linked-unreached"
        found[(check, path, readable(sym))] = ["kept by %s" % (", ".join(sorted(where)) or "no executable")]
    return found


# ----------------------------------------------------------------------------------------------------------------------
# Planted faults: each check must report the fault planted in a copy of the tree, and the list must refuse an entry that no longer occurs.

def plant(texts, path, before, add):
    """A copy of `texts` with `add` inserted into `path` before the first `before`, or appended when `before` is None."""
    copy = dict(texts)
    t = copy.get(path, "")
    if before is None:
        copy[path] = t + add
    else:
        i = t.index(before)
        copy[path] = t[:i] + add + t[i:]
    return copy


def self_test(texts, found, listed):
    """The planted faults: a dead function, one after a function with a braced default argument, a function only tests call, an unused value of an enum a template parameter shares a name with, a macro never used, an override nothing calls, one declared over several lines and one with a braced default argument, an unreached shader, a flag whose value only the help prints, a flag parsed and dropped, an unreached Python function, an option only another module's namespace reads, an unbuilt test source a record names, and a stale list entry.
    The faults one check reports share a copy of the tree, so each check runs once."""
    # The planted flags go before the --ignore-eos branch, found by its comparison whatever the parser names its spelling.
    anchor = re.search(r'else if \(\w+ == "--ignore-eos"\)', texts[MAIN])
    if not anchor:
        raise AssertionError("dead-code self-test: main.cpp has no --ignore-eos branch to plant flags before")
    flag_line = anchor.group(0)
    names = plant(texts, "src/core/utf8.hpp", None, "\nnamespace utf8 {\ninline int planted_dead_function() { return 1; }\n"
                  "inline int planted_probe() { return 2; }\nstruct PlantedArgs { int n = 0; };\n"
                  "inline int planted_defaulted(PlantedArgs a = {}) { return a.n; }\ninline int planted_after_default() { return 4; }\n}\n"
                  "#define PLANTED_MACRO 1\n")
    names = plant(plant(names, "tests/json.cpp", "int main(", "static int planted_use = utf8::planted_probe();\n"),
                  "src/core/json.hpp", "Null, Bool,", "PlantedKind, ")
    flags = plant(plant(texts, MAIN, flag_line, 'else if (a == "--planted-flag") exec.planted_knob = true;\n                '
                        'else if (a == "--planted-noop") flag_value(argc, argv, i, a);\n                '),
                  MAIN, '<< "  --depth N', "<< defaults.planted_knob\n            ")
    python = plant(plant(texts, "tests/version.py", None, "\n\ndef planted_helper():\n    return 3\n"),
                   "tools/fetch_test_models.py", "    args = parser.parse_args(argv)", '    parser.add_argument("--model", help="planted")\n')
    cases = [
        (name_findings, names, [
            ("a dead function", ("unused", "src/core/utf8.hpp", "planted_dead_function")),
            ("a dead function after one with a braced default argument", ("unused", "src/core/utf8.hpp", "planted_after_default")),
            ("a function only a test calls", ("test-only", "src/core/utf8.hpp", "planted_probe")),
            ("an unused value of an enum a template parameter shares a name with", ("unused", "src/core/json.hpp", "T::PlantedKind")),
            ("a macro never used", ("macro", "src/core/utf8.hpp", "PLANTED_MACRO"))]),
        (override_findings, plant(texts, "src/core/utf8.hpp", None, "\nnamespace utf8 {\nstruct PlantedArg { int n = 0; };\n"
                                  "struct PlantedBase {\n    virtual int planted_virtual() = 0;\n    virtual int planted_lines(int a,\n                              int b) = 0;\n"
                                  "    virtual int planted_default(PlantedArg p = {}) = 0;\n};\n"
                                  "struct PlantedImpl : PlantedBase {\n    int planted_virtual() override { return 1; }\n"
                                  "    int planted_lines(int a,\n                      int b) override { return a + b; }\n"
                                  "    int planted_default(PlantedArg p = {}) override { return p.n; }\n};\n}\n"), [
            ("an override nothing calls", ("override", "src/core/utf8.hpp", "PlantedImpl::planted_virtual")),
            ("an override declared over several lines", ("override", "src/core/utf8.hpp", "PlantedImpl::planted_lines")),
            ("an override with a braced default argument", ("override", "src/core/utf8.hpp", "PlantedImpl::planted_default"))]),
        (shader_findings, dict(texts, **{SHADERS + "planted.comp": "#version 450\nvoid main() {}\n"}), [
            ("an unreached shader", ("shader", SHADERS + "planted.comp", "not built"))]),
        (shader_findings, plant(texts, VK, '    X(K_ADD,',
                               '    X(K_PLANTED, "planted", kSpvAdd, sizeof(kSpvAdd), 2, nullptr) \\\n'), [
            ("an orphan kernel catalog ID", ("shader", VK, "K_PLANTED"))]),
        (shader_findings, dict(texts, **{VK: texts[VK].replace("kSpvAdd, sizeof(kSpvAdd),", "kSpvSiluMul, sizeof(kSpvSiluMul),", 1)}), [
            ("an embedded module omitted from the catalog", ("shader", VK, "kSpvAdd"))]),
        (shader_findings, dict(texts, **{VK: texts[VK].replace('X(K_ADD, "add",', 'X(K_ADD, "",', 1)}), [
            ("a missing kernel diagnostic name", ("shader", VK, "K_ADD.name"))]),
        (flag_findings, flags, [
            ("a flag whose value only the help prints", ("flag", MAIN, "--planted-flag")),
            ("a flag parsed and dropped", ("flag", MAIN, "--planted-noop"))]),
        (python_findings, python, [
            ("an unreached Python function", ("python", "tests/version.py", "planted_helper")),
            ("an option only another module's namespace reads", ("python", "tools/fetch_test_models.py", "--model"))]),
        (file_findings, plant(dict(texts, **{"tests/planted.cpp": "int main() { return 0; }\n"}), "docs/STATUS.md", None,
                              "\n`tests/planted.cpp` was built by hand.\n"), [
            ("an unbuilt test source a record names", ("file", "tests/planted.cpp", "unreached"))]),
    ]
    ok, count = True, 1
    for check, tree, faults in cases:
        got = check(tree)
        for what, key in faults:
            count += 1
            if key not in got or key in found:
                print("  self-test: %s was not reported as %s" % (what, " | ".join(key)))
                ok = False
    stale = dict(listed)
    stale[("unused", "src/core/utf8.hpp", "planted_gone")] = (1, "a planted entry")
    if common.settle_findings(found, SOURCE_CHECKS, stale, say=lambda *a: None):
        print("  self-test: a listed finding that does not occur was accepted")
        ok = False
    return ok, count


def run():
    t0 = time.time()
    texts, _ = common.read_tree()
    listed = common.load_known_findings()
    found = source_findings(texts)
    planted_ok, planted = self_test(texts, found, listed)
    ok = common.settle_findings(found, SOURCE_CHECKS, listed)
    counts = collections.Counter(k[0] for k in found)
    print("dead-code: %d findings (%s) against the list; %d planted faults; %.1f s  [%s]" % (
        len(found), ", ".join("%s %d" % (c, counts[c]) for c in SOURCE_CHECKS if counts[c]), planted, time.time() - t0,
        "ok" if ok and planted_ok else "FAIL"))
    return ok and planted_ok


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--compile":
        compile_launcher(sys.argv[2:])
    parser = argparse.ArgumentParser(description="Dead-code checks; without options, the source checks the suite runs.")
    parser.add_argument("--linked", metavar="BUILD", help="build every target at -O0 with the Vulkan backend into BUILD and check what the executables keep (Linux, GCC)")
    parser.add_argument("--jobs", type=int, default=4, help="parallel build jobs for --linked")
    args = parser.parse_args()
    if not args.linked:
        return 0 if run() else 1
    if not sys.platform.startswith("linux") or not shutil.which("nm"):
        parser.error("--linked needs Linux with GCC, GNU ld, binutils, CMake, the Vulkan headers and glslc")
    t0 = time.time()
    texts, _ = common.read_tree()
    build = os.path.abspath(args.linked)
    linked_build(common.ROOT, build, args.jobs)
    t1 = time.time()
    listed = common.load_known_findings()
    found = linked_findings(texts, build)
    ok = common.settle_findings(found, LINKED_CHECKS, listed)
    counts = collections.Counter(k[0] for k in found)
    print("dead-code --linked: %d findings (%s) against the list; build %.0f s, analysis %.1f s  [%s]" % (
        len(found), ", ".join("%s %d" % (c, counts[c]) for c in LINKED_CHECKS if counts[c]), t1 - t0, time.time() - t1, "ok" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
