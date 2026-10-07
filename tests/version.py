import hashlib
import os
from pathlib import Path
import re

import common

SOURCES = "cmake/numerics-sources.txt"


def numerics_list(root):
    """The roots, the out-list and the allow-list of cmake/numerics-sources.txt, each a path with its reason."""
    ins, outs, allows = [], {}, {}
    for line in (Path(root) / SOURCES).read_text(encoding="ascii").splitlines():
        m = re.match(r"(in|out|allow) +([^ |]+)(?: *\| *(.*))?$", line)
        if not m:
            assert not line.strip() or line.startswith("#"), "%s: not an entry: %r" % (SOURCES, line)
        elif m[1] == "in":
            ins.append(m[2])
        else:
            assert m[3], "%s: %s %s gives no reason" % (SOURCES, m[1], m[2])
            (outs if m[1] == "out" else allows)[m[2]] = m[3]
    return ins, outs, allows


def is_out(path, outs):
    return any(path == o or path.startswith(o + "/") for o in outs)


def numerics_files(root, ins, outs):
    """Every file under the roots that no out line names, in path order."""
    files = []
    for i in ins:
        p = Path(root) / i
        if p.is_dir():
            files += [f.relative_to(root).as_posix() for f in p.rglob("*") if f.is_file()]
        else:
            files.append(i)
    return sorted(f for f in files if not is_out(f, outs))


def fingerprint(root, files, changed=None):
    """The numerics fingerprint as the build computes it: each file as "path hash\\n", the hash the SHA-256 of the file's bytes in hex with CRLF read as LF, and the SHA-256 of those lines.
    `changed` maps a path to bytes that stand for its own, for the planted faults."""
    text = ""
    for f in files:
        data = (changed or {}).get(f)
        if data is None:
            data = (Path(root) / f).read_bytes()
        text += "%s %s\n" % (f, hashlib.sha256(data.replace(b"\r\n", b"\n").hex().encode("ascii")).hexdigest())
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def included(root, path):
    """The repository files a source includes by a quoted name, from src/ or beside itself."""
    out = []
    text = (Path(root) / path).read_text(encoding="utf-8", errors="replace")
    for name in re.findall(r'^\s*#\s*include\s+"([^"]+)"', text, re.M):
        for base in ("src", os.path.dirname(path)):
            target = os.path.normpath(os.path.join(base, name)).replace("\\", "/")
            if (Path(root) / target).is_file():
                out.append(target)
                break
    return out


def check_list(root):
    """What holds the list complete: every src/ file is in or named out, every line names something, and no file inside includes one outside that no allow line argues."""
    ins, outs, allows = numerics_list(root)
    files = numerics_files(root, ins, outs)
    problems = []
    for path in list(ins) + list(outs) + list(allows):
        if not (Path(root) / path).exists():
            problems.append("%s names %s, which does not exist" % (SOURCES, path))
    for f in sorted(p.relative_to(root).as_posix() for p in (Path(root) / "src").rglob("*") if p.is_file()):
        if f not in files and not is_out(f, outs):
            problems.append("%s is neither in the fingerprint nor named out" % f)
    for a in allows:
        if not is_out(a, outs):
            problems.append("%s allows %s, which is not out" % (SOURCES, a))
    used = set()
    for f in files:
        if not f.startswith("src/"):
            continue
        for target in included(root, f):
            if is_out(target, outs):
                if target in allows:
                    used.add(target)
                else:
                    problems.append("%s, in the fingerprint, includes %s, which is out and not allowed" % (f, target))
    for a in sorted(set(allows) - used):
        problems.append("%s allows %s, which nothing in the fingerprint includes" % (SOURCES, a))
    return files, outs, problems


def planted(root, files, outs):
    """Planted faults: a byte changed inside moves the fingerprint, one outside does not, a line ending does not, and each rule of the list catches its fault."""
    base = fingerprint(root, files)
    inside = next(f for f in files if f.startswith("src/quant/"))
    data = (Path(root) / inside).read_bytes()
    assert fingerprint(root, files, {inside: data + b" "}) != base, "a byte changed in %s left the fingerprint" % inside
    assert fingerprint(root, files, {inside: data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")}) == base, "a line ending changed in %s moved the fingerprint" % inside
    outside = "src/cli/main.cpp"
    assert is_out(outside, outs) and outside not in files, outside
    assert fingerprint(root, numerics_files(root, *numerics_list(root)[:2])) == base, "the fingerprint is not the same twice"
    import shutil, tempfile
    with tempfile.TemporaryDirectory() as tmp:
        for part in ("src/core", "src/quant", "cmake"):
            shutil.copytree(Path(root) / part, Path(tmp) / part)
        for name in ("CMakeLists.txt", "build.bat"):
            (Path(tmp) / name).write_bytes(b"")
        listing = (Path(tmp) / SOURCES).read_text(encoding="ascii")
        kept = "\n".join(l for l in listing.splitlines() if not re.match(r"(out|allow) ", l) or re.search(r" src/core/", l)) + "\n"
        (Path(tmp) / SOURCES).write_text(kept, encoding="ascii")
        clean = [p for p in check_list(tmp)[2] if "nothing in the fingerprint includes" not in p]
        assert not clean, clean
        faults = (("an out line naming nothing", kept + "out src/core/gone.hpp | planted\n", None, "does not exist"),
                  ("an include of an out file", kept, ("src/quant/quant.hpp", b'#include "core/utf8.hpp"\n'), "includes src/core/utf8.hpp, which is out and not allowed"),
                  ("an allow line for a file that is in", kept + "allow src/core/fp16.hpp | planted\n", None, "which is not out"),
                  ("an entry without a reason", kept + "out src/core/fp16.hpp\n", None, "gives no reason"))
        for what, text, add, want in faults:
            (Path(tmp) / SOURCES).write_text(text, encoding="ascii")
            saved = None
            if add:
                saved = (Path(tmp) / add[0]).read_bytes()
                (Path(tmp) / add[0]).write_bytes(add[1] + saved)
            try:
                found = check_list(tmp)[2]
            except AssertionError as e:
                found = [str(e)]
            assert any(want in p for p in found), "planted fault not caught: %s: %s" % (what, found)
            if add:
                (Path(tmp) / add[0]).write_bytes(saved)
        out_file = Path(tmp) / "src/core/json.hpp"
        (Path(tmp) / SOURCES).write_text(kept, encoding="ascii")
        ins, outs2, _ = numerics_list(tmp)
        before = fingerprint(tmp, numerics_files(tmp, ins, outs2))
        out_file.write_bytes(out_file.read_bytes() + b" ")
        assert fingerprint(tmp, numerics_files(tmp, ins, outs2)) == before, "a byte changed in an out file moved the fingerprint"
    return 7


def run():
    p = common.run_process(["--version"], timeout=10)
    assert p.returncode == 0 and not p.stderr, (p.returncode, p.stderr)
    output = p.stdout.decode("ascii").strip()
    match = re.fullmatch(r"llmx (\d+\.\d+\.\d+)\+(unknown|g[0-9a-f]{12,40}(?:\.dirty)?) numerics ([0-9a-f]{16})", output)
    assert match, output
    config = (Path(common.ROOT) / "CMakeLists.txt").read_text(encoding="ascii")
    release = re.search(r"project\(llmx VERSION ([0-9.]+)", config)
    assert release and match[1] == release[1], (output, config)
    usage = common.run_process([], timeout=10)
    assert usage.returncode == 1 and usage.stdout.startswith(p.stdout.split(b" numerics ")[0] + b" - "), usage.stdout
    files, outs, problems = check_list(common.ROOT)
    assert not problems, "\n".join(problems)
    want = fingerprint(common.ROOT, files)
    assert match[3] == want[:16], "the binary's numerics fingerprint %s is not the tree's %s: it was built from other sources, or the build and this test compute it apart" % (match[3], want[:16])
    faults = planted(common.ROOT, files, outs)
    print("version: release/build identifier and usage banner agree; the numerics fingerprint %s is the tree's over %d files, every src/ file in or named out, "
          "no file inside including one outside unargued; %d planted faults  [ok]" % (want[:16], len(files), faults))
    return True
