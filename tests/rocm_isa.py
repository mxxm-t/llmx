"""rocm-isa: the ROCm kernels as built, without a device.
What it holds and why is in docs/TESTS.md.
"""
import os
import re
import subprocess
import sys
import tempfile

TARGETS = ["gfx906", "gfx942", "gfx1151"]
KERNEL = "llmx_delta_step"
# The end label sits at the kernel's last statement, so only its epilogue lies past it.
END_SLACK = 64

INS = re.compile(r"^\s+(\S.*?)\s*//\s*([0-9A-Fa-f]+):")
LIT = re.compile(r"s_mov_b32 s\d+, (-?0x[0-9a-fA-F]+|-?\d+)$")


def prefetch_finding(dis, start, size):
    """None when the kernel's one code prefetch reads only its own code up to near its end, else what is wrong."""
    end = start + size
    # gfx11's s_delay_alu only schedules, so it is not the instruction that follows.
    ins = [(int(m.group(2), 16), m.group(1).strip()) for m in (INS.search(l) for l in dis.splitlines()) if m and not m.group(1).startswith("s_delay_alu")]
    # A pc followed by s_add_u32 is a pc-relative address of data; any other is the prefetch's read of the kernel's code.
    at = [i for i, (_, t) in enumerate(ins) if t.startswith("s_getpc_b64") and not (i + 1 < len(ins) and ins[i + 1][1].startswith("s_add_u32"))]
    if len(at) != 1:
        return "%d code reads from s_getpc_b64, want the prefetch's one" % len(at)
    i = at[0]
    if i + 1 >= len(ins):
        return "nothing after s_getpc_b64"
    pc, nxt = ins[i + 1]
    m = LIT.match(nxt)
    if not m:
        return "s_getpc_b64 is not followed by the end label's distance (%s), so its reads have no bound" % nxt
    lit = int(m.group(1), 0)
    if lit >= 1 << 31:
        lit -= 1 << 32
    if lit <= 0:
        return "end label %+d from pc 0x%x, not after it" % (lit, pc)
    if pc + lit > end:
        return "end label 0x%x past the symbol end 0x%x" % (pc + lit, end)
    if end - (pc + lit) > END_SLACK:
        return "end label 0x%x is %d B before the symbol end 0x%x, so the prefetch misses the kernel's tail" % (pc + lit, end - (pc + lit), end)
    lines = (lit + (pc & 63)) // 64
    last = (pc & ~63) + 64 * (lines - 1) + 4
    if last > end:
        return "last read 0x%x past the symbol end 0x%x" % (last, end)
    return None


def planted():
    """Each fault the check exists for must be reported, and the good form must pass."""
    def dis(*lines):
        return "".join("\t%s // %012X: 0\n" % (t, a) for a, t in lines)
    start, size = 0x1000, 0x2000
    good = dis((0x1100, "s_getpc_b64 s[4:5]"), (0x1104, "s_mov_b32 s6, 0x1ee4"), (0x110C, "s_and_b32 s4, s4, -64"))
    cases = {
        "fixed": (dis((0x1100, "s_getpc_b64 s[4:5]"), (0x1104, "s_and_b32 s4, s4, -64")), True),
        "before": (dis((0x1100, "s_getpc_b64 s[4:5]"), (0x1104, "s_mov_b32 s6, 0xffffff00")), True),
        "past": (dis((0x1100, "s_getpc_b64 s[4:5]"), (0x1104, "s_mov_b32 s6, 0x2000")), True),
        "short": (dis((0x1100, "s_getpc_b64 s[4:5]"), (0x1104, "s_mov_b32 s6, 0x100")), True),
        "zero": (dis((0x1100, "s_getpc_b64 s[4:5]"), (0x1104, "s_mov_b32 s6, 0")), True),
        "two": (good + dis((0x1200, "s_getpc_b64 s[8:9]"), (0x1204, "s_mov_b32 s6, 0x1de4")), True),
        "address": (good + dis((0x1200, "s_getpc_b64 s[8:9]"), (0x1204, "s_add_u32 s8, s8, 0xffed96b4")), False),
        "unbounded": (dis((0x1100, "s_getpc_b64 s[8:9]"), (0x1104, "s_andn2_b32 s8, s8, 63"), (0x1200, "s_getpc_b64 s[4:5]"), (0x1204, "s_add_u32 s4, s4, 0xffed96b4")), True),
        "address only": (dis((0x1200, "s_getpc_b64 s[4:5]"), (0x1204, "s_add_u32 s4, s4, 0xffed96b4")), True),
        "none": (dis((0x1100, "s_endpgm")), True),
        "good": (good, False),
    }
    bad = []
    for name, (text, fails) in cases.items():
        if (prefetch_finding(text, start, size) is not None) != fails:
            bad.append(name)
    return bad


def main():
    if len(sys.argv) != 3:
        print("usage: rocm_isa.py <build>/generated/rocm <ROCm llvm bin dir>")
        return 2
    gen, bin_dir = sys.argv[1], sys.argv[2]
    bad = planted()
    if bad:
        print("FAIL planted cases not reported right: %s" % ", ".join(bad))
        return 1
    bundle = os.path.join(gen, "f32.hipfb")
    tool = lambda n: os.path.join(bin_dir, n)
    listed = subprocess.run([tool("clang-offload-bundler"), "--list", "--type=o", "--input=" + bundle], capture_output=True, text=True, check=True).stdout
    # A build for other targets is supported (docs/BUILD.md), so only the targets the bundle holds are read.
    built = [t for t in TARGETS if "hipv4-amdgcn-amd-amdhsa--" + t in listed.split()]
    for t in TARGETS:
        if t not in built:
            print("SKIP %s: not built in %s" % (t, bundle))
    if not built:
        return 77
    fails = 0
    tmp = tempfile.TemporaryDirectory()
    for t in built:
        triple = "hipv4-amdgcn-amd-amdhsa--" + t
        co = os.path.join(tmp.name, t + ".co")
        subprocess.run([tool("clang-offload-bundler"), "--unbundle", "--type=o", "--input=" + bundle, "--targets=" + triple, "--output=" + co], check=True)
        sym = subprocess.run([tool("llvm-readelf"), "-s", co], capture_output=True, text=True, check=True).stdout
        found = [l.split() for l in sym.splitlines() if l.split()[-1:] == [KERNEL]]
        # The symbol is listed in both .dynsym and .symtab.
        found = sorted({(f[1], f[2]) for f in found if len(f) >= 8 and f[3] == "FUNC"})
        if len(found) != 1:
            print("FAIL %s: %d %s symbols" % (t, len(found), KERNEL))
            fails += 1
            continue
        start, size = int(found[0][0], 16), int(found[0][1], 0)
        dis = subprocess.run([tool("llvm-objdump"), "-d", "--disassemble-symbols=" + KERNEL, co], capture_output=True, text=True, check=True).stdout
        why = prefetch_finding(dis, start, size)
        print("%s %s %s %s" % ("FAIL" if why else "PASS", t, KERNEL, why or "code prefetch inside the kernel"))
        fails += why is not None
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
