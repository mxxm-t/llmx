"""cuda-sass: the CUDA kernels as built, without a device.
What it holds and why is in docs/TESTS.md.
"""
import os
import re
import subprocess
import sys

# The toolkit release the spill list below was made on.
SPILL_TOOLKIT = "12.9"
ARCHS = ["sm_80", "sm_86", "sm_89", "sm_90", "sm_120"]

# Kernel families that spill today on some target, each with the step of docs/CUDA.md (Order of work) whose port is held to remove the spills; which kernels of a family spill moves between targets and toolkits.
SPILL_ALLOWED = {
    "llmx_q4_k_": "step 3, the F16 class",
    "llmx_q5_k_": "step 3, the F16 class",
    "llmx_q6_k_": "step 3, the F16 class",
    "llmx_q8_0_": "step 3, the F16 class",
    "llmx_matmul_rows_q8_0": "step 3, the F16 class",
    "llmx_group_": "step 6, tensor groups",
    "llmx_mxfp4_": "step 8, MXFP4",
    "llmx_hc_": "step 8, the deepseek4 operations",
    "llmx_indexer_": "step 8, the deepseek4 operations",
    "llmx_moe_": "step 8, the routed MoE operations",
    "llmx_q5_0_": "step 8, Q5_0",
    "llmx_q5_1_": "step 8, Q5_1",
    "llmx_iq4_nl_": "step 8, IQ4_NL",
}


def family(fn):
    return next((f for f in SPILL_ALLOWED if fn.startswith(f)), None)

def spills(report):
    """Kernel -> set of targets on which ptxas reports spill stores or loads."""
    out, arch, fn = {}, None, None
    for line in report.splitlines():
        m = re.search(r"Compiling entry function '(\w+)' for '(sm_\d+)'", line)
        if m:
            fn, arch = m.group(1), m.group(2)
            continue
        m = re.search(r"Function properties for (\w+)", line)
        if m:
            fn = m.group(1)
            continue
        m = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", line)
        if m and fn and (int(m.group(1)) or int(m.group(2))):
            out.setdefault(fn, set()).add(arch)
    return out


def sass_by_function(cuobjdump, fatbin, arch):
    text = subprocess.run([cuobjdump, "-sass", "-arch", arch, fatbin], capture_output=True, text=True, check=True).stdout
    out, fn = {}, None
    for line in text.splitlines():
        m = re.match(r"\s*Function : (\w+)", line)
        if m:
            fn = m.group(1)
            out[fn] = []
        elif fn:
            out[fn].append(line)
    return out


def main():
    if len(sys.argv) != 3:
        print("usage: cuda_sass.py GENERATED_CUDA_DIR CUOBJDUMP")
        return 2
    gen, cuobjdump = sys.argv[1], sys.argv[2]
    failures = []

    def require(ok, what):
        print(("PASS " if ok else "FAIL ") + what)
        if not ok:
            failures.append(what)

    spilled = {}
    for kernel in ("f32", "int16"):
        fatbin = os.path.join(gen, kernel + ".fatbin")
        elf = subprocess.run([cuobjdump, "--list-elf", fatbin], capture_output=True, text=True, check=True).stdout
        ptx = subprocess.run([cuobjdump, "--list-ptx", fatbin], capture_output=True, text=True, check=True).stdout
        missing = [a for a in ARCHS if "." + a + "." not in elf]
        require(not missing, f"{kernel}: SASS for every claimed target" + (" (missing " + ", ".join(missing) + ")" if missing else ""))
        require("sm_80" in ptx, f"{kernel}: PTX for compute_80, which later targets compile at load")
        with open(os.path.join(gen, kernel + ".ptxas"), encoding="utf-8", errors="replace") as f:
            for fn, archs in spills(f.read()).items():
                spilled.setdefault(fn, set()).update(archs)

    unexpected = sorted(fn for fn in spilled if not family(fn))
    require(not unexpected, "no kernel spills to local memory outside the listed families" + ("".join(f"\n  {fn} on {', '.join(sorted(spilled[fn]))}" for fn in unexpected)))
    stale = sorted(f for f in SPILL_ALLOWED if not any(family(fn) == f for fn in spilled))
    release = re.search(r"release (\d+\.\d+)", subprocess.run([cuobjdump, "--version"], capture_output=True, text=True, check=True).stdout)
    release = release.group(1) if release else "unknown"
    if release == SPILL_TOOLKIT:
        require(not stale, "every listed family still spills" + ("".join(f"\n  {f} no longer spills: remove its line" for f in stale)))
    elif stale:
        print(f"NOTE toolkit {release}, the spill list holds for {SPILL_TOOLKIT}: no longer spilling here: {', '.join(stale)}")

    for arch in ARCHS:
        funcs = sass_by_function(cuobjdump, os.path.join(gen, "f32.fatbin"), arch)
        matmul = [fn for fn in funcs if fn.startswith("llmx_matmul") and fn.endswith("_f32")]
        no_ffma = [fn for fn in matmul if not any("FFMA" in l for l in funcs[fn])]
        require(matmul and not no_ffma, f"{arch}: the {len(matmul)} F32 matmul kernels sum with FFMA" + (" (not " + ", ".join(no_ffma) + ")" if no_ffma else ""))
        ftz = sorted(fn for fn in matmul if any(".FTZ" in l for l in funcs[fn]))
        require(not ftz, f"{arch}: no F32 matmul kernel flushes subnormals" + (" (" + ", ".join(ftz[:8]) + ")" if ftz else ""))
    for kernel in ("f32", "int16"):
        text = subprocess.run([cuobjdump, "-ptx", os.path.join(gen, kernel + ".fatbin")], capture_output=True, text=True, check=True).stdout
        asked = sorted(set(re.findall(r"([a-z0-9]+(?:\.[a-z0-9]+)*\.ftz(?:\.[a-z0-9]+)*)", text)) - {"ex2.approx.ftz.f32"})
        require(not asked, f"{kernel}: the PTX asks for flushing only in the math library's ex2.approx.ftz" + (" (" + ", ".join(asked[:8]) + ")" if asked else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
