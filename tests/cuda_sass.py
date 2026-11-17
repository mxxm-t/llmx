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
    "llmx_q4_k_rows16_": "step 4, first tuning on real shapes, the Q4_K 16-row build",
    "llmx_matmul_rows_q8_0": "step 4, first tuning on real shapes, the Q8_0 row build",
    "llmx_q5_k_rows": "step 4, first tuning on real shapes, the Q5_K row builds at 255 registers on sm_120 and 64 on sm_90",
    "llmx_q6_k_rows8_": "step 4, the int8 class",
    "llmx_q8_0_half8_": "step 4, the int8 class, the pair mask of the scale exchange at sm_80's 80 registers",
    "llmx_q4_k_routed": "step 8, the routed MoE operations",
    "llmx_q5_k_routed": "step 8, the routed MoE operations",
    "llmx_q6_k_routed": "step 8, the routed MoE operations",
    "llmx_q8_0_routed": "step 8, the routed MoE operations",
    "llmx_group_": "step 6, tensor groups",
    "llmx_mxfp4_": "step 8, MXFP4",
    "llmx_route_step_": "step 8, the deepseek4 one-launch router, its eight sorted keys a lane on one wave: a few dozen bytes on sm_90 and sm_120",
    "llmx_indexer_": "step 8, the deepseek4 operations",
    "llmx_sparse_": "step 8, the deepseek4 sparse attention: llmx_sparse_scores_tile on sm_120 and the few-row builds on sm_90, which no card has run",
    "llmx_moe_": "step 8, the routed MoE operations",
    "llmx_q4_0_": "step 8, Q4_0",
    "llmx_q4_1_": "step 8, Q4_1",
    "llmx_q5_0_": "step 8, Q5_0",
    "llmx_q5_1_": "step 8, Q5_1",
    "llmx_iq4_nl_": "step 8, IQ4_NL",
    "llmx_matmul_rows_iq4_nl": "step 8, IQ4_NL",
}


# The int8 class's witnesses: the tensor-core prompt tiles of every int8 tile type, and a Q8_0 and a K-quant row build on the 8-bit twin.
IMMA_KERNELS = ("llmx_q8_0_tile8", "llmx_q8_0_tile8_tall") + tuple(f"llmx_{t}_tile8" for t in ("q4_0", "q4_1", "q5_0", "q5_1", "iq4_nl", "mxfp4", "q4_k", "q5_k", "q6_k"))
DP4A_KERNELS = ("llmx_q8_0_half8_1", "llmx_q4_k_rows8_1")

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
        # The int8 class (docs/CUDA.md, step 4): the Q8_0 prompt tiles sum on the integer tensor cores and the 8-bit rows on the four-way dot.
        funcs = sass_by_function(cuobjdump, os.path.join(gen, "int16.fatbin"), arch)
        no_imma = [fn for fn in IMMA_KERNELS if not any("IMMA" in l for l in funcs.get(fn, []))]
        require(not no_imma, f"{arch}: the int8 prompt tiles sum with IMMA" + (" (not " + ", ".join(no_imma) + ")" if no_imma else ""))
        no_dp4a = [fn for fn in DP4A_KERNELS if not any("IDP.4A" in l for l in funcs.get(fn, []))]
        require(not no_dp4a, f"{arch}: the int8 rows sum with IDP.4A" + (" (not " + ", ".join(no_dp4a) + ")" if no_dp4a else ""))
    for kernel in ("f32", "int16"):
        text = subprocess.run([cuobjdump, "-ptx", os.path.join(gen, kernel + ".fatbin")], capture_output=True, text=True, check=True).stdout
        asked = sorted(set(re.findall(r"([a-z0-9]+(?:\.[a-z0-9]+)*\.ftz(?:\.[a-z0-9]+)*)", text)) - {"ex2.approx.ftz.f32"})
        require(not asked, f"{kernel}: the PTX asks for flushing only in the math library's ex2.approx.ftz" + (" (" + ", ".join(asked[:8]) + ")" if asked else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
