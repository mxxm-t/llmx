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
    "llmx_sparse_": "step 8, the deepseek4 sparse attention: the few-row builds on sm_90, which no card has run",
    "llmx_moe_": "step 8, the routed MoE operations",
    "llmx_q4_0_": "step 8, Q4_0, the twin builds of Q5_0",
    "llmx_q4_1_": "step 8, Q4_1, the twin builds of Q5_1",
    "llmx_q5_0_": "step 8, Q5_0",
    "llmx_q5_1_": "step 8, Q5_1",
    "llmx_iq4_nl_": "step 8, IQ4_NL",
    "llmx_q4_k_tile8_tall": "change 3, the tall K tile: 4 bytes on sm_90 at two blocks an SM, one loop-invariant word reloaded a step, which the tall loop trim removes (docs/STATUS.md, CUDA prompt gap, change 3, second part, Left)",
    "llmx_q5_k_tile8_tall": "change 3, the tall K tile: 8 bytes on sm_90 at two blocks an SM, one loop-invariant pair reloaded a step, which the tall loop trim removes (docs/STATUS.md, CUDA prompt gap, change 3, second part, Left)",
    "llmx_q2_k_": "step 8, Q2_K's twin builds: up to 36 bytes, where the 256-thread bound lets ptxas trade them for a block an SM, and two blocks an SM spilled more and was not taken (docs/STATUS.md)",
    "llmx_q3_k_": "step 8, Q3_K's twin builds: up to 48 bytes, where the 256-thread bound lets ptxas trade them for a block an SM, and two blocks an SM spilled more and was not taken (docs/STATUS.md)",
    "llmx_iq4_xs_": "step 8, IQ4_XS's twin builds: up to 44 bytes, where the 256-thread bound lets ptxas trade them for a block an SM, and two blocks an SM spilled more and was not taken (docs/STATUS.md)",
    "llmx_iq2_xxs_": "step 8, IQ2_XXS's twin builds: up to 44 bytes, where the 256-thread bound lets ptxas trade them for a block an SM, and two blocks an SM spilled more and was not taken (docs/STATUS.md)",
    "llmx_iq2_xs_": "step 8, IQ2_XS's twin builds: up to 44 bytes, where the 256-thread bound lets ptxas trade them for a block an SM, and two blocks an SM spilled more and was not taken (docs/STATUS.md)",
    "llmx_iq3_s_": "step 8, IQ3_S's twin builds: up to 44 bytes, where the 256-thread bound lets ptxas trade them for a block an SM, and two blocks an SM spilled more and was not taken (docs/STATUS.md)",
    "llmx_iq2_s_": "step 8, IQ2_S's twin builds: up to 48 bytes, where the 256-thread bound lets ptxas trade them for a block an SM, and two blocks an SM spilled more and was not taken (docs/STATUS.md)",
    "llmx_iq3_xxs_": "step 8, IQ3_XXS's twin builds: up to 44 bytes, where the 256-thread bound lets ptxas trade them for a block an SM, and two blocks an SM spilled more and was not taken (docs/STATUS.md)",
    "llmx_iq1_s_": "step 8, IQ1_S's twin builds: up to 44 bytes, where the 256-thread bound lets ptxas trade them for a block an SM, and two blocks an SM spilled more and was not taken (docs/STATUS.md)",
    "llmx_iq1_m_": "step 8, IQ1_M's twin builds: up to 36 bytes, where the 256-thread bound lets ptxas trade them for a block an SM, and two blocks an SM spilled more and was not taken (docs/STATUS.md)",
    "llmx_q1_0_": "step 8, Q1_0's twin builds: up to 36 bytes, where the 256-thread bound lets ptxas trade them for a block an SM, and two blocks an SM spilled more and was not taken (docs/STATUS.md)",
    "llmx_q2_0_": "step 8, Q2_0's twin builds: up to 36 bytes, where the 256-thread bound lets ptxas trade them for a block an SM, and two blocks an SM spilled more and was not taken (docs/STATUS.md)",
    "llmx_nvfp4_": "step 8, NVFP4's twin builds: up to 36 bytes, where the 256-thread bound lets ptxas trade them for a block an SM, and two blocks an SM spilled more and was not taken (docs/STATUS.md)",
    "llmx_matmul_rows_iq2_xs": "step 8, IQ2_XS's exact row build: 8 bytes on sm_120, where the 256-thread bound lets ptxas trade them for a block an SM, and two blocks an SM spilled more and was not taken (docs/STATUS.md)",
    "llmx_matmul_rows_iq4_xs": "step 8, IQ4_XS's exact row build: 4 bytes on sm_120, where the 256-thread bound lets ptxas trade them for a block an SM, and two blocks an SM spilled more and was not taken (docs/STATUS.md)",
    "llmx_delta_step": "step 5, the qwen35 delta step: 8 bytes on sm_86, sm_89 and sm_90 since it took the int8 half-block form of feat/qwen27-delta-q8, and two blocks an SM spilled more and was not taken (docs/STATUS.md)",
}


# The int8 class's witnesses: the tensor-core prompt tiles of every int8 tile type, dense and routed, and a Q8_0 and a K-quant row build on the 8-bit twin.
INT8_TILE_TYPES = ("q4_0", "q4_1", "q5_0", "q5_1", "iq4_nl", "mxfp4", "q4_k", "q5_k", "q6_k")
# A type's 8-bit routed groups of up to 16 entries, and MXFP4's of up to 32, run in the stream (routed_stream), which is no tile and sums an entry's terms in block order on four-way dots.
IMMA_KERNELS = ("llmx_q8_0_tile8", "llmx_q8_0_tile8_tall") + tuple(f"llmx_{t}_{b}8" for t in INT8_TILE_TYPES for b in ("tile", "routed_tile", "routed_mid") if (t, b) != ("mxfp4", "routed_mid")) + tuple(f"llmx_{t}_tile8_tall" for t in ("q4_k", "q5_k", "q6_k"))
DP4A_KERNELS = ("llmx_q8_0_half8_1", "llmx_q4_k_rows8_1")
# The tensor-core K tile's main loop at 128 outputs: most instructions per 4096 multiply-adds (one m16n8k32) from its back branch on every target, the most measured at toolkit 12.9 rounded up plus one (docs/CUDA.md, change 3).
# Measured 26.7, 29.2 and 38.3 over the fold's floors of about 18.5, 18.5 and 28.5, so a fold or staging regrowth fails; a ceiling drops with each trim.
TALL_LOOP = {"llmx_q4_k_tile8_tall": 28, "llmx_q5_k_tile8_tall": 31, "llmx_q6_k_tile8_tall": 40}

# sizeof(TwinMap), which backend_core.hpp pins with a static_assert: a kernel taking the map has a by-value PTX parameter of this many bytes.
TWIN_MAP_BYTES = 120

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


def local_objects(report):
    """(kernel, target) pairs whose stack frame holds more than ptxas's spill stores: a local object, such as a copied parameter."""
    out, arch, fn = set(), None, None
    for line in report.splitlines():
        m = re.search(r"Compiling entry function '(\w+)' for '(sm_\d+)'", line)
        if m:
            fn, arch = m.group(1), m.group(2)
            continue
        m = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores", line)
        if m and fn and int(m.group(1)) > int(m.group(2)):
            out.add((fn, arch))
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


def imma_loop(lines):
    """(instructions, multiply-adds) of the longest loop holding IMMA, read from its back branch, or None."""
    ins = []
    for line in lines:
        m = re.match(r"\s+/\*([0-9a-f]{4,})\*/\s+(?:@!?U?P[0-9T] )?([A-Z0-9_.]+)[^;]*;", line)
        if m:
            ins.append((int(m.group(1), 16), m.group(2), line))
    at = {a: i for i, (a, _, _) in enumerate(ins)}
    best = None
    for i, (a, op, line) in enumerate(ins):
        target = re.search(r"0x([0-9a-f]+)", line.split(op, 1)[1]) if op == "BRA" else None
        if target and int(target.group(1), 16) < a and int(target.group(1), 16) in at:
            body = ins[at[int(target.group(1), 16)]:i + 1]
            macs = sum(int(m.group(1)) * int(m.group(2)) * int(m.group(3)) for _, o, _ in body for m in [re.match(r"IMMA\.(\d\d)(\d)(\d\d)", o)] if m)
            if macs and (best is None or len(body) > best[0]):
                best = (len(body), macs)
    return best


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

    spilled, objects = {}, set()
    for kernel in ("f32", "int16"):
        fatbin = os.path.join(gen, kernel + ".fatbin")
        elf = subprocess.run([cuobjdump, "--list-elf", fatbin], capture_output=True, text=True, check=True).stdout
        ptx = subprocess.run([cuobjdump, "--list-ptx", fatbin], capture_output=True, text=True, check=True).stdout
        missing = [a for a in ARCHS if "." + a + "." not in elf]
        require(not missing, f"{kernel}: SASS for every claimed target" + (" (missing " + ", ".join(missing) + ")" if missing else ""))
        require("sm_80" in ptx, f"{kernel}: PTX for compute_80, which later targets compile at load")
        with open(os.path.join(gen, kernel + ".ptxas"), encoding="utf-8", errors="replace") as f:
            report = f.read()
            for fn, archs in spills(report).items():
                spilled.setdefault(fn, set()).update(archs)
            objects |= local_objects(report)

    unexpected = sorted(fn for fn in spilled if not family(fn))
    require(not unexpected, "no kernel spills to local memory outside the listed families" + ("".join(f"\n  {fn} on {', '.join(sorted(spilled[fn]))}" for fn in unexpected)))
    stale = sorted(f for f in SPILL_ALLOWED if not any(family(fn) == f for fn in spilled))
    release = re.search(r"release (\d+\.\d+)", subprocess.run([cuobjdump, "--version"], capture_output=True, text=True, check=True).stdout)
    release = release.group(1) if release else "unknown"
    if release == SPILL_TOOLKIT:
        require(not stale, "every listed family still spills" + ("".join(f"\n  {f} no longer spills: remove its line" for f in stale)))
    elif stale:
        print(f"NOTE toolkit {release}, the spill list holds for {SPILL_TOOLKIT}: no longer spilling here: {', '.join(stale)}")

    # Every kernel taking a TwinMap reads it in place (LLMX_GRID_CONST), so none keeps a local object or touches local memory beyond the spills the list above governs.
    # The takers are read from the PTX, and every _major build must be among them.
    f32_ptx = subprocess.run([cuobjdump, "-ptx", os.path.join(gen, "f32.fatbin")], capture_output=True, text=True, check=True).stdout
    takers = set(re.findall(r"\.param \.align \d+ \.b8 (llmx_\w+)_param_\d+\[%d\]" % TWIN_MAP_BYTES, f32_ptx))
    for arch in ARCHS:
        funcs = sass_by_function(cuobjdump, os.path.join(gen, "f32.fatbin"), arch)
        matmul = [fn for fn in funcs if fn.startswith("llmx_matmul") and fn.endswith("_f32")]
        no_ffma = [fn for fn in matmul if not any("FFMA" in l for l in funcs[fn])]
        require(matmul and not no_ffma, f"{arch}: the {len(matmul)} F32 matmul kernels sum with FFMA" + (" (not " + ", ".join(no_ffma) + ")" if no_ffma else ""))
        ftz = sorted(fn for fn in matmul if any(".FTZ" in l for l in funcs[fn]))
        require(not ftz, f"{arch}: no F32 matmul kernel flushes subnormals" + (" (" + ", ".join(ftz[:8]) + ")" if ftz else ""))
        unheld = sorted(fn for fn in funcs if fn.endswith("_major") and fn not in takers)
        absent = sorted(takers - set(funcs))
        kept = sorted(fn for fn in takers if (fn, arch) in objects)
        local = sorted(fn for fn in takers & set(funcs) if arch not in spilled.get(fn, ()) and any(re.search(r"\b(LDL|STL)\b", l) for l in funcs[fn]))
        bad = [f"{fn} keeps a local object" for fn in kept] + [f"{fn} uses local memory" for fn in local] + [f"{fn} takes no map" for fn in unheld] + [f"{fn} has no SASS" for fn in absent]
        require(takers and not bad, f"{arch}: the {len(takers)} kernels taking a twin map keep no local object and use local memory only to spill" + (" (" + ", ".join(bad) + ")" if bad else ""))
        # The int8 class (docs/CUDA.md, step 4): the Q8_0 prompt tiles sum on the integer tensor cores and the 8-bit rows on the four-way dot.
        funcs = sass_by_function(cuobjdump, os.path.join(gen, "int16.fatbin"), arch)
        no_imma = [fn for fn in IMMA_KERNELS if not any("IMMA" in l for l in funcs.get(fn, []))]
        require(not no_imma, f"{arch}: the int8 prompt tiles sum with IMMA" + (" (not " + ", ".join(no_imma) + ")" if no_imma else ""))
        for fn, most in TALL_LOOP.items():
            loop = imma_loop(funcs.get(fn, []))
            per = loop[0] * 4096 / loop[1] if loop else None
            local = arch not in spilled.get(fn, ()) and any(re.search(r"\b(LDL|STL)\b", l) for l in funcs.get(fn, []))
            require(per is not None and per <= most and not local, f"{arch}: {fn}'s main loop takes at most {most} instructions per 4096 multiply-adds and no local memory" + (f" ({per:.1f}" + (", local memory" if local else "") + ")" if per is not None else " (no IMMA loop)"))
        no_dp4a = [fn for fn in DP4A_KERNELS if not any("IDP.4A" in l for l in funcs.get(fn, []))]
        require(not no_dp4a, f"{arch}: the int8 rows sum with IDP.4A" + (" (not " + ", ".join(no_dp4a) + ")" if no_dp4a else ""))
    for kernel in ("f32", "int16"):
        text = subprocess.run([cuobjdump, "-ptx", os.path.join(gen, kernel + ".fatbin")], capture_output=True, text=True, check=True).stdout
        asked = sorted(set(re.findall(r"([a-z0-9]+(?:\.[a-z0-9]+)*\.ftz(?:\.[a-z0-9]+)*)", text)) - {"ex2.approx.ftz.f32"})
        require(not asked, f"{kernel}: the PTX asks for flushing only in the math library's ex2.approx.ftz" + (" (" + ", ".join(asked[:8]) + ")" if asked else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
