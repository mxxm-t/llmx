"""Hand check: a tensor group's sums against the vendor collective library's all-reduce on the same devices.

Usage: python3 tools/sums_vs_reference.py --test BIN --reference ALL_REDUCE_PERF [--before BIN] [--widths 2,3,4] [--cycles 3]

BIN is llmx-backend-rocm-test or llmx-backend-cuda-test, whose `vsnccl` mode times a joined group's in-place F32 sum per size.
ALL_REDUCE_PERF is rccl-tests' or nccl-tests' all_reduce_perf, run single-process over the same first W visible devices.
The arms alternate in one session, cycle by cycle, and the check fails wherever llmx's median is slower than the reference's at any size or width.
An optional --before BIN, an earlier build's test binary, is timed in the same cycles and printed beside them without being judged.
A run that fails or passes --timeout is reported with its last line and the session goes on; a cell llmx or the reference has no time for fails.
"""

import argparse
import re
import statistics
import subprocess
import sys

# The sizes vsnccl times: rows of the 27B's 5120-wide hidden state, the 65537-float odd row, and a 256-float decode row.
MODEL_SIZES = [5120 * r for r in (1, 2, 4, 6, 8, 13, 32, 128, 512)] + [65537]
SMALL_SIZES = [256]


def run(argv, timeout):
    try:
        p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        out = e.output.decode(errors="replace") if isinstance(e.output, bytes) else (e.output or "")
        return 124, out + "\ntimed out after %d s" % timeout
    return p.returncode, p.stdout


def llmx_arm(test, width, model, timeout):
    rc, out = run([test, "vsnccl", "01", str(width), str(model)], timeout)
    got = {}
    for line in out.splitlines():
        m = re.match(r"ARM llmx order \S+ n (\d+) us ([\d.]+)", line)
        if m:
            got[int(m.group(1))] = float(m.group(2))
    if rc != 0 or not got:
        raise RuntimeError("vsnccl width %d model %d exited %d: %s" % (width, model, rc, (out.strip().splitlines() or ["no output"])[-1]))
    return got


def reference_arm(reference, width, n, timeout):
    # The same call count vsnccl takes per timed repetition, and 15 repetitions whose median is kept.
    iters = 10 if n >= 100000 else 32
    argv = [reference, "-g", str(width), "-b", str(n * 4), "-e", str(n * 4), "-d", "float", "-o", "sum",
            "-n", str(iters), "-w", "5", "-c", "0", "-N", "15"]
    rc, out = run(argv, timeout)
    times = []
    for line in out.splitlines():
        f = line.split()
        # Data rows: size count type redop root, then out-of-place time algbw busbw wrong, then in-place time algbw busbw wrong.
        if len(f) >= 13 and f[0] == str(n * 4) and f[1] == str(n):
            times.append(float(f[9]))
    if rc != 0 or not times:
        raise RuntimeError("all_reduce_perf width %d n %d exited %d: %s" % (width, n, rc, (out.strip().splitlines() or ["no output"])[-1]))
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--test", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--before")
    ap.add_argument("--widths", default="2,3,4")
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=300)
    a = ap.parse_args()
    widths = [int(w) for w in a.widths.split(",")]
    times = {"llmx": {}, "before": {}, "reference": {}}
    tests = {"llmx": a.test, "before": a.before}
    errors = []
    for w in widths:
        for c in range(a.cycles):
            arms = ["llmx", "before", "reference"] if a.before else ["llmx", "reference"]
            for arm in arms if c % 2 == 0 else arms[::-1]:
                try:
                    if arm == "reference":
                        for n in SMALL_SIZES + MODEL_SIZES:
                            times[arm].setdefault((w, n), []).append(reference_arm(a.reference, w, n, a.timeout))
                        continue
                    for model, sizes in ((5120, MODEL_SIZES), (256, SMALL_SIZES)):
                        got = llmx_arm(tests[arm], w, model, a.timeout)
                        for n in sizes:
                            if n not in got:
                                raise RuntimeError("vsnccl width %d printed no time for %d floats" % (w, n))
                            times[arm].setdefault((w, n), []).append(got[n])
                except RuntimeError as e:
                    errors.append("%s width %d cycle %d: %s" % (arm, w, c + 1, e))
                    print("ERROR " + errors[-1], flush=True)
            print("width %d cycle %d of %d done" % (w, c + 1, a.cycles), flush=True)
    slower = missing = 0
    med = lambda arm, w, n: statistics.median(times[arm][(w, n)]) if times[arm].get((w, n)) else None
    show = lambda v: "%.2f" % v if v is not None else "-"
    print("| width | floats |%s llmx us | reference us | llmx / reference |" % (" before us |" if a.before else ""))
    print("|---|---|%s---|---|---|" % ("---|" if a.before else ""))
    for w in widths:
        for n in sorted(SMALL_SIZES + MODEL_SIZES):
            lm, rm = med("llmx", w, n), med("reference", w, n)
            bad = lm is not None and rm is not None and lm > rm
            slower += bad
            missing += lm is None or rm is None
            before = " %s |" % show(med("before", w, n)) if a.before else ""
            ratio = "%.3f%s" % (lm / rm, " SLOWER" if bad else "") if lm is not None and rm is not None else "- NO TIME"
            print("| %d | %d |%s %s | %s | %s |" % (w, n, before, show(lm), show(rm), ratio))
    for e in errors:
        print("ERROR " + e)
    if slower or missing:
        print("FAIL: llmx is slower than the reference at %d and untimed at %d of %d cells" % (slower, missing, len(widths) * (len(SMALL_SIZES) + len(MODEL_SIZES))))
        return 1
    print("PASS: llmx is at least as fast as the reference at every size and width")
    return 0


if __name__ == "__main__":
    sys.exit(main())
