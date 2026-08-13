import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import run as cli

# Perf-regression gate: run the `bench` command and assert the hot paths beat a
# generous floor, so catastrophic slowdowns fail loudly without being flaky.
# This is a smoke gate, not a benchmark harness — the printed numbers are what
# you compare across commits when changing a hot path.

FLOOR_GFLOPS = 0.5   # matmul matvec on 2048x2048 must beat this (very generous)


def parse(lines):
    out = {}
    for ln in lines:
        if "matmul" in ln:
            out["matmul_ms"] = float(ln.split()[3])
            out["matmul_gflops"] = float(ln.split()[5])
        elif "rms_norm" in ln:
            out["rms_norm_ms"] = float(ln.split()[3])
        elif "rope" in ln:
            out["rope_ms"] = float(ln.split()[3])
    return out


def run():
    rc, out = cli(["bench", "--size", "2048", "--iters", "5"])
    assert rc == 0, "bench command failed"
    res = parse(out.splitlines())
    assert "matmul_gflops" in res, "bench output missing matmul line:\n" + out
    assert res["matmul_gflops"] >= FLOOR_GFLOPS, (
        "matmul %.2f GFLOPS below floor %.2f" % (res["matmul_gflops"], FLOOR_GFLOPS))
    print("perf: matmul %.2f GFLOPS (%.3f ms), rms_norm %.3f ms, rope %.3f ms  [ok]"
          % (res["matmul_gflops"], res["matmul_ms"], res["rms_norm_ms"], res["rope_ms"]))
    return True


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
