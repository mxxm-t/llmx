import os
import math
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import run as cli

# Perf-regression gate: run the `bench` command and assert the hot paths beat a
# generous floor, so catastrophic slowdowns fail loudly without being flaky.
# This is a smoke gate, not a benchmark harness — the printed numbers are what
# you compare across commits when changing a hot path.

# matmul matvec on 2048x2048 must beat this. Dev machine (Ryzen 7 5800X, 8
# cores, AVX2) hits ~18 GFLOPS; 8.0 catches catastrophic regressions (e.g. the
# ~3.6 GFLOPS per-row-cpuid slowdown) with ~2x margin while staying non-flaky.
FLOOR_GFLOPS = 8.0

# End-to-end TPS on the synthetic Qwen3 model (2 layers, 256 embd). Dev machine
# hits ~4200 (prefill) / ~3700 (decode) tok/s; floors are ~4x below so only
# catastrophic model/runtime regressions fail while staying non-flaky.
FLOOR_PREFILL_TPS = 1000.0
FLOOR_DECODE_TPS = 800.0


def parse(lines):
    out = {}
    for ln in lines:
        p = ln.split()
        if "matmul" in ln:
            out["matmul_ms"] = float(p[3])
            out["matmul_gflops"] = float(p[5])
        elif "prefill" in ln:
            out["prefill_ms"] = float(p[4])
            out["prefill_tps"] = float(p[6])
        elif "decode" in ln:
            out["decode_ms"] = float(p[4])
            out["decode_tps"] = float(p[6])
        elif "rms_norm" in ln:
            out["rms_norm_ms"] = float(p[3])
        elif "rope" in ln:
            out["rope_ms"] = float(p[3])
    return out


def run(enforce_floor=True):
    # These floors were established with one worker; keep the workload fixed.
    rc, out = cli(["bench", "--size", "2048", "--iters", "5", "--threads", "1"])
    assert rc == 0, "bench command failed"
    res = parse(out.splitlines())
    assert "matmul_gflops" in res, "bench output missing matmul line:\n" + out
    assert "prefill_tps" in res and "decode_tps" in res, \
        "bench output missing prefill/decode lines:\n" + out
    assert all(math.isfinite(v) and v >= 0 for v in res.values()), "invalid benchmark output: " + out
    assert all(res[k] > 0 for k in ("matmul_gflops", "prefill_tps", "decode_tps")), "zero throughput: " + out
    assert not enforce_floor or res["matmul_gflops"] >= FLOOR_GFLOPS, (
        "matmul %.2f GFLOPS below floor %.2f" % (res["matmul_gflops"], FLOOR_GFLOPS))
    assert not enforce_floor or res["prefill_tps"] >= FLOOR_PREFILL_TPS, (
        "prefill %.1f tok/s below floor %.1f" % (res["prefill_tps"], FLOOR_PREFILL_TPS))
    assert not enforce_floor or res["decode_tps"] >= FLOOR_DECODE_TPS, (
        "decode %.1f tok/s below floor %.1f" % (res["decode_tps"], FLOOR_DECODE_TPS))
    print("perf: matmul %.2f GFLOPS (%.3f ms), prefill %.0f tok/s, decode %.0f tok/s, "
          "rms_norm %.3f ms, rope %.3f ms  [ok]"
          % (res["matmul_gflops"], res["matmul_ms"], res["prefill_tps"], res["decode_tps"],
             res["rms_norm_ms"], res["rope_ms"]))
    if not enforce_floor:
        print("perf: hardware-specific floors disabled; timings are diagnostic only")
    return True


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
