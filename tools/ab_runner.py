"""Paired A/B runner for the docs/DEVICE-EXECUTION.md migration steps and
the docs/KV-CACHE.md block-size screening.

The bar for each step is no measured regression, so the runner is built to
make a negative result stick: the plan, including the advance rule and the
contamination criteria, is written and hashed BEFORE any timing, and the
runner refuses to start if a plan already exists. Samples are never dropped;
activity flags are diagnostic only.

Arms rotate within each round and the rotation reverses on odd rounds, so a
drift in machine load falls on every arm equally and no arm always runs
first. Every arm is monitored by tools/monitor_windows.py, which is what
makes the activity annotation comparable across arms.

One base and one or more candidates; every candidate is scored against the
base by paired ratios within a round. Usage, in two phases so the rule cannot
be written after seeing data:

    python tools/ab_runner.py plan --out DIR --base A.exe --cand B.exe \\
        [--cand NAME=C.exe ...] --model M.gguf --prompt P.txt
    python tools/ab_runner.py run --out DIR
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
RATE = re.compile(r"^(pp|tg): .*?, ([0-9.]+) tok/s", re.M)
KV = re.compile(r"^kv: allocated ([0-9]+) bytes, peak ([0-9]+) bytes, used ([0-9]+) bytes", re.M)

# Frozen before any timing. A step advances unless a phase shows a real
# regression: mean or median below the noise band, or a baseline win count that
# is itself unlikely by chance.
#
# The win count needs a significance threshold, not a majority. Under no real
# difference the count is Binomial(pairs, 0.5), so "baseline wins no more than
# half" rejects a genuinely neutral change about half the time -- with 9 pairs,
# P(wins <= 4) is exactly 256/512. A criterion that fails a coin flip is not a
# gate. The threshold below is the smallest count whose one-sided tail is at
# most 5%, so a neutral change passes about 95% of the time and a consistent
# regression still fails.
#
# noise_fraction is CALIBRATED BY AN A/A RUN, not guessed. Publishing the same
# binary as both arms on this harness moved prefill +1.59% and decode -1.62% by
# median over 15 pairs, with zero code difference. A 1% band therefore failed
# its own A/A, which means it was measuring the harness rather than the change.
# 3% leaves roughly a factor of two over the observed A/A spread. Re-run the
# A/A on new hardware or a new workload before trusting this number there: if
# the A/A does not pass, the band is wrong and no result from it means anything.
ADVANCE = {
    "noise_fraction": 0.03,
    "alpha": 0.05,
    "rule": "For each phase, over the PAIRED per-round ratios cand/base: both "
            "the mean and the median ratio are at least (1 - noise_fraction), "
            "AND baseline paired wins are below the smallest count whose "
            "one-sided binomial tail under p=0.5 is at most alpha. Ratios are "
            "paired so a drift within the run cancels inside each pair; "
            "comparing marginal medians instead let a run that drifted "
            "289 -> 222 tok/s report -11.15% on identical code.",
    "contamination": "A pair is flagged when either arm's whole-run CPU exceeds "
                     "the session median by 25%. Flags are reported, never used "
                     "to drop, replace or re-run a pair.",
}


def win_threshold(pairs, alpha):
    """Smallest baseline-win count whose one-sided tail under p=0.5 is <= alpha."""
    from math import comb
    total = 2 ** pairs
    for k in range(pairs, -1, -1):
        tail = sum(comb(pairs, i) for i in range(k, pairs + 1))
        if tail / total > alpha:
            return k + 1
    return 0


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cmd_plan(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / "plan.json"
    if plan_path.exists():
        sys.exit("plan.json exists; refusing to rewrite a frozen plan")
    # Absolute native paths: CreateProcess does not accept a relative
    # forward-slash command token, and a frozen plan must not depend on cwd.
    arms = {"base": str(Path(args.base).resolve())}
    for i, c in enumerate(args.cand):
        name, _, exe = c.rpartition("=")
        arms[name or ("cand" if len(args.cand) == 1 else f"cand{i}")] = str(Path(exe).resolve())
    args.model = Path(args.model).resolve()
    args.prompt = Path(args.prompt).resolve()
    plan = {
        "kind": "device-execution A/B",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": str(args.model),
        "model_sha256": sha256(args.model),
        "prompt": str(args.prompt),
        "prompt_sha256": sha256(args.prompt),
        "arms": {k: {"exe": str(v), "sha256": sha256(v)} for k, v in arms.items()},
        "threads": args.threads,
        "max_tokens": args.max_tokens,
        "warmup_rounds": 1,
        "measured_rounds": args.rounds,
        "order": "arms rotate by one place per round; the rotation reverses on odd rounds",
        "advance": ADVANCE,
        "limits": "One model and one prompt. Host load is recorded, not "
                  "controlled. This runner covers only the regression half of "
                  "the bar, baseline against candidate. The external "
                  "reference floor is a separate matched comparison and is "
                  "NOT run here; a step that passes this runner still has to "
                  "clear that floor.",
    }
    text = json.dumps(plan, indent=2, sort_keys=True)
    plan_path.write_text(text, encoding="ascii")
    (out / "plan.sha256").write_text(
        hashlib.sha256(text.encode("ascii")).hexdigest(), encoding="ascii")
    print(f"frozen plan: {plan_path}")


def invoke(exe, model, prompt_text, threads, max_tokens):
    # A hung arm used to hang the whole run, and the generated text is decoded
    # with the ANSI codepage unless this says otherwise, which raises on any
    # non-ASCII token.
    proc = subprocess.run(
        [str(exe), "generate", str(model), prompt_text, "-n", str(max_tokens),
         "--temp", "0", "--seed", "1", "--threads", str(threads), "--verbose"],
        capture_output=True, encoding="utf-8", errors="replace", timeout=1800)
    if proc.returncode != 0:
        raise RuntimeError(f"{exe} exited {proc.returncode}: {proc.stderr[-400:]}")
    text = proc.stdout + proc.stderr
    rates = dict((m.group(1), float(m.group(2))) for m in RATE.finditer(text))
    if "pp" not in rates or "tg" not in rates:
        raise RuntimeError(f"{exe} produced no rate lines")
    # Binaries before the paged cache print no kv line; record what is there.
    kv = KV.search(text)
    if kv:
        rates["kv_allocated"] = int(kv.group(1))
        rates["kv_peak"] = int(kv.group(2))
        rates["kv_used"] = int(kv.group(3))
    return rates


def monitored(out, tag, fn):
    """Run fn under monitor_windows.py, returning (result, monitor_path)."""
    log = out / f"monitor-{tag}.jsonl"
    stop = out / f"stop-{tag}"
    mon = subprocess.Popen(
        [sys.executable, str(TOOLS / "monitor_windows.py"), "--output", str(log),
         "--seconds", "600", "--interval", "1", "--stop-file", str(stop)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        return fn(), log
    finally:
        stop.touch()
        try:
            mon.wait(timeout=30)
        except subprocess.TimeoutExpired:
            # Leaving it running would put an unaccounted process alongside
            # every later arm, which is exactly what it is here to detect.
            mon.kill()
            mon.wait()
        stop.unlink(missing_ok=True)


def cmd_run(args):
    out = Path(args.out)
    plan = json.loads((out / "plan.json").read_text(encoding="ascii"))
    frozen = (out / "plan.sha256").read_text(encoding="ascii").strip()
    live = hashlib.sha256(json.dumps(plan, indent=2, sort_keys=True)
                          .encode("ascii")).hexdigest()
    if frozen != live:
        sys.exit("plan.json changed after freezing; refusing to run")
    for name, arm in plan["arms"].items():
        if sha256(arm["exe"]) != arm["sha256"]:
            sys.exit(f"arm {name} binary changed since the plan was frozen")

    prompt_text = Path(plan["prompt"]).read_text(encoding="utf-8")
    order = list(plan["arms"])
    samples = []
    total = plan["warmup_rounds"] + plan["measured_rounds"]
    for rnd in range(total):
        measured = rnd >= plan["warmup_rounds"]
        k = rnd % len(order)
        seq = order[k:] + order[:k]
        if rnd % 2:
            seq = seq[::-1]
        for arm in seq:
            tag = f"r{rnd}-{arm}"
            rates, log = monitored(out, tag, lambda: invoke(
                plan["arms"][arm]["exe"], plan["model"], prompt_text,
                plan["threads"], plan["max_tokens"]))
            samples.append({"round": rnd, "arm": arm, "measured": measured,
                            "pp": rates["pp"], "tg": rates["tg"],
                            "kv_allocated": rates.get("kv_allocated"),
                            "kv_peak": rates.get("kv_peak"),
                            "kv_used": rates.get("kv_used"),
                            "monitor": log.name})
            print(f"{tag}: pp {rates['pp']:.2f} tg {rates['tg']:.2f}")
    (out / "samples.json").write_text(
        json.dumps(samples, indent=2), encoding="ascii")
    print(f"wrote {out / 'samples.json'} ({len(samples)} samples, none dropped)")


def cmd_report(args):
    """Apply the frozen rule to collected samples. The verdict is computed
    here, from the plan's own criteria, so it is reproducible rather than
    recomputed by hand each time."""
    import statistics as st
    out = Path(args.out)
    plan = json.loads((out / "plan.json").read_text(encoding="ascii"))
    samples = json.loads((out / "samples.json").read_text(encoding="ascii"))
    adv = plan["advance"]
    # A run is scored by the rule frozen with it, never by a newer one. A plan
    # missing a criterion predates that criterion, and rescoring it under the
    # current rule would be choosing the test after seeing the data.
    missing = [k for k in ("noise_fraction", "alpha") if k not in adv]
    if missing:
        sys.exit(f"plan predates {', '.join(missing)}; it must be scored by the "
                 f"rule frozen with it. Re-measure under a new plan instead.")
    measured = [s for s in samples if s["measured"]]
    rounds = sorted({s["round"] for s in measured})
    pick = lambda r, a, ph: next(s[ph] for s in measured
                                 if s["round"] == r and s["arm"] == a)
    limit = win_threshold(len(rounds), adv["alpha"])
    cands = [a for a in plan["arms"] if a != "base"]
    verdict, report = True, {}
    for cand in cands:
      phases = {}
      kv = next((s for s in reversed(measured) if s["arm"] == cand and s.get("kv_used")), None)
      if kv:
          phases["kv_bytes"] = {"allocated": kv["kv_allocated"], "peak": kv["kv_peak"],
                                "used": kv["kv_used"]}
      for ph, name in (("pp", "prefill"), ("tg", "decode")):
        b = [pick(r, "base", ph) for r in rounds]
        c = [pick(r, cand, ph) for r in rounds]
        wins = sum(1 for i in range(len(b)) if b[i] > c[i])
        keep = 1 - adv["noise_fraction"]
        # Paired ratios, not a ratio of marginals. The arms alternate so that
        # drift cancels WITHIN a pair; comparing median(cand) to median(base)
        # discards that pairing and lets a monotonic drift dominate. A run
        # whose throughput fell 289 -> 222 across 15 rounds reported a -11.15%
        # unpaired prefill median on identical code, against +1.24% paired.
        ratios = sorted(c[i] / b[i] for i in range(len(b)))
        mean_ratio = sum(ratios) / len(ratios)
        median_ratio = st.median(ratios)
        ok = (mean_ratio >= keep and median_ratio >= keep and wins < limit)
        verdict &= ok
        phases[name] = {
            "base_mean": st.mean(b), "cand_mean": st.mean(c),
            "base_median": st.median(b), "cand_median": st.median(c),
            "paired_mean_change_percent": (mean_ratio - 1) * 100,
            "paired_median_change_percent": (median_ratio - 1) * 100,
            "unpaired_mean_change_percent": (st.mean(c) / st.mean(b) - 1) * 100,
            "baseline_wins": wins, "pairs": len(b), "fail_at_wins": limit,
            "paired_change_percent": [(c[i] / b[i] - 1) * 100
                                      for i in range(len(b))],
            "advance": "pass" if ok else "fail"}
        print(f"{cand} {name}: paired mean {phases[name]['paired_mean_change_percent']:+.2f}% "
              f"median {phases[name]['paired_median_change_percent']:+.2f}% "
              f"baseline wins {wins}/{len(b)} (fail at >= {limit}) "
              f"-> {'PASS' if ok else 'FAIL'}")
      report[cand] = phases
    print("ADVANCE:", "pass" if verdict else "fail")
    (out / "report.json").write_text(json.dumps(
        {"advance": "pass" if verdict else "fail", "candidates": report,
         "plan_sha256": (out / "plan.sha256").read_text(encoding="ascii").strip()},
        indent=2), encoding="ascii")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("plan")
    f.add_argument("--out", required=True)
    f.add_argument("--base", required=True)
    f.add_argument("--cand", required=True, action="append",
                   help="candidate exe, or NAME=exe; repeatable")
    f.add_argument("--model", required=True)
    f.add_argument("--prompt", required=True)
    f.add_argument("--threads", type=int, default=6)
    f.add_argument("--max-tokens", type=int, default=32)
    f.add_argument("--rounds", type=int, default=9)
    f.set_defaults(func=cmd_plan)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.set_defaults(func=cmd_run)
    rep = sub.add_parser("report")
    rep.add_argument("--out", required=True)
    rep.set_defaults(func=cmd_report)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
