"""Paired A/B runner for the docs/DEVICE-EXECUTION.md migration steps.

The bar for each step is no measured regression, so the runner is built to
make a negative result stick: the plan, including the advance rule and the
contamination criteria, is written and hashed BEFORE any timing, and the
runner refuses to start if a plan already exists. Samples are never dropped;
activity flags are diagnostic only.

Arms alternate within each round and the order reverses on odd rounds, so a
drift in machine load falls on both arms equally. Every arm is monitored by
tools/monitor_windows.py, which is what makes the activity annotation
comparable across arms.

Usage, in two phases so the rule cannot be written after seeing data:

    python tools/ab_runner.py plan --out DIR --base A.exe --cand B.exe \\
        --model M.gguf --prompt P.txt
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

# Frozen before any timing. A step advances only if neither phase regresses by
# more than NOISE, and the baseline does not win a majority of pairs.
ADVANCE = {
    "noise_fraction": 0.01,
    "rule": "For each phase: candidate mean and median are at least "
            "(1 - noise_fraction) x baseline, AND baseline paired wins are not "
            "more than half the pairs. Otherwise the step does not advance.",
    "contamination": "A pair is flagged when either arm's whole-run CPU exceeds "
                     "the session median by 25%. Flags are reported, never used "
                     "to drop, replace or re-run a pair.",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cmd_plan(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / "plan.json"
    if plan_path.exists():
        sys.exit("plan.json exists; refusing to rewrite a frozen plan")
    arms = {"base": args.base, "cand": args.cand}
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
        "order": "arms alternate within a round; order reverses on odd rounds",
        "advance": ADVANCE,
        "limits": "One model and one prompt. Host load is recorded, not "
                  "controlled. This runner covers only the regression half of "
                  "the bar, baseline against candidate. The external "
                  "mx-llama.cpp floor is a separate matched comparison and is "
                  "NOT run here; a step that passes this runner still has to "
                  "clear that floor.",
    }
    text = json.dumps(plan, indent=2, sort_keys=True)
    plan_path.write_text(text, encoding="ascii")
    (out / "plan.sha256").write_text(
        hashlib.sha256(text.encode("ascii")).hexdigest(), encoding="ascii")
    print(f"frozen plan: {plan_path}")


def invoke(exe, model, prompt_text, threads, max_tokens):
    proc = subprocess.run(
        [str(exe), "generate", str(model), prompt_text, "-n", str(max_tokens),
         "--temp", "0", "--seed", "1", "--threads", str(threads), "--verbose"],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{exe} exited {proc.returncode}: {proc.stderr[-400:]}")
    rates = dict((m.group(1), float(m.group(2)))
                 for m in RATE.finditer(proc.stdout + proc.stderr))
    if "pp" not in rates or "tg" not in rates:
        raise RuntimeError(f"{exe} produced no rate lines")
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
        mon.wait(timeout=30)
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
    order = ["base", "cand"]
    samples = []
    total = plan["warmup_rounds"] + plan["measured_rounds"]
    for rnd in range(total):
        measured = rnd >= plan["warmup_rounds"]
        seq = order if rnd % 2 == 0 else order[::-1]
        for arm in seq:
            tag = f"r{rnd}-{arm}"
            rates, log = monitored(out, tag, lambda: invoke(
                plan["arms"][arm]["exe"], plan["model"], prompt_text,
                plan["threads"], plan["max_tokens"]))
            samples.append({"round": rnd, "arm": arm, "measured": measured,
                            "pp": rates["pp"], "tg": rates["tg"],
                            "monitor": log.name})
            print(f"{tag}: pp {rates['pp']:.2f} tg {rates['tg']:.2f}")
    (out / "samples.json").write_text(
        json.dumps(samples, indent=2), encoding="ascii")
    print(f"wrote {out / 'samples.json'} ({len(samples)} samples, none dropped)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("plan")
    f.add_argument("--out", required=True)
    f.add_argument("--base", required=True)
    f.add_argument("--cand", required=True)
    f.add_argument("--model", required=True)
    f.add_argument("--prompt", required=True)
    f.add_argument("--threads", type=int, default=6)
    f.add_argument("--max-tokens", type=int, default=32)
    f.add_argument("--rounds", type=int, default=9)
    f.set_defaults(func=cmd_plan)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.set_defaults(func=cmd_run)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
