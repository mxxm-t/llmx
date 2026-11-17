"""The drafting set: one model on three fixed prompts, with and without a drafter.

It prints each prompt's prefill and decode rates, drafts fed and kept and whether the greedy ids equal those without drafts (docs/SPECULATIVE.md, section 10).

Usage:
  python tools/drafting_set.py --exe build/llmx --model M.gguf --drafter lookup,embedded
    [--draft-max 3] [--prompts hard,medium,easy] [--path raw] [--repeats 1] [--json OUT]
    [-- extra generate flags such as --device vulkan:0]
"""

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "tests", "data", "drafting")

# name: (file, SHA-256 of its bytes).
PROMPTS = {
    "hard": ("hard.txt", "1e7ae9363da5c59f9fe43a4372f9804b60fe6fe27d46071eac6e934683d6c78c"),
    "medium": ("medium.txt", "6ff3b2e04b925c523bfe205fb035ca1fbd60cc6cb7eca9c7ef4f6c087a4f70b6"),
    "easy": ("easy.txt", "8c17d07b675dfbb60ae4c3534f08fda800a63022921b96e93c4e998576e57173"),
}
MAX_TOKENS = 512


def fixture(name):
    """The prompt file of `name`, after checking its bytes against the recorded SHA-256."""
    file, sha = PROMPTS[name]
    path = os.path.join(DATA, file)
    with open(path, "rb") as f:
        got = hashlib.sha256(f.read()).hexdigest()
    if got != sha:
        sys.exit("%s: SHA-256 %s, the set records %s; the drafting set's prompts never change" % (path, got, sha))
    return path


def counts(line):
    """The kept/fed pairs of a `drafts kept:` line, by draft position."""
    return [tuple(int(v) for v in pair.split("/")) for pair in line.split(":", 1)[1].split()]


def run(args, prompt, path, drafter):
    cmd = [args.exe, "generate", args.model, "--file", prompt, "-n", str(MAX_TOKENS), "--temp", "0", "--verbose"]
    if path == "chat":
        cmd.append("--chat")
    if drafter:
        cmd += ["--drafter", args.drafter, "--draft-max", str(args.draft_max)]
    cmd += args.extra
    p = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        sys.exit("failed (%d): %s\n%s" % (p.returncode, " ".join(cmd), p.stderr[-2000:]))
    r = {"kept": 0, "fed": 0}
    for line in p.stdout.splitlines():
        m = re.match(r"(pp|tg): (\d+) tok, \d+ ms, ([\d.]+) tok/s$", line)
        if m:
            r[m.group(1)] = float(m.group(3))
            r[m.group(1) + "_tokens"] = int(m.group(2))
        elif line.startswith("ids:"):
            r["ids"] = line[4:].strip()
        elif line.startswith("drafts kept:") or line.startswith("own drafts kept:"):
            pairs = counts(line)
            r["kept"] += sum(k for k, _ in pairs)
            r["fed"] += sum(f for _, f in pairs)
    if not all(k in r for k in ("pp", "tg", "ids")):
        sys.exit("no pp, tg or ids line from: " + " ".join(cmd))
    return r


def summary(runs):
    """Medians of an arm's rates, and its drafting counts, which greedy runs repeat exactly."""
    first = runs[0]
    # The first token comes from the prompt's logits; each later pass gives its sampled token and the drafts it kept.
    generated = first["tg_tokens"] - 1
    passes = generated - first["kept"]
    return {
        "pp": statistics.median(r["pp"] for r in runs),
        "tg": statistics.median(r["tg"] for r in runs),
        "fed": first["fed"],
        "kept": first["kept"],
        "acceptance": first["kept"] / first["fed"] if first["fed"] else None,
        "tokens_per_pass": generated / passes if passes else None,
        "prompt_tokens": first["pp_tokens"],
        "generated": first["tg_tokens"],
    }


def table(name, path, off, on, same):
    def pct(a, b):
        return "%+.1f%%" % (100.0 * (b - a) / a)

    def fmt(v, spec):
        return "-" if v is None else spec % v

    print("\n%s (%s, %d prompt tokens, %d generated)" % (name, path, off["prompt_tokens"], off["generated"]))
    print("| arm | prefill tok/s | decode tok/s | drafts fed | kept | acceptance | tokens/pass | ids equal |")
    print("|---|---:|---:|---:|---:|---:|---:|---|")
    print("| off | %.2f | %.2f | - | - | - | 1.00 | - |" % (off["pp"], off["tg"]))
    print("| drafter | %.2f | %.2f | %d | %d | %s | %s | %s |" % (on["pp"], on["tg"], on["fed"], on["kept"], fmt(on["acceptance"], "%.3f"), fmt(on["tokens_per_pass"], "%.2f"), "yes" if same else "NO"))
    print("| change | %s | %s | | | | | |" % (pct(off["pp"], on["pp"]), pct(off["tg"], on["tg"])))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--exe", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--drafter", required=True, help="the --drafter value of the drafting arm, such as lookup,embedded or a drafter file")
    ap.add_argument("--draft-max", type=int, default=3)
    ap.add_argument("--prompts", default="hard,medium,easy", help="comma-separated; final and gate results report all three")
    ap.add_argument("--path", choices=("raw", "chat"), default="chat", help="raw feeds each prompt's text as it is, for a model without a chat template")
    ap.add_argument("--repeats", type=int, default=1, help="pairs of off/on runs a prompt, run off, on, on, off")
    ap.add_argument("--json", help="write every run's figures here")
    ap.add_argument("extra", nargs="*", help="after --, flags both arms pass to generate")
    args = ap.parse_args()
    names = args.prompts.split(",")
    for name in names:
        if name not in PROMPTS:
            ap.error("unknown prompt %r; the set is %s" % (name, ",".join(PROMPTS)))
    print("model %s, drafter %s, draft-max %d, flags %s" % (args.model, args.drafter, args.draft_max, " ".join(args.extra) or "(none)"))
    record = {"model": args.model, "drafter": args.drafter, "draft_max": args.draft_max, "extra": args.extra, "prompts": {}}
    failed = False
    for name in names:
        prompt = fixture(name)
        path = args.path
        arms = {False: [], True: []}
        for i in range(args.repeats):
            for drafter in ((False, True) if i % 2 == 0 else (True, False)):
                arms[drafter].append(run(args, prompt, path, drafter))
        ids = arms[False][0]["ids"]
        same = all(r["ids"] == ids for r in arms[False] + arms[True])
        failed |= not same
        off, on = summary(arms[False]), summary(arms[True])
        table(name, path, off, on, same)
        record["prompts"][name] = {"path": path, "ids_equal": same, "off": off, "drafter": on, "runs": {"off": arms[False], "drafter": arms[True]}}
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=1)
    if failed:
        sys.exit("greedy ids differ with and without drafts")


if __name__ == "__main__":
    main()
