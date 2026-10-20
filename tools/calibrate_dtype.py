"""Calibrate the tiny-fixture error budget of each activation dtype (docs/PRECISION.md).

A dtype class's budget is what the class itself costs against HF in float32: the same HF model, its weights exact, the input of every matrix product rounded to the dtype and the sums in float32, over tiny fixtures at their own weights and at further seeds.
The 16-bit classes take the fixtures of tests/f32.py, tests/moe.py and tests/mxfp4.py; the int8 class takes the 512-token Q8_0 fixtures of tests/int8.py, rounding each matrix product's input at every row per block of 32 to 8-bit integers by the activation twin's rule at 127 levels.
A router keeps float32 inputs by its operation role; weight storage does not select activation precision.
The largest error of each class, times MARGIN, is its frozen budget.
Run once in the HF reference environment of tools/gen_baseline.py, then commit tests/data/dtype_budget.json, before any candidate path is measured against it.

    python tools/calibrate_dtype.py --output FILE
    python tools/calibrate_dtype.py --family q5-1 --output FILE
    python tools/calibrate_dtype.py --family q5-0 --output FILE
"""

import argparse
import json
import math
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tests"))
sys.path.insert(0, HERE)

SEEDS = (0, 1, 2, 3)      # further weights beside each fixture's own
MARGIN = 2.0              # budget over the largest error measured
DTYPES = ("f16", "bf16", "int8")
BUDGET = os.path.join(HERE, "..", "tests", "data", "dtype_budget.json")


def round_int8(x, torch):
    """`x` rounded per block of 32 of its last dimension to 8-bit integers as the activation twin rounds them (xquant.glsl at 127 levels): d = amax / 127, q = x * (127 / amax) rounded half away from zero, the value q * d, all in float32."""
    blocks = x.reshape(-1, 32)
    amax = blocks.abs().amax(dim=1, keepdim=True)
    d = amax / 127.0
    inverse = torch.where(amax > 0, 127.0 / amax, torch.zeros_like(amax))
    r = blocks * inverse
    q = (torch.sign(r) * torch.floor(r.abs() + 0.5)).clamp(-127.0, 127.0)
    return (q * d).reshape(x.shape)


def rounding(model, torch, dtype):
    """Hooks rounding each matrix product's input to `dtype`, routers excepted; returns the handles."""
    if dtype == "int8":
        def to(x):
            return round_int8(x, torch)
    else:
        target = {"f16": torch.float16, "bf16": torch.bfloat16}[dtype]

        def to(x):
            return x.to(target).to(torch.float32)
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and not name.endswith("mlp.gate"):
            handles.append(module.register_forward_pre_hook(lambda _, args: (to(args[0]),) + args[1:]))
    return handles


def routing_hooks(model, trace):
    """Record selected expert sets, ignoring their order, at every reference router call."""
    routers = [(name, module) for name, module in model.named_modules() if name.endswith("mlp.gate")]
    used = getattr(model.config, "num_experts_per_tok", 0)
    if used and not routers:
        raise ValueError("routed reference has no observed router")

    def hook(name):
        def record(_, args, output):
            choices = output.topk(used, dim=-1).indices.sort(dim=-1).values.tolist()
            trace.append({"router": name, "experts": choices})
        return record
    return [module.register_forward_hook(hook(name)) for name, module in routers]


def errors(model, torch, dtype, texts=None, contexts=(4, 16), routing=None):
    """The largest full-vocabulary logit error and windowed NLL error of `dtype` against float32, over reference_outputs' texts."""
    from gen_baseline import reference_outputs
    trace = []
    observers = routing_hooks(model, trace) if routing is not None else []
    try:
        cases, nll = reference_outputs(model, torch, texts, contexts)
        original = list(trace)
        trace.clear()
        handles = rounding(model, torch, dtype)
        try:
            rcases, rnll = reference_outputs(model, torch, texts, contexts)
        finally:
            for h in handles:
                h.remove()
        if routing is not None:
            if len(original) != len(trace):
                raise ValueError("reference router call count changed")
            routing.update(stable=original == trace, calls=len(trace))
            for at, (before, after) in enumerate(zip(original, trace)):
                if before != after:
                    routing["first_change"] = {"call": at, "original": before, "rounded": after}
                    break
    finally:
        for h in observers:
            h.remove()
    logit = max(max(abs(a - b) for a, b in zip(c["logits"], r["logits"])) for c, r in zip(cases, rcases))
    return logit, max(abs(a["mean_nll"] - b["mean_nll"]) for a, b in zip(nll, rnll))


def models():
    """Each tiny fixture family of the 16-bit classes at its own weights and at SEEDS: (name, seed, model, weights)."""
    from gen_baseline import tiny_moe, tiny_mxfp4, tiny_qwen3
    from mxfp4 import VARIANTS
    for tied in (False, True):
        for seed in (12345,) + SEEDS:
            model, weights = tiny_qwen3(tied, seed)
            yield "f32-%s" % ("tied" if tied else "untied"), seed, model, weights
    for seed in (67890,) + SEEDS:
        model, weights = tiny_moe(seed)
        yield "moe", seed, model, weights
    for name, tied, moe in VARIANTS:
        for seed in (12345,) + SEEDS:
            model, _, weights, _ = tiny_mxfp4(tied, moe, seed)
            yield "mxfp4-" + name, seed, model, weights


def int8_models(directory):
    """tests/int8.py's fixtures at their own weights and at SEEDS: (name, seed, model, file SHA-256)."""
    from gen_baseline import tiny_int8
    import int8
    for fixture, own in (("dense-untied", int8.DENSE_SEED), ("dense-tied", int8.DENSE_SEED), ("moe", int8.MOE_SEED)):
        for seed in (own,) + SEEDS:
            model, sha256 = tiny_int8(fixture, seed, os.path.join(directory, "%s-%d.gguf" % (fixture, seed)))
            yield fixture, seed, model, sha256


def family_models(family):
    """Raw-block fixture weights, including split shapes, independent of runtime output."""
    from gen_baseline import tiny_block_model
    from block_fixture import VARIANTS, packed_hash
    from f32 import weight_hash
    import q5_0
    import q5_1
    assert family in ("q5-0", "q5-1"), "unknown calibration family"
    fixture = {"q5-0": q5_0.fixture, "q5-1": q5_1.fixture}[family]
    for split in (False, True):
        for name, tied, moe in VARIANTS:
            for seed in (12345,) + SEEDS:
                config, weights, packed = fixture(tied, moe, seed, split=split)
                model = tiny_block_model(config, weights, tied, moe)
                yield model, {"fixture": name, "split": split, "seed": seed,
                              "config": config, "weights_sha256": weight_hash(weights),
                              "packed_sha256": packed_hash(packed)}


def budgets(runs):
    result = {}
    for dtype, samples in runs.items():
        if not samples or any(not math.isfinite(r[key]) or r[key] < 0
                              for r in samples for key in ("logit_error", "nll_error")):
            raise ValueError("invalid reference errors for " + dtype)
        accepted = [r for r in samples if r.get("status") != "refused"]
        if not accepted:
            raise ValueError("no routing-stable reference errors for " + dtype)
        logit = max(r["logit_error"] for r in accepted)
        nll = max(r["nll_error"] for r in accepted)
        result[dtype] = {"max_logit_error": logit, "max_nll_error": nll,
                         "logit_budget": logit * MARGIN, "nll_budget": nll * MARGIN, "runs": samples}
    return result


def family_budget(family, torch, transformers):
    # Extend only the named family; existing families and the original class
    # budgets remain frozen, including on a reproducibility run.
    with open(BUDGET, encoding="utf-8") as f:
        doc = json.load(f)
    runs = {d: [] for d in DTYPES}
    for model, identity in family_models(family):
        for dtype in DTYPES:
            routing = {}
            logit, nll = errors(model, torch, dtype, routing=routing)
            if not isinstance(routing.get("stable"), bool):
                raise ValueError("missing reference routing witness")
            status = "accepted" if routing["stable"] else "refused"
            runs[dtype].append(dict(identity, logit_error=logit, nll_error=nll, routing=routing, status=status))
            print("%s %s split=%s seed=%d %s %s logit %.8f nll %.8f" %
                  (family, identity["fixture"], identity["split"], identity["seed"], dtype, status, logit, nll), flush=True)
    measured = {"torch_version": torch.__version__, "transformers_version": transformers.__version__,
                "threads": torch.get_num_threads(), "margin": MARGIN, "dtypes": budgets(runs)}
    families = doc.setdefault("families", {})
    if family in families and families[family] != measured:
        raise ValueError("reference differs from frozen family budget: " + family)
    families[family] = measured
    return doc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", required=True, help="new result file; never overwrites a frozen budget")
    parser.add_argument("--family", choices=("q5-0", "q5-1"), help="extend the frozen budget for this raw-block fixture family only")
    args = parser.parse_args(argv)
    import torch
    import transformers
    if torch.__version__ != "2.5.1+cpu" or transformers.__version__ != "4.55.2":
        raise SystemExit("calibration requires torch 2.5.1+cpu and transformers 4.55.2")
    torch.set_num_threads(1)
    if args.family:
        doc = family_budget(args.family, torch, transformers)
        with open(args.output, "x", encoding="utf-8", newline="\n") as f:
            json.dump(doc, f, indent=1)
            f.write("\n")
        print("wrote %s" % args.output)
        return
    runs = {d: [] for d in DTYPES}
    from f32 import weight_hash
    for name, seed, model, weights in models():
        for d in ("f16", "bf16"):
            logit, nll = errors(model, torch, d)
            runs[d].append({"fixture": name, "seed": seed, "weights_sha256": weight_hash(weights), "logit_error": logit, "nll_error": nll})
            print("%-18s seed %-6d %-4s logit %.8f nll %.8f" % (name, seed, d, logit, nll))
    import int8
    with tempfile.TemporaryDirectory(prefix="llmx_int8_") as directory:
        for name, seed, model, sha256 in int8_models(directory):
            logit, nll = errors(model, torch, "int8", int8.TEXTS, int8.CONTEXTS)
            runs["int8"].append({"fixture": "int8-" + name, "seed": seed, "file_sha256": sha256, "logit_error": logit, "nll_error": nll})
            print("%-18s seed %-6d int8 logit %.8f nll %.8f" % ("int8-" + name, seed, logit, nll))
    doc = {"_comment": "Generated by tools/calibrate_dtype.py: each activation class against HF float32 on its tiny fixtures, weights exact, matrix-product inputs rounded, float32 sums.",
           "torch_version": torch.__version__, "transformers_version": transformers.__version__, "threads": torch.get_num_threads(), "margin": MARGIN, "dtypes": {}}
    doc["dtypes"] = budgets(runs)
    with open(args.output, "x", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    print("wrote %s" % args.output)


if __name__ == "__main__":
    main()
