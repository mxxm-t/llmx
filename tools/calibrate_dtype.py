"""Calibrate the tiny-fixture error budget of each activation dtype (docs/PRECISION.md).

A dtype class's budget is what the class itself costs against HF in float32: the same HF model, its weights exact, the input of every matrix product rounded to the dtype and the sums in float32, over tiny fixtures at their own weights and at further seeds.
The 16-bit classes take the fixtures of tests/f32.py, tests/moe.py and tests/mxfp4.py; the int8 class takes the 512-token Q8_0 fixtures of tests/int8.py, rounding each matrix product's input at every row per block of 32 to 8-bit integers by the activation twin's rule at 127 levels.
A router keeps float32 inputs by its operation role; weight storage does not select activation precision.
The largest error of each class, times MARGIN, is its frozen budget.
Run once in the HF reference environment of tools/gen_baseline.py, then commit tests/data/dtype_budget.json, before any candidate path is measured against it.

    python tools/calibrate_dtype.py --output FILE
    python tools/calibrate_dtype.py --family q5-1 --output FILE
    python tools/calibrate_dtype.py --family q5-0 --output FILE
    python tools/calibrate_dtype.py --family q1-0 --output FILE
    python tools/calibrate_dtype.py --family q2-0 --output FILE
    python tools/calibrate_dtype.py --family q3-k --output FILE
    python tools/calibrate_dtype.py --family iq4-xs --output FILE
    python tools/calibrate_dtype.py --family iq3-s --output FILE
    python tools/calibrate_dtype.py --family iq2-s --output FILE
    python tools/calibrate_dtype.py --family iq2-xxs --output FILE
    python tools/calibrate_dtype.py --family iq2-xs --output FILE
    python tools/calibrate_dtype.py --family iq3-xxs --output FILE
    python tools/calibrate_dtype.py --family iq1-s --output FILE
    python tools/calibrate_dtype.py --family iq1-m --output FILE
    python tools/calibrate_dtype.py --class f16-mma --output FILE
    python tools/calibrate_dtype.py --class bf16-mma --output FILE

The f16-mma class is the F16 class on BF16 weights with each matrix sum taken as the CUDA TF32 tile takes it (int16.gpu, bf16_tile): groups of 8 in k order, each group's exact products aligned to its largest, cut G bits below that product's 24, the cut sum rounded to F32, and the group added to the F32 sum; on sm_86 a group is two such sums of 4, the first's result a term of the second (PROBED).
Its fixtures are the 16-bit classes' with every matrix rounded to BF16 in the reference and the candidate alike, then tests/bf16_tile.py's, whose every width the tile takes, at its own weights and at SEEDS; its error is the largest over the sum models given (by default ENVELOPE), and every Linear but the router takes the model, a superset of the tile's reach.
The F16 class must reproduce first on the 16-bit fixtures, and on the tile's fixture, which the F16 record does not hold, plain F16 on its BF16 weights must stay below the F16 budget.
The bf16-mma class is the same over BF16 inputs and groups of 16, as the BF16 tile takes them at --dtype bf16 (bf16_tile<true>), the frozen BF16 class reproducing first and BF16_MODELS its default sum models.

Q1_0 and Q2_0 also measure prospective F16 costs; each existing BF16 entry
must reproduce unchanged before its missing F16 entry can be added.
This reference tool does not qualify a runtime path.
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
MMA = "f16-mma"
BMMA = "bf16-mma"
# Each tensor-core class: the activations' type, the group of k one instruction sums, and the class whose frozen runs it reproduces first.
CLASSES = {MMA: ("f16", 8), BMMA: ("bf16", 16)}
# Sum models G:align:final:reference[:kN] of the TF32 group sum: the one the sm_86 probe fits bit for bit, then the single-sum envelope kept for unprobed architectures (docs/PRECISION.md).
PROBED = ("1:rz:rz:inputs:k4",)
ENVELOPE = tuple("%d:%s:%s:norm" % (g, a, f) for g in (0, 2, 3) for a in ("rz", "floor") for f in ("rz", "rne"))
# The BF16 group sum has no probe yet: single sums of 16 and chains of 4 and 8 at the probed TF32 alignment, with the single-sum envelope.
BF16_MODELS = ("1:rz:rz:inputs", "1:rz:rz:inputs:k4", "1:rz:rz:inputs:k8") + ENVELOPE
BUDGET = os.path.join(HERE, "..", "tests", "data", "dtype_budget.json")
CUTS = {}                 # sum model -> groups whose alignment cut changed a product, so a record shows the cut was reached


def round_int8(x, torch):
    """`x` rounded per block of 32 of its last dimension to 8-bit integers as the activation twin rounds them (xquant.glsl at 127 levels): d = amax / 127, q = x * (127 / amax) rounded half away from zero, the value q * d, all in float32."""
    blocks = x.reshape(-1, 32)
    amax = blocks.abs().amax(dim=1, keepdim=True)
    d = amax / 127.0
    inverse = torch.where(amax > 0, 127.0 / amax, torch.zeros_like(amax))
    r = blocks * inverse
    q = (torch.sign(r) * torch.floor(r.abs() + 0.5)).clamp(-127.0, 127.0)
    return (q * d).reshape(x.shape)


def mma_linear(x, w, bias, torch, spec, cls=MMA):
    """`x` rounded to the class's input type times `w`, each group of k (8 for f16-mma, 16 for bf16-mma) summed by sum model `spec` (G:align:final:reference[:kN]) from zero and added to the F32 sum in k order.

    With kN a group is chained sums of N, as the probe found m16n8k8 on sm_86 at N = 4: each sum's F32 result joins the next N products as one more term, aligned by its own leading exponent.
    """
    g, align, final, reference, *halves = spec.split(":")
    assert reference in ("norm", "inputs"), "unknown reference " + reference
    assert final in ("rz", "rne"), "unknown final rounding " + final
    form, size = CLASSES[cls]
    chunk = int(halves[0][1:]) if halves else size
    assert not halves or (len(halves) == 1 and halves[0][0] == "k" and chunk < size and size % chunk == 0), "unknown group shape " + ":".join(halves)
    shape, n = x.shape[:-1], x.shape[-1]
    pad = -n % size   # zero products, which change no group's alignment, fill a last group of a width the tile would not take
    x = torch.nn.functional.pad(x.to({"f16": torch.float16, "bf16": torch.bfloat16}[form]).to(torch.float64).reshape(-1, n), (0, pad))
    w = torch.nn.functional.pad(w.to(torch.float64), (0, pad))
    p = (x[:, None, :] * w[None, :, :]).reshape(x.shape[0], w.shape[0], (n + pad) // size, size)   # binary16 or BF16 times BF16 products are exact in float64
    if reference == "norm":
        lead = torch.frexp(p.abs())[1] - 1
    else:   # the sum of the inputs' exponents, one below a product of significands of 2 or more
        ex, ew = torch.frexp(x.abs())[1] - 1, torch.frexp(w.abs())[1] - 1
        lead = (ex[:, None, :] + ew[None, :, :]).reshape(p.shape)
    lead = lead.masked_fill(p == 0, -100000)

    def aligned_sum(terms, leads):
        top = leads.amax(-1, keepdim=True)
        top = top.masked_fill(top == -100000, 0)   # an all-zero group sums to zero at any unit
        unit = torch.ldexp(torch.ones_like(terms[..., :1]), top - 23 - int(g))
        assert unit.min() > 0, "alignment unit underflowed"
        cut = {"rz": torch.trunc, "floor": torch.floor}[align](terms / unit)
        CUTS[spec] = CUTS.get(spec, 0) + int((cut * unit != terms).any(-1).sum())
        v = cut.sum(-1) * unit[..., 0]   # exact for G below about 25, where each cut term stays below 2^27
        if final == "rz":
            m, e = torch.frexp(v)
            v = torch.ldexp(torch.trunc(torch.ldexp(m, torch.full_like(e, 24))), e - 24)
        return v.to(torch.float32).to(torch.float64)

    groups = aligned_sum(p[..., :chunk], lead[..., :chunk])
    for at in range(chunk, size, chunk):
        prior = (torch.frexp(groups)[1] - 1).masked_fill(groups == 0, -100000)
        groups = aligned_sum(torch.cat([p[..., at:at + chunk], groups[..., None]], -1), torch.cat([lead[..., at:at + chunk], prior[..., None]], -1))
    groups = groups.to(torch.float32)
    acc = torch.zeros(groups.shape[:2], dtype=torch.float32)
    for k in range(groups.shape[2]):
        acc = acc + groups[..., k]
    if bias is not None:
        acc = acc + bias.to(torch.float32)
    assert torch.isfinite(acc).all() or not torch.isfinite(x).all(), "non-finite tensor-core group sum"
    return acc.reshape(*shape, w.shape[0])


def bf16_weights(model, torch):
    """Every matrix of `model` rounded to BF16 in place, as a BF16 file holds it; vectors such as norms stay float32."""
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() >= 2:
                p.copy_(p.to(torch.bfloat16).to(torch.float32))


def rounding(model, torch, dtype):
    """Hooks rounding each matrix product's input to `dtype`, routers excepted; returns the handles."""
    cls, _, spec = dtype.partition(":")
    if cls in CLASSES:
        return [module.register_forward_hook(lambda m, args, out: mma_linear(args[0], m.weight, m.bias, torch, spec, cls))
                for name, module in model.named_modules()
                if isinstance(module, torch.nn.Linear) and not name.endswith("mlp.gate")]
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
    from functools import partial
    from spec_decode import IQ1_S, IQ1_M
    import tq
    import nvfp4
    import iq1
    import q5_0
    import q5_1
    import q3_k
    import iq4_xs
    import iq3_s
    import iq2_s
    import iq3_xxs
    import iq2_small
    import low_bits
    assert family in ("q5-0", "q5-1", "q3-k", "iq4-xs", "iq3-s", "iq2-s", "iq2-xxs", "iq2-xs", "iq3-xxs", "iq1-s", "iq1-m", "tq1-0", "tq2-0", "nvfp4", "q1-0", "q2-0"), "unknown calibration family"
    fixture = {"q5-0": q5_0.fixture, "q5-1": q5_1.fixture, "q3-k": q3_k.fixture, "iq4-xs": iq4_xs.fixture,
               "iq3-s": iq3_s.fixture, "iq2-s": iq2_s.fixture, "iq3-xxs": iq3_xxs.fixture,
               "iq2-xxs": iq2_small.fixture_xxs, "iq2-xs": iq2_small.fixture_xs,
               "iq1-s": partial(iq1.fixture, IQ1_S), "iq1-m": partial(iq1.fixture, IQ1_M),
               "tq1-0": tq.fixture_tq1, "tq2-0": tq.fixture_tq2, "nvfp4": nvfp4.fixture,
               "q1-0": low_bits.fixture_q1, "q2-0": low_bits.fixture_q2}[family]
    own_seed = (tq.TQ2_SEED if family == "tq2-0" else nvfp4.NVFP4_SEED if family == "nvfp4" else
                12346 if family == "q2-0" else 12345)
    for split in (False, True):
        for name, tied, moe in VARIANTS:
            for seed in (own_seed,) + SEEDS:
                config, weights, packed = fixture(tied, moe, seed, split=split)
                model = tiny_block_model(config, weights, tied, moe)
                identity = {"fixture": name, "split": split, "seed": seed,
                            "config": config, "weights_sha256": weight_hash(weights),
                            "packed_sha256": packed_hash(packed)}
                del weights, packed
                yield model, identity
                del model


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
    # Extend only the named family; existing families and class budgets stay frozen on reproducibility runs.
    with open(BUDGET, encoding="utf-8") as f:
        doc = json.load(f)
    # Q3_K uses the generic F16 budget; Q1/Q2 measure F16 prospectively before candidate execution.
    dtypes = (("bf16",) if family == "q3-k" else
              ("f16", "bf16") if family in ("tq1-0", "tq2-0", "nvfp4", "q1-0", "q2-0") else DTYPES)
    runs = {d: [] for d in dtypes}
    for model, identity in family_models(family):
        for dtype in dtypes:
            routing = {}
            logit, nll = errors(model, torch, dtype, routing=routing)
            if not isinstance(routing.get("stable"), bool):
                raise ValueError("missing reference routing witness")
            status = "accepted" if routing["stable"] else "refused"
            runs[dtype].append(dict(identity, logit_error=logit, nll_error=nll, routing=routing, status=status))
            print("%s %s split=%s seed=%d %s %s logit %.8f nll %.8f" %
                  (family, identity["fixture"], identity["split"], identity["seed"], dtype, status, logit, nll), flush=True)
        del model
    measured = {"torch_version": torch.__version__, "transformers_version": transformers.__version__,
                "threads": torch.get_num_threads(), "margin": MARGIN, "dtypes": budgets(runs)}
    families = doc.setdefault("families", {})
    if family in families and families[family] != measured:
        prior = families[family]
        bf16_only = dict(measured, dtypes={"bf16": measured["dtypes"]["bf16"]})
        if family not in ("q1-0", "q2-0") or set(prior.get("dtypes", {})) != {"bf16"} or prior != bf16_only:
            raise ValueError("reference differs from frozen family budget: " + family)
    families[family] = measured
    return doc


def mma_budget(torch, transformers, specs, cls=MMA):
    """A tensor-core class over the 16-bit fixtures on BF16 weights, after its frozen input class (F16 or BF16) reproduces on each; refused at or above that class's budget, which its runs then take."""
    from f32 import weight_hash
    with open(BUDGET, encoding="utf-8") as f:
        doc = json.load(f)
    import bf16_tile
    from gen_baseline import tiny_bf16_tile
    form = CLASSES[cls][0]
    frozen = {(r["fixture"], r["seed"]): r for r in doc["dtypes"][form]["runs"]}
    f16 = doc["dtypes"][form]
    tile = ((("bf16-tile-%s" % ("tied" if tied else "untied"), seed) + tiny_bf16_tile(tied, seed)) for tied in (False, True) for seed in (12345,) + SEEDS)
    runs, model_max = [], {}
    for name, seed, model, weights in list(models()) + list(tile):
        texts = (bf16_tile.TEXTS, bf16_tile.CONTEXTS) if name.startswith("bf16-tile") else (None, (4, 16))
        digest = weight_hash(weights)
        if name.startswith("bf16-tile"):
            plain = errors(model, torch, form, *texts)
            if plain[0] >= f16["logit_budget"] or plain[1] >= f16["nll_budget"]:
                raise ValueError("plain %s on %s seed %d costs more than its frozen budget" % (form, name, seed))
        else:
            prior = frozen[(name, seed)]
            logit, nll = errors(model, torch, form)
            if (digest, logit, nll) != (prior["weights_sha256"], prior["logit_error"], prior["nll_error"]):
                raise ValueError("frozen %s class does not reproduce on %s seed %d" % (form, name, seed))
            bf16_weights(model, torch)
            plain = errors(model, torch, form)
        check = errors(model, torch, cls + ":40:rz:rne:norm", *texts)   # near-exact group sums: the plain input class up to F32 sum order
        per = []
        for spec in specs:
            before = CUTS.get(spec, 0)
            l, n = errors(model, torch, cls + ":" + spec, *texts)
            per.append({"model": spec, "logit_error": l, "nll_error": n, "cut_groups": CUTS[spec] - before})
        for r in per:
            worst = model_max.setdefault(r["model"], {"logit_error": 0.0, "nll_error": 0.0})
            worst.update(logit_error=max(worst["logit_error"], r["logit_error"]), nll_error=max(worst["nll_error"], r["nll_error"]))
            worst["cut_groups"] = worst.get("cut_groups", 0) + r["cut_groups"]
        run = {"fixture": name, "seed": seed, "weights_sha256": digest, "weights": "bf16",
               "logit_error": max(r["logit_error"] for r in per), "nll_error": max(r["nll_error"] for r in per),
               form + "_logit_error": plain[0], form + "_nll_error": plain[1], "wide_logit_error": check[0], "wide_nll_error": check[1]}
        runs.append(run)
        print("%-18s seed %-6d %s logit %.8f nll %.8f (input class on bf16 weights %.8f %.8f, wide %.8f %.8f)" %
              (name, seed, cls, run["logit_error"], run["nll_error"], plain[0], plain[1], check[0], check[1]), flush=True)
        del model
    if not all(m["cut_groups"] for m in model_max.values()):
        raise ValueError("a sum model's alignment cut changed no product")
    measured = dict(budgets({cls: runs})[cls], sum_models=list(specs), model_max=model_max, weights="bf16")
    if measured["max_logit_error"] >= f16["logit_budget"] or measured["max_nll_error"] >= f16["nll_budget"]:
        raise ValueError("%s costs more than the %s budget allows; it needs a budget of its own" % (cls, form))
    doc["dtypes"][cls] = measured
    return doc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", required=True, help="new result file; never overwrites a frozen budget")
    parser.add_argument("--family", choices=("q5-0", "q5-1", "q3-k", "iq4-xs", "iq3-s", "iq2-s", "iq2-xxs", "iq2-xs", "iq3-xxs", "iq1-s", "iq1-m", "tq1-0", "tq2-0", "nvfp4", "q1-0", "q2-0"), help="measure this raw-block fixture family; Q3_K measures BF16 only, Q1_0/Q2_0 F16 and BF16")
    parser.add_argument("--class", dest="cls", choices=tuple(CLASSES), help="measure the f16-mma class of the TF32 tile or the bf16-mma class of the BF16 tile on BF16 weights")
    parser.add_argument("--sum-models", help="sum models G:align:final:reference[:kN], comma separated (default: f16-mma the probed model and the envelope, bf16-mma BF16_MODELS)")
    args = parser.parse_args(argv)
    import torch
    import transformers
    if torch.__version__ != "2.5.1+cpu" or transformers.__version__ != "4.55.2":
        raise SystemExit("calibration requires torch 2.5.1+cpu and transformers 4.55.2")
    torch.set_num_threads(1)
    if args.cls:
        specs = args.sum_models.split(",") if args.sum_models else list(PROBED + ENVELOPE if args.cls == MMA else BF16_MODELS)
        doc = mma_budget(torch, transformers, specs, args.cls)
        with open(args.output, "x", encoding="utf-8", newline="\n") as f:
            json.dump(doc, f, indent=1)
            f.write("\n")
        print("wrote %s" % args.output)
        return
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
