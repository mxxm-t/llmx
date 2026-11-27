"""Generate the tiny inkling2_loop reference with transformers' own Inkling2LoopForCausalLM, pinned and offline.

The fixture of tests/fixture_inkling2.py is built from its seed, loaded into HF's text model in float32 with eager attention on one CPU thread, and run whole, in chunks and token by token (docs/INKLING2.md, 6. Correctness).
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS.parent / "tests"))
sys.path.insert(0, str(TOOLS))
import fixture_inkling2 as fixture  # noqa: E402
from fixture_deepseek4 import _stream  # noqa: E402
from minimax_reference import _short, _sha256  # noqa: E402

PYTHON_VERSION = "3.12.13"
TRANSFORMERS_VERSION = "5.24.0"
MODELING = "models/inkling2_loop/modeling_inkling2_loop.py"
OUTPUT_NAME = "inkling2_reference.json"
BASE_SEED = 24_2026
SEEDS_TRIED = 64
SEQUENCE_SEED = 24_1122
# Past the window (5), the relative extent (7), the log-scale floor (8) and many convolution ring wraps.
LENGTH = 48
CHUNK = 7
PROMPT = "the quick brown fox"
GREEDY_TOKENS = 16
MARGIN_FLOOR = 2e-5
EQUAL_BOUND = 1e-5


def sequence(length):
    """The scored ids: 2 to 255 from the fixture's generator, so neither BOS nor EOS nor a padded row appears."""
    rng = _stream(SEQUENCE_SEED, "sequence")
    return [2 + (rng.next() >> 40) % 254 for _ in range(length)]


def pinned_environment():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    import transformers
    for name, found, wanted in (("Python", platform.python_version(), PYTHON_VERSION), ("transformers", transformers.__version__, TRANSFORMERS_VERSION)):
        if found != wanted:
            raise RuntimeError("inkling2 reference requires %s %s, found %s" % (name, wanted, found))
    root = Path(transformers.__file__).resolve().parent
    return torch, _sha256(root / MODELING), importlib.import_module("transformers.models.inkling2_loop.modeling_inkling2_loop"), \
        importlib.import_module("transformers.models.inkling2_loop.configuration_inkling2_loop")


def _build(torch, modeling, configuration, seed):
    hf_config = configuration.Inkling2LoopTextConfig(**fixture.CONFIG)
    hf_config._attn_implementation = "eager"
    model = modeling.Inkling2LoopForCausalLM(hf_config).float().eval()
    state = fixture.make_state(fixture.CONFIG, seed)
    want = {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}
    if want != {name: tuple(shape) for name, (shape, _) in state.items()}:
        raise RuntimeError("inkling2 reference: HF's parameters are not the fixture's: %s" % sorted(set(want) ^ set(state)))
    model.load_state_dict({name: torch.tensor(values, dtype=torch.float32).reshape(shape) for name, (shape, values) in state.items()}, strict=True)
    return model, state, hf_config


class _Trace:
    """The margin at the cut of every router top-k and how many rows each clip shortens, observed without changing HF's operations."""

    def __init__(self, model, torch):
        from torch.utils._python_dispatch import TorchDispatchMode
        self.margins, self.clipped, self.rows, self.handles = [], 0, 0, []
        trace = self

        class Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                out = func(*args, **(kwargs or {}))
                if str(func) == "aten.topk.default":
                    k = int(args[1])
                    rows = args[0].detach().double().reshape(-1, args[0].shape[-1]).sort(dim=-1, descending=True).values
                    if rows.shape[-1] > k:
                        trace.margins.extend((rows[:, k - 1] - rows[:, k]).tolist())
                return out

        self.mode = Mode()
        for name, module in model.named_modules():
            if type(module).__name__ in ("Inkling2LoopScaledL2Clip", "Inkling2LoopLogScaledL2Clip"):
                self.handles.append(module.register_forward_hook(self._clip))

    def _clip(self, module, inputs, output):
        x = inputs[0].float() * module.scale.float()
        threshold = module.threshold if hasattr(module, "threshold") else module.l2_clip.threshold
        norms = x.norm(dim=-1)
        self.clipped += int((norms > threshold).sum())
        self.rows += norms.numel()

    def close(self):
        for handle in self.handles:
            handle.remove()


def _case(torch, modeling, configuration, seed, ids):
    model, state, hf_config = _build(torch, modeling, configuration, seed)
    trace = _Trace(model, torch)
    try:
        with torch.no_grad(), trace.mode:
            whole = model(torch.tensor([ids]), use_cache=False).logits[0]
            from transformers.cache_utils import DynamicCache
            cache, rows = DynamicCache(config=hf_config), []
            for token in ids:
                rows.append(model(torch.tensor([[token]]), past_key_values=cache, use_cache=True).logits[0, -1])
            cached = torch.stack(rows)
            cache, rows = DynamicCache(config=hf_config), []
            for at in range(0, len(ids), CHUNK):
                rows.append(model(torch.tensor([ids[at:at + CHUNK]]), past_key_values=cache, use_cache=True).logits[0])
            chunked = torch.cat(rows)
            prompt = list(PROMPT.encode("ascii"))
            cache = DynamicCache(config=hf_config)
            logits, greedy, greedy_margin = model(torch.tensor([prompt]), past_key_values=cache, use_cache=True).logits[0, -1], [], math.inf
            for _ in range(GREEDY_TOKENS):
                top = logits.double().topk(2)
                greedy_margin = min(greedy_margin, float(top.values[0] - top.values[1]))
                greedy.append(int(top.indices[0]))
                logits = model(torch.tensor([[greedy[-1]]]), past_key_values=cache, use_cache=True).logits[0, -1]
    finally:
        trace.close()
    if min(trace.margins) < MARGIN_FLOOR or greedy_margin < MARGIN_FLOOR:
        return None
    cached_diff, chunked_diff = float((whole - cached).abs().max()), float((whole - chunked).abs().max())
    if not torch.isfinite(whole).all() or max(cached_diff, chunked_diff) > EQUAL_BOUND:
        raise RuntimeError("inkling2 reference: cached rows differ from the whole prompt's by %.3g, chunked by %.3g" % (cached_diff, chunked_diff))
    if not 0 < trace.clipped < trace.rows:
        raise RuntimeError("inkling2 reference: %d of %d clip rows shortened; the fixture must clip some and not all" % (trace.clipped, trace.rows))
    logits = whole.double()
    nll = [float(torch.logsumexp(logits[t], 0) - logits[t, ids[t + 1]]) for t in range(len(ids) - 1)]
    return {
        "config": fixture.CONFIG, "seed": seed, "state_sha256": fixture.state_sha256(state), "sequence": ids,
        "logits": [[_short(v) for v in row] for row in whole.tolist()], "nll": [_short(v) for v in nll], "mean_nll": sum(nll) / len(nll),
        "cached_max_abs_diff": cached_diff, "chunked_max_abs_diff": chunked_diff, "chunk": CHUNK,
        "router_margin": {"minimum": min(trace.margins), "count": len(trace.margins)}, "clipped_rows": [trace.clipped, trace.rows],
        "prompt": PROMPT, "greedy": greedy, "greedy_margin": greedy_margin,
    }


def generate(output_dir):
    """Write OUTPUT_NAME with the fixture's golden and its file's SHA-256, replacing it only once complete."""
    torch, modeling_sha, modeling, configuration = pinned_environment()
    torch.set_num_threads(1)
    ids = sequence(LENGTH)
    for seed in range(BASE_SEED, BASE_SEED + SEEDS_TRIED):
        case = _case(torch, modeling, configuration, seed, ids)
        if case:
            break
        print("seed %d has a router or greedy pick within %.3g of a tie" % (seed, MARGIN_FLOOR))
    else:
        raise RuntimeError("inkling2 reference: no seed of %d clears the margin floor" % SEEDS_TRIED)
    doc = {"_comment": "Generated by tools/inkling2_reference.py from transformers' Inkling2LoopForCausalLM, float32 eager on one CPU thread; the weights are tests/fixture_inkling2.py's of the seed.",
           "python_version": PYTHON_VERSION, "torch_version": torch.__version__, "transformers_version": TRANSFORMERS_VERSION, "modeling_sha256": modeling_sha,
           "margin_floor": MARGIN_FLOOR, "equal_bound": EQUAL_BOUND, "fixture": case}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="inkling2_reference_", dir=out) as stage:
        path = fixture.write_inkling2(Path(stage) / "inkling2.gguf", fixture.CONFIG, fixture.make_state(fixture.CONFIG, case["seed"]))
        case["file_sha256"] = _sha256(path)
        staged = Path(stage) / OUTPUT_NAME
        staged.write_text(json.dumps(doc, separators=(",", ":")) + "\n", encoding="utf-8")
        os.replace(staged, out / OUTPUT_NAME)
    print("seed %d, router margin %.3g, greedy margin %.3g, clipped %d of %d, cached within %.3g, chunked within %.3g, mean nll %.4f" % (
        case["seed"], case["router_margin"]["minimum"], case["greedy_margin"], case["clipped_rows"][0], case["clipped_rows"][1],
        case["cached_max_abs_diff"], case["chunked_max_abs_diff"], case["mean_nll"]))
    return out / OUTPUT_NAME


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output_dir", nargs="?", default=str(TOOLS.parent / "tests" / "data"), help="where inkling2_reference.json is written")
    generate(parser.parse_args().output_dir)
