"""Generate the tiny minimax-m3 references with transformers' own MiniMaxM3VLForCausalLM, pinned and offline.

Each fixture of tests/fixture_minimax.py is built from its seed, loaded into HF's text model in float32 with eager attention on one CPU thread, and run whole and token by token (docs/MINIMAX.md, Classes and checks).
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import struct
import sys
import tempfile

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS.parent / "tests"))
import fixture_minimax as fixture  # noqa: E402

PYTHON_VERSION = "3.12.13"
TORCH_VERSION = "2.11.0+cpu"
TRANSFORMERS_VERSION = "5.23.0"
# The transformers sources the reference runs, by SHA-256.
SOURCE_SHA256 = {
    "models/minimax_m3_vl/modeling_minimax_m3_vl.py": "52b4860cedbb15a19036f33b04415c9204fa44e1a553e6efa2ca52d965a06452",
    "models/minimax_m3_vl/configuration_minimax_m3_vl.py": "be3b1b3a15dda7c6c215fe954eede1e624bdc13cf4c21d1bd28ba7aa364ea8fb",
    "cache_utils.py": "bf5e6269ba573a541462f41f0388a50ec114299dc736f97c344cc6e3345a8fa3",
}
OUTPUT_NAME = "minimax_m3_reference.json"
# Each fixture's first weight seed tried; the first from it whose selections all clear the floor is kept.
BASE_SEED = 31_2026
SEEDS_TRIED = 64
SEQUENCE_SEED = 31_1115
LENGTHS = {"small": 64, "wide": 272}
PROMPT = "the quick brown fox"
GREEDY_TOKENS = 24
# Every selection's margin at its cut, and every greedy pick's over the runner-up, must clear the per-logit bound, and the cached rows equal the whole prompt's within half of it.
MARGIN_FLOOR = 2e-5
EQUAL_BOUND = 1e-5


def sequence(length):
    """The scored ids: 2 to 127 from the fixture's generator, so neither BOS nor EOS appears."""
    rng = fixture._stream(SEQUENCE_SEED, "sequence")
    return [2 + (rng.next() >> 40) % 126 for _ in range(length)]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_environment(python_version, torch_version, transformers_version, source_hashes):
    """Refuse a Python, torch, transformers or transformers source other than the pinned ones."""
    for name, found, wanted in (("Python", python_version, PYTHON_VERSION), ("torch", torch_version, TORCH_VERSION),
                                ("transformers", transformers_version, TRANSFORMERS_VERSION)):
        if found != wanted:
            raise RuntimeError("minimax reference requires %s %s, found %s" % (name, wanted, found))
    if source_hashes != SOURCE_SHA256:
        raise RuntimeError("minimax reference: the installed transformers sources are not the pinned ones")


def pinned_environment():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    import transformers
    root = Path(transformers.__file__).resolve().parent
    check_environment(platform.python_version(), torch.__version__, transformers.__version__,
                      {name: _sha256(root / name) for name in SOURCE_SHA256})
    return torch, importlib.import_module("transformers.models.minimax_m3_vl.modeling_minimax_m3_vl"), \
        importlib.import_module("transformers.models.minimax_m3_vl.configuration_minimax_m3_vl")


def _short(value):
    """The shortest decimal that reads back as the same f32."""
    f32 = lambda v: struct.unpack("<f", struct.pack("<f", v))[0]
    value = f32(float(value))
    for digits in range(1, 10):
        text = "%.*g" % (digits, value)
        if f32(float(text)) == value:
            return float(text)
    return value


class _Trace:
    """The margin at the cut of every top-k the routers and indexers run, and each indexer's kept block ids, observed without changing HF's operations."""

    def __init__(self, model, torch):
        from torch.utils._python_dispatch import TorchDispatchMode
        self.owner, self.margins, self.kept, self.handles = [], {"router": [], "index": []}, [], []
        trace = self

        class Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                out = func(*args, **(kwargs or {}))
                if str(func) == "aten.topk.default" and trace.owner:
                    k = int(args[1])
                    rows = args[0].detach().double().reshape(-1, args[0].shape[-1]).sort(dim=-1, descending=True).values
                    if rows.shape[-1] > k:
                        cut, after = rows[:, k - 1], rows[:, k]
                        finite = torch.isfinite(cut) & torch.isfinite(after)
                        trace.margins[trace.owner[-1]].extend((cut - after)[finite].tolist())
                return out

        self.mode = Mode()
        for name, module in model.named_modules():
            label = "index" if name.endswith(".indexer") else "router" if name.endswith(".mlp.gate") else None
            if label:
                self.handles.append(module.register_forward_pre_hook(lambda _m, _i, label=label: self.owner.append(label)))
                self.handles.append(module.register_forward_hook(self._after, always_call=True))

    def _after(self, module, _inputs, output):
        if self.owner.pop() == "index":
            self.kept.append(output.detach().clone())
        return output

    def close(self):
        for handle in self.handles:
            handle.remove()


def _build(torch, modeling, configuration, config, seed):
    """HF's model of `config` holding the fixture's state of `seed`, every parameter and buffer checked against the fixture's names and shapes."""
    hf_config = configuration.MiniMaxM3VLTextConfig(**config)
    hf_config._attn_implementation = "eager"
    model = modeling.MiniMaxM3VLForCausalLM(hf_config).float().eval()
    state = fixture.make_state(config, seed)
    want = {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}
    if want != {name: tuple(shape) for name, (shape, _) in state.items()}:
        raise RuntimeError("minimax reference: HF's parameters are not the fixture's: %s" % sorted(set(want) ^ set(state)))
    model.load_state_dict({name: torch.tensor(values, dtype=torch.float32).reshape(shape) for name, (shape, values) in state.items()}, strict=True)
    if model.model.rotary_emb.inv_freq.numel() * 2 != config["rotary_dim"]:
        raise RuntimeError("minimax reference: HF's rope turns another width than rotary_dim")
    return model, state


def _first_sparse(kept, block):
    """The first position whose kept blocks, in the whole forward's selections, are fewer than its visible ones."""
    first = None
    for ids in kept:
        valid = (ids >= 0).sum(dim=-1)[0]
        for position in range(valid.shape[-1]):
            if int(valid[:, position].min()) < position // block + 1:
                first = position if first is None else min(first, position)
                break
    return first


def _case(torch, modeling, configuration, config, seed, ids):
    model, state = _build(torch, modeling, configuration, config, seed)
    trace = _Trace(model, torch)
    try:
        with torch.no_grad(), trace.mode:
            whole = model(torch.tensor([ids]), use_cache=False).logits[0]
            kept = list(trace.kept)
            cache, rows = None, []
            for token in ids:
                out = model(torch.tensor([[token]]), past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                rows.append(out.logits[0, -1])
            cached = torch.stack(rows)
            prompt = list(PROMPT.encode("ascii"))
            out = model(torch.tensor([prompt]), use_cache=True)
            cache, logits, greedy, greedy_margin = out.past_key_values, out.logits[0, -1], [], math.inf
            for _ in range(GREEDY_TOKENS):
                top = logits.double().topk(2)
                greedy_margin = min(greedy_margin, float(top.values[0] - top.values[1]))
                greedy.append(int(top.indices[0]))
                out = model(torch.tensor([[greedy[-1]]]), past_key_values=cache, use_cache=True)
                cache, logits = out.past_key_values, out.logits[0, -1]
    finally:
        trace.close()
    least = {kind: min(values) for kind, values in trace.margins.items()}
    if not all(trace.margins.values()) or min(least.values()) < MARGIN_FLOOR or greedy_margin < MARGIN_FLOOR:
        return None
    if not torch.isfinite(whole).all() or float((whole - cached).abs().max()) > EQUAL_BOUND:
        raise RuntimeError("minimax reference: the token-by-token rows differ from the whole prompt's by %.3g" % float((whole - cached).abs().max()))
    first = _first_sparse(kept, config["index_block_size"])
    if first is None or first >= len(ids):
        raise RuntimeError("minimax reference: no selection drops a block within the sequence")
    # The same weights keeping every block: how far the sparse rows sit from dense attention, so matching them proves the selection ran.
    dense_model, _ = _build(torch, modeling, configuration, dict(config, index_topk_blocks=config["max_position_embeddings"] // config["index_block_size"] + 1), seed)
    with torch.no_grad():
        dense = dense_model(torch.tensor([ids]), use_cache=False).logits[0]
    logits = whole.double()
    nll = [float(torch.logsumexp(logits[t], 0) - logits[t, ids[t + 1]]) for t in range(len(ids) - 1)]
    return {
        "config": config, "seed": seed, "state_sha256": fixture.state_sha256(state), "sequence": ids,
        "logits": [[_short(v) for v in row] for row in whole.tolist()], "nll": [_short(v) for v in nll], "mean_nll": sum(nll) / len(nll),
        "cached_max_abs_diff": float((whole - cached).abs().max()), "first_sparse_position": first,
        "dense_max_abs_diff": float((whole[first:] - dense[first:]).abs().max()),
        "margins": {kind: {"minimum": least[kind], "count": len(values)} for kind, values in trace.margins.items()},
        "prompt": PROMPT, "greedy": greedy, "greedy_margin": greedy_margin,
    }


def generate(output_dir):
    """Write OUTPUT_NAME with each fixture's golden and its file's SHA-256, replacing it only once every fixture has one."""
    torch, modeling, configuration = pinned_environment()
    torch.set_num_threads(1)
    doc = {"_comment": "Generated by tools/minimax_reference.py from transformers' MiniMaxM3VLForCausalLM, float32 eager on one CPU thread; the weights are tests/fixture_minimax.py's of each seed.",
           "python_version": PYTHON_VERSION, "torch_version": TORCH_VERSION, "transformers_version": TRANSFORMERS_VERSION, "source_sha256": SOURCE_SHA256,
           "margin_floor": MARGIN_FLOOR, "equal_bound": EQUAL_BOUND, "fixtures": {}}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="minimax_reference_", dir=out) as stage:
        for name, config in fixture.FIXTURES.items():
            ids = sequence(LENGTHS[name])
            for seed in range(BASE_SEED, BASE_SEED + SEEDS_TRIED):
                case = _case(torch, modeling, configuration, config, seed, ids)
                if case:
                    break
                print("%s: seed %d has a selection or greedy pick within %.3g of a tie" % (name, seed, MARGIN_FLOOR))
            else:
                raise RuntimeError("minimax reference: no seed of %d clears the margin floor for %s" % (SEEDS_TRIED, name))
            path = fixture.write_minimax(Path(stage) / (name + ".gguf"), config, fixture.make_state(config, case["seed"]))
            case["file_sha256"] = _sha256(path)
            doc["fixtures"][name] = case
            print("%s: seed %d, margins %s, greedy margin %.3g, first sparse position %d, dense rows %.3g away, cached rows within %.3g" % (
                name, case["seed"], {k: v["minimum"] for k, v in case["margins"].items()}, case["greedy_margin"], case["first_sparse_position"],
                case["dense_max_abs_diff"], case["cached_max_abs_diff"]))
        staged = Path(stage) / OUTPUT_NAME
        staged.write_text(json.dumps(doc, separators=(",", ":")) + "\n", encoding="utf-8")
        os.replace(staged, out / OUTPUT_NAME)
    return out / OUTPUT_NAME


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output_dir", nargs="?", default=str(TOOLS.parent / "tests" / "data"), help="where minimax_m3_reference.json is written")
    generate(parser.parse_args().output_dir)
