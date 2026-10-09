import hashlib
import json
import math
import os
import tempfile

import common
import moe
from f32 import write_model


# The tiny Q8_0 fixtures of --dtype int8 (docs/PRECISION.md): prompts of 160 to 256 tokens in a 512-token context, read whole, in windows and one token a step, so the row kernels and the prompt tile both run.
# Every row width is a multiple of 64, so a device's prompt tile takes each projection; the weights are tests/moe.py's Q8_0 draws.
DENSE_CONFIG = {"block_count": 2, "embedding_length": 128, "feed_forward_length": 256,
                "attention.head_count": 2, "attention.head_count_kv": 1,
                "attention.key_length": 64, "context_length": 512}
# Every token takes all eight experts, so no routing near a tie can turn over under rounding and set the budget (docs/STATUS.md, `--dtype int8` at every row count); each expert still runs its routed products over its own entries.
MOE_CONFIG = dict(moe.Q8_CONFIG, context_length=512, expert_used_count=8)
DENSE_SEED = 40961
MOE_SEED = moe.Q8_SEED
ROUTER_SCALE = 16.0

PASSAGE = ("The river rises in the high moors and runs south through a narrow valley of old mills, where the water was once "
           "turned to drive looms and grind grain for the villages along its banks. By the nineteenth century the mills had "
           "given way to a canal, and barges carried coal and stone to the coast, until the railway came and the canal fell "
           "quiet. Today the towpath is a walking route, the locks are kept by volunteers, and herons stand in the shallows "
           "below the weirs. In spring the meadows flood, and the lower fields hold water for weeks; in summer the river runs "
           "low and clear over gravel, and children wade at the fords where carts once crossed before the stone bridges were built.")
# Three prompts of 160, 200 and 256 tokens, one a byte; the last also gives the windows of 128 and 256 tokens.
TEXTS = [PASSAGE[0:160], PASSAGE[160:360], PASSAGE[360:616]]
CONTEXTS = (128, 256)
assert len(TEXTS[-1]) % 128 == 0 and PASSAGE.isascii()


def tensors(fixture, seed):
    """A fixture's tensors as tests/f32.py's write_model takes quantized ones: `dense-untied`, `dense-tied` or `moe`."""
    if fixture == "moe":
        return moe.q8_tensors(ROUTER_SCALE, seed, MOE_CONFIG)
    return moe.q8_tensors(None, seed, DENSE_CONFIG, range(DENSE_CONFIG["block_count"]), fixture == "dense-tied")


def write_fixture(path, fixture, seed):
    """Writes one fixture to `path` and returns the file's SHA-256."""
    config, arch = (MOE_CONFIG, "qwen3moe") if fixture == "moe" else (DENSE_CONFIG, "qwen3")
    write_model(path, [], config=config, arch=arch, quantized=tensors(fixture, seed))
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def golden():
    """tests/data/baseline_int8.json: HF float32 on each fixture's own file, checked against this module's texts and windows."""
    with open(os.path.join(os.path.dirname(__file__), "data", "baseline_int8.json"), encoding="utf-8") as f:
        doc = json.load(f)
    assert doc["texts"] == TEXTS and doc["contexts"] == list(CONTEXTS), "int8 fixture texts or windows changed"
    assert doc["dense_config"] == DENSE_CONFIG and doc["moe_config"] == MOE_CONFIG, "int8 fixture configs changed"
    return doc


def errors(model, fixture, dtype):
    """The largest logit and NLL errors of `model` under `dtype` against its golden, at the frozen bounds of the paths the runs witnessed; the paths seen."""
    worst, worst_nll, paths = 0.0, 0.0, set()
    for case in fixture["cases"]:
        p = common.run_process(["logits", model, case["text"], "--top", "257", "--dtype", dtype], cache="f32", text=True)
        assert p.returncode == 0, "int8 fixture logits failed: " + p.stdout + p.stderr
        bounds = common.hf_execution(p.stderr)
        paths |= witnessed(p.stderr)
        worst = max(worst, common.hf_logit_error("int8 fixture " + dtype, dict(zip(*common.parse_logits(p.stdout))), case["logits"], bound=bounds["logit"]))
    # Each window in batched passes and one token a step, so decode rows meet the reference too.
    for case in fixture["perplexity"]:
        for mode in ([], ["--per-token"]):
            p = common.run_process(["perplexity", model, TEXTS[-1], "-c", str(case["context"]), "--dtype", dtype] + mode, cache="f32", text=True)
            assert p.returncode == 0, "int8 fixture perplexity failed: " + p.stdout + p.stderr
            bounds = common.hf_execution(p.stderr)
            paths |= witnessed(p.stderr)
            error = abs(float(common.perplexity_fields(p.stdout)["mean NLL"]) - case["mean_nll"])
            assert math.isfinite(error) and error < bounds["nll"], "int8 fixture %s%s NLL error %.8f, bound %.8f" % (dtype, " per token" if mode else "", error, bounds["nll"])
            worst_nll = max(worst_nll, error)
    return worst, worst_nll, paths


def witnessed(stderr):
    """The matrix paths one run's witness names."""
    record = json.loads(next(line for line in stderr.splitlines() if line.startswith("matrix-paths: "))[len("matrix-paths: "):])
    return {path for device in record["devices"] for path in device}


def run():
    """The fixtures against HF float32 under f16 and under int8, each at the frozen budget of the paths it took: where the device runs int8 every run witnesses block-int8, and where it does not int8 widens to f16, as the record says."""
    if common.f32_cache_skip("int8") or common.tensor_width_skip("int8"):
        return common.SKIPPED
    doc = golden()
    report = []
    native = None
    with tempfile.TemporaryDirectory(prefix="llmx_int8_") as directory:
        for fixture in doc["fixtures"]:
            model = os.path.join(directory, fixture["name"] + ".gguf")
            assert write_fixture(model, fixture["name"], fixture["seed"]) == fixture["file_sha256"], "int8 fixture %s changed" % fixture["name"]
            logit, nll, paths = errors(model, fixture, "f16")
            report.append("%s f16 %.5f/%.6f" % (fixture["name"], logit, nll))
            assert "block-int8" not in paths, "f16 took 8-bit inputs"
            if native is None:
                p = common.run_process(["logits", model, TEXTS[0], "--dtype", "int8"], text=True)
                assert p.returncode == 0, "int8 logits failed: " + p.stdout + p.stderr
                record = next(line for line in p.stderr.splitlines() if line.startswith("dtype: "))
                native = " fallback to " not in record
                assert native or ("dtype: int8 -> f16 " in record and "(warning: emulated or wider fallback)" in record), record
            logit, nll, paths = errors(model, fixture, "int8")
            assert ("block-int8" in paths) == native, "int8 witnessed %s where the record says %s" % (sorted(paths), "native" if native else "widened")
            report.append("%s int8%s %.5f/%.6f" % (fixture["name"], "" if native else " (as f16)", logit, nll))
    print("int8: the 512-token Q8_0 fixtures against HF float32 at the frozen budgets of the paths taken; logit/NLL errors %s  [ok]" % ", ".join(report))
    return True


if __name__ == "__main__":
    run()
