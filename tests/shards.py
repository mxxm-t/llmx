import json
import os
import tempfile

import common
import f32
from common import run_f32_cache as cli


def run():
    if common.f32_cache_skip("shards"):
        return True
    with open(os.path.join(os.path.dirname(__file__), "data", "baseline_f32.json"), encoding="utf-8") as stream:
        golden = json.load(stream)
    worst = 0.0
    cases = 0
    with tempfile.TemporaryDirectory(prefix="llmx_sharded_hf_") as directory:
        for fixture in golden["fixtures"]:
            weights = f32.tensors(fixture["tied"])
            assert f32.weight_hash(weights) == fixture["weights_sha256"]
            path = f32.write_model(os.path.join(directory, "model.gguf"), weights, shards=3)
            error, count = common.check_hf_fixture("sharded", path, fixture["cases"], fixture["perplexity"],
                                                   f32.TEXTS[-1], (3,))
            worst = max(worst, error)
            cases += count
            os.remove(path.replace("00001-of-", "00003-of-"))
            rc, _ = cli(["info", path])
            assert rc != 0, "missing shard accepted by CLI"
    print("shards: %d full-logit HF cases, tied/untied, threads, PPL and missing-shard CLI; max error %.8f  [ok]" % (cases, worst))
    return True
