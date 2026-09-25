import os
import tempfile

import common
import f32
from common import run_f32_cache as cli


def run():
    if common.f32_cache_skip("shards"):
        return True
    worst = 0.0
    cases = 0
    with tempfile.TemporaryDirectory(prefix="llmx_sharded_hf_") as directory:
        for fixture in f32.golden("baseline_f32.json")["fixtures"]:
            path = f32.write_model(os.path.join(directory, "model.gguf"), fixture["weights"], shards=3)
            error, count = common.check_hf_fixture("sharded", path, fixture["cases"], fixture["perplexity"],
                                                   f32.TEXTS[-1], (3,))
            worst = max(worst, error)
            cases += count
            os.remove(path.replace("00001-of-", "00003-of-"))
            rc, _ = cli(["info", path])
            assert rc != 0, "missing shard accepted by CLI"
    print("shards: %d full-logit HF cases, tied/untied, threads, PPL and missing-shard CLI; max error %.8f  [ok]" % (cases, worst))
    return True
