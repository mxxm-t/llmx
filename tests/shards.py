import json
import math
import os
import tempfile

import f32
from common import run as cli


def run():
    if os.environ.get("LLMX_CACHE_TYPE", "f32") != "f32":
        print("shards: SKIP - the exact F32 gate needs f32 caches (LLMX_CACHE_TYPE=%s)" % os.environ["LLMX_CACHE_TYPE"])
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
            for threads in (1, 4):
                for case in fixture["cases"]:
                    rc, out = cli(["logits", path, case["text"], "--top", "257",
                                   "--threads", str(threads), "--ubatch", "3"])
                    assert rc == 0, "sharded logits failed: " + out
                    got = {int(p[0]): float(p[1]) for line in out.splitlines()
                           if len(p := line.split()) == 2 and p[0].isdigit()}
                    assert set(got) == set(range(257)), "missing sharded logits"
                    error = max(abs(got[i] - expected) for i, expected in enumerate(case["logits"]))
                    assert all(math.isfinite(v) for v in got.values()) and error < 2e-5
                    worst = max(worst, error)
                    cases += 1
                for case in fixture["perplexity"]:
                    rc, out = cli(["perplexity", path, f32.TEXTS[-1], "--threads", str(threads),
                                   "-c", str(case["context"])])
                    assert rc == 0, "sharded PPL failed: " + out
                    fields = dict(line.split(":", 1) for line in out.splitlines())
                    error = abs(float(fields["mean NLL"]) - case["mean_nll"])
                    assert math.isfinite(error) and error < 1e-5
            os.remove(path.replace("00001-of-", "00003-of-"))
            rc, _ = cli(["info", path])
            assert rc != 0, "missing shard accepted by CLI"
    print("shards: %d full-logit HF cases, tied/untied, threads, PPL and missing-shard CLI; max error %.8f  [ok]" % (cases, worst))
    return True
