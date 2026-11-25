"""The tiny minimax-m3 fixtures held to HF's MiniMaxM3VLForCausalLM (docs/MINIMAX.md, Classes and checks).

Each fixture of tests/fixture_minimax.py is written from the committed golden's seed, its file held to the reference's SHA-256.
`llmx logits` reads the golden's sequence through --then-ids, and every position's logits must be HF's within 2e-5 and each target's NLL within 1e-5: whole at lengths either side of the first sparse drop, token by token (--per-token), at ubatch 1, 3 and 16 and at one and three threads.
Greedy `generate` after the golden's prompt gives HF's ids, and no selection or greedy pick of the golden sits within 2e-5 of a tie.
On a configured device every check runs with the model whole on the device, then again with --cpu-moe, the attention on the device and the experts on the CPU.
On the CPU, llmx-route-probe found beside the executable prints the rows `logits` prints at its settings byte for byte, and each routed layer's picks at every position are the top k of the router scores it wrote.
"""
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile

import common
import fixture_minimax

GOLDEN = os.path.join(common.ROOT, "tests", "data", "minimax_m3_reference.json")
F32 = ["--cache-type-k", "f32", "--cache-type-v", "f32"]
LOGIT_BOUND = 2e-5
NLL_BOUND = 1e-5


def rows(path, directory, ids, vocab, flags=()):
    """The logits rows `logits` prints for `ids`, read through --then-ids after an empty text, every position with the whole vocabulary."""
    text = os.path.join(directory, "empty.txt")
    open(text, "w").close()
    tail = os.path.join(directory, "sequence.ids")
    with open(tail, "w") as f:
        f.write(" ".join(str(i) for i in ids))
    p = common.run_process(["logits", path, "--file", text, "--then-ids", tail, "--last", str(len(ids)), "--top", str(vocab)] + F32 + list(flags), text=True)
    assert p.returncode == 0, "minimax: logits %s gave exit %d, stderr %r" % (" ".join(flags), p.returncode, p.stderr[-400:])
    lines = p.stdout.splitlines()
    assert lines[0] == "tokens: %d" % len(ids), lines[0]
    result = []
    for position, line in enumerate(lines[1:]):
        fields = line.split()
        assert int(fields[0]) == position and len(fields) == 1 + 2 * vocab, "minimax: row %d of %s: %r" % (position, flags, line[:80])
        result.append({int(i): float(v) for i, v in zip(fields[1::2], fields[2::2])})
    assert len(result) == len(ids), (flags, len(result))
    return result


def nll(row, target):
    peak = max(row.values())
    return peak + math.log(sum(math.exp(v - peak) for v in row.values())) - row[target]


def held(name, got, golden):
    """The largest logit and NLL errors of `got`, rows from position 0, against the golden, each asserted within its bound."""
    sequence, logits = golden["sequence"], golden["logits"]
    logit_error = max(abs(row[i] - logits[t][i]) for t, row in enumerate(got) for i in range(len(logits[t])))
    nll_error = max([abs(nll(got[t], sequence[t + 1]) - golden["nll"][t]) for t in range(len(got) - 1)] or [0.0])
    assert logit_error <= LOGIT_BOUND, "minimax: %s logits differ from HF's by %.3g (bound %g)" % (name, logit_error, LOGIT_BOUND)
    assert nll_error <= NLL_BOUND, "minimax: %s NLL differs from HF's by %.3g (bound %g)" % (name, nll_error, NLL_BOUND)
    return logit_error, nll_error


def gate(name, path, directory, golden, floor, extra=()):
    """The whole lengths either side of the first sparse drop, then the whole sequence token by token and over passes and threads, each run with `extra` flags."""
    sequence, vocab, first = golden["sequence"], golden["config"]["vocab_size"], golden["first_sparse_position"]
    closest = min([m["minimum"] for m in golden["margins"].values()] + [golden["greedy_margin"]])
    assert closest >= floor, "minimax: %s: a selection or greedy pick sits %.3g from a tie" % (name, closest)
    # HF's rows from the first drop on sit this far from dense attention's, so rows within the bound took the selection.
    assert golden["dense_max_abs_diff"] > 10 * LOGIT_BOUND, "minimax: %s: the sparse rows are %.3g from dense attention's" % (name, golden["dense_max_abs_diff"])
    lengths = sorted({1, 2, first, first + 1, first + 2, len(sequence)})
    runs = [("whole %d" % n, sequence[:n], []) for n in lengths]
    runs += [("decode", sequence, ["--per-token"]), ("ubatch 1", sequence, ["--ubatch", "1"]), ("ubatch 3", sequence, ["--ubatch", "3"]),
             ("ubatch 16", sequence, ["--ubatch", "16"]), ("threads 1", sequence, ["--threads", "1"]), ("threads 3", sequence, ["--threads", "3"])]
    worst, whole = [0.0, 0.0], None
    for label, ids, flags in runs:
        got = rows(path, directory, ids, vocab, flags + list(extra))
        if len(ids) == len(sequence) and not flags:
            whole = got
        worst = [max(a, b) for a, b in zip(worst, held("%s %s" % (name, label), got, golden))]
    mean = sum(nll(whole[t], sequence[t + 1]) for t in range(len(sequence) - 1)) / (len(sequence) - 1)
    assert abs(mean - golden["mean_nll"]) <= NLL_BOUND, "minimax: %s mean NLL %.9f against HF's %.9f" % (name, mean, golden["mean_nll"])
    p = common.run_process(["generate", path, golden["prompt"], "-n", str(len(golden["greedy"])), "--temp", "0", "--ignore-eos", "--verbose"] + F32 + list(extra))
    assert p.returncode == 0, "minimax: generate failed: %s" % p.stderr.decode("utf-8", "replace")
    line = re.search(rb"^ids:(.*)$", common.cli_stdout(p.stdout), re.M)
    ids = [int(i) for i in line[1].replace(b",", b" ").split()]
    assert ids == golden["greedy"], "minimax: %s greedy ids %s, HF's %s" % (name, ids, golden["greedy"])
    print("minimax: %s%s: %d runs from 1 to %d tokens, first sparse at %d (dense rows %.3g away), every logit within %.3g of HF's (bound %g), each target's NLL within %.3g (bound %g), "
          "mean NLL %.9f against %.9f, %d greedy ids HF's, no selection or pick nearer a tie than %.3g"
          % (name, " " + " ".join(extra) if extra else "", len(runs), len(sequence), first, golden["dense_max_abs_diff"], worst[0], LOGIT_BOUND, worst[1], NLL_BOUND, mean, golden["mean_nll"], len(ids), closest))


def probe_path():
    return os.path.join(os.path.dirname(common.exe_path()), "llmx-route-probe" + (".exe" if os.name == "nt" else ""))


def routes(tool, name, path, directory, golden, state, ubatch):
    """llmx-route-probe's rows against `logits` at its settings, byte for byte, and its picks against the top k of sigmoid(score) + bias, which no golden selection holds within 2e-5 of a tie."""
    sys.path.insert(0, os.path.join(common.ROOT, "tools"))
    import minimax_file_reference
    config, sequence = golden["config"], golden["sequence"]
    ids = os.path.join(directory, "probe.ids")
    with open(ids, "w") as f:
        f.write(" ".join(str(i) for i in sequence))
    prefix = os.path.join(directory, "probe")
    p = subprocess.run([tool, path, ids, prefix, str(len(sequence)), str(ubatch)], capture_output=True, encoding="utf-8", errors="replace", timeout=300)
    assert p.returncode == 0, "minimax: llmx-route-probe gave exit %d: %s" % (p.returncode, p.stderr[-400:])
    empty = os.path.join(directory, "empty.txt")
    open(empty, "w").close()
    p = common.run_process(["logits", path, "--file", empty, "--then-ids", ids, "--last", str(len(sequence)), "--top", str(config["vocab_size"]), "--device", "cpu",
                            "--dtype", "f32", "--load-mode", "mapped", "--ubatch", str(ubatch)] + F32, text=True)
    assert p.returncode == 0, "minimax: logits gave exit %d" % p.returncode
    with open(prefix + ".txt", encoding="utf-8") as f:
        assert f.read() == p.stdout, "minimax: %s: llmx-route-probe's rows at ubatch %d are not those logits prints" % (name, ubatch)
    got = minimax_file_reference.read_routes(prefix + ".routes")
    routed = [i for i, kind in enumerate(config["mlp_layer_types"]) if kind == "sparse"]
    assert sorted(got) == routed, "minimax: %s: routes of layers %s, the model routes %s" % (name, sorted(got), routed)
    k, n = config["num_experts_per_tok"], config["num_local_experts"]
    for layer in routed:
        picks, scores, k_got, n_got = got[layer]
        assert (k_got, n_got, len(picks)) == (k, n, len(sequence) * k), "minimax: %s layer %d: %s" % (name, layer, (k_got, n_got, len(picks)))
        bias = state["model.layers.%d.mlp.gate.e_score_correction_bias" % layer][1]
        for t in range(len(sequence)):
            row = [1.0 / (1.0 + math.exp(-scores[t * n + e])) + bias[e] for e in range(n)]
            best = sorted(range(n), key=lambda e: (-row[e], e))[:k]
            assert sorted(picks[t * k:(t + 1) * k]) == sorted(best), "minimax: %s layer %d position %d picked %s, the scores' top %d %s" % (name, layer, t, list(picks[t * k:(t + 1) * k]), k, best)
    return len(routed)


def run(require=False):
    if common.tensor_width_skip("minimax") or common.f32_cache_skip("minimax"):
        return common.SKIPPED
    with open(GOLDEN, encoding="utf-8") as f:
        document = json.load(f)
    with tempfile.TemporaryDirectory(prefix="llmx_minimax_") as directory:
        for name, golden in document["fixtures"].items():
            assert golden["config"] == fixture_minimax.FIXTURES[name], "minimax: the golden's %s config is not the fixture's" % name
            state = fixture_minimax.make_state(golden["config"], golden["seed"])
            assert fixture_minimax.state_sha256(state) == golden["state_sha256"], "minimax: %s's weights are not the reference's" % name
            path = str(fixture_minimax.write_minimax(os.path.join(directory, name + ".gguf"), golden["config"], state))
            with open(path, "rb") as f:
                assert hashlib.sha256(f.read()).hexdigest() == golden["file_sha256"], "minimax: the %s file is not the reference's" % name
            gate(name, path, directory, golden, document["margin_floor"])
            if any(d != "cpu" for d in os.environ.get("LLMX_DEVICE", "cpu").split(",")):
                # The experts' offload keeps the attention on the device and moves the routed experts to the CPU, held to HF's bounds the same way.
                gate(name, path, directory, golden, document["margin_floor"], ["--cpu-moe"])
                continue
            tool = probe_path()
            if not os.path.exists(tool):
                assert not require, "minimax: %s not found beside the executable" % tool
                print("minimax: route probe SKIP - %s not found beside the executable" % tool)
                continue
            for ubatch in (3, len(golden["sequence"])):
                layers = routes(tool, name, path, directory, golden, state, ubatch)
            print("minimax: %s: llmx-route-probe's rows are logits' byte for byte at ubatch 3 and %d, and its picks in %d routed layers the top %d of its scores"
                  % (name, len(golden["sequence"]), layers, golden["config"]["num_experts_per_tok"]))
    return True
