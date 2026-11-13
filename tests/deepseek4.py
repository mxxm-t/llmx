"""The deepseek41 module's names held to the step 2 writer's files (docs/DEEPSEEK4.md, step 4).

Both layouts are written from the committed golden by tests/fixture_deepseek4.py, standard library only.
The gate (step 7, piece 7): `llmx logits` on the fork layout reads the golden's 200 ids through --then-ids, and every position's 128 logits must be the reference's within 2e-5 and every target's NLL, taken from those rows, within 1e-5.
It runs as the reference's 11 runs do, each prefix whole at 1, 2, 3, 64, 127, 128, 129 and 200 tokens and the decode path from the first token (--per-token), and the whole sequence again at ubatch 1, 3 and 16, and at one and three threads.
No selection of the golden sits within 2e-5 of a tie (margin_floor), so no run needs forced mode while the logits hold.
The upstream layout is refused by its key.
On a configured device the same gate runs; a device whose backend lacks one of the model's ops refuses the file at load by that op's name, and the component reports SKIP.
Its cache is kept as native, the model's packed formats and its default, or f32: the whole, decode and ubatch 3 runs under native print the f32 runs' logits, and f16 or two different sides are refused by the flag before the model is built.
`serve` gives the CLI's ids alone, together, after a prefix split, from a kept checkpoint and through a pause (check_serve), `generate --drafter lookup` gives the ids it gives with drafts off (`common.check_drafts`), and `info` lists the fork file's keys and tensors, and llmx-arch-deepseek4-test, found beside the executable, holds the file to the one its own C++ list builds.
On one ROCm or Vulkan device the backend's test beside the executable runs its scratch check on the served file (check_scratch): its passes grow the device by no more than the fit priced and the backend keeps.
"""
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile

import common
import fixture_deepseek4

GOLDEN = os.path.join(common.ROOT, "tests", "data", "deepseek41_reference.json")
DSPARK = os.path.join(common.ROOT, "tests", "data", "deepseek41_dspark.json")
UPSTREAM = "inference: unsupported metadata deepseek41.ple.layers"
DEVICE = "which the backend of its device does not implement"
F32 = ["--cache-type-k", "f32", "--cache-type-v", "f32"]
NATIVE = ["--cache-type-k", "native", "--cache-type-v", "native"]
LOGIT_BOUND = 2e-5
NLL_BOUND = 1e-5


def tool_path(name="llmx-arch-deepseek4-test"):
    return os.path.join(os.path.dirname(common.exe_path()), name + (".exe" if os.name == "nt" else ""))


# The scratch check of the one device's backend test (docs/TESTS.md, backend-rocm-scratch and vulkan-scratch) on `path`.
def check_scratch(path, require):
    devices = os.environ.get("LLMX_DEVICE", "cpu").split(",")
    kind, _, index = devices[0].partition(":")
    tools = {"rocm": ("llmx-backend-rocm-test", "scratch"), "vulkan": ("llmx-vulkan-lifetime-test", "--scratch")}
    if len(devices) != 1 or kind not in tools or not index.isdigit():
        return
    tool = tool_path(tools[kind][0])
    if not os.path.exists(tool):
        assert not require, "deepseek4: %s not found beside the executable" % tool
        print("deepseek4: scratch not checked - %s not found beside the executable" % tool)
        return
    p = subprocess.run([tool, tools[kind][1], path, index], capture_output=True, encoding="utf-8", errors="replace", timeout=600)
    assert p.returncode == 0, "deepseek4: the scratch check failed on %s: %s%s" % (devices[0], p.stdout[-2000:], p.stderr[-2000:])
    grew = [line for line in p.stdout.splitlines() if line.startswith("scratch fitted")]
    print("deepseek4: on %s the served file's passes grow the device by no more than its backend keeps (%s)" % (devices[0], grew[-1] if grew else "no line"))


def refused(path, text, flags=()):
    """`logits` on `path` with `flags` exits 1 with `text` as its error, its stdout at most the prompt's token count printed before the first pass."""
    p = common.run_process(["logits", path, "a"] + list(flags), text=True)
    assert p.returncode == 1 and text in p.stderr and all(line.startswith("tokens: ") for line in p.stdout.splitlines()), \
        "deepseek4: logits on %s gave exit %d, stdout %r, stderr %r" % (os.path.basename(path), p.returncode, p.stdout, p.stderr)


def rows(path, directory, ids, flags=(), cache=F32):
    """The logits rows `logits` prints for `ids`, read through --then-ids after an empty text, every position with all 128 ids, as a list of id-to-logit maps."""
    text = os.path.join(directory, "empty.txt")
    open(text, "w").close()
    tail = os.path.join(directory, "sequence.ids")
    with open(tail, "w") as f:
        f.write(" ".join(str(i) for i in ids))
    args = ["logits", path, "--file", text, "--then-ids", tail, "--last", str(len(ids)), "--top", "128"] + cache + list(flags)
    p = common.run_process(args, text=True)
    assert p.returncode == 0, "deepseek4: logits %s gave exit %d, stderr %r" % (" ".join(flags), p.returncode, p.stderr[-400:])
    lines = p.stdout.splitlines()
    assert lines[0] == "tokens: %d" % len(ids), lines[0]
    result = []
    for position, line in enumerate(lines[1:]):
        fields = line.split()
        assert int(fields[0]) == position and len(fields) == 1 + 2 * 128, "deepseek4: row %d of %s: %r" % (position, flags, line[:80])
        result.append({int(i): float(v) for i, v in zip(fields[1::2], fields[2::2])})
    assert len(result) == len(ids), (flags, len(result))
    return result


def nll(row, target):
    peak = max(row.values())
    return peak + math.log(sum(math.exp(v - peak) for v in row.values())) - row[target]


def held(name, got, document):
    """The largest logit and NLL errors of `got`, rows from position 0, against the golden, each asserted within its bound."""
    sequence, logits = document["sequence"], document["logits"]
    logit_error = max(abs(row[i] - logits[t][i]) for t, row in enumerate(got) for i in range(len(logits[t])))
    nll_error = max([abs(nll(got[t], sequence[t + 1]) - document["nll"][t]) for t in range(len(got) - 1)] or [0.0])
    assert logit_error <= LOGIT_BOUND, "deepseek4: %s logits differ from the reference's by %.3g (bound %g)" % (name, logit_error, LOGIT_BOUND)
    assert nll_error <= NLL_BOUND, "deepseek4: %s NLL differs from the reference's by %.3g (bound %g)" % (name, nll_error, NLL_BOUND)
    return logit_error, nll_error


def gate(path, directory, document):
    """The reference's runs, then the whole sequence over passes and threads; returns the largest errors."""
    sequence = document["sequence"]
    closest = min(margins["minimum"] for margins in document["margins"].values())
    assert closest >= document["margin_floor"], "deepseek4: a selection sits %.3g from a tie, which needs forced mode" % closest
    runs = [("whole %d" % length, sequence[:length], []) for length in sorted(set(r["length"] for r in document["runs"] if r["kind"] == "whole"))]
    runs += [("decode", sequence, ["--per-token"]), ("ubatch 1", sequence, ["--ubatch", "1"]), ("ubatch 3", sequence, ["--ubatch", "3"]),
             ("ubatch 16", sequence, ["--ubatch", "16"]), ("threads 1", sequence, ["--threads", "1"]), ("threads 3", sequence, ["--threads", "3"])]
    worst = [0.0, 0.0]
    whole = None
    for name, ids, flags in runs:
        got = rows(path, directory, ids, flags)
        if len(ids) == len(sequence) and not flags:
            whole = got
        errors = held(name, got, document)
        worst = [max(a, b) for a, b in zip(worst, errors)]
        # The packed cache reads back the bits f32 stores, so its logits print as the f32 run's; the default is native when no cache type is configured.
        if name in ("whole %d" % len(sequence), "decode", "ubatch 3"):
            assert rows(path, directory, ids, flags, NATIVE) == got, "deepseek4: %s under native differs from f32" % name
            if not os.environ.get("LLMX_CACHE_TYPE"):
                assert rows(path, directory, ids, flags, []) == got, "deepseek4: %s at the default cache type differs from f32" % name
    mean = sum(nll(whole[t], sequence[t + 1]) for t in range(len(sequence) - 1)) / (len(sequence) - 1)
    assert abs(mean - document["mean_nll"]) <= NLL_BOUND, "deepseek4: mean NLL %.9f against the reference's %.9f" % (mean, document["mean_nll"])
    print("deepseek4: %d runs, every position's 128 logits within %.3g of the reference (bound %g), each target's NLL within %.3g (bound %g), mean NLL %.9f against %.9f, no selection nearer a tie than %.3g"
          % (len(runs), worst[0], LOGIT_BOUND, worst[1], NLL_BOUND, mean, document["mean_nll"], closest))


def cli_ids(model, prompt, n, cache):
    """The ids greedy `generate` draws after `prompt`, with `cache` as both cache types."""
    p = common.run_process(["generate", model, prompt, "-n", str(n), "--temp", "0", "--ignore-eos", "--verbose", "--cache-type-k", cache, "--cache-type-v", cache])
    assert p.returncode == 0, "deepseek4: generate failed: %s" % p.stderr.decode("utf-8", "replace")
    line = re.search(rb"^ids:(.*)$", common.cli_stdout(p.stdout), re.M)
    return [int(i) for i in line[1].replace(b",", b" ").split()]


def greedy(text, n=24, logprobs=False):
    body = {"prompt": text, "temperature": 0, "max_tokens": n, "ignore_eos": True}
    return dict(body, logprobs=True, top_logprobs=5) if logprobs else body


def check_serve(model, cache, whole, extra=()):
    """Section 5.6 of docs/DEEPSEEK4.md on the fork file with a context of 1024, both cache sides `cache`: greedy ids equal the CLI's, and a forked request's log-probabilities a fresh server's.
    Without `whole` only the 129 donor, the 128-token shared prefix, the follow-up turn through the host tier and the pause without checkpoints run; with it also four prompts alone and together, donors of 1, 3, 127 and 128 tokens and pauses with checkpoints.
    `extra` flags go to every server, such as a drafter, whose drafts must leave the CLI's ids as they are."""
    import server
    flags = ["--cache-type-k", cache, "--cache-type-v", cache, *extra]
    cpu = all(d == "cpu" for d in os.environ.get("LLMX_DEVICE", "cpu").split(","))
    # The fixture's vocabulary is single characters without a space.
    prompts = ["helloworld", "abc", "thequickbrownfox", "0123456789"]
    if whole:
        srv = server.Server(model, "--max-seqs", "4", *flags)
        try:
            server.alone_and_together(srv, [greedy(text) for text in prompts])
            for text in prompts:
                assert server.post_ok(srv, "/v1/generate", greedy(text))["ids"] == cli_ids(model, text, 24, cache), text
        finally:
            srv.close()
        print("deepseek4: %s: served greedy ids alone and four at once equal the CLI's" % cache)
    # One character a token, so a prefix of s characters is a donor of s tokens; a prompt keeps a checkpoint at its last whole block short of its end, the model's largest KV block (128 on the CPU, 64 on the devices).
    text = "".join(chr(97 + (i * 7) % 26) for i in range(200))
    # A 160-token prompt and a donor of its first 128 and 32 others: one length, so one row class, and a checkpoint at 128 under either block.
    short = text[:160]
    other = text[:128] + "".join(chr(65 + (i * 3) % 26) for i in range(32))
    want = {p: cli_ids(model, p, 24, cache) for p in (text, short)}
    for split, donor, prompt in [(s, text[:s], text) for s in ((1, 3, 127, 128, 129) if whole else (129,))] + [(128, other, short)]:
        srv = server.Server(model, "--max-seqs", "4", *flags)
        try:
            server.post_ok(srv, "/v1/generate", greedy(donor, 1))
            reply = server.post_ok(srv, "/v1/generate", greedy(prompt))
            assert reply["ids"] == want[prompt], (split, len(donor), reply["ids"], want[prompt])
            # A device takes the donor's rows only where both prompts fall in one row class (Model::row_class): the CPU's always do, and prompts of one length do everywhere.
            reuse = {128} if donor == other or split > 128 and cpu else {0, 128} if split > 128 else {0}
            assert reply["reused_tokens"] in reuse, (split, len(donor), reply["reused_tokens"], srv.get("/v1/health")["reuse"])
        finally:
            srv.close()
    print("deepseek4: %s: after donors of %s tokens of a 200-token prompt and a 160-token donor sharing 128 of a 160-token prompt, a fresh server gives the CLI's ids, forking the 128-token checkpoint"
          % (cache, "1, 3, 127, 128 and 129" if whole else "129"))
    first = "".join(chr(97 + (i * 5) % 26) for i in range(300))
    follow = first + "andthen?"
    fresh = server.Server(model, "--max-seqs", "4", *flags)
    try:
        alone = server.post_ok(fresh, "/v1/generate", greedy(follow, logprobs=True))
    finally:
        fresh.close()
    # The first turn's checkpoint at 256 is moved to the host by a 900-token request on a pool of 1024, then brought back for the follow-up turn.
    srv = server.Server(model, "--max-seqs", "4", "--kv-tokens", "1024", "--host-cache-bytes", str(1 << 28), *flags)
    try:
        server.post_ok(srv, "/v1/generate", greedy(first))
        server.post_ok(srv, "/v1/generate", greedy("".join(chr(65 + (i * 11) % 26) for i in range(900)), 8))
        again = server.post_ok(srv, "/v1/generate", greedy(follow, logprobs=True))
        health = srv.get("/v1/health")["reuse"]
        assert again["reused_tokens"] == 256 and health["host"]["since_start"]["promotions"] > 0, (again["reused_tokens"], health)
        assert again["ids"] == cli_ids(model, follow, 24, cache) and again["logprobs"] == alone["logprobs"], (again["ids"], alone["ids"])
    finally:
        srv.close()
    print("deepseek4: %s: a follow-up turn forks its first turn's checkpoint brought back from the host, with the CLI's ids and a fresh server's log-probabilities" % cache)
    uncapped = [{"prompt": text, "temperature": 0, "ignore_eos": True} for text in prompts]
    for kept in ((["--state-checkpoints", "0"], []) if whole else (["--state-checkpoints", "0"],)):
        # Its blocks span 128 tokens and a request takes its prompt and 256 more as it is admitted, so a pool of 1024 admits two of the four, which then pause each other.
        srv = server.Server(model, "--max-seqs", "4", "--kv-tokens", "1024", *kept, *flags)
        try:
            alone = [server.post_ok(srv, "/v1/completions", body)["choices"][0]["text"] for body in uncapped]
            for (status, reply), body, want_text in zip(srv.together("/v1/completions", uncapped), uncapped, alone):
                assert status == 200 and reply["choices"][0]["text"] == want_text, (body, reply, want_text)
            health = srv.get("/v1/health")
            assert health["requests"]["now"]["active"] == 0 and health["requests"]["now"]["paused"] == 0 and health["pressure"]["since_start"]["pauses"] > 0, (kept, health["requests"]["now"], health["pressure"])
        finally:
            srv.close()
    print("deepseek4: %s: four uncapped requests on a pool of 1024 tokens pause and resume with the text each gives alone, %s" % (cache, "with and without checkpoints" if whole else "without checkpoints"))


def dspark(fork, directory, args, state, engram):
    """The tiny DSpark sidecar written from its golden: the digest the reference recorded, drafts that leave generate's output as it is at --draft-max 1, 4, 8 and 16, and the refusals of a sidecar whose experts are the trunk's, whose taps pass the trunk or whose shared metadata differs."""
    with open(DSPARK, encoding="utf-8") as f:
        document = json.load(f)
    _, dargs, dstate, _ = fixture_deepseek4.dspark_fixture(document)
    d = document["dspark"]
    coded = d["format_margins"]
    assert sorted(coded) == ["block", "trunk_compressed", "trunk_index_keys", "trunk_index_queries", "trunk_window", "window"], "deepseek4: the DSpark golden's coding sites are %s" % sorted(coded)
    for site, margin in coded.items():
        assert 0 < margin <= 0.5, "deepseek4: the DSpark golden's %s codings sit %.3g code spacings from a rounding point, outside (0, 0.5]" % (site, margin)
    assert len(d["window_codes"]) == dargs["n_mtp_layers"] and all(len(rows) == len(document["sequence"]) for rows in d["window_codes"]), "deepseek4: the DSpark golden's window codes are not a row a stage and position"
    sidecar = str(fixture_deepseek4.write_dflash(os.path.join(directory, "dflash.gguf"), dargs, dstate, engram))
    with open(sidecar, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == document["dspark"]["file"]["sha256"], "deepseek4: the dflash writer's file is not the reference's"
    common.check_drafts("deepseek4", fork, n=40, drafter=sidecar)
    print("deepseek4: generate with the DSpark sidecar at --draft-max 1, 4, 8 and 16, greedy and seeded, gives the ids it gives with drafts off")
    print("deepseek4: the DSpark golden's window rows sit %.3g and its block rows %.3g from a rounding point, the trunk's codings %.3g"
          % (coded["window"], coded["block"], min(v for k, v in coded.items() if k.startswith("trunk_"))))
    trunk_experts = dict(dstate)
    for e in range(args["n_routed_experts"]):
        for w in ("w1", "w2", "w3"):
            trunk_experts["mtp.0.ffn.experts.%d.%s.weight" % (e, w)] = state["layers.0.ffn.experts.%d.%s.weight" % (e, w)]
    meta, tensors = fixture_deepseek4.map_dflash(dargs, dstate, engram)
    window = dict(meta, **{"dflash.attention.sliding_window": (fixture_deepseek4.GGUF_UINT32, args["window_size"] // 2)})
    cases = [(fixture_deepseek4.map_dflash(dargs, trunk_experts, engram), "experts blk.0.ffn_gate_exps.weight are the target's blk.0.ffn_gate_exps.weight"),
             (fixture_deepseek4.map_dflash(dict(dargs, dspark_target_layer_ids=[3, 4, args["n_layers"]]), dstate, engram),
              "the DSpark taps must be ascending distinct layers of the model"),
             ((window, tensors), "metadata dflash.attention.sliding_window differs from the target's")]
    for i, ((m, t), text) in enumerate(cases):
        bad = os.path.join(directory, "bad%d.gguf" % i)
        fixture_deepseek4.write_gguf(bad, m, t)
        p = common.run_process(["generate", fork, "ab", "-n", "1", "--drafter", bad], text=True)
        assert p.returncode != 0 and text in p.stderr, "deepseek4: the sidecar %d was not refused with %r: %s" % (i, text, p.stderr)
    print("deepseek4: a sidecar whose experts are the trunk's, whose taps pass the trunk or whose window differs is refused")
    return sidecar, dargs, dstate


def dspark_serve(model, side):
    """Section 5.6 with DSpark beside the server: a prompt's first token and its log-probabilities are the plain server's, and replies greedy and seeded, alone and four at once, at --draft-max 1, 3 and 5, give the plain server's ids."""
    import server
    texts = ["helloworld", "abcabcabcabcabc", "thequickbrownfox", "0123456789"]
    bodies = [{"prompt": text, "temperature": temp, "seed": 5, "max_tokens": 40, "ignore_eos": True} for text in texts for temp in (0, 0.8)]
    first = [dict(greedy(text, 1, logprobs=True)) for text in texts]
    plain = server.Server(model, "--max-seqs", "4", *F32)
    try:
        want = [server.post_ok(plain, "/v1/generate", body)["ids"] for body in bodies]
        want_first = [server.post_ok(plain, "/v1/generate", body) for body in first]
    finally:
        plain.close()
    for k in ("1", "3", "5"):
        srv = server.Server(model, "--max-seqs", "4", "--drafter", side, "--draft-max", k, *F32)
        try:
            for body, reply in zip(first, want_first):
                got = server.post_ok(srv, "/v1/generate", body)
                assert [got[x] for x in ("ids", "logprobs", "top_logprobs")] == [reply[x] for x in ("ids", "logprobs", "top_logprobs")], (k, body["prompt"], got, reply)
            server.alone_and_together(srv, bodies)
            for body, ids in zip(bodies, want):
                assert server.post_ok(srv, "/v1/generate", body)["ids"] == ids, (k, body)
            health = srv.get("/v1/health")["drafting"]
            assert health["since_start"]["drafted"] > 0 and health["since_start"]["failed"] == 0, health
        finally:
            srv.close()
    print("deepseek4: a server with the DSpark sidecar gives the plain server's first-token log-probabilities, and its ids greedy and seeded, alone and four at once, at --draft-max 1, 3 and 5")


def run(require=False):
    # The golden's two output groups split at width 2 alone; the group-legal file below holds widths 2 and 4.
    if os.environ.get("LLMX_TENSOR_WIDTH", "1") not in ("1", "2") and common.tensor_width_skip("deepseek4"):
        return common.SKIPPED
    with open(GOLDEN, encoding="utf-8") as f:
        document = json.load(f)
    args, state, engram = fixture_deepseek4.fixture_from_document(document)
    with tempfile.TemporaryDirectory(prefix="llmx_deepseek4_") as directory:
        paths = {layout: fixture_deepseek4.write_deepseek41(os.path.join(directory, layout + ".gguf"), args, state, engram, layout)
                 for layout in fixture_deepseek4.LAYOUTS}
        fork = str(paths["fork"])
        # A device whose backend lacks one of the model's ops refuses the file at load by the op's name, and nothing is compared.
        if any(d != "cpu" for d in os.environ.get("LLMX_DEVICE", "cpu").split(",")):
            p = common.run_process(["logits", fork, "a"] + F32, text=True)
            if p.returncode == 1 and DEVICE in p.stderr:
                print("deepseek4: SKIP - the device refuses the model at load (%s), so nothing was compared" % p.stderr.strip().splitlines()[-1])
                return common.SKIPPED
        gate(fork, directory, document)
        refused(str(paths["upstream"]), UPSTREAM)
        print("deepseek4: the upstream layout is refused by its key")
        refused(fork, "--cache-type-k: this model keeps its cache in native or f32, not f16", ["--cache-type-k", "f16", "--cache-type-v", "f32"])
        refused(fork, "--cache-type-v: this model keeps its cache in native or f32, not f16", ["--cache-type-k", "f32", "--cache-type-v", "f16"])
        refused(fork, "--cache-type-k and --cache-type-v: this model keeps one cache type for both sides, not f32 and native", ["--cache-type-k", "f32", "--cache-type-v", "native"])
        print("deepseek4: an f16 cache side and two different sides refused by the flags; native and the default print the f32 runs' logits")
        served = fixture_deepseek4.write_deepseek41(os.path.join(directory, "serve.gguf"), dict(args, max_seq_len=1024), state, engram, "fork")
        check_serve(served, "f32", True)
        check_serve(served, "native", False)
        check_scratch(served, require)
        common.check_drafts("deepseek4", fork)
        print("deepseek4: generate with --drafter lookup at 1, 4, 8 and 16 drafts, greedy and seeded, gives the ids it gives with drafts off")
        sidecar, dspark_args, dspark_state = dspark(fork, directory, args, state, engram)
        served_side = str(fixture_deepseek4.write_dflash(os.path.join(directory, "serve-dflash.gguf"), dict(dspark_args, max_seq_len=1024), dspark_state, engram))
        dspark_serve(served, served_side)
        check_serve(served, "f32", False, ["--drafter", served_side, "--draft-max", "5"])
        print("deepseek4: with the DSpark sidecar, forks of donors and of a checkpoint brought back from the host and pauses give the CLI's ids without drafts")
        metadata, tensors = fixture_deepseek4.map_deepseek41(args, state, engram, "fork")
        p = common.run_process(["info", fork], text=True)
        assert p.returncode == 0, "deepseek4: info failed: %s" % p.stderr
        listed = set(line.split(" = ", 1)[0].strip() for line in p.stdout.splitlines() if " = " in line)
        names = set(line.split()[1] for line in p.stdout.splitlines() if line.startswith("  ") and " shape=" in line)
        assert set(metadata) <= listed, "deepseek4: info lists no %s" % sorted(set(metadata) - listed)
        assert names == set(t[0] for t in tensors), "deepseek4: info's tensors differ from the writer's: %s" % sorted(names ^ set(t[0] for t in tensors))
        print("deepseek4: info lists the fork file's %d keys and %d tensors" % (len(metadata), len(tensors)))
        tool = tool_path()
        if not os.path.exists(tool):
            assert not require, "deepseek4: %s not found beside the executable" % tool
            print("deepseek4: the C++ list not compared - %s not found beside the executable" % tool)
            return True
        p = subprocess.run([tool, GOLDEN, "--gguf", fork], capture_output=True, encoding="utf-8", errors="replace", timeout=600)
        assert p.returncode == 0, "deepseek4: the writer's fork file is not the test's: %s%s" % (p.stdout, p.stderr)
        print("deepseek4: the writer's fork file holds the arch-deepseek4 test's keys and tensors")
        q = subprocess.run([tool, DSPARK, "--dspark", fork, sidecar], capture_output=True, encoding="utf-8", errors="replace", timeout=600)
        assert q.returncode == 0, "deepseek4: DSpark's drafts are not the reference's: %s%s" % (q.stdout, q.stderr)
        print("\n".join("deepseek4: " + line.split(": ", 1)[1] for line in q.stdout.splitlines() if line.startswith("arch-deepseek4: DSpark")))
        # Four output groups make every split role divide at widths 2 and 4 (docs/DEEPSEEK4.md, section 7).
        gargs, gstate, gengram = fixture_deepseek4.fixture_from_document(document, o_groups=4)
        grouped = str(fixture_deepseek4.write_deepseek41(os.path.join(directory, "groups.gguf"), gargs, gstate, gengram, "fork"))
        g = subprocess.run([tool, GOLDEN, "--groups", grouped], capture_output=True, encoding="utf-8", errors="replace", timeout=600)
        assert g.returncode == 0, "deepseek4: tensor groups: %s%s" % (g.stdout, g.stderr)
        for line in (p.stdout + g.stdout).splitlines():
            if line.startswith(("arch-deepseek4: layer ", "arch-deepseek4: a 512-row", "arch-deepseek4: the writer's file over", "arch-deepseek4: a 12", "arch-deepseek4: one CPU", "arch-deepseek4: a fork", "arch-deepseek4: two CPUs", "arch-deepseek4: an abort", "arch-deepseek4: a group", "arch-deepseek4: every member")):
                print("deepseek4: " + line.split(": ", 1)[1])
    return True
