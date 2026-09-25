import json
import math
from pathlib import Path
import re
import tempfile

import common
import f32


# Thread counts are the CPU backend's, and a device backend reports none, so every command here names the CPU instead of the configured device.
def invoke(args, data=None):
    p = common.run_process(args + ["--device", "cpu"], input=data, timeout=30)
    assert p.returncode == 0, (args, p.returncode, p.stderr)
    return p.stdout.replace(b"\r\n", b"\n"), p.stderr


def check_perplexity_threads(model, automatic, weights):
    golden = json.loads((Path(__file__).parent / "data/baseline_f32.json").read_text())
    fixture = next(case for case in golden["fixtures"] if not case["tied"])
    assert golden["config"] == f32.CONFIG
    assert fixture["weights_sha256"] == f32.weight_hash(weights)
    case = fixture["perplexity"][0]
    worst = 0.0
    checked = 0
    for count in (None, 0, 1, 4):
        for batch in (None, 0, 1, 3):
            for alias in (("--threads-batch", "-tb") if batch is not None else ("--threads-batch",)):
                for per_token in (False, True):
                    flags = ["--verbose", "--ubatch", "3",
                             "--cache-type-k", "f32", "--cache-type-v", "f32",
                             "-c", str(case["context"])]
                    if count is not None:
                        flags += ["--threads", str(count)]
                    if batch is not None:
                        flags += [alias, str(batch)]
                    if per_token:
                        flags += ["--per-token"]
                    out, err = invoke(["perplexity", str(model), f32.TEXTS[-1]] + flags)
                    expected = count or automatic
                    if not per_token:
                        expected = batch or expected
                    counts = re.findall(rb"^threads: (prefill|decode) (\d+)\r?$", err, re.M)
                    phase = b"decode" if per_token else b"prefill"
                    assert counts == [(phase, str(expected).encode())], (count, batch, alias, per_token, counts, expected)
                    error = abs(float(common.perplexity_fields(out.decode("utf-8"))["mean NLL"]) - case["mean_nll"])
                    assert math.isfinite(error) and error < 1e-5, (count, batch, per_token, error)
                    worst = max(worst, error)
                    checked += 1
    print("threads: perplexity effective counts/aliases and HF NLL, %d cases, max error %.8f  [ok]" % (checked, worst))


def run():
    bench = ["bench", "--size", "32", "--iters", "1", "--p", "1", "--n", "1"]
    automatic = None
    for count in (None, 0, 1, 4):
        flags = [] if count is None else ["--threads", str(count)]
        out, _ = invoke(bench + flags)
        match = re.search(rb"^bench: threads (\d+)$", out, re.M)
        assert match, out
        actual = int(match[1])
        if count is None:
            automatic = actual
            assert 1 <= automatic <= 64, automatic
        assert actual == (count or automatic), (count, actual, automatic)

    fixture = json.loads((Path(__file__).parent / "data/baseline_chat.json").read_text())
    case = fixture["cases"][0]
    spec = case["spec"]
    weights = f32.tensors(False)
    assert fixture["weights_sha256"] == f32.weight_hash(weights)
    assert fixture["config"] == f32.CONFIG
    assert spec["template"] == "{{ messages[-1]['content'] }}"
    assert spec["max_tokens"] == 1
    with tempfile.TemporaryDirectory(prefix="llmx_threads_") as directory:
        model = Path(directory) / "threads.gguf"
        f32.write_model(model, weights, spec["template"])
        check_perplexity_threads(model, automatic, weights)
        for count in (None, 0, 1, 4):
            for batch in (None, 0, 1, 3):
                flags = ["--temp", "0", "-n", "1", "--verbose", "--ubatch", "3"]
                if count is not None:
                    flags += ["--threads", str(count)]
                if batch is not None:
                    flags += ["--threads-batch", str(batch)]
                decode = count or automatic
                prefill = batch or decode
                for command in ("generate", "chat"):
                    args = [command, str(model)]
                    if command == "generate":
                        args += [spec["inputs"][0]]
                        data, turns = None, 1
                    else:
                        args += ["--system", ""]
                        data = ("\n".join(spec["inputs"]) + "\n").encode()
                        turns = len(spec["inputs"])
                    out, err = invoke(args + flags, data)
                    counts = re.findall(rb"^threads: (prefill|decode) (\d+)\r?$", err, re.M)
                    expected = [(b"prefill", str(prefill).encode()),
                                (b"decode", str(decode).encode())] * turns
                    assert counts == expected, (command, count, batch, counts, expected)
                    replies = b"".join(bytes(t["reply_ids"]) + b"\n" for t in case["turns"][:turns])
                    if command == "chat":
                        assert out == b"Chat ready (type your message; Ctrl+C to quit)\n" + replies, out
                    else:
                        generated = re.search(rb"^pp: [^\n]*\n(.*?)^tg: ", out, re.M | re.S)
                        assert generated and generated[1] == replies, out
    print("threads: auto/explicit counts, prefill restore, follow-up chat and HF replies  [ok]")
    return True
