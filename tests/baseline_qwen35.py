import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import common
from common import device_args, require
from baseline_8b import file_sha256
from qwen35 import refusal
import spec_decode

# The qwen35 family's real-model check: llmx on a pinned Qwen3.5 file against the HF goldens of its model, which tools/gen_baseline.py qwen35 made with HF's full forward in float32 on the pinned checkpoint.
# The tokenizer cases are the family's golden, since every checkpoint pinned here has the same tokenizer.json.
# tests/baseline.py runs the gate's qwen35 files at 512-token windows; the 4096-token windows and the files outside the gate run by hand through main().

DATA = Path(__file__).resolve().parent / "data"
TOKENIZER_GOLDEN = "baseline_tokenizer_qwen35.json"
VOCAB_SIZE = 248320
MODEL_CONTEXT = 262144
PPL_GOLDENS = {512: "baseline_perplexity.json", 4096: "baseline_perplexity_4096.json"}

# Every golden this check reads, with the SHA-256 of its text with LF line endings.
GOLDEN_SHA256 = {
    "baseline_tokenizer_qwen35.json": "fdf0bcfb75d77345f1321fcd6c98aa0b48abda41306371a499b659a8dce3cb6a",
    "qwen35-0.8b/baseline_logits.json": "faa5d386b9eb0842e3f01f442fdd5478a744026d5606e6cff11f302a9ec3525b",
    "qwen35-0.8b/baseline_chat.json": "9d0c6903633934efcc260ae38de4185f99c10824801957b11efa4fbd6acba3b1",
    "qwen35-0.8b/baseline_perplexity.json": "42c5ef92ec3da19b8ef67b81aa5cfafd55b7c9a9413a76c4e65df9a57d8e98f4",
    "qwen35-0.8b/baseline_perplexity_4096.json": "6c8bd8601709199c5ddde46d637ece9242efe03222a403a808ab0dca0a6c789a",
    "qwen35-4b/baseline_logits.json": "2a2fbb27a70a6344835e751432bcc5a47aae73e542cb1e11562de09a4ef7d361",
    "qwen35-4b/baseline_chat.json": "0fc52ab64ddb855c00bd441a702d91d7fb2c40bcafa6320636baf21740f5bab8",
    "qwen35-4b/baseline_perplexity.json": "bea1ad500f60333c546c328523efb0d9edf24deb7d8f99decb1b6e67e3359cbb",
    "qwen35-4b/baseline_perplexity_4096.json": "e19d2510d6acccb10fb292c530899e47843d7caff97915e1a93cb9d544af01e4",
}

# Each file's bounds against its model's goldens, set from llmx's first measurement on it, the largest NLL delta over both window lengths, both cache types and both ways of scoring plus a margin (docs/ASSETS.md); a file without them is measured and fails.
# The 0.8B Q4_K_M has none: its own quantization moves HF's top-1 on two prompts, which its file-exact goldens show (docs/STATUS.md).
BOUNDS = {
    "Qwen3.5-0.8B-Q8_0.gguf": {"top5_overlap": 5, "continuous_nll": 0.02, "window_nll": 0.02},
    "Qwen3.5-4B-Q4_K_M.gguf": {"top5_overlap": 4, "continuous_nll": 0.07, "window_nll": 0.08},
}
# A file against its file-exact goldens is held to this file's bounds whatever its type, since the format's loss is on both sides.
FILE_EXACT_BOUNDS = "Qwen3.5-0.8B-Q8_0.gguf"
MAX_PLAUSIBLE_LOGIT = 100.0
# What a file without bounds is measured at: every rule of the validators except the bounds themselves.
MEASURE_ONLY = {"top5_overlap": 0, "max_abs_logit": MAX_PLAUSIBLE_LOGIT, "continuous_nll": math.inf, "window_nll": math.inf}


def ids_sha256(ids):
    """The digest a golden keeps in place of a long id list: SHA-256 of the ids in decimal, joined by commas."""
    return hashlib.sha256(",".join(map(str, ids)).encode("ascii")).hexdigest()


def wiki_excerpt(chars):
    """The first `chars` characters of tests/data/wiki.test.raw, its line endings read as LF, as every perplexity golden takes its text."""
    with open(DATA / "wiki.test.raw", encoding="utf-8") as f:
        return f.read(chars)


def load_golden(name):
    """The golden `name`, a path under tests/data, refused unless its text has the pinned SHA-256."""
    # Git may check JSON out with CRLF; the generator writes LF.
    text = (DATA / name).read_text(encoding="utf-8")
    require(name in GOLDEN_SHA256 and hashlib.sha256(text.encode("utf-8")).hexdigest() == GOLDEN_SHA256[name],
            "qwen35 golden changed or not pinned: " + name)
    return json.loads(text)


def goldens_for(file):
    """The directory of the goldens made for the pinned file `file`, which lists it in each of them, and those goldens by name."""
    for directory in sorted({name.split("/")[0] for name in GOLDEN_SHA256 if "/" in name}):
        docs = {name.split("/")[1]: load_golden(name) for name in GOLDEN_SHA256 if name.startswith(directory + "/")}
        if all(file in doc["gguf_files"] for doc in docs.values()):
            return directory, docs
        require(not any(file in doc["gguf_files"] for doc in docs.values()), "%s lists %s in only some of its goldens" % (directory, file))
    raise ValueError("no qwen35 goldens are made for " + file)


def check_chat_template(model, expected):
    """The chat template in the file `model` must be the one the chat golden was rendered with."""
    template = spec_decode.GGUF(model).value("tokenizer.chat_template")
    require(template is not None, "the model file has no chat template")
    digest = hashlib.sha256(template.encode("utf-8")).hexdigest()
    require(digest == expected, "the file's chat template has SHA-256 %s, the golden's %s" % (digest, expected))
    return {"sha256": digest}


def ppl_text(doc):
    text = wiki_excerpt(doc["chars"])
    require(hashlib.sha256(text.encode("utf-8")).hexdigest() == doc["text_sha256"], "PPL text hash differs")
    return text


def check_ids_digest(output, n_tokens, digest):
    ids = common.parse_ids(output)
    require(len(ids) == n_tokens and ids_sha256(ids) == digest, "token IDs differ from HF: %d tokens, SHA-256 %s" % (len(ids), ids_sha256(ids)))
    return {"tokens": len(ids)}


def model_goldens(model, file_exact=None):
    """The goldens the pinned qwen35 file `model` is held to, by name, and their bounds, None until they are set.
    They are its model's goldens, or with `file_exact` the logit and 512-token perplexity goldens tools/gen_baseline.py file-exact made from this very file, at the Q8_0 file's bounds whatever the file's type."""
    spec = pinned(model)
    if not file_exact:
        return goldens_for(spec["file"])[1], BOUNDS.get(spec["file"])
    docs = {}
    for name in ("baseline_logits.json", PPL_GOLDENS[512]):
        doc = json.loads(Path(file_exact, name).read_text(encoding="utf-8"))
        require(doc.get("weights", {}).get("sha256") == spec["sha256"], "the goldens in %s were not made from %s" % (file_exact, spec["file"]))
        docs[name] = doc
    return docs, BOUNDS.get(FILE_EXACT_BOUNDS)


def check_file(model, context, run, check, write, docs, bounds):
    """Every check of the pinned qwen35 file `model` against the goldens `docs` at `bounds`, with perplexity windows of `context` tokens.
    `run(label, arguments)` runs llmx and gives (exit code, output), `check(label, validator, *arguments)` records one check, and `write(name, data)` keeps a file beside the results and gives its path.
    Returns "skip" when llmx does not run the architecture here (qwen35.REFUSALS), "unbounded" when there are no bounds yet, which measures every check at MEASURE_ONLY, and "checked" otherwise."""
    limits = dict(bounds or MEASURE_ONLY, max_abs_logit=MAX_PLAUSIBLE_LOGIT)

    def output(label, arguments):
        rc, out = run(label, arguments)
        require(rc == 0, "%s failed (exit %d): %s" % (label, rc, out.strip()[-200:]))
        return out

    for index, case in enumerate(load_golden(TOKENIZER_GOLDEN)["cases"]):
        label = "tokenizer-%02d" % index
        check(label, common.check_ids, output(label, ["tokenize", model, case["text"]]), case["ids"])
    chats = []
    if "baseline_chat.json" in docs:
        check("chat-template", check_chat_template, model, docs["baseline_chat.json"]["template_sha256"])
        chats = docs["baseline_chat.json"]["cases"]
    for index, case in enumerate(chats):
        label = "chat-%02d-ids" % index
        check(label, common.check_ids, output(label, ["tokenize", model, case["text"]]), case["token_ids"])
    for index, case in enumerate(docs["baseline_logits.json"]["cases"]):
        label = "logits-%02d-ids" % index
        check(label, common.check_ids, output(label, ["tokenize", model, case["text"]]), case["token_ids"])
    doc = docs[PPL_GOLDENS[context]]
    text = ppl_text(doc)
    check("ppl-ids", check_ids_digest, output("ppl-ids", ["tokenize", model, text]), doc["n_tokens"], doc["token_ids_sha256"])

    prompts = [("logits-%02d" % i, case, ["logits", model, case["text"]]) for i, case in enumerate(docs["baseline_logits.json"]["cases"])]
    for index, case in enumerate(chats):
        prompts.append(("chat-%02d" % index, case, ["logits", model, "--file", write("chat-%02d.txt" % index, case["text"].encode("utf-8"))]))
    for label, case, arguments in prompts:
        rc, out = run(label, arguments + ["--top", "10", "--threads", "6"])
        if refusal(rc, out):
            return "skip"
        require(rc == 0, "%s failed (exit %d): %s" % (label, rc, out.strip()[-200:]))
        check(label, common.check_logits, out, case, VOCAB_SIZE, limits)
    excerpt = write("excerpt.txt", text.encode("utf-8"))
    for index, case in enumerate(common.ppl_cases(doc)):
        for mode in common.PPL_MODES:
            label = "ppl-%02d" % index + ("-per-token" if mode == "per-token" else "")
            check(label, common.check_ppl, output(label, common.ppl_command(model, excerpt, case, mode)), case, doc["n_tokens"], MODEL_CONTEXT, limits)
    return "checked" if bounds else "unbounded"


def pinned(model):
    """The entry of tests/data/fixtures.json for the qwen35 file `model`, refused unless the file has its SHA-256."""
    from baseline import pinned_fixture
    spec = pinned_fixture(os.path.basename(model))
    require(spec is not None and spec["family"] == "qwen35", "%s is not a pinned qwen35 fixture" % model)
    require(file_sha256(Path(model)) == spec["sha256"], "%s does not have the pinned SHA-256" % model)
    return spec


def run_hosted():
    """The 512-token check of every qwen35 fixture of the gate on disk, which the hosted HF job downloads, in the suite's form: a line for each file, and False on the first failure."""
    import tempfile
    from baseline import PINNED, snapshot_path
    for spec in (s for s in PINNED if s["family"] == "qwen35" and s["gate"]):
        name = "baseline-qwen35[%s]" % spec["file"]
        if os.environ.get("LLMX_BASELINE_GGUF"):
            print("%s: SKIP - LLMX_BASELINE_GGUF selects one Qwen3 fixture" % name)
            continue
        model = snapshot_path(spec["repo"], spec["revision"], spec["file"])
        model = str(model) if model.exists() else None
        if not model:
            print("%s: SKIP - fixture model not on disk" % name)
            continue
        failures, count = [], [0]

        def run(label, arguments):
            return common.run(arguments)

        def check(label, validator, *arguments):
            count[0] += 1
            try:
                validator(*arguments)
            except (ValueError, OverflowError) as error:
                failures.append("%s: %s" % (label, error))

        with tempfile.TemporaryDirectory(prefix="llmx_qwen35_") as directory:
            def write(file, data):
                path = os.path.join(directory, file)
                with open(path, "wb") as f:
                    f.write(data)
                return path

            try:
                status = check_file(model, 512, run, check, write, *model_goldens(model))
            except ValueError as error:
                failures.append(str(error))
                status = "failed"
        if failures:
            print("%s: %d of %d checks fail:" % (name, len(failures), count[0]))
            for failure in failures:
                print("    " + failure)
            return False
        if status == "skip":
            print("%s: SKIP - llmx does not run the qwen35 architecture here (%d tokenizer and chat id checks pass)" % (name, count[0]))
        elif status == "unbounded":
            print("%s: FAIL - no bounds in tests/baseline_qwen35.py; set them from `tests/baseline_qwen35.py --model` on this file" % name)
            return False
        else:
            print("%s: %d checks  [ok]" % (name, count[0]))
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description="The qwen35 real-model HF check of one pinned file, with a report; no downloads.")
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="new directory for raw output and report")
    parser.add_argument("--context", type=int, choices=sorted(PPL_GOLDENS), default=512, help="the perplexity windows to score, in tokens")
    parser.add_argument("--device", help="run the commands that take --device on this device or comma-separated list, e.g. vulkan:0")
    parser.add_argument("--file-exact", metavar="DIR", help="hold the file to the goldens tools/gen_baseline.py file-exact made from it, at 512-token windows")
    args = parser.parse_args(argv)
    if args.file_exact and args.context != 512:
        parser.error("file-exact goldens score 512-token windows")
    if args.device:
        os.environ["LLMX_DEVICE"] = args.device
    # The file is known by its name, which a link into the HF cache keeps and its target does not.
    exe, model, out = args.exe.resolve(), Path(os.path.abspath(args.model)), args.output_dir.resolve()
    if out.exists():
        parser.error("output directory already exists; use a new path")
    out.mkdir(parents=True)
    report = {"status": "running", "model": str(model), "executable": str(exe), "device": args.device or "cpu",
              "context": args.context, "commands": [], "checks": []}

    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="ascii")

    def run(label, arguments):
        command = [str(exe)] + device_args(arguments)
        record = {"label": label, "command": command, "status": "running"}
        report["commands"].append(record)
        save()
        print(label, flush=True)
        try:
            result = subprocess.run(command, capture_output=True, timeout=3600)
            stdout, stderr, rc = result.stdout, result.stderr, result.returncode
            record.update(status="exited", returncode=rc)
        except subprocess.TimeoutExpired as error:
            stdout, stderr, rc = error.stdout or b"", error.stderr or b"", None
            record.update(status="timeout", timeout_seconds=3600)
        (out / (label + ".stdout")).write_bytes(stdout)
        (out / (label + ".stderr")).write_bytes(stderr)
        save()
        return rc, (stdout + stderr).decode("utf-8", "replace") if rc else stdout.decode("utf-8")

    def check(label, validator, *arguments):
        try:
            result = dict(status="pass", **validator(*arguments))
        except (ValueError, OverflowError) as error:
            result = dict(status="fail", error=str(error))
        report["checks"].append(dict(label=label, **result))
        save()

    def write(name, data):
        (out / name).write_bytes(data)
        return str(out / name)

    save()
    try:
        report["executable_sha256"] = file_sha256(exe)
        report["model_sha256"] = file_sha256(model)
        rc, version = run("version", ["--version"])
        require(rc == 0, "llmx --version failed")
        report["version"] = version.strip()
        docs, bounds = model_goldens(str(model), args.file_exact)
        report["bounds"] = bounds
        status = check_file(str(model), args.context, run, check, write, docs, bounds)
        failures = [item["label"] for item in report["checks"] if item["status"] == "fail"]
        require(not failures, "failed checks: " + ", ".join(failures))
        if status == "skip":
            report["status"] = "skip"
            print("qwen35 HF check SKIP - llmx does not run the qwen35 architecture here; %d tokenizer and chat id checks pass" % len(report["checks"]), flush=True)
        elif status == "unbounded":
            report["status"] = "unbounded"
            print("qwen35 HF check FAIL - %s has no bounds in tests/baseline_qwen35.py; %d checks measured, see report.json"
                  % (model.name, len(report["checks"])), flush=True)
        else:
            report["status"] = "pass"
            print("qwen35 HF check PASS: %d checks" % len(report["checks"]), flush=True)
    except (OSError, ValueError, OverflowError) as error:
        report.update(status="fail", error=str(error))
        print("qwen35 HF check FAIL: " + str(error), file=sys.stderr, flush=True)
    finally:
        save()
    return 0 if report["status"] in ("pass", "skip") else 1


if __name__ == "__main__":
    sys.exit(main())
