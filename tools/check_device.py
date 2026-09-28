"""Compare a model's device execution with CPU, bounded by an existing-type control on the same architecture.

Captures retain every full-vocabulary row of the pinned HF excerpt in batched and per-token execution, then 64 argmax steps after prefill, without stopping at EOS.
This supplements independent HF correctness; it does not measure format quality or replace the HF gate.
"""
import argparse
import array
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import common
from baseline_8b import file_sha256

FIXTURE = Path(common.ROOT) / "tests/data/baseline_perplexity.json"



def captured_rows(prefix, phase, rows, vocab):
    """Native binary32 rows from this host, with neither truncation nor trailing bytes."""
    path = Path(str(prefix) + "." + phase + ".bin")
    common.require(type(rows) is int and rows > 0 and type(vocab) is int and vocab >= 6, "invalid capture dimensions")
    common.require(path.stat().st_size == rows * vocab * 4, "capture size differs from its row count: " + str(path))
    with path.open("rb") as stream:
        for _ in range(rows):
            row = array.array("f")
            row.frombytes(stream.read(vocab * 4))
            common.require(len(row) == vocab, "capture truncated during reading")
            yield row


def capture_metadata(path, tokens, version):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    common.require(set(doc) == {"version", "architecture", "vocab", "tokens", "greedy", "storage_types"}, "unexpected capture metadata")
    common.require(doc["version"] == version and isinstance(doc["architecture"], str), "capture build identity or architecture invalid")
    common.require(type(doc["vocab"]) is int and doc["vocab"] >= 6, "invalid capture vocabulary")
    common.require(doc["tokens"] == tokens and all(type(i) is int for i in doc["tokens"]), "capture token IDs differ")
    common.require(len(doc["greedy"]) == 64 and all(type(i) is int and 0 <= i < doc["vocab"] for i in doc["greedy"]), "invalid greedy capture IDs")
    common.require(isinstance(doc["storage_types"], list) and doc["storage_types"] and
                   all(type(i) is int and i >= 0 for i in doc["storage_types"]) and
                   doc["storage_types"] == sorted(set(doc["storage_types"])), "invalid capture storage types")
    return doc


def compare_pair(cpu, device, tokens, limits=None):
    common.require(cpu["metadata"]["architecture"] == device["metadata"]["architecture"] and
                   cpu["metadata"]["vocab"] == device["metadata"]["vocab"], "capture architecture or vocabulary differs")
    vocab = cpu["metadata"]["vocab"]
    result = {}
    for phase in ("batched", "decode"):
        a = captured_rows(cpu["prefix"], phase, len(tokens), vocab)
        b = captured_rows(device["prefix"], phase, len(tokens), vocab)
        result[phase] = common.check_device_rows(a, b, tokens, None if limits is None else limits[phase]["max_logit_gap"])
    result["greedy"] = common.check_device_greedy(
        captured_rows(cpu["prefix"], "greedy", 64, vocab), captured_rows(device["prefix"], "greedy", 64, vocab),
        cpu["metadata"]["greedy"], device["metadata"]["greedy"])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, default=Path(common.EXE), help="llmx executable, with llmx-model-logits beside it")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True, help="existing weight type on the same model or architecture, selected before measuring")
    parser.add_argument("--device", required=True, help="device or device list to compare with CPU")
    parser.add_argument("--output", type=Path, required=True, help="new directory for commands, raw captures and report.json")
    parser.add_argument("--cache-type", choices=("f16", "f32"), default="f16")
    parser.add_argument("--ubatch", type=int, default=512)
    parser.add_argument("--n-cpu-moe", type=int, default=0, help="device-side expert offload count; -1 for all; CPU references use 0")
    args = parser.parse_args(argv)
    if args.ubatch < 1 or args.n_cpu_moe < -1:
        parser.error("ubatch must be positive and n-cpu-moe at least -1")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "scope": "device versus CPU, supplemental to independent HF", "commands": [],
              "cache": args.cache_type, "ubatch": args.ubatch, "threads": 6, "byte_order": sys.byteorder,
              "device": args.device, "n_cpu_moe": args.n_cpu_moe}

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def command(label, cmd):
        record = {"label": label, "argv": list(map(str, cmd)), "started": datetime.now(timezone.utc).isoformat()}
        report["commands"].append(record)
        save()
        with (output / (label + ".stdout")).open("wb") as stdout, (output / (label + ".stderr")).open("wb") as stderr:
            proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr)
            record["pid"] = proc.pid
            save()
            try:
                record["exit"] = proc.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                record["timeout"] = True
                raise
        record["completed"] = datetime.now(timezone.utc).isoformat()
        save()
        common.require(record["exit"] == 0, "command failed: " + label)
        return output / (label + ".stdout")

    try:
        exe = args.exe.resolve()
        tool = exe.with_name("llmx-model-logits" + exe.suffix)
        models = {"control": args.control.resolve(), "candidate": args.model.resolve()}
        report["hashes"] = {str(p): file_sha256(p) for p in [exe, tool, FIXTURE, *models.values()]}
        common.require(report["hashes"][str(models["control"])] != report["hashes"][str(models["candidate"])], "control and candidate are the same model file")
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        tokens = fixture["token_ids"]
        common.require(len(tokens) == fixture["n_tokens"] and len(tokens) >= 2, "fixture token count differs")
        ids = output / "tokens.txt"
        ids.write_text(" ".join(map(str, tokens)), encoding="ascii")
        version = command("version", [str(exe), "--version"]).read_text().strip()
        common.require(version.startswith("llmx "), "unrecognized executable version")
        report["version"] = version
        pairs = {}
        for name, model in models.items():
            tokenized = command(name + "-tokens", [str(exe), "tokenize", str(model), fixture["text"]])
            common.check_ids(tokenized.read_text(encoding="utf-8"), tokens)
            pair = []
            for role, device, experts in [("cpu", "cpu", 0), ("device", args.device, args.n_cpu_moe)]:
                prefix = output / (name + "-" + role)
                stdout = command(prefix.name, [str(tool), str(model), str(ids), str(prefix), device,
                                               args.cache_type, str(args.ubatch), str(experts)])
                metadata = capture_metadata(stdout, tokens, version[5:])
                captured = {"prefix": str(prefix), "metadata": metadata,
                            "hashes": {phase: file_sha256(Path(str(prefix) + "." + phase + ".bin")) for phase in ("batched", "decode", "greedy")}}
                pair.append(captured)
            if name == "candidate":
                common.require(pair[0]["metadata"]["architecture"] == pairs["control"][0]["metadata"]["architecture"], "control must have the same architecture")
            pairs[name] = pair
            report[name] = {"captures": pair, "comparison": compare_pair(*pair, tokens, None if name == "control" else report["control"]["comparison"])}
            save()
            print(name + ": passed", flush=True)
        report["status"] = "pass"
    except Exception as error:
        report["status"] = "fail"
        report["error"] = str(error)
    finally:
        report["completed"] = datetime.now(timezone.utc).isoformat()
        save()
    print(report["status"] + ": " + str(output / "report.json"), flush=True)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
