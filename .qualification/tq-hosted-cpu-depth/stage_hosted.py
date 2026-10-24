#!/usr/bin/env python3
"""Stage the audited TQ build/probes and frozen inputs for hosted CPU captures."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback


ROOT = Path(__file__).resolve().parent
PINS = json.loads((ROOT / "artifact-pins.json").read_text(encoding="utf-8"))
TEMPLATE = ROOT / "bundle-template"
INPUT_TEMPLATE = ROOT / "input-template"
FIXTURE_TEMPLATE = ROOT / "fixtures"
MODEL_REPO = "Majipa/Qwen3-0.6B-GGUF-Quantized"
OVERLAY_PARENT = PINS["candidate_revision"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def safe_extract(archive_path, destination):
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive_path, "r:gz") as archive:
        names = [member.name for member in archive.getmembers()]
        assert len(names) == len(set(names)), "duplicate build archive entries"
        for name in names:
            rel = PurePosixPath(name)
            assert not rel.is_absolute() and ".." not in rel.parts, name
        archive.extractall(destination, filter="data")


def verify_build(build_root, receipt_path):
    pin = PINS["matched_build"]
    assert sha(receipt_path) == pin["candidate_build_receipt_sha256"]
    receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    assert receipt["status"] == "passed"
    assert receipt["revision"] == PINS["candidate_revision"]
    assert receipt["image_id"] == pin["image_id"]
    assert receipt["attach_exit_code"] == 0
    assert receipt["container_state"]["ExitCode"] == 0
    assert receipt["container_state"]["OOMKilled"] is False
    assert receipt["removed"] is True
    rows = receipt["build_artifacts"]
    assert len(rows) == 850
    for item in rows:
        rel = PurePosixPath(item["path"])
        assert not rel.is_absolute() and ".." not in rel.parts
        path = build_root.joinpath(*rel.parts)
        assert path.is_file() and path.stat().st_size == item["bytes"]
        assert sha(path) == item["sha256"], item["path"]
    assert sha(build_root / "llmx") == pin["candidate_cli_sha256"]
    assert sha(build_root / "libllmx-vulkan.a") == pin["candidate_library_sha256"]
    return receipt


def verify_probes(probe_artifact_root):
    pin = PINS["probe_build"]
    execution_path = probe_artifact_root / "probe-build" / "execution.json"
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    assert sha(execution_path) == pin["execution_sha256"]
    assert execution["status"] == "passed"
    assert execution["candidate_revision"] == PINS["candidate_revision"]
    assert execution["candidate_tree"] == PINS["candidate_tree"]
    assert execution["hosted_image_id"] == PINS["matched_build"]["image_id"]
    assert execution["loaded_image_id"] == execution["hosted_image_id"]
    assert execution["container_attach_exit_code"] == 0
    assert execution["container_state_final"]["OOMKilled"] is False
    assert execution["container_remove_exit_code"] == 0
    assert execution["container_absent_after_cleanup"] is True
    root = probe_artifact_root / "probe-build" / "probes"
    expected = {"default-probe": pin["default_probe_sha256"],
                "f32-probe": pin["f32_probe_sha256"],
                "probe-build.json": pin["probe_build_receipt_sha256"]}
    for name, digest in expected.items():
        assert sha(root / name) == digest, name
    record = json.loads((root / "probe-build.json").read_text(encoding="utf-8"))
    assert record["library_sha256"] == PINS["matched_build"]["candidate_library_sha256"]
    assert record["probes"] == {"default": pin["default_probe_sha256"],
                                 "f32": pin["f32_probe_sha256"]}
    return root


def make_runtime_bundle(bundle, paths, validate_staged_inputs=False):
    shutil.copytree(TEMPLATE, bundle, ignore=shutil.ignore_patterns("__pycache__"))
    plan_path = bundle / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["candidate_source"]["source_path"] = str(paths["source"])
    plan["candidate_source"]["build_path"] = str(paths["build"])
    plan["candidate_source"]["build_receipt_path"] = str(paths["build_receipt"])
    plan["model_root"] = str(paths["models"])
    plan["input_root"] = str(paths["inputs"])
    plan["fixture_root"] = str(paths["fixtures"])
    plan["reference_source"] = str(paths["reference_source"])
    plan["probe_owner"]["probe_build_root"] = str(paths["probes"])
    plan["probe_owner"]["probe_build_receipt_path"] = str(paths["probes"] / "probe-build.json")
    plan["remote_root"] = str(paths["results"])
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    pins = {p.name: sha(p) for p in sorted(bundle.iterdir())
            if p.is_file() and p.name != "bundle-pins.json"}
    save(bundle / "bundle-pins.json", {"schema": "tq-clean-depth-capture-bundle-v2",
                                       "files": pins})
    spec = importlib.util.spec_from_file_location("tq_run_capture", bundle / "run-capture.py")
    run_capture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_capture)
    assert run_capture.verify_bundle()
    assert run_capture.self_test()["passed"]
    staged_input_check = None
    if validate_staged_inputs:
        models, input_root, fixtures, common = run_capture.verify_inputs(plan)
        assert set(models) == set(plan["models"])
        staged_input_check = {"models": {name: value["sha256"] for name, value in models.items()},
                              "input_manifest_sha256": sha(input_root / "complete.json"),
                              "fixtures": {name: value["sha256"] for name, value in fixtures.items()},
                              "common_py_sha256": sha(common)}
    cpu_cells = [item["id"] for item in plan["candidate_captures"]["cells"]
                 if item["device"] == "cpu"]
    assert len(cpu_cells) == 8
    return {"bundle_pin_files": len(pins), "bundle_pins_sha256": sha(bundle / "bundle-pins.json"),
            "cpu_cell_ids": cpu_cells, "capture_self_test": "passed",
            "staged_input_check": staged_input_check}


def stage(args):
    pin = PINS["matched_build"]
    assert subprocess.check_output(["git", "rev-parse", "HEAD^"], cwd=args.repo, text=True).strip() == OVERLAY_PARENT
    assert subprocess.check_output(["git", "rev-parse", OVERLAY_PARENT], cwd=args.repo, text=True).strip() == OVERLAY_PARENT
    assert not args.out.exists(), "refuse an existing staging root"
    args.out.mkdir(parents=True)
    results = args.out / "results"
    results.mkdir()
    receipt_path = results / "stage-receipt.json"
    record = {"schema": "tq-hosted-cpu-depth-stage-v1", "passed": False,
              "candidate_revision": OVERLAY_PARENT, "started_unix": time.time(),
              "artifacts": {"matched_build_id": pin["artifact_id"],
                            "matched_build_digest": pin["artifact_digest"],
                            "probe_build_id": PINS["probe_build"]["artifact_id"],
                            "probe_build_digest": PINS["probe_build"]["artifact_digest"]}}
    save(receipt_path, record)
    try:
        matched = args.matched_root.resolve()
        probe_artifact = args.probe_root.resolve()
        assert sha(matched / "build-output.tar.gz") == pin["build_output_sha256"]
        assert sha(matched / "build-image.tar.gz") == pin["image_archive_sha256"]
        image_line = (matched / "image-id.txt").read_text(encoding="ascii").strip()
        assert image_line == pin["image_id"]
        work = args.out / "assets"
        work.mkdir()
        build_extract = work / "matched"
        safe_extract(matched / "build-output.tar.gz", build_extract)
        build_root = build_extract / "arms" / "candidate-build"
        build_receipt = build_extract / "arms" / "candidate" / "build-result.json"
        build = verify_build(build_root, build_receipt)
        probe_source = verify_probes(probe_artifact)
        probe_root = work / "probes"
        shutil.copytree(probe_source, probe_root)
        for name in ("default-probe", "f32-probe"):
            probe_path = probe_root / name
            os.chmod(probe_path, 0o755)
            assert sha(probe_path) == PINS["probe_build"]["default_probe_sha256" if name == "default-probe" else "f32_probe_sha256"]
        docker_load = subprocess.run(["docker", "load", "--input", str(matched / "build-image.tar.gz")],
                                     text=True, capture_output=True, timeout=600)
        (results / "image-load.log").write_text(docker_load.stdout + docker_load.stderr,
                                                encoding="utf-8")
        assert docker_load.returncode == 0, docker_load.returncode
        inspected = json.loads(subprocess.check_output(
            ["docker", "image", "inspect", pin["image_id"]], text=True))[0]
        assert inspected["Id"] == pin["image_id"]
        record["image"] = {"id": inspected["Id"], "archive_sha256": sha(matched / "build-image.tar.gz"),
                           "loaded": True}

        model_specs = PINS["model_files"]
        assert args.model in model_specs
        assert set(model_specs) == set(json.loads((TEMPLATE / "plan.json").read_text(encoding="utf-8"))["models"])
        model_root = work / "models"
        model_root.mkdir()
        from huggingface_hub import hf_hub_download
        staged_models = {}
        for model_id, model_spec in sorted(model_specs.items()):
            cached = hf_hub_download(repo_id=model_spec["repo"], filename=model_spec["file"],
                                     revision=model_spec["revision"], cache_dir=str(work / "hf-cache"))
            model_path = model_root / model_spec["file"]
            shutil.copyfile(cached, model_path)
            assert model_path.stat().st_size == model_spec["bytes"] and sha(model_path) == model_spec["sha256"]
            staged_models[model_id] = {"file": model_spec["file"], "bytes": model_path.stat().st_size,
                                       "sha256": sha(model_path), "repo": model_spec["repo"],
                                       "revision": model_spec["revision"]}

        inputs = work / "inputs"
        shutil.copytree(INPUT_TEMPLATE, inputs)
        complete = json.loads((inputs / "complete.json").read_text(encoding="utf-8"))
        assert sha(inputs / "complete.json") == PINS["input_manifest_sha256"]
        for rel, item in complete["artifacts"].items():
            path = inputs / rel
            assert path.is_file() and path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], rel
        fixtures = work / "fixtures"
        shutil.copytree(FIXTURE_TEMPLATE, fixtures)
        assert {name: sha(fixtures / ("depth-" + name + ".json"))
                for name in ("raw", "chat")} == PINS["fixture_sha256"]

        source = work / "source"
        subprocess.run(["git", "worktree", "add", "--detach", str(source), OVERLAY_PARENT],
                       cwd=args.repo, check=True, timeout=60, capture_output=True, text=True)
        source_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        source_tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=source, text=True).strip()
        source_status = subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True)
        assert source_head == PINS["candidate_revision"] and source_tree == PINS["candidate_tree"]
        assert not source_status.strip()

        bundle = work / "bundle"
        # Preserve the source manifest's original CRLF representation on the
        # reference-only mount; the clean candidate source remains LF checkout.
        source_common = source / "tests" / "common.py"
        source_common_bytes = source_common.read_bytes()
        assert b"\r" not in source_common_bytes
        expected_common = json.loads((TEMPLATE / "plan.json").read_text(encoding="utf-8"))["reference_source_manifest_sha256"]["tests/common.py"]
        assert sha(source_common) == "7ba2710cc9f9b20977314ab3e4806c1c0641c5ed9dd18bba8acdb336551e96f5"
        reference_common_bytes = source_common_bytes.replace(b"\n", b"\r\n")
        assert hashlib.sha256(reference_common_bytes).hexdigest() == expected_common
        reference_source = work / "reference-source"
        (reference_source / "tests").mkdir(parents=True)
        (reference_source / "tests" / "common.py").write_bytes(reference_common_bytes)
        assert sha(reference_source / "tests" / "common.py") == expected_common

        runtime_paths = {"source": source, "reference_source": reference_source,
                         "build": build_root, "build_receipt": build_receipt,
                         "models": model_root, "inputs": inputs, "fixtures": fixtures,
                         "probes": probe_root, "results": results}
        bundle_check = make_runtime_bundle(bundle, runtime_paths, validate_staged_inputs=True)
        record.update(passed=True, build_receipt_sha256=sha(build_receipt),
                      build_artifact_count=len(build["build_artifacts"]),
                      probe_receipt_sha256=sha(probe_root / "probe-build.json"),
                      probe_sha256={name: sha(probe_root / name)
                                    for name in ("default-probe", "f32-probe")},
                      model=staged_models[args.model], staged_models=staged_models,
                      input_manifest_sha256=sha(inputs / "complete.json"),
                      fixture_sha256=PINS["fixture_sha256"], source={"head": source_head,
                      "tree": source_tree, "clean": True}, bundle=bundle_check,
                      path_roots={key: str(value) for key, value in runtime_paths.items()})
    except Exception as error:
        record["error"] = repr(error)
        record["traceback"] = traceback.format_exc()
    finally:
        record["finished_unix"] = time.time()
        save(receipt_path, record)
    return 0 if record.get("passed") else 1


def self_test():
    plan = json.loads((TEMPLATE / "plan.json").read_text(encoding="utf-8"))
    assert plan["candidate_source"]["revision"] == PINS["candidate_revision"]
    assert len([row for row in plan["candidate_captures"]["cells"] if row["device"] == "cpu"]) == 8
    assert len([row for row in plan["candidate_captures"]["cells"] if row["device"] == "vulkan:0"]) == 8
    assert PINS["matched_build"]["image_id"] == "sha256:e369e69469f20d3940d3689abdbcc33545a5db0361807119ef003b1beb759d85"
    input_manifest = json.loads((INPUT_TEMPLATE / "complete.json").read_text(encoding="utf-8"))
    assert sha(INPUT_TEMPLATE / "complete.json") == PINS["input_manifest_sha256"]
    for rel, item in input_manifest["artifacts"].items():
        path = INPUT_TEMPLATE / rel
        assert path.is_file() and path.stat().st_size == item["bytes"] and sha(path) == item["sha256"]
    for name, digest in PINS["fixture_sha256"].items():
        assert sha(FIXTURE_TEMPLATE / ("depth-" + name + ".json")) == digest
    with tempfile.TemporaryDirectory(prefix="tq-hosted-bundle-contract-") as tmp:
        temp = Path(tmp)
        fake_paths = {name: temp / name for name in
                      ("source", "reference_source", "build", "build_receipt", "models", "inputs",
                       "fixtures", "probes", "results")}
        contract = make_runtime_bundle(temp / "bundle", fake_paths)
        assert contract["cpu_cell_ids"] == [
            "cpu-tq1-0-auto-f16-raw", "cpu-tq1-0-auto-f16-chat",
            "cpu-tq1-0-f32-raw", "cpu-tq1-0-f32-chat",
            "cpu-tq2-0-auto-f16-raw", "cpu-tq2-0-auto-f16-chat",
            "cpu-tq2-0-f32-raw", "cpu-tq2-0-f32-chat"]
    print(json.dumps({"passed": True, "cpu_cells": 8, "probe_artifacts_pinned": 2,
                      "input_artifacts": len(input_manifest["artifacts"]),
                      "bundle_assembly": contract,
                      "scope": "offline plan/input pin checks only"}, separators=(",", ":")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--matched-root", type=Path)
    parser.add_argument("--probe-root", type=Path)
    parser.add_argument("--model", choices=sorted(PINS["model_files"]))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    assert args.matched_root and args.probe_root and args.model and args.out
    return stage(args)


if __name__ == "__main__":
    raise SystemExit(main())
