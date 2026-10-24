"""Fail-closed outer adapter for the pinned TQ depth probe/capture payloads."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import sys
import time
import traceback


HERE = Path(__file__).resolve().parent
PLAN_PATH = HERE / "plan.json"
PINS_PATH = HERE / "bundle-pins.json"
LABEL = "llmx.tqpair.f16kvdepth"
GIB = 1024 ** 3


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_owner(name, path):
    spec = importlib.util.spec_from_file_location(name, HERE / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_hex(value, what):
    assert isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), (what, value)


def verify_bundle():
    pins = json.loads(PINS_PATH.read_text(encoding="utf-8"))["files"]
    actual = {p.name for p in HERE.iterdir() if p.is_file() and p.name != "bundle-pins.json"}
    assert set(pins) == actual, {"missing": sorted(actual - set(pins)), "stale": sorted(set(pins) - actual)}
    for name, digest in pins.items():
        assert sha(HERE / name) == digest, name
    return sha(PINS_PATH)


def verify_build_receipt(receipt, plan, image_id):
    source = plan["candidate_source"]
    assert image_id == plan["image_identity"]["id"]
    assert receipt["status"] == source["build_receipt_status"] == "passed"
    assert receipt["revision"] == source["revision"]
    assert receipt["source_head_after"] == source["revision"]
    assert not receipt["source_status_after"].strip()
    assert receipt["attach_exit_code"] == source["attach_exit_code"] == 0
    state = receipt.get("container_state", {})
    assert state.get("ExitCode") == source["container_exit_code"] == 0
    assert state.get("OOMKilled") is source["container_oom_killed"] is False
    assert receipt.get("removed") is True
    assert receipt["image_id"] == image_id
    artifacts = {item["path"]: item for item in receipt["build_artifacts"]}
    assert len(artifacts) == source["artifact_inventory_count"]
    for item in source["build_artifacts"]:
        got = artifacts[item["path"]]
        assert got["sha256"] == item["sha256"] and got["bytes"] == item["bytes"], item["path"]
    assert artifacts["llmx"]["sha256"] == source["cli_sha256"]
    assert artifacts["libllmx-vulkan.a"]["sha256"] == source["library_sha256"]
    assert artifacts["generated/config.hpp"]["sha256"] == source["generated_config_sha256"]
    assert artifacts["generated/llmx-build-info.hpp"]["sha256"] == source["generated_build_info_sha256"]
    return {"revision": receipt["revision"], "image_id": receipt["image_id"],
            "artifact_count": len(artifacts), "cli_sha256": source["cli_sha256"],
            "library_sha256": source["library_sha256"]}


def verify_receipt_file(path, plan, image_id):
    expected = plan["candidate_source"]["build_receipt_sha256"]
    assert expected and sha(path) == expected, "hosted build receipt digest mismatch"
    receipt = json.loads(Path(path).read_text(encoding="utf-8"))
    return verify_build_receipt(receipt, plan, image_id), receipt


def verify_probe(plan, probe_root, policy):
    owner = plan["probe_owner"]
    receipt_path = Path(probe_root) / "probe-build.json"
    assert owner["probe_build_receipt_sha256"] and sha(receipt_path) == owner["probe_build_receipt_sha256"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert set(receipt) == {"commands", "probes", "sources", "library_sha256"}
    assert set(receipt["probes"]) == {"default", "f32"}
    assert receipt["sources"] == plan["probe_source_sha256"]
    assert receipt["library_sha256"] == plan["candidate_source"]["library_sha256"]
    binary = Path(probe_root) / ("default-probe" if policy == "default" else "f32-probe")
    assert sha(binary) == receipt["probes"][policy]
    assert owner["default_probe_sha256"] == receipt["probes"]["default"]
    assert owner["f32_probe_sha256"] == receipt["probes"]["f32"]
    return receipt, binary


def verify_probe_output(plan, output):
    receipt_path = Path(output) / "probe-build.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert set(receipt) == {"commands", "probes", "sources", "library_sha256"}
    assert set(receipt["probes"]) == {"default", "f32"}
    assert receipt["sources"] == plan["probe_source_sha256"]
    assert receipt["library_sha256"] == plan["candidate_source"]["library_sha256"]
    for policy in ("default", "f32"):
        binary = Path(output) / (policy + "-probe")
        assert binary.is_file() and sha(binary) == receipt["probes"][policy], policy
    return {"receipt_sha256": sha(receipt_path),
            "default_probe_sha256": receipt["probes"]["default"],
            "f32_probe_sha256": receipt["probes"]["f32"],
            "sources": receipt["sources"], "library_sha256": receipt["library_sha256"]}


def verify_inputs(plan):
    models = {}
    for name, spec in plan["models"].items():
        path = Path(plan["model_root"]) / spec["file"]
        assert path.stat().st_size == spec["bytes"]
        assert sha(path) == spec["sha256"], name
        models[name] = {"path": path, "bytes": spec["bytes"], "sha256": spec["sha256"]}
    input_root = Path(plan["input_root"])
    manifest = input_root / "complete.json"
    assert sha(manifest) == plan["input_manifest_sha256"]
    complete = json.loads(manifest.read_text(encoding="utf-8"))
    listed = complete["artifacts"]
    assert {name: item["sha256"] for name, item in listed.items()} == plan["input_artifacts"]
    for name, item in listed.items():
        path = input_root / name
        assert path.is_file() and sha(path) == item["sha256"], name
    fixtures = {}
    for kind, digest in plan["fixtures"].items():
        path = Path(plan["fixture_root"]) / ("depth-" + kind + ".json")
        assert sha(path) == digest, kind
        fixtures[kind] = {"path": path, "sha256": digest}
    common = Path(plan["reference_source"]) / "tests/common.py"
    assert sha(common) == plan["reference_source_manifest_sha256"]["tests/common.py"]
    return models, input_root, fixtures, common


def input_identity(plan):
    models, input_root, fixtures, common = verify_inputs(plan)
    return {"models": {key: {"bytes": value["bytes"], "sha256": value["sha256"]}
                        for key, value in models.items()},
            "manifest_sha256": sha(input_root / "complete.json"),
            "artifacts": plan["input_artifacts"],
            "fixtures": {key: value["sha256"] for key, value in fixtures.items()},
            "common_py_sha256": sha(common)}


def verify_source(plan, source):
    source = Path(source)
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    tree = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD^{tree}"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True)
    expected = plan["candidate_source"]
    assert head == expected["revision"] and tree == expected["source_tree_git_sha1"] and not status.strip()
    return {"head": head, "tree": tree, "clean": True}


def verify_build(plan, build):
    source = plan["candidate_source"]
    receipt_path = Path(source["build_receipt_path"])
    assert sha(receipt_path) == source["build_receipt_sha256"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    image = plan["image_identity"]["id"]
    identity = verify_build_receipt(receipt, plan, image)
    build = Path(build)
    for item in receipt["build_artifacts"]:
        rel = PurePosixPath(item["path"])
        assert not rel.is_absolute() and ".." not in rel.parts
        path = build.joinpath(*rel.parts)
        assert path.is_file() and path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], item["path"]
    return identity, receipt


def admission_plan(plan, admission, root):
    required = {"cpu_set", "cpu_quota", "threads", "container_name"}
    assert set(admission) == required | {"device"}, (set(admission), required)
    cpu_set = admission["cpu_set"]
    assert re.fullmatch(r"[0-9]+-[0-9]+", cpu_set) and cpu_set
    assert isinstance(admission["cpu_quota"], (int, float)) and admission["cpu_quota"] > 0
    assert isinstance(admission["threads"], int) and admission["threads"] > 0
    assert re.fullmatch(r"llmx-tq-clean-depth-[a-z0-9-]+", admission["container_name"])
    r = plan["outer_runner"]["execution_policy"]
    resources = {"cpu_set": cpu_set, "cpu_quota": admission["cpu_quota"],
                 "memory_bytes": r["memory_bytes"], "memory_swap_bytes": r["memory_swap_bytes"],
                 "minimum_host_available_bytes": r["minimum_host_available_bytes"],
                 "minimum_live_reserve_bytes": r["minimum_live_reserve_bytes"],
                 "minimum_disk_free_bytes": r["minimum_disk_free_bytes"],
                 "pids_limit": r["pids_limit"], "runtime": r["runtime"], "tmpfs": r["tmpfs"]}
    return {"root": str(root), "resources": resources,
            "device": admission["device"]}


def verify_image(plan):
    expected = plan["image_identity"]["id"]
    actual = json.loads(subprocess.check_output(["docker", "image", "inspect", expected], text=True))[0]
    assert actual["Id"] == expected
    return actual


def verify_container(info, name, image, cpuset, quota, memory, expected_mounts, device):
    host = info["HostConfig"]
    assert info["Name"] == "/" + name and info["Image"] == image
    assert info["Config"]["Labels"].get(LABEL) == name
    assert host["NetworkMode"] == "none" and host["Runtime"] == "runc"
    assert host["CpusetCpus"] == cpuset and host["NanoCpus"] == int(quota * 1_000_000_000)
    assert host["Memory"] == memory and host["MemorySwap"] == memory
    assert host["PidsLimit"] == 256 and host["ReadonlyRootfs"] and not host["Privileged"]
    assert any(row["Name"] == "core" and row["Soft"] == row["Hard"] == 0 for row in host["Ulimits"])
    assert host.get("Tmpfs", {}).get("/tmp") == "rw,size=512m"
    mounts = {row["Destination"]: row for row in info["Mounts"] if row["Type"] == "bind"}
    assert set(mounts) == set(expected_mounts), (set(mounts), set(expected_mounts))
    for dst, source in expected_mounts.items():
        row = mounts[dst]
        assert Path(row["Source"]).resolve() == Path(source).resolve(), (dst, row["Source"], source)
        assert row["RW"] is (dst == "/out"), (dst, row["RW"])
    if device:
        nodes = [row["PathOnHost"] for row in host.get("Devices", [])]
        assert device["node"] in nodes
    else:
        assert not host.get("Devices") and not host.get("DeviceRequests")


def payload_argv(plan, mode, cell):
    if mode == "probe-build":
        return ["python3", "-B", "/bundle/build-probes.py"]
    device_arg = "cpu" if cell["device"] == "cpu" else "0"
    return ["python3", "-B", "/bundle/capture-one.py", cell["model"],
            cell["policy"], cell["variant"], device_arg]


def build_argv(plan, admission, output, model=None, mode="probe-build", cell=None):
    args = ["docker", "create", "--init", "--name", admission["container_name"],
            "--cidfile", str(Path(output) / "container.cid"), "--label", LABEL + "=" + admission["container_name"],
            "--network", "none", "--runtime", "runc", "--cpuset-cpus", admission["cpu_set"],
            "--cpus", str(admission["cpu_quota"]), "--memory", str(plan["outer_runner"]["execution_policy"]["memory_bytes"]),
            "--memory-swap", str(plan["outer_runner"]["execution_policy"]["memory_swap_bytes"]),
            "--pids-limit", "256", "--ulimit", "core=0", "--read-only", "--tmpfs", "/tmp:rw,size=512m",
            "--security-opt", "no-new-privileges"]
    device = admission.get("device")
    if device:
        args += ["--device", device["node"]]
    b = plan["candidate_source"]["build_path"]
    s = plan["candidate_source"]["source_path"]
    i = plan["input_root"]
    f = plan["fixture_root"]
    refs = plan["reference_source"]
    mounts = [(str(HERE), "/bundle", True), (b, "/b", True), (s, "/source", True),
              (i, "/inputs", True), (f, "/fixtures", True), (refs, "/reference-source", True),
              (str(output), "/out", False)]
    if model:
        mounts.append((str(model), "/model.gguf", True))
        mounts.append((plan["probe_owner"]["probe_build_root"], "/probes", True))
    for src, dst, readonly in mounts:
        assert Path(src).exists(), src
        args += ["--mount", "type=bind,src=" + src + ",dst=" + dst + (",readonly" if readonly else "")]
    payload = payload_argv(plan, mode, cell) if model is not None else payload_argv(plan, "probe-build", None)
    args += [plan["image_identity"]["id"]] + payload
    return args


def self_test():
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    receipt_path = HERE / "a0fa-build-result.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    image = plan["image_identity"]["id"]
    verify_build_receipt(receipt, plan, image)
    rejected = []
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tampered = Path(tmp) / "build-result.json"
        tampered.write_bytes(receipt_path.read_bytes() + b" ")
        try:
            verify_receipt_file(tampered, plan, image)
        except AssertionError:
            rejected.append("receipt_sha")
        else:
            raise AssertionError("mutated receipt bytes were accepted")
    bad_source = json.loads(json.dumps(receipt)); bad_source["revision"] = "0" * 40
    try:
        verify_build_receipt(bad_source, plan, image)
    except AssertionError:
        rejected.append("source_revision")
    else:
        raise AssertionError("mutated source revision was accepted")
    bad_plan = json.loads(json.dumps(plan)); bad_plan["image_identity"]["id"] = "sha256:" + "0" * 64
    try:
        verify_build_receipt(receipt, bad_plan, image)
    except AssertionError:
        rejected.append("image_identity")
    else:
        raise AssertionError("mutated image identity was accepted")
    assert rejected == ["receipt_sha", "source_revision", "image_identity"]
    cpu_cell = next(row for row in plan["candidate_captures"]["cells"] if row["device"] == "cpu")
    vulkan_cell = next(row for row in plan["candidate_captures"]["cells"] if row["device"] == "vulkan:0")
    assert payload_argv(plan, "probe-build", None) == ["python3", "-B", "/bundle/build-probes.py"]
    assert payload_argv(plan, "capture", cpu_cell)[-4:] == [cpu_cell["model"], cpu_cell["policy"], cpu_cell["variant"], "cpu"]
    assert payload_argv(plan, "capture", vulkan_cell)[-4:] == [vulkan_cell["model"], vulkan_cell["policy"], vulkan_cell["variant"], "0"]
    with tempfile.TemporaryDirectory() as tmp:
        staged = json.loads(json.dumps(plan))
        for key in ("build_path", "source_path"):
            staged["candidate_source"][key] = tmp
        for key in ("input_root", "fixture_root", "reference_source"):
            staged[key] = tmp
        staged["probe_owner"]["probe_build_root"] = tmp
        model = Path(tmp) / "model.gguf"
        model.touch()
        admission = {"container_name": "llmx-tq-clean-depth-mount-check", "cpu_set": "0-3",
                     "cpu_quota": 4, "device": None}
        argv = build_argv(staged, admission, Path(tmp), model, "capture", cpu_cell)
        mounts = [argv[i + 1] for i, value in enumerate(argv) if value == "--mount"]
        by_destination = {next(part[4:] for part in value.split(",") if part.startswith("dst=")): value
                          for value in mounts}
        assert set(by_destination) == {"/bundle", "/b", "/source", "/inputs", "/fixtures",
                                       "/reference-source", "/out", "/model.gguf", "/probes"}
        assert by_destination["/probes"] == "type=bind,src=" + tmp + ",dst=/probes,readonly"
        assert all(value.endswith(",readonly") for key, value in by_destination.items() if key != "/out")
        assert not by_destination["/out"].endswith(",readonly")
    # These functions are patched to fail so a regression cannot turn self-test
    # into a container start or inference launch.
    old_run, old_output, old_popen = subprocess.run, subprocess.check_output, subprocess.Popen
    def forbidden(*_args, **_kwargs):
        raise AssertionError("self-test attempted external process execution")
    subprocess.run = subprocess.check_output = subprocess.Popen = forbidden
    try:
        old_argv = sys.argv
        try:
            sys.argv = [str(HERE / "run-capture.py"), "probe-build"]
            try:
                main()
            except AssertionError as error:
                assert "explicitly authorized" in str(error)
            else:
                raise AssertionError("execution gate accepted an unarmed invocation")
        finally:
            sys.argv = old_argv
    finally:
        subprocess.run, subprocess.check_output, subprocess.Popen = old_run, old_output, old_popen
    return {"passed": True, "tested": rejected + ["unarmed_execution_refused"],
            "docker_calls": 0, "inference_calls": 0, "scope": "pure receipt/source/image contract only"}


def execute(args):
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    pin_hash = verify_bundle()
    assert args.admission and args.resources_released
    admission = json.loads(Path(args.admission).read_text(encoding="utf-8"))
    cell = None
    if args.mode == "capture":
        cell = next((row for row in plan["candidate_captures"]["cells"] if row["id"] == args.cell), None)
        assert cell is not None
        assert cell["device"] == ("cpu" if not cell["device"].startswith("vulkan") else "vulkan:0")
        if cell["device"] == "cpu":
            assert not admission.get("device"), "CPU capture must not map a GPU"
        else:
            assert admission.get("device"), "Vulkan capture requires freshly checked device identity"
    else:
        assert not args.cell and not admission.get("device"), "probe build is CPU-only"

    root = Path(plan["remote_root"])
    source_path = Path(plan["candidate_source"]["source_path"])
    build_path = Path(plan["candidate_source"]["build_path"])
    model_by_cell = plan["models"].get(cell["model"]) if cell else None
    model_path = Path(plan["model_root"]) / model_by_cell["file"] if cell else None
    output_root = root / ("probes" if args.mode == "probe-build" else "captures")
    if args.mode == "capture":
        output_root = output_root / args.run_id / args.cell
    else:
        output_root = output_root / args.run_id
    output_root.mkdir(parents=True, exist_ok=False)
    record_path = output_root / "outer-execution.json"
    record = {"mode": args.mode, "cell": args.cell, "run_id": args.run_id,
              "started": time.time(), "passed": False, "removed": False,
              "scope": "one bounded probe build or one candidate capture; no HF comparison"}
    cid = child = None
    create_attempted = False
    owner = gpu_owner = runtime_plan = None
    name = admission["container_name"]
    old_handlers = {sig: signal.signal(sig, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt("signal " + str(signum))))
                    for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        owner = load_owner("tq_resource_owner", "resource-owner.py")
        gpu_owner = load_owner("tq_gpu_owner", "gpu-owner.py")
        runtime_plan = admission_plan(plan, admission, root)
        label = owner.LABEL
        assert label == LABEL
        record["bundle_pins_sha256"] = pin_hash
        record["source"] = verify_source(plan, source_path)
        record["build"], _ = verify_build(plan, build_path)
        models, inroot, fixtures, common = verify_inputs(plan)
        record["inputs"] = input_identity(plan)
        record["image"] = verify_image(plan)["Id"]
        if cell:
            model_ref = model_by_cell
            variant = cell["variant"]
            policy = cell["policy"]
            probe_root = Path(plan["probe_owner"]["probe_build_root"])
            probe_record, probe_path = verify_probe(plan, probe_root, policy)
            record["probe"] = {"receipt_sha256": sha(probe_root / "probe-build.json"),
                                "binary_sha256": sha(probe_path), "policy": policy}
            record["model"] = {"path": str(model_path), "sha256": model_ref["sha256"], "bytes": model_ref["bytes"]}
            record["fixture"] = fixtures[variant]["sha256"]
            out = output_root
        else:
            probe_root = Path(plan["probe_owner"]["probe_build_root"])
            assert not probe_root.exists() or not any(probe_root.iterdir()), "probe output must be a fresh empty directory"
            probe_root.mkdir(parents=True, exist_ok=True)
            out = probe_root
        # The existing resource owner enforces current affinity, disk, memory and
        # non-overlapping active-container checks. Vulkan adds its current PCI/VRAM/fuser owner.
        record["resource_admission"] = owner.resources(runtime_plan)
        if cell and cell["device"] == "vulkan:0":
            m = {"device": admission["device"], "resources": {"cpuset": admission["cpu_set"],
                  "cpus": str(admission["cpu_quota"])}, "candidate": {"root": str(root)}, "container": name}
            record["gpu_admission"] = gpu_owner.preflight(m)
        assert subprocess.run(["docker", "inspect", name], capture_output=True, timeout=30).returncode != 0
        # Pin every read-only input immediately before container creation.
        before = {"source": record["source"], "build": record["build"], "image": record["image"],
                  "inputs": record["inputs"]}
        if cell:
            before.update(model=sha(model_path), fixture=sha(Path(plan["fixture_root"]) / ("depth-" + cell["variant"] + ".json")),
                          probe=sha(probe_path), probe_receipt=sha(probe_root / "probe-build.json"))
        record["before"] = before
        create_argv = build_argv(plan, admission, out, model_path,
                                 mode=args.mode, cell=cell)
        record["create_argv"] = create_argv
        record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        create_attempted = True
        created = subprocess.run(create_argv, capture_output=True, text=True, timeout=30, check=True)
        cid = created.stdout.strip()
        assert re.fullmatch(r"[0-9a-f]{64}", cid)
        info = owner.inspect_owned(cid, name, plan["image_identity"]["id"])
        expected_mounts = {"/bundle": HERE, "/b": build_path, "/source": source_path,
                           "/inputs": Path(plan["input_root"]), "/fixtures": Path(plan["fixture_root"]),
                           "/reference-source": Path(plan["reference_source"]), "/out": out}
        if cell:
            expected_mounts["/model.gguf"] = model_path
            expected_mounts["/probes"] = probe_root
        verify_container(info, name, plan["image_identity"]["id"], admission["cpu_set"],
                         admission["cpu_quota"], runtime_plan["resources"]["memory_bytes"], expected_mounts,
                         admission.get("device") if cell and cell["device"] == "vulkan:0" else None)
        # Recheck admission after creation, before starting the bounded payload.
        owner.resources(runtime_plan)
        with (output_root / "run.log").open("xb") as log, (output_root / "activity.jsonl").open("x", encoding="utf-8") as activity:
            child = subprocess.Popen(["docker", "start", "-a", cid], stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + (plan["outer_runner"]["execution_policy"]["timeout_seconds"] if cell else plan["probe_build_resources"]["timeout_seconds"])
            while child.poll() is None:
                info = owner.inspect_owned(cid, name, plan["image_identity"]["id"])
                sample = owner.monitor(info, runtime_plan)
                peer_unused, sample["peer_memory_reservations"] = owner.peer_reservations(record["resource_admission"]["active_container_cpus"])
                cgroup_current = 0
                if "memory.current" in sample:
                    try: cgroup_current = int(sample["memory.current"])
                    except ValueError: pass
                sample["required_host_available_bytes"] = owner.live_host_floor(runtime_plan, {"memory.current": cgroup_current}) + peer_unused
                activity.write(json.dumps(sample) + "\n"); activity.flush()
                assert sample["host_available_bytes"] >= sample["required_host_available_bytes"], sample
                assert sample["disk_free_bytes"] >= runtime_plan["resources"]["minimum_disk_free_bytes"], sample
                if time.monotonic() >= deadline:
                    record["timed_out"] = True
                    subprocess.run(["docker", "kill", cid], check=False, timeout=30)
                    raise TimeoutError("bounded outer stage exceeded its timeout")
                time.sleep(1)
        record["attach_exit_code"] = child.returncode
        info = owner.inspect_owned(cid, name, plan["image_identity"]["id"])
        record["container_state"] = info["State"]
        if args.mode == "probe-build":
            record["probe_build"] = verify_probe_output(plan, out)
            record["passed_payload"] = child.returncode == 0
        else:
            payload = json.loads((out / "capture" / "execution.json").read_text(encoding="utf-8"))
            record["payload_execution_sha256"] = sha(out / "capture" / "execution.json")
            record["payload_passed"] = payload.get("passed") is True and payload.get("comparison", "").startswith("pending")
            record["passed_payload"] = child.returncode == 0 and record["payload_passed"]
        after = {"source": verify_source(plan, source_path), "build": verify_build(plan, build_path)[0],
                 "image": verify_image(plan)["Id"], "inputs": input_identity(plan)}
        if cell:
            after.update(model=sha(model_path), fixture=sha(Path(plan["fixture_root"]) / ("depth-" + cell["variant"] + ".json")),
                         probe=sha(probe_path), probe_receipt=sha(probe_root / "probe-build.json"))
        record["after"] = after
        assert after == before, "read-only inputs or pins changed during stage"
        record["passed"] = bool(record["passed_payload"] and info["State"].get("ExitCode") == 0 and not info["State"].get("OOMKilled"))
    except BaseException:
        record["error"] = traceback.format_exc()
    finally:
        previous = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            if not cid and (output_root / "container.cid").is_file():
                cid = (output_root / "container.cid").read_text().strip()
            if not cid and create_attempted:
                result = subprocess.run(["docker", "ps", "-aq", "--no-trunc", "--filter", "name=^/" + name + "$"], capture_output=True, text=True, timeout=30)
                rows = result.stdout.splitlines()
                assert len(rows) <= 1, "multiple containers match unique stage name"
                if rows:
                    cid = rows[0]
            if cid:
                assert owner is not None, "cannot clean a container without the pinned identity owner"
                owner.inspect_owned(cid, name, plan["image_identity"]["id"])
                info = json.loads(subprocess.check_output(["docker", "inspect", cid], text=True))[0]
                record["container_state"] = info["State"]
                (output_root / "container.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
                subprocess.run(["docker", "rm", "-f", cid], check=True, timeout=60)
                record["removed"] = subprocess.run(["docker", "inspect", cid], capture_output=True, timeout=30).returncode != 0
            else:
                record["removed"] = True
            if child is not None:
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait(timeout=10)
        except BaseException:
            record["cleanup_error"] = traceback.format_exc()
        for sig, handler in previous.items():
            signal.signal(sig, old_handlers[sig])
        state = record.get("container_state", {})
        record["passed"] = bool(record.get("passed") and record.get("removed") and
                                state.get("ExitCode") == 0 and not state.get("OOMKilled", False))
        record["finished"] = time.time()
        try:
            record["artifacts"] = {str(p.relative_to(output_root)): sha(p) for p in sorted(output_root.rglob("*")) if p.is_file() and p.name != "outer-execution.json"}
            record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        except OSError:
            pass
    return 0 if record.get("passed") else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("probe-build", "capture"))
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--execute", action="store_true", help="explicitly permit the pinned Docker stage")
    parser.add_argument("--resources-released", action="store_true", help="root confirms peer workload ended and fresh admission is appropriate")
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--cell")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
        return 0
    assert args.execute and args.resources_released, "refuse Docker unless explicitly authorized after fresh resource release"
    assert args.admission and args.run_id and re.fullmatch(r"[A-Za-z0-9-]{8,64}", args.run_id)
    if args.mode == "capture":
        assert args.cell
    return execute(args)


if __name__ == "__main__":
    sys.exit(main())
