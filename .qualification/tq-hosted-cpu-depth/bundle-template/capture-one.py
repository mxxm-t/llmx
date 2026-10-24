import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time

kind, policy, variant, device = sys.argv[1:]
assert kind in ("tq1-0", "tq2-0")
assert policy in ("default", "f32") and variant in ("raw", "chat")
assert device in ("cpu", "0")
plan = json.loads(Path("/bundle/plan.json").read_text(encoding="utf-8"))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validate_probe_build(record, receipt_path, plan, policy, probe):
    assert set(record) == {"commands", "probes", "sources", "library_sha256"}
    assert set(record["probes"]) == {"default", "f32"}
    assert record["sources"] == plan["probe_source_sha256"]
    assert record["library_sha256"] == plan["candidate_source"]["library_sha256"]
    expected_receipt = plan["probe_owner"]["probe_build_receipt_sha256"]
    assert expected_receipt and sha(receipt_path) == expected_receipt
    expected_probe = record["probes"][policy]
    assert isinstance(expected_probe, str) and len(expected_probe) == 64
    assert sha(probe) == expected_probe
    return expected_probe


def prepare_output(root):
    root = Path(root)
    assert root.is_dir(), "the runner must mount an existing output root"
    output = root / "capture"
    assert not output.exists(), "refuse an existing capture child"
    output.mkdir()
    return output


out_root = Path("/out")
out = prepare_output(out_root)
record = {"kind":kind, "policy":policy, "variant":variant, "device":device,
          "passed":False, "comparison":"pending independent reference outputs"}
(out/"execution.json").write_text(json.dumps(record, indent=2)+"\n")
try:
    spec = plan["models"][kind]
    model = Path("/model.gguf")
    probe_build = json.loads(Path("/probes/probe-build.json").read_text(encoding="utf-8"))
    probe = Path("/probes") / ("default-probe" if policy == "default" else "f32-probe")
    expected_probe = validate_probe_build(probe_build, Path("/probes/probe-build.json"), plan, policy, probe)
    assert model.stat().st_size == spec["bytes"] and sha(model) == spec["sha256"]
    inputs = Path("/inputs")
    assert sha(inputs/"complete.json") == plan["input_manifest_sha256"]
    for name, digest in plan["input_artifacts"].items():
        item = inputs / name
        assert item.is_file() and sha(item) == digest, name
    fixture = Path("/fixtures") / ("depth-"+variant+".json")
    assert sha(fixture) == plan["fixtures"][variant]
    runtime_paths = ["/b/llmx", "/b/libllmx-vulkan.a", "/b/generated/config.hpp",
                     "/b/generated/llmx-build-info.hpp", "/reference-source/tests/common.py"]
    runtime_before = {p:sha(p) for p in runtime_paths}
    for key, field in (("cli_sha256","/b/llmx"),("library_sha256","/b/libllmx-vulkan.a"),
                       ("generated_config_sha256","/b/generated/config.hpp"),
                       ("generated_build_info_sha256","/b/generated/llmx-build-info.hpp")):
        assert plan["candidate_source"][key] and sha(field)==plan["candidate_source"][key], key
    device_arg = "cpu" if device == "cpu" else device
    argv = [str(probe),str(model),str(fixture),device_arg]
    started = time.monotonic()
    with (out/"capture.stdout").open("xb") as stdout, (out/"capture.stderr").open("xb") as stderr:
        result = subprocess.run(argv, stdout=stdout, stderr=stderr, timeout=2100, check=False)
    record.update(command=argv, exit=result.returncode, seconds=time.monotonic()-started,
                  probe_sha256=sha(probe), model_sha256=sha(model), fixture_sha256=sha(fixture),
                  runtime_files_before=runtime_before, stdout_sha256=sha(out/"capture.stdout"),
                  stderr_sha256=sha(out/"capture.stderr"))
    assert result.returncode == 0, "probe child failed"
    rows=[line.split() for line in (out/"capture.stdout").read_text(encoding="utf-8").splitlines()
          if re.match(r"^\d+ ",line)]
    assert [int(row[0]) for row in rows] == list(range(16383,16895))
    assert all(len(row)==101 and all(math.isfinite(float(row[j])) for j in range(2,101,2)) for row in rows)
    stderr=(out/"capture.stderr").read_text(encoding="utf-8")
    assert "tokenizer: 16384 exact ids" in stderr
    release=re.findall(r'^#define LLMX_RELEASE_VERSION "([^"]+)"$',Path("/b/generated/config.hpp").read_text(encoding="utf-8"),re.M)
    revision=re.findall(r'^#define LLMX_BUILD_REVISION "([^"]+)"$',Path("/b/generated/llmx-build-info.hpp").read_text(encoding="utf-8"),re.M)
    assert len(release)==len(revision)==1
    expected_banner=f"version: {release[0]}+{revision[0]}; dtype: {'auto -> f16' if policy=='default' else 'f32 -> f32'}; tokenizer: 16384 exact ids"
    assert [line for line in stderr.splitlines() if line.startswith("version: ")] == [expected_banner]
    path_lines=[json.loads(line[len("matrix-paths: "):]) for line in stderr.splitlines()
                if line.startswith("matrix-paths: ")]
    assert len(path_lines)==1 and len(path_lines[0]["devices"])==1
    expected_dtype="f32" if policy=="f32" else "f16"
    assert path_lines[0]["dtype"]==expected_dtype
    path_key="f32" if policy=="f32" else "auto-f16"
    expected=plan["static_expected_matrix_paths"][path_key]
    if isinstance(expected,dict): expected=expected["cpu" if device=="cpu" else "vulkan:0"]
    actual=sorted(set(path_lines[0]["devices"][0]))
    assert actual==sorted(expected),(actual,expected)
    record.update(positions=512,matrix_paths=path_lines[0],actual_path_categories=actual)
    after={p:sha(p) for p in runtime_paths}
    assert after==runtime_before
    assert model.stat().st_size==spec["bytes"] and sha(model)==spec["sha256"]
    assert sha(fixture)==plan["fixtures"][variant]
    assert sha(probe)==expected_probe
    record["runtime_files_after"]=after
    record["passed"]=True
except BaseException as error:
    record["error"]=repr(error)
finally:
    (out/"execution.json").write_text(json.dumps(record,indent=2,ensure_ascii=True)+"\n",encoding="utf-8")
if not record["passed"]:
    raise SystemExit(1)
