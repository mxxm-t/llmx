import ast
import hashlib
import json
from pathlib import Path
import tempfile

HERE = Path(__file__).resolve().parent
SOURCE = (HERE / "capture-one.py").read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)
FUNCTIONS = {node.name: node for node in TREE.body if isinstance(node, ast.FunctionDef)}
NS = {"Path": Path, "hashlib": hashlib}
NS["sha"] = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name in ("validate_probe_build", "prepare_output"):
    exec(compile(ast.Module(body=[FUNCTIONS[name]], type_ignores=[]), str(HERE / "capture-one.py"), "exec"), NS)

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    (root / "lib.a").write_bytes(b"library")
    (root / "default-probe").write_bytes(b"default")
    (root / "f32-probe").write_bytes(b"f32")
    sources = {"quant-depth-default-probe.cpp": "a" * 64,
               "quant-depth-f32-probe.cpp": "b" * 64}
    receipt = {
        "commands": [["c++", "default"], ["c++", "f32"]],
        "probes": {"default": NS["sha"](root / "default-probe"),
                   "f32": NS["sha"](root / "f32-probe")},
        "sources": sources,
        "library_sha256": NS["sha"](root / "lib.a"),
    }
    receipt_path = root / "probe-build.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    plan = {
        "probe_source_sha256": sources,
        "candidate_source": {"library_sha256": receipt["library_sha256"]},
        "probe_owner": {"probe_build_receipt_sha256": NS["sha"](receipt_path)},
    }
    assert NS["validate_probe_build"](receipt, receipt_path, plan, "default", root / "default-probe") == receipt["probes"]["default"]
    assert NS["validate_probe_build"](receipt, receipt_path, plan, "f32", root / "f32-probe") == receipt["probes"]["f32"]
    mutations = []
    for field, mutate in (
        ("schema_keys", lambda r: r.pop("sources")),
        ("source_hashes", lambda r: r["sources"].update({"quant-depth-f32-probe.cpp": "c" * 64})),
        ("library_hash", lambda r: r.update(library_sha256="d" * 64)),
        ("probe_hash", lambda r: r["probes"].update(default="e" * 64)),
    ):
        bad = json.loads(json.dumps(receipt))
        mutate(bad)
        try:
            NS["validate_probe_build"](bad, receipt_path, plan, "default", root / "default-probe")
        except (AssertionError, KeyError):
            mutations.append(field)
        else:
            raise AssertionError(f"mutated receipt was accepted: {field}")
    capture = NS["prepare_output"](root)
    assert capture == root / "capture" and capture.is_dir()
    try:
        NS["prepare_output"](root)
    except AssertionError:
        mutations.append("existing_capture_child")
    else:
        raise AssertionError("existing capture child was accepted")
    (root / "probe-build.json").write_bytes(b"tampered")
    try:
        NS["validate_probe_build"](receipt, receipt_path, plan, "default", root / "default-probe")
    except AssertionError:
        mutations.append("receipt_file_hash")
    else:
        raise AssertionError("tampered receipt bytes were accepted")

result = {"passed": True, "scope": "mocked capture output and probe-build receipt contract only; no inference or build",
          "schema": ["commands", "probes", "sources", "library_sha256"],
          "valid_policies": ["default", "f32"], "rejected_mutations": mutations}
(HERE / "capture-contract-smoke.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))