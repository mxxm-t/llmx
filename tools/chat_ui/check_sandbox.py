"""Manual integration check using an existing Linux Docker image, without GPUs."""

import argparse
import json
from pathlib import Path
import tempfile
import time

from workspace import Workspace


def wait(workspace, job, seconds=45):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        result = workspace.job(job["id"])
        if result["status"] != "running":
            return result
        time.sleep(0.1)
    workspace.cancel(job["id"])
    raise AssertionError("Container check did not finish within its time limit")


def check(image, host):
    script = '''import json, os, pathlib, socket
checks = {"uid": os.getuid(), "docker_socket_absent": not pathlib.Path("/var/run/docker.sock").exists()}
try:
    pathlib.Path("/write-test").write_text("blocked")
    checks["read_only_root"] = False
except OSError:
    checks["read_only_root"] = True
sock = socket.socket()
sock.settimeout(0.5)
try:
    sock.connect(("1.1.1.1", 443))
    checks["network_blocked"] = False
except OSError:
    checks["network_blocked"] = True
finally:
    sock.close()
pathlib.Path("src/result.txt").write_text("changed")
print(json.dumps(checks), flush=True)
'''
    with tempfile.TemporaryDirectory(prefix="llmx-workspace-check-") as directory:
        workspace = Workspace(directory, image, host)
        workspace.write("src/check.py", script, 0)
        workspace.checkpoint("Before integration check")
        point = workspace.snapshot()["checkpoints"][-1]["id"]
        result = wait(workspace, workspace.start("python src/check.py"))
        assert result["status"] == "completed", result
        checks = json.loads(result["output"])
        assert checks == dict(uid=65534, docker_socket_absent=True, read_only_root=True, network_blocked=True), checks
        assert workspace.read("src/result.txt")["content"] == "changed"
        assert "result.txt" in workspace.diff(point)["diff"]
        workspace.restore(point, workspace.snapshot()["revision"])
        assert not any(file["path"] == "src/result.txt" for file in workspace.snapshot()["files"])
        job = workspace.start("printf 'started\\n'; sleep 60")
        deadline = time.monotonic() + 20
        while "started" not in workspace.job(job["id"])["output"]:
            assert time.monotonic() < deadline, workspace.job(job["id"])
            time.sleep(0.1)
        workspace.cancel(job["id"])
        assert wait(workspace, job, 15)["status"] == "cancelled"
        result = wait(workspace, workspace.start("ln -s /etc/passwd outside.txt"))
        assert result["status"] == "failed" and "link" in result["error"], result
        assert not any(file["path"] == "outside.txt" for file in workspace.snapshot()["files"])
        workspace.close()
        return dict(image=image, docker_host=host, checks=checks, file_roundtrip=True, checkpoint_restore=True,
                    cancellation=True, symlink_refusal=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Existing image with sh, tar, sleep and Python")
    parser.add_argument("--docker-host")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check(args.image, args.docker_host)
    report = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(report + "\n", encoding="utf-8")
    print(report)
