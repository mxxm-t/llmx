"""Download and verify the pinned HF fixtures using only the Python stdlib."""

import hashlib
from pathlib import Path
import sys
import tempfile
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from baseline import BASELINE_MODELS


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(spec):
    destination = (Path.home() / ".cache" / "huggingface" / "hub" /
                   ("models--" + spec["repo"].replace("/", "--")) /
                   "snapshots" / spec["revision"] / spec["file"])
    if destination.is_file() and sha256(destination) == spec["sha256"]:
        print("verified " + spec["file"], flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = "https://huggingface.co/%s/resolve/%s/%s" % (spec["repo"], spec["revision"], spec["file"])
    print("downloading " + spec["file"], flush=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as f:
            temporary = Path(f.name)
            digest = hashlib.sha256()
            with urllib.request.urlopen(url, timeout=60) as response:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    digest.update(block)
                    f.write(block)
        if digest.hexdigest() != spec["sha256"]:
            raise RuntimeError("SHA-256 mismatch for " + spec["file"])
        temporary.replace(destination)
        print("verified " + spec["file"], flush=True)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


if __name__ == "__main__":
    for spec in BASELINE_MODELS:
        fetch(spec)
