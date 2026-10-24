#!/usr/bin/env python3
"""Check immutable upstream artifact IDs and metadata without printing credentials."""
import json
import os
from pathlib import Path
import urllib.request


ROOT = Path(__file__).resolve().parent
PINS = json.loads((ROOT / "artifact-pins.json").read_text(encoding="utf-8"))
REPO = "mxxm-t/llmx"


def main():
    token = os.environ["GH_TOKEN"]
    checked = []
    for key in ("matched_build", "probe_build"):
        pin = PINS[key]
        req = urllib.request.Request(
            f"https://api.github.com/repos/{REPO}/actions/artifacts/{pin['artifact_id']}",
            headers={"Accept": "application/vnd.github+json",
                     "Authorization": "Bearer " + token,
                     "X-GitHub-Api-Version": "2022-11-28",
                     "User-Agent": "llmx-tq-hosted-capture-preflight"})
        with urllib.request.urlopen(req, timeout=30) as response:
            item = json.load(response)
        assert item["id"] == pin["artifact_id"]
        assert item["name"] == pin["artifact_name"]
        assert item["digest"] == pin["artifact_digest"]
        assert item["expired"] is False
        assert item["workflow_run"]["id"] == pin["workflow_run_id"]
        assert item["workflow_run"]["head_sha"] == pin["workflow_head"]
        checked.append({"kind": key, "id": item["id"], "digest": item["digest"],
                        "workflow_run_id": item["workflow_run"]["id"],
                        "workflow_head": item["workflow_run"]["head_sha"]})
    print(json.dumps({"passed": True, "artifacts": checked}, separators=(",", ":")))


if __name__ == "__main__":
    main()
