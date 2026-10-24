#!/usr/bin/env python3
"""Choose a fresh four-CPU hosted-runner admission for one CPU capture."""
import json
import os
from pathlib import Path
import re
import sys


def main():
    allowed = sorted(os.sched_getaffinity(0))
    chosen = None
    for start in allowed:
        candidate = list(range(start, start + 4))
        if all(cpu in allowed for cpu in candidate):
            chosen = candidate
            break
    if chosen is None:
        raise RuntimeError("runner does not expose four contiguous CPUs")
    admission = {"cpu_set": f"{chosen[0]}-{chosen[-1]}", "cpu_quota": 4,
                 "threads": 4, "device": None,
                 "container_name": "llmx-tq-clean-depth-hosted-" + sys.argv[1]}
    assert re.fullmatch(r"llmx-tq-clean-depth-[a-z0-9-]+", admission["container_name"])
    Path(sys.argv[2]).write_text(json.dumps(admission, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
