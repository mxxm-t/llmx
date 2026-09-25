from pathlib import Path
import re

import common


def run():
    p = common.run_process(["--version"], timeout=10)
    assert p.returncode == 0 and not p.stderr, (p.returncode, p.stderr)
    output = p.stdout.decode("ascii").strip()
    match = re.fullmatch(r"llmx (\d+\.\d+\.\d+)\+(unknown|g[0-9a-f]{12,40}(?:\.dirty)?)", output)
    assert match, output
    config = (Path(common.ROOT) / "CMakeLists.txt").read_text(encoding="ascii")
    release = re.search(r"project\(llmx VERSION ([0-9.]+)", config)
    assert release and match[1] == release[1], (output, config)
    usage = common.run_process([], timeout=10)
    assert usage.returncode == 1 and usage.stdout.startswith(p.stdout.rstrip() + b" - "), usage.stdout
    print("version: release/build identifier and usage banner agree  [ok]")
    return True
