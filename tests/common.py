import os
import subprocess
import sys
import struct

# Shared helpers for the llmx test suite. All tests generate their own
# fixtures in a temp dir and clean up after themselves; no real models needed.

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(ROOT, "llmx.exe")


def exe_path():
    if not os.path.exists(EXE):
        raise SystemExit("llmx.exe not found. Run build.bat (Windows) or the CMake build first.")
    return EXE


def run(args, cwd=None):
    """Run the llmx CLI, returning (returncode, stdout_text)."""
    p = subprocess.run([exe_path()] + args, capture_output=True, text=True, cwd=cwd or ROOT)
    return p.returncode, p.stdout


def write_bin(path, floats):
    """Write a list of floats as little-endian f32."""
    with open(path, "wb") as f:
        for x in floats:
            f.write(struct.pack("<f", x))


def read_bin_floats(path):
    with open(path, "rb") as f:
        data = f.read()
    return struct.unpack("<%df" % (len(data) // 4), data)


def max_err(a, b):
    return max(abs(x - y) for x, y in zip(a, b))
