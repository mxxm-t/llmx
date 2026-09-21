import os
import subprocess
import sys
import struct

# Shared helpers for synthetic tests and the optional real-model HF baseline.

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(ROOT, "llmx.exe" if os.name == "nt" else "llmx")


def exe_path():
    if not os.path.exists(EXE):
        raise SystemExit("Executable not found: %s. Build first and pass --exe to tests/run_tests.py." % EXE)
    return EXE


# Commands that take --device. LLMX_DEVICE is test configuration, like
# LLMX_BASELINE_GGUF: it never reaches the binary except as this flag.
DEVICE_COMMANDS = {"generate", "chat", "logits", "perplexity", "bench"}


def device_args(args):
    device = os.environ.get("LLMX_DEVICE")
    if device and args and args[0] in DEVICE_COMMANDS and "--device" not in args:
        return list(args) + ["--device", device]
    return list(args)


def run(args, cwd=None):
    """Run the llmx CLI, returning (returncode, stdout_text)."""
    p = subprocess.run([exe_path()] + device_args(args), capture_output=True, text=True,
                       encoding="utf-8", cwd=cwd or ROOT)
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
