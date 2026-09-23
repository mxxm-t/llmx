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


# Commands that take --device.
# LLMX_DEVICE is test configuration, like LLMX_BASELINE_GGUF: it never reaches the binary except as this flag.
DEVICE_COMMANDS = {"generate", "chat", "logits", "perplexity", "bench", "serve"}


def device_args(args, cache=None):
    args = list(args)
    if not args or args[0] not in DEVICE_COMMANDS:
        return args
    device = os.environ.get("LLMX_DEVICE")
    if device and "--device" not in args:
        args += ["--device", device]
    # LLMX_CACHE_TYPE, set by run_tests.py --cache-type, runs the same commands with both cache sides stored as that type; test configuration like LLMX_DEVICE, reaching the binary only as flags.
    # `cache` is a component asking for a type because its fixtures need it, which an explicit LLMX_CACHE_TYPE overrides.
    want = os.environ.get("LLMX_CACHE_TYPE") or cache
    if want and "--cache-type-k" not in args:
        args += ["--cache-type-k", want, "--cache-type-v", want]
    return args


def run(args, cwd=None, cache=None):
    """Run the llmx CLI, returning (returncode, stdout_text)."""
    p = subprocess.run([exe_path()] + device_args(args, cache), capture_output=True, text=True,
                       encoding="utf-8", cwd=cwd or ROOT)
    # A failure's diagnostic is on stderr; hand it back with the output so a caller can tell a missing device kernel from a wrong answer.
    return p.returncode, p.stdout if p.returncode == 0 else p.stdout + p.stderr


def device_lacks_kernel(rc, out):
    """True when the selected device refused the model for want of a kernel,
    which a test reports as skipped rather than failed."""
    return (rc != 0 and bool(os.environ.get("LLMX_DEVICE")) and
            ("unsupported matrix type" in out or "unsupported embedding type" in out))


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


# The components that check exact f32 arithmetic against independently generated fixtures run the CLI through this, so they keep their f32 cache sides now that the runtime stores f16 by default.
def run_f32_cache(args, cwd=None):
    return run(args, cwd, cache="f32")
