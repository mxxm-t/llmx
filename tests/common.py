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
    # LLMX_LAYER_SHARES, set by run_tests.py --layer-shares, fixes each listed device's proportion of the layers, so a split the fit would not choose on this machine is tested anyway.
    shares = os.environ.get("LLMX_LAYER_SHARES")
    if shares and "--layer-shares" not in args:
        args += ["--layer-shares", shares]
    # LLMX_CACHE_TYPE, set by run_tests.py --cache-type, runs the same commands with both cache sides stored as that type; test configuration like LLMX_DEVICE, reaching the binary only as flags.
    # `cache` is a component asking for a type because its fixtures need it, which an explicit LLMX_CACHE_TYPE overrides.
    want = os.environ.get("LLMX_CACHE_TYPE") or cache
    if want and "--cache-type-k" not in args:
        args += ["--cache-type-k", want, "--cache-type-v", want]
    return args


# How far from the reference's 5th logit two tokens may sit and still trade places at the top-5 boundary: near ties there reorder with the rounding of any backend that sums in another order.
TOP5_TIE_MARGIN = 0.1


def top5_overlap(ids, ref_ids, ref_logits, margin=TOP5_TIE_MARGIN):
    """The reference's top-5 tokens found in `ids[:5]`, a boundary swap counting as agreement.

    A reference top-5 token missing from `ids[:5]` is forgiven only when the reference puts it within `margin` of its own 5th logit,
    and only against a token `ids[:5]` holds instead that the reference puts within `margin` below that logit; a strong token that
    vanishes, or one pulled in from further down, still counts as a miss.
    """
    top, fifth = set(ref_ids[:5]), ref_logits[4]
    ref = dict(zip(ref_ids, ref_logits))
    got = set(ids[:5])
    missing = [t for t in top - got if ref[t] - fifth <= margin]
    extra = [t for t in got - top if t in ref and fifth - ref[t] <= margin]
    return len(got & top) + min(len(missing), len(extra))


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
