import math
import os
from pathlib import Path
import re
import tempfile

import common
import f32
from chat import BANNER, replies


# Thread counts are the CPU backend's, and a device backend reports none, so every command here names the CPU instead of the configured device.
def invoke(args, data=None):
    p = common.run_process(args + ["--device", "cpu"], input=data, timeout=30)
    assert p.returncode == 0, (args, p.returncode, p.stderr)
    return p.stdout, p.stderr


def plain_cgroup_path(path):
    """A cgroup path without its trailing slashes when it is absolute and does not climb with "..", else None."""
    if not path.startswith("/") or "/../" in path + "/":
        return None
    return path.rstrip("/") or "/"


def unescape(field):
    """A mountinfo field with its octal escapes decoded, "\\040" for a space."""
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), field)


def cgroup_mounts(text):
    """(root, mount point) of each cgroup v2 mount and of each v1 mount holding the cpu controller in mountinfo text, in the order listed."""
    mounts = {"v2": [], "v1": []}
    for line in text.splitlines():
        fields = line.split()
        if "-" not in fields[6:]:
            continue
        dash = fields.index("-", 6)
        if len(fields) < dash + 4:
            continue
        kind = "v2" if fields[dash + 1] == "cgroup2" else "v1" if fields[dash + 1] == "cgroup" and "cpu" in fields[dash + 3].split(",") else None
        root, point = plain_cgroup_path(unescape(fields[3])), unescape(fields[4])
        if kind and root and point.startswith("/"):
            mounts[kind].append((root, point))
    return mounts


def cgroup_quotas():
    """The CPUs each cgroup CPU quota over this process allows, rounded up: v2's cpu.max and v1's cfs quota over its period, in its own cgroup and each above it up to its mount's point."""
    try:
        lines = Path("/proc/self/cgroup").read_text().splitlines()
        mounts = cgroup_mounts(Path("/proc/self/mountinfo").read_text())
    except OSError:
        return []
    quotas = []
    taken = set()
    for line in lines:
        if line.count(":") < 2:
            continue
        hierarchy, controllers, path = line.split(":", 2)
        if hierarchy == "0" and not controllers:
            kind, files = "v2", ("cpu.max",)
        elif "cpu" in controllers.split(","):
            kind, files = "v1", ("cpu.cfs_quota_us", "cpu.cfs_period_us")
        else:
            continue
        # Only the first line of each kind counts, and one whose path climbs or is relative reads nothing.
        if kind in taken:
            continue
        taken.add(kind)
        path = plain_cgroup_path(path)
        if path is None:
            continue
        for root, point in mounts[kind]:
            if root != "/" and path != root and not path.startswith(root + "/"):
                continue
            parts = [part for part in (path if root == "/" else path[len(root):]).split("/") if part]
            for depth in range(len(parts), -1, -1):
                try:
                    fields = " ".join((Path(point, *parts[:depth]) / name).read_text() for name in files).split()
                except OSError:
                    continue
                if len(fields) == 2 and fields[0] not in ("max", "-1"):
                    quotas.append(-(-int(fields[0]) // int(fields[1])))
            break
    return quotas


def windows_kernel():
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32")
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.GetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    kernel.GetProcessGroupAffinity.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.USHORT), ctypes.POINTER(wintypes.USHORT)]
    kernel.GetActiveProcessorGroupCount.restype = wintypes.WORD
    kernel.GetActiveProcessorCount.argtypes = [wintypes.WORD]
    kernel.GetActiveProcessorCount.restype = wintypes.DWORD
    kernel.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p]
    return ctypes, wintypes, kernel


def job_quota():
    """The CPUs the CPU rate hard cap of this process's job object allows over the active processors, rounded up, on Windows; None elsewhere or without one."""
    if os.name != "nt":
        return None
    ctypes, wintypes, kernel = windows_kernel()
    info = (wintypes.DWORD * 2)()  # JOBOBJECT_CPU_RATE_CONTROL_INFORMATION: ControlFlags, CpuRate
    if not kernel.QueryInformationJobObject(None, 15, info, ctypes.sizeof(info), None):
        return None
    flags, rate = info[0], info[1]
    processors = kernel.GetActiveProcessorCount(0xFFFF)
    if flags & 0x5 != 0x5 or not 1 <= rate <= 10000 or not processors:
        return None
    return -(-rate * processors // 10000)


def affinity():
    """The CPUs this process's affinity allows: on Windows the process mask when its threads lie in one processor group, else every active processor of their groups; None where it cannot be read."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    if os.name != "nt":
        return None
    ctypes, wintypes, kernel = windows_kernel()
    process = kernel.GetCurrentProcess()
    count = wintypes.USHORT(max(kernel.GetActiveProcessorGroupCount(), 1))
    groups = (wintypes.USHORT * count.value)()
    if not kernel.GetProcessGroupAffinity(process, ctypes.byref(count), groups) or not count.value:
        return None
    if count.value > 1:
        return sum(kernel.GetActiveProcessorCount(groups[i]) for i in range(count.value)) or None
    allowed, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not kernel.GetProcessAffinityMask(process, ctypes.byref(allowed), ctypes.byref(system)):
        return None
    return bin(allowed.value).count("1") or None


def expected_automatic():
    """The count the CPU backend should take when given none, read here apart from llmx: the fewest of the hardware threads, the CPUs the affinity allows and the CPUs the CPU quotas allow, four when none can be read, from 1 to 64."""
    counts = [n for n in [os.cpu_count(), affinity(), job_quota()] + cgroup_quotas() if n]
    return max(1, min(64, min(counts) if counts else 4))


def check_perplexity_threads(model, automatic, weights_sha256):
    fixture = next(case for case in f32.golden("baseline_f32.json")["fixtures"] if not case["tied"])
    # The model under test holds the untied fixture's weights, whose NLL is the reference here.
    assert fixture["weights_sha256"] == weights_sha256
    case = fixture["perplexity"][0]
    worst = 0.0
    checked = 0
    bounds = set()
    for count in (None, 0, 1, 4):
        for batch in (None, 0, 1, 3):
            for alias in (("--threads-batch", "-tb") if batch is not None else ("--threads-batch",)):
                # Per-token scoring runs on --threads and refuses a batch count, as the cli component checks.
                for per_token in ((False, True) if batch is None else (False,)):
                    flags = ["--verbose", "--ubatch", "3",
                             "--cache-type-k", "f32", "--cache-type-v", "f32",
                             "-c", str(case["context"])]
                    if count is not None:
                        flags += ["--threads", str(count)]
                    if batch is not None:
                        flags += [alias, str(batch)]
                    if per_token:
                        flags += ["--per-token"]
                    out, err = invoke(["perplexity", str(model), f32.TEXTS[-1]] + flags)
                    expected = count or automatic
                    if not per_token:
                        expected = batch or expected
                    counts = re.findall(rb"^threads: (prefill|decode) (\d+)\r?$", err, re.M)
                    phase = b"decode" if per_token else b"prefill"
                    assert counts == [(phase, str(expected).encode())], (count, batch, alias, per_token, counts, expected)
                    error = abs(float(common.perplexity_fields(out.decode("utf-8"))["mean NLL"]) - case["mean_nll"])
                    bound = common.hf_execution(err.decode("utf-8"))["nll"]
                    assert math.isfinite(error) and error < bound, (count, batch, per_token, error, bound)
                    bounds.add(bound)
                    worst = max(worst, error)
                    checked += 1
    print("threads: perplexity effective counts/aliases and HF NLL, %d cases, max error %.8f, witnessed bounds %s  [ok]" % (checked, worst, sorted(bounds)))


def run():
    bench = ["bench", "--size", "32", "--iters", "1", "--p", "1", "--n", "1"]
    automatic = None
    for count in (None, 0, 1, 4):
        flags = [] if count is None else ["--threads", str(count)]
        out, _ = invoke(bench + flags)
        match = re.search(rb"^bench: threads (\d+)\r?$", out, re.M)
        assert match, out
        actual = int(match[1])
        if count is None:
            automatic = actual
            assert automatic == expected_automatic(), (automatic, expected_automatic())
        assert actual == (count or automatic), (count, actual, automatic)

    fixture = f32.golden("baseline_chat.json")
    case = fixture["cases"][0]
    spec = case["spec"]
    assert spec["template"] == "{{ messages[-1]['content'] }}"
    assert spec["max_tokens"] == 1
    with tempfile.TemporaryDirectory(prefix="llmx_threads_") as directory:
        model = Path(directory) / "threads.gguf"
        f32.write_model(model, fixture["weights"], spec["template"])
        check_perplexity_threads(model, automatic, fixture["weights_sha256"])
        for count in (None, 0, 1, 4):
            for batch in (None, 0, 1, 3):
                flags = ["--temp", "0", "-n", "1", "--verbose", "--ubatch", "3"]
                if count is not None:
                    flags += ["--threads", str(count)]
                if batch is not None:
                    flags += ["--threads-batch", str(batch)]
                decode = count or automatic
                prefill = batch or decode
                for command in ("generate", "chat"):
                    args = [command, str(model)]
                    if command == "generate":
                        args += [spec["inputs"][0]]
                        data, turns = None, 1
                    else:
                        args += ["--system", ""]
                        data = ("\n".join(spec["inputs"]) + "\n").encode()
                        turns = len(spec["inputs"])
                    out, err = invoke(args + flags, data)
                    counts = re.findall(rb"^threads: (prefill|decode) (\d+)\r?$", err, re.M)
                    expected = [(b"prefill", str(prefill).encode()),
                                (b"decode", str(decode).encode())] * turns
                    assert counts == expected, (command, count, batch, counts, expected)
                    if command == "chat":
                        assert common.cli_stdout(out) == BANNER + replies(case, turns), out
                    else:
                        assert common.generate_text(out) + b"\n" == replies(case, turns), out
    print("threads: auto count %d, explicit counts, prefill restore, follow-up chat and HF replies  [ok]" % automatic)
    return True
