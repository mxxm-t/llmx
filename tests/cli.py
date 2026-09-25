import math
import os
import re
import subprocess
import tempfile

import common
import f32
import moe


# The command-line surface the numerical components do not reach: a device that cannot run, the file listing, and the command lines refused as usage errors.
# It needs no device, so it runs in every job; each command here that takes a device names its own.


def check_device(model):
    """A build without the Vulkan backend refuses a Vulkan device, and a Vulkan build refuses a device it cannot open; neither runs on the CPU instead.
    The model commands and the synthetic bench each reach the device through their own call."""
    for args in (["logits", model, "a"], ["bench", "--size", "32", "--iters", "1", "--p", "1", "--n", "1"]):
        p = common.run_process(args + ["--device", "vulkan:0"], text=True, timeout=120)
        messages = ("error: --device vulkan: this build has no Vulkan backend", "error: vulkan: ")
        if p.returncode == 0:
            # Only a Vulkan build with a device 0 gets here, so an index no machine has stands in for the missing device.
            p = common.run_process(args + ["--device", "vulkan:999999"], text=True, timeout=120)
            messages = ("error: vulkan: no device at index 999999",)
        assert p.returncode == 1 and not p.stdout and any(m in p.stderr for m in messages), (args, p.returncode, p.stdout, p.stderr)


def check_info(directory):
    """`info` names the architecture and the layer count, and lists every tensor written with its type, shape and size."""
    weights = moe.tensors()
    model = f32.write_model(os.path.join(directory, "tiny-moe.gguf"), weights, config=moe.CONFIG, arch="qwen3moe")
    rc, out = common.run(["info", model])
    assert rc == 0, out
    lines = out.splitlines()
    assert '  general.architecture = "qwen3moe"' in lines, out
    assert "  qwen3moe.block_count = %d" % moe.CONFIG["block_count"] in lines, out
    # write_model adds one Q8_0 block ahead of the weights.
    expected = [("Q8_0", "unused.weight", [32], 34)]
    expected += [("F32", name, shape, 4 * math.prod(shape)) for name, _, shape, _ in weights]
    assert "Tensors: %d" % len(expected) in lines, out
    listed = [(kind, name, [int(n) for n in shape.split(", ")], int(size)) for kind, name, shape, size in
              re.findall(r"^  (\S+) (\S+) shape=\[([\d, ]+)\] elements=\d+ bytes=(\d+)$", out, re.M)]
    assert listed == expected, listed


def usage_error(args, page):
    """`args` must be refused as a usage error: status 2, nothing on stdout, and `page`'s help on stderr followed by the reason.
    The line is run as written, without the configured device flags, so a check of missing arguments stays one."""
    p = subprocess.run([common.exe_path()] + args, capture_output=True, timeout=10)
    err = p.stderr.decode("utf-8", "replace")
    assert p.returncode == 2 and not p.stdout, (args, p.returncode, p.stdout, err)
    assert err.startswith("llmx ") and ("Usage: llmx " + page) in err and "\nerror: " in err, (args, err)


def check_usage_errors():
    """Every command line a command cannot take exits with status 2 and the command's page, before any model file is opened."""
    # An unknown command is named and refused with status 2, with or without help after it.
    for args in (["frobnicate"], ["frobnicate", "--help"]):
        p = subprocess.run([common.exe_path()] + args, capture_output=True, timeout=10)
        assert p.returncode == 2 and not p.stdout and b"unknown command: frobnicate" in p.stderr, (args, p)
    # Every refusal below happens while the command line is read, before the model file is opened, so a path that does not exist is enough.
    model = "missing.gguf"
    for args, page in ((["--help", "chat"], "<command>"), (["serve"], "serve"), (["pull"], "pull"), (["pull", "Qwen/Qwen3-0.6B-GGUF"], "pull"),
                       (["pull", "a/b:q8_0", "--parallel", "0"], "pull"), (["pull", "a/b:q8_0", "--parallel", "17"], "pull"),
                       (["pull", "a/b:q8_0", "--mirror", "x"], "pull"), (["pull", "a/b:q8_0", "--revision"], "pull"),
                       (["generate"], "generate"), (["generate", model], "generate"),
                       (["info"], "info"), (["tokenize", model], "tokenize"), (["detokenize", model, "1", "2"], "detokenize"),
                       (["dequantize", model, "a.json"], "dequantize"), (["quantize", "a.json", "a.bin", "a.gguf", "q2_k"], "quantize"),
                       (["logits", model], "logits"), (["perplexity", model, "--file"], "perplexity")):
        usage_error(args, page)
    # Flags and arguments a command would ignore or overwrite are refused rather than dropped, and so is a stream of experts with none on the CPU.
    for args, page in ((["chat", model, "hello"], "chat"), (["generate", model, "a", "b"], "generate"),
                       (["generate", model, "a", "--stop", "x", "--stop", "y"], "generate"),
                       (["generate", model, "a", "--system", "x"], "generate"),
                       (["logits", model, "text", "--file"], "logits"), (["logits", model, "-f", "a.txt", "--file", "b.txt"], "logits"),
                       (["perplexity", model, "text", "-f", "a.txt"], "perplexity"),
                       (["bench", "--threads", "2", "--r", "2"], "bench"), (["bench", "--cpu-moe"], "bench"),
                       (["bench", "--model", model, "--size", "64"], "bench"), (["bench", "--model", model, "--iters", "2"], "bench"),
                       (["bench", "--model", model, "--profile", "--device", "cpu"], "bench"),
                       (["bench", "--model", model, "--profile", "--device", "vulkan:0,vulkan:1"], "bench"),
                       (["bench", "--model", model, "--depth", "4", "--seqs", "2", "--device", "cpu"], "bench"),
                       (["logits", model, "a", "--moe-stream-from", "4"], "logits"),
                       (["serve", model, "--moe-stream-from", "1", "--n-cpu-moe", "0"], "serve")):
        usage_error(args, page)
    # Numbers are decimal, whole where the flag counts, and within the flag's range; a flag at the end of the line has no value.
    numbers = [("-n", ["0", "-1", "x", ""]), ("--threads", ["-1", "-0", "x", "4x", "+4", "0x4", "1.5", "2147483648", ""]),
               ("-tb", ["-1"]), ("--ubatch", ["0", "-5"]), ("--topk", ["-1"]), ("--seed", ["-1", "0x10", "18446744073709551616"]),
               ("--temp", ["-0.5", "x", "nan", "inf", "1e39", "0x1p1", "1,5"]), ("--topp", ["1.5", "-0.1"]), ("--penalty", ["0.5"]),
               ("--n-cpu-moe", ["-2"]), ("--moe-stream-from", ["-1"]), ("--layer-shares", ["1,x", "-1", "1000000"]),
               ("--cache-type-k", ["q8_0", "F16", ""]), ("-ctv", ["bf16"])]
    for flag, values in numbers:
        for value in values:
            usage_error(["generate", model, "a", flag, value], "generate")
        usage_error(["generate", model, "a", flag], "generate")
    for args, page in ((["serve", model, "--port", "65536"], "serve"), (["serve", model, "--port", "-1"], "serve"),
                       (["serve", model, "--max-seqs", "0"], "serve"), (["serve", model, "--max-queue", "0"], "serve"),
                       (["serve", model, "-c", "0"], "serve"), (["logits", model, "a", "--top", "0"], "logits"),
                       (["logits", model, "a", "--last", "0"], "logits"), (["perplexity", model, "a", "--chunks", "-1"], "perplexity"),
                       (["bench", "--size", "48"], "bench"), (["bench", "--p", "0"], "bench"), (["bench", "--depth", "-1"], "bench")):
        usage_error(args, page)


def run():
    with tempfile.TemporaryDirectory(prefix="llmx_cli_") as directory:
        model = f32.write_model(os.path.join(directory, "tiny-f32.gguf"), f32.tensors(False))
        check_device(model)
        check_info(directory)
    check_usage_errors()
    print("cli: a Vulkan device refused without the backend or without the device, info's architecture, layers and tensors, "
          "and usage errors exiting 2 with the command's page  [ok]")
    return True


if __name__ == "__main__":
    run()
