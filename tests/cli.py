import math
import os
import re
import tempfile

import common
import f32
import moe


# The command-line surface the numerical components do not reach: a device that cannot run, and the file listing.
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


def run():
    with tempfile.TemporaryDirectory(prefix="llmx_cli_") as directory:
        model = f32.write_model(os.path.join(directory, "tiny-f32.gguf"), f32.tensors(False))
        check_device(model)
        check_info(directory)
    print("cli: a Vulkan device refused without the backend or without the device, and info's architecture, layers and tensors  [ok]")
    return True


if __name__ == "__main__":
    run()
