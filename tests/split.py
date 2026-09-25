import os
import subprocess
import tempfile

import common
import f32
import moe


# The layer split against one device through llmx-split-check (tools/split_check.cpp), on CPU backends: raw logits compared bit for bit over the prompt path, the prefill a split pipelines over its stages, greedy decode steps and a decoding sequence beside a fresh prompt.
# The tool names its own devices and cache type, so the configured device, shares and cache type do not reach it.
# Each split runs with f16 caches, the default, and with f32 caches, since a split is exact at either.
# Three decode steps after the 13-token text fill the tiny models' 16-token context.
UBATCHES = (1, 3, 16)
CACHE_TYPES = ("f16", "f32")
STEPS = 3


def tool_path():
    return os.path.join(os.path.dirname(common.exe_path()), "llmx-split-check" + (".exe" if os.name == "nt" else ""))


def run(require=False):
    tool = tool_path()
    if not os.path.exists(tool):
        assert not require, "split: %s not found beside the executable" % tool
        print("split: SKIP - %s not found beside the executable" % tool)
        return True
    runs = 0
    with tempfile.TemporaryDirectory(prefix="llmx_split_") as directory:
        text = os.path.join(directory, "text.txt")
        with open(text, "w", encoding="utf-8", newline="") as f:
            f.write(f32.TEXTS[-1])
        models = [(f32.write_model(os.path.join(directory, "tiny-f32-%s.gguf" % name), f32.tensors(tied)), ["cpu,cpu"])
                  for name, tied in (("tied", True), ("untied", False))]
        models.append((f32.write_model(os.path.join(directory, "tiny-moe.gguf"), moe.tensors(), config=moe.CONFIG, arch="qwen3moe"),
                       ["cpu,cpu", "cpu,cpu,cpu"]))
        for model, splits in models:
            for split in splits:
                for ubatch in UBATCHES:
                    for cache in CACHE_TYPES:
                        args = [tool, model, text, "cpu", split, str(STEPS), str(ubatch), cache]
                        p = subprocess.run(args, capture_output=True, encoding="utf-8", errors="replace", timeout=60)
                        assert p.returncode == 0 and ": 13 tokens, %s caches;" % cache in p.stdout and "bit-identical" in p.stdout, \
                            "split differs from one device: %s\n%s%s" % (" ".join(args[1:]), p.stdout, p.stderr)
                        runs += 1
    print("split: %d runs of the tiny F32 (tied, untied) and MoE models over 2 and 3 CPU backends at ubatch %s with %s caches, "
          "bit-identical to one  [ok]" % (runs, "/".join(map(str, UBATCHES)), " and ".join(CACHE_TYPES)))
    return True


if __name__ == "__main__":
    run()
