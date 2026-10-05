import os
import re
from pathlib import Path
import tempfile

import common
import f32
import tensor_split


APPEND = ("{% for message in messages %}"
          "{% if message['role'] == 'assistant' %}b{% endif %}"
          "{{ message['content'] }}{% endfor %}"
          "{% if add_generation_prompt %}b{% endif %}")
CASES = [
    {"name": "no-generation-suffix", "template": "{{ messages[-1]['content'] }}",
     "inputs": ["a", "ab", "abc"], "max_tokens": 1},
    {"name": "rewrite-longer", "template": "{{ messages[-1]['content'] }}",
     "inputs": ["a", "aaa"], "max_tokens": 1},
    {"name": "append", "template": APPEND,
     "inputs": ["a", "c", "d"], "max_tokens": 1},
    {"name": "token-limit", "template": APPEND,
     "inputs": ["a", "c"], "max_tokens": 2},
    {"name": "stop-before-feed", "template": APPEND,
     "inputs": ["a", "c", "d"], "max_tokens": 2, "stop": "J"},
    {"name": "eos", "template": "{{ messages[-1]['content'] }}",
     "inputs": ["a", "ab", "abc"], "max_tokens": 1, "eos_id": 205},
    {"name": "rewrite-and-shorten",
     "template": "{{ messages[-1]['content'] }}{% if add_generation_prompt %}b{% endif %}",
     "inputs": ["abc", "a", "c", "c"], "max_tokens": 1},
    {"name": "identical-prompt", "template": "ab",
     "inputs": ["a", "c"], "max_tokens": 1, "eos_id": 74},
    {"name": "elif-else-and-trailing-text",
     "template": "{% if messages[-1]['content'] == 'a' %}a"
                 "{% elif messages[-1]['content'] == 'b' %}ab{% else %}abc{% endif %}d",
     "inputs": ["a", "b", "c"], "max_tokens": 1},
]
BANNER = b"Chat ready (type your message; Ctrl+C to quit)\n"


def replies(case, turns=None):
    """What `chat` prints after its banner for the first `turns` turns of a golden case, every turn by default: each reply's bytes and a line feed."""
    return b"".join(bytes(turn["reply_ids"]) + b"\n" for turn in case["turns"][:turns])


def run():
    fixture = f32.golden("baseline_chat.json")
    assert [case["spec"] for case in fixture["cases"]] == CASES
    # Under a tensor width the model is the tensor-split fixtures' shape, which a group splits whole, and a reply is held to the same conversation at every thread count and ubatch rather than to HF's, which the tensor-split component holds the shape's numerics to.
    grouped = common.tensor_width() > 1
    weights = f32.tensors(True, config=tensor_split.CONFIG, vocab=tensor_split.VOCAB) if grouped else fixture["weights"]
    shape = {"config": dict(tensor_split.CONFIG, context_length=f32.CONFIG["context_length"]), "tokens": tensor_split.TOKENS} if grouped else {}
    def write(model, template, eos_id=None):
        f32.write_model(model, weights, template, eos_id, **shape)
    with tempfile.TemporaryDirectory(prefix="llmx_chat_") as directory:
        model = Path(directory) / "chat.gguf"
        for case in fixture["cases"]:
            spec = case["spec"]
            write(model, spec["template"], spec.get("eos_id"))
            expected = None if grouped else replies(case)
            # Under a tensor width the first reply is held to `generate` on the same group reading the first turn's prompt as the reference rendered it, which holds no reply of the reference's model; the later turns' rendered prompts do, so they are held to the same conversation at every thread count and ubatch.
            first = None
            if grouped:
                prompt = Path(directory) / "first.txt"
                prompt.write_bytes(bytes(case["turns"][0]["prompt_ids"]))
                g = common.run_process(["generate", str(model), "--file", str(prompt), "--temp", "0", "-n", str(spec["max_tokens"]), "--threads", "1"]
                                       + (["--stop", spec["stop"]] if "stop" in spec else []), timeout=30)
                assert g.returncode == 0, (spec["name"], g.returncode, g.stderr)
                first = common.generate_text(g.stdout) + b"\n"
            for threads in (1, 4):
                for ubatch in (1, 3, 16):
                    args = ["chat", str(model), "--system", "",
                            "--temp", "0", "-n", str(spec["max_tokens"]),
                            "--threads", str(threads), "--ubatch", str(ubatch)]
                    if "stop" in spec:
                        args += ["--stop", spec["stop"]]
                    p = common.run_process(args, input=("\n".join(spec["inputs"]) + "\n").encode(), timeout=30)
                    assert p.returncode == 0, (spec["name"], p.returncode, p.stderr)
                    output = common.cli_stdout(p.stdout)
                    if expected is None:
                        assert output.startswith(BANNER), output
                        expected = output[len(BANNER):]
                        assert expected.startswith(first), (spec["name"], expected, first)
                    assert output == BANNER + expected, (spec["name"], output, expected)
        write(model, "{% if false %}x{% endif %}")
        p = common.run_process(["chat", str(model)], input=b"a\n", timeout=30)
        assert p.returncode == 1 and b"empty prompt" in p.stderr, (p.returncode, p.stderr)
        # A template the renderer refuses stops chat and serve before either takes a turn or listens, while generate, which renders no template, still runs on the file.
        write(model, "{% filter upper %}{{ messages[0]['content'] }}{% endfilter %}")
        p = common.run_process(["chat", str(model)], input=b"a\n", timeout=30)
        refusal = b"error: the model's chat template is refused: the tag 'filter' is not supported"
        assert p.returncode == 1 and refusal in p.stderr and not p.stdout, (p.returncode, p.stderr, p.stdout)
        p = common.run_process(["serve", str(model), "--host", "127.0.0.1", "--port", str(common.free_port())], timeout=60)
        assert p.returncode == 1 and refusal in p.stderr and b"serving" not in p.stderr, (p.returncode, p.stderr)
        p = common.run_process(["generate", str(model), "a", "-n", "2", "--temp", "0"], timeout=30)
        assert p.returncode == 0, (p.returncode, p.stderr)
        common.generate_text(p.stdout)
        for cmd in (["generate", str(model), "a", "--chat", "-n", "2"], ["logits", str(model), "a", "--chat"]):
            p = common.run_process(cmd, timeout=30)
            assert p.returncode == 1 and refusal in p.stderr and not p.stdout, (cmd, p.returncode, p.stderr, p.stdout)
        # --chat reads the text as one user message rendered with the generation prompt and no system message, as the chat routes render it: "d" is "adb" here.
        write(model, "{% for message in messages %}{% if message['role'] == 'user' %}a{% else %}c{% endif %}"
                     "{{ message['content'] }}{% endfor %}{% if add_generation_prompt %}b{% endif %}")
        with tempfile.TemporaryDirectory(prefix="llmx_chat_ids_") as ids_dir:
            then = Path(ids_dir) / "then.ids"
            then.write_text("3 1 4")
            pairs = [(["generate", str(model), "d", "--chat"], ["generate", str(model), "adb"], ["-n", "3", "--temp", "0"]),
                     (["logits", str(model), "d", "--chat"], ["logits", str(model), "adb"], ["--top", "5"]),
                     (["logits", str(model), "d", "--chat"], ["logits", str(model), "adb"], ["--then-ids", str(then), "--last", "4", "--top", "3"])]
            for chat_args, raw_args, rest in pairs:
                got, want = (common.run_process(a + rest, timeout=30) for a in (chat_args, raw_args))
                assert got.returncode == 0 and want.returncode == 0, (chat_args, got.stderr, want.stderr)
                if chat_args[0] == "generate":
                    assert common.generate_text(got.stdout) == common.generate_text(want.stdout), (got.stdout, want.stdout)
                else:
                    assert got.stdout == want.stdout and got.stdout.startswith(b"tokens: "), (rest, got.stdout, want.stdout)
        case = fixture["cases"][0]
        spec = case["spec"]
        write(model, spec["template"])
        expected = None if grouped else replies(case)
        args = ["chat", str(model), "--system", "", "--temp", "0", "-n", "1", "--threads", "1"]
        for verbose in (False, True):
            p = common.run_process(args + (["--verbose"] if verbose else []),
                                   input=("\n".join(spec["inputs"]) + "\n").encode(), timeout=30)
            assert p.returncode == 0, p.stderr
            if expected is None:
                expected = common.cli_stdout(p.stdout)[len(BANNER):]
            assert common.cli_stdout(p.stdout) == BANNER + expected
            if not verbose:
                requested = os.environ.get("LLMX_DTYPE", "auto").encode("ascii")
                assert re.fullmatch(rb"dtype: " + requested + rb" -> (?:f16|bf16|f32|int8) \(model declares bf16\); [^\r\n]+\r?\n", p.stderr), p.stderr
                continue
            percents = [int(x) for x in re.findall(rb"Loading tensor data: (\d+)%", p.stderr)]
            assert percents[0] == 0 and percents[-1] == 100
            assert all(a < b for a, b in zip(percents, percents[1:])), percents
            assert p.stderr.index(b"Reading model metadata") < p.stderr.index(b"Loading tensor data")
            assert p.stderr.index(b"Loading tensor data: 100%") < p.stderr.index(b"Preparing model")
            assert p.stderr.count(b"Processing ") == len(spec["inputs"])
            assert p.stderr.count(b"Generating...") == len(spec["inputs"])
        # The devices are made before the file is read, so neither of these reads it; no machine has a Vulkan device at index 999999.
        # A bad cache type or layer share is a usage error before any file is opened, which the cli component checks.
        for bad in (["--device", "bogus"], ["--device", "vulkan:999999"]):
            p = common.run_process(args + ["--verbose"] + bad, input=b"a\n", timeout=30)
            assert p.returncode == 1 and b"Reading model metadata" not in p.stderr, (bad, p.returncode, p.stderr)
    print("chat: follow-up replies %s; append, rewrite, reset, stop/EOS and token limit; a refused template stops chat, serve and --chat, not generate  [ok]"
          % ("on the tensor-split shape at tensor width %d, the first reply generate's on the same group and every reply the same at every thread count and ubatch" % common.tensor_width() if grouped else "vs HF"))
    print("chat --chat: generate and logits read one user message as the template renders it  [ok]")
    print("chat progress: completed loading percentages and per-turn phases stay on stderr  [ok]")
    print("chat flags: a device that cannot be made fails before the file is read  [ok]")
    return True
