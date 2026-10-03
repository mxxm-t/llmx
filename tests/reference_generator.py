import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
import tempfile
import types
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch, call


SCRIPT = Path(__file__).resolve().parents[1] / "tools/gen_baseline.py"
SPEC = importlib.util.spec_from_file_location("llmx_gen_baseline", SCRIPT)
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)


def write_routed_layer(path):
    """A qwen35moe GGUF at `path` holding one routed layer, written as the converter writes one: HF's gate_up_proj of [E, 2I, H] as the gate and up stacks apart, and the shared expert's gate of [1, H] as a vector.
    Returns the HF parameters it holds, {name: (shape, values)}."""
    import spec_decode
    experts, rows, width, shared = 3, 2, 4, 5
    written, state = [], {}

    def add(gguf, hf, shape, values, stored=None):
        written.append((gguf, stored or list(reversed(shape)), spec_decode.F32, struct.pack("<%df" % len(values), *values)))
        state[hf] = (shape, values)

    layer = "model.layers.0.mlp."
    gate = [1000.0 + i for i in range(experts * rows * width)]
    up = [2000.0 + i for i in range(experts * rows * width)]
    size = rows * width
    written += [(name, [width, rows, experts], spec_decode.F32, struct.pack("<%df" % len(half), *half))
                for name, half in (("blk.0.ffn_gate_exps.weight", gate), ("blk.0.ffn_up_exps.weight", up))]
    state[layer + "experts.gate_up_proj"] = ([experts, 2 * rows, width], [v for e in range(experts) for half in (gate, up) for v in half[e * size:(e + 1) * size]])
    add("blk.0.ffn_down_exps.weight", layer + "experts.down_proj", [experts, width, rows], [3000.0 + i for i in range(experts * width * rows)])
    add("blk.0.ffn_gate_inp.weight", layer + "gate.weight", [experts, width], [4000.0 + i for i in range(experts * width)])
    add("blk.0.ffn_gate_shexp.weight", layer + "shared_expert.gate_proj.weight", [shared, width], [5000.0 + i for i in range(shared * width)])
    add("blk.0.ffn_up_shexp.weight", layer + "shared_expert.up_proj.weight", [shared, width], [6000.0 + i for i in range(shared * width)])
    add("blk.0.ffn_down_shexp.weight", layer + "shared_expert.down_proj.weight", [width, shared], [7000.0 + i for i in range(shared * width)])
    add("blk.0.ffn_gate_inp_shexp.weight", layer + "shared_expert_gate.weight", [1, width], [8000.0 + i for i in range(width)], [width])
    metadata = {"general.architecture": (8, "qwen35moe"), "qwen35moe.block_count": (4, 1), "qwen35moe.ssm.group_count": (4, 2),
                "qwen35moe.ssm.time_step_rank": (4, 2), "qwen35moe.ssm.state_size": (4, 2), "qwen35moe.ssm.inner_size": (4, 4)}
    spec_decode.write_gguf(path, metadata, written)
    return state


class ReferenceGenerator(unittest.TestCase):
    def test_defaults_and_invalid_selection(self):
        default = generator.parse_args([])
        self.assertEqual(default.repo, "Qwen/Qwen3-0.6B")
        self.assertEqual(default.revision, "c1899de289a04d12100db370d81485cdf75e47ca")
        self.assertEqual(default.threads, 6)
        self.assertEqual(generator.parse_args(["tokenizer-qwen35"]).output_dir, generator.OUT_DIR)
        self.assertEqual(generator.parse_args(["qwen35-tiny"]).output_dir, generator.OUT_DIR)
        alternate = ["logits", "--repo", "Qwen/Qwen3-8B", "--revision", "a" * 40]
        invalid = [
            ["typo"], ["logits", "--revision", "main"],
            ["logits", "--repo", "Qwen/Qwen3-8B"], alternate,
            alternate + ["--output-dir", generator.OUT_DIR],
            ["logits", "--gguf-repo", "Qwen/Qwen3-8B-GGUF"],
            ["logits", "--threads", "0"], ["tokenizer", "--threads", "2"],
            ["f32", "--revision", "b" * 40], ["f32", "--threads", "2"],
            ["logits", "--repo", str(SCRIPT.parent)],
            ["tokenizer-qwen35", "--revision", "c" * 40], ["tokenizer-qwen35", "--threads", "2"],
            ["tokenizer-qwen35", "--repo", "Qwen/Qwen3.5-9B", "--revision", "c" * 40],
            ["tokenizer-qwen35", "--gguf-repo", "a/b", "--gguf-file", "c.gguf"],
            ["qwen35-tiny", "--threads", "2"], ["qwen35-tiny", "--revision", "c" * 40],
            ["file-exact"],
        ]
        with contextlib.redirect_stderr(io.StringIO()):
            for argv in invalid:
                with self.subTest(argv=argv), self.assertRaises(SystemExit) as error:
                    generator.parse_args(argv)
                self.assertEqual(error.exception.code, 2)
        with tempfile.TemporaryDirectory(prefix="llmx_reference_args_") as directory:
            selected = generator.parse_args(alternate + ["--output-dir", directory,
                "--gguf-repo", "Qwen/Qwen3-8B-GGUF", "--gguf-file", "Qwen3-8B-Q8_0.gguf"])
            self.assertEqual(selected.repo, "Qwen/Qwen3-8B")
            self.assertTrue(Path(selected.output_dir).is_absolute())
            self.assertTrue(Path(selected.output_dir).samefile(directory))
            self.assertEqual(selected.gguf_file, "Qwen3-8B-Q8_0.gguf")

    def test_file_exact_selection(self):
        with tempfile.TemporaryDirectory(prefix="llmx_reference_file_exact_") as directory:
            pinned, local = (os.path.join(directory, name) for name in ("Qwen3-0.6B-IQ4_XS.gguf", "local.gguf"))
            for path in (pinned, local):
                Path(path).write_bytes(b"GGUF")
            out = os.path.join(directory, "goldens")
            invalid = [["file-exact", "--output-dir", out], ["logits", "--weights-gguf", pinned, "--output-dir", out],
                       ["file-exact", "--weights-gguf", pinned],
                       ["file-exact", "--weights-gguf", pinned, "--output-dir", generator.OUT_DIR],
                       ["file-exact", "--weights-gguf", pinned, "--output-dir", out, "--gguf-repo", "a/b", "--gguf-file", "b.gguf"],
                       ["file-exact", "--weights-gguf", pinned + ".missing", "--output-dir", out]]
            with contextlib.redirect_stderr(io.StringIO()):
                for argv in invalid:
                    with self.subTest(argv=argv), self.assertRaises(SystemExit) as error:
                        generator.parse_args(argv)
                    self.assertEqual(error.exception.code, 2)
            # A pinned file labels the goldens with its repository, and any other file with its name alone.
            args = generator.parse_args(["file-exact", "--weights-gguf", pinned, "--output-dir", out, "--threads", "2"])
            self.assertEqual((args.gguf_repo, args.gguf_file, args.threads), ("unsloth/Qwen3-0.6B-GGUF", "Qwen3-0.6B-IQ4_XS.gguf", 2))
            self.assertTrue(Path(args.weights_gguf).samefile(pinned))
            args = generator.parse_args(["file-exact", "--weights-gguf", local, "--output-dir", out])
            self.assertEqual((args.gguf_repo, args.gguf_file), (None, "local.gguf"))

    def test_file_exact_run_fails_when_the_device_checks_nothing(self):
        import baseline
        with tempfile.TemporaryDirectory(prefix="llmx_file_exact_run_") as directory:
            model = os.path.join(directory, "model.gguf")
            Path(model).write_bytes(b"GGUF")
            weights = {"file": "model.gguf", "sha256": hashlib.sha256(b"GGUF").hexdigest()}
            for name in ("baseline_logits.json", "baseline_perplexity.json"):
                Path(directory, name).write_text(json.dumps({"weights": weights}), encoding="utf-8")
            # A check returns None when the device has no kernel for the model, which fails a file-exact run.
            cases = (((None, True), False), ((True, None), False), ((False, True), False), ((True, True), True))
            with patch.dict(os.environ, {"LLMX_DEVICE": "vulkan:0"}), patch.object(baseline, "ppl_excerpt", return_value="excerpt.txt"):
                for (logits, ppl), passed in cases:
                    with self.subTest(logits=logits, ppl=ppl), contextlib.redirect_stdout(io.StringIO()):
                        with patch.object(baseline, "check_model_logits", return_value=logits), patch.object(baseline, "check_model_ppl", return_value=ppl):
                            self.assertIs(baseline.run_file_exact(directory, model), passed)

    def test_gguf_tensors_take_their_hf_parameters(self):
        import f32
        import spec_decode
        # One map names the tiny models' parameters and file-exact's; here it is spelled out once more for two block tensors, the embedding and the head.
        self.assertEqual([f32.hf_name(name) for name in ("blk.12.attn_q.weight", "blk.0.ffn_down.weight", "token_embd.weight", "output.weight")],
                         ["model.layers.12.self_attn.q_proj.weight", "model.layers.0.mlp.down_proj.weight", "model.embed_tokens.weight", "lm_head.weight"])
        for name in ("blk.0.ffn_gate_exps.weight", "rope_freqs.weight", "blk.x.attn_q.weight"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                f32.hf_name(name)
        weights = f32.tensors(False)
        # The tiny F32 model as a GGUF: every tensor reaches its HF parameter with its dimensions reversed and its values unchanged.
        with tempfile.TemporaryDirectory(prefix="llmx_reference_state_") as directory:
            path = os.path.join(directory, "tiny.gguf")
            spec_decode.write_gguf(path, {"general.architecture": (8, "qwen3")},
                                   [(name, shape, spec_decode.F32, struct.pack("<%df" % len(values), *values))
                                    for name, _, shape, values in weights])
            state, types = generator.gguf_state(path, numpy=False)
            self.assertEqual(types, {"F32": len(weights)})
            self.assertEqual(set(state), {hf for _, hf, _, _ in weights})
            for _, hf, shape, values in weights:
                self.assertEqual(state[hf], (list(reversed(shape)), values))
            spec_decode.write_gguf(path, {"general.architecture": (8, "llama")}, [])
            with self.assertRaises(SystemExit):
                generator.gguf_state(path, numpy=False)

    def test_qwen3moe_mixed_stacks_are_split_in_expert_order(self):
        import spec_decode
        experts, rows, width = 3, 2, 32
        # Each slot/row has different values; the F32 router is not an expert.
        gate = [float(i - 71) for i in range(experts * rows * width)]
        up = [float((i % 31) - 15) / 8 for i in range(len(gate))]
        down = [float((i % 255) - 127) / 16 for i in range(len(gate))]
        router = [float(i) / 8 for i in range(experts * width)]
        written = [
            ("blk.0.ffn_gate_exps.weight", [width, rows, experts], spec_decode.F32, struct.pack("<%df" % len(gate), *gate)),
            ("blk.0.ffn_up_exps.weight", [width, rows, experts], spec_decode.BF16,
             b"".join(struct.pack("<I", struct.unpack("<I", struct.pack("<f", v))[0])[2:] for v in up)),
            ("blk.0.ffn_down_exps.weight", [width, rows, experts], spec_decode.Q8_0,
             b"".join(struct.pack("<e32b", 1 / 16, *[int(v * 16) for v in down[i:i + 32]]) for i in range(0, len(down), 32))),
            ("blk.0.ffn_gate_inp.weight", [width, experts], spec_decode.F32, struct.pack("<%df" % len(router), *router)),
        ]
        with tempfile.TemporaryDirectory(prefix="llmx_qwen3moe_mapping_") as directory:
            path = os.path.join(directory, "mixed.gguf")
            spec_decode.write_gguf(path, {"general.architecture": (8, "qwen3moe")}, written)
            state, types = generator.gguf_state(path, numpy=False)
            self.assertEqual(types, {"F32": 4, "BF16": 3, "Q8_0": 3})
            self.assertEqual(len(state), 10)
            for projection, values in (("gate", gate), ("up", up), ("down", down)):
                for expert in range(experts):
                    key = "model.layers.0.mlp.experts.%d.%s_proj.weight" % (expert, projection)
                    self.assertEqual(state[key], ([rows, width], values[expert * rows * width:(expert + 1) * rows * width]))
            self.assertEqual(state["model.layers.0.mlp.gate.weight"], ([experts, width], router))
            if importlib.util.find_spec("numpy"):
                arrays, types_np = generator.gguf_state(path)
                self.assertEqual(types_np, types)
                self.assertEqual({k: (shape, list(values)) for k, (shape, values) in arrays.items()}, state)
            spec_decode.write_gguf(path, {"general.architecture": (8, "qwen3moe")},
                                   [(written[0][0], [width, rows * experts], written[0][2], written[0][3])])
            with self.assertRaisesRegex(SystemExit, "three dimensions"):
                generator.gguf_state(path, numpy=False)

    def test_fixtures_are_pinned_once(self):
        import baseline
        import baseline_qwen35
        pinned = baseline.PINNED
        self.assertEqual(len({spec["file"] for spec in pinned}), len(pinned))
        for spec in pinned:
            with self.subTest(file=spec["file"]):
                self.assertEqual(set(spec), {"family", "repo", "file", "revision", "sha256", "size", "gate", "hosted"})
                self.assertIn(spec["family"], ("qwen3", "qwen35"))
                self.assertRegex(spec["revision"], r"^[0-9a-f]{40}$")
                self.assertRegex(spec["sha256"], r"^[0-9a-f]{64}$")
                self.assertTrue(type(spec["size"]) is int and spec["size"] > 0)
                self.assertTrue(type(spec["gate"]) is bool and type(spec["hosted"]) is bool)
                self.assertEqual(baseline.pinned_fixture(spec["file"]), spec)
        # The gate is the models with bounds, in the file's order, each family's by its own: Qwen3's in tests/baseline.py, the qwen35 files' hosted ones in tests/baseline_qwen35.py.
        self.assertEqual([spec["file"] for spec in baseline.BASELINE_MODELS], [spec["file"] for spec in pinned if spec["gate"] and spec["family"] == "qwen3"])
        self.assertEqual([spec["file"] for spec in pinned if spec["gate"] and spec["family"] == "qwen35"],
                         [spec["file"] for spec in pinned if spec["family"] == "qwen35" and spec["hosted"] and baseline_qwen35.bounds_for(spec)])
        # UD-Q8_K_XL has joined the gate; IQ4_XS and Q2_K remain its planned hosted additions. The other three 0.6B files are checked by hand, and 8B BF16 is a manual fixture.
        mixed = baseline.pinned_fixture("Qwen3-0.6B-UD-Q8_K_XL.gguf")
        self.assertTrue(mixed["gate"] and mixed["hosted"])
        self.assertEqual(baseline.BOUNDS[mixed["file"]], baseline.BOUNDS["Qwen3-0.6B-Q8_0.gguf"])
        later = [spec for spec in pinned if not spec["gate"] and spec["family"] == "qwen3"]
        self.assertEqual(sorted(spec["file"] for spec in later if spec["hosted"]),
                         ["Qwen3-0.6B-IQ4_XS.gguf", "Qwen3-0.6B-Q2_K.gguf"])
        self.assertEqual(sorted(spec["file"] for spec in later if not spec["hosted"]),
                         ["Qwen3-0.6B-BF16.gguf", "Qwen3-0.6B-IQ4_NL.gguf", "Qwen3-0.6B-Q3_K_S.gguf", "Qwen3-8B-BF16.gguf"])
        # The qwen35 files: the two 0.8B files hosted and in the gate, the Q4_K_M with its committed file-exact goldens and its own quality bounds, the 4B by hand, and each a file of a checkpoint QWEN35_MODELS pins.
        self.assertEqual(sorted(spec["file"] for spec in pinned if spec["family"] == "qwen35" and spec["gate"]), ["Qwen3.5-0.8B-Q4_K_M.gguf", "Qwen3.5-0.8B-Q8_0.gguf"])
        qwen35 = [spec for spec in pinned if spec["family"] == "qwen35"]
        self.assertEqual(sorted(spec["file"] for spec in qwen35 if spec["hosted"]), ["Qwen3.5-0.8B-Q4_K_M.gguf", "Qwen3.5-0.8B-Q8_0.gguf"])
        self.assertEqual(sorted(spec["file"] for spec in qwen35 if not spec["hosted"]), ["Qwen3.5-4B-Q4_K_M.gguf"])
        self.assertEqual(sorted(spec["file"] for spec in qwen35),
                         sorted(file for model in generator.QWEN35_MODELS.values() for file in model["gguf_files"]))

    def test_qwen35_selection(self):
        args = generator.parse_args(["qwen35", "--model", "Qwen3.5-4B", "--threads", "3"])
        self.assertEqual((args.family, args.qwen35_model, args.repo, args.revision, args.threads),
                         ("qwen35", "Qwen3.5-4B", "Qwen/Qwen3.5-4B", generator.QWEN35_MODELS["Qwen3.5-4B"]["revision"], 3))
        self.assertEqual(Path(args.output_dir), Path(generator.OUT_DIR) / "qwen35-4b")
        self.assertEqual((args.gguf_repo, args.gguf_file), ("lmstudio-community/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf"))
        with tempfile.TemporaryDirectory(prefix="llmx_reference_qwen35_out_") as directory:
            self.assertTrue(Path(generator.parse_args(["qwen35", "--model", "Qwen3.5-0.8B", "--output-dir", directory]).output_dir).samefile(directory))
        invalid = [["qwen35"], ["logits", "--model", "Qwen3.5-0.8B"], ["qwen35", "--model", "Qwen3.5-9B"],
                   ["qwen35", "--model", "Qwen3.5-0.8B", "--repo", "Qwen/Qwen3.5-4B", "--revision", "a" * 40],
                   ["qwen35", "--model", "Qwen3.5-0.8B", "--gguf-repo", "a/b", "--gguf-file", "c.gguf"],
                   ["qwen35", "--model", "Qwen3.5-0.8B", "--weights-gguf", str(SCRIPT)],
                   ["qwen35", "--model", "Qwen3.5-0.8B", "--threads", "0"],
                   # The Qwen3 goldens and another model's goldens are not this model's to write.
                   ["qwen35", "--model", "Qwen3.5-0.8B", "--output-dir", generator.OUT_DIR],
                   ["qwen35", "--model", "Qwen3.5-0.8B", "--output-dir", os.path.join(generator.OUT_DIR, "qwen35-4b")]]
        with contextlib.redirect_stderr(io.StringIO()):
            for argv in invalid:
                with self.subTest(argv=argv), self.assertRaises(SystemExit) as error:
                    generator.parse_args(argv)
                self.assertEqual(error.exception.code, 2)
        # A pinned qwen35 file's file-exact reference is the checkpoint whose goldens are for it, which --repo and --revision cannot replace, and its goldens never go over committed ones.
        with tempfile.TemporaryDirectory(prefix="llmx_reference_qwen35_file_exact_") as directory:
            path = os.path.join(directory, "Qwen3.5-0.8B-Q4_K_M.gguf")
            Path(path).write_bytes(b"GGUF")
            out = os.path.join(directory, "goldens")
            args = generator.parse_args(["file-exact", "--weights-gguf", path, "--output-dir", out])
            self.assertEqual((args.family, args.qwen35_model, args.repo, args.gguf_repo, args.gguf_file),
                             ("qwen35", "Qwen3.5-0.8B", "Qwen/Qwen3.5-0.8B", "unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"))
            for extra in (["--repo", "Qwen/Qwen3.5-4B", "--revision", "a" * 40], ["--revision", "a" * 40]):
                with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    generator.parse_args(["file-exact", "--weights-gguf", path, "--output-dir", out] + extra)
                self.assertEqual(error.exception.code, 2)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                generator.parse_args(["file-exact", "--weights-gguf", path, "--output-dir", os.path.join(generator.OUT_DIR, "qwen35-0.8b")])
            self.assertEqual(error.exception.code, 2)

    def test_qwen35_tensors_undo_the_converter(self):
        import spec_decode
        # A qwen35 file written as the converter writes one, from HF-order tensors of a model with 3 V heads to each of 2 K heads, a linear layer, an attention layer and an MTP block.
        width, vocab, hk, hv, dk, dv = 4, 5, 2, 6, 2, 3
        conv = 2 * hk * dk + hv * dv
        # The converter stores HF's V head h * 3 + s as V head s * 2 + h.
        held = [(j % hk) * (hv // hk) + j // hk for j in range(hv)]

        def tiled(values, rows, block, cols=1, offset=0, columns=False):
            if columns:
                return [values[r * rows + offset + held[j // block] * block + j % block] for r in range(len(values) // rows) for j in range(rows)]
            head_rows = [offset + held[j // block] * block + j % block for j in range(rows - offset)]
            return values[:offset * cols] + [values[r * cols + c] for r in head_rows for c in range(cols)]

        state, written = {}, []

        def add(gguf, hf, shape, stored=None):
            count = 1
            for n in shape:
                count *= n
            values = [0.25 * (len(state) * 64 + i) for i in range(count)]
            state[hf] = (shape, values)
            data = stored(values) if stored else values
            written.append((gguf, list(reversed(shape)) if len(shape) < 3 else [shape[2], shape[0]], spec_decode.F32, struct.pack("<%df" % count, *data)))

        norm = lambda values: [v + 1 for v in values]
        add("token_embd.weight", "model.embed_tokens.weight", [vocab, width])
        add("output_norm.weight", "model.norm.weight", [width], norm)
        add("output.weight", "lm_head.weight", [vocab, width])
        linear = "model.layers.0.linear_attn."
        add("blk.0.attn_norm.weight", "model.layers.0.input_layernorm.weight", [width], norm)
        add("blk.0.post_attention_norm.weight", "model.layers.0.post_attention_layernorm.weight", [width], norm)
        add("blk.0.attn_qkv.weight", linear + "in_proj_qkv.weight", [conv, width], lambda v: tiled(v, conv, dv, width, 2 * hk * dk))
        add("blk.0.attn_gate.weight", linear + "in_proj_z.weight", [hv * dv, width], lambda v: tiled(v, hv * dv, dv, width))
        add("blk.0.ssm_alpha.weight", linear + "in_proj_a.weight", [hv, width], lambda v: tiled(v, hv, 1, width))
        add("blk.0.ssm_beta.weight", linear + "in_proj_b.weight", [hv, width], lambda v: tiled(v, hv, 1, width))
        add("blk.0.ssm_conv1d.weight", linear + "conv1d.weight", [conv, 1, 4], lambda v: tiled(v, conv, dv, 4, 2 * hk * dk))
        add("blk.0.ssm_a", linear + "A_log", [hv], lambda v: tiled([-math.exp(x / 64) for x in v], hv, 1))
        add("blk.0.ssm_dt.bias", linear + "dt_bias", [hv], lambda v: tiled(v, hv, 1))
        add("blk.0.ssm_norm.weight", linear + "norm.weight", [dv])
        add("blk.0.ssm_out.weight", linear + "out_proj.weight", [width, hv * dv], lambda v: tiled(v, hv * dv, dv, columns=True))
        attention = "model.layers.1.self_attn."
        for name, hf, shape in (("attn_q", "q_proj", [8, width]), ("attn_k", "k_proj", [2, width]), ("attn_v", "v_proj", [2, width]),
                                ("attn_output", "o_proj", [width, 4])):
            add("blk.1.%s.weight" % name, attention + hf + ".weight", shape)
        for name, hf in (("attn_q_norm", "q_norm"), ("attn_k_norm", "k_norm")):
            add("blk.1.%s.weight" % name, attention + hf + ".weight", [2], norm)
        add("blk.1.ffn_down.weight", "model.layers.1.mlp.down_proj.weight", [width, 3])
        add("blk.2.nextn.eh_proj.weight", "mtp.fc.weight", [width, 2 * width])
        del state["mtp.fc.weight"]
        # The A_log a file holds is -exp(A_log) rounded to f32, so its log comes back within that rounding.
        state[linear + "A_log"] = ([hv], [x / 64 for x in state[linear + "A_log"][1]])
        metadata = {"general.architecture": (8, "qwen35"), "qwen35.block_count": (4, 3), "qwen35.nextn_predict_layers": (4, 1),
                    "qwen35.ssm.group_count": (4, hk), "qwen35.ssm.time_step_rank": (4, hv), "qwen35.ssm.state_size": (4, dk),
                    "qwen35.ssm.inner_size": (4, hv * dv)}
        with tempfile.TemporaryDirectory(prefix="llmx_reference_qwen35_state_") as directory:
            path = os.path.join(directory, "tiny.gguf")
            spec_decode.write_gguf(path, metadata, written)
            got, types = generator.gguf_state(path, numpy=False)
            self.assertEqual(types, {"F32": len(state)})
            self.assertEqual(set(got), set(state))
            for key, (shape, values) in state.items():
                with self.subTest(key=key):
                    self.assertEqual(got[key][0], shape)
                    if key.endswith("A_log"):
                        self.assertTrue(all(abs(a - b) < 1e-6 for a, b in zip(got[key][1], values)))
                    else:
                        self.assertEqual(got[key][1], values)
            spec_decode.write_gguf(path, metadata, written + [("blk.0.ssm_extra", [hv], spec_decode.F32, struct.pack("<%df" % hv, *[0.0] * hv))])
            with self.assertRaisesRegex(SystemExit, "blk.0.ssm_extra"):
                generator.gguf_state(path, numpy=False)

    def test_qwen35moe_tensors_join_the_expert_halves(self):
        import spec_decode
        with tempfile.TemporaryDirectory(prefix="llmx_reference_qwen35moe_state_") as directory:
            path = os.path.join(directory, "tiny.gguf")
            state = write_routed_layer(path)
            layer = "model.layers.0.mlp."
            (experts, rows, width), joined = state[layer + "experts.gate_up_proj"]
            rows //= 2
            got, types = generator.gguf_state(path, numpy=False)
            self.assertEqual((got, types), (state, {"F32": len(state)}))
            model = spec_decode.GGUF(path)
            # The layered reference decodes one layer's tensors at a time; the up stack alone gives nothing, and the gate stack gives the whole gate_up_proj.
            by_name = {t.name: t for t in model.tensors}
            self.assertEqual(list(generator.qwen35_tensors(model, False, [by_name["blk.0.ffn_up_exps.weight"]])), [])
            self.assertEqual([(name, shape, values) for name, shape, values, _ in generator.qwen35_tensors(model, False, [by_name["blk.0.ffn_gate_exps.weight"]])],
                             [(layer + "experts.gate_up_proj", [experts, 2 * rows, width], joined)])
            if importlib.util.find_spec("numpy"):
                self.assertEqual({key: (shape, list(values)) for key, (shape, values) in generator.gguf_state(path)[0].items()}, state)

    def test_qwen35_reference_loads_as_pinned(self):
        # load_qwen35 against doubles: the versions, float32 and eager attention, transformers' torch forms, and the keys a load may leave out.
        dtype = object()

        def load(torch_version="2.5.1+cpu", transformers_version="5.17.0", tokenizers_version="0.23.2", installed=(), missing=(), mismatched=(), unexpected=()):
            model = MagicMock()
            model.cpu.return_value = model
            info = {"missing_keys": list(missing), "mismatched_keys": list(mismatched),
                    "unexpected_keys": ["mtp.fc.weight", "model.visual.blocks.0.attn.qkv.weight"] + list(unexpected)}
            causal = SimpleNamespace(from_pretrained=MagicMock(return_value=(model, info)))
            modeling = SimpleNamespace(torch_chunk_gated_delta_rule=MagicMock(return_value="chunked"), torch_recurrent_gated_delta_rule=MagicMock())
            torch = SimpleNamespace(__version__=torch_version, float32=dtype, set_num_threads=MagicMock())
            modules = {"torch": torch, "tokenizers": SimpleNamespace(__version__=tokenizers_version),
                       "transformers": SimpleNamespace(__version__=transformers_version, Qwen3_5ForCausalLM=causal),
                       "transformers.models": SimpleNamespace(), "transformers.models.qwen3_5": SimpleNamespace(modeling_qwen3_5=modeling),
                       "transformers.models.qwen3_5.modeling_qwen3_5": modeling}
            args = generator.parse_args(["qwen35", "--model", "Qwen3.5-0.8B", "--threads", "2"])
            with patch.dict(sys.modules, modules), patch.object(generator, "qwen35_checkpoint", return_value="checkpoint"), \
                 patch.object(generator, "Qwen35Tokenizer"), patch.object(importlib.util, "find_spec", side_effect=lambda name: name in installed or None):
                result = generator.load_qwen35(args)
            return args, torch, causal, modeling, model, result

        args, torch, causal, modeling, model, result = load()
        causal.from_pretrained.assert_called_once_with("checkpoint", dtype=dtype, attn_implementation="eager", output_loading_info=True)
        torch.set_num_threads.assert_called_once_with(2)
        model.eval.assert_called_once_with()
        self.assertIs(result[3], model)
        self.assertEqual(args.qwen35["gguf_files"], generator.QWEN35_MODELS["Qwen3.5-0.8B"]["gguf_files"])
        # The goldens are HF's full forward: its chunked form is counted and its token-by-token recurrence refused.
        self.assertEqual(modeling.torch_chunk_gated_delta_rule(1), "chunked")
        self.assertEqual(args.chunked_calls, 1)
        with self.assertRaisesRegex(SystemExit, "recurrence"):
            modeling.torch_recurrent_gated_delta_rule(1)
        refusals = [({"torch_version": "2.6.0+cpu"}, "2.5.1"), ({"transformers_version": "5.16.0"}, "5.17.0"),
                    ({"tokenizers_version": "0.22.1"}, "0.23.2"), ({"installed": ("kernels",)}, "kernels"),
                    ({"installed": ("fla",)}, "fla"), ({"installed": ("causal_conv1d",)}, "causal_conv1d"),
                    ({"missing": ("model.norm.weight",)}, "model.norm.weight"),
                    ({"mismatched": ("model.layers.0.linear_attn.A_log",)}, "model.layers.0.linear_attn.A_log"),
                    ({"unexpected": ("model.language_model.extra",)}, "model.language_model.extra")]
        for change, message in refusals:
            with self.subTest(change=change), self.assertRaisesRegex(SystemExit, message):
                load(**change)

    def test_qwen35_checkpoint_files_are_held_to_their_digests(self):
        with tempfile.TemporaryDirectory(prefix="llmx_reference_qwen35_checkpoint_") as directory:
            files = {"config.json": b"{}", "model.safetensors": b"weights"}
            for name, raw in files.items():
                (Path(directory) / name).write_bytes(raw)
            spec = {"repo": "a/b", "revision": "c" * 40, "sha256": {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()}}
            hub = SimpleNamespace(hf_hub_download=MagicMock(side_effect=lambda repo, name, revision: str(Path(directory) / name)))
            with patch.dict(sys.modules, {"huggingface_hub": hub}):
                self.assertTrue(Path(generator.qwen35_checkpoint(spec)).samefile(directory))
                self.assertEqual(hub.hf_hub_download.call_args_list, [call("a/b", name, revision="c" * 40) for name in files])
                with self.assertRaisesRegex(SystemExit, "model.safetensors"):
                    generator.qwen35_checkpoint(dict(spec, sha256=dict(spec["sha256"], **{"model.safetensors": "0" * 64})))

    def test_committed_qwen35_goldens_are_the_generators(self):
        # Each model's goldens record the generator's checkpoint, files, prompts, conversations, template and excerpts, and the consumer pins every one of them.
        import baseline_qwen35
        pinned = set()
        for spec in generator.QWEN35_MODELS.values():
            directory = Path(generator.OUT_DIR) / spec["directory"]
            docs = {path.name: json.loads(path.read_text(encoding="utf-8")) for path in directory.glob("*.json")}
            self.assertEqual(sorted(docs), sorted(["baseline_logits.json", "baseline_chat.json"] + list(baseline_qwen35.PPL_GOLDENS.values())))
            pinned.update(spec["directory"] + "/" + name for name in docs)
            for name, doc in docs.items():
                with self.subTest(golden=spec["directory"] + "/" + name):
                    self.assertEqual((doc["reference_repo"], doc["reference_revision"], doc["gguf_files"]), (spec["repo"], spec["revision"], spec["gguf_files"]))
                    self.assertEqual((doc["reference_dtype"], doc["attention"], doc["transformers_version"]), ("float32", "eager", generator.QWEN35_ENV["transformers"]))
            self.assertEqual([case["text"] for case in docs["baseline_logits.json"]["cases"]], generator.LOGIT_PROMPTS)
            chat = docs["baseline_chat.json"]
            self.assertEqual(chat["template_sha256"], spec["template_sha256"])
            self.assertEqual([(case["name"], case["messages"]) for case in chat["cases"]], [(name, messages) for name, messages in generator.QWEN35_CHATS])
            for context, name in baseline_qwen35.PPL_GOLDENS.items():
                self.assertEqual((docs[name]["chars"], docs[name]["chunk_cases"][0]["context_size"]), (generator.QWEN35_PPL[context], context))
        # A file's committed file-exact goldens hold the generator's prompts and 512-token excerpt, run on the checkpoint its entry names with that very file's weights.
        import baseline
        for sha256, directory in baseline_qwen35.FILE_EXACT.items():
            spec = next(spec for spec in baseline.PINNED if spec["sha256"] == sha256)
            model = next(model for model in generator.QWEN35_MODELS.values() if spec["file"] in model["gguf_files"])
            docs = {path.name: json.loads(path.read_text(encoding="utf-8")) for path in (Path(generator.OUT_DIR) / directory).glob("*.json")}
            self.assertEqual(sorted(docs), ["baseline_logits.json", baseline_qwen35.PPL_GOLDENS[512]])
            pinned.update(directory + "/" + name for name in docs)
            for name, doc in docs.items():
                with self.subTest(golden=directory + "/" + name):
                    self.assertEqual((doc["reference_repo"], doc["reference_revision"]), (model["repo"], model["revision"]))
                    self.assertEqual((doc["weights"]["file"], doc["weights"]["sha256"]), (spec["file"], sha256))
                    self.assertEqual((doc["reference_dtype"], doc["attention"], doc["transformers_version"]), ("float32", "eager", generator.QWEN35_ENV["transformers"]))
            self.assertEqual([case["text"] for case in docs["baseline_logits.json"]["cases"]], generator.LOGIT_PROMPTS)
            self.assertEqual(docs[baseline_qwen35.PPL_GOLDENS[512]]["chars"], generator.QWEN35_PPL[512])
        self.assertEqual(pinned | {baseline_qwen35.TOKENIZER_GOLDEN}, set(baseline_qwen35.GOLDEN_SHA256))

    @unittest.skipUnless(sys.platform == "win32", "Windows short-path aliases")
    def test_short_windows_output_directory(self):
        import ctypes
        short_path = ctypes.windll.kernel32.GetShortPathNameW
        short_path.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
        short_path.restype = ctypes.c_uint
        with tempfile.TemporaryDirectory(prefix="llmx_reference_short_path_") as directory:
            resolved = str(Path(directory).resolve())
            buffer = ctypes.create_unicode_buffer(32768)
            length = short_path(resolved, buffer, len(buffer))
            self.assertTrue(0 < length < len(buffer))
            if buffer.value == resolved:
                self.skipTest("Filesystem does not provide a short-path alias")
            with patch.object(tempfile, "tempdir", buffer.value):
                self.test_defaults_and_invalid_selection()

    def test_logits_passes_reference_identity_to_loaders(self):
        with tempfile.TemporaryDirectory(prefix="llmx_reference_logits_") as directory:
            args = generator.parse_args(["logits", "--output-dir", directory,
                                         "--revision", "b" * 40, "--threads", "2"])
            ids = MagicMock()
            ids.shape = (1, 2)
            ids.__getitem__.return_value.tolist.return_value = [11, 12]
            tokenizer = MagicMock(return_value=SimpleNamespace(input_ids=ids))
            model = MagicMock()
            model.cpu.return_value = model
            auto_tokenizer = SimpleNamespace(from_pretrained=MagicMock(return_value=tokenizer))
            auto_model = SimpleNamespace(from_pretrained=MagicMock(return_value=model))
            dtype = object()
            torch = SimpleNamespace(__version__="test-torch", float32=dtype,
                set_num_threads=MagicMock(), inference_mode=contextlib.nullcontext,
                topk=lambda logits, count: SimpleNamespace(
                    indices=SimpleNamespace(tolist=lambda: list(range(count))),
                    values=[float(i) for i in range(count)]))
            transformers = SimpleNamespace(__version__="test-transformers",
                AutoTokenizer=auto_tokenizer, AutoModelForCausalLM=auto_model)
            with patch.dict(sys.modules, torch=torch, transformers=transformers), contextlib.redirect_stdout(io.StringIO()):
                generator.gen_logits(args)
            auto_tokenizer.from_pretrained.assert_called_once_with(args.repo, revision="b" * 40)
            auto_model.from_pretrained.assert_called_once_with(args.repo, revision="b" * 40,
                torch_dtype=dtype, attn_implementation="eager")
            torch.set_num_threads.assert_called_once_with(2)
            model.cpu.assert_called_once_with()
            model.eval.assert_called_once_with()
            self.assertEqual(tokenizer.call_args_list, [call(text, add_special_tokens=False, return_tensors="pt")
                                                       for text in generator.LOGIT_PROMPTS])
            self.assertEqual(model.call_args_list, [call(ids, use_cache=False)] * len(generator.LOGIT_PROMPTS))
            doc = json.loads((Path(directory) / "baseline_logits.json").read_text())
            self.assertEqual(doc["reference_revision"], "b" * 40)
            self.assertEqual(doc["reference_dtype"], "float32")
            self.assertEqual(doc["attention"], "eager")
            self.assertEqual(doc["threads"], 2)
            self.assertEqual(doc["torch_version"], "test-torch")
            self.assertTrue(all(case["token_ids"] == [11, 12] for case in doc["cases"]))

    def test_qwen35_tokenizer_golden_keeps_reached_merges_and_the_files_token_types(self):
        # The byte map writes the space of "ab c" as U+0120; HF cuts the text before it, and the merge of "b" with it joins across that cut, so it must be kept for a pretokenizer that does not cut there.
        # The added tokens take the GGUF files' types, control for a special token or one written <|name|> and user-defined for the rest, and a token only the config adds is kept apart.
        space = "\u0120"
        alphabet = ["a", "b", "c", "x", "y", space]
        merges = ["a b", space + " c", "x y", "b " + space, "ab " + space + "c"]
        vocab = {t: i for i, t in enumerate(alphabet + ["ab", space + "c", "xy", "b" + space, "ab" + space + "c"])}
        added = [{"id": 20, "content": "<s>", "special": True}, {"id": 21, "content": "<t>", "special": False},
                 {"id": 22, "content": "<|x|>", "special": False}]
        spec = {"model": {"vocab": vocab, "merges": merges}, "added_tokens": added}
        config = {"added_tokens_decoder": {str(a["id"]): a for a in added + [{"id": 23, "content": "<u>", "special": True}]}}
        byte_level = MagicMock()
        byte_level.return_value.pre_tokenize_str = lambda text: [(text.replace(" ", space), (0, len(text)))]
        byte_level.alphabet.return_value = alphabet
        with tempfile.TemporaryDirectory(prefix="llmx_reference_qwen35_") as directory:
            files = {"tokenizer.json": json.dumps(spec).encode(), "tokenizer_config.json": json.dumps(config).encode()}
            for name, raw in files.items():
                (Path(directory) / name).write_bytes(raw)
            digests = {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()}
            hub = SimpleNamespace(hf_hub_download=MagicMock(side_effect=lambda repo, name, revision: str(Path(directory) / name)))
            encoded = SimpleNamespace(ids=[vocab["ab"], vocab[space + "c"]])
            tokenizers = SimpleNamespace(__version__="test-tokenizers",
                Tokenizer=SimpleNamespace(from_file=MagicMock(return_value=SimpleNamespace(encode=MagicMock(return_value=encoded)))),
                pre_tokenizers=SimpleNamespace(ByteLevel=byte_level))
            with patch.dict(sys.modules, {"tokenizers": tokenizers, "tokenizers.pre_tokenizers": tokenizers.pre_tokenizers,
                                          "huggingface_hub": hub}), \
                 patch.object(generator, "QWEN35_CASES", ["ab c"]), contextlib.redirect_stdout(io.StringIO()):
                # A file whose digest is not the pinned one is refused before anything is written.
                with patch.object(generator, "QWEN35_SHA256", dict(digests, **{"tokenizer_config.json": "0" * 64})), \
                     self.assertRaisesRegex(SystemExit, "tokenizer_config.json"):
                    generator.gen_tokenizer_qwen35(directory)
                self.assertFalse((Path(directory) / "baseline_tokenizer_qwen35.json").exists())
                hub.hf_hub_download.reset_mock()
                with patch.object(generator, "QWEN35_SHA256", digests):
                    generator.gen_tokenizer_qwen35(directory)
            self.assertEqual(hub.hf_hub_download.call_args_list,
                             [call(generator.QWEN35_REPO, name, revision=generator.QWEN35_REVISION) for name in ("tokenizer.json", "tokenizer_config.json")])
            doc = json.loads((Path(directory) / "baseline_tokenizer_qwen35.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["merges"], ["a b", space + " c", "b " + space, "ab " + space + "c"])
        self.assertEqual(doc["tokens"], {str(i): t for t, i in vocab.items() if t != "xy"} | {"20": "<s>", "21": "<t>", "22": "<|x|>"})
        self.assertEqual(doc["config_tokens"], {"23": "<u>"})
        self.assertEqual(doc["token_types"], {"20": 3, "21": 4, "22": 3, "23": 3})
        self.assertEqual(doc["cases"], [{"text": "ab c", "ids": encoded.ids}])
        self.assertEqual(doc["tokenizers_version"], "test-tokenizers")
        self.assertEqual((doc["tokenizer_json_sha256"], doc["tokenizer_config_json_sha256"]), (digests["tokenizer.json"], digests["tokenizer_config.json"]))

    def test_committed_qwen35_tokenizer_golden_is_the_generators(self):
        # Changing the generator's texts, commit or pinned digests without regenerating the golden fails here.
        doc = json.loads((Path(generator.OUT_DIR) / "baseline_tokenizer_qwen35.json").read_text(encoding="utf-8"))
        self.assertEqual([case["text"] for case in doc["cases"]], generator.QWEN35_CASES)
        self.assertEqual((doc["tokenizer_repo"], doc["tokenizer_revision"]), (generator.QWEN35_REPO, generator.QWEN35_REVISION))
        self.assertEqual({"tokenizer.json": doc["tokenizer_json_sha256"], "tokenizer_config.json": doc["tokenizer_config_json_sha256"]},
                         generator.QWEN35_SHA256)

    def test_qwen35_environment_is_offline_at_its_versions_without_replacements(self):
        # Every qwen35 reference comes from HF's own torch functions in one pinned environment, so another torch, transformers or tokenizers version, or a package HF's qwen3_5 code would run in their place, is refused.
        modeling = SimpleNamespace()
        self.assertEqual(generator.QWEN35_ENV["transformers"], generator.CHAT_REFERENCE["transformers"])

        def modules(**versions):
            found = dict(generator.QWEN35_ENV, **versions)
            return {"torch": SimpleNamespace(__version__=found["torch"]), "tokenizers": SimpleNamespace(__version__=found["tokenizers"]),
                    "transformers": SimpleNamespace(__version__=found["transformers"]), "transformers.models.qwen3_5.modeling_qwen3_5": modeling}

        with patch.dict(os.environ), patch.dict(sys.modules, modules()), \
             patch.object(generator.importlib.util, "find_spec", return_value=None) as find:
            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            self.assertIs(generator.qwen35_environment()[2], modeling)
            self.assertEqual((os.environ["HF_HUB_OFFLINE"], os.environ["TRANSFORMERS_OFFLINE"]), ("1", "1"))
            self.assertEqual(sorted(c.args[0] for c in find.call_args_list), ["causal_conv1d", "fla", "kernels"])
        for change in ({"torch": "2.6.0+cpu"}, {"transformers": "5.16.0"}, {"tokenizers": "0.22.1"}):
            with self.subTest(change=change), patch.dict(os.environ), patch.dict(sys.modules, modules(**change)), \
                 patch.object(generator.importlib.util, "find_spec", return_value=None), self.assertRaisesRegex(SystemExit, re.escape(list(change.values())[0])):
                generator.qwen35_environment()
        for package in generator.QWEN35_REPLACEMENTS:
            with self.subTest(package=package), patch.dict(os.environ), patch.dict(sys.modules, modules()), \
                 patch.object(generator.importlib.util, "find_spec", side_effect=lambda name: object() if name == package else None), \
                 self.assertRaisesRegex(SystemExit, "the %s package" % package):
                generator.qwen35_environment()

    def test_qwen35_loading_holds_float32_eager_attention_and_the_keys(self):
        # The model loads from local files in float32 with eager attention, may leave only mtp.* and model.visual.* keys unused, and may miss none.
        dtype = object()
        torch = SimpleNamespace(float32=dtype)
        keys = ["model.norm.weight", "mtp.fc.weight", "model.visual.blocks.0.attn.qkv.weight"]

        def load(keys=keys, attention="eager", param=dtype, **report):
            model = MagicMock()
            model.state_dict.return_value = dict.fromkeys(["model.norm.weight", "lm_head.weight"])
            model.config._attn_implementation = attention
            model.parameters.return_value = [SimpleNamespace(dtype=dtype), SimpleNamespace(dtype=param)]
            info = dict({"missing_keys": set(), "unexpected_keys": set(), "mismatched_keys": set(), "error_msgs": []}, **report)
            loader = MagicMock(return_value=(model, info))
            got, unused = generator.load_qwen35_tiny("checkpoint", keys, torch, SimpleNamespace(Qwen3_5ForCausalLM=SimpleNamespace(from_pretrained=loader)))
            loader.assert_called_once_with("checkpoint", dtype=dtype, attn_implementation="eager", local_files_only=True, output_loading_info=True)
            self.assertIs(got, model)
            model.eval.assert_called_once_with()
            return unused

        self.assertEqual(load(), ["model.visual.blocks.0.attn.qkv.weight", "mtp.fc.weight"])
        refused = [({"keys": keys + ["model.layers.0.mlp.bias"]}, "model.layers.0.mlp.bias"),
                   ({"unexpected_keys": {"model.layers.9.gate"}}, "model.layers.9.gate"),
                   ({"missing_keys": {"lm_head.weight"}}, "lm_head.weight"),
                   ({"mismatched_keys": {("model.norm.weight", (37,), (38,))}}, "model.norm.weight"),
                   ({"error_msgs": ["size mismatch"]}, "size mismatch"),
                   ({"attention": "sdpa"}, "eager attention"), ({"param": object()}, "float32")]
        for change, message in refused:
            with self.subTest(change=change), self.assertRaisesRegex(SystemExit, message):
                load(**change)

    def test_committed_tiny_qwen35_goldens_are_the_generators(self):
        # The goldens come from HF's recurrence at the pinned version in float32 with eager attention, the full forward, which runs the chunked form, lands elsewhere, and only the MTP block's keys go unused.
        import qwen35
        doc = qwen35.golden()
        self.assertEqual((doc["torch_version"], doc["transformers_version"], doc["dtype"], doc["attention"]),
                         (generator.QWEN35_ENV["torch"], generator.QWEN35_ENV["transformers"], "float32", "eager"))
        bases = [fixture["name"] for fixture in doc["fixtures"] if not fixture["mtp"]]
        for fixture in doc["fixtures"]:
            with self.subTest(fixture=fixture["name"]):
                self.assertGreater(fixture["recurrent_steps"], 0)
                if fixture["mtp"]:
                    self.assertTrue(fixture["unused_keys"] and all(key.startswith("mtp.") for key in fixture["unused_keys"]))
                    self.assertIn(fixture["base"], bases)
                else:
                    self.assertEqual(fixture["unused_keys"], [])
                    self.assertGreater(fixture["full_forward_distance"], 0)
                    self.assertGreaterEqual(fixture["greedy"]["min_gap"], generator.QWEN35_GREEDY_GAP)

    def test_tiled_v_head_order(self):
        # tests/qwen35.py owns the order for the tiny writer, the file-exact references and the layered reference's GGUF comparison.
        import qwen35
        self.assertEqual(qwen35.tiled_order(2, 6), [0, 3, 1, 4, 2, 5])
        self.assertEqual(qwen35.tiled_order(16, 16), list(range(16)))
        # GGUF V head j reads K head j mod Hk, and the HF head it holds reads K head floor(i / r), the same one.
        for k_heads, v_heads in ((16, 32), (16, 48)):
            order = qwen35.tiled_order(k_heads, v_heads)
            self.assertEqual(sorted(order), list(range(v_heads)))
            self.assertTrue(all(order[j] // (v_heads // k_heads) == j % k_heads for j in range(v_heads)))
        self.assertEqual(qwen35.tiled(list("abcdef"), 2, 6, 1), list("adbecf"))

    def test_the_writers_tiled_order_maps_hf_dt_bias_onto_a_4b_gguf(self):
        # HF's dt_bias of one Qwen3.5-4B layer, put in the tiny qwen35 writer's tiled order, is that layer's ssm_dt.bias in a 4B GGUF bit for bit, and in HF's own order it is not.
        # So a misreading of the order that the writer and the kernels share fails here.
        import qwen35
        doc = json.loads((Path(generator.OUT_DIR) / "qwen35_4b_dt_bias.json").read_text(encoding="utf-8"))
        hf, gguf = doc["hf"]["values"], doc["gguf"]["values"]
        k_heads, v_heads = doc["gguf"]["ssm.group_count"], doc["gguf"]["ssm.time_step_rank"]
        self.assertEqual((len(hf), len(gguf), v_heads % k_heads), (v_heads, v_heads, 0))
        self.assertGreater(v_heads, k_heads)
        # HF stores bf16, which widens to float32 exactly.
        self.assertTrue(all(struct.unpack("<I", struct.pack("<f", value))[0] & 0xffff == 0 for value in hf))

        def packed(values):
            return struct.pack("<%df" % len(values), *values)

        self.assertEqual(packed(qwen35.tiled(hf, k_heads, v_heads, 1)), packed(gguf))
        self.assertNotEqual(packed(hf), packed(gguf))


LAYERED_SCRIPT = SCRIPT.parent / "gen_layered_reference.py"
LAYERED_SPEC = importlib.util.spec_from_file_location("llmx_gen_layered_reference", LAYERED_SCRIPT)
layered = importlib.util.module_from_spec(LAYERED_SPEC)
LAYERED_SPEC.loader.exec_module(layered)


class LayeredReference(unittest.TestCase):
    """tools/gen_layered_reference.py, which needs torch to run, through what it does without it: its arguments, its environment, its inputs, and the records it committed."""

    def test_peak_memory_on_this_platform(self):
        peak = layered.peak_gib()
        self.assertTrue(math.isfinite(peak))
        self.assertGreater(peak, 0)

    def test_peak_memory_posix_units(self):
        for platform, units in (("linux", 2 ** 20), ("darwin", 2 ** 30)):
            resource = SimpleNamespace(RUSAGE_SELF=17, getrusage=MagicMock(return_value=SimpleNamespace(ru_maxrss=units * 1.25)))
            with self.subTest(platform=platform), patch.object(layered.sys, "platform", platform), \
                 patch.dict(sys.modules, {"resource": resource}):
                self.assertEqual(layered.peak_gib(), 1.25)
                resource.getrusage.assert_called_once_with(resource.RUSAGE_SELF)

    def test_snapshot_sets_offline_before_hub_import(self):
        import builtins
        original = builtins.__import__
        download = MagicMock(return_value=os.path.join("cache", "revision", "config.json"))
        seen = []

        def importing(name, *args, **kwargs):
            if name == "huggingface_hub":
                seen.append((os.environ.get("HF_HUB_OFFLINE"), os.environ.get("TRANSFORMERS_OFFLINE")))
                return SimpleNamespace(hf_hub_download=download)
            return original(name, *args, **kwargs)

        with patch.dict(os.environ), patch.object(builtins, "__import__", importing):
            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            self.assertEqual(layered.snapshot("a/model", "b" * 40), os.path.join("cache", "revision"))
        self.assertEqual(seen, [("1", "1")])
        download.assert_called_once_with("a/model", "config.json", revision="b" * 40, local_files_only=True)

    def test_layered_tokenizer_loads_are_explicitly_offline(self):
        loader = MagicMock(side_effect=RuntimeError("stopped before reference setup"))
        transformers = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=loader))
        with tempfile.TemporaryDirectory(prefix="llmx_layered_offline_") as directory:
            Path(directory, "config.json").write_text(json.dumps({"model_type": "qwen3_moe"}))
            gguf = Path(directory, "model.gguf")
            gguf.write_bytes(b"GGUF")
            args = SimpleNamespace(repo="a/model", revision="b" * 40, threads=1, file_exact=True,
                                   gguf=str(gguf), gguf_repo="a/gguf", gguf_revision="c" * 40)
            for entry in (layered.goldens, layered.equality):
                with self.subTest(entry=entry.__name__), patch.object(layered, "snapshot", return_value=directory), \
                     patch.object(layered, "runtime", return_value=(None, transformers)), \
                     patch.dict(sys.modules, {"transformers": transformers}), contextlib.redirect_stdout(io.StringIO()), \
                     self.assertRaisesRegex(RuntimeError, "stopped before reference setup"):
                    entry(args)
                loader.assert_called_with(args.repo, revision=args.revision, local_files_only=True)

    def test_tokenizer_writer_can_use_the_cached_file_without_hub(self):
        tok = SimpleNamespace(encode=lambda text, add_special_tokens: SimpleNamespace(ids=[len(text)]))
        tokenizers = SimpleNamespace(Tokenizer=SimpleNamespace(from_file=MagicMock(return_value=tok)))
        hub = SimpleNamespace(hf_hub_download=MagicMock(return_value="downloaded.json"))
        with tempfile.TemporaryDirectory(prefix="llmx_layered_tokenizer_") as directory, \
             patch.dict(sys.modules, {"tokenizers": tokenizers, "huggingface_hub": hub}), \
             contextlib.redirect_stdout(io.StringIO()):
            args = SimpleNamespace(repo="a/model", revision="b" * 40, gguf_repo="a/gguf", gguf_file="model.gguf", output_dir=directory)
            generator.gen_tokenizer(args, "cached.json")
            hub.hf_hub_download.assert_not_called()
            tokenizers.Tokenizer.from_file.assert_called_with("cached.json")
            offline = Path(directory, "baseline_tokenizer.json").read_bytes()
            generator.gen_tokenizer(args)
            hub.hf_hub_download.assert_called_once_with(args.repo, "tokenizer.json", revision=args.revision)
            tokenizers.Tokenizer.from_file.assert_called_with("downloaded.json")
            self.assertEqual(Path(directory, "baseline_tokenizer.json").read_bytes(), offline)

    def test_qwen3moe_runtime_keeps_its_existing_reference_versions(self):
        versions = {"torch": "2.5.1+cpu", "transformers": "4.55.2", "numpy": "2.2.6"}
        modules = {name: SimpleNamespace(__version__=version) for name, version in versions.items()}
        modules["torch"].set_num_threads = MagicMock()
        with tempfile.TemporaryDirectory(prefix="llmx_moe_environment_") as directory, \
             patch.dict(os.environ), patch.dict(sys.modules, modules):
            Path(directory, "config.json").write_text(json.dumps({"model_type": "qwen3_moe"}))
            self.assertEqual(layered.runtime(2, directory), (modules["torch"], modules["transformers"]))
            modules["torch"].set_num_threads.assert_called_once_with(2)
            self.assertEqual((os.environ["HF_HUB_OFFLINE"], os.environ["TRANSFORMERS_OFFLINE"]), ("1", "1"))
            for name, module in modules.items():
                with self.subTest(name=name), patch.object(module, "__version__", "unsupported"), self.assertRaisesRegex(SystemExit, name):
                    layered.runtime(2, directory)
    def test_arguments(self):
        with tempfile.TemporaryDirectory(prefix="llmx_layered_args_") as directory:
            gguf = os.path.join(directory, "model.gguf")
            Path(gguf).write_bytes(b"GGUF")
            out = os.path.join(directory, "goldens")
            good = ["goldens", "--repo", "Qwen/Qwen3.5-9B", "--revision", "a" * 40, "--gguf", gguf, "--output-dir", out]
            args = layered.parse_args(good + ["--gguf-repo", "a/b-GGUF", "--gguf-revision", "b" * 40])
            self.assertEqual((args.threads, args.gguf_repo, args.gguf_revision, args.file_exact), (6, "a/b-GGUF", "b" * 40, False))
            # A file-exact run's writers take the GGUF's weights and label the goldens with the record of them.
            exact = layered.parse_args(good + ["--file-exact"])
            self.assertTrue(exact.file_exact)
            self.assertEqual((lambda w: (w.weights_gguf, w.weights))(layered.writer_args(exact, {"file": "model.gguf"})), (gguf, {"file": "model.gguf"}))
            self.assertEqual((lambda w: (w.weights_gguf, w.weights))(layered.writer_args(exact)), (None, None))
            self.assertTrue(Path(args.output_dir).is_absolute())
            check = layered.parse_args(["equality", "--output", out, "--threads", "2"])
            self.assertEqual((check.threads, check.repo, check.revision, check.layers), (2, layered.EQUALITY_REPO, layered.EQUALITY_REVISION, None))
            cut = ["equality", "--output", out, "--repo", "Qwen/Qwen3.6-35B-A3B", "--revision", "c" * 40, "--layers", "4"]
            self.assertEqual((lambda a: (a.repo, a.revision, a.layers))(layered.parse_args(cut)), ("Qwen/Qwen3.6-35B-A3B", "c" * 40, 4))
            invalid = [["typo"], ["goldens"], ["equality"], good[:-2], good + ["--threads", "0"],
                       good[:4] + ["main"] + good[5:], good[:4] + ["A" * 40] + good[5:],
                       good[:2] + [directory] + good[3:], good[:6] + [gguf + ".missing"] + good[7:],
                       good + ["--gguf-revision", "b" * 40], good + ["--gguf-repo", "a/b", "--gguf-revision", "b" * 39],
                       good[:-1] + [generator.OUT_DIR], good + ["--layers", "4"],
                       cut[:-1] + ["0"], cut[:6] + ["main"] + cut[7:], cut[:4] + [directory] + cut[5:]]
            with contextlib.redirect_stderr(io.StringIO()):
                for argv in invalid:
                    with self.subTest(argv=argv), self.assertRaises(SystemExit) as error:
                        layered.parse_args(argv)
                    self.assertEqual(error.exception.code, 2)

    def test_writers_record_only_the_checkpoint_the_layered_run_names(self):
        # tools/gen_baseline.py's writers take the layered run's own arguments, which hold no pinned qwen35 checkpoint's loader record, so the goldens name only the checkpoint that ran.
        with tempfile.TemporaryDirectory(prefix="llmx_layered_writer_") as directory:
            gguf = os.path.join(directory, "model.gguf")
            Path(gguf).write_bytes(b"GGUF")
            args = layered.parse_args(["goldens", "--repo", "Qwen/Qwen3.5-9B", "--revision", "a" * 40, "--gguf", gguf, "--output-dir", os.path.join(directory, "goldens")])
        torch, transformers = (SimpleNamespace(__version__=generator.QWEN35_ENV[name]) for name in ("torch", "transformers"))
        self.assertEqual(generator.reference_metadata(layered.writer_args(args), torch, transformers),
                         {"reference_repo": "Qwen/Qwen3.5-9B", "reference_revision": "a" * 40, "reference_dtype": "float32", "attention": "eager", "device": "cpu",
                          "threads": 6, "torch_version": generator.QWEN35_ENV["torch"], "transformers_version": generator.QWEN35_ENV["transformers"]})

    def test_runtime_is_the_qwen35_environment_on_its_threads(self):
        # The layered reference runs offline in tools/gen_baseline.py's qwen35 environment, so it refuses what that refuses.
        torch = SimpleNamespace(__version__=generator.QWEN35_ENV["torch"], set_num_threads=MagicMock())
        transformers = SimpleNamespace(__version__=generator.QWEN35_ENV["transformers"])
        modules = {"torch": torch, "transformers": transformers, "tokenizers": SimpleNamespace(__version__=generator.QWEN35_ENV["tokenizers"]),
                   "transformers.models.qwen3_5.modeling_qwen3_5": SimpleNamespace()}
        with patch.dict(os.environ), patch.dict(sys.modules, modules):
            for package in generator.QWEN35_REPLACEMENTS:
                with self.subTest(package=package), \
                     patch.object(importlib.util, "find_spec", side_effect=lambda name, package=package: object() if name == package else None), \
                     self.assertRaises(SystemExit) as error:
                    layered.runtime(2)
                self.assertIn("the %s package" % package, str(error.exception))
            with patch.object(importlib.util, "find_spec", return_value=None):
                self.assertEqual(layered.runtime(2), (torch, transformers))
                torch.set_num_threads.assert_called_once_with(2)
                self.assertEqual((os.environ["HF_HUB_OFFLINE"], os.environ["TRANSFORMERS_OFFLINE"]), ("1", "1"))
                for module, version in ((torch, "2.6.0+cpu"), (transformers, "5.18.0")):
                    with self.subTest(version=version), patch.object(module, "__version__", version), self.assertRaises(SystemExit):
                        layered.runtime(2)

    def test_checkpoint_plan_takes_only_keys_the_model_takes_or_ignores(self):
        # transformers' loading rules as the tool reads them: every renaming, then at most one converter, which also names the pattern it matched.
        class WeightRenaming:
            def __init__(self, source, target):
                self.source, self.target = source, target

        class WeightConverter:
            __init__ = WeightRenaming.__init__

        def rename_source_key(key, renamings, converters, prefix, meta):
            for renaming in renamings:
                key = key.replace(renaming.source, renaming.target)
            for converter in converters:
                if converter.source in key:
                    return key.replace(converter.source, converter.target), converter.source
            return key, None

        core = types.ModuleType("transformers.core_model_loading")
        core.WeightRenaming, core.WeightConverter, core.rename_source_key = WeightRenaming, WeightConverter, rename_source_key
        conversion = types.ModuleType("transformers.conversion_mapping")
        conversion.get_model_conversion_mapping = lambda model: [WeightRenaming("model.language_model.", "model."), WeightConverter("mlp_fused", "mlp")]
        safetensors = types.ModuleType("safetensors")
        safetensors.safe_open = MagicMock()
        names = ["model.embed_tokens.weight", "model.layers.0.mlp.weight", "model.norm.weight", "lm_head.weight"]

        def model(tied):
            return SimpleNamespace(state_dict=lambda: dict.fromkeys(names), base_model_prefix="model",
                                   config=SimpleNamespace(tie_word_embeddings=tied))

        stored = {"model.language_model.embed_tokens.weight": "a.safetensors", "model.language_model.layers.0.mlp.weight": "b.safetensors",
                  "model.language_model.norm.weight": "b.safetensors", "lm_head.weight": "b.safetensors",
                  "mtp.fc.weight": "b.safetensors", "model.visual.blocks.0.weight": "a.safetensors"}
        with tempfile.TemporaryDirectory(prefix="llmx_layered_plan_") as directory, \
             patch.dict(sys.modules, {"transformers": types.ModuleType("transformers"), "transformers.core_model_loading": core,
                                      "transformers.conversion_mapping": conversion, "safetensors": safetensors}):
            def plan(weight_map, tied=False, cut=None):
                with open(os.path.join(directory, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
                    json.dump({"weight_map": weight_map}, f)
                return layered.checkpoint_plan(directory, model(tied), cut)

            found, unexpected = plan(stored)
            self.assertEqual(found, {name: (stored[key], key) for name, key in (
                ("model.embed_tokens.weight", "model.language_model.embed_tokens.weight"),
                ("model.layers.0.mlp.weight", "model.language_model.layers.0.mlp.weight"),
                ("model.norm.weight", "model.language_model.norm.weight"), ("lm_head.weight", "lm_head.weight"))})
            self.assertEqual(unexpected, ["model.visual.blocks.0.weight", "mtp.fc.weight"])
            untied = {key: file for key, file in stored.items() if key != "lm_head.weight"}
            self.assertNotIn("lm_head.weight", plan(untied, tied=True)[0])
            # A model cut to its first layer leaves the later layers' keys unused, and only such a model does.
            deeper = dict(stored, **{"model.language_model.layers.1.mlp.weight": "b.safetensors"})
            self.assertEqual(plan(deeper, cut=1)[1], ["model.language_model.layers.1.mlp.weight", "model.visual.blocks.0.weight", "mtp.fc.weight"])
            self.assertEqual((layered.past_cut("model.layers.3.mlp.weight", 3), layered.past_cut("model.layers.2.mlp.weight", 3),
                              layered.past_cut("mtp.layers.3.mlp.weight", 3), layered.past_cut("model.layers.3.mlp.weight", None)), (True, False, False, False))
            converted = dict(untied, **{"model.language_model.layers.0.mlp_fused.weight": "b.safetensors"})
            del converted["model.language_model.layers.0.mlp.weight"]
            for weight_map, tied, error in ((untied, False, "no checkpoint key gives lm_head.weight"),
                                            (deeper, False, "neither takes nor ignores, such as model.language_model.layers.1.mlp.weight"),
                                            (dict(stored, **{"model.language_model.layers.0.extra": "a.safetensors"}), False, "neither takes nor ignores"),
                                            ({key: file for key, file in stored.items() if "norm" not in key}, False, "no checkpoint key gives model.norm.weight"),
                                            (converted, True, "converts model.language_model.layers.0.mlp_fused.weight on load"),
                                            (dict(stored, **{"model.norm.weight": "a.safetensors"}), False, "both give model.norm.weight")):
                with self.subTest(error=error), self.assertRaises(SystemExit) as refused:
                    plan(weight_map, tied)
                self.assertIn(error, str(refused.exception))

    def test_config_chooses_the_model_and_cuts_its_layers(self):
        def config_class(kind):
            def load(directory, local_files_only):
                self.assertTrue(local_files_only)
                return SimpleNamespace(kind=kind, num_hidden_layers=4, layer_types=["linear_attention"] * 3 + ["full_attention"])
            return SimpleNamespace(from_pretrained=load)
        transformers = SimpleNamespace(Qwen3_5TextConfig=config_class("dense"), Qwen3_5ForCausalLM="dense model",
                                       Qwen3_5MoeTextConfig=config_class("moe"), Qwen3_5MoeForCausalLM="moe model")
        with tempfile.TemporaryDirectory(prefix="llmx_layered_config_") as directory:
            def chosen(raw, layers=None):
                with open(os.path.join(directory, "config.json"), "w", encoding="utf-8") as f:
                    json.dump(raw, f)
                causal, config = layered.text_model(transformers, directory, layers)
                return causal, config.kind, config._attn_implementation, config.num_hidden_layers, config.layer_types

            moe = {"model_type": "qwen3_5_moe", "text_config": {"model_type": "qwen3_5_moe_text"}}
            self.assertEqual(chosen(moe), ("moe model", "moe", "eager", 4, ["linear_attention"] * 3 + ["full_attention"]))
            self.assertEqual(chosen(moe, 3), ("moe model", "moe", "eager", 3, ["linear_attention"] * 3))
            self.assertEqual(chosen({"model_type": "qwen3_5_text"})[:2], ("dense model", "dense"))
            for raw, layers, error in ((moe, 5, "has 4 decoder layers, not 5"), (moe, 0, "has 4 decoder layers, not 0"),
                                       ({"model_type": "qwen3_next"}, None, "holds a qwen3_next model")):
                with self.subTest(error=error), self.assertRaises(SystemExit) as refused:
                    chosen(raw, layers)
                self.assertIn(error, str(refused.exception))

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy, which the file-exact reference decodes with")
    def test_file_weights_give_one_layer_of_the_file(self):
        # A file-exact run reads one module's parameters at a time from the GGUF, as HF names them.
        torch = SimpleNamespace(from_numpy=lambda values: values)
        with tempfile.TemporaryDirectory(prefix="llmx_layered_file_") as directory:
            path = os.path.join(directory, "tiny.gguf")
            state = write_routed_layer(path)
            weights = layered.FileWeights(path)
            got = weights.read(torch, sorted(state))
            self.assertEqual({name: (list(value.shape), value.reshape(-1).tolist()) for name, value in got.items()}, state)
            self.assertEqual(weights.types, {"F32": len(state)})
            for names in (sorted(state)[1:], sorted(state) + ["model.layers.0.mlp.extra.weight"]):
                with self.subTest(names=len(names)), self.assertRaisesRegex(SystemExit, "the GGUF gives"):
                    weights.read(torch, names)

    def test_provenance_names_the_f32_tensors_it_had_nothing_to_compare_with(self):
        provenance = layered.Provenance.__new__(layered.Provenance)
        provenance.spec = SimpleNamespace(F32=0)
        provenance.gguf = SimpleNamespace(tensors=[SimpleNamespace(name=name, type=kind) for name, kind in (
            ("blk.0.attn_norm.weight", 0), ("blk.0.attn_qkv.weight", 12), ("blk.64.nextn.enorm.weight", 0))])
        provenance.equal, provenance.differ, provenance.linear_layers, provenance.grouped = 1, [], 0, 0
        provenance.k_heads = provenance.v_heads = 16
        provenance.compared = {"blk.0.attn_norm.weight"}
        self.assertEqual(provenance.record()["f32_not_compared"], ["blk.64.nextn.enorm.weight"])
        provenance.compared.add("blk.64.nextn.enorm.weight")
        self.assertNotIn("f32_not_compared", provenance.record())

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy, which the GGUF comparison reads tensors with")
    def test_float32_steps_count_across_zero(self):
        import numpy
        below_one = struct.unpack("<f", struct.pack("<I", 0x3F7FFFFF))[0]
        tiny = struct.unpack("<f", struct.pack("<I", 1))[0]
        a = numpy.array([1.0, 1.0, -1.0, 0.0, -0.0, tiny, -2.5], dtype=numpy.float32)
        b = numpy.array([1.0, below_one, -below_one, -0.0, tiny, -tiny, -2.5], dtype=numpy.float32)
        self.assertEqual(layered.float32_steps(a, b).tolist(), [0, 1, 1, 0, 1, 2, 0])

    def test_label_keeps_the_writers_fields_in_order(self):
        extra = {"gguf_repo": None, "gguf_file": "m.gguf", "gguf_sha256": "0" * 64}
        with tempfile.TemporaryDirectory(prefix="llmx_layered_label_") as directory:
            path = os.path.join(directory, "golden.json")
            # The logit and tokenizer writers name the GGUF, and the perplexity writer does not.
            for written, labelled in (({"_comment": "writer", "gguf_repo": None, "gguf_file": "m.gguf", "cases": []},
                                       [("_comment", layered.COMMENT), ("gguf_repo", None), ("gguf_file", "m.gguf"), ("gguf_sha256", "0" * 64), ("cases", [])]),
                                      ({"_comment": "writer", "transformers_version": "5.17.0", "text": "t"},
                                       [("_comment", layered.COMMENT), ("transformers_version", "5.17.0"), ("gguf_repo", None), ("gguf_file", "m.gguf"),
                                        ("gguf_sha256", "0" * 64), ("text", "t")])):
                generator._write(path, written)
                layered.label_golden(path, extra)
                with open(path, encoding="utf-8") as f:
                    self.assertEqual(list(json.load(f).items()), labelled)
            generator._write(path, {"_comment": "writer", "text": "t"})
            with self.assertRaises(SystemExit):
                layered.label_golden(path, extra)

    def test_one_window_rule_gives_every_perplexity_golden(self):
        import baseline_layered
        paths = [Path(generator.OUT_DIR) / "baseline_perplexity.json", Path(generator.OUT_DIR) / "qwen3-8b" / "baseline_perplexity.json"]
        paths += [golden["data"] / "baseline_perplexity.json" for golden in baseline_layered.GOLDENS.values()]
        for path in paths:
            with self.subTest(path=path.parent.name):
                doc = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(doc["text"], generator.ppl_text())
                self.assertEqual([(case["context_size"], case["max_chunks"]) for case in doc["chunk_cases"]], list(generator.PPL_WINDOWS))
                for case in doc["chunk_cases"]:
                    spans = generator.ppl_window_bounds(doc["n_tokens"], case["context_size"], case["max_chunks"])
                    self.assertEqual((len(spans), sum(end - start for start, end in spans), sum(end - start - 1 for start, end in spans)),
                                     (case["chunks"], case["used_tokens"], case["n_scored"]))

    def test_committed_equality_records_show_the_layered_forward_equal(self):
        import baseline_layered
        # The dense record runs Qwen3.5-0.8B whole, and the routed one the checkpoint of the Qwen3.6-35B-A3B goldens cut to its first four layers, three linear and one full attention.
        routed = next(json.loads((golden["data"] / "baseline_logits.json").read_text(encoding="utf-8"))
                      for golden in baseline_layered.GOLDENS.values() if golden["name"].startswith("Qwen3.6-35B-A3B"))
        records = {"layered_equality.json": (layered.EQUALITY_REPO, layered.EQUALITY_REVISION, None, None),
                   "layered_equality_qwen3.6-35b-a3b.json": (routed["reference_repo"], routed["reference_revision"],
                                                             "the first 4, of which 1 full attention", routed["layered"]["experts_implementation"])}
        for name, (repo, revision, layers, experts) in records.items():
            with self.subTest(record=name):
                doc = json.loads((Path(generator.OUT_DIR) / name).read_text(encoding="utf-8"))
                self.assertEqual((doc["reference_repo"], doc["reference_revision"], doc.get("decoder_layers"), doc.get("experts_implementation")),
                                 (repo, revision, layers, experts))
                self.assertEqual((doc["transformers_version"], doc["torch_version"], doc["reference_dtype"], doc["attention"]),
                                 (generator.QWEN35_ENV["transformers"], generator.QWEN35_ENV["torch"], "float32", "eager"))
                self.assertTrue(doc["parameters"] > 0 and doc["parameters_equal"] == doc["parameters"] and doc["inv_freq_equal"])
                # Every input the goldens take, each prompt and the text whole and in every window, with the full forward's logits bit for bit.
                tokens = next(case["tokens"] for case in doc["cases"] if case["label"] == "perplexity")
                self.assertEqual([case["label"] for case in doc["cases"]], layered.row_labels(len(generator.LOGIT_PROMPTS), tokens))
                for case in doc["cases"]:
                    with self.subTest(case=case["label"]):
                        self.assertRegex(case["full_sha256"], r"^[0-9a-f]{64}$")
                        self.assertEqual((case["layered_sha256"], case["max_abs_difference"]), (case["full_sha256"], 0.0))

    def test_committed_layered_goldens_are_the_generators(self):
        import baseline_layered
        for digest, golden in baseline_layered.GOLDENS.items():
            with self.subTest(golden=golden["name"]):
                docs = {name: json.loads((golden["data"] / name).read_text(encoding="utf-8")) for name in golden["fixture_sha256"]}
                self.assertEqual([case["text"] for case in docs["baseline_tokenizer.json"]["cases"]], generator.CASES)
                self.assertEqual([case["text"] for case in docs["baseline_logits.json"]["cases"]], generator.LOGIT_PROMPTS)
                self.assertTrue(all(doc["_comment"] == layered.COMMENT and doc["gguf_sha256"] == digest for doc in docs.values()))
                for name in ("baseline_logits.json", "baseline_perplexity.json"):
                    doc = docs[name]
                    self.assertEqual((doc["transformers_version"], doc["torch_version"], doc["reference_dtype"], doc["attention"], doc["threads"]),
                                     (generator.QWEN35_ENV["transformers"], generator.QWEN35_ENV["torch"], "float32", "eager", 6))
                    self.assertRegex(doc["reference_revision"], r"^[0-9a-f]{40}$")
                    self.assertTrue(doc["layered"]["safetensors_sha256"])
                    # A routed model's goldens name the experts implementation HF's default ran.
                    self.assertEqual(doc["layered"].get("experts_implementation"), "grouped_mm" if golden["name"].startswith("Qwen3.6-35B-A3B") else None)


class DtypeCalibration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("llmx_calibrate_dtype", SCRIPT.with_name("calibrate_dtype.py"))
        cls.calibration = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.calibration)

    def test_rounds_matrix_inputs_but_not_router_or_norm(self):
        class Linear:
            def __init__(self):
                self.register_forward_pre_hook = MagicMock()
        query, gate, router, expert, head = [Linear() for _ in range(5)]
        norm = SimpleNamespace(register_forward_pre_hook=MagicMock())
        model = SimpleNamespace(named_modules=lambda: [("layer.self_attn.q_proj", query), ("layer.mlp.gate_proj", gate),
            ("layer.mlp.gate", router), ("layer.mlp.experts.0.up_proj", expert), ("lm_head", head), ("layer.norm", norm)])
        torch = SimpleNamespace(nn=SimpleNamespace(Linear=Linear), float16="f16", bfloat16="bf16", float32="f32")
        for dtype in ("f16", "bf16"):
            handles = self.calibration.rounding(model, torch, dtype)
            self.assertEqual(len(handles), 4)
            for module in (query, gate, expert, head):
                hook = module.register_forward_pre_hook.call_args.args[0]
                value, extra = MagicMock(), object()
                result = hook(module, (value, extra))
                value.to.assert_called_once_with(dtype)
                value.to.return_value.to.assert_called_once_with("f32")
                self.assertIs(result[0], value.to.return_value.to.return_value)
                self.assertIs(result[1], extra)
        router.register_forward_pre_hook.assert_not_called()
        norm.register_forward_pre_hook.assert_not_called()

    def test_failed_reference_removes_rounding_hooks(self):
        handles = [MagicMock(), MagicMock()]
        reference = SimpleNamespace(reference_outputs=MagicMock(side_effect=[([], []), RuntimeError("reference failed")]))
        with patch.dict(sys.modules, {"gen_baseline": reference}), patch.object(self.calibration, "rounding", return_value=handles):
            with self.assertRaisesRegex(RuntimeError, "reference failed"):
                self.calibration.errors(object(), object(), "f16")
        for handle in handles:
            handle.remove.assert_called_once_with()

    def test_pinned_environment_and_frozen_output(self):
        torch = SimpleNamespace(__version__="2.5.1+cpu", set_num_threads=MagicMock(), get_num_threads=lambda: 1)
        transformers = SimpleNamespace(__version__="4.55.2")
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"torch": torch, "transformers": transformers}):
            path = Path(directory) / "budget.json"
            with patch.object(self.calibration, "models", return_value=[("test", 7, object(), [])]), patch.object(self.calibration, "errors", return_value=(0.25, 0.125)), contextlib.redirect_stdout(io.StringIO()):
                self.calibration.main(["--output", str(path)])
                torch.set_num_threads.assert_called_once_with(1)
                doc = json.loads(path.read_text())
                self.assertEqual(doc["threads"], 1)
                for dtype in ("f16", "bf16"):
                    self.assertEqual((doc["dtypes"][dtype]["logit_budget"], doc["dtypes"][dtype]["nll_budget"]), (0.5, 0.25))
                original = path.read_bytes()
                with self.assertRaises(FileExistsError):
                    self.calibration.main(["--output", str(path)])
                self.assertEqual(path.read_bytes(), original)
            for module in (torch, transformers):
                with patch.object(module, "__version__", "wrong"), patch.object(self.calibration, "models") as models:
                    with self.assertRaisesRegex(SystemExit, "requires torch"):
                        self.calibration.main(["--output", str(path)])
                    models.assert_not_called()
                self.assertEqual(path.read_bytes(), original)


def run():
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(case) for case in (ReferenceGenerator, LayeredReference, DtypeCalibration))
    result = unittest.TextTestRunner(stream=sys.stdout).run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
