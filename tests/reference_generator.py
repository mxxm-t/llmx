import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch, call


SCRIPT = Path(__file__).resolve().parents[1] / "tools/gen_baseline.py"
SPEC = importlib.util.spec_from_file_location("llmx_gen_baseline", SCRIPT)
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)


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

    def test_fixtures_are_pinned_once(self):
        import baseline
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
        # The gate is the models with bounds, in the file's order.
        self.assertEqual([spec["file"] for spec in baseline.BASELINE_MODELS], [spec["file"] for spec in pinned if spec["gate"]])
        # The six Qwen3 files pinned ahead of their types join the gate with them: the hosted HF job is to download UD-Q8_K_XL, IQ4_XS and Q2_K, and the other three are checked by hand.
        later = [spec for spec in pinned if not spec["gate"] and spec["family"] == "qwen3"]
        self.assertEqual(sorted(spec["file"] for spec in later if spec["hosted"]),
                         ["Qwen3-0.6B-IQ4_XS.gguf", "Qwen3-0.6B-Q2_K.gguf", "Qwen3-0.6B-UD-Q8_K_XL.gguf"])
        self.assertEqual(sorted(spec["file"] for spec in later if not spec["hosted"]),
                         ["Qwen3-0.6B-BF16.gguf", "Qwen3-0.6B-IQ4_NL.gguf", "Qwen3-0.6B-Q3_K_S.gguf"])
        # The qwen35 files join the gate with their bounds: the two 0.8B files in the hosted HF job, the 4B by hand, and each is a file of a checkpoint QWEN35_MODELS pins.
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
                   ["qwen35", "--model", "Qwen3.5-0.8B", "--threads", "0"]]
        with contextlib.redirect_stderr(io.StringIO()):
            for argv in invalid:
                with self.subTest(argv=argv), self.assertRaises(SystemExit) as error:
                    generator.parse_args(argv)
                self.assertEqual(error.exception.code, 2)
        # A pinned qwen35 file's file-exact reference is the checkpoint whose goldens are for it, which --repo cannot replace.
        with tempfile.TemporaryDirectory(prefix="llmx_reference_qwen35_file_exact_") as directory:
            path = os.path.join(directory, "Qwen3.5-0.8B-Q4_K_M.gguf")
            Path(path).write_bytes(b"GGUF")
            out = os.path.join(directory, "goldens")
            args = generator.parse_args(["file-exact", "--weights-gguf", path, "--output-dir", out])
            self.assertEqual((args.family, args.qwen35_model, args.repo, args.gguf_repo, args.gguf_file),
                             ("qwen35", "Qwen3.5-0.8B", "Qwen/Qwen3.5-0.8B", "unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"))
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                generator.parse_args(["file-exact", "--weights-gguf", path, "--output-dir", out, "--repo", "Qwen/Qwen3.5-4B", "--revision", "a" * 40])

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

    def test_qwen35_reference_loads_as_pinned(self):
        # load_qwen35 against doubles: the versions, float32 and eager attention, transformers' torch forms, and the keys a load may leave out.
        dtype = object()

        def load(transformers_version="5.17.0", installed=(), missing=(), unexpected=()):
            model = MagicMock()
            model.cpu.return_value = model
            info = {"missing_keys": list(missing), "mismatched_keys": [],
                    "unexpected_keys": ["mtp.fc.weight", "model.visual.blocks.0.attn.qkv.weight"] + list(unexpected)}
            causal = SimpleNamespace(from_pretrained=MagicMock(return_value=(model, info)))
            modeling = SimpleNamespace(torch_chunk_gated_delta_rule=MagicMock(return_value="chunked"), torch_recurrent_gated_delta_rule=MagicMock())
            torch = SimpleNamespace(__version__="2.5.1+cpu", float32=dtype, set_num_threads=MagicMock())
            modules = {"torch": torch, "tokenizers": SimpleNamespace(__version__="0.23.2"),
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
        refusals = [({"transformers_version": "5.16.0"}, "5.17.0"), ({"installed": ("kernels",)}, "kernels"),
                    ({"installed": ("fla",)}, "fla"), ({"missing": ("model.norm.weight",)}, "model.norm.weight"),
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
        # A model whose files are all checked by hand may have none yet, and a model with a hosted file has all of them.
        import baseline
        import baseline_qwen35
        pinned = set()
        for spec in generator.QWEN35_MODELS.values():
            directory = Path(generator.OUT_DIR) / spec["directory"]
            docs = {path.name: json.loads(path.read_text(encoding="utf-8")) for path in directory.glob("*.json")}
            if not docs and not any(baseline.pinned_fixture(file)["hosted"] for file in spec["gguf_files"]):
                continue
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

    def test_qwen35_environment_is_offline_at_its_version_without_replacements(self):
        # The goldens come from HF's own torch functions at transformers 5.17.0, so another version, or a package HF's qwen3_5 code would run in their place, is refused.
        modeling = SimpleNamespace()

        def modules(version):
            return {"torch": SimpleNamespace(__version__="test-torch"), "transformers": SimpleNamespace(__version__=version),
                    "transformers.models.qwen3_5.modeling_qwen3_5": modeling}

        with patch.dict(os.environ), patch.dict(sys.modules, modules("5.17.0")), \
             patch.object(generator.importlib.util, "find_spec", return_value=None) as find:
            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            self.assertIs(generator.qwen35_environment()[2], modeling)
            self.assertEqual((os.environ["HF_HUB_OFFLINE"], os.environ["TRANSFORMERS_OFFLINE"]), ("1", "1"))
            self.assertEqual(sorted(c.args[0] for c in find.call_args_list), ["causal_conv1d", "fla", "kernels"])
        with patch.dict(os.environ), patch.dict(sys.modules, modules("5.16.0")), \
             patch.object(generator.importlib.util, "find_spec", return_value=None), self.assertRaisesRegex(SystemExit, "5.17.0"):
            generator.qwen35_environment()
        for package in generator.QWEN35_REPLACEMENTS:
            with self.subTest(package=package), patch.dict(os.environ), patch.dict(sys.modules, modules("5.17.0")), \
                 patch.object(generator.importlib.util, "find_spec", side_effect=lambda name: object() if name == package else None), \
                 self.assertRaisesRegex(SystemExit, package):
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
        self.assertEqual((doc["transformers_version"], doc["dtype"], doc["attention"]), (generator.QWEN35_TRANSFORMERS, "float32", "eager"))
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


def run():
    result = unittest.TextTestRunner(stream=sys.stdout).run(unittest.defaultTestLoader.loadTestsFromTestCase(ReferenceGenerator))
    return result.wasSuccessful()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
