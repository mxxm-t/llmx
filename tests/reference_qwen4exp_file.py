"""Independent sharded storage checks and optional pinned-HF proof of bounded Qwen4Exp adapters."""

import contextlib
import io
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import spec_decode
import qwen4exp
import qwen4exp_reference
import qwen4exp_file_reference as file_reference


class ShardStorage(unittest.TestCase):
    def test_cli_refuses_invalid_inputs_before_model_loading(self):
        with patch.object(file_reference, "environment") as environment:
            with self.assertRaisesRegex(ValueError, "attention"):
                file_reference.FileReference([], {}, attention="unsupported")
            environment.assert_not_called()
        with tempfile.TemporaryDirectory(prefix="qwen4exp_file_cli_") as directory:
            root = Path(directory)
            config, output, ids_file = root / "config.json", root / "out.json", root / "ids.json"
            config.write_text(json.dumps({"vocab_size": 32}), encoding="utf-8")
            ids_file.write_text("[1,2]", encoding="utf-8")
            base = ["--config", str(config), "--gguf", str(root / "missing.gguf"), "--output", str(output)]
            invalid = [["--input-ids", value] for value in ("[]", "[true]", "[1.5]", "[-1]", "[32]", "invalid")]
            invalid += [["--input-ids", "[1]", flag, value] for flag, value in
                        (("--last", "-1"), ("--top", "-1"), ("--top", "33"), ("--attention", "other"), ("--threads", "0"))]
            invalid += [["--input-ids", "[1,2]", "--max-tokens", "1"],
                        ["--input-ids", "[1]", "--input-ids-file", str(ids_file)],
                        ["--input-ids-file", str(root / "missing.json")], []]
            with patch.object(file_reference, "FileReference") as constructor, contextlib.redirect_stderr(io.StringIO()):
                for arguments in invalid:
                    with self.subTest(arguments=arguments), self.assertRaises(SystemExit):
                        file_reference.main(base + arguments)
                output.write_text("preserve", encoding="utf-8")
                with self.assertRaises(SystemExit):
                    file_reference.main(base + ["--input-ids-file", str(ids_file)])
                self.assertEqual(output.read_text(encoding="utf-8"), "preserve")
                constructor.assert_not_called()

    def test_shard_headers_bounds_and_contradictions(self):
        with tempfile.TemporaryDirectory(prefix="qwen4exp_file_shards_") as directory:
            root = Path(directory)
            first, second = root / "first.gguf", root / "second.gguf"
            metadata = {"general.architecture": (spec_decode.STRING, "qwen4exp"),
                        "split.no": (4, 0), "split.count": (4, 2), "split.tensors.count": (10, 2)}
            tensor = ("a", [2, 2], spec_decode.F32, struct.pack("<4f", 1, 2, 3, 4))
            spec_decode.write_gguf(first, metadata, [tensor])
            tail = {"split.no": (4, 1), "split.count": (4, 2), "split.tensors.count": (10, 2)}
            spec_decode.write_gguf(second, tail, [("b", *tensor[1:])])
            source = file_reference.Shards([second, first], 16)
            self.assertEqual(source.value("general.architecture"), "qwen4exp")
            self.assertEqual(len(source.tensors), 2)
            if spec_decode.np is not None:
                raw = spec_decode.GGUF.raw
                def bounded(file, tensor, first=0, count=None):
                    self.assertIsNotNone(count)
                    return raw(file, tensor, first, count)
                with patch.object(spec_decode.GGUF, "raw", bounded):
                    self.assertEqual(source.read("b", 2, 2).tolist(), [3, 4])
                with self.assertRaisesRegex(ValueError, "memory limit"):
                    file_reference.Shards([first, second], 4).read("a")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                file_reference.Shards([first], 16)
            bad = dict(tail, **{"general.architecture": (spec_decode.STRING, "qwen3")})
            spec_decode.write_gguf(second, bad, [("b", *tensor[1:])])
            with self.assertRaisesRegex(ValueError, "metadata differs"):
                file_reference.Shards([first, second], 16)
            spec_decode.write_gguf(second, tail, [tensor])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                file_reference.Shards([first, second], 16)


class PinnedHFAdapters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch, _ = file_reference.environment()
        cls.torch.set_num_threads(1)

    def test_sdpa_source_pin_refused_before_model_allocation(self):
        with patch.object(file_reference, "environment", return_value=(self.torch, None)), \
             patch.object(file_reference.reference, "_sha256_file", return_value="0" * 64) as digest, \
             patch.object(file_reference, "Shards") as shards, \
             patch("transformers.models.qwen4_exp.modeling_qwen4_exp.Qwen4ExpForCausalLM") as model:
            with self.assertRaisesRegex(ValueError, "SDPA integration source hash"):
                file_reference.FileReference([], qwen4exp_reference.TINY_CONFIG, attention="sdpa")
            self.assertEqual(digest.call_args.args[0].name, "sdpa_attention.py")
            shards.assert_not_called()
            model.assert_not_called()
        self.assertEqual(len(qwen4exp_reference.SOURCE_SHA256), 3)

    def test_original_hf_sdpa_selection_routes_and_head_geometry(self):
        from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
        from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM
        torch = self.torch
        for production_heads, length in ((False, 2057), (True, 129)):
            config = dict(qwen4exp_reference.TINY_CONFIG, num_hidden_layers=4,
                          layer_types=["linear_attention", "qwen_sparse_attention"] * 2,
                          max_position_embeddings=4096, indexer_compress_ratio=4, indexer_budget=2048)
            if production_heads:
                config.update(num_attention_heads=24, num_key_value_heads=2, head_dim=256, indexer_head_dim=128,
                              rope_parameters=dict(rope_theta=10000000.0, rope_type="default", partial_rotary_factor=.25,
                                                   mrope_section=[11, 11, 10]))
            hf_config = Qwen4ExpTextConfig(**config)
            hf_config._attn_implementation = "sdpa"
            hf_config._experts_implementation = "eager"
            torch.manual_seed(qwen4exp_reference.SEED + 19)
            model = Qwen4ExpForCausalLM(hf_config).eval().requires_grad_(False)
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name.endswith("hc_norm.weight"):
                        parameter.copy_((torch.arange(parameter.numel()).reshape(parameter.shape) % 7 - 3) / 16)
                    elif name.endswith(".ple.conv1d.weight"):
                        parameter.normal_(0, .01)
            ids = ((torch.arange(length) * 7 + 2) % 30 + 1)[None, :]
            for position in (62, 127, 2047, 2052):
                if position < length:
                    ids[0, position] = config["eos_token_id"]
            selected, routes, handles = [], [], []
            for name, module in model.named_modules():
                if name.endswith(".indexer"):
                    def mask(module, args, output):
                        selected.append((output == 0 if output.is_floating_point() else output).clone())
                    handles.append(module.register_forward_hook(mask))
                elif name.endswith(".mlp.gate"):
                    handles.append(module.register_forward_hook(lambda module, args, output: routes.append(output[2].clone())))
            try:
                with torch.inference_mode():
                    model.set_attn_implementation("eager")
                    expected = model(ids, use_cache=False).logits
                    old_selected, old_routes = selected[:], routes[:]
                    selected.clear()
                    routes.clear()
                    model.set_attn_implementation("sdpa")
                    if production_heads:
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                            actual = model(ids, use_cache=False).logits
                        self.assertIn("aten::_scaled_dot_product_flash_attention_for_cpu", {event.key for event in profile.key_averages()})
                    else:
                        actual = model(ids, use_cache=False).logits
                self.assertEqual(len(selected), 2)
                self.assertEqual(len(routes), 4)
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(old_selected, selected)))
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(old_routes, routes)))
                error = (actual - expected).abs().max().item()
                self.assertLessEqual(error, 2e-7)
                print("HF eager/SDPA: %d tokens, production heads %s, selected sets/router IDs identical, max logit error %.9g"
                      % (length, production_heads, error))
            finally:
                for handle in handles:
                    handle.remove()

    def test_nonselfinverse_value_head_order_and_nonzero_hyper_norms(self):
        from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
        from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM
        torch = self.torch
        config = dict(qwen4exp_reference.TINY_CONFIG, linear_num_value_heads=6)
        hf_config = Qwen4ExpTextConfig(**config)
        hf_config._attn_implementation = hf_config._experts_implementation = "eager"
        torch.manual_seed(qwen4exp_reference.SEED + 2)
        full = Qwen4ExpForCausalLM(hf_config).eval().requires_grad_(False)
        with torch.no_grad():
            for name, parameter in full.named_parameters():
                if name.endswith("hc_norm.weight"):
                    parameter.copy_((torch.arange(parameter.numel()).reshape(parameter.shape) % 7 - 3) / 16)
        state = {name: (tuple(value.shape), value.reshape(-1).tolist()) for name, value in full.state_dict().items()}
        with tempfile.TemporaryDirectory(prefix="qwen4exp_nonselfinverse_") as directory:
            path = Path(directory) / "model.gguf"
            qwen4exp.write_qwen4exp(path, config, state)
            adapted = file_reference.FileReference([path], config)
            order = qwen4exp._v_order(2, 6)
            self.assertNotEqual(order, adapted.head_order)
            ids = torch.tensor([qwen4exp_reference.CASES[0]["input_ids"]], dtype=torch.long)
            with torch.inference_mode():
                error = (adapted.forward(ids).logits - full(ids, use_cache=False).logits).abs().max().item()
            self.assertLessEqual(error, 2e-7)
            print("Qwen4Exp Hv6/Hk2 nonselfinverse order with nonzero HC norms: max logit error %.9g; mapping roundtrip %s"
                  % (error, adapted.mapping_roundtrip))

    def test_wide_vocabulary_nll_uses_float64_scoring(self):
        torch = self.torch
        vocab = 248320
        logits = torch.full((3, vocab), -10.0, dtype=torch.float32)
        logits[0, 0] = 0.0
        logits[1, -1] = 0.0
        ids = torch.tensor([1, 0, vocab - 1], dtype=torch.long)
        original = logits.clone()
        expected = math.log(math.fsum([1.0] + [math.exp(-10.0)] * (vocab - 1)))
        got = file_reference.score_nll(torch, logits, ids)
        old = torch.nn.functional.cross_entropy(logits[:-1], ids[1:], reduction="none")
        self.assertEqual(got.dtype, torch.float64)
        self.assertTrue(torch.equal(logits, original))
        self.assertLessEqual(max(abs(value - expected) for value in got.tolist()), 2e-14)
        error = max(abs(value - expected) for value in old.tolist())
        self.assertGreater(error, 1e-6)
        self.assertEqual(file_reference.score_nll(torch, logits[:1], ids[:1]).numel(), 0)
        print("Wide-vocabulary NLL: float64 analytic error %.9g; former F32 cross-entropy error %.9g"
              % (max(abs(value - expected) for value in got.tolist()), error))

    def test_original_hf_full_cached_and_failure_recovery(self):
        from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
        from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM
        torch = self.torch
        artifact = Path(__file__).resolve().parent / "data" / "qwen4exp_reference.json"
        document = json.loads(artifact.read_text(encoding="utf-8"))
        config = document["config"]
        state = qwen4exp.state_from_document(document)
        hf_config = Qwen4ExpTextConfig(**config)
        hf_config._attn_implementation = "eager"
        hf_config._experts_implementation = "eager"
        full = Qwen4ExpForCausalLM(hf_config).eval().requires_grad_(False)
        grouped_config = Qwen4ExpTextConfig(**config)
        grouped_config._attn_implementation = "eager"
        grouped = Qwen4ExpForCausalLM(grouped_config).eval().requires_grad_(False)
        original = {name: torch.tensor(values, dtype=torch.int64 if name.endswith(("layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes")) else torch.float32).reshape(shape)
                    for name, (shape, values) in state.items()}
        full.load_state_dict(original, strict=True)
        grouped.load_state_dict(original, strict=True)
        self.assertEqual(full.config._experts_implementation, "eager")
        self.assertEqual(grouped.config._experts_implementation, "grouped_mm")
        metadata, tensors = qwen4exp.map_qwen4exp(config, state)
        with tempfile.TemporaryDirectory(prefix="qwen4exp_hf_file_") as directory:
            root = Path(directory)
            first, second = root / "first.gguf", root / "ple.gguf"
            metadata = dict(metadata, **{"split.no": (4, 0), "split.count": (4, 2), "split.tensors.count": (10, len(tensors))})
            ordinary = [tensor for tensor in tensors if tensor[0] != "per_layer_token_embd.weight"]
            table = [tensor for tensor in tensors if tensor[0] == "per_layer_token_embd.weight"]
            spec_decode.write_gguf(first, metadata, ordinary)
            spec_decode.write_gguf(second, {"split.no": (4, 1), "split.count": (4, 2), "split.tensors.count": (10, len(tensors))}, table)
            adapted = file_reference.FileReference([first, second], config, max_tokens=32, max_decoded_bytes=1 << 20)
            reads = []
            decode_span = spec_decode.GGUF.decode_span

            def observed(file, tensor, start, count, numpy=True, tensor_sources=None):
                reads.append((tensor.name, start, count, tensor.count))
                return decode_span(file, tensor, start, count, numpy, tensor_sources)

            maximum = 0.0
            grouped_difference = 0.0
            router_ids = {"eager": [], "grouped": [], "adapted": []}
            router_hooks = []
            for label, model in (("eager", full), ("grouped", grouped), ("adapted", adapted.model)):
                for layer in model.model.layers:
                    def capture(module, args, output, label=label):
                        router_ids[label].append(output[2].tolist())
                    router_hooks.append(layer.mlp.gate.register_forward_hook(capture))
            with patch.object(spec_decode.GGUF, "decode_span", observed), torch.inference_mode():
                for case in document["cases"]:
                    ids = torch.tensor([case["input_ids"]], dtype=torch.long)
                    for recorded in router_ids.values():
                        recorded.clear()
                    expected = full(ids, use_cache=False).logits
                    grouped_logits = grouped(ids, use_cache=False).logits
                    canonical = torch.tensor(case["full_logits"], dtype=torch.float32).unsqueeze(0)
                    self.assertLessEqual((grouped_logits - canonical).abs().max().item(), 2e-7)
                    grouped_difference = max(grouped_difference, (expected - grouped_logits).abs().max().item())
                    got = adapted.forward(ids).logits
                    self.assertEqual(router_ids["eager"], router_ids["grouped"])
                    self.assertEqual(router_ids["eager"], router_ids["adapted"])
                    error = (expected - got).abs().max().item()
                    maximum = max(maximum, error)
                    self.assertLessEqual(error, 2e-7)
                    if case is document["cases"][0]:
                        target_count = ids.shape[1] - 1
                        nondivisor = next((rows for rows in range(2, target_count)
                                           if target_count % rows), target_count + 1)
                        expected_nll = file_reference.score_nll(torch, expected[0], ids[0])
                        before_hooks = [(dict(module._forward_pre_hooks), dict(module._forward_hooks))
                                        for module in adapted.model.modules()]
                        for rows in (1, nondivisor, max(1, target_count - 1), target_count + 1):
                            with patch.object(adapted, "load", wraps=adapted.load) as load:
                                actual_nll = adapted.sequence_nll(ids, rows=rows)
                            self.assertEqual(actual_nll.numel(), target_count)
                            self.assertEqual(actual_nll.dtype, torch.float64)
                            self.assertLessEqual((actual_nll - expected_nll).abs().max().item(), 4e-7)
                            self.assertEqual(sum(call.args[1] == "lm_head." for call in load.call_args_list), 1)
                            self.assertTrue(all(parameter.is_meta for parameter in adapted.model.parameters()))
                            self.assertTrue(all(embedding.weight.numel() == 0 for embedding in adapted.ple_embeddings))
                            self.assertEqual([(dict(module._forward_pre_hooks), dict(module._forward_hooks))
                                              for module in adapted.model.modules()], before_hooks)
                    full_cache = adapted_cache = None
                    start = 0
                    for length in document["slice_lengths"]:
                        part = ids[:, start:start + length]
                        expected_step = full(part, past_key_values=full_cache, use_cache=True)
                        actual_step = adapted.forward(part, past_key_values=adapted_cache, use_cache=True)
                        full_cache, adapted_cache = expected_step.past_key_values, actual_step.past_key_values
                        error = (expected_step.logits - actual_step.logits).abs().max().item()
                        maximum = max(maximum, error)
                        self.assertLessEqual(error, 2e-7)
                        start += length
                    if case["decode_token"] is not None:
                        token = torch.tensor([[case["decode_token"]]], dtype=torch.long)
                        expected_step = full(token, past_key_values=full_cache, use_cache=True)
                        actual_step = adapted.forward(token, past_key_values=adapted_cache, use_cache=True)
                        self.assertLessEqual((expected_step.logits - actual_step.logits).abs().max().item(), 2e-7)
            for hook in router_hooks:
                hook.remove()
            a_error = 0.0
            norm_error = 0.0
            for name, shape in adapted.expected.items():
                if name.endswith(".A_log") or ("norm" in name and name.endswith(".weight")):
                    error = (torch.from_numpy(adapted._parts(name, shape)) - original[name]).abs().max().item()
                    if name.endswith(".A_log"):
                        a_error = max(a_error, error)
                    else:
                        norm_error = max(norm_error, error)
            for name, _, count, total in reads:
                if name.endswith(("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")) or name == "per_layer_token_embd.weight":
                    self.assertLess(count, total, name)
            self.assertFalse(any(not parameter.is_meta for parameter in adapted.model.parameters()))
            self.assertTrue(all(embedding.weight.numel() == 0 for embedding in adapted.ple_embeddings))
            self.assertFalse(any(module._forward_pre_hooks or module._forward_hooks for module in adapted.model.modules()))

            def fail(name, module):
                if name == "model.layers.1":
                    raise RuntimeError("injected reference failure")
            ids = torch.tensor([document["cases"][0]["input_ids"]], dtype=torch.long)
            with self.assertRaisesRegex(RuntimeError, "injected"):
                adapted.forward(ids, on_load=fail)
            self.assertTrue(all(parameter.is_meta for parameter in adapted.model.parameters()))
            self.assertFalse(any(module._forward_pre_hooks or module._forward_hooks for module in adapted.model.modules()))
            with torch.inference_mode():
                self.assertLessEqual((adapted.forward(ids).logits - full(ids, use_cache=False).logits).abs().max().item(), 2e-7)
            def fail_head(name, module):
                if name == "lm_head":
                    raise RuntimeError("injected head callback failure")
            with self.assertRaisesRegex(RuntimeError, "head callback"):
                adapted.sequence_nll(ids, rows=3, on_load=fail_head)
            self.assertTrue(all(parameter.is_meta for parameter in adapted.model.parameters()))
            self.assertTrue(all(embedding.weight.numel() == 0 for embedding in adapted.ple_embeddings))
            self.assertFalse(any(module._forward_pre_hooks or module._forward_hooks for module in adapted.model.modules()))
            expected_nll = file_reference.score_nll(torch, full(ids, use_cache=False).logits[0], ids[0])
            self.assertLessEqual((adapted.sequence_nll(ids, rows=3) - expected_nll).abs().max().item(), 4e-7)
            with patch.object(adapted.model.lm_head, "forward", side_effect=RuntimeError("injected head scoring failure")):
                with self.assertRaisesRegex(RuntimeError, "head scoring"):
                    adapted.sequence_nll(ids, rows=3)
            self.assertTrue(all(parameter.is_meta for parameter in adapted.model.parameters()))
            self.assertFalse(any(module._forward_pre_hooks or module._forward_hooks for module in adapted.model.modules()))
            with patch.object(adapted.model.lm_head, "forward", return_value=torch.full((3, config["vocab_size"]), float("nan"))):
                with self.assertRaisesRegex(ValueError, "nonfinite"):
                    adapted.sequence_nll(ids, rows=3)
            self.assertTrue(all(parameter.is_meta for parameter in adapted.model.parameters()))
            self.assertLessEqual((adapted.sequence_nll(ids, rows=3) - expected_nll).abs().max().item(), 4e-7)
            for rows in (0, -1, 1.5, True):
                with self.assertRaisesRegex(ValueError, "row count"):
                    adapted.sequence_nll(ids, rows=rows)
            with self.assertRaisesRegex(ValueError, "token limit"):
                adapted.sequence_nll(ids[:, :1])
            complete = adapted.forward(ids).logits[0]
            all_values = file_reference.output_values(torch, complete, ids[0])
            self.assertEqual(all_values["logits"], complete.tolist())
            self.assertEqual(all_values["nll"], file_reference.score_nll(torch, complete, ids[0]).tolist())
            self.assertEqual(all_values["output_coverage"], "all")
            self.assertEqual(all_values["scoring_scope"], "whole-sequence next-token targets")
            for keep in (1, ids.shape[1] + 1):
                selected = adapted.forward(ids, logits_to_keep=keep).logits[0]
                self.assertLessEqual((selected - complete[-keep:]).abs().max().item(), 2e-7)
                summary = file_reference.output_values(torch, selected, ids[0], top=5)
                self.assertEqual(summary["n_scored"], selected.shape[0] - 1)
            picks = torch.tensor([0, 3], dtype=torch.long)
            self.assertLessEqual((adapted.forward(ids, logits_to_keep=picks).logits[0] - complete[picks]).abs().max().item(), 2e-7)
            tail = adapted.forward(ids, logits_to_keep=4).logits[0]
            self.assertLessEqual((tail - complete[-4:]).abs().max().item(), 2e-7)
            values = file_reference.output_values(torch, tail, ids[0], top=5)
            self.assertNotIn("logits", values)
            self.assertEqual(values["positions"], list(range(ids.shape[1] - 4, ids.shape[1])))
            self.assertEqual(values["output_coverage"], "tail")
            self.assertEqual(values["n_scored"], 3)
            self.assertEqual(values["nll"], file_reference.score_nll(torch, tail, ids[0, -4:]).tolist())
            self.assertEqual(values["nll_target_positions"], list(range(ids.shape[1] - 3, ids.shape[1])))
            self.assertEqual(values["nll_target_ids"], ids[0, -3:].tolist())
            self.assertEqual(values["nll_target_logits"], tail[:-1].gather(1, ids[0, -3:, None]).squeeze(1).tolist())
            for row, ranking, logits in zip(tail, values["top_ids"], values["top_logits"]):
                expected_ids = sorted(range(row.numel()), key=lambda token: (-row[token].item(), token))[:5]
                self.assertEqual(ranking, expected_ids)
                self.assertEqual(logits, row[ranking].tolist())
            tied = file_reference.output_values(torch, torch.tensor([[1., 2., 2., 2., 0.]]), torch.tensor([1]), top=2)
            self.assertEqual(tied["top_ids"], [[1, 2]])
            self.assertEqual(tied["nll"], [])
            self.assertEqual(tied["nll_target_logits"], [])
            self.assertIsNone(tied["perplexity"])
            outside = file_reference.output_values(torch, torch.tensor([[4., 3., 2., 1., -5.], [5., 4., 3., -6., 2.], [1., 0., 0., 0., 0.]]),
                                                  torch.tensor([1, 2, 0, 4, 3]), top=2)
            self.assertEqual(outside["nll_target_positions"], [3, 4])
            self.assertEqual(outside["nll_target_ids"], [4, 3])
            self.assertEqual(outside["nll_target_logits"], [-5., -6.])
            self.assertTrue(all(target not in top for target, top in zip(outside["nll_target_ids"], outside["top_ids"])))
            sdpa = file_reference.FileReference([first, second], config, attention="sdpa")
            self.assertEqual(sdpa.attention_source_sha256, {"integrations/sdpa_attention.py": file_reference.SDPA_SOURCE_SHA256})
            self.assertEqual(adapted.attention_source_sha256, {})
            self.assertLessEqual((sdpa.forward(ids, logits_to_keep=4).logits[0] - tail).abs().max().item(), 2e-7)
            config_path, ids_path, output = root / "config.json", root / "ids.json", root / "output.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            ids_path.write_text(json.dumps(ids[0].tolist()), encoding="utf-8")
            default_output = root / "default.json"
            with patch.object(file_reference, "FileReference", return_value=adapted) as constructor:
                file_reference.main(["--config", str(config_path), "--gguf", str(first), str(second),
                                     "--input-ids", ids_path.read_text(encoding="utf-8"), "--output", str(default_output)])
                self.assertEqual(constructor.call_args.kwargs["attention"], "eager")
            default = json.loads(default_output.read_text(encoding="utf-8"))
            self.assertEqual(default["attention"], "eager")
            self.assertEqual(default["logits"], complete.tolist())
            self.assertEqual(default["nll"], all_values["nll"])
            self.assertEqual((default["last"], default["top"]), (0, 0))
            arguments = ["--config", str(config_path), "--gguf", str(first), str(second), "--input-ids-file", str(ids_path),
                         "--output", str(output), "--attention", "sdpa", "--last", "4", "--top", "5"]
            with patch.object(file_reference, "FileReference", return_value=sdpa) as constructor:
                file_reference.main(arguments)
                self.assertEqual(constructor.call_args.kwargs["attention"], "sdpa")
            written = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(written["attention"], "sdpa")
            self.assertEqual(written["attention_source_sha256"], sdpa.attention_source_sha256)
            self.assertEqual(written["transformers_source_sha256"], qwen4exp_reference.SOURCE_SHA256)
            self.assertEqual(written["positions"], values["positions"])
            self.assertEqual(written["top_ids"], values["top_ids"])
            self.assertEqual(written["scoring_scope"], "tail next-token targets")
            self.assertNotIn("logits", written)
            self.assertEqual(set(root.iterdir()), {first, second, config_path, ids_path, output, default_output})
            with self.assertRaisesRegex(ValueError, "token limit"):
                adapted.forward(torch.zeros((1, 33), dtype=torch.long))
            with self.assertRaisesRegex(ValueError, "outside vocabulary"):
                adapted.forward(torch.tensor([[config["vocab_size"]]], dtype=torch.long))
            print("Qwen4Exp file-exact adapters: three full prompts, cached slices, two decode steps, selected rows/experts and failure recovery; max logit error %.9g" % maximum)
            print("HF eager vs canonical grouped_mm max logit difference %.9g; all full-prompt router IDs identical; inverse A_log error %.9g, norm offset error %.9g"
                  % (grouped_difference, a_error, norm_error))


if __name__ == "__main__":
    unittest.main()
