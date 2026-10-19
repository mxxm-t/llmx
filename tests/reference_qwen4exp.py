"""Dependency-free checks for the tiny Qwen4-Exp HF fixture specification."""

import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest


MODULE = Path(__file__).resolve().parents[1] / "tools" / "qwen4exp_reference.py"
SPEC = importlib.util.spec_from_file_location("llmx_qwen4exp_reference", MODULE)
reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reference)
ARTIFACT = Path(__file__).resolve().parent / "data" / "qwen4exp_reference.json"


def assert_finite_shape(test, value, shape, label):
    test.assertIsInstance(value, list, label)
    test.assertEqual(len(value), shape[0], label)
    if len(shape) == 1:
        for item in value:
            test.assertIsInstance(item, (int, float), label)
            test.assertNotIsInstance(item, bool, label)
            test.assertTrue(math.isfinite(item), label)
        return
    for item in value:
        assert_finite_shape(test, item, shape[1:], label)


class Qwen4ExpReferenceSpec(unittest.TestCase):
    def test_reference_pins_are_full_and_fixture_names_are_unique(self):
        self.assertEqual(len(reference.MODEL_REVISION), 40)
        self.assertEqual(len(reference.TRANSFORMERS_COMMIT), 40)
        self.assertEqual(reference.TRANSFORMERS_VERSION, "5.16.1")
        self.assertEqual(len({reference.OUTPUT_NAME, reference.CONFIG_NAME, reference.WEIGHTS_NAME}), 3)

    def test_environment_refuses_wrong_dependency_and_source_pins(self):
        args = (reference.TORCH_VERSION, reference.TRANSFORMERS_VERSION,
                reference.NUMPY_VERSION, reference.SAFETENSORS_VERSION,
                reference.SOURCE_SHA256)
        with self.assertRaisesRegex(RuntimeError, "requires Torch"):
            reference._check_environment("0.0", *args[1:])
        changed = dict(reference.SOURCE_SHA256)
        changed["modeling_qwen4_exp.py"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "source SHA-256"):
            reference._check_environment(*args[:4], changed)
        with self.assertRaisesRegex(RuntimeError, "optional accelerated"):
            reference._check_environment(*args, ("fla",))

    def test_incomplete_generation_does_not_replace_existing_artifacts(self):
        with tempfile.TemporaryDirectory(prefix="llmx_qwen4exp_stage_") as directory:
            root = Path(directory)
            stage = root / "stage"
            output = root / "output"
            stage.mkdir()
            output.mkdir()
            for name in (reference.CONFIG_NAME, reference.WEIGHTS_NAME, reference.OUTPUT_NAME):
                (output / name).write_text("old:" + name, encoding="utf-8")
            (stage / reference.CONFIG_NAME).write_text("new config", encoding="utf-8")
            (stage / reference.WEIGHTS_NAME).write_text("new weights", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "staging is incomplete"):
                reference._publish_staged(stage, output)
            for name in (reference.CONFIG_NAME, reference.WEIGHTS_NAME, reference.OUTPUT_NAME):
                self.assertEqual((output / name).read_text(encoding="utf-8"), "old:" + name)

    def test_nonfinite_decode_logits_are_refused(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, "non-finite decode"):
                reference._require_finite([[0.0, value]], "decode logit")

    def test_config_exercises_every_requested_layer_family(self):
        config = reference.TINY_CONFIG
        self.assertEqual(config["layer_types"], ["linear_attention", "qwen_sparse_attention"])
        self.assertEqual(config["ple_layer_ids"], [1])
        self.assertEqual(config["hc_count"], 4)
        self.assertEqual(config["rope_parameters"]["partial_rotary_factor"], 0.5)
        self.assertEqual(config["rope_parameters"]["mrope_section"], [1, 1, 0])
        self.assertEqual(config["indexer_n_heads"], 4)
        self.assertGreater(config["num_experts"], config["num_experts_per_tok"])
        self.assertGreater(config["num_experts_per_tok"], 1)
        self.assertGreater(config["ngram_size"], 1)
        self.assertEqual(config["ple_embed_dim"] % ((config["ngram_size"] - 1) * config["heads_per_ngram"]), 0)

    def test_qsa_candidate_selection_truncates_and_keeps_partial_tail(self):
        counts = reference.selection_counts()
        self.assertEqual(counts[-1], (5, 13))
        self.assertLess(counts[-1][0], counts[-1][1])
        self.assertEqual(counts[-1][0] % reference.TINY_CONFIG["indexer_compress_ratio"], 1)
        self.assertEqual(reference.TINY_CONFIG["indexer_budget"] % reference.TINY_CONFIG["indexer_compress_ratio"], 0)

    def test_reference_cases_cover_eos_interior_and_terminal_positions(self):
        eos_cases = [
            [i for i, token in enumerate(case["input_ids"]) if token == reference.EOS]
            for case in reference.CASES
        ]
        self.assertIn([4], eos_cases)
        self.assertIn([12], eos_cases)
        self.assertEqual(sum(reference.SLICE_LENGTHS), len(reference.CASES[0]["input_ids"]))
        self.assertEqual(len({case["name"] for case in reference.CASES}), len(reference.CASES))

    def test_slice_partition_and_decode_cover_prefill_and_single_token_steps(self):
        self.assertGreater(reference.SLICE_LENGTHS[0], 1)
        self.assertTrue(all(length > 0 for length in reference.SLICE_LENGTHS))
        self.assertEqual(reference.SLICE_LENGTHS, (3, 2, 4, 4))
        self.assertTrue(all(case["decode_token"] is not None for case in reference.CASES[:2]))

    def test_persistence_reference_crosses_cache_blocks_without_eos(self):
        document = json.loads(ARTIFACT.read_text(encoding="utf-8"))
        case = document["persistence_case"]
        self.assertEqual(case["config_overrides"], {"max_position_embeddings": 512})
        self.assertEqual(document["config"], reference.TINY_CONFIG)
        self.assertEqual(document["config"]["max_position_embeddings"], 32)
        self.assertEqual(case["prefix_tokens"], 257)
        self.assertEqual(len(case["input_ids"]), 269)
        self.assertTrue(all(type(token) is int and 0 <= token < reference.EOS for token in case["input_ids"]))
        self.assertGreater(len(set(case["input_ids"])), 20)
        for key in ("prefix_logits", "full_logits"):
            assert_finite_shape(self, case[key], (document["config"]["vocab_size"],), key)
        self.assertTrue(math.isfinite(case["max_abs_prefix_vs_full_prefix"]))
        self.assertLessEqual(case["max_abs_prefix_vs_full_prefix"], 2e-5)

    def test_checked_in_qsa_ple_and_hyper_connection_captures(self):
        document = json.loads(ARTIFACT.read_text(encoding="utf-8"))
        config = document["config"]
        seq_len = len(document["cases"][0]["input_ids"])
        heads = config["indexer_n_heads"]
        key_dim = config["indexer_head_dim"]
        hidden_stream_width = config["hidden_size"] * config["hc_count"]
        row_width = config["ple_embed_dim"] // (
            (config["ngram_size"] - 1) * config["heads_per_ngram"])

        self.assertEqual(len(document["cases"]), len(reference.CASES))
        for case, spec in zip(document["cases"], reference.CASES):
            with self.subTest(case=case["name"]):
                self.assertEqual(case["name"], spec["name"])
                self.assertEqual(case["input_ids"], spec["input_ids"])
                qsa = case["qsa_indexer"]
                for phase in ("full", "sliced", "cached"):
                    capture = qsa[phase]
                    assert_finite_shape(self, capture["query_projection_pre_norm_rope"],
                                        (seq_len, heads, key_dim), phase + " query projection")
                    assert_finite_shape(self, capture["raw_index_keys_pre_pool_norm_rope"],
                                        (seq_len, key_dim), phase + " raw index keys")
                    memberships = capture["selected_token_ids"]
                    self.assertEqual(len(memberships), seq_len)
                    for query, selected in enumerate(memberships):
                        expected_count, visible_count = reference.selection_counts(sequence_length=seq_len)[query]
                        self.assertEqual(len(selected), expected_count)
                        self.assertEqual(selected, sorted(set(selected)))
                        self.assertTrue(all(isinstance(token, int) and 0 <= token < visible_count
                                            and token <= query for token in selected))
                full_ids = qsa["full"]["selected_token_ids"]
                self.assertEqual(qsa["sliced"]["selected_token_ids"], full_ids)
                self.assertEqual(qsa["cached"]["selected_token_ids"], full_ids)
                decode = qsa["decode"]
                if case["decode_token"] is None:
                    self.assertIsNone(decode)
                else:
                    assert_finite_shape(self, decode["query_projection_pre_norm_rope"],
                                        (1, heads, key_dim), "decode query projection")
                    assert_finite_shape(self, decode["raw_index_keys_pre_pool_norm_rope"],
                                        (1, key_dim), "decode raw index keys")
                    self.assertEqual(len(decode["selected_token_ids"]), 1)
                    self.assertTrue(all(0 <= token < seq_len + 1 for token in decode["selected_token_ids"][0]))
                expected_legacy = [qsa["cached"]["selected_token_ids"][i:i + 1] for i in range(seq_len)]
                if decode is not None:
                    expected_legacy.append(decode["selected_token_ids"])
                self.assertEqual(case["indexer_selected_ids"], expected_legacy)

                ple = case["ple_intermediates"]
                self.assertEqual(ple["layer"], 0)
                assert_finite_shape(self, ple["input"], (1, seq_len, hidden_stream_width), "PLE input")
                assert_finite_shape(self, ple["ngram_embeddings"],
                                    (1, seq_len, (config["ngram_size"] - 1) * config["heads_per_ngram"], row_width),
                                    "PLE embeddings")
                assert_finite_shape(self, ple["output"], (1, seq_len, hidden_stream_width), "PLE output")
                hc = case["attention_hyper_connection"]
                self.assertEqual(hc["layer"], 1)
                assert_finite_shape(self, hc["mixed_input"], (1, seq_len, config["hidden_size"]), "HC mixed input")
                assert_finite_shape(self, hc["original_streams"], (1, seq_len, hidden_stream_width), "HC streams")
                assert_finite_shape(self, hc["injection_gates"], (1, seq_len, config["hc_count"]), "HC gates")


if __name__ == "__main__":
    unittest.main()
