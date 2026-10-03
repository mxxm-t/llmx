"""Focused hand check of the Qwen3-MoE layered reference in its pinned HF environment.

python tests/layered_moe_reference.py

Creates only tiny temporary GGUFs/configs; reads but never regenerates accepted goldens.
"""

import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import gen_baseline as generator
import gen_layered_reference as layered
import moe
import common
import spec_decode


class LayeredMoe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch, cls.transformers = generator.qwen3moe_environment()
        cls.torch.set_num_threads(1)
        print("torch %s; transformers %s; CPU, float32, eager, one thread"
              % (cls.torch.__version__, cls.transformers.__version__), flush=True)

    def test_mixed_file_and_existing_q8_goldens(self):
        torch = self.torch
        c = moe.Q8_CONFIG
        config = self.transformers.Qwen3MoeConfig(
            vocab_size=257, hidden_size=c["embedding_length"], intermediate_size=c["feed_forward_length"],
            moe_intermediate_size=c["expert_feed_forward_length"], num_hidden_layers=c["block_count"],
            num_attention_heads=c["attention.head_count"], num_key_value_heads=c["attention.head_count_kv"],
            head_dim=c["attention.key_length"], max_position_embeddings=c["context_length"], rope_theta=10000.0,
            rms_norm_eps=1e-6, tie_word_embeddings=False, attention_dropout=0.0,
            num_experts=c["expert_count"], num_experts_per_tok=c["expert_used_count"],
            norm_topk_prob=True, decoder_sparse_step=1, mlp_only_layers=list(moe.DENSE_LAYERS))
        config._attn_implementation = "eager"
        golden = json.loads((Path(__file__).parent / "data/baseline_moe_q8.json").read_text())
        checked = 0
        golden_errors = {}
        with tempfile.TemporaryDirectory(prefix="llmx_layered_moe_") as directory:
            config.save_pretrained(directory)
            for variant, scale in moe.Q8_VARIANTS:
                path = str(Path(directory) / (variant + ".gguf"))
                self.assertEqual(moe.write_q8_model(path, scale), moe.Q8_FILES[variant])
                full = self.transformers.Qwen3MoeForCausalLM(config).float().eval().requires_grad_(False)
                full.load_state_dict(generator.moe_q8_state(path, torch), strict=True)
                # The synthetic writer adds an intentionally unused tensor.
                # The file-exact loader must refuse it, not silently ignore a weight.
                with self.assertRaisesRegex(ValueError, "unused.weight"):
                    layered.Layered(torch, directory, gguf=path)
                original = spec_decode.GGUF(path)
                path = str(Path(directory) / (variant + "-parameters.gguf"))
                spec_decode.write_gguf(path, original.metadata,
                                      [(t.name, t.shape, t.type, original.raw(t)) for t in original.tensors if t.name != "unused.weight"])
                reference = layered.Layered(torch, directory, gguf=path)
                cases = next(v["cases"] for v in golden["variants"] if v["name"] == variant)
                ids = [torch.tensor([list(case["prompt"].encode("ascii")) + case["forced_ids"]]) for case in cases]

                def live_layer(name, module):
                    live = [n for n, p in reference.model.named_parameters() if not p.is_meta]
                    self.assertTrue(live)
                    self.assertTrue(all(n.startswith(name + ".") for n in live), live)
                    self.assertTrue(all(not b.is_meta for b in reference.rotary.buffers()))

                reference.run(ids, on_layer=live_layer)
                self.assertTrue(all(p.is_meta for p in reference.model.parameters()))
                with torch.inference_mode():
                    for row, case in zip(ids, cases):
                        expected = full(row, use_cache=False).logits
                        got = reference(row).logits
                        self.assertEqual(expected.numpy().tobytes(), got.numpy().tobytes())
                        rows = expected[0, len(case["prompt"]) - 1:]
                        error = max(abs(v - ref) for r, refrow in zip(rows.tolist(), case["rows"]) for v, ref in zip(r, refrow))
                        golden_errors[variant] = max(golden_errors.get(variant, 0), error)
                        # The existing consumer bounds the gated variant only; near-tie routing remains diagnostic across HF builds.
                        if variant == "gated":
                            self.assertLessEqual(error, common.F32_HF_LOGIT_BOUND)
                        checked += expected.numel()
                # Every repeat reloads weights but must not inflate metadata counts.
                counts = dict(reference.file.types)
                reference.logits.clear()
                reference.run(ids[:1])
                self.assertEqual(reference.file.types, counts)
                self.assertEqual(sum(counts.values()), len(dict(full.named_parameters())))

            # A real mixed storage file uses the same exact decoder/mapping owner.
            original = spec_decode.GGUF(path)
            written = []
            for t in original.tensors:
                kind, raw = t.type, original.raw(t)
                if t.name == "blk.0.ffn_up_exps.weight":
                    values = original.decode(t, numpy=False)
                    raw = b"".join(struct.pack("<f", v)[2:] for v in values)
                    kind = spec_decode.BF16
                written.append((t.name, t.shape, kind, raw))
            mixed = str(Path(directory) / "mixed.gguf")
            spec_decode.write_gguf(mixed, original.metadata, written)
            full.load_state_dict(generator.moe_q8_state(mixed, torch), strict=True)
            reference = layered.Layered(torch, directory, gguf=mixed)

            def fail(name, module):
                if name == "model.layers.1":
                    raise RuntimeError("injected reference failure")

            with self.assertRaisesRegex(RuntimeError, "injected reference failure"):
                reference.run(ids[:1], on_layer=fail)
            self.assertFalse(reference.logits)
            self.assertTrue(all(p.is_meta for p in reference.model.parameters()))
            self.assertTrue(all(not m._forward_pre_hooks and not m._forward_hooks for m in reference.model.modules()))
            reference.run(ids)
            with torch.inference_mode():
                for row in ids:
                    expected = full(row, use_cache=False).logits
                    self.assertEqual(expected.numpy().tobytes(), reference(row).logits.numpy().tobytes())
                    checked += expected.numel()
            self.assertTrue(all(p.is_meta for p in reference.model.parameters()))
            # The same owner also reads a pinned checkpoint for its equality command; no Hub access is needed for this tiny local checkpoint.
            full.save_pretrained(directory)
            checkpoint = layered.Layered(torch, directory)
            checkpoint.run(ids[:1])
            self.assertEqual(reference(ids[0]).logits.numpy().tobytes(), checkpoint(ids[0]).logits.numpy().tobytes())
            checked += checkpoint(ids[0]).logits.numel()
            self.assertTrue(all(p.is_meta for p in checkpoint.model.parameters()))
        print("layered/full exact: %d float32 logits; both original Q8 golden variants unchanged; mixed BF16/Q8/F32 and failure recovery passed"
              % checked, flush=True)
        print("existing golden max errors %s; gated bound %g, near-tie diagnostic only"
              % (golden_errors, common.F32_HF_LOGIT_BOUND), flush=True)


if __name__ == "__main__":
    unittest.main()
