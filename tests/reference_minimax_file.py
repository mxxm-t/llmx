"""Checks of the file-exact MiniMax-M3.1 HF reference (tools/minimax_file_reference.py).

Rows parses `llmx logits` output without torch; PinnedHF, which needs the minimax reference environment (docs/ASSETS.md), holds the adapter on each tiny fixture's file to HF's own model with eager experts, bit for bit.
"""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import fixture_minimax  # noqa: E402
import spec_decode  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "data" / "minimax_m3_reference.json"


class Rows(unittest.TestCase):
    def test_rows_take_every_id_and_refuse_a_partial_row(self):
        if spec_decode.np is None:
            self.skipTest("numpy required by the comparison")
        import minimax_file_reference as reference
        with tempfile.TemporaryDirectory(prefix="minimax_rows_") as directory:
            path = Path(directory) / "logits.txt"
            path.write_text("tokens: 5\n3 2 1.5 0 -0.25 1 0.125\n4 0 2.0 1 1.0 2 0.5\n", encoding="utf-8")
            rows = reference.llmx_rows(path, 3)
            self.assertEqual(sorted(rows), [3, 4])
            self.assertEqual(rows[3].tolist(), [-0.25, 0.125, 1.5])
            path.write_text("tokens: 5\n3 2 1.5 0 -0.25\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "whole vocabulary"):
                reference.llmx_rows(path, 3)

    def test_routes_join_a_layers_records_and_refuse_a_cut_record(self):
        import struct
        import minimax_file_reference as reference
        with tempfile.TemporaryDirectory(prefix="minimax_routes_") as directory:
            path = Path(directory) / "row.routes"
            record = lambda layer, ids, scores: struct.pack("<iiii", layer, len(ids), 1, 2) + struct.pack("<%dI" % len(ids), *ids) + struct.pack("<%df" % len(scores), *scores)
            path.write_bytes(record(3, [1], [0.5, -0.5]) + record(4, [0], [1.0, 2.0]) + record(3, [0, 1], [0.25, 0.75, -1.0, 1.0]))
            routes = reference.read_routes(path)
            self.assertEqual(sorted(routes), [3, 4])
            self.assertEqual((list(routes[3][0]), list(routes[3][1]), routes[3][2:]), ([1, 0, 1], [0.5, -0.5, 0.25, 0.75, -1.0, 1.0], (1, 2)))
            path.write_bytes(record(3, [1], [0.5, -0.5])[:-4])
            with self.assertRaisesRegex(ValueError, "cut short"):
                reference.read_routes(path)


class PinnedHF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import minimax_reference
        try:
            cls.torch, _, _ = minimax_reference.pinned_environment()
        except (ImportError, RuntimeError) as error:
            raise unittest.SkipTest("the pinned minimax reference environment: %s" % error)
        cls.torch.set_num_threads(1)

    def test_fixture_files_give_hf_logits(self):
        import minimax_file_reference as reference
        import minimax_reference
        torch, modeling, configuration = minimax_reference.pinned_environment()
        golden = json.loads(GOLDEN.read_text(encoding="utf-8"))["fixtures"]
        for name, case in golden.items():
            with self.subTest(fixture=name), tempfile.TemporaryDirectory(prefix="minimax_file_") as directory:
                root = Path(directory)
                (root / "config.json").write_text(json.dumps(case["config"]), encoding="utf-8")
                path = fixture_minimax.write_minimax(root / "model.gguf", case["config"], fixture_minimax.make_state(case["config"], case["seed"]))
                runner = reference.FileReference(root, [path], 1 << 30, 2)
                ids = case["sequence"]
                logits, trace = runner.forward(self.torch.tensor([ids]), len(ids))
                runner.pool.shutdown()
                # The adapter decodes one expert at a time, so it runs HF's eager expert loop; the golden ran HF's default grouped products.
                model, _ = minimax_reference._build(torch, modeling, configuration, case["config"], case["seed"])
                model.config._experts_implementation = "eager"
                with torch.no_grad():
                    eager = model(torch.tensor([ids]), use_cache=False).logits[0]
                self.assertTrue(torch.equal(logits, eager))
                grouped = float((logits - torch.tensor(case["logits"])).abs().max())
                self.assertLessEqual(grouped, 1e-6)
                self.assertEqual(trace["first_sparse_position"], case["first_sparse_position"])
                self.assertEqual(len(trace["layers"]), case["config"]["num_hidden_layers"])
                print("%s: %d positions bit for bit with eager experts, %.3g from the grouped golden, first sparse position %d" % (name, len(ids), grouped, trace["first_sparse_position"]))

    def test_routes_are_followed_only_where_hf_cannot_order_the_experts(self):
        # Experts 0 and 1 of the last routed layer score equal in F32 at every position, at the cut behind experts 2 and 3; HF's top-k takes either.
        # Being the last, it leaves every other layer's picks HF's own, which the routes give them.
        from array import array
        import struct
        import minimax_file_reference as reference
        import minimax_reference
        torch, modeling, configuration = minimax_reference.pinned_environment()
        case = json.loads(GOLDEN.read_text(encoding="utf-8"))["fixtures"]["small"]
        config, ids = case["config"], case["sequence"]
        layer = len(config["mlp_layer_types"]) - 1 - config["mlp_layer_types"][::-1].index("sparse")
        prefix = "model.layers.%d.mlp.gate." % layer
        state = fixture_minimax.make_state(config, case["seed"])
        shape, weight = state[prefix + "weight"]
        width = shape[1]
        state[prefix + "weight"] = (shape, weight[:width] + weight[:width] + weight[2 * width:])
        bias = [0.0] * config["num_local_experts"]
        bias[0] = bias[1] = 50.0
        bias[2] = bias[3] = 100.0
        state[prefix + "e_score_correction_bias"] = ((len(bias),), bias)

        def eager(values):
            model = modeling.MiniMaxM3VLForCausalLM(configuration.MiniMaxM3VLTextConfig(**config)).float().eval()
            model.config._attn_implementation = model.config._experts_implementation = "eager"
            model.load_state_dict({name: torch.tensor(v, dtype=torch.float32).reshape(s) for name, (s, v) in values.items()}, strict=True)
            with torch.no_grad():
                return model(torch.tensor([ids]), use_cache=False).logits[0]

        def biased(values):
            out = dict(state)
            out[prefix + "e_score_correction_bias"] = ((len(values),), values)
            return out
        # Expert 1 a step below expert 0 leaves HF no tie, and the weights come from the unbiased scores, so these are the two picks of the tie.
        lower, upper = eager(biased(bias[:1] + [49.0] + bias[2:])), eager(biased([49.0] + bias[1:]))
        self.assertGreater(float((lower - upper).abs().max()), 1e-3)
        with tempfile.TemporaryDirectory(prefix="minimax_tie_") as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            path = fixture_minimax.write_minimax(root / "model.gguf", config, state)
            runner = reference.FileReference(root, [path], 1 << 30, 2)
            # HF's own picks and router logits, which llmx-route-probe writes as llmx's, a record a pass, then read back.
            seen = {}

            def keep(i):
                def hook(module, args, out):
                    seen[i] = (out[0].clone(), out[2].clone())
                return hook
            hooks = [runner.model.model.layers[i].mlp.gate.register_forward_hook(keep(i)) for i in runner.experts]
            own, _ = runner.forward(self.torch.tensor([ids]), len(ids))
            for hook in hooks:
                hook.remove()
            k = config["num_experts_per_tok"]

            def routes(picks):
                file = root / "routes"
                with open(file, "wb") as f:
                    for i, (logits, hf) in seen.items():
                        chosen = picks(i, hf)
                        for begin in range(0, len(ids), 5):
                            rows = min(5, len(ids) - begin)
                            f.write(struct.pack("<iiii", i, rows, k, logits.shape[1]))
                            f.write(array("I", chosen[begin:begin + rows].flatten().tolist()).tobytes())
                            f.write(array("f", logits[begin:begin + rows].flatten().tolist()).tobytes())
                return reference.read_routes(file)

            def tied(expert):
                return lambda i, hf: torch.tensor([[2, 3, expert]] * len(ids)) if i == layer else hf
            followed = []
            for expert, expected in ((0, lower), (1, upper)):
                logits, trace = runner.forward(self.torch.tensor([ids]), len(ids), routes=routes(tied(expert)))
                self.assertLessEqual(float((logits - expected).abs().max()), 1e-6)
                got = trace["routes"]
                self.assertEqual((got["beyond"], got["exact_ties"]), (0, got["followed"]))
                self.assertTrue(all(p["layer"] == layer and p["llmx_only"] == [expert] for p in got["picks"]))
                followed.append(got["followed"])
            # HF took one of the two at every position, so the two followings together cover each position once.
            self.assertEqual(sum(followed), len(ids))
            # A pick outside the tie is not HF's to give up: it stays HF's own and is counted beyond the margin.
            logits, trace = runner.forward(self.torch.tensor([ids]), len(ids), routes=routes(tied(5)))
            runner.pool.shutdown()
            self.assertTrue(torch.equal(logits, own))
            self.assertEqual((trace["routes"]["beyond"], trace["routes"]["followed"]), (len(ids), 0))
        print("router tie: llmx's picks taken at %d and %d positions, %d picks beyond the margin kept HF's" % (followed[0], followed[1], len(ids)))

if __name__ == "__main__":
    unittest.main()
