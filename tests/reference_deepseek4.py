"""Torch-free checks of the tiny deepseek41 reference: its tool's pins and refusals, the committed golden against the generator, both tiny files and the element decoders."""

import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import tempfile
import types
import unittest
from unittest.mock import patch

import fixture_deepseek4 as fixture
import spec_decode as sd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("llmx_deepseek4_reference", ROOT / "tools" / "deepseek4_reference.py")
reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reference)
GOLDEN = Path(__file__).resolve().parent / "data" / reference.OUTPUT_NAME
# Set by run(): numpy's absence fails the numpy read-back instead of skipping it.
REQUIRE = {"numpy": False}


def port_names():
    """NAMES of tools/deepseek4_kernel.py, read without importing torch."""
    tree = ast.parse((ROOT / "tools" / "deepseek4_kernel.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "NAMES" for t in node.targets):
            return set(ast.literal_eval(node.value))
    raise AssertionError("the kernel port has no NAMES")


def golden():
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


class Decoders(unittest.TestCase):
    def test_e4m3_hand_values(self):
        for byte, value in ((0x00, 0.0), (0x01, 2.0 ** -9), (0x07, 7 * 2.0 ** -9), (0x08, 2.0 ** -6), (0x30, 0.5),
                            (0x38, 1.0), (0x39, 1.125), (0x40, 2.0), (0x7E, 448.0), (0xB8, -1.0), (0xFE, -448.0)):
            self.assertEqual(sd.e4m3(byte), value, hex(byte))
        self.assertEqual(math.copysign(1.0, sd.e4m3(0x80)), -1.0)
        for byte in (0x7F, 0xFF):
            self.assertTrue(math.isnan(sd.e4m3(byte)))

    def test_e4m3_every_code(self):
        positive = [sd.e4m3(byte) for byte in range(0x7F)]
        self.assertTrue(all(a < b for a, b in zip(positive, positive[1:])))
        for byte in range(0x7F):
            self.assertEqual(sd.e4m3(byte | 0x80), -positive[byte])
        self.assertEqual(sum(math.isfinite(sd.e4m3(byte)) for byte in range(256)), 254)

    def test_e8m0_every_code(self):
        self.assertEqual(sd.e8m0(0), 2.0 ** -127)
        self.assertEqual(sd.e8m0(127), 1.0)
        self.assertEqual(sd.e8m0(254), 2.0 ** 127)
        self.assertTrue(math.isnan(sd.e8m0(255)))
        for byte in range(255):
            self.assertEqual(sd.e8m0(byte), math.ldexp(1.0, byte - 127))

    def test_e2m1_every_code(self):
        self.assertEqual([sd.e2m1(c) for c in range(8)], [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
        self.assertEqual([sd.e2m1(c) for c in range(9, 16)], [-0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
        self.assertEqual(math.copysign(1.0, sd.e2m1(8)), -1.0)

    def test_fp4_e4m3_16_block(self):
        block = bytes([0x30]) + bytes([0x21, 0xF7, 0x80, 0x00, 0x00, 0x00, 0x00, 0x9A])
        values = sd.fp4_e4m3_16(block)
        self.assertEqual(values[:6], [0.25, 0.5, 3.0, -3.0, 0.0, -0.0])
        self.assertEqual(values[-2:], [-0.5, -0.25])
        self.assertEqual(len(values), 16)


class Tool(unittest.TestCase):
    def test_pins_are_full(self):
        self.assertRegex(reference.REVISION, r"^[0-9a-f]{40}$")
        self.assertEqual(len(reference.SOURCE), 13)
        for relative, (oid, sha256) in reference.SOURCE.items():
            self.assertRegex(oid, r"^[0-9a-f]{40}$", relative)
            self.assertRegex(sha256, r"^[0-9a-f]{64}$", relative)
        self.assertEqual(reference.git_blob_oid(b""), "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391")

    def test_requirements_pin_the_reference_versions(self):
        pins = reference.read_requirements()
        self.assertEqual(len(pins), 26)
        for name, version in (("torch", "2.11.0+cpu"), ("numpy", "2.2.6"), ("safetensors", "0.8.0"),
                              ("tokenizers", "0.23.2"), ("sympy", "1.14.0"), ("pillow", "13.0.0")):
            self.assertEqual(pins[name], version)
        self.assertFalse(set(pins) & set(reference.FORBIDDEN_MODULES))

    def test_environment_refusals(self):
        pins = reference.read_requirements()
        reference.check_environment(reference.PYTHON_VERSION, dict(pins, pip="25.0"), pins)
        with self.assertRaisesRegex(RuntimeError, "requires Python"):
            reference.check_environment("3.12.12", pins, pins)
        with self.assertRaisesRegex(RuntimeError, "requires torch 2.11.0"):
            reference.check_environment(reference.PYTHON_VERSION, dict(pins, torch="2.12.0+cpu"), pins)
        with self.assertRaisesRegex(RuntimeError, "requires sympy"):
            reference.check_environment(reference.PYTHON_VERSION, {k: v for k, v in pins.items() if k != "sympy"}, pins)
        for extra in ("tilelang", "transformers", "deep-gemm"):
            with self.assertRaisesRegex(RuntimeError, "unpinned distributions: " + extra):
                reference.check_environment(reference.PYTHON_VERSION, dict(pins, **{extra: "1.0"}), pins)

    def test_source_refusals(self):
        with tempfile.TemporaryDirectory(prefix="llmx_deepseek4_source_") as directory:
            root = Path(directory)
            files = {"inference/model.py": b"from kernel import act_quant\n", "config.json": b"{}\n"}
            pins = {}
            for relative, data in files.items():
                (root / relative).parent.mkdir(parents=True, exist_ok=True)
                (root / relative).write_bytes(data)
                pins[relative] = (reference.git_blob_oid(data), hashlib.sha256(data).hexdigest())
            with patch.object(reference, "SOURCE", pins):
                self.assertEqual(reference.check_sources(root), files)
                (root / "config.json").write_bytes(b"{} \n")
                with self.assertRaisesRegex(RuntimeError, "not the pinned file"):
                    reference.check_sources(root)
                (root / "config.json").unlink()
                with self.assertRaisesRegex(RuntimeError, "lacks config.json"):
                    reference.check_sources(root)

    def test_kernel_imports_and_module_audit(self):
        source = "import torch\nfrom kernel import (\n    act_quant,\n    sparse_attn,\n)\n"
        self.assertEqual(reference.kernel_imports(source), {"act_quant", "sparse_attn"})
        with self.assertRaisesRegex(RuntimeError, "imports kernel as a module"):
            reference.kernel_imports("import kernel\n")
        self.assertEqual(port_names(), {"act_quant", "fp4_act_quant", "fp4_gemm", "fp8_gemm", "hc_split_sinkhorn", "sparse_attn"})
        self.assertLessEqual(set(golden()["kernel_imports"]), port_names())
        reference.check_imports({"torch": None, "model": None}, False)
        for name in reference.FORBIDDEN_MODULES:
            with self.assertRaisesRegex(RuntimeError, name):
                reference.check_imports({"torch": None, name: None}, False)
        with self.assertRaisesRegex(RuntimeError, "process group"):
            reference.check_imports({}, True)

    def test_incomplete_staging_replaces_nothing(self):
        with tempfile.TemporaryDirectory(prefix="llmx_deepseek4_stage_") as directory:
            stage, output = Path(directory) / "stage", Path(directory) / "out"
            stage.mkdir()
            output.mkdir()
            names = (reference.OUTPUT_NAME,) + tuple(reference.GGUF_NAMES.values())
            for name in names:
                (output / name).write_text("old", encoding="utf-8")
            (stage / reference.OUTPUT_NAME).write_text("new", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "staging is incomplete"):
                reference.publish_staged(stage, output)
            for name in names:
                self.assertEqual((output / name).read_text(encoding="utf-8"), "old")

    def test_margin_floor_refusal(self):
        reference.check_margins({"router": [1.0, reference.MARGIN_FLOOR]})
        with self.assertRaisesRegex(RuntimeError, "router selection margin"):
            reference.check_margins({"router": [1.0, reference.MARGIN_FLOOR / 2]})

    def test_gen_baseline_delegates(self):
        import gen_baseline
        seen = {}
        with patch.dict("sys.modules", deepseek4_reference=types.SimpleNamespace(generate=lambda out, src: seen.update(out=out, src=src))):
            with tempfile.TemporaryDirectory(prefix="llmx_deepseek4_args_") as directory:
                gen_baseline.main(["deepseek41-tiny", "--output-dir", directory, "--checkpoint", directory])
                self.assertEqual(seen, {"out": os.path.abspath(directory), "src": directory})
                for argv in (["deepseek41-tiny"], ["deepseek41-tiny", "--checkpoint", os.path.join(directory, "absent")],
                             ["deepseek41-tiny", "--checkpoint", directory, "--threads", "2"],
                             ["deepseek41-tiny", "--checkpoint", directory, "--revision", "c" * 40]):
                    with self.assertRaises(SystemExit), patch("sys.stderr"):
                        gen_baseline.parse_args(argv)


class Golden(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = golden()

    def test_golden_is_the_generators(self):
        doc = self.doc
        self.assertEqual((doc["repo"], doc["revision"]), (reference.REPO, reference.REVISION))
        self.assertEqual(doc["source"], {k: {"oid": o, "sha256": s} for k, (o, s) in reference.SOURCE.items()})
        self.assertEqual(doc["python_version"], reference.PYTHON_VERSION)
        self.assertEqual(doc["requirements"], reference.read_requirements())
        self.assertEqual(doc["requirements_sha256"], reference.text_sha256(reference.REQUIREMENTS))
        self.assertEqual(doc["port_sha256"], reference.text_sha256(ROOT / "tools" / "deepseek4_kernel.py"))
        self.assertEqual((doc["seed"], doc["sequence_seed"]), (reference.SEED, reference.SEQUENCE_SEED))
        self.assertEqual(doc["args"], fixture.TINY_ARGS)
        self.assertEqual(doc["sequence"], reference.sequence())
        self.assertEqual((doc["margin_floor"], doc["equal_bound"]), (reference.MARGIN_FLOOR, reference.EQUAL_BOUND))
        self.assertEqual([(r["kind"], r["length"]) for r in doc["runs"]],
                         [("whole", n) for n in reference.WHOLE_LENGTHS] + [("prefill_then_decode", n) for n in reference.DECODE_FROM])
        self.assertTrue(all(r["max_abs_diff"] <= reference.EQUAL_BOUND for r in doc["runs"]))
        self.assertTrue(all(r["outputs_max_abs_diff"] <= reference.EQUAL_BOUND for r in doc["runs"]))
        self.assertEqual(len(doc["logits"]), reference.SEQUENCE_LENGTH)
        self.assertTrue(all(len(row) == fixture.TINY_ARGS["vocab_size"] and all(math.isfinite(v) for v in row) for row in doc["logits"]))

    def test_selections_clear_the_floor_and_hold_together(self):
        doc, args = self.doc, fixture.TINY_ARGS
        for kind, margin in doc["margins"].items():
            self.assertGreaterEqual(margin["minimum"], reference.MARGIN_FLOOR, kind)
            self.assertGreater(margin["count"], 0, kind)
        self.assertEqual(sorted(doc["selections"]["index"]), [str(n) for n in args["index_source_layers"]])
        blocks = doc["selections"]["candidate"][str(args["candidate_source_layer"])]
        size = args["candidate_block_size"]
        for position, ids in enumerate(doc["selections"]["index"]["7"]):
            self.assertLessEqual(len(ids), args["index_topk"])
            self.assertTrue(all(i // size in blocks[position] for i in ids), position)
            self.assertTrue(all(i < (position + 1) // args["compress_ratios"][7] for i in ids))
        self.assertEqual(len(doc["selections"]["index"]["7"][-1]), args["index_topk"])
        for layer, rows in doc["selections"]["router"].items():
            self.assertTrue(all(len(row) == args["n_activated_experts"] for row in rows), layer)
        self.assertGreater(doc["swiglu_clamped"], 0)
        self.assertEqual(doc["activation_qdq_skipped"], args["n_layers"])
        for block, parts in doc["sinkhorn"].items():
            for part, positions in parts.items():
                self.assertEqual(sorted(positions, key=int), [str(p) for p in reference.SINKHORN_POSITIONS], (block, part))
        self.assertEqual(sorted(doc["outputs"], key=int), [str(n) for n in range(args["n_layers"])])
        for block, parts in doc["outputs"].items():
            self.assertEqual(sorted(parts), sorted(reference.SUBLAYERS), block)
            for part, positions in parts.items():
                self.assertEqual(sorted(positions, key=int), [str(p) for p in reference.SINKHORN_POSITIONS], (block, part))
                self.assertTrue(all(len(row) == args["dim"] and all(math.isfinite(v) for v in row) for row in positions.values()), (block, part))

    def test_engram_pins(self):
        engram, args = self.doc["engram"], fixture.TINY_ARGS
        self.assertEqual(engram["token_map_sha256"], reference.TOKEN_MAP_SHA256)
        self.assertEqual(len(engram["token_map"]), args["vocab_size"])
        self.assertEqual(engram["token_map_prefix_sha256"],
                         hashlib.sha256(struct.pack("<%di" % args["vocab_size"], *engram["token_map"])).hexdigest())
        differs = engram["mapped_differs"]
        self.assertNotEqual(differs["raw"], differs["mapped"])
        self.assertEqual(engram["token_map"][differs["raw"]], differs["mapped"])
        self.assertEqual(self.doc["sequence"][differs["position"]], differs["raw"])
        self.assertEqual(self.doc["sequence"][reference.PAD_POSITION], args["engram_pad_id"])
        self.assertEqual(engram["mapped_pad_id"], engram["token_map"][args["engram_pad_id"]])
        hashes = engram["hash_ids"]
        self.assertEqual(len(hashes), reference.SEQUENCE_LENGTH)
        cols = (args["engram_max_ngram_size"] - 1) * args["engram_n_heads"]
        for position in range(3):
            for layer, ids in enumerate(hashes[position]):
                self.assertEqual(len(ids), cols)
                offsets, primes = engram["offsets"][layer], engram["primes"][layer]
                for column, hash_id in enumerate(ids):
                    self.assertTrue(offsets[column] <= hash_id < offsets[column] + primes[column])
        self.assertEqual(engram["no_convolution"]["convolution"], False)
        layer, position = reference.ZERO_KEY
        self.assertEqual(engram["zero_rows"], {str(layer): sorted(hashes[position][args["engram_layer_ids"].index(layer)])})
        zero = engram["zero_key"]
        self.assertEqual((zero["layer"], zero["position"]), (layer, position))
        sigmoid = f32(1.0 / (1.0 + math.exp(-f32(math.sqrt(f32(1e-6))))))
        self.assertEqual(f32(zero["gate"]), sigmoid)
        self.assertNotEqual(zero["gate"], 0.5)
        self.assertEqual(engram["gates"][str(layer)][str(position)], [zero["gate"]] * args["hc_mult"])
        for engram_layer in args["engram_layer_ids"]:
            gates = engram["gates"][str(engram_layer)]
            self.assertEqual(sorted(gates, key=int), ["0", "1", "2", "3", str(position)])

    def test_tiny_files_are_the_state(self):
        doc = self.doc
        args, state, engram = fixture.fixture_from_document(doc)
        self.assertEqual(fixture.state_sha256(state), doc["state_sha256"])
        numpy = sd.np is not None
        if not numpy and REQUIRE["numpy"]:
            self.fail("numpy is required for the read-back of the tiny deepseek41 files")
        with tempfile.TemporaryDirectory(prefix="llmx_deepseek4_gguf_") as directory:
            for layout in fixture.LAYOUTS:
                path = fixture.write_deepseek41(Path(directory) / ("%s.gguf" % layout), args, state, engram, layout)
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), doc["files"][layout]["sha256"], layout)
                model = sd.GGUF(path)
                decoded = {t.name: model.decode(t, numpy=numpy) for t in model.tensors}
                decoded = {name: [float(v) for v in values] for name, values in decoded.items()}
                self._held_to_state(layout, model, decoded, args, state)

    def _held_to_state(self, layout, model, decoded, args, state):
        bits = lambda values: struct.pack("<%df" % len(values), *values)  # noqa: E731
        seen = set()
        for name, (shape, values) in state.items():
            target = fixture.gguf_name(name, layout)
            if target is not None:
                self.assertEqual(bits(decoded[target]), bits(values), (layout, name))
                seen.add(target)
        tables = []
        for layer in range(args["n_layers"]):
            p = "layers.%d." % layer
            for proj, target in (("w1", "ffn_gate_exps"), ("w2", "ffn_down_exps"), ("w3", "ffn_up_exps")):
                stacked = [v for e in range(args["n_routed_experts"]) for v in state[p + "ffn.experts.%d.%s.weight" % (e, proj)][1]]
                self.assertEqual(bits(decoded["blk.%d.%s.weight" % (layer, target)]), bits(stacked), (layout, layer, proj))
                seen.add("blk.%d.%s.weight" % (layer, target))
            if layer not in args["engram_layer_ids"]:
                continue
            values, scales = state[p + "engram.embed.weight"][1], state[p + "engram.embed.scale"][1]
            table = [v * scales[i // 32] for i, v in enumerate(values)]
            if layout == "fork":
                self.assertEqual(bits(decoded["blk.%d.engram_embed.weight" % layer]), bits(table), layer)
                seen.add("blk.%d.engram_embed.weight" % layer)
                self.assertIn("deepseek41.engram.token_map", model.metadata)
            else:
                product = [a * b for a, b in zip(state[p + "engram.q_weight"][1], state[p + "engram.k_weight"][1])]
                self.assertEqual(bits(decoded["blk.%d.engram_gate.weight" % layer]), bits(product), layer)
                seen.add("blk.%d.engram_gate.weight" % layer)
                tables.extend(table)
        if layout == "upstream":
            self.assertEqual(bits(decoded["per_layer_token_embd.weight"]), bits(tables))
            seen.add("per_layer_token_embd.weight")
            self.assertFalse(any(key.startswith("deepseek41.engram.") and key not in (
                "deepseek41.engram.pad_token_id", "deepseek41.engram.token_map") for key in model.metadata))
            self.assertIn("deepseek41.ple.head_offsets", model.metadata)
        else:
            self.assertFalse(any(key.startswith("deepseek41.ple.") for key in model.metadata))
        self.assertEqual(seen, set(decoded), layout)


def run(require=False):
    REQUIRE["numpy"] = require
    suite = unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    return unittest.TextTestRunner().run(suite).wasSuccessful()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    sys.exit(0 if run() else 1)
