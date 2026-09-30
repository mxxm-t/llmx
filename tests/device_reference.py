import math
import array
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch
import sys
import unittest

import common
import f32

sys.path.insert(0, str(Path(common.ROOT) / "tools"))
import check_device


class DeviceReference(unittest.TestCase):
    row = [8.0, 6.0, 5.0, 4.0, 3.0, 1.0, -1.0, -3.0]
    ids = [0, 1, 2, 3]

    def rows(self):
        return [self.row[:] for _ in self.ids]

    def test_complete_rows_and_offset(self):
        cpu = self.rows()
        device = [[x + 0.004 for x in row] for row in cpu]
        result = common.check_device_rows(iter(cpu), iter(device), self.ids, 0.005)
        self.assertEqual((result["rows"], result["scored"], result["top1_checked"]), (4, 3, 4))
        self.assertAlmostEqual(result["max_logit_gap"], 0.004)
        self.assertLess(result["nll_delta"], 1e-12)
        with self.assertRaisesRegex(ValueError, "logit gap"):
            common.check_device_rows(cpu, device, self.ids, 0.003)

    def test_rank_damage_and_near_ties(self):
        for field, replacement, reason in [(0, 5.9, "top-1"), (2, -2.0, "top-5")]:
            device = self.rows()
            device[1][field] = replacement
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                common.check_device_rows(self.rows(), device, self.ids)
        cpu = [[8.0, 7.98, 5.0, 4.0, 3.0, 2.95, -1.0, -3.0] for _ in self.ids]
        device = [[7.99, 8.0, 5.0, 4.0, 2.97, 3.0, -1.0, -3.0] for _ in self.ids]
        got = common.check_device_rows(cpu, device, self.ids, 0.1)
        self.assertEqual(got["top1_checked"], 0)
        self.assertEqual(got["min_top5_overlap"], 5)

    def test_control_measures_without_candidate_acceptance(self):
        cpu, device = self.rows(), self.rows()
        device[0][0] = 0.0
        measured = common.check_device_rows(cpu, device, self.ids, calibration=True)
        self.assertEqual(measured["max_logit_gap"], 8.0)
        self.assertEqual(measured["top1_fail_positions"], [0])
        self.assertFalse(measured["criteria_passed"])
        with self.assertRaisesRegex(ValueError, "top-1"):
            common.check_device_rows(cpu, device, self.ids, measured["max_logit_gap"])
        device[0][0] = math.nan
        with self.assertRaises(ValueError):
            common.check_device_rows(cpu, device, self.ids, calibration=True)
        cpu = [self.row[:] for _ in range(64)]
        device = [self.row[:] for _ in range(64)]
        device[8][1] = 9.0
        ids = [0] * 8 + [1] + [0] * 55
        measured = common.check_device_greedy(cpu, device, [0] * 64, ids, calibration=True)
        self.assertFalse(measured["agrees_until_tie"])
        self.assertEqual(measured["first_divergence"], 8)
        with self.assertRaisesRegex(ValueError, "greedy"):
            common.check_device_greedy(cpu, device, [0] * 64, ids)

    def test_mean_nll_damage(self):
        device = self.rows()
        for row in device:
            row[0] += 1.0
        with self.assertRaisesRegex(ValueError, "NLL"):
            common.check_device_rows(self.rows(), device, self.ids, 2.0)

    def test_malformed_rows_and_limits(self):
        for cpu, device, ids in [([], [], []), ([self.row], [self.row], [0]),
                                 (self.rows()[:-1], self.rows(), self.ids),
                                 (self.rows(), self.rows() + [self.row], self.ids),
                                 (self.rows(), [self.row[:-1]] * 4, self.ids),
                                 (self.rows(), self.rows(), [0, -1, 2, 3]),
                                 (self.rows(), self.rows(), [0, True, 2, 3]),
                                 (self.rows(), self.rows(), [0, 8, 2, 3])]:
            with self.subTest(ids=ids, rows=len(cpu)), self.assertRaises(ValueError):
                common.check_device_rows(cpu, device, ids)
        for value in (math.nan, math.inf, -math.inf):
            for side in (0, 1):
                rows = [self.rows(), self.rows()]
                rows[side][2][7] = value
                with self.assertRaises(ValueError):
                    common.check_device_rows(*rows, self.ids)
        for limit in (-1.0, math.nan, math.inf):
            with self.assertRaises(ValueError):
                common.check_device_rows(self.rows(), self.rows(), self.ids, limit)

    def test_greedy_prefix_before_first_near_tie(self):
        cpu = [self.row[:] for _ in range(64)]
        device = [self.row[:] for _ in range(64)]
        ids = [0] * 64
        self.assertEqual(common.check_device_greedy(cpu, device, ids, ids)["matched_prefix"], 64)
        cpu[7][1] = 7.95
        device[7] = [7.96, 8.0, 5.0, 4.0, 3.0, 1.0, -1.0, -3.0]
        for row in device[8:]:
            row[2] = 9.0
        got = common.check_device_greedy(cpu, device, ids, [0] * 7 + [1] + [2] * 56)
        self.assertEqual((got["matched_prefix"], got["first_near_tie"], got["first_divergence"]), (7, 7, 7))

    def test_greedy_damage_is_refused(self):
        cpu = [self.row[:] for _ in range(64)]
        device = [self.row[:] for _ in range(64)]
        device[8][1] = 9.0
        with self.assertRaisesRegex(ValueError, "greedy"):
            common.check_device_greedy(cpu, device, [0] * 64, [0] * 8 + [1] + [0] * 55)
        for rows, ids in [(device[:-1], [0] * 63), (device + [self.row], [0] * 65),
                          (cpu, [1] * 64), (cpu, [True] * 64)]:
            with self.assertRaises(ValueError):
                common.check_device_greedy(cpu, rows, [0] * 64, ids)
        cpu[0][1] = 7.95
        device[63][7] = math.nan
        with self.assertRaises(ValueError):
            common.check_device_greedy(cpu, device, [0] * 64, [0] * 64)


class NativeCapture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="llmx_device_reference_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "capture-\u6a21\u578b"
        self.root.mkdir()
        self.ids = self.root / "ids.txt"
        self.ids.write_text("97 98 99", encoding="ascii")
        self.model, self.control = self.root / "model.gguf", self.root / "control.gguf"
        self.goldens = f32.golden("baseline_f32.json")["fixtures"]
        for path, tied in [(self.model, False), (self.control, True)]:
            f32.write_model(path, f32.tensors(tied), config=dict(f32.CONFIG, context_length=128))
        self.tool = Path(common.EXE).with_name("llmx-model-logits" + (".exe" if os.name == "nt" else ""))
        self.version = subprocess.check_output([common.EXE, "--version"], text=True).strip()[5:]

    def capture(self, cache="f32", ubatch="2", ids=None, shares=None):
        prefix = self.root / "capture"
        cmd = [str(self.tool), str(self.model), str(ids or self.ids), str(prefix), "cpu", cache, ubatch, "0"]
        if shares is not None:
            cmd.append(shares)
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=120)
        return prefix, result

    def test_full_capture_against_independent_hf(self):
        prefix, result = self.capture()
        self.assertEqual(result.returncode, 0, result.stderr)
        meta_path = self.root / "capture.json"
        meta_path.write_text(result.stdout, encoding="utf-8")
        meta = check_device.capture_metadata(meta_path, [97, 98, 99], self.version)
        golden = next(x for x in self.goldens if not x["tied"])
        expected = {case["text"]: case["logits"] for case in golden["cases"]}
        for phase in ("batched", "decode"):
            rows = list(check_device.captured_rows(prefix, phase, 3, 257))
            for text, row in zip(("a", "ab", "abc"), rows):
                common.hf_logit_error("captured " + phase, dict(enumerate(row)), expected[text])
        greedy = list(check_device.captured_rows(prefix, "greedy", 64, 257))
        self.assertEqual(common.check_device_greedy(greedy, greedy, meta["greedy"], meta["greedy"])["matched_prefix"], 64)
        for key, value in [("vocab", True), ("version", "wrong"), ("tokens", [97, 98, 98]), ("greedy", [0] * 63), ("storage_types", [True])]:
            meta_path.write_text(json.dumps(dict(meta, **{key: value})))
            with self.assertRaises(ValueError):
                check_device.capture_metadata(meta_path, [97, 98, 99], self.version)

    def test_explicit_layer_shares(self):
        prefix, result = self.capture()
        self.assertEqual(result.returncode, 0, result.stderr)
        original = {phase: Path(str(prefix) + "." + phase + ".bin").read_bytes()
                    for phase in ("batched", "decode", "greedy")}
        metadata = result.stdout
        _, result = self.capture(shares="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, metadata)
        for phase, data in original.items():
            self.assertEqual(Path(str(prefix) + "." + phase + ".bin").read_bytes(), data)
        for shares in ("", "-1", "1.5", "0", "1,1", "2147483648"):
            with self.subTest(shares=shares):
                _, result = self.capture(shares=shares)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("share", result.stderr.lower())

    def test_bad_inputs_and_capture_lengths(self):
        for text in ("", "1", "97 -1", "97 257", "97 1.5", "97 x", "97 4294967296"):
            self.ids.write_text(text)
            _, result = self.capture()
            self.assertNotEqual(result.returncode, 0, text)
            self.assertIn("token ID", result.stderr)
        self.ids.write_text("97 98 99")
        for cache, ubatch in [("bad", "2"), ("f32", "0"), ("f32", "2.5")]:
            _, result = self.capture(cache, ubatch)
            self.assertNotEqual(result.returncode, 0)
        prefix, result = self.capture("f16", "3")
        self.assertEqual(result.returncode, 0, result.stderr)
        path = Path(str(prefix) + ".decode.bin")
        original = path.read_bytes()
        for data in (original[:-1], original + b"x"):
            path.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "capture size"):
                list(check_device.captured_rows(prefix, "decode", 3, 257))

    def test_runner_records_control_error_without_waiving_candidate(self):
        fixture = self.root / "fixture.json"
        fixture.write_text(json.dumps({"text": "abc", "token_ids": [97, 98, 99], "n_tokens": 3}))
        output = self.root / "calibration"
        original = check_device.capture_metadata
        def damage_control(path, tokens, version):
            metadata = original(path, tokens, version)
            if path.name == "control-device.stdout":
                binary = output / "control-device.batched.bin"
                rows = array.array("f")
                rows.frombytes(binary.read_bytes())
                best = max(range(metadata["vocab"]), key=lambda i: rows[i])
                rows[best] += 20.0
                binary.write_bytes(rows.tobytes())
            return metadata
        args = ["--exe", common.EXE, "--model", str(self.model), "--control", str(self.control),
                "--device", "cpu", "--cache-type", "f32", "--output", str(output)]
        with patch.object(check_device, "FIXTURE", fixture), patch.object(check_device, "capture_metadata", damage_control), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(check_device.main(args), 0)
        report = json.loads((output / "report.json").read_text())
        self.assertFalse(report["control"]["comparison"]["batched"]["criteria_passed"])
        self.assertTrue(report["candidate"]["comparison"]["batched"]["criteria_passed"])
        self.assertGreater(report["control"]["comparison"]["batched"]["nll_delta"], 0.01)

    def test_runner_and_full_vocabulary_damage(self):
        fixture = self.root / "fixture.json"
        fixture.write_text(json.dumps({"text": "abc", "token_ids": [97, 98, 99], "n_tokens": 3}))
        output = self.root / "run"
        args = ["--exe", common.EXE, "--model", str(self.model), "--control", str(self.control),
                "--device", "cpu", "--cache-type", "f32", "--ubatch", "2", "--output", str(output)]
        with patch.object(check_device, "FIXTURE", fixture), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(check_device.main(args), 0)
        report = json.loads((output / "report.json").read_text())
        self.assertEqual(len(report["commands"]), 7)
        self.assertEqual(report["candidate"]["comparison"]["batched"]["rows"], 3)
        self.assertEqual(report["candidate"]["comparison"]["decode"]["max_logit_gap"], 0)
        pair = report["candidate"]["captures"]
        path = Path(pair[1]["prefix"] + ".batched.bin")
        rows = array.array("f")
        rows.frombytes(path.read_bytes())
        low = min(range(257), key=lambda i: rows[i])
        rows[low] -= 0.01
        path.write_bytes(rows.tobytes())
        with self.assertRaisesRegex(ValueError, "logit gap"):
            check_device.compare_pair(*pair, [97, 98, 99], report["control"]["comparison"])
        args[args.index("--control") + 1] = str(self.model)
        args[-1] = str(self.root / "same-model")
        with patch.object(check_device, "FIXTURE", fixture), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(check_device.main(args), 1)
        self.assertIn("same model file", json.loads((self.root / "same-model/report.json").read_text())["error"])


def run(require=False):
    tool = Path(common.EXE).with_name("llmx-model-logits" + (".exe" if os.name == "nt" else ""))
    cases = [DeviceReference] + ([NativeCapture] if tool.exists() else [])
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(case) for case in cases)
    ok = unittest.TextTestRunner(stream=sys.stdout).run(suite).wasSuccessful()
    if not tool.exists():
        print("device-reference: " + ("FAIL" if require else "SKIP") + " - llmx-model-logits missing beside executable")
        return False if require or not ok else common.SKIPPED
    return ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
