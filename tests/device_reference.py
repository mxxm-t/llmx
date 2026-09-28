import math
import sys
import unittest

import common


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
        cpu = [[8.0, 7.95, 5.0, 4.0, 3.0, 2.95, -1.0, -3.0] for _ in self.ids]
        device = [[7.96, 8.0, 5.0, 4.0, 2.97, 3.0, -1.0, -3.0] for _ in self.ids]
        got = common.check_device_rows(cpu, device, self.ids, 0.1)
        self.assertEqual(got["top1_checked"], 0)
        self.assertEqual(got["min_top5_overlap"], 5)

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


def run(require=False):
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DeviceReference)
    return unittest.TextTestRunner(stream=sys.stdout).run(suite).wasSuccessful()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
