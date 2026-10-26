"""Run from the repository root: python tests/test_tfe_diagnostics.py."""

import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

import torch

spec = importlib.util.spec_from_file_location("diagnose_tfe", Path(__file__).resolve().parents[1] / "diagnose_tfe.py")
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


class DiagnosticsTest(unittest.TestCase):
    def test_uniform_gate_is_not_reported_as_scale_preference(self):
        stats = diagnostics.GateStats()
        stats.update(torch.zeros(2, 3, 4, 5, requires_grad=True))
        result = stats.report()
        for value in result["mean_weight"]:
            self.assertAlmostEqual(value, 1 / 3, places=6)
        self.assertAlmostEqual(result["mean_normalized_entropy"], 1, places=6)
        self.assertEqual(result["mean_spatial_std_per_image"], [0, 0, 0])
        self.assertEqual(result["near_uniform_fraction_max_minus_min_lt_0.05"], 1)
        self.assertEqual(result["strong_preference_fraction_max_ge_0.8"], 0)

    def test_spatial_selection_is_visible_and_weighted_by_pixel_count(self):
        stats = diagnostics.GateStats()
        logits = torch.full((1, 3, 2, 4), -30.0)
        logits[:, 0, :, :2] = 30
        logits[:, 2, :, 2:] = 30
        stats.update(logits)
        result = stats.report()
        self.assertAlmostEqual(result["mean_weight"][0], 0.5)
        self.assertAlmostEqual(result["mean_weight"][2], 0.5)
        self.assertAlmostEqual(result["mean_spatial_std_per_image"][0], 0.5)
        self.assertLess(result["mean_normalized_entropy"], 0.001)
        self.assertEqual(result["strong_preference_fraction_max_ge_0.8"], 1)
        stats.update(torch.zeros(1, 3, 2, 2))
        result = stats.report()
        self.assertEqual(result["feature_pixels"], 12)
        self.assertAlmostEqual(result["mean_weight"][0], (4 + 4 / 3) / 12, places=6)

    def test_csv_uses_matching_metrics_at_best_epoch_and_handles_header_spaces(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            path.write_text("epoch, metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B)\n"
                            "1,0.4,0.3,0.5,0.2\n2,0.6,0.5,0.7,0.4\n3,0.7,0.4,0.6,0.3\n")
            result = diagnostics.summarize_csv(path)
        self.assertEqual(result["best_row"]["epoch"], 2)
        self.assertEqual(result["best_row"]["metrics/precision(B)"], 0.6)
        self.assertEqual(result["last_row"]["epoch"], 3)

    def test_empty_statistics_fail_clearly(self):
        with self.assertRaises(ValueError):
            diagnostics.GateStats().report()


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
