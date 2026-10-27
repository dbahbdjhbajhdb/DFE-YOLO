"""Core behavior tests; run from the repo root with PyTorch and torchvision."""

import importlib.util
from pathlib import Path
import sys
import unittest

import torch
from torchvision.ops import batched_nms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("pair_core", ROOT / "ultralytics/utils/learned_nms.py")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


class LearnedNMSTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_num_threads(2)
        self.det = torch.tensor([[0, 0, 10, 10, .9, 0], [.2, .2, 10.2, 10.2, .8, 0],
                                 [2, 0, 12, 10, .7, 0], [0, 0, 10, 10, .6, 1]], dtype=torch.float32)

    def test_no_rescue_exactly_matches_class_aware_nms(self):
        expected = batched_nms(self.det[:, :4], self.det[:, 4], self.det[:, 5], .5)
        torch.testing.assert_close(core.greedy_nms(self.det, .5), expected)
        torch.testing.assert_close(core.greedy_nms(self.det, .5, rescue=torch.zeros(4, 4, dtype=torch.bool)), expected)

    def test_rescued_candidate_can_still_suppress_later_duplicate(self):
        det = self.det[:3].clone()
        det[1, :4] = torch.tensor([2.1, 0, 12.1, 10])
        det[1, 4], det[2, 4] = .6, .8
        rescue = torch.zeros(3, 3, dtype=torch.bool)
        rescue[0, 1:] = True
        rescue[1:, 0] = True
        self.assertEqual(core.greedy_nms(det, .5, rescue=rescue).tolist(), [0, 2])

    def test_pair_selection_only_includes_same_class_small_conflicts(self):
        pairs = core.conflict_pairs(self.det, torch.tensor([True, True, False, True]), .5)
        self.assertEqual(pairs.tolist(), [[0, 1]])
        self.assertEqual(len(core.conflict_pairs(self.det, torch.ones(4, dtype=torch.bool), .5, cap=1)), 1)

    def test_geometry_roi_and_classifier_are_symmetric(self):
        pairs = torch.tensor([[0, 2], [2, 0]])
        geometry = core.pair_geometry(self.det, pairs, (64, 96))
        torch.testing.assert_close(geometry[0], geometry[1])
        rois = core.pair_rois(torch.randn(1, 8, 16, 24), self.det, pairs, (64, 96))
        self.assertEqual(tuple(rois.shape), (2, 3, 8, 3, 3))
        torch.testing.assert_close(rois[0, 0], rois[1, 1])
        for embedding in (8, 16):
            model = core.PairClassifier(8, embedding=embedding)
            logits = model(geometry, rois)
            torch.testing.assert_close(logits[0], logits[1])
            nn_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
            nn_loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_matching_distinguishes_duplicates_different_targets_and_ambiguity(self):
        gt = self.det[[0, 2], :4]
        pairs = torch.tensor([[0, 1], [0, 2]])
        labels = core.pair_targets(self.det, pairs, gt, torch.zeros(2))
        self.assertEqual(labels.tolist(), [0, 1])
        ambiguous_gt = self.det[[0, 1], :4]
        labels = core.pair_targets(self.det, pairs, ambiguous_gt, torch.zeros(2))
        self.assertEqual(labels.tolist(), [-1, -1])
        det = self.det.clone()
        det[1, :4] += 50
        self.assertEqual(core.pair_targets(det, pairs[:1], gt, torch.zeros(2)).tolist(), [0])

    def test_learned_nms_rescues_only_eligible_pairs(self):
        model = core.PairClassifier(8).eval()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.head[-1].bias.fill_(20)
        feature = torch.randn(1, 8, 16, 24)
        kept = core.learned_nms(self.det, feature, torch.ones(4, dtype=torch.bool), (64, 96), model, .9)
        self.assertEqual(kept.tolist(), [0, 1, 2, 3])
        kept = core.learned_nms(self.det, feature, torch.zeros(4, dtype=torch.bool), (64, 96), model, .9)
        self.assertEqual(kept.tolist(), [0, 3])
        kept = core.learned_nms(self.det, feature, torch.ones(4, dtype=torch.bool), (64, 96), model, 1.01)
        self.assertEqual(kept.tolist(), [0, 3])

    def test_empty_candidates(self):
        prediction = torch.zeros(6, 10)
        self.assertEqual(core.candidates(prediction).shape, (0, 6))
        self.assertEqual(len(core.greedy_nms(torch.zeros(0, 6))), 0)


class TrainingProtocolTest(unittest.TestCase):
    def test_no_scene_group_leakage(self):
        from learned_nms_experiment import grouped_split
        rows = [{"group": group, "image": f"{group}_{i}"} for group in ("a", "b", "c") for i in range(3)]
        train, val = grouped_split(rows, 0, .2)
        self.assertFalse({r["group"] for r in train} & {r["group"] for r in val})
        self.assertEqual(len(train) + len(val), len(rows))

    def test_unsafe_or_uninformative_classifier_disables_overrides(self):
        from learned_nms_experiment import classification_metrics
        scores = torch.tensor([.9, .9, .9, .9])
        labels = torch.tensor([1., 0., 1., 0.])
        result = classification_metrics(scores, labels, min_tp=1)
        self.assertAlmostEqual(result["pair_AP"], .5)
        self.assertEqual(result["threshold"], 1.01)
        result = classification_metrics(torch.tensor([.99, .98, .2, .1]), labels=torch.tensor([1., 1., 0., 0.]), min_tp=1)
        self.assertEqual(result["recall"], 1)
        self.assertEqual(result["precision"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
