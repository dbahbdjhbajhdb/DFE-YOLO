"""Run with python tests/test_adaptive_tfe.py; only PyTorch is required."""

import importlib.util
from pathlib import Path
import unittest

import torch

source = Path(__file__).with_name("adaptive_tfe.py")
if not source.exists():
    source = Path(__file__).resolve().parents[1] / "ultralytics/nn/modules/adaptive_tfe.py"
spec = importlib.util.spec_from_file_location("adaptive_tfe", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
AdaptiveTFE = module.AdaptiveTFE


class AdaptiveTFETest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(2)

    def inputs(self, odd=False, requires_grad=False):
        sizes = [(31, 47), (16, 24), (8, 12)] if odd else [(32, 48), (16, 24), (8, 12)]
        return [torch.randn(2, c, h, w, requires_grad=requires_grad)
                for c, (h, w) in zip([8, 16, 32], sizes)]

    def test_odd_rectangular_inputs_and_spatial_probability_maps(self):
        model = AdaptiveTFE([8, 16, 32], 12).eval()
        output, weights = model.forward_with_weights(self.inputs(odd=True))
        self.assertEqual(tuple(output.shape), (2, 12, 16, 24))
        self.assertEqual(tuple(weights.shape), (2, 3, 16, 24))
        torch.testing.assert_close(weights.sum(1), torch.ones(2, 16, 24))
        self.assertTrue((weights >= 0).all() and (weights <= 1).all())
        self.assertGreater(weights[0, 0].std().item(), 1e-6)
        self.assertFalse(torch.equal(weights[0], weights[1]))

    def test_gradients_reach_all_scales_and_weight_predictor(self):
        model = AdaptiveTFE([8, 16, 32], 12)
        inputs = self.inputs(requires_grad=True)
        model(inputs).square().mean().backward()
        for tensor in inputs:
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(tensor.grad.abs().sum().item(), 0)
        for layer in [*model.weight_reduce, model.weight_logits]:
            for parameter in layer.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_forced_scale_selection_changes_actual_fused_features(self):
        model = AdaptiveTFE([8, 16, 32], 12).eval()
        inputs = self.inputs()
        with torch.no_grad():
            aligned = model._align(inputs)
            model.weight_logits.weight.zero_()
            for index in range(3):
                model.weight_logits.bias.fill_(-30)
                model.weight_logits.bias[index] = 30
                output, weights = model.forward_with_weights(inputs)
                torch.testing.assert_close(output, model.refine(aligned[index]))
                torch.testing.assert_close(weights[:, index], torch.ones_like(weights[:, index]))

    def test_mean_and_concat_controls_have_same_output_interface(self):
        for mode in ["mean", "concat"]:
            model = AdaptiveTFE([8, 16, 32], 12, fusion=mode).eval()
            inputs = self.inputs()
            output, weights = model.forward_with_weights(inputs)
            self.assertEqual(tuple(output.shape), (2, 12, 16, 24))
            aligned = model._align(inputs)
            if mode == "mean":
                torch.testing.assert_close(output, model.refine(sum(aligned) / 3))
                torch.testing.assert_close(weights, torch.full_like(weights, 1 / 3))
            else:
                self.assertIsNone(weights)
                torch.testing.assert_close(
                    output, model.refine(model.concat_project(torch.cat(aligned, 1)))
                )

    def test_cpu_mixed_precision_forward_backward(self):
        model = AdaptiveTFE([8, 16, 32], 12)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output, weights = model.forward_with_weights(self.inputs())
            loss = output.float().square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(model.weight_logits.weight.grad).all())
        torch.testing.assert_close(weights.float().sum(1), torch.ones(2, 16, 24), atol=0.008, rtol=0)

    def test_invalid_configuration_and_resolution_order(self):
        with self.assertRaises(ValueError):
            AdaptiveTFE([8, 16], 12)
        with self.assertRaises(ValueError):
            AdaptiveTFE([8, 16, 32], 12, fusion="invalid")
        with self.assertRaises(ValueError):
            AdaptiveTFE([8, 16, 32], 12)(list(reversed(self.inputs())))


if __name__ == "__main__":
    unittest.main(verbosity=2)
