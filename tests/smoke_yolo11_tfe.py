"""YOLO11 detection-loss checks for baseline, mean, adaptive, and concat."""

from pathlib import Path
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics.nn.modules import AdaptiveTFE
from ultralytics.nn.tasks import DetectionModel, yaml_model_load


def main():
    torch.set_num_threads(2)
    folder = ROOT / "ultralytics/cfg/models/11"
    configs = {"baseline": "yolo11n.yaml", "mean": "yolo11n-tfe-mean.yaml",
               "adaptive": "yolo11n-adaptive-tfe.yaml", "concat": "yolo11n-tfe-concat.yaml"}
    for mode, filename in configs.items():
        torch.manual_seed(0)
        config = yaml_model_load(folder / filename)
        model = DetectionModel(config, nc=10, verbose=False)
        model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
        assert model.stride.tolist() == [8, 16, 32]
        nodes = [m for m in model.modules() if isinstance(m, AdaptiveTFE)]
        assert len(nodes) == (0 if mode == "baseline" else 2)
        assert all(node.fusion == mode for node in nodes)
        model.train()
        batch = {
            "img": torch.randn(2, 3, 128, 128),
            "batch_idx": torch.tensor([0, 1]),
            "cls": torch.tensor([[0.0], [3.0]]),
            "bboxes": torch.tensor([[0.4, 0.5, 0.06, 0.06], [0.6, 0.3, 0.12, 0.12]]),
        }
        loss, _ = model(batch)
        assert torch.isfinite(loss).all()
        loss.sum().backward()
        if mode == "adaptive":
            for node in nodes:
                gradient = node.weight_logits.weight.grad
                assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        model.eval()
        with torch.no_grad():
            prediction, _ = model(batch["img"])
        assert prediction.shape == (2, 14, 336)
        assert torch.isfinite(prediction).all()
        print(mode, "parameters:", sum(p.numel() for p in model.parameters()),
              "strides:", model.stride.tolist(), "loss/backward/inference OK")


if __name__ == "__main__":
    main()
