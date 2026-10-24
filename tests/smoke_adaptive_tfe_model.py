"""Full model smoke test. Run from the repo root with its dependencies installed."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from ultralytics.nn.tasks import DetectionModel


def main():
    torch.manual_seed(17)
    torch.set_num_threads(2)
    path = Path(__file__).resolve().parents[1] / "ultralytics/cfg/models/v8/yolov8-adaptive-tfe-p2.yaml"
    config = yaml.safe_load(path.read_text())
    config["scale"] = "n"
    for mode in ("adaptive", "mean", "concat"):
        cfg = deepcopy(config)
        for row in cfg["head"]:
            if row[2] == "AdaptiveTFE":
                row[3][-1] = mode
        model = DetectionModel(cfg, nc=2, verbose=False)
        model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
        model.train()
        images = torch.randn(2, 3, 128, 128)
        batch = {
            "img": images,
            "batch_idx": torch.tensor([0, 1]),
            "cls": torch.tensor([[0.0], [1.0]]),
            "bboxes": torch.tensor([[0.4, 0.5, 0.06, 0.06], [0.6, 0.3, 0.12, 0.12]]),
        }
        loss, components = model(batch)
        assert torch.isfinite(loss).all()
        loss.sum().backward()
        assert model.stride.tolist() == [4.0, 8.0, 16.0, 32.0]
        if mode == "adaptive":
            for index in (10, 12):
                grad = model.model[index].weight_logits.weight.grad
                assert torch.isfinite(grad).all() and grad.abs().sum() > 0
        model.eval()
        with torch.no_grad():
            prediction, _ = model(images)
        assert prediction.shape == (2, 6, 1360)
        assert torch.isfinite(prediction).all()
        print(mode, "parameters:", sum(p.numel() for p in model.parameters()),
              "loss components:", components.tolist(), "forward/backward/eval OK")


if __name__ == "__main__":
    main()
