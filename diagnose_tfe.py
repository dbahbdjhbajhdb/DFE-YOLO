"""Inspect existing training CSVs or AdaptiveTFE gates without retraining.

Examples (run in the repository root):
    python diagnose_tfe.py results runs/detect/baseline/results.csv runs/detect/tfe/results.csv
    python diagnose_tfe.py weights --weights runs/detect/tfe/weights/best.pt --source path/to/images
"""

import argparse
import csv
import json
import math
from pathlib import Path


class GateStats:
    """Streaming scale statistics; never retains feature maps or autograd graphs."""

    def __init__(self):
        self.pixels = self.images = 0
        self.weight_sum = [0.0] * 3
        self.spatial_std_sum = [0.0] * 3
        self.entropy_sum = self.uniform = self.strong = 0.0

    def update(self, logits):
        import torch

        with torch.no_grad():
            weights = logits.detach().float().softmax(1)
            batch, _, height, width = weights.shape
            self.images += batch
            self.pixels += batch * height * width
            sums = weights.sum((0, 2, 3)).cpu().tolist()
            stds = weights.flatten(2).std(2, unbiased=False).sum(0).cpu().tolist()
            self.weight_sum = [a + b for a, b in zip(self.weight_sum, sums)]
            self.spatial_std_sum = [a + b for a, b in zip(self.spatial_std_sum, stds)]
            entropy = -(weights * weights.clamp_min(1e-12).log()).sum(1) / math.log(3)
            self.entropy_sum += entropy.sum().item()
            self.uniform += ((weights.max(1).values - weights.min(1).values) < 0.05).sum().item()
            self.strong += (weights.max(1).values >= 0.8).sum().item()

    def report(self):
        if not self.pixels:
            raise ValueError("No inference batches were observed")
        return {
            "image_forwards": self.images,
            "feature_pixels": self.pixels,
            "scale_order": ["high_resolution", "middle_resolution", "low_resolution"],
            "mean_weight": [x / self.pixels for x in self.weight_sum],
            "mean_spatial_std_per_image": [x / self.images for x in self.spatial_std_sum],
            "mean_normalized_entropy": self.entropy_sum / self.pixels,
            "near_uniform_fraction_max_minus_min_lt_0.05": self.uniform / self.pixels,
            "strong_preference_fraction_max_ge_0.8": self.strong / self.pixels,
        }


def summarize_csv(path):
    keys = ["epoch", "metrics/precision(B)", "metrics/recall(B)",
            "metrics/mAP50(B)", "metrics/mAP50-95(B)"]
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(handle)]
    if not rows or any(k not in rows[0] for k in keys):
        raise ValueError(f"{path}: expected nonempty Ultralytics detection results.csv with {keys}")
    valid = [r for r in rows if all(math.isfinite(float(r[k])) for k in keys)]
    if not valid:
        raise ValueError(f"{path}: no finite metric rows")
    best = max(valid, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    return {"file": str(path), "epochs_logged": len(rows),
            "selection": "row with highest validation mAP50-95; not an independent test score",
            "best_row": {k: float(best[k]) for k in keys},
            "last_row": {k: float(rows[-1][k]) for k in keys}}


def inspect_weights(args):
    from ultralytics import YOLO
    from ultralytics.nn.modules import AdaptiveTFE

    model = YOLO(args.weights)
    stats, handles = {}, []

    def attach_after_warmup(predictor):
        # on_predict_start runs after the predictor's dummy warmup forward.
        for name, module in predictor.model.named_modules():
            if isinstance(module, AdaptiveTFE) and module.fusion == "adaptive":
                accumulator = stats.setdefault(name, GateStats())
                handles.append(module.weight_logits.register_forward_hook(
                    lambda layer, inputs, output, acc=accumulator: acc.update(output)
                ))
        if not handles:
            raise ValueError("Checkpoint has no active AdaptiveTFE gates. Check the model and fusion mode.")

    model.add_callback("on_predict_start", attach_after_warmup)
    stream = model.predict(source=args.source, device=args.device, imgsz=args.imgsz,
                           batch=1, stream=True, save=False, verbose=False)
    try:
        for count, _ in enumerate(stream, 1):
            if count >= args.max_images:
                break
    finally:
        stream.close()
        for handle in handles:
            handle.remove()
    if not stats:
        raise ValueError("No images were processed")
    return {"checkpoint": args.weights, "nodes": {name: acc.report() for name, acc in stats.items()},
            "notes": ["Includes background and any letterbox padding; no object-specific masking.",
                      "Global weight statistics cannot alone prove usefulness or explain accuracy."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    results = commands.add_parser("results")
    results.add_argument("csv_files", nargs="+")
    weights = commands.add_parser("weights")
    weights.add_argument("--weights", required=True)
    weights.add_argument("--source", required=True)
    weights.add_argument("--device", default="0")
    weights.add_argument("--imgsz", type=int, default=640)
    weights.add_argument("--max-images", type=int, default=32)
    args = parser.parse_args()
    if args.command == "results":
        output = [summarize_csv(p) for p in args.csv_files]
    else:
        if args.max_images < 1:
            parser.error("--max-images must be positive")
        output = inspect_weights(args)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
