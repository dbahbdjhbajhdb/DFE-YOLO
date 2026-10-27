"""Export YOLO11 conflict pairs, train a geometry+ROI classifier, and evaluate learned NMS.

Run from the repository root. See README_learned_nms.md for the experiment protocol.
"""

import argparse
from bisect import bisect_right
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import ops
from ultralytics.utils.learned_nms import (PairClassifier, candidates, conflict_pairs, greedy_nms,
                                          learned_nms, pair_geometry, pair_rois, pair_targets)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_tensors(path):
    return torch.load(path, map_location="cpu", weights_only=True)


class PairValidator(DetectionValidator):
    """Use the existing dataset transforms and AP evaluator with a replacement postprocess."""

    def __init__(self, *args, experiment, checkpoint=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.experiment, self.checkpoint = experiment, checkpoint
        self.feature = self.tap = None
        self.export_rows = []
        self.random = random.Random(experiment.seed)

    def init_metrics(self, model):
        super().init_metrics(model)
        if self.end2end or self.args.agnostic_nms or self.args.single_cls:
            raise ValueError("Only standard class-aware YOLO detection is supported")
        if self.experiment.command == "export" or self.experiment.mode == "learned":
            detector = next((m for m in model.modules() if isinstance(m, DetectionModel)), None)
            if detector is None:
                raise ValueError("A PyTorch YOLO detection checkpoint is required")
            self.tap = detector.model[self.experiment.feature_layer].register_forward_hook(self.capture)
        if self.checkpoint is not None:
            self.classifier = PairClassifier(self.checkpoint["meta"]["channels"]).to(self.device).eval()
            self.classifier.load_state_dict(self.checkpoint["state_dict"])

    def capture(self, module, inputs, output):
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            raise ValueError("feature-layer must output one BCHW tensor")
        self.feature = output.detach()

    def preprocess(self, batch):
        self.current_batch = super().preprocess(batch)
        return self.current_batch

    def export_image(self, detections, feature, small, image_shape, image_index):
        cfg, batch = self.experiment, self.current_batch
        pairs = conflict_pairs(detections, small, cfg.iou, cfg.pair_cap)
        mask = batch["batch_idx"] == image_index
        gt = ops.xywh2xyxy(batch["bboxes"][mask])
        gt *= gt.new_tensor([image_shape[1], image_shape[0], image_shape[1], image_shape[0]])
        labels = pair_targets(detections, pairs, gt, batch["cls"][mask].flatten(), cfg.match_iou)
        valid = (labels >= 0).nonzero().flatten().cpu().tolist()
        # Random sampling is label-independent; held-out precision is not computed
        # on an artificially class-balanced validation set.
        if len(valid) > cfg.pairs_per_image:
            valid = self.random.sample(valid, cfg.pairs_per_image)
        if not valid:
            return
        selected = torch.tensor(valid, device=detections.device)
        pairs, labels = pairs[selected], labels[selected]
        image_path = batch["im_file"][image_index]
        name = hashlib.sha1(str(image_path).encode()).hexdigest() + ".pt"
        torch.save({"geometry": pair_geometry(detections, pairs, image_shape).cpu(),
                    "rois": pair_rois(feature, detections, pairs, image_shape).half().cpu(),
                    "labels": labels.cpu()}, Path(cfg.out) / name)
        stem = Path(image_path).stem
        group = stem.split("_")[0] if cfg.grouping == "prefix" else str(image_path)
        row = {"file": name, "image": str(image_path), "group": group,
               "count": len(labels), "positives": int(labels.sum())}
        self.export_rows.append(row)
        with (Path(cfg.out) / "manifest.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def postprocess(self, preds):
        raw = preds[0] if isinstance(preds, (tuple, list)) else preds
        if raw.ndim != 3 or raw.shape[1] != 4 + self.nc:
            raise ValueError("Expected standard decoded detection predictions [B,4+nc,N]")
        cfg, batch = self.experiment, self.current_batch
        image_shape = batch["img"].shape[-2:]
        result = []
        for index, prediction in enumerate(raw):
            det = candidates(prediction, cfg.conf, cfg.max_candidates)
            feature = None if self.feature is None else self.feature[index:index + 1].float()
            if feature is not None:
                if self.checkpoint is not None and feature.shape[1] != self.checkpoint["meta"]["channels"]:
                    raise ValueError("ROI channel count does not match pair-classifier checkpoint")
                self.export_channels = feature.shape[1]
            original_boxes = ops.scale_boxes(image_shape, det[:, :4].clone(), batch["ori_shape"][index],
                                             batch["ratio_pad"][index])
            area = (original_boxes[:, 2:] - original_boxes[:, :2]).clamp_min(0).prod(1)
            small = (area > 0) & (area <= cfg.small_size**2)
            if cfg.command == "export":
                self.export_image(det, feature, small, image_shape, index)
            if cfg.command == "eval" and cfg.mode == "learned":
                keep = learned_nms(det, feature, small, image_shape, self.classifier,
                                   self.checkpoint["threshold"], cfg.iou, cfg.max_det, cfg.pair_cap)
            else:
                keep = greedy_nms(det, cfg.iou, cfg.max_det)
            selected = det[keep]
            result.append({"bboxes": selected[:, :4], "conf": selected[:, 4], "cls": selected[:, 5],
                           "extra": selected[:, 6:]})
        return result

    def finalize_metrics(self):
        if self.tap is not None:
            self.tap.remove()
        return super().finalize_metrics()


class PairDataset(Dataset):
    def __init__(self, folder, rows):
        self.folder, self.rows, self.ends = Path(folder), rows, []
        count = 0
        for row in rows:
            count += row["count"]
            self.ends.append(count)

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    @lru_cache(maxsize=8)
    def shard(self, index):
        return load_tensors(self.folder / self.rows[index]["file"])

    def __getitem__(self, index):
        image = bisect_right(self.ends, index)
        offset = index - (self.ends[image - 1] if image else 0)
        shard = self.shard(image)
        return shard["geometry"][offset], shard["rois"][offset].float(), shard["labels"][offset]


def grouped_split(rows, seed, fraction):
    groups = sorted({row["group"] for row in rows})
    if len(groups) < 2:
        raise ValueError("Need at least two scene/image groups; inspect naming or use --grouping image")
    random.Random(seed).shuffle(groups)
    count = max(1, min(len(groups) - 1, round(len(groups) * fraction)))
    held_out = set(groups[:count])
    return ([r for r in rows if r["group"] not in held_out],
            [r for r in rows if r["group"] in held_out])


def classification_metrics(probability, labels, required_precision=0.98, min_tp=5):
    """Conservative threshold from held-out pairs. Disable overrides if no threshold qualifies."""
    probability, labels = probability.cpu(), labels.cpu()
    positive_count = int(labels.sum())
    order = probability.argsort(descending=True, stable=True)
    ranked = labels[order]
    tp = ranked.cumsum(0)
    precision = tp / torch.arange(1, len(labels) + 1)
    # AP over unique score thresholds; tied scores must be evaluated together.
    end = torch.cat((probability[order][1:] != probability[order][:-1], torch.tensor([True])))
    recall = tp[end] / max(positive_count, 1)
    ap = float(((recall - torch.cat((torch.zeros(1), recall[:-1]))) * precision[end]).sum())
    chosen = {"threshold": 1.01, "precision": 0.0, "recall": 0.0, "true_positives": 0}
    for threshold in torch.linspace(0.5, 0.99, 50).tolist():
        selected = probability >= threshold
        true_positive = int(labels[selected].sum())
        p = true_positive / max(int(selected.sum()), 1)
        r = true_positive / max(positive_count, 1)
        if p >= required_precision and true_positive >= min_tp and r > chosen["recall"]:
            chosen = {"threshold": threshold, "precision": p, "recall": r, "true_positives": true_positive}
    return {"pair_AP": ap, "positive_pairs": positive_count, "negative_pairs": len(labels) - positive_count,
            **chosen}


def train_classifier(args):
    folder = Path(args.pairs)
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    if not meta.get("complete"):
        raise ValueError("Pair export was not completed")
    rows = [json.loads(line) for line in (folder / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    train_rows, val_rows = grouped_split(rows, args.seed, args.val_fraction)
    for name, split in (("train", train_rows), ("validation", val_rows)):
        positives = sum(row["positives"] for row in split)
        total = sum(row["count"] for row in split)
        if not 0 < positives < total:
            raise ValueError(f"{name}: need both classes, found {positives} different-object pairs / {total} total. "
                             "Export more images or inspect pair matching; do not force ambiguous pairs into positives.")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Use an empty output folder: {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "split.json").write_text(json.dumps({"train": train_rows, "validation": val_rows}, indent=2), encoding="utf-8")
    torch.manual_seed(args.seed)
    device = torch.device("cpu" if args.device == "cpu" else f"cuda:{args.device}")
    model = PairClassifier(meta["channels"]).to(device)
    train_data, val_data = PairDataset(folder, train_rows), PairDataset(folder, val_rows)
    loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=0)
    validation = DataLoader(val_data, batch_size=args.batch, num_workers=0)
    positives = sum(r["positives"] for r in train_rows)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(train_data) - positives) / positives, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history, best = [], -1.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for geometry, rois, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(geometry.to(device), rois.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(labels)
        model.eval()
        probabilities, targets = [], []
        with torch.no_grad():
            for geometry, rois, labels in validation:
                probabilities.append(model(geometry.to(device), rois.to(device)).sigmoid().cpu())
                targets.append(labels)
        metrics = classification_metrics(torch.cat(probabilities), torch.cat(targets), args.precision, args.min_tp)
        metrics.update(epoch=epoch + 1, train_loss=total_loss / len(train_data))
        history.append(metrics)
        print(json.dumps(metrics))
        # Select by held-out pair AP; detection AP must be evaluated independently.
        if metrics["pair_AP"] > best:
            best = metrics["pair_AP"]
            torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "meta": meta, "threshold": metrics["threshold"], "validation": metrics}, out / "best.pt")
        (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Classifier parameters: {sum(p.numel() for p in model.parameters())}; checkpoint: {out / 'best.pt'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for action in ("export", "eval"):
        sub = commands.add_parser(action)
        sub.add_argument("--weights", required=True, help="Frozen trained YOLO11 .pt")
        sub.add_argument("--data", required=True)
        sub.add_argument("--device", default="0")
        sub.add_argument("--imgsz", type=int, default=640)
        sub.add_argument("--conf", type=float, default=0.001)
        sub.add_argument("--iou", type=float, default=0.5)
        sub.add_argument("--small-size", type=float, default=32, help="sqrt(box area) in original image pixels")
        sub.add_argument("--max-candidates", type=int, default=3000)
        sub.add_argument("--pair-cap", type=int, default=1024)
        sub.add_argument("--feature-layer", type=int, default=2, help="Standard YOLO11 P2 backbone layer")
        sub.add_argument("--max-det", type=int, default=1000)
        sub.add_argument("--seed", type=int, default=0)
        sub.add_argument("--project", default="runs/learned_nms")
        if action == "export":
            sub.add_argument("--out", required=True)
            sub.add_argument("--pairs-per-image", type=int, default=64)
            sub.add_argument("--match-iou", type=float, default=0.5)
            sub.add_argument("--grouping", choices=("prefix", "image"), default="prefix")
        else:
            sub.add_argument("--mode", choices=("nms", "learned"), required=True)
            sub.add_argument("--pair-model")
            sub.add_argument("--split", choices=("val", "test"), default="val")
    fit = commands.add_parser("train")
    fit.add_argument("--pairs", required=True)
    fit.add_argument("--out", required=True)
    fit.add_argument("--device", default="0")
    fit.add_argument("--epochs", type=int, default=30)
    fit.add_argument("--batch", type=int, default=128)
    fit.add_argument("--lr", type=float, default=0.001)
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument("--val-fraction", type=float, default=0.2)
    fit.add_argument("--precision", type=float, default=0.98)
    fit.add_argument("--min-tp", type=int, default=5)
    args = parser.parse_args()
    if args.command == "train":
        if not (0 < args.val_fraction < 1 and 0 < args.precision <= 1 and args.epochs > 0 and args.batch > 0 and args.min_tp > 0):
            parser.error("Invalid split fraction, precision, epochs, batch, or minimum true positives")
        train_classifier(args)
        return
    if not (0 <= args.conf <= 1 and 0 < args.iou < 1 and args.small_size > 0
            and args.max_candidates > 0 and args.pair_cap > 0 and args.max_det > 0):
        parser.error("Invalid candidate, IoU, size, or budget settings")
    detector_hash = file_hash(args.weights)
    checkpoint = None
    if args.command == "export":
        if args.pairs_per_image < 1 or not 0 < args.match_iou <= 1:
            parser.error("Invalid pair sample count or GT match threshold")
        out = Path(args.out)
        if out.exists() and any(out.iterdir()):
            raise FileExistsError(f"Use an empty export directory: {out}")
        out.mkdir(parents=True, exist_ok=True)
        meta = {**vars(args), "detector_sha256": detector_hash, "complete": False, "roi_size": 3}
        (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        (out / "manifest.jsonl").touch()
    elif args.mode == "learned":
        if not args.pair_model:
            parser.error("--pair-model is required for learned NMS")
        checkpoint = load_tensors(args.pair_model)
        meta = checkpoint["meta"]
        if meta["detector_sha256"] != detector_hash:
            raise ValueError("Pair classifier belongs to a different detector checkpoint")
        for key in ("imgsz", "conf", "iou", "small_size", "max_candidates", "pair_cap", "feature_layer"):
            if getattr(args, key) != meta[key]:
                raise ValueError(f"{key} differs from export settings ({meta[key]}); keep proposal settings matched")
    validator_holder = []

    def make_validator(**kwargs):
        instance = PairValidator(**kwargs, experiment=args, checkpoint=checkpoint)
        validator_holder.append(instance)
        return instance

    model = YOLO(args.weights)
    metrics = model.val(data=args.data, validator=make_validator, split="train" if args.command == "export" else args.split,
                        imgsz=args.imgsz, batch=1, device=args.device, workers=0, conf=args.conf, iou=args.iou,
                        max_det=args.max_det, plots=False, half=False, augment=False, compile=False,
                        project=args.project, name="export" if args.command == "export" else args.mode)
    if args.command == "export":
        validator = validator_holder[0]
        if not validator.export_rows:
            raise ValueError("No reliable conflict pairs exported; inspect detector proposals, labels, and matching")
        meta.update(complete=True, channels=validator.export_channels,
                    images_with_pairs=len(validator.export_rows),
                    pairs=sum(row["count"] for row in validator.export_rows),
                    positives=sum(row["positives"] for row in validator.export_rows))
        (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(json.dumps(meta, indent=2))
    else:
        summary = {"mode": args.mode, "detector_sha256": detector_hash,
                   "metrics": {k: float(v) for k, v in metrics.results_dict.items()},
                   "speed_ms_per_image": metrics.speed,
                   "pair_threshold": None if checkpoint is None else checkpoint["threshold"],
                   "settings": vars(args)}
        (validator_holder[0].save_dir / "postprocess_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
