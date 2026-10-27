"""Experimental geometry + ROI pair classifier for class-aware greedy NMS.

Positive means 'two different real objects; preserve this pair'. This module
requires frozen-detector ROI features and never modifies detection scores.
"""

import torch
from torch import nn
from torchvision.ops import batched_nms, box_iou, roi_align


class PairClassifier(nn.Module):
    def __init__(self, channels, roi_size=3, embedding=16):
        super().__init__()
        self.channels, self.roi_size, self.embedding = channels, roi_size, embedding
        self.encoder = nn.Sequential(nn.Flatten(1), nn.Linear(channels * roi_size**2, embedding),
                                     nn.LayerNorm(embedding), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(16 + 3 * embedding, 32), nn.SiLU(),
                                  nn.Linear(32, 1))

    def forward(self, geometry, rois):
        # Symmetric in the first two ROIs: swapping boxes cannot change the result.
        embeddings = self.encoder(rois.flatten(0, 1)).reshape(-1, 3, self.embedding)
        a, b, context = embeddings.unbind(1)
        return self.head(torch.cat((geometry, (a + b) / 2, (a - b).abs(), context), 1)).squeeze(1)


def candidates(prediction, conf=0.001, limit=3000):
    """Decode one standard YOLO detection output [4+nc,N], using multi-label candidates."""
    if prediction.ndim != 2 or prediction.shape[0] <= 4:
        raise ValueError("Expected standard YOLO [4+nc,N] predictions, not end-to-end detections")
    prediction = prediction.float().T
    indices, classes = torch.where(prediction[:, 4:] > conf)
    xywh = prediction[indices, :4]
    boxes = torch.cat((xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2), 1)
    output = torch.cat((boxes, prediction[indices, classes + 4, None], classes[:, None].float()), 1)
    valid = torch.isfinite(output).all(1) & ((boxes[:, 2:] - boxes[:, :2]) > 0).all(1)
    output = output[valid]
    return output[output[:, 4].argsort(descending=True)[:limit]]


def conflict_pairs(detections, small, iou_threshold, cap=1024):
    """Same-class small-box conflicts; cap high-confidence pairs, fall back to NMS otherwise."""
    overlap = box_iou(detections[:, :4], detections[:, :4])
    conflicts = ((overlap > iou_threshold) & (detections[:, 5, None] == detections[None, :, 5])
                 & small[:, None] & small[None, :]).triu(1)
    pairs = conflicts.nonzero()
    if len(pairs) > cap:
        confidence = detections[pairs[:, 0], 4] * detections[pairs[:, 1], 4]
        pairs = pairs[confidence.topk(cap).indices]
    return pairs


def pair_geometry(detections, pairs, image_shape):
    """Sixteen symmetric geometry/score descriptors. No ground truth at inference."""
    a, b = detections[pairs[:, 0]], detections[pairs[:, 1]]
    wa, wb = (a[:, 2:4] - a[:, :2]).clamp_min(1e-6), (b[:, 2:4] - b[:, :2]).clamp_min(1e-6)
    aa, ab = wa.prod(1), wb.prod(1)
    intersection_wh = (torch.minimum(a[:, 2:4], b[:, 2:4]) - torch.maximum(a[:, :2], b[:, :2])).clamp_min(0)
    intersection = intersection_wh.prod(1)
    amin, amax = torch.minimum(aa, ab), torch.maximum(aa, ab)
    center_distance = ((a[:, :2] + a[:, 2:4] - b[:, :2] - b[:, 2:4]) / (wa + wb)).abs()
    aspect_a, aspect_b = (wa[:, 0] / wa[:, 1]).log(), (wb[:, 0] / wb[:, 1]).log()
    score_min, score_max = torch.minimum(a[:, 4], b[:, 4]), torch.maximum(a[:, 4], b[:, 4])
    features = [intersection / (aa + ab - intersection).clamp_min(1e-6),
                intersection / amin.clamp_min(1e-6), amin / (image_shape[0] * image_shape[1]),
                amax / (image_shape[0] * image_shape[1]), center_distance[:, 0], center_distance[:, 1],
                intersection_wh[:, 0] / torch.minimum(wa[:, 0], wb[:, 0]),
                intersection_wh[:, 1] / torch.minimum(wa[:, 1], wb[:, 1]), amin / amax,
                torch.minimum(wa[:, 0], wb[:, 0]) / torch.maximum(wa[:, 0], wb[:, 0]),
                torch.minimum(wa[:, 1], wb[:, 1]) / torch.maximum(wa[:, 1], wb[:, 1]),
                torch.minimum(aspect_a, aspect_b), torch.maximum(aspect_a, aspect_b),
                score_min, score_max, score_max - score_min]
    return torch.stack(features, 1)


def pair_rois(feature, detections, pairs, image_shape, roi_size=3):
    """ROIAlign for box A, box B, and a slightly expanded union context, all in one pass."""
    a, b = detections[pairs[:, 0], :4], detections[pairs[:, 1], :4]
    union = torch.cat((torch.minimum(a[:, :2], b[:, :2]), torch.maximum(a[:, 2:], b[:, 2:])), 1)
    margin = (union[:, 2:] - union[:, :2]) * 0.1
    union = union + torch.cat((-margin, margin), 1)
    boxes = torch.stack((a, b, union), 1).flatten(0, 1).clone()
    # Explicit x/y scaling supports rectangular inputs and rounding differences.
    boxes[:, [0, 2]] *= feature.shape[-1] / image_shape[1]
    boxes[:, [1, 3]] *= feature.shape[-2] / image_shape[0]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, feature.shape[-1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, feature.shape[-2])
    rois = roi_align(feature.float(), [boxes], output_size=roi_size, spatial_scale=1,
                     sampling_ratio=2, aligned=True)
    return rois.reshape(len(pairs), 3, feature.shape[1], roi_size, roi_size)


def pair_targets(detections, pairs, gt_boxes, gt_classes, match_iou=0.5, margin=0.1, background_iou=0.1):
    """Return 1=different GTs, 0=duplicate/background conflict, -1=ambiguous (exclude)."""
    labels = detections.new_full((len(pairs),), -1)
    if not len(gt_boxes) or not len(pairs):
        return labels
    overlaps = box_iou(detections[:, :4], gt_boxes)
    overlaps = overlaps * (detections[:, 5, None] == gt_classes[None, :])
    best, identity = overlaps.max(1)
    second = overlaps.topk(2, dim=1).values[:, 1] if len(gt_boxes) > 1 else torch.zeros_like(best)
    reliable = (best >= match_iou) & ((best - second) >= margin)
    background = best <= background_iou
    i, j = pairs.T
    both = reliable[i] & reliable[j]
    labels[both] = (identity[i[both]] != identity[j[both]]).float()
    labels[(reliable[i] & background[j]) | (reliable[j] & background[i])] = 0
    return labels


def greedy_nms(detections, iou_threshold=0.5, max_det=300, rescue=None):
    """Rescued pairs remain in the candidate queue; later duplicates can still suppress them."""
    if not len(detections):
        return torch.empty(0, dtype=torch.long, device=detections.device)
    if rescue is None or not rescue.any():
        return batched_nms(detections[:, :4], detections[:, 4], detections[:, 5], iou_threshold)[:max_det]
    order, kept = detections[:, 4].argsort(descending=True), []
    while order.numel() and len(kept) < max_det:
        first = order[0]
        kept.append(first)
        rest = order[1:]
        overlap = box_iou(detections[first, :4][None], detections[rest, :4])[0]
        suppress = ((overlap > iou_threshold) & (detections[first, 5] == detections[rest, 5])
                    & ~rescue[first, rest])
        order = rest[~suppress]
    return torch.stack(kept)


@torch.no_grad()
def learned_nms(detections, feature, small, image_shape, classifier, threshold,
                iou_threshold=0.5, max_det=300, pair_cap=1024):
    if threshold > 1:
        return greedy_nms(detections, iou_threshold, max_det)
    pairs = conflict_pairs(detections, small, iou_threshold, pair_cap)
    rescue = torch.zeros((len(detections), len(detections)), dtype=torch.bool, device=detections.device)
    for block in pairs.split(128):
        if not len(block):
            continue
        geometry = pair_geometry(detections, block, image_shape)
        rois = pair_rois(feature, detections, block, image_shape, classifier.roi_size)
        probability = classifier(geometry, rois).sigmoid()
        selected = block[probability >= threshold]
        rescue[selected[:, 0], selected[:, 1]] = True
        rescue[selected[:, 1], selected[:, 0]] = True
    return greedy_nms(detections, iou_threshold, max_det, rescue)
