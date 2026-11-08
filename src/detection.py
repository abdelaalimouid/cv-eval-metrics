"""Object detection metrics: IoU, AP at one IoU threshold, and COCO mAP@[.5:.95].

Example: the two dogs in data/images/dogs_beach.jpg, with hand-drawn ground-truth
boxes and two made-up detectors:
  * "tight"  - boxes close to the ground truth
  * "sloppy" - boxes that just pass IoU 0.5
Both get the same AP@0.5, which is the failure case shown on the slide.
"""
import numpy as np
from PIL import Image, ImageDraw

from common import IMAGES, FIGURES, BLUE, ORANGE, save_json

# Boxes are (x1, y1, x2, y2) in pixels of the 1280x772 image.
GROUND_TRUTH = [(96, 156, 524, 616), (444, 280, 1104, 610)]

# (box, confidence score). The third box is a false positive on the sand, so
# precision drops at the end of the ranking.
TIGHT = [((104, 166, 516, 606), 0.92), ((452, 290, 1094, 604), 0.88),
         ((900, 620, 1200, 760), 0.71)]
SLOPPY = [((40, 100, 600, 700), 0.92), ((400, 200, 1230, 700), 0.88),
          ((900, 620, 1200, 760), 0.71)]


def iou(a, b):
    """Intersection over union of two (x1, y1, x2, y2) boxes."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter)


def precision_recall(preds, gts, thr):
    """Greedy COCO-style matching in score order. Returns precision and recall arrays."""
    matched, tp = set(), []
    for box, _ in sorted(preds, key=lambda p: -p[1]):
        ious = [iou(box, g) if i not in matched else -1 for i, g in enumerate(gts)]
        best = int(np.argmax(ious))
        if ious[best] >= thr:
            matched.add(best)
            tp.append(1)
        else:
            tp.append(0)
    tp = np.array(tp)
    cum_tp = np.cumsum(tp)
    precision = cum_tp / np.arange(1, len(tp) + 1)
    recall = cum_tp / len(gts)
    return precision, recall


def average_precision(preds, gts, thr):
    """COCO 101-point interpolated AP at one IoU threshold."""
    p, r = precision_recall(preds, gts, thr)
    envelope = np.maximum.accumulate(p[::-1])[::-1]  # best precision at recall >= r
    points = [envelope[r >= t].max() if np.any(r >= t) else 0.0 for t in np.linspace(0, 1, 101)]
    return float(np.mean(points))


def coco_map(preds, gts):
    """Mean AP over IoU thresholds 0.50, 0.55, ..., 0.95."""
    return float(np.mean([average_precision(preds, gts, t) for t in np.arange(0.5, 0.96, 0.05)]))


def draw(preds, name):
    im = Image.open(IMAGES / "dogs_beach.jpg").convert("RGB")
    d = ImageDraw.Draw(im)
    for g in GROUND_TRUTH:
        d.rectangle(g, outline=BLUE, width=8)
    for box, score in preds[:2]:
        d.rectangle(box, outline=ORANGE, width=8)
    im.save(FIGURES / f"detection_{name}.jpg", quality=88)


def main():
    out = {"ground_truth": GROUND_TRUTH}
    for name, preds in [("tight", TIGHT), ("sloppy", SLOPPY)]:
        p, r = precision_recall(preds, GROUND_TRUTH, 0.5)
        out[name] = {
            "iou_dog1": round(iou(preds[0][0], GROUND_TRUTH[0]), 3),
            "iou_dog2": round(iou(preds[1][0], GROUND_TRUTH[1]), 3),
            "precision_at_0.5": [round(x, 3) for x in p],
            "recall_at_0.5": [round(x, 3) for x in r],
            "AP50": round(average_precision(preds, GROUND_TRUTH, 0.5), 3),
            "AP75": round(average_precision(preds, GROUND_TRUTH, 0.75), 3),
            "mAP_50_95": round(coco_map(preds, GROUND_TRUTH), 3),
        }
        draw(preds, name)
    save_json("detection", out)
    for name in ("tight", "sloppy"):
        o = out[name]
        print(f"{name:7s} IoU {o['iou_dog1']}/{o['iou_dog2']}  AP50 {o['AP50']}  "
              f"AP75 {o['AP75']}  mAP@[.5:.95] {o['mAP_50_95']}")
        print(f"        precision {o['precision_at_0.5']}  recall {o['recall_at_0.5']}")


if __name__ == "__main__":
    main()
