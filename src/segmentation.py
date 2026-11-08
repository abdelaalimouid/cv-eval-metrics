"""Segmentation metrics: pixel accuracy, IoU (Jaccard), Dice and mean IoU.

Example: the yellow school bus in data/images/school_bus.jpg. The reference mask
is a polygon traced by hand around the bus, the way segmentation labels are
usually made. Three predictions:
  * "dilated" - the mask grown by 8 pixels (slight over-segmentation)
  * "shifted" - the mask moved 60 pixels to the right
  * "empty"   - predicts background everywhere
The empty mask still gets high pixel accuracy: the failure case on the slide.
"""
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from common import IMAGES, FIGURES, BLUE, ORANGE, save_json

# Hand-traced outline of the bus, (x, y) pixels in the 1280x867 image.
BUS_POLYGON = [(206, 343), (251, 311), (491, 304), (738, 343), (741, 571), (677, 576),
               (667, 631), (599, 631), (590, 599), (498, 599), (489, 649), (434, 654),
               (421, 622), (286, 622), (274, 649), (219, 649), (206, 617), (197, 434)]


def bus_mask():
    """Boolean mask of the hand-traced bus polygon."""
    w, h = Image.open(IMAGES / "school_bus.jpg").size
    canvas = Image.new("L", (w, h), 0)
    ImageDraw.Draw(canvas).polygon(BUS_POLYGON, fill=1)
    return np.asarray(canvas).astype(bool)


def scores(pred, gt):
    """Pixel accuracy, IoU and Dice for the bus class, plus mIoU over {background, bus}."""
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou_bus = inter / union
    dice = 2 * inter / (pred.sum() + gt.sum())
    iou_bg = np.logical_and(~pred, ~gt).sum() / np.logical_or(~pred, ~gt).sum()
    return {
        "pixel_accuracy": round(float((pred == gt).mean()), 3),
        "iou_bus": round(float(iou_bus), 3),
        "dice_bus": round(float(dice), 3),
        "miou": round(float((iou_bus + iou_bg) / 2), 3),
    }


def overlay(pred, gt, name):
    im = np.asarray(Image.open(IMAGES / "school_bus.jpg").convert("RGB")).astype(float)
    edge = gt ^ ndimage.binary_erosion(gt, iterations=4)
    tint = np.array(Image.new("RGB", (1, 1), ORANGE)).reshape(3).astype(float)
    im[pred] = 0.45 * im[pred] + 0.55 * tint
    im[edge] = np.array(Image.new("RGB", (1, 1), BLUE)).reshape(3)
    Image.fromarray(im.astype(np.uint8)).save(FIGURES / f"segmentation_{name}.jpg", quality=88)


def main():
    gt = bus_mask()
    preds = {
        "dilated": ndimage.binary_dilation(gt, iterations=8),
        "shifted": np.roll(gt, 60, axis=1),
        "empty": np.zeros_like(gt),
    }
    out = {"bus_pixel_share": round(float(gt.mean()), 3)}
    for name, p in preds.items():
        out[name] = scores(p, gt)
        overlay(p, gt, name)
    save_json("segmentation", out)
    print(f"bus covers {out['bus_pixel_share']:.1%} of the pixels")
    for name in preds:
        print(f"{name:8s} {out[name]}")


if __name__ == "__main__":
    main()
