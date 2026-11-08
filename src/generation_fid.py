"""Distribution metrics for image generation: FID, KID, precision and recall.

There is no generator here. As in Heusel et al. (2017), we stand in a "generated"
set with real image crops and then damage them on purpose:
  * real set R        : 200 random 299x299 crops of the six photos (seed 0)
  * "generated" set G : 200 other crops of the same photos (seed 1)
  * G is then blurred or noised at increasing strength.
A good judge should rise as the damage rises (works case).

Fooled case: FID of two undamaged sets drawn from the same photos changes with
the number of images alone, because the Frechet estimate is biased at small N.
KID is unbiased and stays near zero.

Features: the standard FID Inception-v3 pool3 layer (2048-d) from pytorch-fid.
"""
import numpy as np
import torch
from PIL import Image, ImageFilter
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import IMAGES, FIGURES, INK, BLUE, ORANGE, GREY, save_json

torch.manual_seed(0)
N, CROP = 200, 299
PHOTOS = sorted(IMAGES.glob("*.jpg"))


def random_crops(n, seed):
    """n random CROPxCROP crops from the photos, each resized so its short side is 400."""
    rng = np.random.default_rng(seed)
    images = []
    for p in PHOTOS:
        im = Image.open(p).convert("RGB")
        s = 400 / min(im.size)
        images.append(im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS))
    crops = []
    for _ in range(n):
        im = images[rng.integers(len(images))]
        x = rng.integers(0, im.width - CROP + 1)
        y = rng.integers(0, im.height - CROP + 1)
        crops.append(im.crop((x, y, x + CROP, y + CROP)))
    return crops


def blur(crops, sigma):
    return [c.filter(ImageFilter.GaussianBlur(sigma)) if sigma else c for c in crops]


def noise(crops, sigma, seed=2):
    rng = np.random.default_rng(seed)
    out = []
    for c in crops:
        a = np.asarray(c).astype(float) + rng.normal(0, sigma, (CROP, CROP, 3))
        out.append(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)))
    return out


MODEL = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).eval()


@torch.no_grad()
def features(crops):
    feats = []
    for i in range(0, len(crops), 50):
        batch = np.stack([np.asarray(c) for c in crops[i:i + 50]]).astype(np.float32) / 255
        x = torch.from_numpy(batch).permute(0, 3, 1, 2)
        feats.append(MODEL(x)[0].squeeze(-1).squeeze(-1).numpy())
    return np.concatenate(feats).astype(np.float64)


def fid(a, b):
    return float(calculate_frechet_distance(a.mean(0), np.cov(a, rowvar=False),
                                            b.mean(0), np.cov(b, rowvar=False)))


def kid(a, b):
    """Unbiased MMD^2 with the cubic polynomial kernel (Binkowski et al., 2018)."""
    d = a.shape[1]
    k = lambda x, y: (x @ y.T / d + 1) ** 3
    kaa, kbb, kab = k(a, a), k(b, b), k(a, b)
    m, n = len(a), len(b)
    return float((kaa.sum() - np.trace(kaa)) / (m * (m - 1))
                 + (kbb.sum() - np.trace(kbb)) / (n * (n - 1)) - 2 * kab.mean())


def knn_radii(x, k=3):
    d = np.linalg.norm(x[:, None] - x[None], axis=-1)
    return np.sort(d, axis=1)[:, k]  # column 0 is the point itself


def precision_recall(real, fake, k=3):
    """Kynkaanniemi et al. (2019): share of samples inside the other set's k-NN balls."""
    d = np.linalg.norm(real[:, None] - fake[None], axis=-1)  # real x fake
    precision = float((d <= knn_radii(real, k)[:, None]).any(axis=0).mean())
    recall = float((d <= knn_radii(fake, k)[None, :]).any(axis=1).mean())
    return precision, recall


def main():
    real = features(random_crops(N, seed=0))
    gen_crops = random_crops(N, seed=1)
    out = {"n_images": N, "blur": [], "noise": [], "sample_size": []}

    for sigma in [0, 1, 2, 4]:
        g = features(blur(gen_crops, sigma))
        p, r = precision_recall(real, g)
        out["blur"].append({"sigma": sigma, "fid": round(fid(real, g), 1),
                            "kid_x1000": round(1000 * kid(real, g), 2),
                            "precision": round(p, 3), "recall": round(r, 3)})
    for sigma in [0, 15, 30, 60]:
        g = features(noise(gen_crops, sigma))
        p, r = precision_recall(real, g)
        out["noise"].append({"sigma": sigma, "fid": round(fid(real, g), 1),
                             "kid_x1000": round(1000 * kid(real, g), 2),
                             "precision": round(p, 3), "recall": round(r, 3)})

    clean = features(gen_crops)
    for n in [25, 50, 100, 200]:
        out["sample_size"].append({"n": n, "fid": round(fid(real[:n], clean[:n]), 1),
                                   "kid_x1000": round(1000 * kid(real[:n], clean[:n]), 2)})
    save_json("generation_fid", out)

    for key in ("blur", "noise", "sample_size"):
        for row in out[key]:
            print(key, row)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), dpi=150)
    for ax, key, label in [(axes[0], "blur", "Gaussian blur sigma (px)"),
                           (axes[1], "noise", "Gaussian noise sigma (0-255)")]:
        xs = [r["sigma"] for r in out[key]]
        ax.plot(xs, [r["fid"] for r in out[key]], "-o", color=BLUE, lw=3)
        ax.set_xlabel(label)
        ax.set_ylabel("FID")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "fid_vs_damage.png")

    fig, ax = plt.subplots(figsize=(5, 3.6), dpi=150)
    ns = [r["n"] for r in out["sample_size"]]
    ax.plot(ns, [r["fid"] for r in out["sample_size"]], "-o", color=ORANGE, lw=3)
    ax.set_xlabel("images per set (no damage)")
    ax.set_ylabel("FID")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "fid_vs_sample_size.png")


if __name__ == "__main__":
    main()
