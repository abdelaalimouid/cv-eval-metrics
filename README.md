# Evaluation metrics for computer vision: detection, segmentation, generation

Code for the homework *Evaluation Metrics for Computer Vision Tasks* (Master Data & IA, ML Evaluation, 2026).
Each script computes the metrics of one task on real, freely licensed photos and shows two cases:
one where the metric **works** and one where it is **fooled**. Every number in the presentation comes from
these scripts.

| Task | Metrics | Script |
|---|---|---|
| Object detection | IoU, AP@0.5, AP@0.75, COCO mAP@[.5:.95] | `src/detection.py` |
| Semantic segmentation | pixel accuracy, IoU, Dice, mean IoU | `src/segmentation.py` |
| Image generation: realism and variety | FID, KID, precision and recall | `src/generation_fid.py` |
| Image generation: prompt match | CLIPScore | `src/generation_clipscore.py` |
| Image generation: small worked examples | Elo update, precision fooled by outliers | `src/generation_toy.py` |

## Run it

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./run_all.sh
```

The whole run takes about 5 minutes on a laptop CPU; no GPU is needed. The first run downloads two sets of
pretrained weights: the standard FID Inception-v3 (about 92 MB, from the pytorch-fid GitHub release) and
CLIP ViT-B/32 (about 600 MB, `openai/clip-vit-base-patch32` on Hugging Face).

Each script writes its numbers to `results/<name>.json` and its pictures to `figures/`.

## Layout

```
data/images/        six public-domain or CC0 photos, credits in data/images/CREDITS.md
src/common.py       paths, colours and save_json()
src/detection.py
src/segmentation.py
src/generation_fid.py
src/generation_clipscore.py
src/generation_toy.py
src/make_damage_examples.py   picture of the damage levels used by generation_fid.py
results/            JSON output of every script
figures/            images used on the slides
run_all.sh
```

---

## 1. Object detection: `src/detection.py`

**Data.** `dogs_beach.jpg` (1280 x 772). Two ground-truth boxes, one per dog, drawn by hand.
Two made-up detectors, each with a box per dog plus one false alarm on the sand:

* *tight*: boxes close to the truth
* *sloppy*: boxes much larger than the dogs, just above IoU 0.5

**Metrics.**

* `iou(a, b)`: intersection area divided by union area of two `(x1, y1, x2, y2)` boxes.
* `precision_recall(preds, gts, thr)`: sorts detections by confidence and matches each one greedily to the
  unmatched ground-truth box with the highest IoU; a match counts if IoU ≥ `thr`.
* `average_precision(...)`: COCO-style 101-point interpolated AP. The precision curve is made monotone from the
  right, then sampled at recall 0, 0.01, ..., 1.
* `coco_map(...)`: AP averaged over IoU thresholds 0.50, 0.55, ..., 0.95.

**Results** (`results/detection.json`):

| Detector | IoU dog 1 / dog 2 | AP50 | AP75 | mAP@[.5:.95] |
|---|---|---|---|---|
| tight | 0.921 / 0.926 | 1.00 | 1.00 | 0.90 |
| sloppy | 0.586 / 0.525 | 1.00 | 0.00 | 0.15 |

**Works:** tight boxes get top marks at every threshold.
**Fooled:** AP50 gives the sloppy detector the same perfect score. Only stricter thresholds separate them,
which is why COCO reports mAP over IoU 0.5 to 0.95.

Figures: `figures/detection_tight.jpg`, `figures/detection_sloppy.jpg` (blue = ground truth, orange = prediction).

## 2. Segmentation: `src/segmentation.py`

**Data.** `school_bus.jpg` (1280 x 867). The ground-truth mask is a polygon traced by hand around the bus
(`BUS_POLYGON`); the bus covers 14.7% of the pixels. Three predictions:

* *dilated*: the true mask grown by 8 pixels
* *shifted*: the true mask moved 60 pixels to the right
* *empty*: background everywhere

**Metrics.** For the bus class: pixel accuracy = share of pixels labelled correctly;
IoU = |A ∩ B| / |A ∪ B|; Dice = 2|A ∩ B| / (|A| + |B|) = 2 IoU / (1 + IoU).
Mean IoU averages the IoU of the bus and the background classes.

**Results** (`results/segmentation.json`):

| Prediction | Pixel accuracy | IoU (bus) | Dice (bus) | mIoU |
|---|---|---|---|---|
| dilated 8 px | 0.988 | 0.922 | 0.959 | 0.954 |
| shifted 60 px | 0.956 | 0.741 | 0.851 | 0.845 |
| empty | 0.853 | 0.000 | 0.000 | 0.426 |

**Works:** IoU and Dice drop clearly when the mask is shifted; pixel accuracy hardly moves.
**Fooled:** predicting nothing still gives 85.3% pixel accuracy, because most pixels are background.

Figures: `figures/segmentation_*.jpg` (blue outline = ground truth, orange = prediction).

## 3. Image generation, realism and variety: `src/generation_fid.py`

There is no generator in this repo. Following Heusel et al. (2017), real images stand in for generated
ones and are then damaged on purpose, so we know which set should score worse.

**Data.** 200 random 299 x 299 crops of the six photos (seed 0) form the *real* set. 200 other crops
(seed 1) form the *generated* set, which is then blurred (Gaussian, σ = 1, 2, 4 px) or noised
(Gaussian, σ = 15, 30, 60 on a 0 to 255 scale). `figures/damage_examples.jpg` shows one crop at each level.

**Metrics.** All use the 2048-d pool3 features of the FID Inception-v3 network (pytorch-fid).

* FID: Fréchet distance between Gaussians fitted to both feature sets,
  ‖μr − μg‖² + Tr(Σr + Σg − 2(Σr Σg)^½).
* KID: unbiased squared MMD with the cubic polynomial kernel k(x, y) = (x·y / d + 1)³ (Bińkowski et al., 2018),
  reported × 1000.
* Precision and recall (Kynkäänniemi et al., 2019): the share of generated samples inside the k = 3
  nearest-neighbour balls of the real samples, and the other way round.

**Results** (`results/generation_fid.json`), 200 images per set:

| Damage | FID | KID × 1000 | Precision | Recall |
|---|---|---|---|---|
| clean | 25.6 | −0.98 | 0.965 | 0.910 |
| blur σ 1 | 54.6 | 8.69 | 0.305 | 0.310 |
| blur σ 2 | 119.6 | 35.58 | 0.015 | 0.050 |
| blur σ 4 | 197.3 | 70.72 | 0.000 | 0.000 |
| noise σ 15 | 78.2 | 18.05 | 0.160 | 0.235 |
| noise σ 30 | 137.4 | 45.70 | 0.015 | 0.025 |
| noise σ 60 | 253.9 | 119.87 | 0.000 | 0.000 |

Same undamaged crops, fewer images per set:

| Images per set | FID | KID × 1000 |
|---|---|---|
| 25 | 81.8 | −9.31 |
| 50 | 63.5 | −3.54 |
| 100 | 39.9 | −1.03 |
| 200 | 25.6 | −0.98 |

**Works:** all four metrics move monotonically with the damage.
**Fooled:** FID depends on the number of images. 25 clean crops (81.8) look worse than 200 crops blurred at
σ 1 (54.6). FID's estimate is biased at small sample sizes (Chong & Forsyth, 2020); KID is unbiased and stays
near zero (a negative value is normal for an unbiased estimator of a quantity that is zero).

Caveat: two sets of 200 crops from the same six photos overlap heavily, so absolute values are not comparable
with published FIDs (which use 10k to 50k images). Only the trends matter here.

## 4. Image generation, prompt match: `src/generation_clipscore.py`

**Metric.** CLIPScore (Hessel et al., 2021): CLIP-S = 2.5 · max(cos(image, caption), 0), with CLIP ViT-B/32
and each caption prefixed by “A photo depicts”, as in the paper's reference code.

**Results** (`results/generation_clipscore.json`):

| Image | Caption | CLIP-S |
|---|---|---|
| horse_grey | a horse grazing in a meadow | 0.798 |
| horse_grey | a red bus on a city street | 0.314 |
| horse_grey | the horse is eating the grass | 0.770 |
| horse_grey | **the grass is eating the horse** | **0.789** |
| dogs_beach | two dogs playing on a beach | 0.815 |
| dogs_beach | three dogs playing on a beach | 0.810 |
| dogs_beach | one dog playing on a beach | 0.746 |
| school_bus | a yellow school bus | 0.812 |
| school_bus | **a blue school bus** | **0.835** |

**Works:** a matching caption scores far above an unrelated one.
**Fooled:** CLIP behaves like a bag of words: the swapped sentence scores higher than the true one, “two”
and “three” dogs differ by 0.005, and “blue” beats “yellow” for an orange-yellow bus.

## 5. Small worked examples: `src/generation_toy.py`

* **Precision fooled by outliers** (Naeem et al., 2020). In 2-D, 200 real points around the origin and
  200 generated points, half of them a bad cluster far away. Precision is 0.50. Adding three real outliers
  next to the bad cluster gives them large k-NN balls, and precision becomes 1.00 with the same fakes.
* **Elo update** used by arenas: two models at 1000, expected score 0.5 each; with K = 32, the winner goes
  to 1016 and the loser to 984.

---

## Sources

* Lin et al., *Microsoft COCO: Common Objects in Context*, ECCV 2014.
* Cheng et al., *Boundary IoU*, CVPR 2021.
* Heusel et al., *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium* (FID), NeurIPS 2017.
* Bińkowski et al., *Demystifying MMD GANs* (KID), ICLR 2018.
* Chong & Forsyth, *Effectively Unbiased FID and Inception Score and Where to Find Them*, CVPR 2020.
* Kynkäänniemi et al., *Improved Precision and Recall Metric for Assessing Generative Models*, NeurIPS 2019.
* Kynkäänniemi et al., *The Role of ImageNet Classes in Fréchet Inception Distance*, ICLR 2023.
* Naeem et al., *Reliable Fidelity and Diversity Metrics for Generative Models*, ICML 2020.
* Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP), ICML 2021.
* Hessel et al., *CLIPScore: A Reference-free Evaluation Metric for Image Captioning*, EMNLP 2021.
* Lin et al., *Evaluating Text-to-Visual Generation with Image-to-Text Generation* (VQAScore), ECCV 2024.

Photos: see `data/images/CREDITS.md`. All are public domain or CC0.
