"""Save one crop at every damage level used in generation_fid.py.

Output: figures/damage_examples.jpg, a 4x2 grid shown on the FID slide.
  top row:    clean, blur sigma 1, 2, 4
  bottom row: clean, noise sigma 15, 30, 60
"""
from PIL import Image

from common import FIGURES
from generation_fid import random_crops, blur, noise

TILE, GAP = 299, 12
crop = random_crops(1, seed=1)
rows = [[crop[0]] + [blur(crop, s)[0] for s in (1, 2, 4)],
        [crop[0]] + [noise(crop, s)[0] for s in (15, 30, 60)]]
grid = Image.new("RGB", (4 * TILE + 3 * GAP, 2 * TILE + GAP), (243, 244, 246))
for r, row in enumerate(rows):
    for c, im in enumerate(row):
        grid.paste(im, (c * (TILE + GAP), r * (TILE + GAP)))
grid.save(FIGURES / "damage_examples.jpg", quality=90)
print("saved", grid.size)
