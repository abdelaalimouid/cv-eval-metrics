#!/usr/bin/env bash
# Runs every example in order. Results go to results/*.json, figures to figures/.
# The first run downloads the FID Inception weights (~92 MB, GitHub) and
# CLIP ViT-B/32 (~600 MB, Hugging Face).
set -euo pipefail
cd "$(dirname "$0")/src"
python detection.py
python segmentation.py
python generation_toy.py
python generation_clipscore.py
python generation_fid.py
python make_damage_examples.py
