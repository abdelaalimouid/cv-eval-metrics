"""Shared paths and small helpers for every example script."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "data" / "images"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

# Deck palette, so the figures match the slides.
INK, PAPER, BLUE, ORANGE, GREY = "#15171C", "#F3F4F6", "#2F5BEA", "#F2541B", "#A3A9B5"


def save_json(name, data):
    """Write a result file to results/<name>.json and return its path."""
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(data, indent=2))
    return path
