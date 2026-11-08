"""Prompt-matching metric: CLIPScore (Hessel et al., EMNLP 2021).

CLIP-S(c, v) = 2.5 * max(cos(c, v), 0), with CLIP ViT-B/32 and the caption
prefixed by "A photo depicts", as in the paper's reference code.

Works case: a matching caption scores far above an unrelated one.
Fooled case: swapping who does what ("the horse is eating the grass" vs
"the grass is eating the horse") or changing a count barely moves the score.
"""
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from common import IMAGES, save_json

MODEL_ID = "openai/clip-vit-base-patch32"
CASES = {
    "horse_grey.jpg": ["a horse grazing in a meadow", "a red bus on a city street",
                       "the horse is eating the grass", "the grass is eating the horse"],
    "dogs_beach.jpg": ["two dogs playing on a beach", "three dogs playing on a beach",
                       "one dog playing on a beach"],
    "school_bus.jpg": ["a yellow school bus", "a blue school bus"],
}

model = CLIPModel.from_pretrained(MODEL_ID).eval()
processor = CLIPProcessor.from_pretrained(MODEL_ID)


@torch.no_grad()
def clip_scores(image_path, captions):
    image = Image.open(image_path).convert("RGB")
    texts = [f"A photo depicts {c}" for c in captions]
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
    img = model.get_image_features(pixel_values=inputs["pixel_values"])
    txt = model.get_text_features(input_ids=inputs["input_ids"],
                                  attention_mask=inputs["attention_mask"])
    img = img / img.norm(dim=-1, keepdim=True)
    txt = txt / txt.norm(dim=-1, keepdim=True)
    cos = (txt @ img.T).squeeze(-1)
    return [round(float(2.5 * max(c, 0)), 4) for c in cos]


def main():
    out = {}
    for name, captions in CASES.items():
        out[name] = dict(zip(captions, clip_scores(IMAGES / name, captions)))
        print(name)
        for c, s in out[name].items():
            print(f"  {s:.4f}  {c}")
    save_json("generation_clipscore", out)


if __name__ == "__main__":
    main()
