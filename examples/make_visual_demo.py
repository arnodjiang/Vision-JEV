"""Generate independent synthetic images for integration smoke testing, not evaluation."""

import json
from pathlib import Path

from PIL import Image, ImageDraw

out = Path("outputs/visual-demo").resolve()
out.mkdir(parents=True, exist_ok=True)
for split, revenue, profit in [("train", 12, 3), ("eval", 19, 4)]:
    image = Image.new("RGB", (320, 160), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 10), "Financial summary", fill="black", font_size=20)
    draw.rectangle((15, 45, 305, 145), outline="black", width=2)
    draw.line((15, 95, 305, 95), fill="black", width=2)
    draw.line((170, 45, 170, 145), fill="black", width=2)
    draw.text((25, 60), "Revenue", fill="black", font_size=20)
    draw.text((185, 60), f"{revenue} USD", fill="black", font_size=20)
    draw.text((25, 110), "Profit", fill="black", font_size=20)
    draw.text((185, 110), f"{profit} USD", fill="black", font_size=20)
    image.save(out / f"{split}.png")
    record = {
        "id": f"visual-{split}",
        "group_id": f"synthetic-{split}",
        "image": f"{split}.png",
        "task": "extract",
        "question": "What is the revenue?" if split == "train" else "What is the profit?",
        "candidates": [
            {
                "id": "revenue",
                "text": f"{revenue} USD",
                "value": revenue,
                "bbox": [0.53, 0.28, 0.95, 0.59],
            },
            {
                "id": "profit",
                "text": f"{profit} USD",
                "value": profit,
                "bbox": [0.53, 0.59, 0.95, 0.91],
            },
            {"id": "none", "text": "Not present", "is_null": True},
        ],
        "target": "revenue" if split == "train" else "profit",
    }
    (out / f"{split}.jsonl").write_text(json.dumps(record) + "\n")
print(out)
