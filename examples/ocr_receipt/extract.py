"""Extract 2–6 requested keys from an OCR document using a trained checkpoint."""

import argparse
import json
import re
from pathlib import Path


def make_records(ocr, keys, image):
    if not 2 <= len(keys) <= 6 or len(set(keys)) != len(keys):
        raise ValueError("Supply 2–6 unique keys")
    if any(not isinstance(k, str) or not k.strip() for k in keys):
        raise ValueError("Keys must be nonempty strings")
    width, height = ocr["width"], ocr["height"]
    candidates = []
    for span in ocr["spans"]:
        text = span["text"]
        # Parse only visible OCR text, never annotations or requested target labels.
        parts = re.split(r":\s*|\s{2,}", text, maxsplit=1)
        value = parts[-1] if len(parts) == 2 else text
        candidates.append(
            {
                "id": span["id"],
                "text": text,
                "value": value,
                "bbox": [v / (width if j % 2 == 0 else height) for j, v in enumerate(span["bbox"])],
            }
        )
    candidates.append({"id": "none", "text": "Not present in the document", "is_null": True})
    return [
        {
            "id": f"field-{i}",
            "group_id": "ocr-document",
            "image": str(image),
            "task": "extract",
            "question": f"Extract the field: {key}. Select not present if absent.",
            "candidates": candidates,
        }
        for i, key in enumerate(keys)
    ]


def format_fields(keys, predictions):
    return {
        "source": "model",
        "fields": {
            key: {
                "value": p["value"],
                "confidence": p["probabilities"][p["candidate_id"]],
                "confidence_kind": "uncalibrated_candidate_probability",
                "calibrated": False,
                "abstained": p["abstained"],
                "candidate_id": p["candidate_id"],
                "evidence": p["evidence"],
                "probabilities": p["probabilities"],
            }
            for key, p in zip(keys, predictions, strict=True)
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocr", type=Path, default=Path(__file__).with_name("ocr.json"))
    parser.add_argument("--keys", nargs="+", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    ocr = json.loads(args.ocr.read_text())
    records = make_records(ocr, args.keys, (args.ocr.parent / ocr["image"]).resolve())
    from vision_jev.schema import validate

    for record in records:
        validate(record)
    if args.prepare_only:
        print(json.dumps({"keys": args.keys, "requests": records}, indent=2))
        return
    if not args.checkpoint:
        parser.error("--checkpoint is required for inference; use --prepare-only to inspect inputs")
    from vision_jev.pipeline import VisionJEVPipeline

    pipeline = VisionJEVPipeline(args.checkpoint, device=args.device)
    document = {k: v for k, v in records[0].items() if k not in ("question", "task")}
    document["fields"] = [
        {"key": key, "question": r["question"]} for key, r in zip(args.keys, records, strict=True)
    ]
    print(json.dumps(pipeline(document, threshold=args.threshold), indent=2))


if __name__ == "__main__":
    main()
