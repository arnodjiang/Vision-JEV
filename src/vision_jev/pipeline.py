"""Python API for trained checkpoints with independent question batches."""

from pathlib import Path

import torch
from transformers import AutoProcessor

from .model import VisionJEV
from .processing import RecordProcessor


def format_prediction(record, probabilities, threshold):
    index = max(range(len(probabilities)), key=probabilities.__getitem__)
    candidate = record["candidates"][index]
    abstained = probabilities[index] < threshold or candidate.get("is_null", False)
    return {
        "id": record["id"],
        "candidate_id": candidate["id"],
        "value": None if abstained else candidate.get("value", candidate["text"]),
        "evidence": {k: candidate[k] for k in ("bbox", "page", "text") if k in candidate},
        "probabilities": {c["id"]: p for c, p in zip(record["candidates"], probabilities)},
        "abstained": abstained,
        "calibrated": False,
    }


class VisionJEVPipeline:
    def __init__(self, checkpoint, device="cpu", max_length=8192):
        self.device = device
        processor = AutoProcessor.from_pretrained(Path(checkpoint) / "processor")
        self.process = RecordProcessor(processor, max_length)
        self.model = VisionJEV.load(checkpoint).to(device).eval()

    def __call__(self, record, threshold=0.0):
        return self.batch([record], threshold)[0]

    @torch.inference_mode()
    def batch(self, records, threshold=0.0):
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0,1]")
        batch = {k: v.to(self.device) for k, v in self.process.batch(records).items()}
        probs = self.model(**batch)["logits"].softmax(-1).cpu().tolist()
        return [
            format_prediction(r, p[: len(r["candidates"])], threshold)
            for r, p in zip(records, probs)
        ]
