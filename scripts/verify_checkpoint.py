"""Verify independent-row and merge consistency on user-supplied smoke inputs."""

import argparse
import json
from pathlib import Path

import torch

from vision_jev.pipeline import VisionJEVPipeline
from vision_jev.schema import read_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    records = read_records(args.data)
    pipe = VisionJEVPipeline(args.checkpoint, device=args.device)

    def logits(rows):
        batch = {k: v.to(args.device) for k, v in pipe.process.batch(rows).items()}
        return pipe.model(**batch)["logits"].cpu()

    with torch.inference_mode():
        together = logits(records)
        max_batch_error = 0.0
        for i, record in enumerate(records):
            single = logits([record])[0]
            joint = together[i, : len(record["candidates"])]
            torch.testing.assert_close(single, joint, atol=1e-4, rtol=1e-4)
            max_batch_error = max(max_batch_error, (single - joint).abs().max().item())
        if not hasattr(pipe.model.backbone, "merge_and_unload"):
            raise ValueError("Merge verification requires an adapter checkpoint")
        pipe.model.backbone = pipe.model.backbone.merge_and_unload()
        merged = logits(records)
        valid = torch.isfinite(together)
        torch.testing.assert_close(together[valid], merged[valid], atol=1e-4, rtol=1e-4)
    result = {
        "records": len(records),
        "batch_logit_max_abs_error": max_batch_error,
        "merge_logit_max_abs_error": (together[valid] - merged[valid]).abs().max().item(),
        "atol": 1e-4,
        "rtol": 1e-4,
        "scope": "Consistency only; no QA or speed claim",
    }
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
