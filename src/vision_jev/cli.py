import argparse
import hashlib
import json
import random
import time
from pathlib import Path

from .schema import assert_disjoint, read_records


def main():
    parser = argparse.ArgumentParser(description="Vision-JEV research prototype")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--data", required=True)
    train.add_argument(
        "--eval-data", required=True, help="Held-out labelled records; group leakage checked"
    )
    train.add_argument("--output", required=True)
    train.add_argument("--base", default="Qwen/Qwen3.5-0.8B")
    train.add_argument("--revision", default=None)
    train.add_argument("--mode", choices=["head", "lora"], default="lora")
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--lr", type=float, default=2e-4)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--shuffle-candidates", action="store_true")
    train.add_argument("--shuffle-fields", action="store_true")
    predict = sub.add_parser("predict")
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--data", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument(
        "--threshold", type=float, default=0.0, help="Uncalibrated probability threshold"
    )
    merge = sub.add_parser("merge", help="Merge trained LoRA into the multimodal backbone")
    merge.add_argument("--checkpoint", required=True)
    merge.add_argument("--output", required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--predictions", required=True)
    for p in (train, predict):
        p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
        p.add_argument("--max-length", type=int, default=8192)
    args = parser.parse_args()
    if args.command == "evaluate":
        from .metrics import score

        records = read_records(args.data, True)
        predictions = [
            json.loads(s) for s in Path(args.predictions).read_text().splitlines() if s.strip()
        ]
        print(json.dumps(score(records, predictions), indent=2))
        return
    import torch
    from transformers import AutoProcessor

    from .model import VisionJEV
    from .processing import RecordProcessor

    if args.command == "merge":
        if Path(args.output).exists():
            parser.error("Output already exists")
        model = VisionJEV.load(args.checkpoint)
        if hasattr(model.backbone, "merge_and_unload"):
            model.backbone = model.backbone.merge_and_unload()
        processor = AutoProcessor.from_pretrained(Path(args.checkpoint) / "processor")
        config = json.loads((Path(args.checkpoint) / "vision_jev.json").read_text())
        metadata = dict(config.get("metadata", {}))
        metadata["merged_from"] = str(Path(args.checkpoint).resolve())
        model.save(args.output, processor, metadata)
        print(f"Saved merged Vision-JEV checkpoint to {args.output}")
        return
    if args.command == "train":
        if args.epochs < 1 or args.lr <= 0:
            parser.error("epochs and lr must be positive")
        records = read_records(args.data, True)
        heldout = read_records(args.eval_data, True)
        assert_disjoint(records, heldout)
        if Path(args.output).exists():
            parser.error("Output already exists; choose a new checkpoint directory")
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        processor = AutoProcessor.from_pretrained(args.base, revision=args.revision)
        model = VisionJEV.from_base(args.base, args.revision)
        model.configure_training(args.mode)
        model.to(args.device)
        process = RecordProcessor(processor, args.max_length)
        optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
        history = []
        for epoch in range(args.epochs):
            model.train()
            random.shuffle(records)
            losses = []
            for record in records:
                r = dict(record)
                r["candidates"] = list(record["candidates"])
                if "fields" in r:
                    r["fields"] = list(r["fields"])
                    if args.shuffle_fields:
                        random.shuffle(r["fields"])
                if args.shuffle_candidates:
                    random.shuffle(r["candidates"])
                batch = {k: v.to(args.device) for k, v in process(r, True).items()}
                optim.zero_grad(set_to_none=True)
                loss = model(**batch)["loss"]
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                losses.append(loss.item())
            model.eval()
            correct = 0
            total = 0
            with torch.inference_mode():
                for r in heldout:
                    batch = {k: v.to(args.device) for k, v in process(r, True).items()}
                    correct += (model(**batch)["logits"].argmax(-1) == batch["labels"]).sum().item()
                    total += batch["labels"].numel()
            result = {
                "epoch": epoch + 1,
                "train_loss": sum(losses) / len(losses),
                "heldout_candidate_accuracy": correct / total,
            }
            history.append(result)
            print(json.dumps(result), flush=True)
        metadata = {
            "base": args.base,
            "revision": args.revision,
            "resolved_revision": getattr(model.backbone.config, "_commit_hash", None),
            "mode": args.mode,
            "seed": args.seed,
            "history": history,
            "train_records": len(records),
            "eval_records": len(heldout),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "train_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
            "eval_sha256": hashlib.sha256(Path(args.eval_data).read_bytes()).hexdigest(),
            "torch": torch.__version__,
        }
        model.save(args.output, processor, metadata)
    else:
        if not 0 <= args.threshold <= 1:
            parser.error("threshold must be between 0 and 1")
        records = read_records(args.data)
        from .pipeline import VisionJEVPipeline

        pipeline = VisionJEVPipeline(args.checkpoint, args.device, args.max_length)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)

        def sync():
            if args.device == "cuda":
                torch.cuda.synchronize()
            elif args.device == "mps":
                torch.mps.synchronize()

        with output.open("x") as f, torch.inference_mode():
            for r in records:
                sync()
                start = time.perf_counter()
                result = pipeline(r, args.threshold)
                sync()
                result["latency_seconds"] = time.perf_counter() - start
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()


if __name__ == "__main__":
    main()
