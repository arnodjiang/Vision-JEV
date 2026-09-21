# Reproduction and experimental protocol

## 1. Offline correctness tests

Install the project with its development dependencies, then run:

```bash
ruff check src tests scripts examples
ruff format --check src tests scripts examples
pytest -q
python -m pip wheel --no-deps --wheel-dir dist .
```

The default test run downloads no models. Processor integration tests skip unless `VISION_JEV_PROCESSOR` points to a local processor directory.

## 2. Official processor checks

The following downloads processor/tokenizer assets, not full model weights:

```bash
python - <<'PY'
from transformers import AutoProcessor
p = AutoProcessor.from_pretrained(
    "Qwen/Qwen3.5-0.8B",
    revision="2fc06364715b967f1860aea9cf38778875588b17",
)
p.save_pretrained("outputs/qwen-processor")
PY
VISION_JEV_PROCESSOR=outputs/qwen-processor pytest -q
```

## 3. Official-model smoke test

Run the visual fixture training, prediction, and merge commands in the README. Use fresh output paths. Check that training loss is finite, adapters change, and the saved checkpoint loads in a separate process.

To compare raw logits for individual requests, independent batch rows, and an in-memory LoRA merge:

```bash
cat outputs/visual-demo/train.jsonl outputs/visual-demo/eval.jsonl \
  > outputs/visual-demo/check.jsonl
python scripts/verify_checkpoint.py --checkpoint checkpoints/demo \
  --data outputs/visual-demo/check.jsonl --report outputs/consistency.json
```

The comparison uses raw logits so that saturated softmax probabilities cannot conceal differences. It checks only the supplied examples. The original development smoke test used one training and one held-out image and achieved 0/1 held-out candidate accuracy: it establishes implementation functionality, not model quality.

## 4. Training data and model selection

Build a separate training corpus from original documents disjoint from the frozen evaluation set. Preserve source IDs across translations, visual variants, and question variants. CLI `group_id` checks cannot discover unlabelled duplicates; audit provenance separately.

Generate candidates before looking at evaluation answers. Annotate candidate targets and evidence independently. Include challenging distractors, missing answers, units, script diversity, and candidate-order changes. Use a development split for hyperparameters, calibration, and abstention thresholds; do not tune on the held-out test set.

## 5. Baselines and ablations

| Comparison | Question |
| --- | --- |
| Original Qwen3.5-0.8B generative QA | How does the component compare with the unchanged backbone? |
| Generative fine-tuning on matched data | Is any gain due to supervision or the readout? |
| OCR plus text-only decisions | What does the image contribute? |
| Frozen backbone plus head | What does LoRA contribute? |
| LoRA plus head | Does adaptation improve extraction? |
| Candidate-order permutations | How sensitive is the system to order? |
| System with and without fallback | How do coverage and quality trade off? |

Match input information, training data, image resolution, precision, hardware, and candidate-generation access. Do not compare a cached multi-query run against uncached single-query generation and report only the favorable ratio.

## 6. Metrics

Report full-denominator QA accuracy, candidate recall, selection accuracy conditional on candidate availability, numeric/unit correctness, abstention coverage, and selective accuracy. Evaluate evidence only where localization labels exist. Calibrate on a development set before reporting calibration-based guarantees.

Measure image processing, candidate generation, prefill, readout, postprocessing, and fallback in end-to-end latency. Separate cold loading, warm requests, repeated-image cache conditions, and batch sizes. Report P50/P95, throughput, peak device memory, and the frequency/cost of fallback. The CLI's `latency_seconds` excludes model loading and external candidate generation and is not a complete system speed measurement.

Use paired comparisons and resample original source cases, not individual correlated language variants. For multilingual tasks, report per-language results and document cross-language aggregation weights. Multi-field throughput experiments require multiple distinct questions per source image; translations of one question do not constitute distinct fields.

## 7. Reporting gate

A future paper should publish frozen data manifests, model revisions, prompts, candidate-generation configuration, training seeds, excluded cases, hardware/runtime details, and confidence intervals. The current repository contains no benchmark ranking or measured QA-efficiency claim.
