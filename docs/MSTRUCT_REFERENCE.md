# MStructQA / MStructBench dataset reference

These notes record a read-only inspection on September 21, 2026. The source MVisQA project was not modified. Local paths below describe that inspected snapshot, not assets bundled with Vision-JEV.

- Local project root: `$MSTRUCT_ROOT`, a placeholder for a separate MStructQA checkout.
- Active snapshot: `data/visual_benchmark/final_128_24lang_v5_visual_types`.
- Public dataset: [arnodjiang/MStructBench](https://huggingface.co/datasets/arnodjiang/MStructBench).
- Local release manifest: `data/hf_publish/MStructBench_visual_types_v1/release.json`.
- Canonical references SHA256 declared by the manifest: `ab1cd0fa0e2dbf212edf43d883e56d996aa66f0cde84539244f576bb06f557f1`.

## Inspected counts

The snapshot contains 128 base IDs, 128 case IDs, 24 image languages, 3,072 PNG images, and 8,960 QA configurations. The base cases comprise 87 charts and 41 tables; answer types comprise 71 numeric, 55 short-text, and 2 list answers.

The complete `benchmark.jsonl` cohort contains 1,631 records marked `accepted` and 7,329 marked `needs_review`. The 1,631-record `validation_release/val.jsonl` is a filtered subset, not the complete 8,960-record public evaluation cohort. Automated acceptance does not imply human verification.

## Evaluation scope

Questions include cell lookup, differences between extrema, conditional filtering and ratio ranking, curve-intersection counting, and semantic explanations. Field lookup and constrained decisions are useful initial subsets, but evaluation must retain the full original cohort and report fallback behavior.

Only images, queries, and protocol-approved source context may enter inference. Reference answers, audits, provenance, and hidden structures in `render.py` are restricted to scoring or auditing; they must not become test candidates or model inputs.

Each case currently has one base QA. Its 70 language configurations are not 70 distinct fields. Multiple-field throughput experiments need a separate construction. External context assembly should follow the source project's `scripts.evaluation.context_input.input_text` to retain context required by some questions.

## Training boundary

Vision-JEV does not automatically convert MStructQA into candidate supervision. The original QA records lack candidate-pointer labels; synthetic labels must not be presented as original human annotations. Keep all 128 cases and their language variants out of training when reporting generalization on this evaluation set.
