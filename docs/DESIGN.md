# Vision-JEV design and experiment plan

## Scope

Adapt Qwen3.5-0.8B for non-generative, candidate-based multimodal information extraction and decisions. The component returns values and evidence references for downstream QA; it does not solve arbitrary open-ended VQA.

Inputs: image I, question q, candidate set C, and optional context x.
Outputs: P(c | I, q, C, x), candidate-provided values and evidence, and abstention status.

## Implemented model changes

Load `Qwen3_5Model` from the official checkpoint, retain its vision and language modules, and bypass the vocabulary output head. Let h_c denote a candidate-end hidden state and h_q the decision state after all candidates:

```text
score(c) = (Wq h_q + bq)^T (Wk h_c + bk) / sqrt(d)
```

Softmax normalizes over the request's candidates. Cross entropy supervises `target`. Rank-16 LoRA adapts language FFN gate/up/down projections across the hybrid layers; vision weights remain frozen by default. Head-only training provides an ablation.

Markers use existing reserved tokenizer tokens without expanding embeddings. Inputs containing special tokens are rejected to prevent forged candidate or decision positions. Oversized sequences raise an error rather than silently truncating images or supervision positions.

Independent questions run as batch rows, with candidates scored together and recurrent states isolated between rows. Repeated images are encoded repeatedly; shared-prefix inference is not implemented. Causal candidate order affects representations: equivariance of the head alone does not make the complete model order-invariant. Use shuffled-candidate training and evaluate permutation robustness.

## Integration with QA

1. An external OCR system, table parser, or fixed task vocabulary supplies candidates.
2. Vision-JEV selects a field or category and retains its evidence references.
3. The application executes known deterministic operations or passes the result to a QA model.
4. Null or low-confidence predictions trigger an application-provided QA fallback.

Values and bounding boxes come from the input candidates. They are not independently recognized strings or predicted boxes. OCR, box regression, learned calculation plans, and free-text translation require additional implementations and supervision.

## Dataset construction

- Freeze held-out source documents and all their translated or rendered variants for evaluation.
- Build training data from non-overlapping documents and independent synthetic charts. Split by original source/document identity, not just QA identifier.
- Candidate generation must not access reference answers or hidden rendering data. References may support label matching and annotation review, but must not fill gaps in test candidates.
- Include headers, cells, units, OCR spans, and evidence, as well as distractors, unanswerable cases, and cross-language labels.
- Report candidate recall, accuracy conditional on coverage, and end-to-end accuracy over all examples.
- Audit source identities upstream. The library's `group_id` checks detect known identifier overlap, not paraphrases or near-duplicates automatically.

## Experiments

Baselines: original Qwen3.5-0.8B, generative fine-tuning on matched data, OCR plus a text model, frozen backbone plus head, and LoRA plus head.

Quality metrics include end-to-end QA accuracy, numeric and unit correctness, candidate recall, candidate accuracy, and abstention coverage. Evidence accuracy needs separate human or structural labels. For multilingual evaluation, report results by language and distinguish matched-language from cross-language questions. Bootstrap by source document to account for correlated variants.

End-to-end latency must include image preprocessing, OCR, backbone, head, formatting, and fallback. Fix hardware, resolution, candidate count, batch size, precision, and warmup; report cold starts separately from warm runs. Current CLI timing starts at preprocessing and excludes model loading and external candidate generation, so it is not complete system latency.

Direct readout avoids token decoding, but candidate text adds prefill work. Benefits depend on input length, candidate-generation costs, and fallback frequency. No speedup has been established.

## Milestones and acceptance criteria

1. Prototype: verify visual forward passes, LoRA backward passes, and checkpoint recovery on the real Qwen3.5 hybrid architecture.
2. Task training: pin the official 0.8B revision, train on independent image data, evaluate, and release weights with a model card. Calibrate before selecting abstention thresholds.
3. Multiple fields: isolate full-attention KV caches and DeltaNet recurrent/convolution states at every layer while preserving multimodal positions. Require separate-versus-batched probability equivalence before merging cache optimizations.
4. Learned localization and native extraction: add evidence-localization, span, or set heads to reduce external OCR dependence, and report the additional supervision cost.

One-step synthetic-image training on the official 0.8B model is verified. Formal task training, evaluation, and weight release in milestone 2 remain incomplete; milestones 3 and 4 are not implemented. See [the validation record](VALIDATION.md).
