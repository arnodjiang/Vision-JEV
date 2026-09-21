# Method

## Research question

Can structured extraction and direct decisions reduce the cost of multimodal QA while retaining answer quality across visual and query languages?

This is a hypothesis, not an established result. Version 0.1 implements the model component needed to test it.

## Inputs and outputs

Let I be an image, q a question, x optional inference-time context, and C a set of candidates. Each candidate has an identifier, text, and optionally a typed value and evidence metadata. The model predicts P(c | I, q, x, C).

An `extract` candidate might be an OCR span; a `choice` candidate is an answer category; a `boolean` request contains true and false candidates. A null candidate can represent no supported extraction. Null selection and threshold abstention are explicit in the response.

## Backbone and readout

Qwen3.5's native visual encoder and hybrid language backbone produce token representations. Candidate text ends with an existing reserved marker. A final decision marker follows the question and all candidates.

For candidate representation h_c and decision representation h_q:

```text
score(c) = (Wq h_q + bq)^T (Wk h_c + bk) / sqrt(d)
p(c)     = softmax_C(score(c))
loss     = -log p(target)
```

Both projections are trainable. The model bypasses the vocabulary output head and does not call `generate()`. The implementation uses existing tokenizer tokens, validates marker counts, rejects special-token injection, and fails explicitly instead of truncating oversized examples.

LoRA adapts language FFN `gate_proj`, `up_proj`, and `down_proj` modules across the hybrid stack. Vision weights and the remaining backbone weights are frozen in the default training recipe. `--mode head` provides a frozen-backbone control.

## Independent question batches

Each question occupies its own padded batch row. This preserves independent recurrent state in DeltaNet layers without a custom attention mask. Standard attention-only block masking would not, by itself, isolate recurrent states.

Candidate scores are returned together, but questions do not share a computed prefix in v0.1. Images repeated across questions are encoded repeatedly. Within a question, causal candidate representations depend on candidate order; the complete model is not permutation invariant.

## QA integration

An application supplies candidates, calls Vision-JEV, and routes the structured result to deterministic logic or a QA model. A null or low-confidence prediction can trigger a fallback. Version 0.1 does not implement the candidate generator, calculation planner, or fallback service.

This makes candidate recall an upper bound on extraction coverage. Evidence is selected from supplied metadata; the model does not independently predict bounding boxes or arbitrary strings. Cross-language answer rendering also requires a candidate value mapping or downstream renderer.

## Where efficiency could come from

A direct readout avoids autoregressive answer decoding and can batch independent decisions. However, extra candidate text increases prefill work, image encoding remains, and candidate generation or fallback may dominate total latency. Speedup must therefore be measured with all these costs included.

Current outputs are raw softmax probabilities, not calibrated correctness estimates. Calibration and abstention policies require a separate development set. No speed, calibration, or generalization improvement is claimed by the smoke tests.
