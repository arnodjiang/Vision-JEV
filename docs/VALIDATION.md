# Validation record

Development checks performed on 2026-09-21. This record separates implementation verification from task evaluation.

## Environment

Python 3.12.14, PyTorch 2.14.0, Transformers 5.17.0, PEFT 0.21.0, macOS ARM64 CPU, float32. The original environment is recorded in `requirements-tested.txt`. CUDA DeltaNet kernels were not installed; these runs do not establish serving performance.

## Automated tests

```bash
VISION_JEV_PROCESSOR=outputs/qwen-processor pytest -q
```

**20 tests passed** during repository preparation. Coverage includes malformed input rejection, source-group leakage checks, candidate masking and gradients, missing predictions and abstention metrics, invalid probability rejection, actual hybrid-layer LoRA backward, visual forward, checkpoint recovery, adapter merging, independent-row padding consistency, mixed head/input precision, and real processor integration.

Without the processor environment variable, three optional processor tests skip. The remaining tests use tiny instances of the actual Qwen3.5 architecture and require no downloaded weights. Ruff checks, formatting checks, documentation links, and wheel construction also passed locally. Hosted GitHub Actions results are reported separately by the workflow badge.

## Official Qwen3.5-0.8B integration

Pinned revision: `2fc06364715b967f1860aea9cf38778875588b17`.

| Check | Observed result |
| --- | --- |
| Training data | One synthetic table image |
| Validation data | One separate synthetic table image |
| Optimization | One rank-16 LoRA + head update |
| Trainable parameters | 5,570,816 |
| Total parameters including adapters | 858,556,736 |
| Updated LoRA B matrices | 72 / 72 nonzero |
| Training loss | 0.4418438673 |
| Held-out candidate accuracy | **0 / 1** |
| Held-out NLL | Approximately 24.7364 |
| Save and reload | Separate-process image prediction succeeded |

This checkpoint verifies parameter updates and I/O. It is not a useful trained KIE model and provides no evidence of QA improvement, calibration, or speedup.

The first real-weight attempt exposed BF16 backbone / FP32 head incompatibility. Default loading is now explicitly FP32, readout inputs match the head dtype, and a regression test covers mixed precision.

## Raw-logit consistency

The checked-in `scripts/verify_checkpoint.py` was run on both synthetic images together:

| Comparison | Maximum absolute logit difference |
| --- | ---: |
| Individual vs independent batch rows | 1.239776611328125e-05 |
| Before vs after in-memory LoRA merge | 1.18255615234375e-04 |

Both comparisons passed `torch.testing.assert_close` with **atol=1e-4 and rtol=1e-4**; this is a combined absolute/relative tolerance, not an absolute-only bound. The observations apply to these two examples, not all possible inputs. Earlier probability-only checks also passed, but saturated softmax values can obscure differences; raw-logit checks are the preferred reproduction route.

## Reproduce

Follow [EXPERIMENTS.md](EXPERIMENTS.md) for downloading the official processor, training the synthetic fixture, saving a checkpoint, and running the consistency script. Use new output paths for each run. Only set `HF_HUB_OFFLINE=1` after the selected model revision and processor have been cached.

Generated checkpoints, images, predictions and local reports are excluded from Git. Run metadata records dataset-file hashes, base revision, training mode, seed, parameter counts and loss history. Group-ID validation does not replace a source-provenance audit.

## Unverified research outcomes

Task-ready KIE training, held-out QA evaluation, candidate generation, learned localization, calibration, shared visual-prefix caching, and end-to-end quality–latency comparisons remain future work.

## Shared-document multi-field update

The multi-field implementation passes 24 tests with the local official processor assets enabled. Added checks cover two field readouts in one hybrid-backbone forward, backward gradients, field-level missing-prediction accounting, checkpoint restoration, one image grid per document, and target-label isolation. Tests use a tiny randomly initialized Qwen3.5 backbone, not trained task weights.

An additional offline run using the previous 0.8B smoke adapter could not load its base weights because the required Hugging Face cache snapshot was unavailable. Consequently, this update does not claim a completed official-0.8B multi-field inference run, task quality, or measured latency improvement.
