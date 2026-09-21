# Contributing

Use English for repository documentation, code comments, issues, and pull requests. Multilingual task data may retain its original language. For bug reports, include reproduction steps, package versions, and the full error message with sensitive information removed. For substantial changes, open an issue to discuss the proposed behavior and evaluation before implementation.

Use Python 3.11+ and install `python -m pip install -e '.[dev]'`.

Before submitting a change:

```bash
ruff format src tests scripts examples
ruff check src tests scripts examples
pytest -q
```

Three processor tests are optional and require downloaded tokenizer/processor assets; see [EXPERIMENTS.md](docs/EXPERIMENTS.md). New behavior should have a focused regression test. Explain the concrete problem and validation in the pull request.

Do not commit model weights, private data, credentials, or downloaded benchmark assets. Changes to candidate generation must document which inference-time inputs they consume. Preserve source-group splits, and count failed/missing predictions in the evaluation denominator.

Cache optimizations require separate-versus-batched equivalence tests covering DeltaNet states and full-attention KV, including variable image sizes and padding. Performance claims require measured end-to-end results with hardware, precision, resolution, and candidate-generation/fallback costs. Raw softmax probabilities must not be described as calibrated.

Keep the README's implemented/planned distinction current. Repository citations refer to software until a manuscript actually exists. Original contributions are licensed under the repository's MIT license; upstream assets retain their own terms.
