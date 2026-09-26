# Roadmap

| Stage | Deliverable | Acceptance criterion | Status |
| --- | --- | --- | --- |
| Model prototype | Native vision backbone, pointer readout, LoRA, batch API | Forward/backward, reload and merge tests | Implemented |
| Reproduction | CLI, synthetic images, tests, pinned reference revision | Official 0.8B integration smoke test | Implemented |
| Shared-document fields | 2–6 field readouts, one image encoding and backbone call | Multi-field loss, gradients, marker isolation, checkpoint tests | Implemented; task quality unmeasured |
| KIE supervision | Independent visual extraction/decision corpus | Source-group audit, candidate and evidence labels | Planned |
| Answer verification | Visual support labels for candidate answers, including insufficient evidence | Held-out verification metrics and joint-training ablations | Planned |
| Task checkpoint | Trained 0.8B adapters and model card | Held-out quality and error analysis | Planned |
| QA integration | Candidate generation and explicit fallback | Full-denominator QA evaluation | Planned |
| Calibration | Development-set calibration and rejection policy | Coverage–risk and calibration curves | Planned |
| Serving efficiency | Shared visual-prefix branches | Separate-vs-branch consistency across KV, recurrent and convolution states | Planned |
| Paper experiments | Matched baselines and multilingual results | Reproducible quality–latency frontier | Planned |
