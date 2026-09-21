# Data contract

A JSONL file contains one independent question per line. All language variants, alternate renderings, and questions from the same original document should share `group_id`. Training requires a separate evaluation file; overlapping groups are rejected.

| Field | Required | Meaning |
| --- | --- | --- |
| `id` | Yes | Unique nonempty record ID |
| `group_id` | Yes | Original document identity for split checks |
| `task` | Yes | `extract`, `choice`, or `boolean` |
| `question` | Yes | Nonempty question text |
| `image` | No | Local image path, relative to the JSONL file |
| `context` | No | External prose legitimately available at inference |
| `candidates` | Yes | At least two candidates with unique IDs |
| `target` | Training/scoring | Correct candidate ID; never included in model inputs |

## Extraction

```json
{"id":"q1","group_id":"source-document-1","image":"table.png","task":"extract","question":"What is the revenue?","candidates":[{"id":"cell1","text":"12 USD","value":12,"bbox":[0.1,0.2,0.4,0.3]},{"id":"none","text":"Not present","is_null":true}],"target":"cell1"}
```

Candidate `text` is read by the model. Candidate `value`, normalized `bbox` in xyxy order, and `page` are output metadata. The model does not see candidate IDs or select based on those metadata fields. Include required semantic clues in the candidate text or image.

## Boolean judgment

```json
{"id":"q2","group_id":"source-document-2","task":"boolean","question":"Is revenue larger than profit?","context":"Revenue: 12; profit: 3.","candidates":[{"id":"yes","text":"Yes","value":true},{"id":"no","text":"No","value":false}],"target":"yes"}
```

Boolean requests require exactly two candidates with actual JSON Boolean values, not strings or integers. `choice` uses the same contract with two or more category candidates.

## Output semantics

`candidate_id` is the raw top-ranked candidate, including when abstention occurs. `value` is null on abstention. `probabilities` maps candidate IDs to probabilities; `calibrated` is false. `evidence` retains the selected candidate's supplied metadata and is not a verified localization claim.

Evaluation `accuracy` measures raw candidate accuracy, including abstained predictions. `coverage` and `selective_accuracy` describe accepted predictions. Missing predictions count as incorrect; their target probability is floored at 1e-12 for NLL. These metrics are not the original MStructQA semantic score.

## Candidate construction

Generate candidates using inference inputs only, for example an OCR engine, table parser, or fixed task ontology. Never use an evaluation reference answer or hidden rendering specification to insert the correct option. References may be used to annotate targets and score candidate coverage after candidate generation is frozen.

Record both candidate recall and conditional selection accuracy. A null option helps represent missing evidence but does not repair low candidate recall. The examples in this repository are synthetic fixtures, not a KIE training corpus.
