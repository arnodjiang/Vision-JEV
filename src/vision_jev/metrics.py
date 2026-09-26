"""Candidate-level evaluation, distinct from end-to-end QA semantic accuracy."""

import math

from .schema import validate


def score(records, predictions):
    if not records:
        raise ValueError("Empty evaluation dataset")
    for record in records:
        validate(record, require_target=True)
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Duplicate evaluation IDs")
    by_id = {p["id"]: p for p in predictions}
    if len(by_id) != len(predictions):
        raise ValueError("Duplicate predictions")
    if set(by_id) - {r["id"] for r in records}:
        raise ValueError("Unknown prediction IDs")
    if any("fields" in r for r in records):
        expanded, outputs = [], []
        for i, r in enumerate(records):
            p = by_id.get(r["id"])
            if "fields" not in r:
                expanded.append(dict(r, id=str((i, None))))
                if p is not None:
                    outputs.append(dict(p, id=str((i, None))))
                continue
            if p is not None and set(p.get("fields", {})) - {f["key"] for f in r["fields"]}:
                raise ValueError("Unknown predicted field keys")
            for f in r["fields"]:
                identity = str((i, f["key"]))
                item = {k: v for k, v in r.items() if k != "fields"}
                item.update(
                    id=identity,
                    task="extract",
                    question=f.get("question", f["key"]),
                    target=f["target"],
                )
                expanded.append(item)
                if p is not None and f["key"] in p.get("fields", {}):
                    outputs.append(dict(p["fields"][f["key"]], id=identity))
        return score(expanded, outputs)
    correct = covered = accepted_correct = 0
    nll = 0.0
    for r in records:
        p = by_id.get(r["id"])
        if p is not None:
            ids = {c["id"] for c in r["candidates"]}
            probs = p.get("probabilities", {})
            if p.get("candidate_id") not in ids or set(probs) != ids:
                raise ValueError("Prediction candidates do not match the request")
            if any(
                type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
                for v in probs.values()
            ):
                raise ValueError("Invalid candidate probabilities")
            if not math.isclose(sum(probs.values()), 1.0, abs_tol=1e-5):
                raise ValueError("Candidate probabilities must sum to one")
            if type(p.get("abstained")) is not bool:
                raise ValueError("Prediction requires Boolean abstained")
        hit = bool(p and p.get("candidate_id") == r["target"])
        correct += hit
        accepted = bool(p and not p.get("abstained", False))
        covered += accepted
        accepted_correct += accepted and hit
        probability = p.get("probabilities", {}).get(r["target"], 0) if p else 0
        nll -= math.log(max(float(probability), 1e-12))
    n = len(records)
    return {
        "count": n,
        "missing": n - len(by_id),
        "accuracy": correct / n,
        "coverage": covered / n,
        "selective_accuracy": accepted_correct / covered if covered else None,
        "nll": nll / n,
    }
