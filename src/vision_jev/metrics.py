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
