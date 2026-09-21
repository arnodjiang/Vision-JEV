"""One independent question per record; candidates must come from inference inputs."""

import json
from pathlib import Path

CAND_MARK = "<|fim_suffix|>"
QUERY_MARK = "<|fim_middle|>"


def validate(record, require_target=False):
    if not isinstance(record, dict):
        raise ValueError("Record must be an object")
    for key in ("id", "group_id", "question", "task", "candidates"):
        if key not in record:
            raise ValueError(f"Missing {key}")
    for key in ("id", "group_id"):
        if not isinstance(record[key], str) or not record[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    if not isinstance(record.get("context", ""), str):
        raise ValueError("context must be a string")
    if "image" in record and (not isinstance(record["image"], str) or not record["image"]):
        raise ValueError("image must be a nonempty local path")
    if record["task"] not in ("extract", "choice", "boolean"):
        raise ValueError("task must be extract, choice or boolean")
    if not isinstance(record["question"], str) or not record["question"].strip():
        raise ValueError("question must be nonempty")
    cs = record["candidates"]
    if not isinstance(cs, list) or len(cs) < 2:
        raise ValueError("At least two candidates required (include a null candidate if needed)")
    if any(not isinstance(c, dict) for c in cs):
        raise ValueError("Candidates must be objects")
    ids = [c.get("id") for c in cs]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Candidate IDs must be unique nonempty strings")
    for c in cs:
        if "is_null" in c and type(c["is_null"]) is not bool:
            raise ValueError("is_null must be a Boolean")
        if not isinstance(c.get("text"), str):
            raise ValueError("Candidate text must be a string")
        if "bbox" in c:
            b = c["bbox"]
            if (
                not isinstance(b, (list, tuple))
                or len(b) != 4
                or not all(type(x) in (int, float) and 0 <= x <= 1 for x in b)
                or b[0] > b[2]
                or b[1] > b[3]
            ):
                raise ValueError("bbox must be normalized ordered xyxy")
    if record["task"] == "boolean" and (
        len(cs) != 2
        or any(type(c.get("value")) is not bool for c in cs)
        or {c["value"] for c in cs} != {True, False}
    ):
        raise ValueError("boolean requires true and false candidates")
    if require_target and record.get("target") not in ids:
        raise ValueError("Training target must name a candidate")
    return record


def read_records(path, require_target=False):
    path = Path(path).resolve()
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = validate(json.loads(line), require_target)
        if r.get("image"):
            image = Path(r["image"])
            r["image"] = (
                str((path.parent / image).resolve()) if not image.is_absolute() else str(image)
            )
            if not Path(r["image"]).is_file():
                raise ValueError(f"Missing image: {r['image']}")
        records.append(r)
    if not records:
        raise ValueError("Empty dataset")
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Duplicate record IDs")
    return records


def assert_disjoint(train, evaluation):
    overlap = {r["group_id"] for r in train} & {r["group_id"] for r in evaluation}
    if overlap:
        raise ValueError(f"Source-group leakage: {len(overlap)} overlapping groups")
