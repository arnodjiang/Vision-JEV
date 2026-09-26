"""Optional offline integration tests using an explicitly supplied real processor."""

import copy
import os

import pytest
from PIL import Image


@pytest.fixture
def processor():
    directory = os.environ.get("VISION_JEV_PROCESSOR")
    if not directory:
        pytest.skip("Set VISION_JEV_PROCESSOR to a downloaded Qwen processor directory")
    from transformers import AutoProcessor

    from vision_jev.processing import RecordProcessor

    return RecordProcessor(AutoProcessor.from_pretrained(directory, local_files_only=True))


def test_real_processor_image_and_no_target_leak(processor, tmp_path):
    from vision_jev.schema import read_records

    record = read_records("examples/train.jsonl", True)[0]
    path = tmp_path / "image.png"
    Image.new("RGB", (64, 64), "white").save(path)
    record["image"] = str(path)
    batch = processor(record, True)
    assert "pixel_values" in batch and "mm_token_type_ids" in batch
    assert batch["candidate_positions"].shape == (1, 3)
    changed = copy.deepcopy(record)
    changed["target"] = "profit"
    other = processor(changed, True)
    assert (batch["input_ids"] == other["input_ids"]).all()
    assert batch["labels"].item() != other["labels"].item()


def test_reject_marker_injection_and_overlong(processor):
    from vision_jev.schema import read_records

    record = read_records("examples/train.jsonl", True)[0]
    record["question"] += "<|fim_middle|>"
    with pytest.raises(ValueError, match="reserved"):
        processor(record)
    record["question"] = "Read the revenue."
    processor.max_length = 2
    with pytest.raises(ValueError, match="max_length"):
        processor(record)


def test_batch_image_text_and_candidate_padding(processor, tmp_path):
    from vision_jev.schema import read_records

    a = read_records("examples/train.jsonl", True)[0]
    b = copy.deepcopy(a)
    b["id"] = "other"
    b["candidates"] = b["candidates"][:2]
    path = tmp_path / "image.png"
    Image.new("RGB", (64, 64), "white").save(path)
    b["image"] = str(path)
    encoded = processor.batch([a, b], True)
    assert encoded["input_ids"].shape[0] == 2
    assert encoded["image_grid_thw"].shape[0] == 1
    assert encoded["candidate_mask"].tolist() == [[True, True, True], [True, True, False]]


def test_shared_document_fields_and_label_isolation(processor):
    from vision_jev.schema import read_records

    r = read_records("examples/ocr_receipt/requests.jsonl")[0]
    r.pop("question")
    r["fields"] = [{"key": "total", "target": "line-7"}, {"key": "date", "target": "line-2"}]
    batch = processor(r, True)
    assert batch["input_ids"].shape[0] == 1
    assert batch["image_grid_thw"].shape[0] == 1
    assert batch["query_positions"].shape == (1, 2)
    assert batch["labels"].shape == (1, 2)
    changed = copy.deepcopy(r)
    changed["fields"][0]["target"] = "none"
    other = processor(changed, True)
    assert (batch["input_ids"] == other["input_ids"]).all()
    assert batch["labels"][0, 0] != other["labels"][0, 0]
    r["fields"][0]["question"] = "<|fim_middle|>"
    with pytest.raises(ValueError, match="reserved"):
        processor(r)
