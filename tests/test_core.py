import copy

import pytest
import torch

from vision_jev.metrics import score
from vision_jev.model import CandidateHead, VisionJEV
from vision_jev.schema import assert_disjoint, read_records, validate


def test_group_leakage_and_schema():
    train = read_records("examples/train.jsonl", True)
    assert_disjoint(train, read_records("examples/eval.jsonl", True))
    with pytest.raises(ValueError, match="leakage"):
        assert_disjoint(train, train)
    r = copy.deepcopy(train[0])
    r["candidates"][1]["id"] = r["candidates"][0]["id"]
    with pytest.raises(ValueError):
        validate(r)


def test_pointer_mask_gradients_and_permutation():
    torch.manual_seed(1)
    head = CandidateHead(16, 8)
    h = torch.randn(2, 7, 16, requires_grad=True)
    positions = torch.tensor([[1, 3, 0], [2, 4, 5]])
    mask = torch.tensor([[True, True, False], [True, True, True]])
    logits = head(h, positions, torch.tensor([6, 6]), mask)
    assert logits.softmax(-1)[0, 2] == 0
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1, 0]))
    loss.backward()
    assert h.grad.abs().sum() > 0
    swapped = head(h.detach(), positions[:, [1, 0, 2]], torch.tensor([6, 6]), mask[:, [1, 0, 2]])
    torch.testing.assert_close(swapped, logits.detach()[:, [1, 0, 2]])


def test_missing_predictions_count_as_wrong():
    records = read_records("examples/eval.jsonl", True)
    metrics = score(records, [])
    assert metrics["accuracy"] == 0 and metrics["missing"] == 1
    assert metrics["selective_accuracy"] is None


def tiny_model():
    from transformers import Qwen3_5Config, Qwen3_5Model

    config = Qwen3_5Config(
        text_config={
            "vocab_size": 128,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "layer_types": ["linear_attention", "full_attention"],
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
                "mrope_section": [2, 3, 3],
            },
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "out_hidden_size": 32,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "num_position_embeddings": 16,
        },
        image_token_id=120,
        vision_start_token_id=121,
        vision_end_token_id=122,
    )
    return VisionJEV(Qwen3_5Model(config), projection_size=8)


def test_real_hybrid_architecture_lora_backward():
    model = tiny_model()
    model.configure_training("lora", rank=2)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    result = model(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        candidate_positions=torch.tensor([[2, 4]]),
        query_positions=torch.tensor([5]),
        candidate_mask=torch.ones(1, 2, dtype=torch.bool),
        labels=torch.tensor([1]),
    )
    result["loss"].backward()
    assert torch.isfinite(result["loss"])
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in model.named_parameters()
        if "lora_B" in n
    )
    assert all(
        not p.requires_grad for n, p in model.backbone.named_parameters() if "lora_" not in n
    )


def test_real_vision_forward():
    model = tiny_model()
    ids = torch.tensor([[121, 120, 122, 3, 4, 5]])
    result = model(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        pixel_values=torch.randn(4, 3 * 2 * 16 * 16),
        image_grid_thw=torch.tensor([[1, 2, 2]]),
        mm_token_type_ids=torch.tensor([[0, 1, 0, 0, 0, 0]]),
        candidate_positions=torch.tensor([[3, 4]]),
        query_positions=torch.tensor([5]),
        candidate_mask=torch.ones(1, 2, dtype=torch.bool),
        labels=torch.tensor([0]),
    )
    result["loss"].backward()
    assert torch.isfinite(result["loss"])


@pytest.mark.parametrize("multi", [False, True])
def test_checkpoint_roundtrip(tmp_path, multi):
    class DummyProcessor:
        def save_pretrained(self, directory):
            directory.mkdir()

    model = tiny_model().eval()
    batch = dict(
        input_ids=torch.tensor([[1, 2, 3, 4]]),
        attention_mask=torch.ones(1, 4, dtype=torch.long),
        candidate_positions=torch.tensor([[1, 2]]),
        query_positions=torch.tensor([[2, 3]]) if multi else torch.tensor([3]),
        candidate_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    with torch.no_grad():
        expected = model(**batch)["logits"]
    model.save(tmp_path, DummyProcessor())
    restored = VisionJEV.load(tmp_path).eval()
    with torch.no_grad():
        actual = restored(**batch)["logits"]
    torch.testing.assert_close(actual, expected)


def test_lora_checkpoint_roundtrip(tmp_path):
    class DummyProcessor:
        def save_pretrained(self, directory):
            directory.mkdir()

    base = tiny_model()
    base.backbone.save_pretrained(tmp_path / "base")
    model = VisionJEV.from_base(str(tmp_path / "base"))
    model.configure_training("lora", rank=2)
    batch = dict(
        input_ids=torch.tensor([[1, 2, 3, 4]]),
        attention_mask=torch.ones(1, 4, dtype=torch.long),
        candidate_positions=torch.tensor([[1, 2]]),
        query_positions=torch.tensor([3]),
        candidate_mask=torch.ones(1, 2, dtype=torch.bool),
        labels=torch.tensor([1]),
    )
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    model(**batch)["loss"].backward()
    opt.step()
    model.eval()
    with torch.no_grad():
        expected = model(**batch)["logits"]
    model.save(tmp_path / "checkpoint", DummyProcessor())
    restored = VisionJEV.load(tmp_path / "checkpoint").eval()
    with torch.no_grad():
        actual = restored(**batch)["logits"]
    torch.testing.assert_close(actual, expected)
    restored.backbone = restored.backbone.merge_and_unload()
    with torch.no_grad():
        merged = restored(**batch)["logits"]
    torch.testing.assert_close(merged, expected, atol=1e-6, rtol=1e-5)


def test_independent_hybrid_rows_padding_parity():
    model = tiny_model().eval()
    ids = torch.tensor([[1, 2, 3, 4, 0, 0], [5, 6, 7, 8, 9, 10]])
    masks = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]])
    cp = torch.tensor([[1, 2], [2, 4]])
    qp = torch.tensor([3, 5])
    with torch.no_grad():
        together = model(
            input_ids=ids,
            attention_mask=masks,
            candidate_positions=cp,
            query_positions=qp,
            candidate_mask=torch.ones(2, 2, dtype=torch.bool),
        )["logits"]
        for i, length in enumerate((4, 6)):
            separate = model(
                input_ids=ids[i : i + 1, :length],
                attention_mask=masks[i : i + 1, :length],
                candidate_positions=cp[i : i + 1],
                query_positions=qp[i : i + 1],
                candidate_mask=torch.ones(1, 2, dtype=torch.bool),
            )["logits"]
            torch.testing.assert_close(together[i], separate[0], atol=1e-6, rtol=1e-5)


def test_bfloat16_hidden_float32_head_backward():
    head = CandidateHead(16, 8)
    hidden = torch.randn(1, 4, 16, dtype=torch.bfloat16, requires_grad=True)
    logits = head(
        hidden, torch.tensor([[1, 2]]), torch.tensor([3]), torch.ones(1, 2, dtype=torch.bool)
    )
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1]))
    loss.backward()
    assert torch.isfinite(loss) and hidden.grad is not None


@pytest.mark.parametrize(
    "field,value", [("group_id", []), ("id", ""), ("context", None), ("image", 2)]
)
def test_schema_rejects_invalid_record_fields(field, value):
    record = read_records("examples/train.jsonl", True)[0]
    record[field] = value
    with pytest.raises(ValueError):
        validate(record)


@pytest.mark.parametrize("bbox", [None, [True, 0, 1, 1], [0, 0, float("nan"), 1]])
def test_schema_rejects_invalid_evidence(bbox):
    record = read_records("examples/train.jsonl", True)[0]
    record["candidates"][0]["bbox"] = bbox
    with pytest.raises(ValueError):
        validate(record)


def test_scoring_abstention_and_invalid_distribution():
    records = read_records("examples/eval.jsonl", True)
    p = {
        "id": records[0]["id"],
        "candidate_id": "profit",
        "abstained": True,
        "probabilities": {"profit": 0.7, "revenue": 0.2, "none": 0.1},
    }
    result = score(records, [p])
    assert result["accuracy"] == 1 and result["coverage"] == 0
    p["probabilities"]["profit"] = float("nan")
    with pytest.raises(ValueError, match="probabilities"):
        score(records, [p])
    with pytest.raises(ValueError, match="Empty"):
        score([], [])


def test_multi_field_one_forward_and_loss(tmp_path):
    model = tiny_model()
    model.configure_training("lora", rank=2)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    calls = []
    hook = model.backbone.register_forward_hook(lambda *args: calls.append(1))
    inputs = dict(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        candidate_positions=torch.tensor([[1, 3]]),
        query_positions=torch.tensor([[5, 7]]),
        candidate_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    out = model(**inputs, labels=torch.tensor([[0, 1]]))
    assert calls == [1]
    assert out["logits"].shape == (1, 2, 2)
    out["loss"].backward()
    assert torch.isfinite(out["loss"])
    assert model.head.query.weight.grad.abs().sum() > 0
    hook.remove()
    model.eval()
    full = model(**inputs)["logits"]
    for i in range(2):
        single = model(**dict(inputs, query_positions=inputs["query_positions"][:, i]))["logits"]
        torch.testing.assert_close(full[:, i], single)


def test_multi_field_schema_and_missing_metrics():
    r = read_records("examples/train.jsonl", True)[0]
    r.pop("question")
    r.pop("target")
    r["fields"] = [{"key": "revenue", "target": "revenue"}, {"key": "profit", "target": "profit"}]
    validate(r, True)
    assert score([r], [])["missing"] == 2
    r["fields"][1]["key"] = "revenue"
    with pytest.raises(ValueError, match="unique"):
        validate(r)
