import torch
from PIL import Image

from .schema import CAND_MARK, QUERY_MARK, validate


class RecordProcessor:
    def __init__(self, processor, max_length=8192):
        self.processor = processor
        self.max_length = max_length
        self.marker_ids = []
        for marker in (CAND_MARK, QUERY_MARK):
            ids = processor.tokenizer.encode(marker, add_special_tokens=False)
            if len(ids) != 1 or processor.tokenizer.convert_ids_to_tokens(ids[0]) != marker:
                raise ValueError(f"Tokenizer lacks reserved marker {marker}")
            self.marker_ids.append(ids[0])

    def __call__(self, record, training=False):
        validate(record, require_target=training)
        texts = [record["question"], record.get("context", "")] + [
            c["text"] for c in record["candidates"]
        ]
        # Reject special-token injection, including image placeholders and chat delimiters.
        for text in texts:
            if any(
                token in text
                for token in set(self.processor.tokenizer.all_special_tokens)
                | {CAND_MARK, QUERY_MARK}
            ):
                raise ValueError("User content contains a reserved tokenizer token")
        prompt = f"Task: {record['task']}\nContext: {record.get('context', '')}\nQuestion: {record['question']}\nCandidates:\n"
        prompt += "\n".join(
            f"{i}: {c['text']} {CAND_MARK}" for i, c in enumerate(record["candidates"])
        )
        prompt += f"\nDecide: {QUERY_MARK}"
        content = []
        image = None
        if record.get("image"):
            with Image.open(record["image"]) as im:
                image = im.convert("RGB")
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})
        rendered = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=False
        )
        kwargs = {"text": [rendered], "return_tensors": "pt"}
        if image is not None:
            kwargs["images"] = [image]
        batch = dict(self.processor(**kwargs))
        ids = batch["input_ids"][0]
        if ids.numel() > self.max_length:
            raise ValueError("Input exceeds max_length; reduce resolution/candidates explicitly")
        cp = (ids == self.marker_ids[0]).nonzero().flatten()
        qp = (ids == self.marker_ids[1]).nonzero().flatten()
        if cp.numel() != len(record["candidates"]) or qp.numel() != 1:
            raise ValueError("Marker count mismatch")
        batch.update(
            candidate_positions=cp[None],
            query_positions=qp,
            candidate_mask=torch.ones(1, len(cp), dtype=torch.bool),
        )
        if training:
            batch["labels"] = torch.tensor(
                [[c["id"] for c in record["candidates"]].index(record["target"])]
            )
        return batch

    def batch(self, records, training=False):
        """Independent padded rows; no shared recurrent state or prefix caching."""
        if not records:
            raise ValueError("Empty batch")
        rows = [self(record, training) for record in records]
        batch = {}
        token_keys = ("input_ids", "attention_mask", "mm_token_type_ids", "token_type_ids")
        allowed = set(token_keys) | {
            "pixel_values",
            "image_grid_thw",
            "candidate_positions",
            "query_positions",
            "candidate_mask",
            "labels",
        }
        if any(set(row) - allowed for row in rows):
            raise ValueError("Unsupported processor output for batching")
        for key in token_keys:
            if key not in rows[0]:
                continue
            pad = self.processor.tokenizer.pad_token_id if key == "input_ids" else 0
            batch[key] = torch.nn.utils.rnn.pad_sequence(
                [r[key][0] for r in rows], batch_first=True, padding_value=pad
            )
        for key in ("candidate_positions", "candidate_mask"):
            batch[key] = torch.nn.utils.rnn.pad_sequence(
                [r[key][0] for r in rows], batch_first=True, padding_value=0
            )
        batch["query_positions"] = torch.cat([r["query_positions"] for r in rows])
        if training:
            batch["labels"] = torch.cat([r["labels"] for r in rows])
        for key in ("pixel_values", "image_grid_thw"):
            tensors = [r[key] for r in rows if key in r]
            if tensors:
                batch[key] = torch.cat(tensors)
        return batch
