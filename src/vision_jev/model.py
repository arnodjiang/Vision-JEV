"""Native multimodal backbone with a trainable candidate pointer head."""

import json
from pathlib import Path

import torch
from torch import nn
from transformers import Qwen3_5Model


class CandidateHead(nn.Module):
    def __init__(self, hidden_size, projection_size=128):
        super().__init__()
        self.query = nn.Linear(hidden_size, projection_size)
        self.key = nn.Linear(hidden_size, projection_size)
        self.scale = projection_size**-0.5

    def forward(self, hidden, candidate_positions, query_positions, candidate_mask):
        batch = torch.arange(hidden.shape[0], device=hidden.device)
        q = hidden[batch, query_positions]
        k = hidden[batch[:, None], candidate_positions.clamp_min(0)]
        scores = (
            self.query(q.to(self.query.weight.dtype))[:, None]
            * self.key(k.to(self.key.weight.dtype))
        ).sum(-1) * self.scale
        return scores.float().masked_fill(~candidate_mask, float("-inf"))


class VisionJEV(nn.Module):
    def __init__(self, backbone, projection_size=128):
        super().__init__()
        self.backbone = backbone
        self.projection_size = projection_size
        self.head = CandidateHead(backbone.config.text_config.hidden_size, projection_size)

    @classmethod
    def from_base(cls, name="Qwen/Qwen3.5-0.8B", revision=None):
        backbone = Qwen3_5Model.from_pretrained(name, revision=revision, dtype=torch.float32)
        return cls(backbone)

    def configure_training(self, mode="lora", rank=16):
        self.backbone.requires_grad_(False)
        if mode == "lora":
            from peft import LoraConfig, get_peft_model

            # Dense language FFNs exist in both hybrid layer types. Do not match vision MLPs.
            targets = r"language_model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)"
            self.backbone = get_peft_model(
                self.backbone,
                LoraConfig(
                    r=rank,
                    lora_alpha=2 * rank,
                    target_modules=targets,
                    lora_dropout=0.05,
                    bias="none",
                    revision=getattr(self.backbone.config, "_commit_hash", None),
                ),
            )
        elif mode != "head":
            raise ValueError("mode must be head or lora")

    def forward(self, candidate_positions, query_positions, candidate_mask, labels=None, **inputs):
        out = self.backbone(**inputs, use_cache=False, return_dict=True)
        logits = self.head(
            out.last_hidden_state, candidate_positions, query_positions, candidate_mask
        )
        result = {"logits": logits}
        if labels is not None:
            result["loss"] = nn.functional.cross_entropy(logits, labels)
        return result

    def save(self, directory, processor, metadata=None):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(directory / "backbone", safe_serialization=True)
        processor.save_pretrained(directory / "processor")
        from safetensors.torch import save_file

        save_file(
            {k: v.detach().cpu().contiguous() for k, v in self.head.state_dict().items()},
            str(directory / "head.safetensors"),
        )
        config = {
            "projection_size": self.projection_size,
            "adapter": hasattr(self.backbone, "peft_config"),
            "metadata": metadata or {},
        }
        (directory / "vision_jev.json").write_text(json.dumps(config, indent=2))

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        config = json.loads((directory / "vision_jev.json").read_text())
        if config["adapter"]:
            from peft import PeftConfig, PeftModel

            adapter = PeftConfig.from_pretrained(directory / "backbone")
            base = Qwen3_5Model.from_pretrained(
                adapter.base_model_name_or_path, revision=adapter.revision, dtype=torch.float32
            )
            backbone = PeftModel.from_pretrained(base, directory / "backbone")
        else:
            backbone = Qwen3_5Model.from_pretrained(directory / "backbone", dtype=torch.float32)
        model = cls(backbone, config["projection_size"])
        from safetensors.torch import load_file

        model.head.load_state_dict(load_file(str(directory / "head.safetensors")))
        return model
