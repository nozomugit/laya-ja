"""Low-rank updates for every linear layer of the Laya encoder, and a reader that applies a trained adapter."""

import math

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn

from laya_ja import REVISION, load_base, nli_request, query


class LoRALinear(nn.Module):
    """Frozen linear layer W plus a trainable update (alpha / r) * B @ A. B starts at zero."""

    def __init__(self, base, r, alpha, dropout):
        super().__init__()
        self.base = base
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        update = F.linear(F.linear(self.drop(x), self.lora_A), self.lora_B)
        return self.base(x) + update * self.scale

    def merged(self):
        with torch.no_grad():
            delta = (self.lora_B.float() @ self.lora_A.float()) * self.scale
            self.base.weight.add_(delta.to(self.base.weight.dtype))
        return self.base


def _targets(encoder, kind):
    return [
        n
        for n, m in encoder.named_modules()
        if isinstance(m, kind) and n.startswith("layers.")
    ]


def add_lora(model, r, alpha, dropout=0.0):
    """Freeze the encoder and wrap each linear layer inside its transformer blocks."""
    encoder = model.encoder
    for p in encoder.parameters():
        p.requires_grad_(False)
    names = _targets(encoder, nn.Linear)
    for name in names:
        parent_name, child = name.rsplit(".", 1)
        parent = encoder.get_submodule(parent_name)
        base = getattr(parent, child)
        wrapped = LoRALinear(base, r, alpha, dropout).to(base.weight.device)
        setattr(parent, child, wrapped)
    return names


def merge_lora(model):
    """Fold every update into its base weight, restoring the original module layout."""
    encoder = model.encoder
    for name in _targets(encoder, LoRALinear):
        parent_name, child = name.rsplit(".", 1)
        parent = encoder.get_submodule(parent_name)
        setattr(parent, child, getattr(parent, child).merged())
    for p in encoder.parameters():
        p.requires_grad_(True)


def trainable_state(model):
    return {
        n: p.detach().float().cpu().clone()
        for n, p in model.named_parameters()
        if p.requires_grad
    }


def apply_adapter(model, path):
    """Install a saved adapter (encoder updates and decision head) on the pinned base model."""
    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata()
    if meta["base_revision"] != REVISION:
        raise ValueError("Adapter was trained on a different base revision.")
    add_lora(model, int(meta["rank"]), float(meta["alpha"]))
    result = model.load_state_dict(load_file(str(path)), strict=False)
    if result.unexpected_keys:
        raise ValueError(f"Adapter tensors do not match: {result.unexpected_keys}")
    return model.eval()


class LoraReader:
    """Three-way Japanese inference with one adapter for every corpus; no domain flag."""

    def __init__(self, adapter, model_path=None, device="cpu"):
        self.agent = load_base(model_path, device)
        apply_adapter(self.agent.model, adapter)
        merge_lora(self.agent.model)
        self.agent.model.eval()

    def predict(self, premise, hypothesis):
        return query(self.agent, *nli_request(premise, hypothesis))
