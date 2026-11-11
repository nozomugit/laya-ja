"""Japanese three-way NLI on the fixed premise/hypothesis schema."""

from pathlib import Path

import torch
from laya_ja import REVISION, load, nli_request, query
from safetensors import safe_open
from torch import nn


class PairResidual(nn.Module):
    """Learn a correction from all three semantically ordered Laya marker vectors."""

    def __init__(self):
        super().__init__()
        self.register_buffer("center", torch.zeros(768))
        self.register_buffer("scale", torch.ones(768))
        self.project = nn.Sequential(nn.Linear(768, 32), nn.GELU())
        self.compare = nn.Sequential(
            nn.Linear(96, 32), nn.GELU(), nn.Dropout(0.12), nn.Linear(32, 3)
        )
        nn.init.zeros_(self.compare[-1].weight)
        nn.init.zeros_(self.compare[-1].bias)

    def forward(self, x):
        z = self.project((x - self.center) / self.scale)
        return self.compare(z.flatten(-2))


class EnsembleScorer(nn.Module):
    def __init__(self, prior, heads):
        super().__init__()
        self.prior = prior
        self.heads = nn.ModuleList(heads)

    def forward(self, x):
        delta = torch.stack([h(x) for h in self.heads]).mean(0)
        return self.prior(x) + delta.unsqueeze(-1)


class MultiCorpusNLI:
    """The three marker slots mean entailment, contradiction, neutral, in that order."""

    def __init__(self, model_path=None, device="cpu", variant="multicorpus-nli"):
        regimes = {"multicorpus-nli": "all", "jnli-specialist": "jnli_only"}
        if variant not in regimes:
            raise ValueError(f"Unknown NLI adapter: {variant}")
        self._agent = load("jnli", model_path=model_path, device=device)
        path = Path(__file__).with_name("weights") / f"{variant}.safetensors"
        heads = nn.ModuleList(PairResidual() for _ in range(3))
        with safe_open(str(path), framework="pt") as f:
            if (f.metadata().get("base_revision"), f.metadata().get("regime")) != (
                REVISION,
                regimes[variant],
            ):
                raise ValueError("Incompatible base revision")
            heads.load_state_dict({k: f.get_tensor(k) for k in list(f.keys())})
        prior = self._agent.model.scorer[-1]
        parameter = next(prior.parameters())
        self._agent.model.scorer[-1] = (
            EnsembleScorer(prior, heads)
            .to(device=parameter.device, dtype=parameter.dtype)
            .eval()
        )

    def predict(self, premise, hypothesis):
        return query(self._agent, *nli_request(premise, hypothesis))
