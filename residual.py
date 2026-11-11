"""A per-option residual network above the base model's frozen marker features."""

import torch
from torch import nn


class ResidualScorer(nn.Module):
    def __init__(self, linear, width):
        super().__init__()
        self.linear = linear
        self.linear.requires_grad_(False)
        d = linear.in_features
        self.register_buffer("center", torch.zeros(d))
        self.register_buffer("scale", torch.ones(d))
        self.extra = nn.Sequential(
            nn.Linear(d, width), nn.GELU(), nn.Dropout(0.15), nn.Linear(width, 1)
        )
        nn.init.zeros_(self.extra[-1].weight)
        nn.init.zeros_(self.extra[-1].bias)

    def forward(self, x):
        return self.linear(x) + self.extra((x - self.center) / self.scale)
