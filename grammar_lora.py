"""Adapt the last Laya encoder MLP output projection for grammatical acceptability."""

from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn

from laya_ja import load_base


class LastLayerGrammar(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.Linear(1152, 12, bias=False)
        self.up = nn.Linear(12, 768, bias=False)
        nn.init.zeros_(self.up.weight)
        self.project = nn.Sequential(nn.Linear(768, 64), nn.GELU())
        self.attn = nn.Linear(64, 1)
        self.readout = nn.Sequential(
            nn.LayerNorm(192), nn.Dropout(0.2), nn.Linear(192, 2)
        )

    def forward(self, base, mlp, mask, norm):
        h = norm(base.float() + self.up(self.down(mlp.float())))
        z = self.project(h)
        scores = self.attn(z).squeeze(-1).masked_fill(~mask, -1e4)
        attended = (z * scores.softmax(1)[:, :, None]).sum(1)
        mean = (z * mask[:, :, None]).sum(1) / mask.sum(1, keepdim=True)
        peak = z.masked_fill(~mask[:, :, None], -1e4).amax(1)
        return self.readout(torch.cat([mean, peak, attended], dim=1))


@torch.no_grad()
def layers(agent, tokens):
    pad = agent.tok.pad_token_id
    ids = nn.utils.rnn.pad_sequence(
        [torch.tensor(x) for x in tokens], batch_first=True, padding_value=pad
    ).to(agent.device)
    enc = agent.model.encoder
    cache = {}

    def capture(name):
        def hook(module, args):
            cache[name] = args[0].detach()

        return hook

    h1 = enc.layers[-1].mlp.Wo.register_forward_pre_hook(capture("mlp"))
    h2 = enc.final_norm.register_forward_pre_hook(capture("base"))
    try:
        enc(input_ids=ids, attention_mask=(ids != pad).long())
    finally:
        h1.remove()
        h2.remove()
    return [
        (
            cache["base"][i, 1 : len(x) - 1].half().cpu(),
            cache["mlp"][i, 1 : len(x) - 1].half().cpu(),
        )
        for i, x in enumerate(tokens)
    ]


def tokens(agent, text):
    ids = agent.tok(text, add_special_tokens=False)["input_ids"] or [
        agent.tok.unk_token_id
    ]
    if len(ids) + 2 > min(agent.tok.model_max_length, 8192):
        raise ValueError("Sentence too long")
    return [agent.tok.cls_token_id, *ids, agent.tok.sep_token_id]


class GrammarReader:
    def __init__(self, model_path=None, device="cpu"):
        import json

        self.agent = load_base(model_path, device)
        p = Path(__file__).with_name("weights") / "context"
        self.head = LastLayerGrammar().to(device).eval()
        self.head.load_state_dict(load_file(str(p / "grammar-lora.safetensors")))
        self.cut = json.loads((p / "grammar-threshold.json").read_text())["threshold"]

    @torch.inference_mode()
    def predict(self, text):
        if not text.strip():
            raise ValueError("Provide a sentence")
        b, m = layers(self.agent, [tokens(self.agent, text)])[0]
        b, m = b.to(self.agent.device), m.to(self.agent.device)
        mask = torch.ones((1, len(m)), dtype=torch.bool, device=m.device)
        p = float(
            self.head(b[None], m[None], mask, self.agent.model.encoder.final_norm)[
                0
            ].softmax(0)[1]
        )
        return {"acceptable": p >= self.cut, "probability": p, "threshold": self.cut}
