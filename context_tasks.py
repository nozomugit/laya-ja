"""Laya encoder readouts for Japanese inference, similarity, grammar and common sense."""

import re
from pathlib import Path

import torch
from torch import nn

from laya_ja import LABELS, load_base

D = 768


def lexical(a, b):
    def chunks(s, n):
        return {s[i : i + n] for i in range(len(s) - n + 1)}

    f = [len(a) / 100, len(b) / 100, float(a == b), float(b in a), float(a in b)]
    for n in (1, 2, 3):
        x, y = chunks(a, n), chunks(b, n)
        common = len(x & y)
        f.extend(
            [
                common / max(1, len(x)),
                common / max(1, len(y)),
                common / max(1, len(x | y)),
            ]
        )
    for pattern in [
        "ない|なかった|ません|ぬ",
        "一人|一つ|一個",
        "全員|すべて|全て",
        "だけ|しか",
        "[0-9０-９]+",
        "より|前|後",
        "は",
        "が",
        "を",
        "に",
        "で",
        "と",
        "から",
        "まで",
        "れる|られる",
        "年|月|日",
        "誰|何|いつ|どこ",
        "[ァ-ヶ]",
        "[一-龯]",
    ]:
        x, y = len(re.findall(pattern, a)), len(re.findall(pattern, b))
        f.extend([min(x, 6) / 6, min(y, 6) / 6, (x - y) / max(x + y, 1)])
    return f


F = len(lexical("", ""))


def sequence(tok, parts):
    ids = [tok.cls_token_id]
    regions = []
    for s in parts:
        ts = tok(s, add_special_tokens=False)["input_ids"] or [tok.unk_token_id]
        start = len(ids)
        ids.extend(ts)
        regions.append((start, len(ids)))
        ids.append(tok.sep_token_id)
    if len(ids) > min(tok.model_max_length, 8192):
        raise ValueError("Input exceeds the encoder limit")
    return ids, regions


def item(tok, task, a, options):
    if task == "jcola":
        ids, regs = sequence(tok, [a])
        start, end = regs[0]
        mid = (start + end) // 2
        return ids, [(start, max(start + 1, mid)), (mid, end)], [lexical(a, a)]
    ids, regs = sequence(tok, [a, *options])
    return ids, regs, [lexical(a, b) for b in options]


@torch.inference_mode()
def batch(agent, items):
    pad = agent.tok.pad_token_id
    x = nn.utils.rnn.pad_sequence(
        [torch.tensor(v[0]) for v in items], batch_first=True, padding_value=pad
    ).to(agent.device)
    h = agent.model.encoder(
        input_ids=x, attention_mask=(x != pad).long()
    ).last_hidden_state
    res = []
    for hi, (_, regs, f) in zip(h, items):
        mean = [hi[s:e].mean(0) for s, e in regs]
        first = mean[0].expand(len(f), -1)
        other = torch.stack(mean[1:])
        res.append(torch.cat((first, other, torch.tensor(f, device=hi.device)), dim=-1))
    return torch.stack(res).half().cpu()


class Readout(nn.Module):
    def __init__(self, task):
        super().__init__()
        outputs = {"nli": 3, "jcola": 2, "jsts": 1, "jcommonsenseqa": 1}[task]
        self.task = task
        self.project = nn.Linear(D, 96)
        self.body = nn.Sequential(
            nn.Linear(4 * 96 + F, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(96, outputs),
        )

    def representation(self, x):
        a, b, f = x.float().split([D, D, F], dim=-1)
        a = self.project(nn.functional.layer_norm(a, (D,)))
        b = self.project(nn.functional.layer_norm(b, (D,)))
        return torch.cat([a, b, (a - b).abs(), a * b, f], dim=-1)

    def forward(self, x):
        z = self.body(self.representation(x))
        return z.squeeze(-1) if self.task == "jcommonsenseqa" else z.squeeze(1)


class DomainCorrection(nn.Module):
    def __init__(self):
        super().__init__()
        self.delta = nn.Sequential(
            nn.LayerNorm(4 * 96 + F),
            nn.Linear(4 * 96 + F, 32),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(32, 3),
        )
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(self, x):
        return self.delta(x).squeeze(1)


class ContextReader:
    def __init__(self, task, model_path=None, device="cpu", weights=None, domain=None):
        from safetensors.torch import load_file

        self.task = task
        self.agent = load_base(model_path, device)
        self.head = Readout(task).to(device).eval()
        weights = Path(weights or Path(__file__).with_name("weights") / "context")
        self.head.load_state_dict(load_file(str(weights / (task + ".safetensors"))))
        self.correction = None
        self.domain = domain
        if domain:
            if task != "nli" or domain not in {
                "jnli",
                "jsick",
                "janli",
                "jamp",
                "logic",
            }:
                raise ValueError("Unknown NLI domain")
            self.correction = DomainCorrection().to(device).eval()
            self.correction.load_state_dict(
                load_file(str(weights / ("domain-" + domain + ".safetensors")))
            )

    @torch.inference_mode()
    def predict(self, a, options, symmetric=False):
        views = [item(self.agent.tok, self.task, a, options)]
        if symmetric:
            if self.task != "jsts" or len(options) != 1:
                raise ValueError("Symmetric scoring is for two-sentence similarity")
            views.append(item(self.agent.tok, self.task, options[0], [a]))
        x = batch(self.agent, views).to(self.agent.device)
        z = self.head(x)
        if self.correction is not None:
            z = z + self.correction(self.head.representation(x))
        z = z.mean(0).cpu()
        if self.domain == "janli":
            z = torch.stack([z[0], z[1:].logsumexp(0)])
            labels = ["含意", "非含意"]
            p = z.softmax(-1)
            return {
                "choice": labels[p.argmax().item()],
                "probabilities": dict(zip(labels, p.tolist())),
            }
        if self.task == "jsts":
            return {"similarity": float(z.clamp(0, 5).item())}
        labels = (
            LABELS
            if self.task == "nli"
            else (["容認しにくい", "容認できる"] if self.task == "jcola" else options)
        )
        p = z.softmax(-1)
        return {
            "choice": labels[p.argmax().item()],
            "probabilities": dict(zip(labels, p.tolist())),
        }


class CommonSenseMixture:
    def __init__(self, model_path=None, device="cpu"):
        import json

        from corpus_experiment import old_head

        self.reader = ContextReader("jcommonsenseqa", model_path, device)
        root = Path(__file__).with_name("weights")
        self.previous = old_head(root, "jcommonsenseqa")
        self.weight = json.loads(
            (root / "context" / "common-mixture.json").read_text()
        )["context_weight"]

    @torch.inference_mode()
    def predict(self, question, choices):
        from experiment import extract
        from laya_ja import qa_request

        if len(choices) != 5 or len(set(choices)) != 5:
            raise ValueError("Supply five distinct choices")
        a = self.reader.agent
        v = batch(a, [item(a.tok, "jcommonsenseqa", question, choices)]).to(a.device)
        z = self.reader.head(v).cpu()
        f = extract(
            a, [(qa_request(question, choices), 0, None, "input")], 1, progress=False
        )
        previous = self.previous(f).squeeze(-1)
        p = (z * self.weight + previous * (1 - self.weight))[0].softmax(-1)
        return {
            "choice": choices[p.argmax().item()],
            "probabilities": dict(zip(choices, p.tolist())),
        }
